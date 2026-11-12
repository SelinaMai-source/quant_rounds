"""Inner 权重选择：非默认必须过全部门槛，否则回退默认。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from core.round11.common import daily_from_nav
from core.round23_3.contract import (
    BOOT_LB_Q,
    DEFAULT_W_LGBM,
    DEFAULT_W_TEN,
    INNER_BLOCK,
    INNER_BOOT_N,
    MAXDD_WORSEN,
    MIN_INNER_DAYS,
    MIN_INNER_TRADES,
)
from core.round23_3.inner_adapter import run_windowed_sleeve


@dataclass
class CandidateRecord:
    key: str
    is_default: bool
    inner_utility: float | None
    distance_to_default: float
    n_days: int
    n_trades: int
    maxdd: float | None
    mean_delta: float | None
    block_deltas: list[float | None]
    boot_lb: float | None
    all_gates_pass: bool
    window_fail: bool
    details: dict[str, Any]


def choose_weight(candidates: Sequence[CandidateRecord], *, default_key: str) -> dict[str, Any]:
    """禁止「默认永远因 distance=0 排第一」的旧逻辑。"""
    by_key = {c.key: c for c in candidates}
    if default_key not in by_key:
        raise KeyError(f"default_key missing: {default_key}")
    default = by_key[default_key]
    if default.window_fail:
        return {"chosen": default_key, "reason": "DEFAULT_BUT_WINDOW_FAIL", "chosen_record": default}
    qualified = [c for c in candidates if (not c.is_default) and c.all_gates_pass and (not c.window_fail)]
    if not qualified:
        return {"chosen": default_key, "reason": "FALLBACK_DEFAULT_NO_QUALIFIED", "chosen_record": default}
    qualified_sorted = sorted(
        qualified,
        key=lambda x: (x.distance_to_default, -(x.inner_utility if x.inner_utility is not None else -np.inf)),
    )
    chosen = qualified_sorted[0]
    return {"chosen": chosen.key, "reason": "QUALIFIED_NON_DEFAULT", "chosen_record": chosen}


def _maxdd(nav: pd.DataFrame) -> float:
    r = daily_from_nav(nav)
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def _utility(nav: pd.DataFrame, signal_start, signal_end) -> float | None:
    if nav is None or len(nav) == 0:
        return None
    dates = pd.to_datetime(nav["trade_date"] if "trade_date" in nav.columns else nav.index)
    mask = (dates >= pd.Timestamp(signal_start)) & (dates <= pd.Timestamp(signal_end))
    sub = nav.loc[mask]
    if len(sub) < 2:
        return None
    r = daily_from_nav(sub)
    if len(r) == 0:
        return None
    return float(np.log1p(r).mean())


def _block_bootstrap_lb(diff: np.ndarray, *, n_boot: int = INNER_BOOT_N, width: int = INNER_BLOCK, seed: int = 17) -> float:
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n < 3 * width:
        return float("nan")
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=int(np.ceil(n / width)))
        idx = np.concatenate([np.arange(int(s), int(s) + width) % n for s in starts])[:n]
        means.append(float(diff[idx].mean()))
    return float(np.quantile(means, BOOT_LB_Q))


def evaluate_inner_configs(
    configs: list[tuple[str, pd.DataFrame]],
    *,
    year: int,
    default_key: str,
    dest_root,
    resume: bool = True,
    model_hash: str | None = None,
    prediction_hash: str | None = None,
    model_fit_cutoff: str | None = None,
) -> dict[str, Any]:
    """对预测年 Y：Inner-1=Y-1 H1，Inner-2=Y-1 H2。"""
    from pathlib import Path

    dest_root = Path(dest_root)
    blocks = [
        (pd.Timestamp(year - 1, 1, 1), pd.Timestamp(year - 1, 6, 30), "H1"),
        (pd.Timestamp(year - 1, 7, 1), pd.Timestamp(year - 1, 12, 31), "H2"),
    ]
    # 先跑默认，便于配对
    default_scores = dict(configs)[default_key]
    # Outer OOF 自 2018 起；若 Y-1 无任何分数，内层不可识别 → 强制默认
    prior = default_scores.loc[pd.to_datetime(default_scores["signal_date"]).dt.year.eq(year - 1)]
    if prior.empty:
        return {
            "year": year,
            "chosen": default_key,
            "reason": "FALLBACK_DEFAULT_NO_PRIOR_OOF",
            "records": [
                {
                    "year": year,
                    "key": default_key,
                    "is_default": True,
                    "utility": None,
                    "n_days": 0,
                    "n_trades": 0,
                    "maxdd": None,
                    "mean_delta": None,
                    "block_deltas": [],
                    "boot_lb": None,
                    "all_gates_pass": True,
                    "window_fail": False,
                    "distance_to_default": 0.0,
                    "dd_ok": True,
                    "both_blocks_nonneg": True,
                }
            ],
            "fold_rows": [
                {
                    "year": year,
                    "key": default_key,
                    "block": tag,
                    "signal_start": str(start.date()),
                    "signal_end": str(end.date()),
                    "actual_backtest_start": None,
                    "actual_backtest_end": None,
                    "model_fit_cutoff": model_fit_cutoff,
                    "n_signal_days": 0,
                    "n_nav_days": 0,
                    "n_trades": 0,
                    "model_hash": model_hash,
                    "prediction_hash": prediction_hash,
                    "status": "NO_PRIOR_OOF",
                }
                for start, end, tag in blocks
            ],
        }
    records: list[CandidateRecord] = []
    fold_rows = []
    for key, scores in configs:
        utils = []
        dds = []
        block_deltas: list[float | None] = []
        daily_parts = []
        def_daily_parts = []
        n_days = 0
        n_trades = 0
        window_fail = False
        for start, end, tag in blocks:
            dest = dest_root / f"{year}" / key / tag
            sleeve = run_windowed_sleeve(
                dest,
                scores,
                signal_start=start,
                signal_end=end,
                backtest_start=start,
                backtest_end=end,
                initial_cash=300_000.0,
                extra={
                    "code": f"R233_INNER_{year}_{key}_{tag}",
                    "round": "23.3",
                    "scope": "INNER_REPLAY",
                    "model_fit_cutoff": model_fit_cutoff,
                    "model_hash": model_hash,
                    "prediction_hash": prediction_hash,
                },
                resume=resume,
            )
            if sleeve.get("window_fail") or sleeve.get("status") == "FAIL_FULL_HISTORY_WINDOW":
                window_fail = True
            nav_path = dest / "nav.parquet"
            nav = pd.read_parquet(nav_path) if nav_path.is_file() else pd.DataFrame()
            u = _utility(nav, start, end)
            utils.append(u)
            if len(nav):
                dds.append(_maxdd(nav))
                daily_parts.append(daily_from_nav(nav.loc[pd.to_datetime(nav["trade_date"]).between(start, end)]))
            n_days += int(sleeve.get("n_nav_days_in_signal_block") or 0)
            n_trades += int(sleeve.get("n_trades") or 0)
            fold_rows.append(
                {
                    "year": year,
                    "key": key,
                    "block": tag,
                    "signal_start": sleeve.get("signal_start"),
                    "signal_end": sleeve.get("signal_end"),
                    "actual_backtest_start": sleeve.get("actual_backtest_start"),
                    "actual_backtest_end": sleeve.get("actual_backtest_end"),
                    "model_fit_cutoff": model_fit_cutoff,
                    "n_signal_days": sleeve.get("n_signal_days"),
                    "n_nav_days": sleeve.get("n_nav_days"),
                    "n_trades": sleeve.get("n_trades"),
                    "model_hash": model_hash,
                    "prediction_hash": prediction_hash,
                    "status": sleeve.get("status"),
                }
            )
            # default daily for same block
            def_dest = dest_root / f"{year}" / default_key / tag
            if key != default_key and (def_dest / "nav.parquet").is_file():
                dnav = pd.read_parquet(def_dest / "nav.parquet")
                def_daily_parts.append(daily_from_nav(dnav.loc[pd.to_datetime(dnav["trade_date"]).between(start, end)]))
            elif key == default_key and len(nav):
                def_daily_parts.append(daily_from_nav(nav.loc[pd.to_datetime(nav["trade_date"]).between(start, end)]))

        # 若默认尚未写出，先确保默认块存在
        if key != default_key:
            for start, end, tag in blocks:
                def_dest = dest_root / f"{year}" / default_key / tag
                if not (def_dest / "nav.parquet").is_file():
                    run_windowed_sleeve(
                        def_dest,
                        default_scores,
                        signal_start=start,
                        signal_end=end,
                        backtest_start=start,
                        backtest_end=end,
                        extra={"code": f"R233_INNER_{year}_{default_key}_{tag}", "round": "23.3"},
                        resume=resume,
                    )

        mean_u = float(np.mean([u for u in utils if u is not None])) if any(u is not None for u in utils) else None
        maxdd = float(np.min(dds)) if dds else None
        # paired block deltas
        mean_delta = None
        boot_lb = None
        if key != default_key:
            # rebuild default dailies
            def_parts = []
            cand_parts = []
            for start, end, tag in blocks:
                c_path = dest_root / f"{year}" / key / tag / "nav.parquet"
                d_path = dest_root / f"{year}" / default_key / tag / "nav.parquet"
                if c_path.is_file() and d_path.is_file():
                    cnav = pd.read_parquet(c_path)
                    dnav = pd.read_parquet(d_path)
                    c = daily_from_nav(cnav.loc[pd.to_datetime(cnav["trade_date"]).between(start, end)])
                    d = daily_from_nav(dnav.loc[pd.to_datetime(dnav["trade_date"]).between(start, end)])
                    both = pd.concat([c.rename("c"), d.rename("b")], axis=1, join="inner")
                    if both.isna().any().any() or len(both) == 0:
                        block_deltas.append(None)
                    else:
                        bd = float((both["c"] - both["b"]).mean())
                        block_deltas.append(bd)
                        cand_parts.append(both["c"])
                        def_parts.append(both["b"])
                else:
                    block_deltas.append(None)
            if cand_parts and def_parts:
                cand = pd.concat(cand_parts)
                base = pd.concat(def_parts)
                cand = cand[~cand.index.duplicated(keep="last")].sort_index()
                base = base[~base.index.duplicated(keep="last")].sort_index()
                both = pd.concat([cand.rename("c"), base.rename("b")], axis=1, join="inner")
                if both.isna().any().any():
                    mean_delta = None
                    boot_lb = None
                else:
                    diff = (both["c"] - both["b"]).to_numpy()
                    mean_delta = float(diff.mean())
                    boot_lb = _block_bootstrap_lb(diff)

        # default MaxDD reference
        def_dds = []
        for start, end, tag in blocks:
            p = dest_root / f"{year}" / default_key / tag / "nav.parquet"
            if p.is_file():
                def_dds.append(_maxdd(pd.read_parquet(p)))
        def_dd = float(np.min(def_dds)) if def_dds else None
        dd_ok = True if maxdd is None or def_dd is None else (maxdd >= def_dd - MAXDD_WORSEN)
        both_blocks_nonneg = (
            len(block_deltas) == 2
            and all(d is not None for d in block_deltas)
            and all(float(d) >= 0.0 for d in block_deltas)  # type: ignore[arg-type]
        )
        is_default = key == default_key
        gates = (
            (not window_fail)
            and n_days >= MIN_INNER_DAYS
            and n_trades >= MIN_INNER_TRADES
            and mean_delta is not None
            and mean_delta > 0
            and both_blocks_nonneg
            and dd_ok
            and boot_lb is not None
            and np.isfinite(boot_lb)
            and boot_lb > 0
        )
        if is_default:
            gates = not window_fail
        # distance
        try:
            if key.startswith("t_"):
                dist = abs(float(key.split("_")[1]) - float(DEFAULT_W_TEN))
            elif key.startswith("w_") or key.startswith("l_"):
                dist = abs(float(key.split("_")[1]) - float(DEFAULT_W_LGBM))
            else:
                dist = 0.0 if is_default else 1.0
        except Exception:
            dist = 0.0 if is_default else 1.0
        records.append(
            CandidateRecord(
                key=key,
                is_default=is_default,
                inner_utility=mean_u,
                distance_to_default=float(dist),
                n_days=n_days,
                n_trades=n_trades,
                maxdd=maxdd,
                mean_delta=mean_delta,
                block_deltas=block_deltas,
                boot_lb=boot_lb,
                all_gates_pass=bool(gates),
                window_fail=window_fail,
                details={"dd_ok": dd_ok, "both_blocks_nonneg": both_blocks_nonneg},
            )
        )

    # 确保默认先于非默认评估：若顺序导致默认未跑，上面已补跑
    decision = choose_weight(records, default_key=default_key)
    return {
        "year": year,
        "chosen": decision["chosen"],
        "reason": decision["reason"],
        "records": [
            {
                "year": year,
                "key": r.key,
                "is_default": r.is_default,
                "utility": r.inner_utility,
                "n_days": r.n_days,
                "n_trades": r.n_trades,
                "maxdd": r.maxdd,
                "mean_delta": r.mean_delta,
                "block_deltas": r.block_deltas,
                "boot_lb": r.boot_lb,
                "all_gates_pass": r.all_gates_pass,
                "window_fail": r.window_fail,
                "distance_to_default": r.distance_to_default,
                **r.details,
            }
            for r in records
        ],
        "fold_rows": fold_rows,
    }
