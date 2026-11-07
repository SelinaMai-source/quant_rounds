"""matched placebo：当日真实 candidate 内随机 Top1/Top2。禁止全市场乱抽。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..local_executor import HISTORICAL_PRE_CUTOVER_SELL_TAX_BPS, STAMP_TAX_CUTOVER, historical_sell_tax_bps
from ..paths import ROUND11_ROOT
from ..replay_round7.inject import inject_round7_into_overlay_caches, load_locked_alpha
from ..round9.overlays import build_overlayed
from ..stock_readonly import load_prepared_market
from .common import daily_from_nav, dump_json, load_anchor_nav, sharpe_of, stage_dir, write_md
from .contract import BASELINES
from .cost import load_cost_status

PLACEBO_MD = ROUND11_ROOT / "07_matched_placebo.md"
N_PLACEBO = 1000
SLIP_BPS = 12.3
BUY_FEE = (1.0 + 0.1) / 10_000.0
LIMIT_TOL = 0.005


def _candidates_and_slots() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inject_round7_into_overlay_caches()
    pred = load_locked_alpha()
    overlayed, _, _ = build_overlayed({"kind": "preregistered", "allocator": "B03_conservative"}, pred)
    work = overlayed.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rank"] = work.groupby("signal_date")["score"].rank(method="first", ascending=False)
    slots = (
        work.loc[work["_rank"].isin([1, 2]), ["signal_date", "_rank", "allocation_multiplier", "instrument_id"]]
        .copy()
    )
    cands = work[["signal_date", "instrument_id"]].drop_duplicates()
    return cands, slots, work


def _market_frame(prepared) -> pd.DataFrame:
    mk = prepared.market_lookup.reset_index()
    mk["trade_date"] = pd.to_datetime(mk["trade_date"]).dt.normalize()
    mk["instrument_id"] = mk["instrument_id"].astype(str)
    keep = [c for c in ("trade_date", "instrument_id", "open", "high", "low", "close",
                         "volume", "paused", "limit_up", "limit_down") if c in mk.columns]
    return mk[keep]


def _modeled(ref: pd.Series, side: str, bar: pd.DataFrame) -> pd.Series:
    sign = 1.0 if side == "buy" else -1.0
    modeled = ref.astype(float) * (1.0 + sign * SLIP_BPS / 10_000.0)
    lo = bar["low"].astype(float)
    hi = bar["high"].astype(float)
    if "limit_down" in bar.columns:
        ld = pd.to_numeric(bar["limit_down"], errors="coerce")
        lo = np.maximum(lo, ld.fillna(-np.inf))
    if "limit_up" in bar.columns:
        lu = pd.to_numeric(bar["limit_up"], errors="coerce")
        hi = np.minimum(hi, lu.fillna(np.inf))
    return modeled.clip(lower=lo, upper=hi)


def _buy_ok(bar: pd.DataFrame) -> pd.Series:
    paused = bar["paused"].fillna(False).astype(bool) if "paused" in bar.columns else False
    vol = pd.to_numeric(bar["volume"], errors="coerce")
    opn = pd.to_numeric(bar["open"], errors="coerce")
    lu = pd.to_numeric(bar["limit_up"], errors="coerce") if "limit_up" in bar.columns else pd.Series(np.nan, index=bar.index)
    limit_up_open = lu.notna() & opn.notna() & (opn >= lu - LIMIT_TOL)
    return (~paused) & vol.gt(0) & opn.notna() & ~limit_up_open


def _sell_ok(bar: pd.DataFrame) -> pd.Series:
    paused = bar["paused"].fillna(False).astype(bool) if "paused" in bar.columns else False
    vol = pd.to_numeric(bar["volume"], errors="coerce")
    close = pd.to_numeric(bar["close"], errors="coerce")
    lu = pd.to_numeric(bar["limit_up"], errors="coerce") if "limit_up" in bar.columns else pd.Series(np.nan, index=bar.index)
    ld = pd.to_numeric(bar["limit_down"], errors="coerce") if "limit_down" in bar.columns else pd.Series(np.nan, index=bar.index)
    high = pd.to_numeric(bar["high"], errors="coerce")
    limit_up_hold = lu.notna() & close.notna() & (close >= lu - LIMIT_TOL)
    one_ld = ld.notna() & high.notna() & (high <= ld + 1e-12)
    return (~paused) & vol.gt(0) & close.notna() & ~limit_up_hold & ~one_ld


def build_unit_returns(cands: pd.DataFrame, prepared) -> pd.DataFrame:
    """每个 (signal_date, instrument) 的隔离成交 unit_ret。legacy 5bp 与 corrected 按日出税。"""
    calendar = pd.DatetimeIndex(pd.to_datetime(list(prepared.calendar))).normalize()
    pos = {d: i for i, d in enumerate(calendar)}
    mk = _market_frame(prepared)
    work = cands.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["off"] = work["signal_date"].map(pos)
    work = work.loc[work["off"].notna() & (work["off"] + 2 < len(calendar))].copy()
    work["off"] = work["off"].astype(int)
    work["entry_date"] = calendar[work["off"].to_numpy() + 1]
    buy = work.merge(
        mk, left_on=["entry_date", "instrument_id"], right_on=["trade_date", "instrument_id"],
        how="left", suffixes=("", "_mk"),
    )
    ok = _buy_ok(buy)
    buy = buy.loc[ok].copy()
    if buy.empty:
        return pd.DataFrame(columns=["signal_date", "instrument_id", "unit_legacy", "unit_corrected", "filled"])
    buy["entry_px"] = _modeled(buy["open"], "buy", buy)
    remaining = buy[["signal_date", "instrument_id", "off", "entry_px"]].copy()
    remaining["_id"] = np.arange(len(remaining))
    sold_parts = []
    taken = set()
    for lag in range(0, 10):
        if remaining.empty:
            break
        try_off = remaining["off"].to_numpy() + 2 + lag
        valid = try_off < len(calendar)
        chunk = remaining.loc[valid].copy()
        if chunk.empty:
            break
        chunk["try_date"] = calendar[try_off[valid]]
        merged = chunk.merge(
            mk, left_on=["try_date", "instrument_id"], right_on=["trade_date", "instrument_id"],
            how="left",
        )
        can = _sell_ok(merged)
        got = merged.loc[can].copy()
        if not got.empty:
            got["exit_px"] = _modeled(got["close"], "sell", got)
            sold_parts.append(got[["_id", "signal_date", "instrument_id", "entry_px", "exit_px", "try_date"]])
            taken.update(got["_id"].tolist())
            remaining = remaining.loc[~remaining["_id"].isin(taken)]
    if not sold_parts:
        return pd.DataFrame(columns=["signal_date", "instrument_id", "unit_legacy", "unit_corrected", "filled"])
    sold = pd.concat(sold_parts, ignore_index=True)
    sold["tax_corr"] = np.where(
        pd.to_datetime(sold["try_date"]) < STAMP_TAX_CUTOVER,
        HISTORICAL_PRE_CUTOVER_SELL_TAX_BPS,
        5.0,
    )
    sell_fee_leg = (1.0 + 0.1 + 5.0) / 10_000.0
    sell_fee_corr = (1.0 + 0.1 + sold["tax_corr"].astype(float)) / 10_000.0
    sold["unit_legacy"] = (sold["exit_px"] * (1.0 - sell_fee_leg)) / (sold["entry_px"] * (1.0 + BUY_FEE)) - 1.0
    sold["unit_corrected"] = (sold["exit_px"] * (1.0 - sell_fee_corr)) / (sold["entry_px"] * (1.0 + BUY_FEE)) - 1.0
    sold["filled"] = True
    return sold[["signal_date", "instrument_id", "unit_legacy", "unit_corrected", "filled"]]


def _slot_weights(slots: pd.DataFrame) -> dict[Any, tuple[float, float]]:
    s = slots.copy()
    s["signal_date"] = pd.to_datetime(s["signal_date"]).dt.normalize()
    out: dict[Any, tuple[float, float]] = {}
    for signal, g in s.groupby("signal_date"):
        m1 = g.loc[g["_rank"].eq(1), "allocation_multiplier"]
        m2 = g.loc[g["_rank"].eq(2), "allocation_multiplier"]
        w1 = 0.3 * float(m1.iloc[0]) if len(m1) else 0.0
        w2 = 0.1 * float(m2.iloc[0]) if len(m2) else 0.0
        out[pd.Timestamp(signal)] = (w1, w2)
    return out


def _day_ret(names: list[str], ret_map: dict[str, float], weights: tuple[float, float]) -> float:
    acc = 0.0
    tw = 0.0
    for inst, w in zip(names, weights):
        if inst not in ret_map:
            continue
        acc += w * float(ret_map[inst])
        tw += w
    return float(acc) if tw > 0 else 0.0


def run_placebo(*, resume: bool = True, jobs: int = 1, **_: Any) -> dict[str, Any]:
    _ = jobs
    dest = stage_dir("placebo")
    summary_path = dest / "placebo.json"
    if resume and summary_path.is_file():
        import json

        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("n_placebo") == N_PLACEBO and existing.get("status") == "DONE":
            return existing

    cands, slots, overlayed = _candidates_and_slots()
    loaded = load_prepared_market(warmup_ma5=True)
    fills = build_unit_returns(cands, loaded["prepared"])
    fills["signal_date"] = pd.to_datetime(fills["signal_date"]).dt.normalize()
    fills["instrument_id"] = fills["instrument_id"].astype(str)
    weights = _slot_weights(slots)

    by_day_leg: dict[Any, dict[str, float]] = {}
    by_day_corr: dict[Any, dict[str, float]] = {}
    pools: dict[Any, np.ndarray] = {}
    for signal, g in fills.groupby("signal_date"):
        signal = pd.Timestamp(signal)
        by_day_leg[signal] = dict(zip(g["instrument_id"], g["unit_legacy"].astype(float)))
        by_day_corr[signal] = dict(zip(g["instrument_id"], g["unit_corrected"].astype(float)))
        pools[signal] = g["instrument_id"].astype(str).to_numpy()

    dates = sorted(d for d in pools if d in weights)
    actual_picks: dict[Any, list[str]] = {}
    ov = overlayed.copy()
    ov["signal_date"] = pd.to_datetime(ov["signal_date"]).dt.normalize()
    for signal, g in ov.groupby("signal_date"):
        signal = pd.Timestamp(signal)
        picks = g.sort_values("_rank")["instrument_id"].astype(str).head(2).tolist()
        actual_picks[signal] = picks

    def series_from_picks(picks: dict[Any, list[str]], store: dict[Any, dict[str, float]]) -> pd.Series:
        out = {}
        for signal in dates:
            out[signal] = _day_ret(picks.get(signal) or [], store[signal], weights[signal])
        return pd.Series(out, dtype=float)

    actual = {
        "legacy": series_from_picks(actual_picks, by_day_leg),
        "corrected": series_from_picks(actual_picks, by_day_corr),
    }

    rng = np.random.default_rng(20261008)
    trials: dict[str, list[dict[str, Any]]] = {"legacy": [], "corrected": []}
    for i in range(N_PLACEBO):
        picks = {}
        for signal in dates:
            pool = pools[signal]
            if len(pool) < 2:
                picks[signal] = list(pool)
            else:
                idx = rng.choice(len(pool), size=2, replace=False)
                picks[signal] = [str(pool[idx[0]]), str(pool[idx[1]])]
        for fee_key, store in (("legacy", by_day_leg), ("corrected", by_day_corr)):
            s = series_from_picks(picks, store)
            wealth = float((1.0 + s.fillna(0.0)).prod())
            years = max(len(s) / 252.0, 1e-9)
            trials[fee_key].append({
                "i": i,
                "cagr": wealth ** (1.0 / years) - 1.0,
                "sharpe": sharpe_of(s.fillna(0.0)),
                "n": int(len(s)),
            })

    def _percentile(actual_s: pd.Series, trial_rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
        wealth = float((1.0 + actual_s.fillna(0.0)).prod())
        years = max(len(actual_s) / 252.0, 1e-9)
        actual_v = wealth ** (1.0 / years) - 1.0 if field == "cagr" else sharpe_of(actual_s.fillna(0.0))
        arr = np.array([r[field] for r in trial_rows if r.get(field) is not None], dtype=float)
        arr = arr[np.isfinite(arr)]
        if actual_v is None or not len(arr):
            return {"actual": actual_v, "percentile": None}
        return {
            "actual": float(actual_v),
            "placebo_median": float(np.median(arr)),
            "placebo_p95": float(np.percentile(arr, 95)),
            "percentile": float(np.mean(arr <= actual_v) * 100.0),
            "n": int(len(arr)),
        }

    shuffle_stats: dict[str, Any] = {}
    count_by_day = {pd.Timestamp(d): int(len(p)) for d, p in pools.items()}
    for fee_key in ("legacy", "corrected"):
        act = actual[fee_key]
        meta = pd.DataFrame({"signal_date": act.index, "ret": act.values})
        meta["year"] = pd.to_datetime(meta["signal_date"]).dt.year
        meta["count"] = meta["signal_date"].map(count_by_day)
        meta["count_q"] = pd.qcut(meta["count"].rank(method="first"), 4, labels=False, duplicates="drop")
        rng2 = np.random.default_rng(11)
        shuf = []
        for _ in range(N_PLACEBO):
            parts = []
            for _, block in meta.groupby(["year", "count_q"], sort=False):
                vals = block["ret"].to_numpy().copy()
                rng2.shuffle(vals)
                parts.append(pd.Series(vals, index=block["signal_date"]))
            s = pd.concat(parts).sort_index()
            wealth = float((1.0 + s).prod())
            years = max(len(s) / 252.0, 1e-9)
            shuf.append(wealth ** (1.0 / years) - 1.0)
        actual_cagr = float((1.0 + act).prod() ** (1.0 / max(len(act) / 252.0, 1e-9)) - 1.0)
        arr = np.array(shuf, dtype=float)
        shuffle_stats[fee_key] = {
            "actual_cagr": actual_cagr,
            "shuffle_median": float(np.median(arr)),
            "percentile": float(np.mean(arr <= actual_cagr) * 100.0),
            "note": "year×count-quantile 内打乱信号日收益，破 timing，保留活跃度。",
        }

    v4_nav_sharpe = sharpe_of(daily_from_nav(load_anchor_nav(BASELINES["v4"])))
    cost = load_cost_status()
    v4c_path = ((cost.get("corrected") or {}).get("v4") or {}).get("nav_path")
    v4c_sharpe = sharpe_of(daily_from_nav(pd.read_parquet(v4c_path))) if v4c_path else None

    payload = {
        "status": "DONE",
        "n_placebo": N_PLACEBO,
        "universe": "daily locked-alpha candidate set after gap floor (NOT full market)",
        "method": "PLACEBO_DATE_WEIGHTED_ISOLATED_FILL",
        "matched": [
            "signal_date", "Top1/Top2 slots", "30/10 budget weights",
            "day allocation_multiplier", "exit/fees in unit_ret",
        ],
        "not_full_engine": True,
        "n_signal_days": int(len(dates)),
        "n_candidate_rows": int(len(cands)),
        "n_filled_units": int(len(fills)),
        "legacy": {
            "cagr": _percentile(actual["legacy"], trials["legacy"], "cagr"),
            "sharpe": _percentile(actual["legacy"], trials["legacy"], "sharpe"),
            "engine_v4_sharpe": v4_nav_sharpe,
        },
        "corrected": {
            "cagr": _percentile(actual["corrected"], trials["corrected"], "cagr"),
            "sharpe": _percentile(actual["corrected"], trials["corrected"], "sharpe"),
            "engine_v4_sharpe": v4c_sharpe,
        },
        "timing_shuffle": shuffle_stats,
        "trials_head": {k: v[:5] for k, v in trials.items()},
        "post_hoc_diagnostic": True,
        "historical_sell_tax_helper": str(historical_sell_tax_bps(STAMP_TAX_CUTOVER - pd.Timedelta(days=1))),
    }
    for fee_key in ("legacy", "corrected"):
        pd.DataFrame(trials[fee_key]).to_parquet(dest / f"placebo_{fee_key}.parquet", index=False)
    dump_json(summary_path, payload)
    focus = payload["corrected"]["cagr"]
    lines = [
        "# Round 11 · matched placebo",
        "",
        f"n={N_PLACEBO}。宇宙 = **当日 locked-alpha 候选**（gap floor 后），禁止全市场乱抽。",
        "匹配：日期 / Top1·Top2 槽位 / 30/10 预算权重 / 当日风险乘子 / 退出与费用（unit fill）。",
        "方法 = `PLACEBO_DATE_WEIGHTED_ISOLATED_FILL`（非全引擎 path-dependent）。",
        "",
        "## 关键：corrected 百分位",
        "",
        f"- actual CAGR proxy = `{focus.get('actual')}`",
        f"- placebo median = `{focus.get('placebo_median')}`",
        f"- **percentile = {focus.get('percentile')}**（n={focus.get('n')}）",
        "",
        "## legacy vs corrected",
        "",
        "| 套 | 指标 | actual | placebo median | percentile |",
        "|---|---|---:|---:|---:|",
    ]
    for fee in ("legacy", "corrected"):
        for field in ("cagr", "sharpe"):
            row = payload[fee][field]
            lines.append(
                f"| {fee} | {field} | {row.get('actual')} | {row.get('placebo_median')} | {row.get('percentile')} |"
            )
    lines += [
        "",
        "## year×count timing shuffle",
        "",
        f"- corrected percentile = {(shuffle_stats.get('corrected') or {}).get('percentile')}",
        f"- {(shuffle_stats.get('corrected') or {}).get('note')}",
        "",
    ]
    write_md(PLACEBO_MD, "\n".join(lines))
    return payload


def load_placebo() -> dict[str, Any]:
    import json

    path = ROUND11_ROOT / "placebo" / "placebo.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


__all__ = ["load_placebo", "run_placebo"]
