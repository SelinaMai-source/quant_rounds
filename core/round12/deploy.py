"""预注册部署变体 R12_02–07。3% participation，Top1 优先。不搜 cap。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND10_ROOT, ROUND12_ROOT
from ..round8.overlays import MIN_KEEP
from ..stock_readonly import load_prepared_market
from .common import dump_json, load_json, pct, write_md
from .contract import (
    CAPITALS,
    CONTRACT_CASH,
    DIAGNOSTIC_CAPITAL,
    PARTICIPATION,
    R12_01,
    R12_02,
    R12_03,
    R12_04,
    R12_05,
    R12_06,
    R12_07,
    RANK_NAV,
    experiment_dir,
)
from .execution import load_execution
from .runner import hist_hooks, load_v4_overlayed, run_r12_backtest
from .universe import load_universe

DEPLOY_MD = ROUND12_ROOT / "04_deploy_variants.md"
DEPLOY_JSON = ROUND12_ROOT / "deploy" / "deploy.json"
CAPACITY_MD = ROUND12_ROOT / "05_capacity.md"
CAPACITY_JSON = ROUND12_ROOT / "capacity" / "capacity.json"


def apply_liquidity_priority(
    overlayed: pd.DataFrame,
    *,
    capital: float = CONTRACT_CASH,
    participation: float = PARTICIPATION,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Top1 先分完，Top2 拿剩余。禁止为保 Top2 稀释 Top1。"""
    amount_path = ROUND10_ROOT / "capacity" / "daily_amount.parquet"
    work = overlayed.copy().reset_index(drop=True)
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    if not amount_path.is_file():
        return work.drop(columns=["_rank"], errors="ignore"), {"status": "NO_AMOUNT_CACHE"}
    amt = pd.read_parquet(amount_path)
    amt["trade_date"] = pd.to_datetime(amt["trade_date"]).dt.normalize()
    amt["instrument_id"] = amt["instrument_id"].astype(str)
    merged = work.merge(
        amt.rename(columns={"trade_date": "signal_date"}),
        on=["signal_date", "instrument_id"],
        how="left",
    )
    nav_path = ROUND10_ROOT / "R10_00_baseline_v4" / "trades.parquet"
    nav_map: dict[Any, float] = {}
    if nav_path.is_file():
        t = pd.read_parquet(nav_path)
        t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
        nav_map = t.groupby("signal_date")["entry_portfolio_nav"].median().astype(float).to_dict()
    scale = float(capital) / float(CONTRACT_CASH)
    fracs = {1: float(RANK_NAV[0]), 2: float(RANK_NAV[1])}
    rank = pd.to_numeric(merged["_rank"], errors="coerce").fillna(0).astype(int)
    nav = merged["signal_date"].map(lambda d: float(nav_map.get(d, CONTRACT_CASH)) * scale)
    frac = rank.map(fracs).fillna(0.0)
    mult = pd.to_numeric(merged["allocation_multiplier"], errors="coerce").fillna(0.0)
    merged["_orig_budget"] = nav.to_numpy() * frac.to_numpy() * mult.to_numpy()
    merged["_cap"] = float(participation) * pd.to_numeric(merged.get("amount"), errors="coerce")
    aligned = pd.Series(mult.to_numpy(), index=merged.index, dtype=float)
    n_top1_cut = n_top2_cut = 0
    for day, g in merged.groupby("signal_date", sort=False):
        day_nav = float(nav_map.get(day, CONTRACT_CASH)) * scale
        leftover = None
        g1 = g.loc[g["_rank"].eq(1)]
        g2 = g.loc[g["_rank"].eq(2)]
        if len(g1):
            i = g1.index[0]
            cap = g1.loc[i, "_cap"]
            budget = float(g1.loc[i, "_orig_budget"])
            take = budget if not np.isfinite(cap) else min(budget, float(cap))
            if budget > 1e-9:
                ratio = take / budget
                aligned.loc[i] = float(merged.loc[i, "allocation_multiplier"]) * ratio
                if ratio < 0.999:
                    n_top1_cut += 1
            leftover = max((float(RANK_NAV[0]) + float(RANK_NAV[1])) * day_nav - take, 0.0)
        if len(g2):
            i = g2.index[0]
            cap = g2.loc[i, "_cap"]
            budget = float(g2.loc[i, "_orig_budget"])
            take = budget
            if np.isfinite(cap):
                take = min(take, float(cap))
            if leftover is not None:
                take = min(take, leftover)
            if budget > 1e-9:
                ratio = max(take / budget, 0.0)
                aligned.loc[i] = max(float(merged.loc[i, "allocation_multiplier"]) * ratio, MIN_KEEP)
                if ratio < 0.999:
                    n_top2_cut += 1
            elif leftover is not None and leftover <= 1e-9:
                aligned.loc[i] = MIN_KEEP
                n_top2_cut += 1
    work["allocation_multiplier"] = aligned.clip(MIN_KEEP, 1.0).to_numpy()
    n_clip = n_top1_cut + n_top2_cut
    meta = {
        "status": "applied",
        "rule": "PRIORITY_3PCT_TOP1_FIRST",
        "participation": participation,
        "capital": capital,
        "n_rows": int(len(work)),
        "n_multiplier_reduced": n_clip,
        "n_top1_capped": n_top1_cut,
        "n_top2_capped": n_top2_cut,
        "n_censored_names": n_clip,
        "amount_coverage": float(pd.to_numeric(merged.get("amount"), errors="coerce").notna().mean()),
        "top1_priority": True,
        "no_proportional_dilution_of_top1": True,
    }
    return work.drop(columns=["_rank"], errors="ignore"), meta


def apply_pit_filter(overlayed: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    from .universe import load_pit

    pit = load_pit()
    if pit.empty:
        return overlayed, {"status": "NO_PIT"}
    work = overlayed.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    elig = pit.loc[pit["pit_eligible"], ["trade_date", "instrument_id"]].rename(columns={"trade_date": "signal_date"})
    elig["_ok"] = True
    merged = work.merge(elig, on=["signal_date", "instrument_id"], how="left")
    before = int(len(work))
    # 不删行以免 Top3 补位；不合格压到 MIN_KEEP
    bad = merged["_ok"].isna()
    work.loc[bad.to_numpy(), "allocation_multiplier"] = MIN_KEEP
    return work, {"status": "applied", "n_zeroed": int(bad.sum()), "n_before": before}


def _exec_bps() -> dict[str, Any]:
    exe = load_execution()
    execp = exe.get("exec") or {}
    return {
        "P50": float(execp.get("P50") or 20.0),
        "P75": float(execp.get("P75") or 30.0),
        "P90": float(execp.get("P90") or 50.0),
        "label": execp.get("label") or "FIXED_STRESS_FALLBACK",
        "fallback": bool(exe.get("fallback", True)),
        "classification": exe.get("classification") or "NOT_IDENTIFIABLE",
    }


def _summarize(ran: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    report = ran.get("report") or {}
    m = report.get("metrics") or {}
    trades = None
    out_dir = ran.get("out_dir")
    if out_dir:
        from pathlib import Path

        tp = Path(out_dir) / "trades.parquet"
        if tp.is_file():
            trades = pd.read_parquet(tp)
    fill = {}
    if trades is not None and not trades.empty:
        t = trades.copy()
        t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
        t["_rank"] = t.groupby("signal_date")["score"].rank(method="first", ascending=False)
        fill = {
            "n_trades": int(len(t)),
            "n_days": int(t["signal_date"].nunique()),
            "n_top1": int((t["_rank"] == 1).sum()),
            "n_top2": int((t["_rank"] == 2).sum()),
            "median_target_budget": float(pd.to_numeric(t["target_name_budget"], errors="coerce").median()),
            "median_buy_cost": float(pd.to_numeric(t["buy_cost"], errors="coerce").median()),
        }
    return {
        "code": ran.get("code"),
        "status": ran.get("status"),
        "out_dir": ran.get("out_dir"),
        "cagr": m.get("annualized_return"),
        "sharpe": m.get("sharpe_zero_rate"),
        "max_drawdown": m.get("max_drawdown"),
        "completed_trades": m.get("completed_trades"),
        "total_return": m.get("total_return"),
        "loyo_median": m.get("loyo_median"),
        "loyo_worst": m.get("loyo_worst"),
        "yearly_returns": m.get("yearly_returns") or {},
        "fill": fill,
        **extra,
    }


def _run(
    code: str,
    *,
    prepared_market,
    market_meta,
    resume: bool,
    overlayed,
    buy_bps: float | None,
    cash: float | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    dest = experiment_dir(code)
    hooks = hist_hooks(buy_bps)
    ran = run_r12_backtest(
        dest,
        prepared_market=prepared_market,
        market_meta=market_meta,
        overlayed=overlayed,
        validation_hooks=hooks,
        initial_cash=cash,
        extra_report=extra,
        resume=resume,
    )
    return _summarize(ran, extra)


def run_deploy(*, resume: bool = True, experiment: str = "", **_: Any) -> dict[str, Any]:
    loaded = load_prepared_market(warmup_ma5=True)
    overlayed, overlay_meta, _ = load_v4_overlayed()
    exe = _exec_bps()
    univ = load_universe()
    quality_ok = bool((univ.get("audit") or {}).get("quality_ok_for_R12_02"))
    results: dict[str, Any] = {}
    wanted = {experiment} if experiment else None

    def need(code: str) -> bool:
        return wanted is None or code in wanted or any(code.startswith(w) for w in (wanted or []))

    if need(R12_02):
        if quality_ok:
            ov, meta = apply_pit_filter(overlayed)
            results[R12_02] = _run(
                R12_02,
                prepared_market=loaded["prepared"],
                market_meta=loaded,
                resume=resume,
                overlayed=ov,
                buy_bps=None,
                cash=None,
                extra={"fee_schedule": "HISTORICAL_FEE_CORRECTED", "pit_filter": meta, "role": "conditional_pit"},
            )
        else:
            results[R12_02] = {
                "code": R12_02,
                "status": "SKIPPED_QUALITY",
                "reason": (univ.get("audit") or {}).get("reason") or "survivorship not PASS",
                "ran": False,
            }

    if need(R12_03):
        results[R12_03] = _run(
            R12_03,
            prepared_market=loaded["prepared"],
            market_meta=loaded,
            resume=resume,
            overlayed=overlayed,
            buy_bps=exe["P50"],
            cash=None,
            extra={
                "fee_schedule": "HISTORICAL_FEE_CORRECTED",
                "buy_extra_slippage_bps": exe["P50"],
                "exec_label": exe["label"],
                "fallback": exe["fallback"],
                "parent": R12_01,
            },
        )
    if need(R12_04):
        results[R12_04] = _run(
            R12_04,
            prepared_market=loaded["prepared"],
            market_meta=loaded,
            resume=resume,
            overlayed=overlayed,
            buy_bps=exe["P75"],
            cash=None,
            extra={
                "fee_schedule": "HISTORICAL_FEE_CORRECTED",
                "buy_extra_slippage_bps": exe["P75"],
                "exec_label": exe["label"],
                "fallback": exe["fallback"],
                "parent": R12_01,
            },
        )

    liq_meta_by_cap: dict[str, Any] = {}
    for cash in CAPITALS:
        tag = "300k" if abs(cash - 300_000) < 1 else ("1M" if abs(cash - 1_000_000) < 1 else "3M")
        ov, meta = apply_liquidity_priority(overlayed, capital=cash)
        liq_meta_by_cap[tag] = meta
        code = R12_05 if abs(cash - CONTRACT_CASH) < 1 else f"{R12_05}_{tag}"
        if need(R12_05) or need(code):
            results[code] = _run(
                code,
                prepared_market=loaded["prepared"],
                market_meta=loaded,
                resume=resume,
                overlayed=ov,
                buy_bps=None,
                cash=cash if abs(cash - CONTRACT_CASH) > 1 else None,
                extra={
                    "fee_schedule": "HISTORICAL_FEE_CORRECTED",
                    "liquidity": meta,
                    "capital": cash,
                    "parent": R12_01,
                    "family": R12_05,
                },
            )
        code6 = R12_06 if abs(cash - CONTRACT_CASH) < 1 else f"{R12_06}_{tag}"
        if need(R12_06) or need(code6):
            results[code6] = _run(
                code6,
                prepared_market=loaded["prepared"],
                market_meta=loaded,
                resume=resume,
                overlayed=ov,
                buy_bps=exe["P50"],
                cash=cash if abs(cash - CONTRACT_CASH) > 1 else None,
                extra={
                    "fee_schedule": "HISTORICAL_FEE_CORRECTED",
                    "liquidity": meta,
                    "buy_extra_slippage_bps": exe["P50"],
                    "capital": cash,
                    "role": "DEPLOYMENT_CANDIDATE" if abs(cash - CONTRACT_CASH) < 1 else "capital_variant",
                    "family": R12_06,
                },
            )
        code7 = R12_07 if abs(cash - CONTRACT_CASH) < 1 else f"{R12_07}_{tag}"
        if need(R12_07) or need(code7):
            results[code7] = _run(
                code7,
                prepared_market=loaded["prepared"],
                market_meta=loaded,
                resume=resume,
                overlayed=ov,
                buy_bps=exe["P75"],
                cash=cash if abs(cash - CONTRACT_CASH) > 1 else None,
                extra={
                    "fee_schedule": "HISTORICAL_FEE_CORRECTED",
                    "liquidity": meta,
                    "buy_extra_slippage_bps": exe["P75"],
                    "capital": cash,
                    "family": R12_07,
                },
            )

    # 10M diagnostic only
    ov10, meta10 = apply_liquidity_priority(overlayed, capital=DIAGNOSTIC_CAPITAL)
    results["R12_05_10M_DIAGNOSTIC"] = _run(
        "R12_05_10M_DIAGNOSTIC",
        prepared_market=loaded["prepared"],
        market_meta=loaded,
        resume=resume,
        overlayed=ov10,
        buy_bps=None,
        cash=DIAGNOSTIC_CAPITAL,
        extra={"post_hoc_diagnostic": True, "capital": DIAGNOSTIC_CAPITAL, "liquidity": meta10, "family": R12_05},
    )

    payload = {
        "status": "DONE",
        "exec": exe,
        "R12_02_ran": bool(results.get(R12_02, {}).get("ran", results.get(R12_02, {}).get("status") not in {"SKIPPED_QUALITY"})),
        "results": results,
        "liquidity_meta": liq_meta_by_cap,
        "no_extra_strategy_ids": True,
        "overlay_meta": overlay_meta,
    }
    dump_json(DEPLOY_JSON, payload)
    _write_deploy_md(payload)
    return payload


def _write_deploy_md(payload: dict[str, Any]) -> None:
    lines = [
        "# Round 12 · 预注册部署变体",
        "",
        "正式策略 ID 仅 R12_00–07。资本后缀与 10M 是同一规则下的容量诊断，不是新策略。",
        "流动性：3% signal-day amount，Top1 先分完，Top2 拿剩余。不搜 cap。",
        "",
        f"EXEC = `{payload.get('exec')}`",
        "",
        "| 代码 | CAGR | Sharpe | MaxDD | 笔数 | 说明 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for code, row in (payload.get("results") or {}).items():
        if row.get("status") == "SKIPPED_QUALITY":
            lines.append(f"| `{code}` | — | — | — | — | SKIPPED {row.get('reason')} |")
            continue
        lines.append(
            f"| `{code}` | {pct(row.get('cagr'))} | {row.get('sharpe')} | {pct(row.get('max_drawdown'))} | "
            f"{row.get('completed_trades')} | {row.get('role') or row.get('family') or ''} |"
        )
    lines += ["", "不因结果发明 25bp 变体。", ""]
    write_md(DEPLOY_MD, "\n".join(lines))


def _cap_rating(results: dict[str, Any]) -> str:
    def sharpe(code: str) -> float | None:
        row = results.get(code) or {}
        v = row.get("sharpe")
        return float(v) if v is not None else None

    s300 = sharpe(R12_05)
    s1 = sharpe(f"{R12_05}_1M")
    s3 = sharpe(f"{R12_05}_3M")
    if s300 is None:
        return "HIGH_RISK"
    if s1 is not None and s1 >= 0.8 * s300 and s3 is not None and s3 >= 0.7 * s300:
        return "UP_TO_3M"
    if s1 is not None and s1 >= 0.8 * s300:
        return "UP_TO_1M"
    return "ROBUST_300K_ONLY"


def run_capacity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    deploy = load_deploy()
    results = deploy.get("results") or {}
    rating = _cap_rating(results)
    payload = {
        "status": "DONE",
        "rating": rating,
        "participation": PARTICIPATION,
        "capitals": list(CAPITALS),
        "diagnostic_10m": results.get("R12_05_10M_DIAGNOSTIC"),
        "rows": {
            k: {kk: vv for kk, vv in row.items() if kk in {"cagr", "sharpe", "max_drawdown", "completed_trades", "capital", "fill"}}
            for k, row in results.items()
            if "R12_05" in k or "R12_06" in k or "R12_07" in k
        },
    }
    dump_json(CAPACITY_JSON, payload)
    write_md(
        CAPACITY_MD,
        "\n".join([
            "# Round 12 · 容量复验",
            "",
            f"预注册规则：3% ADV，Top1 优先。评级 = **{rating}**。",
            "10M 仅 diagnostic。",
            "",
        ]),
    )
    return payload


def load_deploy() -> dict[str, Any]:
    return load_json(DEPLOY_JSON)


def load_capacity() -> dict[str, Any]:
    return load_json(CAPACITY_JSON)


__all__ = [
    "apply_liquidity_priority",
    "apply_pit_filter",
    "load_capacity",
    "load_deploy",
    "run_capacity",
    "run_deploy",
]
