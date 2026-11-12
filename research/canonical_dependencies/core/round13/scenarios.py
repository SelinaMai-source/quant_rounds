"""固定情景矩阵 S0–S10 + Pareto / minimax / max-regret。S9 不编切点。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND13_ROOT
from ..round11.common import exclude_year_cagr, max_drawdown_of, rolling_cagr, sharpe_of
from ..round12.deploy import apply_liquidity_priority
from ..round12.runner import hist_hooks
from .candidates import load_candidates, load_daily_map
from .common import cagr_of, dump_json, load_json, pack_daily, pct, write_md
from .contract import (
    CANDIDATE_IDS,
    CONTRACT_CASH,
    R13_00,
    R13_CTRL_V2,
    R13_CTRL_V3,
    S10_UP_RATIO,
    S9_STATUS,
    STAT_IDS,
    experiment_dir,
)
from .freeze import require_frozen
from .runner import load_market, run_r13_backtest
from .weights import load_overlayed

SCEN_JSON = ROUND13_ROOT / "scenarios" / "scenarios.json"
SCEN_MD = ROUND13_ROOT / "03_scenarios.md"
PROMOTION_SCENARIOS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S10")
IDS = ("v2", "v3", "v4", *CANDIDATE_IDS)
CODE_FOR = {"v2": R13_CTRL_V2, "v3": R13_CTRL_V3, "v4": R13_00, **{c: c for c in CANDIDATE_IDS}}


def _ov_code(sid: str) -> str:
    return {"v2": "v2", "v3": "v3", "v4": "v4"}.get(sid, sid)


def _subset_metrics(daily: pd.Series, mask: pd.Series) -> dict[str, Any]:
    sub = daily.loc[mask.reindex(daily.index).fillna(False)]
    return pack_daily(sub)


def _zero_years(daily: pd.Series, years: tuple[int, ...]) -> pd.Series:
    adj = daily.copy()
    adj.loc[adj.index.year.isin(years)] = 0.0
    return adj


def _run_tagged(code: str, overlayed, hooks, cash, extra, resume: bool, loaded) -> dict[str, Any]:
    dest = ROUND13_ROOT / "scenarios" / code
    ran = run_r13_backtest(
        dest,
        overlayed=overlayed,
        validation_hooks=hooks,
        initial_cash=cash,
        extra_report=extra,
        resume=resume,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    report = ran.get("report") or {}
    m = report.get("metrics") or {}
    return {
        "code": code,
        "cagr": m.get("annualized_return"),
        "sharpe": m.get("sharpe_zero_rate"),
        "max_drawdown": m.get("max_drawdown"),
        "completed_trades": m.get("completed_trades"),
        "nav_path": str(dest / "nav.parquet"),
        "status": ran.get("status"),
        **extra,
    }


def run_scenarios(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    dailies = load_daily_map()
    cand = load_candidates()
    loaded = load_market()
    from ..round2.data import load_regime_frame

    regime = load_regime_frame()
    up = pd.to_numeric(regime["up_ratio"], errors="coerce")
    down_p = pd.to_numeric(regime["down_pressure_p250"], errors="coerce") if "down_pressure_p250" in regime.columns else None

    matrix: dict[str, dict[str, Any]] = {sid: {} for sid in IDS}
    extra_runs: dict[str, Any] = {}

    for sid in IDS:
        daily = dailies[sid]
        base = pack_daily(daily)
        row = (cand.get("results") or {}).get(CODE_FOR[sid]) or {}
        if row.get("cagr") is not None:
            base["cagr"] = row.get("cagr")
            base["sharpe"] = row.get("sharpe")
            base["max_drawdown"] = row.get("max_drawdown")
        matrix[sid]["S0"] = {"status": "DONE", **base}

    for sid in IDS:
        ov = load_overlayed(_ov_code(sid))
        extra_runs[f"{sid}_S1"] = _run_tagged(
            f"{sid}_S1_20bp", ov, hist_hooks(20.0), None,
            {"scenario": "S1", "buy_extra_slippage_bps": 20.0, "parent": sid}, resume, loaded,
        )
        extra_runs[f"{sid}_S2"] = _run_tagged(
            f"{sid}_S2_30bp", ov, hist_hooks(30.0), None,
            {"scenario": "S2", "buy_extra_slippage_bps": 30.0, "parent": sid}, resume, loaded,
        )
        ov1, meta1 = apply_liquidity_priority(ov, capital=1_000_000.0)
        extra_runs[f"{sid}_S3"] = _run_tagged(
            f"{sid}_S3_1M", ov1, hist_hooks(), 1_000_000.0,
            {"scenario": "S3", "capital": 1_000_000.0, "liquidity": meta1, "parent": sid}, resume, loaded,
        )
        ov3, meta3 = apply_liquidity_priority(ov, capital=3_000_000.0)
        extra_runs[f"{sid}_S4"] = _run_tagged(
            f"{sid}_S4_3M", ov3, hist_hooks(), 3_000_000.0,
            {"scenario": "S4", "capital": 3_000_000.0, "liquidity": meta3, "parent": sid}, resume, loaded,
        )
        matrix[sid]["S1"] = extra_runs[f"{sid}_S1"]
        matrix[sid]["S2"] = extra_runs[f"{sid}_S2"]
        matrix[sid]["S3"] = extra_runs[f"{sid}_S3"]
        matrix[sid]["S4"] = extra_runs[f"{sid}_S4"]

    for sid in IDS:
        daily = dailies[sid]
        for key, years in (("S5", (2024,)), ("S6", (2025,)), ("S7", (2024, 2025))):
            adj = _zero_years(daily, years)
            pack = pack_daily(adj)
            pack["cagr"] = exclude_year_cagr(daily, years[0]) if len(years) == 1 else pack["cagr"]
            if len(years) == 2:
                adj2 = daily.copy()
                adj2.loc[adj2.index.year.isin(years)] = 0.0
                pack["cagr"] = cagr_of(adj2)
            pack["status"] = "DONE"
            pack["years_zeroed"] = list(years)
            matrix[sid][key] = pack
        rolled = rolling_cagr(daily, 252)
        if len(rolled):
            idx = rolled.idxmin()
            window = daily.loc[idx - pd.tseries.offsets.BDay(400): idx].tail(252)
            matrix[sid]["S8"] = {
                "status": "DONE",
                "cagr": float(rolled.min()) if np.isfinite(rolled.min()) else None,
                "sharpe": sharpe_of(window),
                "max_drawdown": max_drawdown_of(window),
                "window_end": str(pd.Timestamp(idx).date()),
                "n_days": int(len(window)),
            }
        else:
            matrix[sid]["S8"] = {"status": "TOO_SHORT"}
        matrix[sid]["S9"] = {
            "status": S9_STATUS,
            "excluded_from_promotion_worst_median": True,
            "note": "代码库无既有命名 high-vol 切点，不编造。",
        }
        up_idx = up.reindex(daily.index)
        weak = up_idx < S10_UP_RATIO
        matrix[sid]["S10"] = {
            "status": "DONE",
            "rule": f"B01A up_ratio < {S10_UP_RATIO}",
            "n_days": int(weak.fillna(False).sum()),
            **_subset_metrics(daily, weak.fillna(False)),
        }
        if down_p is not None:
            q = down_p.reindex(daily.index)
            hi = q >= 0.75
            matrix[sid]["S9_POST_HOC_DIAGNOSTIC"] = {
                "status": "POST_HOC_DIAGNOSTIC",
                "rule": "down_pressure_p250 >= 0.75 (upper quartile)",
                "n_days": int(hi.fillna(False).sum()),
                **_subset_metrics(daily, hi.fillna(False)),
            }

    # ranks on promotion scenarios (exclude S9)
    rank_tables: dict[str, Any] = {}
    for metric in ("cagr", "sharpe"):
        recs = []
        for scen in PROMOTION_SCENARIOS:
            vals = {}
            for sid in IDS:
                v = (matrix[sid].get(scen) or {}).get(metric)
                vals[sid] = float(v) if v is not None and np.isfinite(v) else np.nan
            s = pd.Series(vals)
            ranks = s.rank(ascending=False, method="average")
            recs.append({"scenario": scen, **{f"rank_{k}": float(ranks[k]) if np.isfinite(ranks[k]) else None for k in IDS}})
            recs[-1].update({f"val_{k}": vals[k] if np.isfinite(vals[k]) else None for k in IDS})
        df = pd.DataFrame(recs)
        rank_tables[metric] = {
            "mean": {k: float(df[f"rank_{k}"].mean()) for k in IDS},
            "std": {k: float(df[f"rank_{k}"].std(ddof=1)) if df[f"rank_{k}"].notna().sum() > 1 else None for k in IDS},
            "top1_freq": {k: float((df[f"rank_{k}"] <= 1.0 + 1e-9).mean()) for k in IDS},
            "top3_freq": {k: float((df[f"rank_{k}"] <= 3.0 + 1e-9).mean()) for k in IDS},
        }

    summary: dict[str, Any] = {}
    for sid in IDS:
        sharpes = []
        cagrs = []
        dds = []
        for scen in PROMOTION_SCENARIOS:
            row = matrix[sid].get(scen) or {}
            if row.get("sharpe") is not None:
                sharpes.append(float(row["sharpe"]))
            if row.get("cagr") is not None:
                cagrs.append(float(row["cagr"]))
            if row.get("max_drawdown") is not None:
                dds.append(float(row["max_drawdown"]))
        summary[sid] = {
            "median_sharpe": float(np.median(sharpes)) if sharpes else None,
            "worst_sharpe": float(np.min(sharpes)) if sharpes else None,
            "median_cagr": float(np.median(cagrs)) if cagrs else None,
            "worst_cagr": float(np.min(cagrs)) if cagrs else None,
            "median_maxdd": float(np.median(dds)) if dds else None,
            "worst_maxdd": float(np.min(dds)) if dds else None,
        }

    # max regret / minimax on Sharpe
    regrets = {sid: [] for sid in IDS}
    for scen in PROMOTION_SCENARIOS:
        vals = {sid: (matrix[sid].get(scen) or {}).get("sharpe") for sid in IDS}
        finite = {k: float(v) for k, v in vals.items() if v is not None and np.isfinite(v)}
        if not finite:
            continue
        best = max(finite.values())
        for sid, v in finite.items():
            regrets[sid].append(best - v)
    max_regret = {sid: (float(np.max(vs)) if vs else None) for sid, vs in regrets.items()}
    minimax = {sid: (summary[sid].get("worst_sharpe")) for sid in IDS}

    # Pareto on (median_sharpe, worst_sharpe, -abs(median_maxdd))
    pts = []
    for sid in IDS:
        s = summary[sid]
        if None in (s.get("median_sharpe"), s.get("worst_sharpe"), s.get("median_maxdd")):
            continue
        pts.append((sid, s["median_sharpe"], s["worst_sharpe"], -abs(s["median_maxdd"])))

    def dominated(a, b):
        return b[1] >= a[1] and b[2] >= a[2] and b[3] >= a[3] and (b[1] > a[1] or b[2] > a[2] or b[3] > a[3])

    pareto = [p[0] for p in pts if not any(dominated(p, q) for q in pts if q[0] != p[0])]

    payload = {
        "status": "DONE",
        "s9": S9_STATUS,
        "s10_rule": f"B01A up_ratio < {S10_UP_RATIO}",
        "promotion_scenarios": list(PROMOTION_SCENARIOS),
        "matrix": matrix,
        "summary": summary,
        "rank_tables": rank_tables,
        "max_regret_sharpe": max_regret,
        "minimax_worst_sharpe": minimax,
        "pareto_ids": pareto,
        "never_invented_s9_cutoff": True,
    }
    dump_json(SCEN_JSON, payload)
    lines = [
        "# Round 13 · 情景矩阵",
        "",
        f"S9 = **{S9_STATUS}**，不进 promotion worst/median。S10 = B01A `up_ratio < {S10_UP_RATIO}`。",
        "",
        "| ID | median Sharpe | worst Sharpe | median CAGR | worst CAGR | max regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sid in IDS:
        s = summary[sid]
        lines.append(
            f"| `{sid}` | {s.get('median_sharpe')} | {s.get('worst_sharpe')} | {pct(s.get('median_cagr'))} | "
            f"{pct(s.get('worst_cagr'))} | {max_regret.get(sid)} |"
        )
    lines += ["", f"Pareto（median/worst Sharpe, -|median MaxDD|）= {pareto}", ""]
    write_md(SCEN_MD, "\n".join(lines))
    return payload


def load_scenarios() -> dict[str, Any]:
    return load_json(SCEN_JSON)


__all__ = ["PROMOTION_SCENARIOS", "load_scenarios", "run_scenarios"]
