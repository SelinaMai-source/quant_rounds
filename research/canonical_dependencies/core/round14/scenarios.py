"""固定情景矩阵 S0–S9。不新发明切点。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND14_ROOT
from ..round11.common import exclude_year_cagr, max_drawdown_of, rolling_cagr, sharpe_of
from ..round12.deploy import apply_liquidity_priority
from ..round12.runner import hist_hooks
from .candidates import load_candidates, load_daily_map
from .common import cagr_of, dump_json, load_json, pack_daily, pct, write_md
from .contract import CANDIDATE_IDS, R14_00, S8_UP_RATIO
from .freeze import require_frozen
from .runner import load_market, run_r14_backtest
from .weights import load_overlayed

SCEN_JSON = ROUND14_ROOT / "scenarios" / "scenarios.json"
SCEN_MD = ROUND14_ROOT / "05_scenarios.md"
PROMOTION_SCENARIOS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")
IDS = ("v4", *CANDIDATE_IDS)
CODE_FOR = {"v4": R14_00, **{c: c for c in CANDIDATE_IDS}}


def _ov_code(sid: str) -> str:
    return "v4" if sid == "v4" else sid


def _subset_metrics(daily: pd.Series, mask: pd.Series) -> dict[str, Any]:
    sub = daily.loc[mask.reindex(daily.index).fillna(False)]
    return pack_daily(sub)


def _zero_years(daily: pd.Series, years: tuple[int, ...]) -> pd.Series:
    adj = daily.copy()
    adj.loc[adj.index.year.isin(years)] = 0.0
    return adj


def _run_tagged(code: str, overlayed, hooks, cash, extra, resume: bool, loaded) -> dict[str, Any]:
    dest = ROUND14_ROOT / "scenarios" / code
    ran = run_r14_backtest(
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
            pack["cagr"] = exclude_year_cagr(daily, years[0]) if len(years) == 1 else cagr_of(adj)
            pack["status"] = "DONE"
            pack["years_zeroed"] = list(years)
            matrix[sid][key] = pack
        rolled = rolling_cagr(daily, 252)
        if len(rolled):
            idx = rolled.idxmin()
            window = daily.loc[idx - pd.tseries.offsets.BDay(400): idx].tail(252)
            matrix[sid]["S9"] = {
                "status": "DONE",
                "cagr": float(rolled.min()) if np.isfinite(rolled.min()) else None,
                "sharpe": sharpe_of(window),
                "max_drawdown": max_drawdown_of(window),
                "window_end": str(pd.Timestamp(idx).date()),
                "n_days": int(len(window)),
            }
        else:
            matrix[sid]["S9"] = {"status": "TOO_SHORT"}
        up_idx = up.reindex(daily.index)
        weak = up_idx < S8_UP_RATIO
        matrix[sid]["S8"] = {
            "status": "DONE",
            "rule": f"B01A up_ratio < {S8_UP_RATIO}",
            "n_days": int(weak.fillna(False).sum()),
            **_subset_metrics(daily, weak.fillna(False)),
        }

    summary: dict[str, Any] = {}
    for sid in IDS:
        sharpes, cagrs, dds = [], [], []
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

    payload = {
        "status": "DONE",
        "s8_rule": f"B01A up_ratio < {S8_UP_RATIO}",
        "s9": "worst historical rolling 12M",
        "promotion_scenarios": list(PROMOTION_SCENARIOS),
        "matrix": matrix,
        "summary": summary,
        "no_new_cutoffs": True,
    }
    dump_json(SCEN_JSON, payload)
    lines = [
        "# Round 14 · 情景矩阵",
        "",
        f"S8 = B01A `up_ratio < {S8_UP_RATIO}`。S9 = 最差滚动 12M。未调切点。",
        "",
        "| ID | median Sharpe | worst Sharpe | median CAGR | worst CAGR |",
        "|---|---:|---:|---:|---:|",
    ]
    for sid in IDS:
        s = summary[sid]
        lines.append(
            f"| `{sid}` | {s.get('median_sharpe')} | {s.get('worst_sharpe')} | "
            f"{pct(s.get('median_cagr'))} | {pct(s.get('worst_cagr'))} |"
        )
    write_md(SCEN_MD, "\n".join(lines) + "\n")
    return payload


def load_scenarios() -> dict[str, Any]:
    return load_json(SCEN_JSON)


__all__ = ["PROMOTION_SCENARIOS", "load_scenarios", "run_scenarios"]
