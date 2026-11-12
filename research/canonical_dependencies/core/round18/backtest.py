"""Canonical ledger 复用 Round17：B03 + HISTORICAL_FEE_CORRECTED。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..round17.backtest import apply_b03, fuse_heads, run_canonical, year_stats
from .common import dump_json, finite_or_none, load_json, observed_from_report


def summarize(ran: dict[str, Any]) -> dict[str, Any]:
    obs = ran.get("observed") or observed_from_report(ran.get("report") or {})
    years = year_stats(obs, years=tuple(range(2018, 2027)))
    return {
        "cagr": finite_or_none(obs.get("cagr")),
        "sharpe": finite_or_none(obs.get("sharpe")),
        "max_drawdown": finite_or_none(obs.get("max_drawdown")),
        "completed_trades": obs.get("completed_trades"),
        "total_return": finite_or_none(obs.get("total_return")),
        "yearly_returns": years.get("years") or obs.get("yearly_returns") or {},
        "median_year": years.get("median_year"),
        "worst_year": years.get("worst_year"),
        "status": ran.get("status"),
        "out_dir": ran.get("out_dir"),
    }


def run_fused_backtest(
    dest: Path,
    five: pd.DataFrame,
    ten: pd.DataFrame,
    *,
    extra: dict[str, Any] | None = None,
    resume: bool = True,
    extra_buy_bps: float | None = None,
    initial_cash: float | None = None,
) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    if resume and (dest / "report.json").is_file() and (dest / "nav.parquet").is_file():
        report = load_json(dest / "report.json")
        nav = pd.read_parquet(dest / "nav.parquet")
        return summarize({"status": "resumed", "report": report, "nav": nav, "out_dir": str(dest)})
    fused = fuse_heads(five, ten)
    overlayed, meta = apply_b03(fused)
    fused.to_parquet(dest / "predictions.parquet", index=False)
    dump_json(dest / "overlay_meta.json", meta)
    ran = run_canonical(
        dest,
        overlayed,
        extra=extra or {},
        resume=resume,
        write=True,
        extra_buy_bps=extra_buy_bps,
        initial_cash=initial_cash,
    )
    obs = summarize(ran)
    dump_json(dest / "summary.json", obs)
    return obs
