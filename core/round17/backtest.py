"""Canonical ledger：B03 overlay + HISTORICAL_FEE_CORRECTED。缓存 prepared_market。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import MARKET_END, PREDICTION_START, ROUND17_ROOT, assert_project_result_path
from ..round7.contract import GAP_FLOOR, TEN_HEAD, TEN_WEIGHT
from ..round11.contract import BASELINES
from ..round12.runner import hist_hooks
from ..round15.runner import run_r15_backtest
from ..round11.common import max_drawdown_of
from .common import clip_nav_window, daily_from_nav, dump_json, metrics_from_nav, observed_from_report, yearly_from_nav
from .contract import A00, DEV_END, DEV_START

_CACHE: dict[str, Any] = {}
KEYS = ["signal_date", "instrument_id"]


def cached_market() -> dict[str, Any]:
    if "market" not in _CACHE:
        from ..round15.runner import load_market

        _CACHE["market"] = load_market()
    return _CACHE["market"]


def cached_overlay_v4() -> pd.DataFrame:
    if "ov4" not in _CACHE:
        from ..round15.runner import build_version_overlayed

        ov4, _, _ = build_version_overlayed("v4")
        _CACHE["ov4"] = ov4.copy()
    return _CACHE["ov4"].copy()


def load_gaps() -> pd.DataFrame:
    if "gaps" not in _CACHE:
        panel_path = TEN_HEAD.panel_dir / "candidate_panel.parquet"
        gaps = pd.read_parquet(panel_path, columns=["trade_date", "instrument_id", "d1_open_gap"])
        gaps = gaps.rename(columns={"trade_date": "signal_date"})
        gaps["signal_date"] = pd.to_datetime(gaps["signal_date"])
        gaps["instrument_id"] = gaps["instrument_id"].astype(str)
        _CACHE["gaps"] = gaps
    return _CACHE["gaps"]


def apply_b03(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    from ..round9.overlays import build_overlayed

    spec = BASELINES["v4"]
    overlayed, meta, _cfg = build_overlayed(dict(spec.overlay_spec), predictions)
    return overlayed, meta


def fuse_heads(five: pd.DataFrame, ten: pd.DataFrame) -> pd.DataFrame:
    """90/10 + gap≥-0.01. Inner join on complete-case universes; never fill missing with 0."""
    import numpy as np

    five_p = five[KEYS + ["score"]].copy().rename(columns={"score": "score_five"})
    ten_p = ten[KEYS + ["score"]].copy().rename(columns={"score": "score_ten"})
    five_p["signal_date"] = pd.to_datetime(five_p["signal_date"])
    ten_p["signal_date"] = pd.to_datetime(ten_p["signal_date"])
    five_p["instrument_id"] = five_p["instrument_id"].astype(str)
    ten_p["instrument_id"] = ten_p["instrument_id"].astype(str)
    work = five_p.merge(ten_p, on=KEYS, how="inner")
    work = work.merge(load_gaps(), on=KEYS, how="left")
    work["d1_open_gap"] = pd.to_numeric(work["d1_open_gap"], errors="coerce")
    work = work.loc[np.isfinite(work["d1_open_gap"]) & work["d1_open_gap"].ge(GAP_FLOOR)].copy()
    if work.empty:
        raise RuntimeError("gap eligibility removed all predictions")
    work["pct_five"] = work.groupby("signal_date")["score_five"].rank(method="average", pct=True)
    work["pct_ten"] = work.groupby("signal_date")["score_ten"].rank(method="average", pct=True)
    work["score"] = TEN_WEIGHT * work["pct_ten"] + (1.0 - TEN_WEIGHT) * work["pct_five"]
    return work[KEYS + ["score"]].sort_values(
        ["signal_date", "score", "instrument_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def run_canonical(
    dest: Path | None,
    overlayed: pd.DataFrame,
    *,
    extra: dict[str, Any] | None = None,
    resume: bool = True,
    write: bool = True,
    start_date: date | None = None,
    end_date: date | None = None,
    extra_buy_bps: float | None = None,
    metric_start: date | None = None,
    metric_end: date | None = None,
    initial_cash: float | None = None,
) -> dict[str, Any]:
    loaded = cached_market()
    dest = Path(dest) if dest is not None else ROUND17_ROOT / "nav" / "_ephemeral"
    if write:
        dest.mkdir(parents=True, exist_ok=True)
        assert_project_result_path(dest / "report.json")
    ran = run_r15_backtest(
        dest,
        overlayed=overlayed,
        validation_hooks=hist_hooks(extra_buy_bps=extra_buy_bps),
        extra_report=extra or {},
        resume=resume,
        write=write,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
    )
    nav = ran.get("nav")
    if nav is None and write and (dest / "nav.parquet").is_file():
        nav = pd.read_parquet(dest / "nav.parquet")
    trades = ran.get("trades")
    if trades is None and write and (dest / "trades.parquet").is_file():
        trades = pd.read_parquet(dest / "trades.parquet")
    report = ran.get("report") or {}
    obs = observed_from_report(report)
    if nav is not None and not nav.empty and (metric_start is not None or metric_end is not None):
        clipped = clip_nav_window(
            nav,
            str(metric_start or PREDICTION_START),
            str(metric_end or MARKET_END),
        )
        if len(clipped) >= 2:
            window_metrics = metrics_from_nav(clipped)
            dret = daily_from_nav(clipped)
            first_asset = float(clipped["total_asset"].iloc[0])
            last_asset = float(clipped["total_asset"].iloc[-1])
            obs = {
                "cagr": window_metrics.get("annualized_return"),
                "sharpe": window_metrics.get("sharpe_zero_rate"),
                "max_drawdown": window_metrics.get("max_drawdown") or max_drawdown_of(dret),
                "completed_trades": obs.get("completed_trades"),
                "total_return": window_metrics.get("total_return") if window_metrics.get("total_return") is not None else (
                    last_asset / first_asset - 1.0 if first_asset > 0 else None
                ),
                "loyo_median": window_metrics.get("loyo_median"),
                "loyo_worst": window_metrics.get("loyo_worst"),
                "yearly_returns": yearly_from_nav(clipped) or window_metrics.get("yearly_returns") or {},
                "window_start": str(metric_start or PREDICTION_START),
                "window_end": str(metric_end or MARKET_END),
            }
    ran["observed"] = obs
    ran["nav"] = nav
    ran["trades"] = trades
    if write and dest is not None:
        dump_json(dest / "summary.json", {k: v for k, v in obs.items() if k != "yearly_returns"} | {
            "code": dest.name,
            "yearly_returns": obs.get("yearly_returns") or {},
        })
    return ran


def discovery_backtest(overlayed: pd.DataFrame, dest: Path | None = None, *, write: bool = False) -> dict[str, Any]:
    """2018–2021 ranking window. Allow 2021 trades to exit into early 2022, then clip NAV."""
    return run_canonical(
        dest,
        overlayed,
        extra={"role": "discovery_2018_2021", "fee_schedule": "HISTORICAL_FEE_CORRECTED"},
        resume=False,
        write=write,
        start_date=DEV_START,
        end_date=date(2022, 3, 1),
        metric_start=DEV_START,
        metric_end=DEV_END,
    )


def year_stats(obs: dict[str, Any], years: tuple[int, ...] = (2018, 2019, 2020, 2021)) -> dict[str, Any]:
    raw = obs.get("yearly_returns") or {}
    vals = []
    mapped = {}
    for y in years:
        v = raw.get(str(y), raw.get(y))
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = None
        mapped[str(y)] = fv
        if fv is not None:
            vals.append(fv)
    if not vals:
        return {"years": mapped, "median_year": None, "worst_year": None}
    return {
        "years": mapped,
        "median_year": float(pd.Series(vals).median()),
        "worst_year": float(min(vals)),
    }


__all__ = [
    "A00",
    "apply_b03",
    "cached_market",
    "cached_overlay_v4",
    "discovery_backtest",
    "fuse_heads",
    "run_canonical",
    "year_stats",
]
