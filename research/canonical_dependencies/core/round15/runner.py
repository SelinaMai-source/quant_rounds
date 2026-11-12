"""Round15 回测入口：复用 Round12 runner。产物只写 result/round15/。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ..local_executor import run_local_fixed_sleeve_backtest, write_backtest_artifacts
from ..paths import MARKET_END, PREDICTION_START, assert_project_result_path
from ..replay_round7.inject import inject_round7_into_overlay_caches, load_locked_alpha
from ..round8.overlays import MIN_KEEP
from ..round8.runner import attach_round8_fields
from ..round11.contract import BASELINES
from ..round11.runner import run_backtest_in_memory
from ..round12.runner import hist_hooks, run_r12_backtest
from .common import observed_from_report
from .contract import RANK_NAV, TOP_K

_CACHE: dict[str, Any] = {}


def load_market() -> dict[str, Any]:
    if "market" not in _CACHE:
        from ..stock_readonly import load_prepared_market

        inject_round7_into_overlay_caches()
        _CACHE["market"] = load_prepared_market(warmup_ma5=True)
    return _CACHE["market"]


def load_alpha() -> pd.DataFrame:
    if "alpha" not in _CACHE:
        inject_round7_into_overlay_caches()
        _CACHE["alpha"] = load_locked_alpha()
    return _CACHE["alpha"]


def build_version_overlayed(version: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    key = f"ov_{version}"
    if key in _CACHE:
        return _CACHE[key]
    pred = load_alpha()
    spec = BASELINES[version]
    kind = (spec.overlay_spec or {}).get("kind")
    if kind == "exact_v2":
        from ..round8.overlays import build_overlayed
    else:
        from ..round9.overlays import build_overlayed
    overlayed, meta, cfg = build_overlayed(dict(spec.overlay_spec), pred)
    _CACHE[key] = (overlayed, meta, cfg)
    return overlayed, meta, cfg


def _with_rank(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    return work


def apply_top2_veto(overlayed: pd.DataFrame, veto_dates) -> pd.DataFrame:
    """零 Top2 用 MIN_KEEP，禁止删行导致 Top3 补位。"""
    work = _with_rank(overlayed)
    dates = set(pd.to_datetime(list(veto_dates)).normalize())
    mask = work["_rank"].eq(2) & work["signal_date"].isin(dates)
    work.loc[mask, "allocation_multiplier"] = MIN_KEEP
    return work.drop(columns=["_rank"], errors="ignore")


def apply_budget_scale(overlayed: pd.DataFrame, dates, factor: float) -> pd.DataFrame:
    work = _with_rank(overlayed)
    idx = set(pd.to_datetime(list(dates)).normalize())
    mask = work["_rank"].le(2) & work["signal_date"].isin(idx)
    scaled = pd.to_numeric(work.loc[mask, "allocation_multiplier"], errors="coerce") * float(factor)
    work.loc[mask, "allocation_multiplier"] = scaled.clip(MIN_KEEP, 1.0)
    return work.drop(columns=["_rank"], errors="ignore")


def apply_top1_skip(overlayed: pd.DataFrame, skip_dates) -> pd.DataFrame:
    work = _with_rank(overlayed)
    dates = set(pd.to_datetime(list(skip_dates)).normalize())
    mask = work["_rank"].eq(1) & work["signal_date"].isin(dates)
    work.loc[mask, "allocation_multiplier"] = MIN_KEEP
    return work.drop(columns=["_rank"], errors="ignore")


def _summary_from_report(code: str, report: dict[str, Any]) -> dict[str, Any]:
    obs = observed_from_report(report)
    obs["code"] = code
    obs["fee_schedule"] = report.get("fee_schedule")
    return obs


def run_r15_backtest(
    dest,
    *,
    overlayed: pd.DataFrame,
    validation_hooks: dict[str, Any] | None = None,
    initial_cash: float | None = None,
    extra_report: dict[str, Any] | None = None,
    resume: bool = True,
    write: bool = True,
    prepared_market=None,
    market_meta: dict[str, Any] | None = None,
    entry_price_overrides: pd.DataFrame | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    loaded = None
    if prepared_market is None or market_meta is None:
        loaded = load_market()
        prepared_market = prepared_market if prepared_market is not None else loaded["prepared"]
        market_meta = market_meta if market_meta is not None else loaded
    extra = {
        "fee_schedule": "HISTORICAL_FEE_CORRECTED",
        "rank_nav": list(RANK_NAV),
        "top_k": TOP_K,
        "evidence_role": "round15_option_top1_exec",
        "note": "Round15 产物只写 result/round15/。不覆盖 legacy。不自动升 CURRENT v5。",
    }
    if extra_report:
        extra.update(extra_report)
    custom = entry_price_overrides is not None or start_date is not None or end_date is not None
    if not custom:
        return run_r12_backtest(
            dest,
            prepared_market=prepared_market,
            market_meta=market_meta,
            overlayed=overlayed,
            validation_hooks=validation_hooks if validation_hooks is not None else hist_hooks(),
            initial_cash=initial_cash,
            extra_report=extra,
            resume=resume,
            write=write,
        )

    dest = Path(dest)
    if write:
        dest.mkdir(parents=True, exist_ok=True)
        assert_project_result_path(dest / "report.json")
    report_path = dest / "report.json"
    if write and resume and report_path.is_file() and (dest / "trades.parquet").is_file() and (dest / "nav.parquet").is_file():
        import json

        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("metrics"):
            return {
                "status": "resumed",
                "code": dest.name,
                "report": existing,
                "out_dir": str(dest),
                "summary": _summary_from_report(dest.name, existing),
            }
    spec = BASELINES["v4"]
    from dataclasses import replace

    from ..round11.runner import _config_for

    config = _config_for({"top_k": TOP_K, "rank_name_budget_nav_fractions": RANK_NAV})
    if initial_cash is not None:
        cash = float(initial_cash)
        config = replace(config, initial_cash=cash, sleeve_cash=cash / float(config.sleeve_count))
    result = run_local_fixed_sleeve_backtest(
        overlayed,
        prepared_market,
        config=config,
        start_date=start_date or PREDICTION_START,
        end_date=end_date or MARKET_END,
        validation_hooks=validation_hooks if validation_hooks is not None else hist_hooks(),
        entry_price_overrides=entry_price_overrides,
    )
    metrics = result["report"].setdefault("metrics", {})
    attach_round8_fields(result["nav"], result["trades"], metrics)
    extra.update({
        "status": "completed_and_validated",
        "experiment_code": dest.name,
        "baseline_version": "v4",
        "parent_artifact": spec.code,
        "window_start": str(start_date or PREDICTION_START),
        "window_end": str(end_date or MARKET_END),
        "entry_price_overrides": entry_price_overrides is not None,
    })
    if market_meta:
        extra["snapshot_id"] = market_meta.get("snapshot_id")
        extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")
    if write:
        written = write_backtest_artifacts(result, dest, extra_report=extra)
        report = written["report"]
        summary = _summary_from_report(dest.name, report)
        (dest / "summary.json").write_text(
            __import__("json").dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "ran",
            "code": dest.name,
            "report": report,
            "out_dir": str(dest),
            "result": result,
            "summary": summary,
        }
    report = dict(result["report"])
    report.update(extra)
    return {
        "status": "ran_in_memory",
        "code": dest.name,
        "report": report,
        "out_dir": str(dest),
        "result": result,
        "summary": _summary_from_report(dest.name, report),
        "nav": result["nav"],
        "trades": result["trades"],
    }


__all__ = [
    "MIN_KEEP",
    "apply_budget_scale",
    "apply_top1_skip",
    "apply_top2_veto",
    "build_version_overlayed",
    "hist_hooks",
    "load_alpha",
    "load_market",
    "run_backtest_in_memory",
    "run_r15_backtest",
]
