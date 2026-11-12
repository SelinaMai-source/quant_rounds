"""Round12 回测入口：复用 Round11 runner / local_executor hooks。产物只写 result/round12/。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from ..local_executor import write_backtest_artifacts
from ..paths import assert_project_result_path
from ..replay_round7.inject import inject_round7_into_overlay_caches, load_locked_alpha
from ..round8.runner import attach_round8_fields
from ..round11.contract import BASELINES
from ..round11.runner import run_backtest_in_memory, run_spec_to
from .common import observed_from_report
from .contract import RANK_NAV, TOP_K


def v4_spec():
    return BASELINES["v4"]


def load_v4_overlayed() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    inject_round7_into_overlay_caches()
    spec = v4_spec()
    pred = load_locked_alpha()
    from ..round9.overlays import build_overlayed

    overlayed, meta, cfg = build_overlayed(dict(spec.overlay_spec), pred)
    return overlayed, meta, cfg


def run_r12_backtest(
    dest: Path,
    *,
    prepared_market,
    market_meta: dict[str, Any] | None,
    overlayed: pd.DataFrame | None = None,
    validation_hooks: dict[str, Any] | None = None,
    initial_cash: float | None = None,
    extra_report: dict[str, Any] | None = None,
    resume: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    dest = Path(dest)
    if write:
        dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "report.json"
    if write:
        assert_project_result_path(report_path)
    if write and resume and report_path.is_file() and (dest / "trades.parquet").is_file() and (dest / "nav.parquet").is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("metrics"):
            return {
                "status": "resumed",
                "code": dest.name,
                "report": existing,
                "out_dir": str(dest),
                "summary": _summary_from_report(dest.name, existing),
            }
    spec = v4_spec()
    if overlayed is None:
        overlayed, overlay_meta, _cfg = load_v4_overlayed()
    else:
        overlay_meta = {"kind": "supplied"}
    result = run_backtest_in_memory(
        spec,
        prepared_market=prepared_market,
        overlayed=overlayed,
        cfg_over={"top_k": TOP_K, "rank_name_budget_nav_fractions": RANK_NAV},
        validation_hooks=validation_hooks,
        initial_cash=initial_cash,
    )
    metrics = result["report"].setdefault("metrics", {})
    attach_round8_fields(result["nav"], result["trades"], metrics)
    extra = {
        "status": "completed_and_validated",
        "experiment_code": dest.name,
        "baseline_version": "v4",
        "parent_artifact": spec.code,
        "overlay": overlay_meta,
        "metric_version": "r3_locked_window_v1",
        "evidence_role": "round12_risk_remediation",
        "note": "Round12 产物只写 result/round12/。不覆盖 legacy。不自动升 v5。",
    }
    if market_meta:
        extra["snapshot_id"] = market_meta.get("snapshot_id")
        extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")
    if extra_report:
        extra.update(extra_report)
    if write:
        written = write_backtest_artifacts(result, dest, extra_report=extra)
        report = written["report"]
        summary = _summary_from_report(dest.name, report)
        (dest / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
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


def _summary_from_report(code: str, report: dict[str, Any]) -> dict[str, Any]:
    obs = observed_from_report(report)
    obs["code"] = code
    obs["fee_schedule"] = report.get("fee_schedule")
    hooks = report.get("validation_hooks")
    if isinstance(hooks, dict):
        obs["buy_extra_slippage_bps"] = hooks.get("buy_extra_slippage_bps") or report.get("buy_extra_slippage_bps")
    else:
        obs["buy_extra_slippage_bps"] = report.get("buy_extra_slippage_bps")
    return obs


def hist_hooks(extra_buy_bps: float | None = None) -> dict[str, Any]:
    hooks: dict[str, Any] = {"historical_stamp_tax": True}
    if extra_buy_bps is not None and float(extra_buy_bps) > 0:
        hooks["buy_extra_slippage_bps"] = float(extra_buy_bps)
    return hooks


__all__ = [
    "hist_hooks",
    "load_v4_overlayed",
    "run_backtest_in_memory",
    "run_r12_backtest",
    "run_spec_to",
    "v4_spec",
]
