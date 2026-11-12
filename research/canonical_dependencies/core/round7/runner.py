"""Formal backtest, baseline comparison, and conclusion for Round 7."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..local_executor import baseline_fixed_sleeve_config, run_local_fixed_sleeve_backtest, write_backtest_artifacts
from ..paths import MARKET_END, PREDICTION_START, ROUND4_ROOT, assert_project_result_path, ensure_round7_tree, round7_dir
from ..round4.runner import attach_round4_fields
from ..stock_readonly import load_prepared_market, sha256_file
from .contract import CANDIDATE_CODE, FIVE_HEAD, TEN_HEAD

HISTORICAL_REFERENCE = {
    "total_return": 24.87723674,
    "max_drawdown": -0.25425,
    "completed_trades": 2158,
    "label": "legacy anchor-replay result; non-binding reference",
}


def _candidate_predictions_path() -> Path:
    return ensure_round7_tree() / "predictions" / "clean_risk" / "predictions.parquet"


def reconcile_labels(trades: pd.DataFrame) -> dict:
    labels = pd.read_parquet(
        TEN_HEAD.panel_dir / "candidate_panel.parquet",
        columns=["trade_date", "instrument_id", "actual_exit_date", "label_exit_rule_net"],
    ).rename(columns={
        "trade_date": "signal_date",
        "actual_exit_date": "label_actual_exit_date",
    })
    for column in ("signal_date", "label_actual_exit_date"):
        labels[column] = pd.to_datetime(labels[column])
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    work = trades.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    work["actual_exit_date"] = pd.to_datetime(work["actual_exit_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    merged = work.merge(labels, on=["signal_date", "instrument_id"], how="left", validate="many_to_one")
    missing = int(merged["label_exit_rule_net"].isna().sum())
    exit_mismatches = int((merged["actual_exit_date"] != merged["label_actual_exit_date"]).sum())
    errors = np.abs(
        pd.to_numeric(merged["ret"], errors="coerce")
        - pd.to_numeric(merged["label_exit_rule_net"], errors="coerce")
    )
    max_error = float(errors.max()) if errors.notna().any() else None
    tolerance = 1e-12
    passed = missing == 0 and exit_mismatches == 0 and max_error is not None and max_error <= tolerance
    return {
        "status": "exact" if passed else "failed",
        "trades_checked": int(len(merged)),
        "missing_labels": missing,
        "exit_date_mismatches": exit_mismatches,
        "return_max_abs_error": max_error,
        "tolerance": tolerance,
    }


def run_backtest(*, resume: bool = True) -> dict:
    dest = round7_dir(CANDIDATE_CODE)
    dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "report.json"
    assert_project_result_path(report_path)
    if resume and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))

    prediction_path = _candidate_predictions_path()
    predictions = pd.read_parquet(prediction_path)
    predictions["signal_date"] = pd.to_datetime(predictions["signal_date"])
    market_meta = load_prepared_market(warmup_ma5=True)
    config = baseline_fixed_sleeve_config(exit_mode="close_from_d2")
    config = replace(
        config,
        top_k=2,
        name_budget=10_000.0,
        name_budget_nav_fraction=None,
        rank_name_budget_nav_fractions=(0.3, 0.1),
        market_up_ratio_min=None,
        market_up_ratio_full_allocation_min=None,
        reduced_name_budget_nav_fraction=None,
        open_gap_min=None,
        open_gap_max=None,
    )
    result = run_local_fixed_sleeve_backtest(
        predictions,
        market_meta["prepared"],
        config=config,
        start_date=PREDICTION_START,
        end_date=MARKET_END,
    )
    metrics = result["report"].setdefault("metrics", {})
    attach_round4_fields(result["nav"], result["trades"], metrics)
    reconciliation = reconcile_labels(result["trades"])
    result["report"]["label_reconciliation"] = reconciliation
    result["report"]["acceptance"] = {
        "max_live_sleeves_lte_2": int(metrics.get("max_live_sleeves", 999)) <= 2,
        "no_negative_sleeve_cash": (
            float(metrics.get("minimum_sleeve_cash", -1.0)) >= -1e-9
            and float(metrics.get("minimum_cash", -1.0)) >= -1e-9
        ),
        "asset_identity": float(metrics.get("asset_identity_max_abs", 1.0)) <= 1e-6,
        "all_positions_exited": int(metrics.get("terminal_positions", 1)) == 0,
        "label_reconciliation": reconciliation["status"] == "exact",
    }
    extra = {
        "status": "completed_and_validated",
        "experiment_code": CANDIDATE_CODE,
        "parent": "baseline_v3=R4_B05",
        "baseline_promoted": False,
        "alpha_contract": "fs01122-derived no-sampling dual-head 90/10 clean recompute",
        "five_head": FIVE_HEAD.to_dict(),
        "ten_head": TEN_HEAD.to_dict(),
        "predictions_file": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "snapshot_id": market_meta.get("snapshot_id"),
        "snapshot_fingerprint": market_meta.get("snapshot_fingerprint"),
        "evidence_role": "consumed_historical_development_recomputed_cleanly",
        "same_open_fill_execution_risk": True,
        "execution_warning": (
            "d1_open_gap is known at 09:25 while the research fill uses the same daily open; "
            "results are not represented as directly executable live performance"
        ),
        "historical_reference_non_binding": HISTORICAL_REFERENCE,
    }
    written = write_backtest_artifacts(result, dest, extra_report=extra)
    report = written["report"]
    attach_round4_fields(result["nav"], result["trades"], report.setdefault("metrics", {}))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def _metric_view(report: dict) -> dict:
    metrics = report.get("metrics") or {}
    return {
        "total_return": metrics.get("total_return"),
        "annualized_return": metrics.get("annualized_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe_zero_rate": metrics.get("sharpe_zero_rate"),
        "completed_trades": metrics.get("completed_trades"),
        "trade_win_rate": metrics.get("trade_win_rate"),
        "yearly_returns": metrics.get("yearly_returns"),
    }


def compare_baseline() -> dict:
    candidate_path = round7_dir(CANDIDATE_CODE) / "report.json"
    baseline_path = ROUND4_ROOT / "R4_B05" / "report.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_metrics = _metric_view(candidate)
    baseline_metrics = _metric_view(baseline)
    delta = {}
    for name in (
        "total_return", "annualized_return", "max_drawdown", "sharpe_zero_rate",
        "completed_trades", "trade_win_rate",
    ):
        left, right = candidate_metrics.get(name), baseline_metrics.get(name)
        delta[name] = None if left is None or right is None else float(left) - float(right)
    audit_path = ensure_round7_tree() / "diagnostics" / "legacy_reference_audit.json"
    legacy_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
    payload = {
        "status": "complete",
        "candidate": CANDIDATE_CODE,
        "baseline": "baseline_v3=R4_B05",
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "delta_candidate_minus_baseline": delta,
        "historical_reference_non_binding": HISTORICAL_REFERENCE,
        "legacy_difference_attribution": legacy_audit,
        "baseline_promoted": False,
        "written_at": datetime.now().astimezone().isoformat(),
    }
    dest = ensure_round7_tree() / "comparison" / "candidate_vs_baseline_v3.json"
    assert_project_result_path(dest)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def conclude() -> dict:
    report = json.loads((round7_dir(CANDIDATE_CODE) / "report.json").read_text(encoding="utf-8"))
    comparison = compare_baseline()
    acceptance = report.get("acceptance") or {}
    required = (
        "max_live_sleeves_lte_2",
        "no_negative_sleeve_cash",
        "asset_identity",
        "all_positions_exited",
        "label_reconciliation",
    )
    technical_pass = all(bool(acceptance.get(name)) for name in required)
    payload = {
        "status": "candidate_pass" if technical_pass else "candidate_fail",
        "candidate": CANDIDATE_CODE,
        "technical_acceptance": {name: bool(acceptance.get(name)) for name in required},
        "performance": comparison["candidate_metrics"],
        "delta_vs_baseline_v3": comparison["delta_candidate_minus_baseline"],
        "baseline_v3_unchanged": True,
        "baseline_v4_created": False,
        "evidence_role": "consumed_historical_development; not final independent OOS",
        "execution_warning": report.get("execution_warning"),
        "written_at": datetime.now().astimezone().isoformat(),
    }
    dest = ensure_round7_tree() / "conclusion.json"
    assert_project_result_path(dest)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload
