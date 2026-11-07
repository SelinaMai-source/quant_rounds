"""Round11 parity 重放：复用已有 overlay，产物只写 result/round11/parity/。不改策略。"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pandas as pd

from ..local_executor import (
    round7_fixed_sleeve_config,
    run_local_fixed_sleeve_backtest,
    write_backtest_artifacts,
)
from ..paths import MARKET_END, PREDICTION_START, assert_project_result_path, ensure_round11_tree
from ..replay_round7.inject import inject_round7_into_overlay_caches, load_locked_alpha, load_locked_risk
from ..round8.runner import attach_round8_fields
from .contract import BASELINES, BaselineSpec, RANK_NAV, TOP_K, replay_dir


def _load_predictions(spec: BaselineSpec) -> pd.DataFrame:
    if spec.predictions == "locked_risk":
        return load_locked_risk()
    return load_locked_alpha()


def _build_overlayed(spec: BaselineSpec, predictions: pd.DataFrame):
    kind = (spec.overlay_spec or {}).get("kind")
    if kind == "v1_clean":
        return predictions.copy(), {"kind": "v1_clean_risk"}, {}
    if kind == "exact_v2":
        from ..round8.overlays import build_overlayed

        return build_overlayed(dict(spec.overlay_spec), predictions)
    from ..round9.overlays import build_overlayed

    return build_overlayed(dict(spec.overlay_spec), predictions)


def _assert_score_unmutated(code: str, overlayed: pd.DataFrame, predictions: pd.DataFrame) -> None:
    left = overlayed[["signal_date", "instrument_id", "score"]].copy()
    right = predictions[["signal_date", "instrument_id", "score"]].copy()
    left["signal_date"] = pd.to_datetime(left["signal_date"]).dt.normalize()
    right["signal_date"] = pd.to_datetime(right["signal_date"]).dt.normalize()
    left["instrument_id"] = left["instrument_id"].astype(str)
    right["instrument_id"] = right["instrument_id"].astype(str)
    merged = left.merge(right, on=["signal_date", "instrument_id"], suffixes=("", "_src"))
    if merged.empty:
        raise RuntimeError(f"{code}: score check merge empty")
    delta = (
        pd.to_numeric(merged["score"], errors="coerce")
        - pd.to_numeric(merged["score_src"], errors="coerce")
    ).abs()
    if not bool((delta.fillna(0.0) <= 1e-12).all()):
        raise RuntimeError(f"{code}: score mutated")


def _config_for(cfg_over: dict[str, Any]):
    config = round7_fixed_sleeve_config(exit_mode="close_from_d2")
    top_k = int(cfg_over.get("top_k") or TOP_K)
    fracs = cfg_over.get("rank_name_budget_nav_fractions")
    if fracs is None:
        fracs = (0.3,) if top_k == 1 else tuple(RANK_NAV)
    return replace(
        config,
        top_k=top_k,
        name_budget=10_000.0,
        name_budget_nav_fraction=None,
        rank_name_budget_nav_fractions=tuple(fracs),
        market_up_ratio_min=None,
        open_gap_min=None,
        open_gap_max=None,
    )


def run_backtest_in_memory(
    spec: BaselineSpec,
    *,
    prepared_market,
    overlayed: pd.DataFrame | None = None,
    cfg_over: dict[str, Any] | None = None,
    validation_hooks: dict[str, Any] | None = None,
    cost_overrides: dict[str, Any] | None = None,
    initial_cash: float | None = None,
) -> dict[str, Any]:
    if overlayed is None:
        predictions = _load_predictions(spec)
        overlayed, _, cfg_over = _build_overlayed(spec, predictions)
    config = _config_for(cfg_over or {})
    if cost_overrides:
        from ..round3.runner import _apply_cost_overrides

        config = _apply_cost_overrides(config, cost_overrides)
    if initial_cash is not None:
        cash = float(initial_cash)
        config = replace(config, initial_cash=cash, sleeve_cash=cash / float(config.sleeve_count))
    return run_local_fixed_sleeve_backtest(
        overlayed,
        prepared_market,
        config=config,
        start_date=PREDICTION_START,
        end_date=MARKET_END,
        validation_hooks=validation_hooks,
    )


def run_spec(
    spec: BaselineSpec,
    *,
    prepared_market,
    market_meta: dict[str, Any] | None,
    resume: bool = True,
) -> dict[str, Any]:
    ensure_round11_tree()
    dest = replay_dir(spec)
    dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "report.json"
    assert_project_result_path(report_path)
    if resume and report_path.is_file() and (dest / "trades.parquet").is_file() and (dest / "nav.parquet").is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("metrics"):
            return {"status": "resumed", "code": spec.replay_code, "report": existing, "out_dir": str(dest)}

    inject_round7_into_overlay_caches()
    predictions = _load_predictions(spec)
    overlayed, overlay_meta, cfg_over = _build_overlayed(spec, predictions)
    _assert_score_unmutated(spec.replay_code, overlayed, predictions)
    result = run_backtest_in_memory(
        spec,
        prepared_market=prepared_market,
        overlayed=overlayed,
        cfg_over=cfg_over,
    )
    metrics = result["report"].setdefault("metrics", {})
    attach_round8_fields(result["nav"], result["trades"], metrics)
    extra = {
        "status": "completed_and_validated",
        "experiment_code": spec.replay_code,
        "baseline_version": spec.version,
        "parent_artifact": spec.code,
        "overlay": overlay_meta,
        "top_k": result["report"].get("config", {}).get("top_k"),
        "rank_name_budget_nav_fractions": result["report"].get("config", {}).get("rank_name_budget_nav_fractions"),
        "metric_version": "r3_locked_window_v1",
        "evidence_role": "round11_parity_replay",
        "note": (
            f"Round11 复现 {spec.version}={spec.code}。产物只写 result/round11/。"
            "不覆盖 legacy。不改策略。"
        ),
    }
    if market_meta:
        extra["snapshot_id"] = market_meta.get("snapshot_id")
        extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")
    written = write_backtest_artifacts(result, dest, extra_report=extra)
    summary = {
        "code": spec.replay_code,
        "version": spec.version,
        "anchor": spec.code,
        "cagr": metrics.get("annualized_return"),
        "sharpe": metrics.get("sharpe_zero_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
        "total_return": metrics.get("total_return"),
        "completed_trades": metrics.get("completed_trades"),
        "loyo_median": metrics.get("loyo_median"),
        "loyo_worst": metrics.get("loyo_worst"),
        "yearly_returns": metrics.get("yearly_returns"),
    }
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ran",
        "code": spec.replay_code,
        "report": written["report"],
        "out_dir": str(dest),
        "result": result,
    }


def run_spec_to(
    spec: BaselineSpec,
    dest,
    *,
    prepared_market,
    market_meta: dict[str, Any] | None,
    resume: bool = True,
    validation_hooks: dict[str, Any] | None = None,
    cost_overrides: dict[str, Any] | None = None,
    initial_cash: float | None = None,
    extra_report: dict[str, Any] | None = None,
    overlayed: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """把一次核验回测写到 dest。禁止写 legacy roundN。"""
    from pathlib import Path

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "report.json"
    assert_project_result_path(report_path)
    if resume and report_path.is_file() and (dest / "trades.parquet").is_file() and (dest / "nav.parquet").is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("metrics"):
            return {"status": "resumed", "code": spec.replay_code, "report": existing, "out_dir": str(dest)}

    inject_round7_into_overlay_caches()
    if overlayed is None:
        predictions = _load_predictions(spec)
        overlayed, overlay_meta, cfg_over = _build_overlayed(spec, predictions)
        _assert_score_unmutated(spec.replay_code, overlayed, predictions)
    else:
        overlay_meta = {"kind": "supplied"}
        cfg_over = {}
    result = run_backtest_in_memory(
        spec,
        prepared_market=prepared_market,
        overlayed=overlayed,
        cfg_over=cfg_over,
        validation_hooks=validation_hooks,
        cost_overrides=cost_overrides,
        initial_cash=initial_cash,
    )
    metrics = result["report"].setdefault("metrics", {})
    attach_round8_fields(result["nav"], result["trades"], metrics)
    extra = {
        "status": "completed_and_validated",
        "experiment_code": dest.name,
        "baseline_version": spec.version,
        "parent_artifact": spec.code,
        "overlay": overlay_meta,
        "metric_version": "r3_locked_window_v1",
        "evidence_role": "round11_validation",
        "note": "Round11 核验产物。只写 result/round11/。不覆盖 legacy。不改策略。",
    }
    if market_meta:
        extra["snapshot_id"] = market_meta.get("snapshot_id")
        extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")
    if extra_report:
        extra.update(extra_report)
    written = write_backtest_artifacts(result, dest, extra_report=extra)
    summary = {
        "code": dest.name,
        "version": spec.version,
        "anchor": spec.code,
        "cagr": metrics.get("annualized_return"),
        "sharpe": metrics.get("sharpe_zero_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
        "total_return": metrics.get("total_return"),
        "completed_trades": metrics.get("completed_trades"),
        "loyo_median": metrics.get("loyo_median"),
        "loyo_worst": metrics.get("loyo_worst"),
        "yearly_returns": metrics.get("yearly_returns"),
        "fee_schedule": written["report"].get("fee_schedule") or extra.get("fee_schedule"),
    }
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ran",
        "code": dest.name,
        "report": written["report"],
        "out_dir": str(dest),
        "result": result,
        "summary": summary,
    }


def run_all_specs(
    *,
    prepared_market,
    market_meta: dict[str, Any] | None,
    resume: bool = True,
) -> dict[str, Any]:
    out = {}
    for key, spec in BASELINES.items():
        out[key] = run_spec(spec, prepared_market=prepared_market, market_meta=market_meta, resume=resume)
    return out


__all__ = ["run_all_specs", "run_backtest_in_memory", "run_spec", "run_spec_to"]
