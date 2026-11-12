"""Round 6 回测入口：复用 local_executor，产物只写 result/round6/{code}/。"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..local_executor import (
    baseline_fixed_sleeve_config,
    run_local_fixed_sleeve_backtest,
    write_backtest_artifacts,
)
from ..metrics import METRIC_VERSION
from ..paths import (
    BACKTEST_VARIANT,
    CAPACITY,
    ENSEMBLE_METHOD,
    MARKET_END,
    PREDICTION_START,
    SIGNAL_END,
    TRIAL_ID,
    assert_project_result_path,
    ensure_round6_tree,
    round6_dir,
)
from ..round3.runner import _apply_cost_overrides, _dump_simple_yaml
from ..round4.runner import attach_round4_fields
from ..stock_readonly import (
    ensemble_dir,
    load_fs00263_ensemble_predictions,
    load_prepared_market,
    sha256_file,
)
from .overlays import apply_date_multiplier, build_date_multiplier
from .registry import LOW_ACTIVITY_TRADES, PARENT_DEFAULT, R6Spec

COMPLETE_YEARS = tuple(range(2018, 2026))


def tracking_row(spec: R6Spec, report: dict[str, Any] | None = None) -> dict[str, Any]:
    m = (report or {}).get("metrics") or {}
    flags = []
    if m.get("low_activity"):
        flags.append("LOW_ACTIVITY")
    if not spec.pit_ok:
        flags.append("PIT_FAIL")
    if spec.diagnostic:
        flags.append("DIAGNOSTIC")
    if spec.forensic_only:
        flags.append("FORENSIC_ONLY")
    return {
        "code": spec.code,
        "title_zh": spec.title_zh,
        "parent": spec.parent,
        "alpha_source": spec.alpha_source,
        "prediction_key": spec.prediction_key,
        "features": list(spec.features),
        "objective": spec.objective,
        "target": spec.target,
        "window_kind": spec.window_kind,
        "representation": spec.representation,
        "fold_rule": spec.fold_rule,
        "cagr": m.get("annualized_return"),
        "sharpe": m.get("sharpe_zero_rate"),
        "max_drawdown": m.get("max_drawdown"),
        "total_return": m.get("total_return"),
        "completed_trades": m.get("completed_trades"),
        "worst_year_return": m.get("worst_complete_year_return") or m.get("worst_year_return"),
        "worst_12m": m.get("worst_rolling_12m"),
        "cagr_ex2024": m.get("cagr_ex2024"),
        "loyo_median": m.get("loyo_median"),
        "loyo_worst": m.get("loyo_worst"),
        "low_activity": m.get("low_activity"),
        "flags": flags,
        "stage": spec.stage,
        "top_k": spec.top_k,
        "forensic_only": spec.forensic_only,
        "diagnostic": spec.diagnostic,
    }


def load_report(code: str) -> dict[str, Any] | None:
    path = round6_dir(code) / "report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_predictions(spec: R6Spec, predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    if spec.alpha_source == "locked_fs00263" or spec.prediction_key in (None, "", "locked"):
        if predictions is not None:
            return predictions
        return load_fs00263_ensemble_predictions(start=PREDICTION_START, signal_end=SIGNAL_END)
    from .train import load_predictions

    return load_predictions(spec.prediction_key or spec.code)


def run_spec(
    spec: R6Spec,
    *,
    predictions=None,
    prepared_market=None,
    market_meta: dict[str, Any] | None = None,
    resume: bool = True,
    run_now: bool = True,
) -> dict[str, Any]:
    ensure_round6_tree()
    dest = round6_dir(spec.code)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")

    existing = load_report(spec.code)
    if resume and existing is not None:
        return {
            "status": existing.get("status") or "resumed",
            "code": spec.code,
            "report": existing,
            "summary": tracking_row(spec, existing),
            "out_dir": str(dest),
        }
    if spec.alpha_source == "skipped":
        payload = {
            "status": "SKIPPED_NO_ARTIFACT",
            "code": spec.code,
            "reason": spec.description_zh,
            "metrics": {},
            "summary": tracking_row(spec),
        }
        (dest / "report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return payload
    if not run_now:
        return {
            "status": "pending",
            "code": spec.code,
            "summary": tracking_row(spec),
            "note": "dry-run / RUN_NOW=False",
        }

    try:
        preds = resolve_predictions(spec, predictions)
    except FileNotFoundError as exc:
        payload = {
            "status": "SKIPPED_NO_PREDICTIONS",
            "code": spec.code,
            "reason": str(exc),
            "summary": tracking_row(spec),
        }
        (dest / "report.json").write_text(
            json.dumps({**payload, "metrics": {}}, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return payload

    if prepared_market is None:
        loaded = load_prepared_market(warmup_ma5=True)
        prepared_market = loaded["prepared"]
        market_meta = loaded

    date_mult, overlay_meta, hooks = build_date_multiplier(spec, preds)
    if overlay_meta.get("status") == "ABANDONED":
        payload = {
            "status": "ABANDONED",
            "code": spec.code,
            "reason": overlay_meta.get("reason"),
            "overlay": overlay_meta,
            "summary": tracking_row(spec),
        }
        (dest / "report.json").write_text(
            json.dumps({**payload, "metrics": {}}, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return payload

    overlayed = apply_date_multiplier(preds, date_mult)
    if spec.random_seed is None:
        merged = overlayed.merge(
            preds[["signal_date", "instrument_id", "score"]],
            on=["signal_date", "instrument_id"],
            suffixes=("", "_src"),
        )
        if merged.empty or not (merged["score"] == merged["score_src"]).all():
            raise RuntimeError(f"{spec.code}: score mutated")
    else:
        rng = np.random.default_rng(int(spec.random_seed))
        overlayed = overlayed.copy()
        overlayed["score"] = rng.random(len(overlayed))

    config = baseline_fixed_sleeve_config(exit_mode=spec.exit_mode)
    if spec.rank_name_budget_nav_fractions is not None:
        config = replace(
            config,
            name_budget=10_000.0,
            name_budget_nav_fraction=None,
            market_up_ratio_min=spec.market_up_ratio_min,
            top_k=int(spec.top_k),
            rank_name_budget_nav_fractions=tuple(spec.rank_name_budget_nav_fractions),
        )
    else:
        config = replace(
            config,
            name_budget_nav_fraction=spec.name_budget_nav_fraction,
            market_up_ratio_min=spec.market_up_ratio_min,
            top_k=int(spec.top_k),
        )
    config = _apply_cost_overrides(config, spec.cost_overrides)

    result = run_local_fixed_sleeve_backtest(
        overlayed,
        prepared_market,
        config=config,
        start_date=PREDICTION_START,
        end_date=MARKET_END,
        overlay_hooks=hooks,
    )
    metrics = result["report"].setdefault("metrics", {})
    attach_round4_fields(result["nav"], result["trades"], metrics)
    metrics["low_activity"] = (
        metrics.get("completed_trades") is not None
        and int(metrics["completed_trades"]) < LOW_ACTIVITY_TRADES
    )
    metrics["parent"] = PARENT_DEFAULT

    extra = {
        "status": "completed_and_validated",
        "experiment_code": spec.code,
        "trial_id": TRIAL_ID,
        "capacity": CAPACITY,
        "ensemble": ENSEMBLE_METHOD if spec.alpha_source == "locked_fs00263" else "local_rank_pct_5seed",
        "variant": BACKTEST_VARIANT,
        "parent": spec.parent,
        "alpha_source": spec.alpha_source,
        "prediction_key": spec.prediction_key,
        "overlay": overlay_meta,
        "overlay_hooks": hooks,
        "name_budget_nav_fraction": spec.name_budget_nav_fraction,
        "market_up_ratio_min": spec.market_up_ratio_min,
        "top_k": spec.top_k,
        "features": list(spec.features),
        "objective": spec.objective,
        "target": spec.target,
        "window_kind": spec.window_kind,
        "representation": spec.representation,
        "forensic_only": spec.forensic_only,
        "metric_version": METRIC_VERSION,
        "cost_overrides": spec.cost_overrides,
        "skip_years": list(spec.skip_years),
        "evidence_role": "round6_historical_development",
        "pit": True,
        "note": (
            "默认对照 baseline_v3=R4_B05。风险层锁死。新训 subsample/colsample=1.0。"
            "产物在课题 result/round6/，未写 Stock/runs。不是 final OOS。"
        ),
    }
    if spec.alpha_source == "locked_fs00263":
        extra["predictions_file"] = str(ensemble_dir() / "predictions.parquet")
        extra["predictions_sha256"] = sha256_file(ensemble_dir() / "predictions.parquet")
    if market_meta:
        extra["snapshot_id"] = market_meta.get("snapshot_id")
        extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")

    written = write_backtest_artifacts(result, dest, extra_report=extra)
    attach_round4_fields(result["nav"], result["trades"], written["report"].setdefault("metrics", {}))
    (dest / "report.json").write_text(
        json.dumps(written["report"], ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    cfg_path = dest / "config.yaml"
    assert_project_result_path(cfg_path)
    cfg_path.write_text(_dump_simple_yaml(spec.to_config()), encoding="utf-8")
    shared_cfg = ensure_round6_tree() / "configs" / f"{spec.code}.json"
    assert_project_result_path(shared_cfg)
    shared_cfg.write_text(
        json.dumps(spec.to_config(), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "code": spec.code,
        "written_at": datetime.now().astimezone().isoformat(),
        "parent": spec.parent,
        "metric_version": METRIC_VERSION,
        "pit": True,
        "alpha_source": spec.alpha_source,
        "prediction_key": spec.prediction_key,
        "snapshot_id": extra.get("snapshot_id"),
        "predictions_sha256": extra.get("predictions_sha256"),
    }
    meta_path = dest / "metadata.json"
    assert_project_result_path(meta_path)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    summary = tracking_row(spec, written["report"])
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    shared_metrics = ensure_round6_tree() / "metrics" / f"{spec.code}.json"
    assert_project_result_path(shared_metrics)
    shared_metrics.write_text(
        json.dumps(written["report"].get("metrics") or {}, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for name in ("nav", "trades"):
        src = dest / f"{name}.parquet"
        if src.is_file():
            target = ensure_round6_tree() / name / f"{spec.code}.parquet"
            shutil.copy2(src, target)
            assert_project_result_path(target)

    return {
        "status": "ran",
        "code": spec.code,
        "out_dir": written["out_dir"],
        "report": written["report"],
        "summary": summary,
        "overlay": overlay_meta,
    }


def run_many(specs: list[R6Spec], *, predictions, prepared, meta, resume: bool, run_now: bool) -> dict[str, Any]:
    out = {}
    for spec in specs:
        out[spec.code] = run_spec(
            spec,
            predictions=predictions,
            prepared_market=prepared,
            market_meta=meta,
            resume=resume,
            run_now=run_now,
        )
    return out


def write_skipped(spec: R6Spec, reason: str, status: str = "SKIPPED_UNSUPPORTED") -> dict[str, Any]:
    dest = round6_dir(spec.code)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")
    payload = {
        "status": status,
        "code": spec.code,
        "reason": reason,
        "metrics": {},
        "summary": tracking_row(spec),
    }
    (dest / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return payload
