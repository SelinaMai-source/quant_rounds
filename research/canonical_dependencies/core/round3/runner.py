"""Round 3 回测入口：复用 local_executor，产物只写 result/round3/{code}/。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..local_executor import (
    baseline_fixed_sleeve_config,
    run_local_fixed_sleeve_backtest,
    write_backtest_artifacts,
)
from ..metrics import LOCKED_BASELINE, METRIC_VERSION
from ..paths import (
    BACKTEST_VARIANT,
    CAPACITY,
    ENSEMBLE_METHOD,
    MARKET_END,
    PREDICTION_START,
    SIGNAL_END,
    TRIAL_ID,
    assert_project_result_path,
    ensure_round3_tree,
    round3_dir,
)
from ..round2.metrics import attach_round2_metrics
from ..stock_readonly import (
    ensemble_dir,
    load_fs00263_ensemble_predictions,
    load_prepared_market,
    sha256_file,
)
from .overlays import apply_date_multiplier, build_date_multiplier
from .registry import R3Spec, r2_source_payload, source_config_hash

LOW_ACTIVITY_TRADES = 437
COMPLETE_YEARS = tuple(range(2018, 2026))


def _apply_cost_overrides(config, overrides: dict[str, Any]):
    if not overrides:
        return config
    costs = config.costs
    extra_slip = overrides.get("extra_slippage_bps")
    if extra_slip:
        costs = replace(costs, slippage_bps=float(costs.slippage_bps) + float(extra_slip))
    fee_mult = overrides.get("fee_mult")
    if fee_mult:
        mult = float(fee_mult)
        costs = replace(
            costs,
            commission_bps=float(costs.commission_bps) * mult,
            transfer_fee_bps=float(costs.transfer_fee_bps) * mult,
            sell_tax_bps=float(costs.sell_tax_bps) * mult,
        )
    config = replace(config, costs=costs)
    buy_adverse = overrides.get("buy_adverse_bps")
    if buy_adverse:
        config = replace(
            config,
            high_gap_threshold=-10.0,
            high_gap_extra_entry_slippage_bps=float(buy_adverse),
        )
    if overrides.get("limit_up_tolerance") is not None:
        config = replace(config, limit_up_tolerance=float(overrides["limit_up_tolerance"]))
    return config


def attach_round3_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    yearly = metrics.get("yearly_returns") or {}
    complete = {str(y): yearly[str(y)] for y in COMPLETE_YEARS if str(y) in yearly}
    if complete:
        worst_key = min(complete, key=complete.get)
        metrics["worst_complete_year"] = worst_key
        metrics["worst_complete_year_return"] = complete[worst_key]
    else:
        metrics["worst_complete_year"] = None
        metrics["worst_complete_year_return"] = None
    ytd = yearly.get("2026")
    metrics["ytd_2026_return"] = ytd
    metrics["ytd_2026_is_partial"] = True
    n_trades = metrics.get("completed_trades")
    metrics["low_activity"] = n_trades is not None and int(n_trades) < LOW_ACTIVITY_TRADES
    metrics["metric_version"] = metrics.get("metric_version") or METRIC_VERSION
    return metrics


def tracking_row(spec: R3Spec, report: dict[str, Any] | None = None) -> dict[str, Any]:
    m = (report or {}).get("metrics") or {}
    n_trades = m.get("completed_trades")
    flags = []
    if spec.uses_g02 and not spec.pit_correct_g02:
        flags.append("INVALID_FOR_DEPLOYMENT")
    if m.get("low_activity"):
        flags.append("LOW_ACTIVITY")
    return {
        "code": spec.code,
        "title_zh": spec.title_zh,
        "source_round2": spec.source_round2,
        "cagr": m.get("annualized_return"),
        "sharpe": m.get("sharpe_zero_rate"),
        "max_drawdown": m.get("max_drawdown"),
        "total_return": m.get("total_return"),
        "final_asset": m.get("final_asset"),
        "completed_trades": n_trades,
        "worst_year_return": m.get("worst_year_return"),
        "worst_complete_year_return": m.get("worst_complete_year_return"),
        "worst_12m": m.get("worst_rolling_12m"),
        "ytd_2026": m.get("ytd_2026_return"),
        "low_activity": m.get("low_activity"),
        "flags": flags,
        "pit_correct_g02": spec.pit_correct_g02,
        "stage": spec.stage,
    }


def load_report(code: str) -> dict[str, Any] | None:
    path = round3_dir(code) / "report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _zyquant_meta() -> dict[str, Any]:
    import zyquant

    return {
        "zyquant_version": getattr(zyquant, "__version__", None),
        "zyquant_file": getattr(zyquant, "__file__", None),
    }


def run_spec(
    spec: R3Spec,
    *,
    predictions=None,
    prepared_market=None,
    market_meta: dict[str, Any] | None = None,
    resume: bool = True,
    run_now: bool = True,
) -> dict[str, Any]:
    ensure_round3_tree()
    dest = round3_dir(spec.code)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")

    existing = load_report(spec.code)
    if resume and existing is not None:
        return {
            "status": "resumed",
            "code": spec.code,
            "report": existing,
            "summary": tracking_row(spec, existing),
            "out_dir": str(dest),
        }
    if not run_now:
        return {
            "status": "pending",
            "code": spec.code,
            "summary": tracking_row(spec),
            "note": "dry-run / RUN_NOW=False",
        }

    if predictions is None:
        predictions = load_fs00263_ensemble_predictions(start=PREDICTION_START, signal_end=SIGNAL_END)
    if prepared_market is None:
        loaded = load_prepared_market(warmup_ma5=True)
        prepared_market = loaded["prepared"]
        market_meta = loaded

    date_mult, overlay_meta, hooks = build_date_multiplier(spec, predictions)
    overlayed = apply_date_multiplier(predictions, date_mult)
    merged = overlayed.merge(
        predictions[["signal_date", "instrument_id", "score"]],
        on=["signal_date", "instrument_id"],
        suffixes=("", "_src"),
    )
    if merged.empty or not (merged["score"] == merged["score_src"]).all():
        raise RuntimeError(f"{spec.code}: score mutated")

    config = baseline_fixed_sleeve_config(exit_mode=spec.exit_mode)
    config = replace(
        config,
        name_budget_nav_fraction=spec.name_budget_nav_fraction,
        market_up_ratio_min=spec.market_up_ratio_min,
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
    attach_round2_metrics(result["nav"], result["trades"], metrics)
    attach_round3_fields(metrics)

    src_payload = r2_source_payload(spec)
    src_hash = source_config_hash(src_payload)
    extra = {
        "status": "completed_and_validated",
        "experiment_code": spec.code,
        "trial_id": TRIAL_ID,
        "capacity": CAPACITY,
        "ensemble": ENSEMBLE_METHOD,
        "variant": BACKTEST_VARIANT,
        "overlay": overlay_meta,
        "overlay_hooks": hooks,
        "name_budget_nav_fraction": spec.name_budget_nav_fraction,
        "market_up_ratio_min": spec.market_up_ratio_min,
        "predictions_file": str(ensemble_dir() / "predictions.parquet"),
        "predictions_sha256": sha256_file(ensemble_dir() / "predictions.parquet"),
        "source_round2": spec.source_round2,
        "source_round2_hash": src_hash,
        "metric_version": METRIC_VERSION,
        "pit_correct_g02": spec.pit_correct_g02,
        "cost_overrides": spec.cost_overrides,
        "skip_years": list(spec.skip_years),
        "evidence_role": "round3_historical_development",
        "note": (
            "Alpha 锁定 fs00263；只改 allocation_multiplier / 费用压力 / 评价窗。"
            "产物在课题 result/round3/，未写 Stock/runs。不是 final OOS。"
        ),
        "locked_baseline_ref": {
            "code": LOCKED_BASELINE["code"],
            "sharpe": LOCKED_BASELINE["sharpe"],
            "cagr": LOCKED_BASELINE["annualized_return"],
            "max_drawdown": LOCKED_BASELINE["max_drawdown"],
        },
    }
    if spec.uses_g02 and not spec.pit_correct_g02:
        extra["deployment_status"] = "INVALID_FOR_DEPLOYMENT"
        extra["invalid_reason"] = "G02 全样本分位（Round 2 原 config 复现）"
    if market_meta:
        extra["snapshot_id"] = market_meta.get("snapshot_id")
        extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")
    extra.update(_zyquant_meta())

    written = write_backtest_artifacts(result, dest, extra_report=extra)
    attach_round3_fields(written["report"].setdefault("metrics", {}))
    (dest / "report.json").write_text(
        json.dumps(written["report"], ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    cfg_path = dest / "config.yaml"
    assert_project_result_path(cfg_path)
    cfg_path.write_text(_dump_simple_yaml(spec.to_config()), encoding="utf-8")
    shared_cfg = ensure_round3_tree() / "configs" / f"{spec.code}.json"
    assert_project_result_path(shared_cfg)
    shared_cfg.write_text(
        json.dumps(spec.to_config(), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "code": spec.code,
        "written_at": datetime.now().astimezone().isoformat(),
        "zyquant_version": extra.get("zyquant_version"),
        "snapshot_id": extra.get("snapshot_id"),
        "snapshot_fingerprint": extra.get("snapshot_fingerprint"),
        "source_round2": spec.source_round2,
        "source_round2_hash": src_hash,
        "metric_version": METRIC_VERSION,
        "costs": {
            "slippage_bps": float(config.costs.slippage_bps),
            "commission_bps": float(config.costs.commission_bps),
            "transfer_fee_bps": float(config.costs.transfer_fee_bps),
            "sell_tax_bps": float(config.costs.sell_tax_bps),
            "overrides": spec.cost_overrides,
        },
        "limit_up_tolerance": float(config.limit_up_tolerance),
        "pit_correct_g02": spec.pit_correct_g02,
        "predictions_sha256": extra["predictions_sha256"],
    }
    meta_path = dest / "metadata.json"
    assert_project_result_path(meta_path)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    summary = tracking_row(spec, written["report"])
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    shared_metrics = ensure_round3_tree() / "metrics" / f"{spec.code}.json"
    assert_project_result_path(shared_metrics)
    shared_metrics.write_text(
        json.dumps(written["report"].get("metrics") or {}, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for name in ("nav", "trades"):
        src = dest / f"{name}.parquet"
        if src.is_file():
            target = ensure_round3_tree() / name / f"{spec.code}.parquet"
            import shutil

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


def _dump_simple_yaml(obj: Any, indent: int = 0) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
    except Exception:
        pass
    spaces = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}\n"
        lines = []
        for key, value in obj.items():
            if isinstance(value, (dict, list, tuple)):
                lines.append(f"{spaces}{key}:")
                lines.append(_dump_simple_yaml(value, indent + 1).rstrip("\n"))
            else:
                lines.append(f"{spaces}{key}: {_yaml_scalar(value)}")
        return "\n".join(lines) + "\n"
    if isinstance(obj, (list, tuple)):
        if not obj:
            return f"{spaces}[]\n"
        lines = []
        for item in obj:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{spaces}-")
                lines.append(_dump_simple_yaml(item, indent + 1).rstrip("\n"))
            else:
                lines.append(f"{spaces}- {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{spaces}{_yaml_scalar(obj)}\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"
