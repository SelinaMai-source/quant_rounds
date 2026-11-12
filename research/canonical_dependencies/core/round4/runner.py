"""Round 4 回测入口：复用 local_executor，产物只写 result/round4/{code}/。"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..local_executor import (
    baseline_fixed_sleeve_config,
    run_local_fixed_sleeve_backtest,
    write_backtest_artifacts,
)
from ..metrics import METRIC_VERSION, clip_nav_for_metrics
from ..paths import (
    BACKTEST_VARIANT,
    CAPACITY,
    ENSEMBLE_METHOD,
    MARKET_END,
    PREDICTION_START,
    SIGNAL_END,
    TRIAL_ID,
    assert_project_result_path,
    ensure_round4_tree,
    round4_dir,
)
from ..round2.metrics import attach_round2_metrics, worst_rolling
from ..round3.runner import _apply_cost_overrides, _dump_simple_yaml, attach_round3_fields
from ..stock_readonly import (
    ensemble_dir,
    load_fs00263_ensemble_predictions,
    load_prepared_market,
    sha256_file,
)
from .overlays import apply_date_multiplier, build_date_multiplier
from .registry import LOW_ACTIVITY_TRADES, PARENT_DEFAULT, R4Spec

COMPLETE_YEARS = tuple(range(2018, 2026))


def attach_round4_fields(nav: pd.DataFrame, trades: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    attach_round2_metrics(nav, trades, metrics)
    attach_round3_fields(metrics)
    clipped = clip_nav_for_metrics(nav) if nav is not None and not nav.empty else nav
    if clipped is None or clipped.empty:
        metrics["metric_version"] = METRIC_VERSION
        return metrics
    ordered = clipped.sort_values("trade_date").copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"])
    asset = ordered.set_index("trade_date")["total_asset"].astype(float)
    daily = asset.pct_change()
    first = ordered["trade_date"].iloc[0]
    last = ordered["trade_date"].iloc[-1]
    years = max((last - first).days / 365.25, 1 / 365.25)
    initial = 300_000.0

    def _flat_cagr(skip: int | None) -> float | None:
        adj = daily.copy()
        if skip is not None:
            adj.loc[adj.index.year == int(skip)] = 0.0
        adj = adj.fillna(0.0)
        wealth = float((1.0 + adj).prod())
        return wealth ** (1.0 / years) - 1.0

    metrics["cagr_ex2024"] = _flat_cagr(2024)
    loyo = {y: _flat_cagr(y) for y in COMPLETE_YEARS}
    vals = [v for v in loyo.values() if v is not None]
    metrics["loyo_cagr"] = {str(k): v for k, v in loyo.items()}
    metrics["loyo_median"] = float(np.median(vals)) if vals else None
    metrics["loyo_worst"] = float(np.min(vals)) if vals else None
    yearly = metrics.get("yearly_returns") or {}
    complete = [yearly[str(y)] for y in COMPLETE_YEARS if str(y) in yearly]
    metrics["yearly_std"] = float(np.std(complete, ddof=1)) if len(complete) >= 2 else None
    if complete:
        logs = [np.log1p(x) for x in complete if x is not None and x > -0.999]
        if logs:
            best_i = int(np.argmax(logs))
            keys = [str(y) for y in COMPLETE_YEARS if str(y) in yearly]
            metrics["best_complete_year"] = keys[best_i] if best_i < len(keys) else None
            total_log = float(np.sum(logs))
            metrics["best_year_log_share"] = float(logs[best_i] / total_log) if total_log else None
    wr = worst_rolling(daily.dropna(), windows=(126, 252, 504))
    metrics["worst_rolling_6m"] = wr.get("worst_rolling_126d")
    metrics["worst_rolling_12m"] = metrics.get("worst_rolling_12m") or wr.get("worst_rolling_252d")
    metrics["worst_rolling_24m"] = wr.get("worst_rolling_504d")
    if len(daily.dropna()) >= 126:
        roll6 = (1.0 + daily.dropna()).rolling(126).apply(lambda x: float(np.prod(x) - 1.0), raw=True)
        metrics["median_rolling_6m"] = float(roll6.median()) if roll6.notna().any() else None
    if len(daily.dropna()) >= 252:
        roll12 = (1.0 + daily.dropna()).rolling(252).apply(lambda x: float(np.prod(x) - 1.0), raw=True)
        metrics["median_rolling_12m"] = float(roll12.median()) if roll12.notna().any() else None
    if len(daily.dropna()) >= 504:
        roll24 = (1.0 + daily.dropna()).rolling(504).apply(lambda x: float(np.prod(x) - 1.0), raw=True)
        metrics["median_rolling_24m"] = float(roll24.median()) if roll24.notna().any() else None
    n_trades = metrics.get("completed_trades")
    metrics["low_activity"] = n_trades is not None and int(n_trades) < LOW_ACTIVITY_TRADES
    metrics["metric_version"] = METRIC_VERSION
    metrics["parent"] = PARENT_DEFAULT
    _ = initial
    return metrics


def tracking_row(spec: R4Spec, report: dict[str, Any] | None = None) -> dict[str, Any]:
    m = (report or {}).get("metrics") or {}
    flags = []
    if m.get("low_activity"):
        flags.append("LOW_ACTIVITY")
    if not spec.pit_ok:
        flags.append("PIT_FAIL")
    if spec.diagnostic:
        flags.append("DIAGNOSTIC")
    return {
        "code": spec.code,
        "title_zh": spec.title_zh,
        "parent": spec.parent,
        "replace_module": spec.replace_module,
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
    }


def load_report(code: str) -> dict[str, Any] | None:
    path = round4_dir(code) / "report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_spec(
    spec: R4Spec,
    *,
    predictions=None,
    prepared_market=None,
    market_meta: dict[str, Any] | None = None,
    resume: bool = True,
    run_now: bool = True,
) -> dict[str, Any]:
    ensure_round4_tree()
    dest = round4_dir(spec.code)
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

    overlayed = apply_date_multiplier(predictions, date_mult)
    if spec.random_seed is None:
        merged = overlayed.merge(
            predictions[["signal_date", "instrument_id", "score"]],
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

    extra = {
        "status": "completed_and_validated",
        "experiment_code": spec.code,
        "trial_id": TRIAL_ID,
        "capacity": CAPACITY,
        "ensemble": ENSEMBLE_METHOD,
        "variant": BACKTEST_VARIANT,
        "parent": spec.parent,
        "replace_module": spec.replace_module,
        "overlay": overlay_meta,
        "overlay_hooks": hooks,
        "name_budget_nav_fraction": spec.name_budget_nav_fraction,
        "market_up_ratio_min": spec.market_up_ratio_min,
        "top_k": spec.top_k,
        "rank_name_budget_nav_fractions": spec.rank_name_budget_nav_fractions,
        "random_seed": spec.random_seed,
        "predictions_file": str(ensemble_dir() / "predictions.parquet"),
        "predictions_sha256": sha256_file(ensemble_dir() / "predictions.parquet"),
        "metric_version": METRIC_VERSION,
        "cost_overrides": spec.cost_overrides,
        "skip_years": list(spec.skip_years),
        "evidence_role": "round4_historical_development",
        "pit": True,
        "note": (
            "默认对照 baseline_v2=R3_03。不重训 fs00263，不改 score。"
            "产物在课题 result/round4/，未写 Stock/runs。不是 final OOS。"
        ),
    }
    if market_meta:
        extra["snapshot_id"] = market_meta.get("snapshot_id")
        extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")

    written = write_backtest_artifacts(result, dest, extra_report=extra)
    attach_round4_fields(written.get("report", {}).get("nav") if False else result["nav"], result["trades"],
                         written["report"].setdefault("metrics", {}))
    (dest / "report.json").write_text(
        json.dumps(written["report"], ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    cfg_path = dest / "config.yaml"
    assert_project_result_path(cfg_path)
    cfg_path.write_text(_dump_simple_yaml(spec.to_config()), encoding="utf-8")
    shared_cfg = ensure_round4_tree() / "configs" / f"{spec.code}.json"
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
        "replace_module": spec.replace_module,
        "snapshot_id": extra.get("snapshot_id"),
        "snapshot_fingerprint": extra.get("snapshot_fingerprint"),
        "predictions_sha256": extra["predictions_sha256"],
        "top_k": spec.top_k,
        "rank_name_budget_nav_fractions": spec.rank_name_budget_nav_fractions,
        "random_seed": spec.random_seed,
    }
    meta_path = dest / "metadata.json"
    assert_project_result_path(meta_path)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    summary = tracking_row(spec, written["report"])
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    shared_metrics = ensure_round4_tree() / "metrics" / f"{spec.code}.json"
    assert_project_result_path(shared_metrics)
    shared_metrics.write_text(
        json.dumps(written["report"].get("metrics") or {}, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for name in ("nav", "trades"):
        src = dest / f"{name}.parquet"
        if src.is_file():
            target = ensure_round4_tree() / name / f"{spec.code}.parquet"
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


def run_many(specs: list[R4Spec], *, predictions, prepared, meta, resume: bool, run_now: bool) -> dict[str, Any]:
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
