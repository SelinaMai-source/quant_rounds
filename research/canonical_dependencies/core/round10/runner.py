"""Round10 正式回测：local_executor → Stock fixed_sleeve。产物只写 result/round10/。"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime
from typing import Any

import pandas as pd

from ..local_executor import (
    round7_fixed_sleeve_config,
    run_local_fixed_sleeve_backtest,
    write_backtest_artifacts,
)
from ..paths import (
    MARKET_END,
    PREDICTION_START,
    ROUND10_ROOT,
    assert_project_result_path,
    ensure_round10_tree,
    round10_dir,
)
from ..replay_round7.inject import inject_round7_into_overlay_caches, load_locked_alpha, load_locked_risk
from ..round3.runner import _apply_cost_overrides
from ..round8.runner import attach_round8_fields
from ..stock_readonly import load_prepared_market
from .contract import CANDIDATE_CODE, PARENT_CODE, RANK_NAV, TOP_K
from .overlays import build_overlayed
from .registry import R10Spec, get_spec

_WORKER: dict[str, Any] = {}


def _config_for(spec: R10Spec, cfg_over: dict[str, Any]):
    config = round7_fixed_sleeve_config(exit_mode="close_from_d2")
    top_k = int(cfg_over.get("top_k") or TOP_K)
    fracs = cfg_over.get("rank_name_budget_nav_fractions")
    if fracs is None:
        fracs = (0.3,) if top_k == 1 else tuple(RANK_NAV)
    cash = float(spec.initial_cash) if spec.initial_cash else 300_000.0
    config = replace(
        config,
        top_k=top_k,
        name_budget=10_000.0,
        name_budget_nav_fraction=None,
        rank_name_budget_nav_fractions=tuple(fracs),
        market_up_ratio_min=None,
        open_gap_min=None,
        open_gap_max=None,
        initial_cash=cash,
        sleeve_cash=cash / 2.0,
    )
    config = _apply_cost_overrides(config, spec.cost_overrides)
    fee_mult = (spec.cost_overrides or {}).get("fee_mult")
    if fee_mult:
        costs = replace(
            config.costs,
            slippage_bps=float(config.costs.slippage_bps) * float(fee_mult),
        )
        config = replace(config, costs=costs)
    return config


def _load_predictions(spec: R10Spec) -> pd.DataFrame:
    if spec.predictions == "d_close":
        path = ROUND10_ROOT / "predictions" / "d_close" / "predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing D-close predictions: {path}")
        frame = pd.read_parquet(path)
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        return frame
    if spec.overlay_spec.get("kind") == "v1_clean":
        return load_locked_risk()
    return load_locked_alpha()


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


def _rank_ids(frame: pd.DataFrame, rank: int) -> pd.DataFrame:
    work = frame.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(
        method="first", ascending=False
    )
    top = work.loc[work["_rank"].eq(rank), ["signal_date", "instrument_id"]]
    return top.sort_values("signal_date").reset_index(drop=True)


def _assert_top1_lock(code: str, overlayed: pd.DataFrame, predictions: pd.DataFrame) -> None:
    left = _rank_ids(overlayed, 1)
    right = _rank_ids(predictions, 1)
    dates = pd.DatetimeIndex(left["signal_date"].unique())
    right = right.loc[right["signal_date"].isin(dates)]
    merged = left.merge(right, on="signal_date", suffixes=("_ov", "_src"))
    if merged.empty:
        raise RuntimeError(f"FAIL_TOP1_MUTATION {code}: top1 merge empty")
    bad = merged["instrument_id_ov"].astype(str) != merged["instrument_id_src"].astype(str)
    if bool(bad.any()):
        n = int(bad.sum())
        raise RuntimeError(f"FAIL_TOP1_MUTATION {code}: top1 mutated on {n} days")


def _assert_top2_identity(code: str, overlayed: pd.DataFrame, predictions: pd.DataFrame) -> None:
    left = pd.concat(
        [_rank_ids(overlayed, 1).assign(_rank=1), _rank_ids(overlayed, 2).assign(_rank=2)],
        ignore_index=True,
    )
    right = pd.concat(
        [_rank_ids(predictions, 1).assign(_rank=1), _rank_ids(predictions, 2).assign(_rank=2)],
        ignore_index=True,
    )
    dates = pd.DatetimeIndex(left["signal_date"].unique())
    right = right.loc[right["signal_date"].isin(dates)]
    merged = left.merge(right, on=["signal_date", "_rank"], suffixes=("_ov", "_src"))
    if merged.empty:
        raise RuntimeError(f"{code}: top2 identity merge empty")
    bad = merged["instrument_id_ov"].astype(str) != merged["instrument_id_src"].astype(str)
    if bool(bad.any()):
        n = int(bad.sum())
        raise RuntimeError(f"{code}: top2 identity mutated on {n} rank-rows")


def run_spec(
    spec: R10Spec,
    *,
    prepared_market,
    market_meta: dict[str, Any] | None,
    resume: bool = True,
) -> dict[str, Any]:
    ensure_round10_tree()
    dest = round10_dir(spec.code)
    dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "report.json"
    if spec.skip:
        payload = {"status": "skipped", "code": spec.code, "reason": spec.skip_reason, "metrics": {}}
        if not report_path.is_file():
            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return payload
    if resume and report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("status") == "skipped" and not spec.skip:
            existing = {}
        if existing.get("metrics") or existing.get("status") in {"skipped", "ABANDONED", "FAILED"}:
            if existing.get("status") != "FAILED":
                return {"status": "resumed", "code": spec.code, "report": existing, "out_dir": str(dest)}

    predictions = _load_predictions(spec)
    overlayed, overlay_meta, cfg_over = build_overlayed(spec.overlay_spec, predictions)
    if spec.lock_score:
        _assert_score_unmutated(spec.code, overlayed, predictions)
    if spec.lock_top1:
        _assert_top1_lock(spec.code, overlayed, predictions)
    if spec.lock_top2 and int(cfg_over.get("top_k") or TOP_K) >= 2:
        _assert_top2_identity(spec.code, overlayed, predictions)

    config = _config_for(spec, cfg_over)
    result = run_local_fixed_sleeve_backtest(
        overlayed,
        prepared_market,
        config=config,
        start_date=PREDICTION_START,
        end_date=MARKET_END,
    )
    metrics = result["report"].setdefault("metrics", {})
    attach_round8_fields(result["nav"], result["trades"], metrics)
    extra = {
        "status": "completed_and_validated",
        "experiment_code": spec.code,
        "parent": spec.parent,
        "trial_id": CANDIDATE_CODE,
        "overlay": overlay_meta,
        "top_k": config.top_k,
        "rank_name_budget_nav_fractions": config.rank_name_budget_nav_fractions,
        "metric_version": "r3_locked_window_v1",
        "evidence_role": "round10_historical_development",
        "post_hoc": bool(spec.post_hoc),
        "predictions": spec.predictions,
        "note": f"正式回测走 Stock fixed_sleeve。parent=baseline_v4={PARENT_CODE}。Top1 LOCK。B03 replacement，不再乘 E02。",
    }
    if market_meta:
        extra["snapshot_id"] = market_meta.get("snapshot_id")
        extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")
    written = write_backtest_artifacts(result, dest, extra_report=extra)
    summary = {
        "code": spec.code,
        "title_zh": spec.title_zh,
        "stage": spec.stage,
        "parent": spec.parent,
        "cagr": metrics.get("annualized_return"),
        "sharpe": metrics.get("sharpe_zero_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
        "total_return": metrics.get("total_return"),
        "completed_trades": metrics.get("completed_trades"),
        "trade_win_rate": metrics.get("trade_win_rate"),
        "sortino": metrics.get("sortino_zero_rate"),
        "calmar": metrics.get("calmar"),
        "worst_year_return": metrics.get("worst_complete_year_return") or metrics.get("worst_year_return"),
        "worst_rolling_12m": metrics.get("worst_rolling_12m"),
        "loyo_median": metrics.get("loyo_median"),
        "loyo_worst": metrics.get("loyo_worst"),
        "loyo_std": metrics.get("loyo_std"),
        "loyo_cagr": metrics.get("loyo_cagr"),
        "cagr_ex2024": metrics.get("cagr_ex2024"),
        "profit_factor": metrics.get("profit_factor"),
        "avg_gross_exposure": metrics.get("avg_gross_exposure"),
        "median_exposure": metrics.get("median_exposure"),
        "mean_cash_ratio": metrics.get("mean_cash_ratio"),
        "full_exposure_days": metrics.get("full_exposure_days"),
        "reduced_exposure_days": metrics.get("reduced_exposure_days"),
        "zero_exposure_days": metrics.get("zero_exposure_days"),
        "top1_pnl": metrics.get("top1_pnl"),
        "top2_pnl": metrics.get("top2_pnl"),
        "top1_mean_return": metrics.get("top1_mean_return"),
        "top2_mean_return": metrics.get("top2_mean_return"),
        "yearly_returns": metrics.get("yearly_returns"),
        "written_at": datetime.now().astimezone().isoformat(),
    }
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (dest / "config.json").write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _mirror_shared(spec.code, dest)
    return {"status": "ran", "code": spec.code, "out_dir": written["out_dir"], "summary": summary}


def _mirror_shared(code: str, dest) -> None:
    dest = dest if hasattr(dest, "joinpath") else round10_dir(code)
    mapping = {
        "nav.parquet": ROUND10_ROOT / "nav" / f"{code}.parquet",
        "trades.parquet": ROUND10_ROOT / "trades" / f"{code}.parquet",
        "summary.json": ROUND10_ROOT / "metrics" / f"{code}.json",
        "config.json": ROUND10_ROOT / "configs" / f"{code}.json",
    }
    for src_name, target in mapping.items():
        src = dest / src_name
        if src.is_file():
            assert_project_result_path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def _worker(code: str) -> tuple[str, dict[str, Any]]:
    spec = _WORKER["jobs"][code]
    try:
        payload = run_spec(
            spec,
            prepared_market=_WORKER["prepared"],
            market_meta=_WORKER["meta"],
            resume=bool(_WORKER.get("resume", True)),
        )
        return code, payload
    except Exception as exc:
        dest = round10_dir(code)
        dest.mkdir(parents=True, exist_ok=True)
        payload = {"status": "FAILED", "code": code, "error": repr(exc), "metrics": {}}
        (dest / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return code, payload


def run_codes(codes: list[str], *, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    inject_round7_into_overlay_caches()
    loaded = load_prepared_market(warmup_ma5=True)
    specs = [get_spec(c) for c in codes]
    _WORKER.update({
        "jobs": {s.code: s for s in specs},
        "prepared": loaded["prepared"],
        "meta": loaded,
        "resume": resume,
    })
    out: dict[str, Any] = {"jobs": {}, "failed": []}
    n_workers = max(1, int(jobs))
    if n_workers == 1:
        for i, spec in enumerate(specs, start=1):
            print(f"[r10 {i}/{len(specs)}] {spec.code}", flush=True)
            code, payload = _worker(spec.code)
            out["jobs"][code] = payload
            if payload.get("status") == "FAILED":
                out["failed"].append(code)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context

        print(f"[r10] workers={n_workers} n={len(specs)}", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("fork")) as pool:
            futs = {pool.submit(_worker, spec.code): spec.code for spec in specs}
            done = 0
            for fut in as_completed(futs):
                done += 1
                code, payload = fut.result()
                out["jobs"][code] = payload
                if payload.get("status") == "FAILED":
                    out["failed"].append(code)
                print(f"[r10 {done}/{len(specs)}] {code} {payload.get('status')}", flush=True)
    out["n"] = len(specs)
    return out


def load_summary(code: str) -> dict[str, Any] | None:
    path = round10_dir(code) / "summary.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    report_path = round10_dir(code) / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report.get("metrics") or {}
        return {
            "code": code,
            "status": report.get("status"),
            "cagr": metrics.get("annualized_return"),
            "sharpe": metrics.get("sharpe_zero_rate"),
            "max_drawdown": metrics.get("max_drawdown"),
            "completed_trades": metrics.get("completed_trades"),
            "yearly_returns": metrics.get("yearly_returns"),
            "loyo_median": metrics.get("loyo_median"),
            "loyo_worst": metrics.get("loyo_worst"),
            "cagr_ex2024": metrics.get("cagr_ex2024"),
        }
    return None


__all__ = ["load_summary", "run_codes", "run_spec"]
