"""Round8 正式回测：local_executor → Stock fixed_sleeve。产物只写 result/round8/。"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..local_executor import (
    round7_fixed_sleeve_config,
    run_local_fixed_sleeve_backtest,
    write_backtest_artifacts,
)
from ..paths import (
    MARKET_END,
    PREDICTION_START,
    ROUND8_ROOT,
    assert_project_result_path,
    ensure_round8_tree,
    round8_dir,
)
from ..replay_round7.inject import inject_round7_into_overlay_caches, load_locked_alpha, load_locked_risk
from ..round3.runner import _apply_cost_overrides
from ..round4.runner import attach_round4_fields
from ..stock_readonly import load_prepared_market
from .contract import CANDIDATE_CODE, RANK_NAV, TOP_K
from .overlays import build_overlayed
from .registry import R8Spec, get_spec

_WORKER: dict[str, Any] = {}


def attach_round8_fields(nav: pd.DataFrame, trades: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    attach_round4_fields(nav, trades, metrics)
    if trades is not None and not trades.empty:
        t = trades.copy()
        t["signal_date"] = pd.to_datetime(t["signal_date"])
        t["dollar"] = pd.to_numeric(t.get("sell_proceeds"), errors="coerce") - pd.to_numeric(
            t.get("buy_cost"), errors="coerce"
        )
        if "ret" in t:
            ret = pd.to_numeric(t["ret"], errors="coerce")
            t["_rank"] = t.groupby("signal_date")["score"].rank(method="first", ascending=False)
            wins = t.loc[t["dollar"] > 0, "dollar"].sum()
            loss = t.loc[t["dollar"] < 0, "dollar"].sum()
            metrics["profit_factor"] = float(wins / abs(loss)) if loss < 0 else None
            for rk, name in ((1, "top1"), (2, "top2")):
                sub = t.loc[t["_rank"].eq(rk)]
                metrics[f"{name}_pnl"] = float(sub["dollar"].sum()) if len(sub) else 0.0
                metrics[f"{name}_mean_return"] = float(sub["ret"].mean()) if len(sub) else None
                metrics[f"{name}_n"] = int(len(sub))
                if len(sub):
                    q = sub["dollar"]
                    metrics[f"{name}_tail_top10_share"] = float(
                        q.nlargest(max(1, int(np.ceil(len(q) * 0.10)))).sum() / q.sum()
                    ) if float(q.sum()) else None
        else:
            metrics["profit_factor"] = None
    if nav is not None and not nav.empty:
        n = nav.copy()
        n["trade_date"] = pd.to_datetime(n["trade_date"])
        exp = pd.to_numeric(n["market_value"], errors="coerce") / pd.to_numeric(n["total_asset"], errors="coerce")
        exp = exp.replace([np.inf, -np.inf], np.nan).dropna()
        metrics["avg_gross_exposure"] = float(exp.mean()) if len(exp) else None
        metrics["median_exposure"] = float(exp.median()) if len(exp) else None
        cash = pd.to_numeric(n["cash"], errors="coerce") / pd.to_numeric(n["total_asset"], errors="coerce")
        metrics["mean_cash_ratio"] = float(cash.mean()) if cash.notna().any() else None
        metrics["full_exposure_days"] = int((exp >= 0.35).sum()) if len(exp) else 0
        metrics["reduced_exposure_days"] = int(((exp > 0.02) & (exp < 0.35)).sum()) if len(exp) else 0
        metrics["zero_exposure_days"] = int((exp <= 0.02).sum()) if len(exp) else 0
    loyo = metrics.get("loyo_cagr") or {}
    vals = [float(v) for v in loyo.values() if v is not None]
    if vals:
        metrics["loyo_std"] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
    return metrics


def _config_for(spec: R8Spec, cfg_over: dict[str, Any]):
    config = round7_fixed_sleeve_config(exit_mode="close_from_d2")
    top_k = int(cfg_over.get("top_k") or TOP_K)
    fracs = cfg_over.get("rank_name_budget_nav_fractions")
    if fracs is None:
        fracs = (0.3,) if top_k == 1 else tuple(RANK_NAV)
    config = replace(
        config,
        top_k=top_k,
        name_budget=10_000.0,
        name_budget_nav_fraction=None,
        rank_name_budget_nav_fractions=tuple(fracs),
        market_up_ratio_min=None,
        open_gap_min=None,
        open_gap_max=None,
    )
    config = _apply_cost_overrides(config, spec.cost_overrides)
    fee_mult = (spec.cost_overrides or {}).get("fee_mult")
    if fee_mult:
        # Round8 成本×k 把滑点一并放大；Round3 的 fee_mult 只动佣金/税
        costs = replace(
            config.costs,
            slippage_bps=float(config.costs.slippage_bps) * float(fee_mult),
        )
        config = replace(config, costs=costs)
    return config


def _load_predictions(spec: R8Spec) -> pd.DataFrame:
    if spec.predictions == "d_close":
        path = ROUND8_ROOT / "predictions" / "d_close" / "predictions.parquet"
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


def _top2_ids(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(
        method="first", ascending=False
    )
    top = work.loc[work["_rank"].le(2), ["signal_date", "_rank", "instrument_id"]]
    return top.sort_values(["signal_date", "_rank"]).reset_index(drop=True)


def _assert_top2_identity(code: str, overlayed: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """关掉 Top2 仓位不得改写当日 score 排序后的 Top2 身份。"""
    left = _top2_ids(overlayed)
    right = _top2_ids(predictions)
    dates = pd.DatetimeIndex(left["signal_date"].unique())
    right = right.loc[right["signal_date"].isin(dates)]
    merged = left.merge(right, on=["signal_date", "_rank"], suffixes=("_ov", "_src"))
    if merged.empty:
        raise RuntimeError(f"{code}: top2 identity merge empty")
    bad = merged["instrument_id_ov"].astype(str) != merged["instrument_id_src"].astype(str)
    if bool(bad.any()):
        n = int(bad.sum())
        raise RuntimeError(f"{code}: top2 identity mutated on {n} rank-rows (would let Top3 fill in)")


def run_spec(
    spec: R8Spec,
    *,
    prepared_market,
    market_meta: dict[str, Any] | None,
    resume: bool = True,
) -> dict[str, Any]:
    ensure_round8_tree()
    dest = round8_dir(spec.code)
    dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "report.json"
    if spec.skip:
        payload = {"status": "skipped", "code": spec.code, "reason": spec.skip_reason, "metrics": {}}
        if not report_path.is_file():
            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return payload
    if resume and report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("metrics") or existing.get("status") in {"skipped", "ABANDONED", "FAILED"}:
            if existing.get("status") != "FAILED":
                return {"status": "resumed", "code": spec.code, "report": existing, "out_dir": str(dest)}

    predictions = _load_predictions(spec)
    if spec.overlay_spec.get("kind") == "v1_clean":
        overlayed = predictions.copy()
        overlay_meta = {"kind": "v1_clean_risk"}
        cfg_over: dict[str, Any] = {}
    elif spec.overlay_spec.get("kind") == "combine_slot":
        raise RuntimeError(f"{spec.code} combine slot must be materialized before run")
    else:
        overlayed, overlay_meta, cfg_over = build_overlayed(spec.overlay_spec, predictions)
        if spec.bottom != 0.20 and "bottom" not in spec.overlay_spec:
            spec.overlay_spec["bottom"] = spec.bottom
            overlayed, overlay_meta, cfg_over = build_overlayed(spec.overlay_spec, predictions)

    _assert_score_unmutated(spec.code, overlayed, predictions)
    if int(cfg_over.get("top_k") or TOP_K) >= 2:
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
        "evidence_role": "round8_historical_development",
        "post_hoc": bool(spec.post_hoc),
        "predictions": spec.predictions,
        "note": "正式回测走 Stock fixed_sleeve。parent=baseline_v2。d1_open_gap 为 execution-time feature。",
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
    dest = dest if hasattr(dest, "joinpath") else round8_dir(code)
    mapping = {
        "nav.parquet": ROUND8_ROOT / "nav" / f"{code}.parquet",
        "trades.parquet": ROUND8_ROOT / "trades" / f"{code}.parquet",
        "summary.json": ROUND8_ROOT / "metrics" / f"{code}.json",
        "config.json": ROUND8_ROOT / "configs" / f"{code}.json",
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
        dest = round8_dir(code)
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
            print(f"[r8 {i}/{len(specs)}] {spec.code}", flush=True)
            code, payload = _worker(spec.code)
            out["jobs"][code] = payload
            if payload.get("status") == "FAILED":
                out["failed"].append(code)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context

        print(f"[r8] workers={n_workers} n={len(specs)}", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("fork")) as pool:
            futs = {pool.submit(_worker, spec.code): spec.code for spec in specs}
            done = 0
            for fut in as_completed(futs):
                done += 1
                code, payload = fut.result()
                out["jobs"][code] = payload
                if payload.get("status") == "FAILED":
                    out["failed"].append(code)
                print(f"[r8 {done}/{len(specs)}] {code} {payload.get('status')}", flush=True)
    out["n"] = len(specs)
    return out


def load_summary(code: str) -> dict[str, Any] | None:
    path = round8_dir(code) / "summary.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    report_path = round8_dir(code) / "report.json"
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


__all__ = ["attach_round8_fields", "load_summary", "run_codes", "run_spec"]
