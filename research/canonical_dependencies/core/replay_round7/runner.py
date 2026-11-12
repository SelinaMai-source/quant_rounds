"""用 Round7 锁分 + Top2 30/10 契约重放 overlay。产物只写 replay_round7。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..experiment_registry import ExperimentSpec
from ..local_executor import (
    round7_fixed_sleeve_config,
    run_local_fixed_sleeve_backtest,
    write_backtest_artifacts,
)
from ..paths import (
    MARKET_END,
    PREDICTION_START,
    assert_project_result_path,
    ensure_replay_round7_tree,
    replay_round7_dir,
)
from ..round2.overlays import apply_date_multiplier
from ..round2.registry import R2Spec
from ..round2.runner import build_date_multiplier as r2_build
from ..round3.overlays import build_date_multiplier as r3_build
from ..round3.registry import R3Spec
from ..round3.runner import _apply_cost_overrides
from ..round4.overlays import build_date_multiplier as r4_build
from ..round4.registry import R4Spec
from ..round4.runner import attach_round4_fields
from ..round5.overlays import build_date_multiplier as r5_build
from ..round5.registry import R5Spec, V3_OVERLAY
from ..stock_readonly import load_prepared_market, sha256_file
from .catalog import ReplayJob, overlay_jobs
from .contract import CANDIDATE_CODE, LOCKED_RISK_KINDS, RANK_NAV, TOP_K
from .inject import inject_round7_into_overlay_caches, load_locked_alpha, load_locked_risk


def uses_locked_risk(job: ReplayJob) -> bool:
    kind = (job.overlay or {}).get("kind")
    if job.round_name == "round1":
        return True
    return kind in LOCKED_RISK_KINDS


def _scaled_fractions(nav_frac: float | None, top_k: int | None, explicit) -> tuple[float, ...] | None:
    if explicit:
        return tuple(float(x) for x in explicit)
    if top_k is not None and int(top_k) == 3:
        return (0.10, 0.10, 0.10)
    base = RANK_NAV
    if nav_frac is None or abs(float(nav_frac) - 0.3) < 1e-12:
        return tuple(base)
    scale = float(nav_frac) / 0.3
    return tuple(min(1.0, max(0.0, x * scale)) for x in base)


def _config_for(job: ReplayJob):
    exit_mode = job.exit_mode or "close_from_d2"
    config = round7_fixed_sleeve_config(exit_mode=exit_mode)
    if job.top_k is not None and int(job.top_k) >= 3:
        top_k = int(job.top_k)
    else:
        top_k = int(TOP_K)
    explicit = job.rank_name_budget_nav_fractions
    if explicit and len(tuple(explicit)) != top_k:
        explicit = None
    fracs = _scaled_fractions(job.name_budget_nav_fraction, top_k, explicit)
    if fracs is not None and len(fracs) != top_k:
        if top_k == 3:
            fracs = (0.10, 0.10, 0.10)
        else:
            fracs = tuple(RANK_NAV)
    config = replace(
        config,
        top_k=int(top_k),
        name_budget=10_000.0,
        name_budget_nav_fraction=None,
        rank_name_budget_nav_fractions=fracs,
        market_up_ratio_min=None,
        open_gap_min=None,
        open_gap_max=None,
    )
    config = _apply_cost_overrides(config, job.cost_overrides)
    return config


def _build_mult(job: ReplayJob, predictions: pd.DataFrame):
    overlay = job.overlay or {}
    kind = overlay.get("kind")
    if job.round_name == "round1" or not kind or kind in {"identity"}:
        idx = pd.DatetimeIndex(pd.to_datetime(predictions["signal_date"].unique())).sort_values()
        return pd.Series(1.0, index=idx), {"kind": kind or "identity"}, None
    if job.round_name == "round2":
        spec = R2Spec(
            code=job.code,
            module="replay",
            title_zh=job.title_zh,
            description_zh=job.title_zh,
            overlay=overlay,
            exit_mode=job.exit_mode,
            name_budget_nav_fraction=job.name_budget_nav_fraction,
            market_up_ratio_min=job.market_up_ratio_min,
            needs_overlay_loop=job.needs_overlay_loop,
        )
        return r2_build(spec, predictions)
    if job.round_name == "round3":
        spec = R3Spec(
            code=job.code,
            title_zh=job.title_zh,
            description_zh=job.title_zh,
            overlay=overlay,
            exit_mode=job.exit_mode,
            name_budget_nav_fraction=job.name_budget_nav_fraction,
            market_up_ratio_min=job.market_up_ratio_min,
            skip_years=tuple(job.skip_years or ()),
            seeds=job.seeds,
            seed_leave_out=job.seed_leave_out,
            cost_overrides=job.cost_overrides,
            pit_correct_g02=job.pit_correct_g02,
        )
        return r3_build(spec, predictions)
    if job.round_name in {"round5", "round6"}:
        spec = R5Spec(
            code=job.code,
            title_zh=job.title_zh,
            description_zh=job.title_zh,
            overlay=overlay or dict(V3_OVERLAY),
            exit_mode=job.exit_mode,
            name_budget_nav_fraction=job.name_budget_nav_fraction,
            market_up_ratio_min=job.market_up_ratio_min,
            skip_years=tuple(job.skip_years or ()),
        )
        return r5_build(spec, predictions)
    spec = R4Spec(
        code=job.code,
        title_zh=job.title_zh,
        description_zh=job.title_zh,
        overlay=overlay,
        exit_mode=job.exit_mode,
        name_budget_nav_fraction=job.name_budget_nav_fraction,
        market_up_ratio_min=job.market_up_ratio_min,
        top_k=int(job.top_k or TOP_K),
        rank_name_budget_nav_fractions=job.rank_name_budget_nav_fractions,
        skip_years=tuple(job.skip_years or ()),
        random_seed=job.random_seed,
        cost_overrides=job.cost_overrides,
    )
    return r4_build(spec, predictions)


def _write_abort(job: ReplayJob) -> dict[str, Any]:
    dest = replay_round7_dir(job.round_name, job.code)
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": job.abort,
        "code": job.code,
        "title_zh": job.title_zh,
        "reason": (
            "本线已是 Top2 30/10。旧 F 组是在 Top1 上再买 Top2，语义重复。"
            "改为 Top3 诊断见 R4_F_TOP3_diag，本组 ABORT_SEMANTIC_SHIFT。"
        ),
        "metrics": {},
    }
    path = dest / "report.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (dest / "summary.json").write_text(
        json.dumps({"code": job.code, "status": job.abort, "title_zh": job.title_zh}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {"status": job.abort, "code": job.code, "out_dir": str(dest)}


def run_job(
    job: ReplayJob,
    *,
    predictions_alpha: pd.DataFrame,
    predictions_risk: pd.DataFrame,
    prepared_market,
    market_meta: dict[str, Any] | None,
    resume: bool = True,
) -> dict[str, Any]:
    ensure_replay_round7_tree()
    dest = replay_round7_dir(job.round_name, job.code)
    dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "report.json"
    if job.abort:
        if resume and report_path.is_file():
            return {"status": "resumed_abort", "code": job.code, "out_dir": str(dest)}
        return _write_abort(job)
    if job.skip or (job.diagnostic_only and job.round_name == "round2"):
        payload = {"status": "skipped", "code": job.code, "reason": job.skip_reason or "diagnostic_only"}
        if not report_path.is_file():
            report_path.write_text(json.dumps({**payload, "metrics": {}}, ensure_ascii=False, indent=2) + "\n")
        return payload
    if resume and report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("metrics") or existing.get("status") in {"ABORT_SEMANTIC_SHIFT", "skipped", "ABANDONED"}:
            return {"status": "resumed", "code": job.code, "report": existing, "out_dir": str(dest)}

    keep_r7_risk = uses_locked_risk(job)
    predictions = predictions_risk if keep_r7_risk else predictions_alpha
    date_mult, overlay_meta, hooks = _build_mult(job, predictions)
    if overlay_meta.get("status") == "ABANDONED":
        payload = {"status": "ABANDONED", "code": job.code, "reason": overlay_meta.get("reason"), "overlay": overlay_meta, "metrics": {}}
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return payload

    overlayed = apply_date_multiplier(predictions, date_mult)
    if job.random_seed is not None:
        rng = np.random.default_rng(int(job.random_seed))
        overlayed = overlayed.copy()
        overlayed["score"] = rng.random(len(overlayed))
    else:
        merged = overlayed.merge(
            predictions[["signal_date", "instrument_id", "score"]],
            on=["signal_date", "instrument_id"],
            suffixes=("", "_src"),
        )
        if merged.empty or not (merged["score"] == merged["score_src"]).all():
            raise RuntimeError(f"{job.code}: score mutated")

    config = _config_for(job)
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
        "experiment_code": job.code,
        "trial_id": CANDIDATE_CODE,
        "capacity": "dual90_10_gap_floor",
        "ensemble": "dual90_10_clean",
        "variant": "rank30_10",
        "parent": CANDIDATE_CODE,
        "used_locked_round7_risk": keep_r7_risk,
        "overlay": overlay_meta,
        "overlay_hooks": hooks,
        "top_k": config.top_k,
        "rank_name_budget_nav_fractions": config.rank_name_budget_nav_fractions,
        "open_gap_min": config.open_gap_min,
        "c03_source": "round7_dual90_10_current_scores_rolling250",
        "metric_version": "r3_locked_window_v1",
        "evidence_role": "round7_baseline_replay_historical_development",
        "note": (
            "正式回测走 Stock fixed_sleeve。旧 roundN / replay_fs01122 未覆盖。"
            "d1_open_gap 为 execution-time feature。"
        ),
    }
    if market_meta:
        extra["snapshot_id"] = market_meta.get("snapshot_id")
        extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")
    written = write_backtest_artifacts(result, dest, extra_report=extra)
    summary = {
        "code": job.code,
        "title_zh": job.title_zh,
        "round": job.round_name,
        "cagr": metrics.get("annualized_return"),
        "sharpe": metrics.get("sharpe_zero_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
        "total_return": metrics.get("total_return"),
        "completed_trades": metrics.get("completed_trades"),
        "trade_win_rate": metrics.get("trade_win_rate"),
        "average_holding_sessions": metrics.get("average_holding_sessions"),
        "worst_year_return": metrics.get("worst_complete_year_return") or metrics.get("worst_year_return"),
        "cagr_ex2024": metrics.get("cagr_ex2024"),
        "yearly_returns": metrics.get("yearly_returns"),
        "used_locked_round7_risk": keep_r7_risk,
        "written_at": datetime.now().astimezone().isoformat(),
    }
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    cfg_path = dest / "config.json"
    assert_project_result_path(cfg_path)
    cfg_path.write_text(
        json.dumps(job.__dict__, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    _ = ExperimentSpec
    _ = sha256_file
    return {"status": "ran", "code": job.code, "out_dir": written["out_dir"], "summary": summary, "report": written["report"]}


_WORKER: dict[str, Any] = {}


def _overlay_worker(code: str) -> tuple[str, dict[str, Any]]:
    job = _WORKER["jobs"][code]
    resume = bool(_WORKER.get("resume", True))
    try:
        payload = run_job(
            job,
            predictions_alpha=_WORKER["alpha"],
            predictions_risk=_WORKER["risk"],
            prepared_market=_WORKER["prepared"],
            market_meta=_WORKER["meta"],
            resume=resume,
        )
        return code, payload
    except Exception as exc:
        dest = replay_round7_dir(job.round_name, job.code)
        dest.mkdir(parents=True, exist_ok=True)
        payload = {"status": "FAILED", "code": job.code, "error": repr(exc), "metrics": {}}
        (dest / "report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return code, payload


def run_overlays(*, resume: bool = True, codes: list[str] | None = None, workers: int = 1) -> dict[str, Any]:
    inject = inject_round7_into_overlay_caches()
    alpha = load_locked_alpha()
    risk = load_locked_risk()
    loaded = load_prepared_market(warmup_ma5=True)
    prepared = loaded["prepared"]
    jobs = overlay_jobs()
    if codes:
        want = set(codes)
        jobs = [j for j in jobs if j.code in want]
    out: dict[str, Any] = {"inject": inject, "jobs": {}}
    failed: list[str] = []
    n_workers = max(1, int(workers))
    _WORKER.update({
        "jobs": {j.code: j for j in jobs},
        "alpha": alpha,
        "risk": risk,
        "prepared": prepared,
        "meta": loaded,
        "resume": resume,
    })

    if n_workers == 1:
        for i, job in enumerate(jobs, start=1):
            print(f"[overlay {i}/{len(jobs)}] {job.code}", flush=True)
            code, payload = _overlay_worker(job.code)
            out["jobs"][code] = payload
            if payload.get("status") == "FAILED":
                failed.append(code)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context

        print(f"[overlay] workers={n_workers} n_jobs={len(jobs)}", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("fork")) as pool:
            futs = {pool.submit(_overlay_worker, job.code): job.code for job in jobs}
            done = 0
            for fut in as_completed(futs):
                done += 1
                code, payload = fut.result()
                out["jobs"][code] = payload
                if payload.get("status") == "FAILED":
                    failed.append(code)
                print(f"[overlay {done}/{len(jobs)}] {code} {payload.get('status')}", flush=True)
    out["failed"] = failed
    out["n_jobs"] = len(jobs)
    out["workers"] = n_workers
    return out
