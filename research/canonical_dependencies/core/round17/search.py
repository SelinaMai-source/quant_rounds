"""2018–2021 真实 5000×5seed。失败记真实异常，指标不填 0。不得静默缩到 500。"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND17_ROOT, assert_project_result_path, ensure_round17_tree
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from ..round7.train import _load_panel
from .backtest import apply_b03, discovery_backtest, fuse_heads, year_stats
from .common import dump_json, file_sha256, finite_or_none, load_json, sha256_json, write_md
from .contract import DEV_END, DEV_START, DISCOVERY_XGB_N_JOBS, N_TRIALS, SEEDS, trial_dir
from .factors import load_new_factors
from .freeze import load_trials, require_frozen
from .train import train_head_open

SEARCH_MD = ROUND17_ROOT / "06_discovery_results.md"
SEARCH_JSON = ROUND17_ROOT / "search" / "search_status.json"
DATA_DIR = ROUND17_ROOT / "data"
KEYS = ["trade_date", "instrument_id"]
_G: dict[str, Any] = {}


def prepare_panels(*, resume: bool = True) -> dict[str, Any]:
    ensure_round17_tree()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "five_training": DATA_DIR / "five_training.parquet",
        "five_candidate": DATA_DIR / "five_candidate.parquet",
        "ten_training": DATA_DIR / "ten_training.parquet",
        "ten_candidate": DATA_DIR / "ten_candidate.parquet",
        "five_frozen": DATA_DIR / "five_frozen.parquet",
        "ten_frozen": DATA_DIR / "ten_frozen.parquet",
    }
    if resume and all(p.is_file() for p in paths.values()):
        return {"status": "resumed", "paths": {k: str(v) for k, v in paths.items()}}
    factors = load_new_factors()
    for head, tkey, ckey in (
        (FIVE_HEAD, "five_training", "five_candidate"),
        (TEN_HEAD, "ten_training", "ten_candidate"),
    ):
        training, candidates = _load_panel(head)
        # reload full panels so we keep extra original columns needed? _load_panel only locked features.
        training = pd.read_parquet(head.panel_dir / "training_panel.parquet")
        candidates = pd.read_parquet(head.panel_dir / "candidate_panel.parquet")
        training["trade_date"] = pd.to_datetime(training["trade_date"])
        candidates["trade_date"] = pd.to_datetime(candidates["trade_date"])
        training["instrument_id"] = training["instrument_id"].astype(str)
        candidates["instrument_id"] = candidates["instrument_id"].astype(str)
        training = training.merge(factors, on=KEYS, how="left")
        candidates = candidates.merge(factors, on=KEYS, how="left")
        training.to_parquet(paths[tkey], index=False)
        candidates.to_parquet(paths[ckey], index=False)
    from ..paths import ROUND7_ROOT

    five_f = pd.read_parquet(ROUND7_ROOT / "models" / "five" / "predictions.parquet")
    ten_f = pd.read_parquet(ROUND7_ROOT / "models" / "ten" / "predictions.parquet")
    five_f.to_parquet(paths["five_frozen"], index=False)
    ten_f.to_parquet(paths["ten_frozen"], index=False)
    meta = {"status": "prepared", "paths": {k: str(v) for k, v in paths.items()}, "filled_with_zero": False}
    dump_json(DATA_DIR / "panels.json", meta)
    return meta


def _init_worker() -> None:
    _G["five_training"] = pd.read_parquet(DATA_DIR / "five_training.parquet")
    _G["five_candidate"] = pd.read_parquet(DATA_DIR / "five_candidate.parquet")
    _G["ten_training"] = pd.read_parquet(DATA_DIR / "ten_training.parquet")
    _G["ten_candidate"] = pd.read_parquet(DATA_DIR / "ten_candidate.parquet")
    _G["five_frozen"] = pd.read_parquet(DATA_DIR / "five_frozen.parquet")
    _G["ten_frozen"] = pd.read_parquet(DATA_DIR / "ten_frozen.parquet")
    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate"):
        _G[key]["trade_date"] = pd.to_datetime(_G[key]["trade_date"])
        _G[key]["instrument_id"] = _G[key]["instrument_id"].astype(str)
    for key in ("five_frozen", "ten_frozen"):
        _G[key]["signal_date"] = pd.to_datetime(_G[key]["signal_date"])
        _G[key]["instrument_id"] = _G[key]["instrument_id"].astype(str)
    from .backtest import cached_market

    cached_market()


def _classify(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    low = msg.lower()
    if "invalid_factor" in low or "missing required columns" in low:
        return "INVALID_FACTOR"
    if "data_unavailable" in low or "fileNotfound" in low or "filenotfound" in low:
        return "DATA_UNAVAILABLE"
    if "backtest" in low or "sleeve" in low:
        return "BACKTEST_FAILED"
    return "TRAINING_FAILED"


def _empty_metrics() -> dict[str, Any]:
    return {
        "dev_sharpe": None,
        "dev_cagr": None,
        "dev_maxdd": None,
        "dev_median_year": None,
        "dev_worst_year": None,
        "dev_OOS_NDCG": None,
        "dev_total_return": None,
        "yearly_returns": None,
    }


def _run_trial(trial: dict[str, Any]) -> dict[str, Any]:
    tid = str(trial["trial_id"])
    dest = trial_dir(tid)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest)
    metrics_path = dest / "metrics.json"
    if metrics_path.is_file():
        existing = load_json(metrics_path)
        if existing.get("status") in {
            "SUCCESS", "INVALID_FACTOR", "TRAINING_FAILED", "BACKTEST_FAILED", "DATA_UNAVAILABLE",
        }:
            return existing
    try:
        five_ex = list(trial.get("five_extras") or [])
        ten_ex = list(trial.get("ten_extras") or [])
        ndcgs = []
        if five_ex:
            five_feats = list(FIVE_FEATURES) + five_ex
            five_pay = train_head_open(
                training=_G["five_training"],
                candidates=_G["five_candidate"],
                features=five_feats,
                target=FIVE_HEAD.target,
                label_end_column=FIVE_HEAD.label_end_column,
                rank_n_bins=FIVE_HEAD.rank_n_bins,
                eval_metric=FIVE_HEAD.eval_metric,
                subsample=0.8,
                colsample_bytree=0.8,
                first_prediction_year=2018,
                last_prediction_year=2021,
                output_dir=None,
                seeds=SEEDS,
                n_jobs=DISCOVERY_XGB_N_JOBS,
                save_models=False,
                resume=False,
            )
            five_pred = five_pay["ensemble"]
            if five_pay.get("mean_fold_ndcg") is not None:
                ndcgs.append(float(five_pay["mean_fold_ndcg"]))
            five_div = five_pay.get("seed_diversity")
        else:
            five_pred = _G["five_frozen"]
            five_div = {"status": "FROZEN_DETERMINISTIC_10_10"}
        if ten_ex:
            ten_feats = list(TEN_FEATURES) + ten_ex
            ten_pay = train_head_open(
                training=_G["ten_training"],
                candidates=_G["ten_candidate"],
                features=ten_feats,
                target=TEN_HEAD.target,
                label_end_column=TEN_HEAD.label_end_column,
                rank_n_bins=TEN_HEAD.rank_n_bins,
                eval_metric=TEN_HEAD.eval_metric,
                subsample=0.8,
                colsample_bytree=0.8,
                first_prediction_year=2018,
                last_prediction_year=2021,
                output_dir=None,
                seeds=SEEDS,
                n_jobs=DISCOVERY_XGB_N_JOBS,
                save_models=False,
                resume=False,
            )
            ten_pred = ten_pay["ensemble"]
            if ten_pay.get("mean_fold_ndcg") is not None:
                ndcgs.append(float(ten_pay["mean_fold_ndcg"]))
            ten_div = ten_pay.get("seed_diversity")
        else:
            ten_pred = _G["ten_frozen"]
            ten_div = {"status": "FROZEN_DETERMINISTIC_10_10"}
        fused = fuse_heads(five_pred, ten_pred)
        fused = fused.loc[pd.to_datetime(fused["signal_date"]).dt.date.between(DEV_START, DEV_END)].copy()
        overlayed, overlay_meta = apply_b03(fused)
        pred_path = dest / "predictions.parquet"
        fused.to_parquet(pred_path, index=False)
        ran = discovery_backtest(overlayed, dest=None, write=False)
        obs = ran.get("observed") or {}
        years = year_stats(obs)
        nav = ran.get("nav")
        if nav is not None and not nav.empty:
            nav.to_parquet(dest / "nav.parquet", index=False)
        rec = {
            "trial_id": tid,
            "status": "SUCCESS",
            "sleeve": trial.get("sleeve"),
            "extras": trial.get("extras"),
            "five_extras": five_ex,
            "ten_extras": ten_ex,
            "feature_set_hash": trial.get("feature_set_hash"),
            "dev_sharpe": finite_or_none(obs.get("sharpe")),
            "dev_cagr": finite_or_none(obs.get("cagr")),
            "dev_maxdd": finite_or_none(obs.get("max_drawdown")),
            "dev_median_year": finite_or_none(years.get("median_year")),
            "dev_worst_year": finite_or_none(years.get("worst_year")),
            "dev_OOS_NDCG": float(np.mean(ndcgs)) if ndcgs else None,
            "dev_total_return": finite_or_none(obs.get("total_return")),
            "yearly_returns": years.get("years"),
            "n_pred_rows": int(len(fused)),
            "prediction_sha256": file_sha256(pred_path),
            "five_diversity": five_div,
            "ten_diversity": ten_div,
            "overlay_kind": (overlay_meta or {}).get("allocator") or (overlay_meta or {}).get("round9_kind"),
            "filled_metrics_with_zero": False,
        }
        dump_json(metrics_path, rec)
        return rec
    except Exception as exc:  # noqa: BLE001 — must record real failure, never fill 0
        rec = {
            "trial_id": tid,
            "status": _classify(exc),
            "sleeve": trial.get("sleeve"),
            "extras": trial.get("extras"),
            "feature_set_hash": trial.get("feature_set_hash"),
            **_empty_metrics(),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(limit=20),
            "filled_metrics_with_zero": False,
        }
        dump_json(metrics_path, rec)
        return rec


def _scan() -> list[dict[str, Any]]:
    root = ROUND17_ROOT / "search" / "trials"
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.glob("R17_T*/metrics.json")):
        rec = load_json(path)
        if rec:
            out.append(rec)
    return out


def run_search(
    *,
    resume: bool = True,
    jobs: int = 8,
    trial_id: str = "",
    trial_start: int = 0,
    trial_end: int = 0,
    dry_run: bool = False,
    **_: Any,
) -> dict[str, Any]:
    require_frozen()
    prepare_panels(resume=resume)
    trials = load_trials()
    if trial_id:
        trials = [t for t in trials if str(t["trial_id"]) == str(trial_id)]
    else:
        start = int(trial_start or 1)
        end = int(trial_end or N_TRIALS)
        trials = [t for t in trials if start <= int(str(t["trial_id"]).replace("R17_T", "")) <= end]
    planned = list(load_trials())
    if dry_run:
        payload = {"status": "DRY_RUN", "n_selected": len(trials), "n_planned": len(planned)}
        dump_json(SEARCH_JSON, payload)
        return payload
    done_ids = set()
    if resume:
        for rec in _scan():
            if rec.get("status") and rec.get("trial_id"):
                done_ids.add(str(rec["trial_id"]))
    todo = [t for t in trials if str(t["trial_id"]) not in done_ids]
    n_jobs = max(1, int(jobs))
    cpu = os.cpu_count() or 1
    if n_jobs * DISCOVERY_XGB_N_JOBS > cpu:
        n_jobs = max(1, cpu // max(DISCOVERY_XGB_N_JOBS, 1))
    if todo:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        _init_worker()
        if n_jobs == 1:
            for trial in todo:
                _run_trial(trial)
        else:
            import multiprocessing as mp
            from concurrent.futures.process import BrokenProcessPool

            ctx = mp.get_context("fork")
            chunk = 256
            done_n = 0
            remaining = list(todo)
            while remaining:
                batch = remaining[:chunk]
                remaining = remaining[chunk:]
                try:
                    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
                        futs = [pool.submit(_run_trial, trial) for trial in batch]
                        for fut in as_completed(futs):
                            try:
                                fut.result()
                            except Exception as exc:  # noqa: BLE001
                                print(f"[round17 search] future error: {type(exc).__name__}: {exc}", flush=True)
                            done_n += 1
                            if done_n % 16 == 0 or not remaining:
                                n_now = len(_scan())
                                print(
                                    f"[round17 search] COMPLETED/{N_TRIALS} = {n_now}/{N_TRIALS} "
                                    f"(this_run={done_n}/{len(todo)} resume_done={len(done_ids)})",
                                    flush=True,
                                )
                                dump_json(SEARCH_JSON, {
                                    "status": "PARTIAL" if n_now < N_TRIALS else "COMPLETE",
                                    "n_planned": N_TRIALS,
                                    "n_completed": n_now,
                                    "jobs": n_jobs,
                                    "xgb_n_jobs": DISCOVERY_XGB_N_JOBS,
                                    "silently_reduced": False,
                                    "resume_command": f"python tools/run_round17.py --stage search --resume --jobs {n_jobs}",
                                })
                                dump_json(ROUND17_ROOT / "_state.json", {
                                    "round": 17,
                                    "last_successful_stage": "search",
                                    "search": {
                                        "status": "PARTIAL" if n_now < N_TRIALS else "COMPLETE",
                                        "n_planned": N_TRIALS,
                                        "n_completed": n_now,
                                        "resume_command": f"python tools/run_round17.py --stage search --resume --jobs {n_jobs}",
                                    },
                                })
                except BrokenProcessPool as exc:
                    print(f"[round17 search] BrokenProcessPool: {exc}; continue leftover on disk resume", flush=True)
                    have = {str(r.get("trial_id")) for r in _scan() if r.get("trial_id")}
                    remaining = [t for t in todo if str(t["trial_id"]) not in have]
                    if n_jobs > 8:
                        n_jobs = max(8, n_jobs // 2)
                        print(f"[round17 search] reducing jobs to {n_jobs}", flush=True)
    scanned = _scan()
    success = [r for r in scanned if r.get("status") == "SUCCESS"]
    failed = [r for r in scanned if r.get("status") not in {"SUCCESS", "PLANNED", None}]
    payload = {
        "status": "COMPLETE" if len(scanned) >= N_TRIALS else "PARTIAL",
        "n_planned": N_TRIALS,
        "n_completed": len(scanned),
        "n_success": len(success),
        "n_failed": len(failed),
        "failed_by_status": pd.Series([r.get("status") for r in failed]).value_counts().to_dict() if failed else {},
        "jobs": n_jobs,
        "xgb_n_jobs": DISCOVERY_XGB_N_JOBS,
        "silently_reduced": False,
        "resume_command": (
            "python tools/run_round17.py --stage search --resume --jobs "
            f"{n_jobs}"
        ),
    }
    dump_json(SEARCH_JSON, payload)
    lines = [
        "# 06 Discovery 2018–2021",
        "",
        f"COMPLETED=`{payload['n_completed']}` / `{N_TRIALS}` SUCCESS=`{payload['n_success']}` FAILED=`{payload['n_failed']}`",
        f"status=`{payload['status']}`",
        "",
        "失败指标为 null，未填 0。",
    ]
    if payload["status"] != "COMPLETE":
        lines += ["", f"未完成。resume：`{payload['resume_command']}`", "不得声称 ROUND17 COMPLETE。"]
    write_md(SEARCH_MD, "\n".join(lines))
    return payload
