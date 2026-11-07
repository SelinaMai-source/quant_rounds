"""冻结公式选 Top100 / Top12。2022–2023 是历史验证切片，不是独立 OOS。"""

from __future__ import annotations

import os
import traceback
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND17_ROOT
from .backtest import apply_b03, cached_overlay_v4, fuse_heads, run_canonical, year_stats
from .common import clip_nav_window, dump_json, finite_or_none, load_json, metrics_from_nav, write_md, yearly_from_nav
from .contract import (
    DEV_END,
    DEV_START,
    HISTORICAL_RETROSPECTIVE_AUDIT,
    MAXDD_WORSEN_CAP,
    N_TRIALS,
    SELECTION_WEIGHTS,
    VAL_END,
    VAL_START,
)
from .freeze import load_trials, require_frozen
from .search import SEARCH_JSON, _scan, prepare_panels

TOP100_JSON = ROUND17_ROOT / "select" / "top100.json"
TOP100_CSV = ROUND17_ROOT / "select" / "top100.csv"
TOP100_MD = ROUND17_ROOT / "07_top100.md"
VAL_JSON = ROUND17_ROOT / "validate" / "validate_2022_2023.json"
VAL_STATUS = ROUND17_ROOT / "validate" / "validate_status.json"
VAL_MD = ROUND17_ROOT / "08_historical_validation_2022_2023.md"
TOP12_JSON = ROUND17_ROOT / "select" / "top12.json"
TOP12_MD = ROUND17_ROOT / "09_top12.md"
AUDIT_JSON = ROUND17_ROOT / "validate" / "retrospective_audit.json"


def _rankpct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(method="average", pct=True)


def a00_discovery_metrics() -> dict[str, Any]:
    dest = ROUND17_ROOT / "parity" / "R17_A00"
    nav_path = dest / "nav.parquet"
    if not nav_path.is_file():
        ov = cached_overlay_v4()
        ran = run_canonical(dest, ov, extra={"code": "R17_A00_disc_slice"}, write=True, resume=False)
        nav = ran["nav"]
        if nav is None or getattr(nav, "empty", True):
            raise RuntimeError("R17_A00 nav missing after replay; cannot score discovery MaxDD gate")
    else:
        nav = pd.read_parquet(nav_path)
    clipped = clip_nav_window(nav, str(DEV_START), str(DEV_END))
    m = metrics_from_nav(clipped)
    years = yearly_from_nav(clipped)
    vals = [float(v) for v in years.values() if v is not None]
    return {
        "dev_sharpe": finite_or_none(m.get("sharpe_zero_rate")),
        "dev_cagr": finite_or_none(m.get("annualized_return")),
        "dev_maxdd": finite_or_none(m.get("max_drawdown")),
        "dev_median_year": float(pd.Series(vals).median()) if vals else None,
        "dev_worst_year": float(min(vals)) if vals else None,
        "yearly_returns": years,
        "window": [str(DEV_START), str(DEV_END)],
    }


def run_select(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and TOP100_JSON.is_file() and TOP12_JSON.is_file() is False:
        existing = load_json(TOP100_JSON)
        if existing.get("n_top100"):
            return existing
    search = load_json(SEARCH_JSON)
    if int(search.get("n_completed") or 0) < N_TRIALS:
        raise RuntimeError(
            f"search incomplete {search.get('n_completed')}/5000; resume: {search.get('resume_command')}"
        )
    rows = _scan()
    a00 = a00_discovery_metrics()
    frame = pd.DataFrame(rows)
    for col in ("dev_sharpe", "dev_median_year", "dev_worst_year", "dev_OOS_NDCG", "dev_maxdd"):
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    valid = frame.loc[frame["status"].eq("SUCCESS")].copy()
    a00_dd = a00.get("dev_maxdd")
    if a00_dd is not None:
        floor = float(a00_dd) - float(MAXDD_WORSEN_CAP)
        valid["maxdd_gate"] = valid["dev_maxdd"] >= floor
    else:
        valid["maxdd_gate"] = True
    scored = valid.loc[
        valid[["dev_sharpe", "dev_median_year", "dev_worst_year", "dev_OOS_NDCG"]].notna().all(axis=1)
        & valid["maxdd_gate"]
    ].copy()
    if scored.empty:
        raise RuntimeError("no valid discovery trials for selection")
    scored["rp_sharpe"] = _rankpct(scored["dev_sharpe"])
    scored["rp_med"] = _rankpct(scored["dev_median_year"])
    scored["rp_worst"] = _rankpct(scored["dev_worst_year"])
    scored["rp_ndcg"] = _rankpct(scored["dev_OOS_NDCG"])
    w = SELECTION_WEIGHTS
    scored["selection_score"] = (
        w["dev_sharpe"] * scored["rp_sharpe"]
        + w["dev_median_year"] * scored["rp_med"]
        + w["dev_worst_year"] * scored["rp_worst"]
        + w["dev_OOS_NDCG"] * scored["rp_ndcg"]
    )
    scored = scored.sort_values(["selection_score", "trial_id"], ascending=[False, True], kind="mergesort")
    top100 = scored.head(100).copy()
    top100["rank"] = np.arange(1, len(top100) + 1)
    keep = [
        "rank", "trial_id", "sleeve", "selection_score", "dev_sharpe", "dev_cagr", "dev_maxdd",
        "dev_median_year", "dev_worst_year", "dev_OOS_NDCG", "feature_set_hash", "extras",
    ]
    for c in keep:
        if c not in top100.columns:
            top100[c] = None
    top100[keep].to_csv(TOP100_CSV, index=False)
    payload = {
        "n_scanned": len(frame),
        "n_success": int((frame["status"] == "SUCCESS").sum()),
        "n_scored": int(len(scored)),
        "n_top100": int(len(top100)),
        "a00_discovery": a00,
        "maxdd_floor": float(a00_dd) - MAXDD_WORSEN_CAP if a00_dd is not None else None,
        "formula": SELECTION_WEIGHTS,
        "tie_break": "trial_id ascending",
        "window": "2018-2021 only",
        "top100": top100[keep].to_dict(orient="records"),
        "all_5000_remain_in_bias_accounting": True,
    }
    dump_json(TOP100_JSON, payload)
    lines = [
        "# 07 Top100（仅 2018–2021 发现窗）",
        "",
        f"scored=`{len(scored)}` top100=`{len(top100)}` A00 dev Sharpe=`{a00.get('dev_sharpe')}` MaxDD=`{a00.get('dev_maxdd')}`",
        "2022+ 未用于本排名。",
        "",
        "| rank | trial_id | sleeve | score | sharpe | med_year | worst_year | ndcg |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rec in payload["top100"][:20]:
        lines.append(
            f"| {rec['rank']} | `{rec['trial_id']}` | {rec['sleeve']} | {rec['selection_score']:.4f} | "
            f"{rec['dev_sharpe']} | {rec['dev_median_year']} | {rec['dev_worst_year']} | {rec['dev_OOS_NDCG']} |"
        )
    lines.append("")
    lines.append("仅列前 20；全表见 select/top100.csv。")
    write_md(TOP100_MD, "\n".join(lines))
    return payload


def _ndcg_for_years(pay: dict[str, Any], years: set[int]) -> float | None:
    vals: list[float] = []
    for report in pay.get("seed_reports") or []:
        for fold in report.get("folds") or []:
            try:
                y = int(fold.get("predict_year"))
            except (TypeError, ValueError):
                continue
            if y not in years:
                continue
            for key, value in fold.items():
                if not str(key).startswith("mean_daily_ndcg"):
                    continue
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    vals.append(v)
    return float(np.mean(vals)) if vals else None


def _validate_one(trial: dict[str, Any], resume: bool = True) -> dict[str, Any]:
    from ..round7.contract import FIVE_HEAD, TEN_FEATURES, TEN_HEAD
    from .contract import DISCOVERY_XGB_N_JOBS, FIVE_FEATURES, SEEDS
    from .search import _G
    from .train import train_head_open

    tid = str(trial["trial_id"])
    dest = ROUND17_ROOT / "validate" / tid
    dest.mkdir(parents=True, exist_ok=True)
    mpath = dest / "metrics.json"
    if resume and mpath.is_file():
        existing = load_json(mpath)
        if existing.get("status") in {
            "SUCCESS", "TRAINING_FAILED", "BACKTEST_FAILED", "INVALID_FACTOR", "DATA_UNAVAILABLE",
        }:
            return existing
    try:
        five_ex = list(trial.get("five_extras") or [])
        ten_ex = list(trial.get("ten_extras") or [])
        ndcgs = []
        if five_ex:
            pay = train_head_open(
                training=_G["five_training"], candidates=_G["five_candidate"],
                features=list(FIVE_FEATURES) + five_ex,
                target=FIVE_HEAD.target, label_end_column=FIVE_HEAD.label_end_column,
                rank_n_bins=FIVE_HEAD.rank_n_bins, eval_metric=FIVE_HEAD.eval_metric,
                subsample=0.8, colsample_bytree=0.8,
                first_prediction_year=2018, last_prediction_year=2023,
                output_dir=None, seeds=SEEDS, n_jobs=DISCOVERY_XGB_N_JOBS, save_models=False, resume=False,
            )
            five_pred = pay["ensemble"]
            v = _ndcg_for_years(pay, {2022, 2023})
            if v is not None:
                ndcgs.append(v)
        else:
            five_pred = _G["five_frozen"]
        if ten_ex:
            pay = train_head_open(
                training=_G["ten_training"], candidates=_G["ten_candidate"],
                features=list(TEN_FEATURES) + ten_ex,
                target=TEN_HEAD.target, label_end_column=TEN_HEAD.label_end_column,
                rank_n_bins=TEN_HEAD.rank_n_bins, eval_metric=TEN_HEAD.eval_metric,
                subsample=0.8, colsample_bytree=0.8,
                first_prediction_year=2018, last_prediction_year=2023,
                output_dir=None, seeds=SEEDS, n_jobs=DISCOVERY_XGB_N_JOBS, save_models=False, resume=False,
            )
            ten_pred = pay["ensemble"]
            v = _ndcg_for_years(pay, {2022, 2023})
            if v is not None:
                ndcgs.append(v)
        else:
            ten_pred = _G["ten_frozen"]
        fused = fuse_heads(five_pred, ten_pred)
        fused = fused.loc[pd.to_datetime(fused["signal_date"]).dt.date.between(VAL_START, VAL_END)].copy()
        overlayed, _ = apply_b03(fused)
        ran = run_canonical(
            dest, overlayed, extra={"role": "hist_val_2022_2023"}, write=True, resume=False,
            start_date=VAL_START, end_date=date(2024, 3, 1),
            metric_start=VAL_START, metric_end=VAL_END,
        )
        obs = ran.get("observed") or {}
        years = year_stats(obs, years=(2022, 2023))
        row = {
            "trial_id": tid,
            "status": "SUCCESS",
            "val_sharpe": finite_or_none(obs.get("sharpe")),
            "val_cagr": finite_or_none(obs.get("cagr")),
            "val_maxdd": finite_or_none(obs.get("max_drawdown")),
            "val_median_year": finite_or_none(years.get("median_year")),
            "val_worst_year": finite_or_none(years.get("worst_year")),
            "val_OOS_NDCG": float(np.mean(ndcgs)) if ndcgs else None,
            "not_independent_oos": True,
            "label": "HISTORICAL_VALIDATION_SLICE",
            "filled_metrics_with_zero": False,
        }
    except Exception as exc:  # noqa: BLE001
        row = {
            "trial_id": tid, "status": "TRAINING_FAILED",
            "val_sharpe": None, "val_cagr": None, "val_maxdd": None,
            "val_median_year": None, "val_worst_year": None, "val_OOS_NDCG": None,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(limit=20),
            "filled_metrics_with_zero": False,
        }
    dump_json(mpath, row)
    return row


def run_validate(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    """Retrain Top100 for 2022–2023 historical validation, freeze Top12. 不看 2024–2026。"""
    from .search import _init_worker

    top = load_json(TOP100_JSON)
    if not top.get("top100"):
        top = run_select(resume=resume)
    require_frozen()
    prepare_panels(resume=True)
    trials_by_id = {t["trial_id"]: t for t in load_trials()}
    ids = [r["trial_id"] for r in top["top100"]]
    if resume and VAL_JSON.is_file() and TOP12_JSON.is_file():
        existing = load_json(VAL_JSON)
        if existing.get("n") == len(ids) and existing.get("top12_frozen"):
            return existing
    done: dict[str, dict[str, Any]] = {}
    todo: list[dict[str, Any]] = []
    for rec in top["top100"]:
        tid = rec["trial_id"]
        mpath = ROUND17_ROOT / "validate" / tid / "metrics.json"
        if resume and mpath.is_file():
            existing = load_json(mpath)
            if existing.get("status") in {
                "SUCCESS", "TRAINING_FAILED", "BACKTEST_FAILED", "INVALID_FACTOR", "DATA_UNAVAILABLE",
            }:
                done[tid] = existing
                continue
        trial = trials_by_id.get(tid)
        if trial is None:
            raise RuntimeError(f"trial {tid} missing from frozen manifest")
        todo.append(trial)
    n_jobs = max(1, int(jobs))
    cpu = os.cpu_count() or 1
    if n_jobs > cpu:
        n_jobs = cpu
    dump_json(VAL_STATUS, {
        "status": "PARTIAL",
        "n_top100": len(ids),
        "n_done": len(done),
        "n_todo": len(todo),
        "resume_command": f"python tools/run_round17.py --stage validate --resume --jobs {n_jobs}",
    })
    if todo:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        _init_worker()
        print(f"[round17 validate] todo={len(todo)} resumed={len(done)} jobs={n_jobs}", flush=True)
        if n_jobs == 1:
            for trial in todo:
                row = _validate_one(trial, resume=resume)
                done[row["trial_id"]] = row
                if len(done) % 5 == 0 or len(done) == len(ids):
                    print(f"[round17 validate] {len(done)}/{len(ids)}", flush=True)
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from concurrent.futures.process import BrokenProcessPool

            ctx = mp.get_context("fork")
            remaining = list(todo)
            while remaining:
                batch = remaining
                remaining = []
                try:
                    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
                        futs = {pool.submit(_validate_one, trial, resume): trial for trial in batch}
                        for fut in as_completed(futs):
                            trial = futs[fut]
                            try:
                                row = fut.result()
                                done[row["trial_id"]] = row
                            except Exception as exc:  # noqa: BLE001
                                print(
                                    f"[round17 validate] {trial.get('trial_id')} {type(exc).__name__}: {exc}",
                                    flush=True,
                                )
                                remaining.append(trial)
                            if len(done) % 5 == 0 or len(done) == len(ids):
                                print(f"[round17 validate] {len(done)}/{len(ids)}", flush=True)
                                dump_json(VAL_STATUS, {
                                    "status": "PARTIAL",
                                    "n_top100": len(ids),
                                    "n_done": len(done),
                                    "resume_command": (
                                        "python tools/run_round17.py --stage validate --resume "
                                        f"--jobs {n_jobs}"
                                    ),
                                })
                except BrokenProcessPool as exc:
                    print(f"[round17 validate] BrokenProcessPool: {exc}", flush=True)
                    have = set(done)
                    remaining = [t for t in batch if str(t["trial_id"]) not in have] + remaining
                    if n_jobs > 4:
                        n_jobs = max(4, n_jobs // 2)
    val_rows = [done[tid] for tid in ids if tid in done]
    if len(val_rows) < len(ids):
        missing = [tid for tid in ids if tid not in done]
        dump_json(VAL_STATUS, {
            "status": "PARTIAL",
            "n_top100": len(ids),
            "n_done": len(done),
            "missing": missing,
            "resume_command": f"python tools/run_round17.py --stage validate --resume --jobs {n_jobs}",
        })
        raise RuntimeError(
            f"validate incomplete {len(done)}/{len(ids)}. "
            f"resume: python tools/run_round17.py --stage validate --resume --jobs {n_jobs}"
        )
    vframe = pd.DataFrame(val_rows)
    ok = vframe.loc[vframe["status"].eq("SUCCESS")].copy()
    for col in ("val_sharpe", "val_median_year", "val_worst_year", "val_OOS_NDCG"):
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    ok = ok.dropna(subset=["val_sharpe", "val_median_year", "val_worst_year", "val_OOS_NDCG"])
    if ok.empty:
        raise RuntimeError("no valid 2022-2023 trials for Top12; metrics not filled with 0")
    ok["rp_sharpe"] = _rankpct(ok["val_sharpe"])
    ok["rp_med"] = _rankpct(ok["val_median_year"])
    ok["rp_worst"] = _rankpct(ok["val_worst_year"])
    ok["rp_ndcg"] = _rankpct(ok["val_OOS_NDCG"])
    w = SELECTION_WEIGHTS
    ok["val_selection_score"] = (
        w["dev_sharpe"] * ok["rp_sharpe"] + w["dev_median_year"] * ok["rp_med"]
        + w["dev_worst_year"] * ok["rp_worst"] + w["dev_OOS_NDCG"] * ok["rp_ndcg"]
    )
    ok = ok.sort_values(["val_selection_score", "trial_id"], ascending=[False, True], kind="mergesort")
    top12 = ok.head(12).copy()
    top12["rank"] = np.arange(1, len(top12) + 1)
    top12_recs = top12[["rank", "trial_id", "val_selection_score", "val_sharpe", "val_cagr", "val_maxdd", "val_median_year", "val_worst_year", "val_OOS_NDCG"]].to_dict(orient="records")
    dump_json(TOP12_JSON, {
        "frozen_before_2024_2026": True,
        "n": len(top12_recs),
        "ids": [r["trial_id"] for r in top12_recs],
        "rows": top12_recs,
        "label": "TOP12_FROZEN",
    })
    dump_json(VAL_JSON, {
        "n": len(val_rows),
        "n_success": int((vframe["status"] == "SUCCESS").sum()),
        "not_independent_oos": True,
        "window": ["2022-01-01", "2023-12-31"],
        "top12_frozen": True,
        "rows": val_rows,
    })
    write_md(VAL_MD, "\n".join([
        "# 08 2022–2023 历史验证",
        "",
        "这是历史验证切片，**不是**独立 OOS。Top12 在查看 2024–2026 之前锁死。",
        f"n=`{len(val_rows)}` success=`{int((vframe['status']=='SUCCESS').sum())}`",
    ]))
    write_md(TOP12_MD, "\n".join([
        "# 09 Top12 冻结",
        "",
        "锁死于查看 2024–2026 之前。之后不改 ID。",
        "",
        "| rank | trial_id | val_score | sharpe |",
        "|---|---|---|---|",
        *[f"| {r['rank']} | `{r['trial_id']}` | {r['val_selection_score']:.4f} | {r['val_sharpe']} |" for r in top12_recs],
        "",
        f"full-history label = `{HISTORICAL_RETROSPECTIVE_AUDIT}` 不是 UNTOUCHED_OOS。",
    ]))
    dump_json(VAL_STATUS, {
        "status": "COMPLETE",
        "n_top100": len(ids),
        "n_done": len(val_rows),
        "n_top12": len(top12_recs),
        "top12_frozen_before_2024_2026": True,
        "resume_command": f"python tools/run_round17.py --stage validate --resume --jobs {n_jobs}",
    })
    dump_json(AUDIT_JSON, {
        "label": HISTORICAL_RETROSPECTIVE_AUDIT,
        "top12_ids_frozen": [r["trial_id"] for r in top12_recs],
        "looked_at_2024_2026": False,
        "note": "IDs locked before full-history fourway. Metrics filled in fourway, IDs never rewritten.",
    })
    return load_json(TOP12_JSON)
