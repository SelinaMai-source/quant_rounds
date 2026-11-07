"""四路 A00/A01/B01/B02。A00 必须对齐 R9_B03。不重调 allocator。"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND17_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from .backtest import apply_b03, cached_overlay_v4, fuse_heads, run_canonical, year_stats
from .common import daily_from_nav, dump_json, finite_or_none, load_json, observed_from_report, write_md
from .contract import A00, A01, DISCOVERY_XGB_N_JOBS, HISTORICAL_RETROSPECTIVE_AUDIT, SEEDS, fourway_dir
from .freeze import load_trials
from .search import _G, _init_worker, prepare_panels
from .train import train_head_open

FOUR_JSON = ROUND17_ROOT / "fourway" / "fourway.json"
FOUR_MD = ROUND17_ROOT / "10_fourway.md"
ATTR_JSON = ROUND17_ROOT / "attribution" / "increments.json"
ATTR_MD = ROUND17_ROOT / "11_attribution.md"
DIV_JSON = ROUND17_ROOT / "diversity" / "finalist_diversity.json"
DIV_MD = ROUND17_ROOT / "12_diversity.md"


def _obs(ran: dict[str, Any]) -> dict[str, Any]:
    obs = ran.get("observed") or observed_from_report(ran.get("report") or {})
    years = year_stats(obs, years=tuple(range(2018, 2027)))
    return {
        "cagr": finite_or_none(obs.get("cagr")),
        "sharpe": finite_or_none(obs.get("sharpe")),
        "max_drawdown": finite_or_none(obs.get("max_drawdown")),
        "completed_trades": obs.get("completed_trades"),
        "total_return": finite_or_none(obs.get("total_return")),
        "yearly_returns": years.get("years") or obs.get("yearly_returns") or {},
        "median_year": years.get("median_year"),
        "worst_year": years.get("worst_year"),
        "status": ran.get("status"),
        "out_dir": ran.get("out_dir"),
    }


def _train_and_bt(code: str, five_ex: list[str], ten_ex: list[str], subsample: float, *, resume: bool) -> dict[str, Any]:
    dest = fourway_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "report.json"
    if resume and report_path.is_file() and (dest / "nav.parquet").is_file():
        return _obs({"status": "resumed", "report": load_json(report_path), "out_dir": str(dest),
                     "nav": __import__("pandas").read_parquet(dest / "nav.parquet")})
    ndcgs = []
    if five_ex or subsample < 1.0:
        feats = list(FIVE_FEATURES) + list(five_ex)
        pay = train_head_open(
            training=_G["five_training"], candidates=_G["five_candidate"], features=feats,
            target=FIVE_HEAD.target, label_end_column=FIVE_HEAD.label_end_column,
            rank_n_bins=FIVE_HEAD.rank_n_bins, eval_metric=FIVE_HEAD.eval_metric,
            subsample=subsample, colsample_bytree=subsample,
            first_prediction_year=2018, last_prediction_year=2026,
            output_dir=dest / "five", seeds=SEEDS, n_jobs=DISCOVERY_XGB_N_JOBS,
            save_models=False, resume=resume,
        )
        five_pred = pay["ensemble"]
        if pay.get("mean_fold_ndcg") is not None:
            ndcgs.append(float(pay["mean_fold_ndcg"]))
        five_div = pay.get("seed_diversity")
    else:
        five_pred = _G["five_frozen"]
        five_div = {"status": "FROZEN_10_10"}
    if ten_ex or subsample < 1.0:
        feats = list(TEN_FEATURES) + list(ten_ex)
        pay = train_head_open(
            training=_G["ten_training"], candidates=_G["ten_candidate"], features=feats,
            target=TEN_HEAD.target, label_end_column=TEN_HEAD.label_end_column,
            rank_n_bins=TEN_HEAD.rank_n_bins, eval_metric=TEN_HEAD.eval_metric,
            subsample=subsample, colsample_bytree=subsample,
            first_prediction_year=2018, last_prediction_year=2026,
            output_dir=dest / "ten", seeds=SEEDS, n_jobs=DISCOVERY_XGB_N_JOBS,
            save_models=False, resume=resume,
        )
        ten_pred = pay["ensemble"]
        if pay.get("mean_fold_ndcg") is not None:
            ndcgs.append(float(pay["mean_fold_ndcg"]))
        ten_div = pay.get("seed_diversity")
    else:
        ten_pred = _G["ten_frozen"]
        ten_div = {"status": "FROZEN_10_10"}
    fused = fuse_heads(five_pred, ten_pred)
    overlayed, meta = apply_b03(fused)
    fused.to_parquet(dest / "predictions.parquet", index=False)
    ran = run_canonical(
        dest, overlayed,
        extra={"code": code, "subsample": subsample, "five_extras": five_ex, "ten_extras": ten_ex,
               "fee_schedule": "HISTORICAL_FEE_CORRECTED"},
        resume=resume, write=True,
    )
    obs = _obs(ran)
    obs["ndcg"] = float(np.mean(ndcgs)) if ndcgs else None
    obs["five_diversity"] = five_div
    obs["ten_diversity"] = ten_div
    obs["code"] = code
    dump_json(dest / "fourway_summary.json", obs)
    return obs


def _fourway_job(spec: tuple[Any, ...]) -> tuple[str, dict[str, Any]]:
    code, five_ex, ten_ex, subsample, resume = spec
    print(f"[round17 fourway] start {code} subsample={subsample}", flush=True)
    obs = _train_and_bt(code, list(five_ex), list(ten_ex), float(subsample), resume=bool(resume))
    print(f"[round17 fourway] done {code} sharpe={obs.get('sharpe')}", flush=True)
    return str(code), obs


def run_fourway(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    prepare_panels(resume=True)
    _init_worker()
    top12 = load_json(ROUND17_ROOT / "select" / "top12.json")
    ids = list(top12.get("ids") or [])
    trials = {t["trial_id"]: t for t in load_trials()}
    a00_dir = ROUND17_ROOT / "parity" / A00
    a00 = _obs({"status": "parity", "report": load_json(a00_dir / "report.json"), "out_dir": str(a00_dir)})
    a00["code"] = A00
    specs: list[tuple[Any, ...]] = [(A01, [], [], 0.8, resume)]
    for i, tid in enumerate(ids, start=1):
        trial = trials[tid]
        five_ex = list(trial.get("five_extras") or [])
        ten_ex = list(trial.get("ten_extras") or [])
        specs.append((f"R17_F{i:02d}_B01", five_ex, ten_ex, 1.0, resume))
        specs.append((f"R17_F{i:02d}_B02", five_ex, ten_ex, 0.8, resume))
    n_jobs = max(1, min(int(jobs), os.cpu_count() or 1, len(specs)))
    results: dict[str, dict[str, Any]] = {}
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if n_jobs == 1:
        for spec in specs:
            code, obs = _fourway_job(spec)
            results[code] = obs
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
            futs = [pool.submit(_fourway_job, spec) for spec in specs]
            for fut in as_completed(futs):
                code, obs = fut.result()
                results[code] = obs
    a01 = results[A01]
    finalists = []
    for i, tid in enumerate(ids, start=1):
        trial = trials[tid]
        b01_id = f"R17_F{i:02d}_B01"
        b02_id = f"R17_F{i:02d}_B02"
        finalists.append({
            "rank": i,
            "source_trial": tid,
            "sleeve": trial.get("sleeve"),
            "extras": trial.get("extras"),
            "B01": results[b01_id],
            "B02": results[b02_id],
            "B01_id": b01_id,
            "B02_id": b02_id,
        })
    payload = {"A00": a00, "A01": a01, "finalists": finalists, "n_finalists": len(finalists)}
    dump_json(FOUR_JSON, payload)
    dump_json(ROUND17_ROOT / "validate" / "retrospective_audit.json", {
        "label": HISTORICAL_RETROSPECTIVE_AUDIT,
        "not_untouched_oos": True,
        "window": ["2018-01-01", "2026-07-24"],
        "top12_ids_unchanged": ids,
        "rows": [
            {
                "source_trial": item["source_trial"],
                "B02_id": item["B02_id"],
                "cagr": (item.get("B02") or {}).get("cagr"),
                "sharpe": (item.get("B02") or {}).get("sharpe"),
                "max_drawdown": (item.get("B02") or {}).get("max_drawdown"),
                "yearly_returns": (item.get("B02") or {}).get("yearly_returns"),
            }
            for item in finalists
        ],
    })
    write_md(FOUR_MD, "\n".join([
        "# 10 四路对照 A00/A01/B01/B02",
        "",
        f"A00 Sharpe=`{a00.get('sharpe')}` A01 Sharpe=`{a01.get('sharpe')}` n_finalists=`{len(finalists)}`",
        "这些是 5000 搜索的后代，不是 24 个独立 OOS 假设。不重调 B03 allocator。",
        "期权 overlay 未并入。",
    ]))
    _increments(payload)
    _finalist_diversity(payload)
    return payload


def _delta(a: dict[str, Any] | None, b: dict[str, Any] | None, key: str) -> float | None:
    if not a or not b:
        return None
    av, bv = a.get(key), b.get(key)
    if av is None or bv is None:
        return None
    return float(bv) - float(av)


def _increments(four: dict[str, Any]) -> dict[str, Any]:
    a00, a01 = four["A00"], four["A01"]
    rows = []
    for item in four.get("finalists") or []:
        b01, b02 = item["B01"], item["B02"]
        rows.append({
            "source_trial": item["source_trial"],
            "B01_id": item["B01_id"],
            "B02_id": item["B02_id"],
            "factor_cagr": _delta(a00, b01, "cagr"),
            "factor_sharpe": _delta(a00, b01, "sharpe"),
            "ensemble_cagr": _delta(a00, a01, "cagr"),
            "ensemble_sharpe": _delta(a00, a01, "sharpe"),
            "joint_cagr": _delta(a00, b02, "cagr"),
            "joint_sharpe": _delta(a00, b02, "sharpe"),
            "interaction_cagr": (
                None if None in (_delta(a00, b02, "cagr"), _delta(a00, b01, "cagr"), _delta(a00, a01, "cagr"))
                else _delta(a00, b02, "cagr") - _delta(a00, b01, "cagr") - _delta(a00, a01, "cagr")
            ),
            "interaction_sharpe": (
                None if None in (_delta(a00, b02, "sharpe"), _delta(a00, b01, "sharpe"), _delta(a00, a01, "sharpe"))
                else _delta(a00, b02, "sharpe") - _delta(a00, b01, "sharpe") - _delta(a00, a01, "sharpe")
            ),
        })
    payload = {
        "definition": {
            "Factor": "B01-A00",
            "Ensemble": "A01-A00",
            "Joint": "B02-A00",
            "Interaction": "B02-B01-A01+A00",
        },
        "a01_luck_not_promotion": True,
        "rows": rows,
    }
    dump_json(ATTR_JSON, payload)
    write_md(ATTR_MD, "\n".join([
        "# 11 归因增量",
        "",
        "Factor=B01−A00；Ensemble=A01−A00；Joint=B02−A00；Interaction=B02−B01−A01+A00。",
        "A01 运气不是晋升理由（R6_A04）。不重调 allocator。",
        "",
        "| trial | factor_sharpe | ensemble_sharpe | joint_sharpe | interaction_sharpe |",
        "|---|---|---|---|---|",
        *[
            f"| `{r['source_trial']}` | {r['factor_sharpe']} | {r['ensemble_sharpe']} | {r['joint_sharpe']} | {r['interaction_sharpe']} |"
            for r in rows
        ],
    ]))
    return payload


def _concentration(yearly: dict[str, Any]) -> str | None:
    if not yearly:
        return None
    items = []
    for y, v in yearly.items():
        try:
            items.append((str(y), float(v)))
        except (TypeError, ValueError):
            continue
    if not items:
        return None
    total = sum(max(v, 0.0) for _, v in items)
    late = sum(max(v, 0.0) for y, v in items if y in {"2024", "2025", "2026"})
    if total > 0 and late / total >= 0.80:
        return "PERFORMANCE_CONCENTRATION_RISK"
    return None


def _finalist_diversity(four: dict[str, Any]) -> dict[str, Any]:
    recs = []
    for item in four.get("finalists") or []:
        recs.append({
            "id": item["B02_id"],
            "five": (item["B02"] or {}).get("five_diversity"),
            "ten": (item["B02"] or {}).get("ten_diversity"),
            "concentration": _concentration((item["B02"] or {}).get("yearly_returns") or {}),
        })
    payload = {
        "leave_one_seed_out": "diagnostic_only_not_used_to_select_four_seeds",
        "finalists": recs,
        "a01": four.get("A01", {}).get("five_diversity"),
    }
    dump_json(DIV_JSON, payload)
    write_md(DIV_MD, "\n".join([
        "# 12 Finalist seed diversity",
        "",
        "leave-one-seed-out 只诊断，不选四 seed。",
        str(recs[:5]),
    ]))
    return payload
