"""Native seed 审计 + group-date 80% 扰动。退化则禁止用 100 个相同 native seed 假装完成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..paths import PREDICTION_START, ROUND20_ROOT, SIGNAL_END, assert_project_result_path
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from ..round17.ensemble import assert_seed_diversity, ensemble_rank_pct
from ..round17.train import complete_cases, ranker_config_from_params
from ..stock_readonly import ensure_stock_src_on_path, sha256_file
from .common import dump_json, finite_or_none, load_json, slim_obs, write_md
from .contract import (
    COLSAMPLE,
    GROUP_DATE_SPEC,
    NATIVE_SEED_DIVERSITY_AUDIT,
    NOT_TRIGGERED,
    SEED_LADDER,
    SUBSAMPLE,
    V5_ANCHOR,
    XGB_N_JOBS,
    exp_dir,
)
from .engine import run_v5_fused_backtest
from .freeze import require_frozen
from .train import new_columns, slim_panel, unique_keep

SEED_JSON = ROUND20_ROOT / "seed" / "seed_robustness.json"
SEED_MD = ROUND20_ROOT / "10_seed_robustness.md"
MASK_ROOT = ROUND20_ROOT / "seed" / "date_masks"
KEYS = ["trade_date", "instrument_id"]
SEED_OFFSET = 10_000


def sample_dates_stratified(dates: pd.Series, *, frac: float, rng: np.random.Generator) -> pd.DatetimeIndex:
    d = pd.to_datetime(pd.Series(dates))
    d = d.dt.normalize().drop_duplicates().sort_values()
    keep: list[Any] = []
    for _, part in d.groupby(d.dt.year, sort=True):
        n = int(len(part))
        k = max(1, int(round(n * float(frac))))
        k = min(k, n)
        chosen = rng.choice(part.to_numpy(), size=k, replace=False)
        keep.extend(list(chosen))
    return pd.DatetimeIndex(pd.to_datetime(keep)).normalize().unique().sort_values()


def build_keep_by_year(
    training: pd.DataFrame,
    *,
    label_end_column: str,
    date_seed: int,
    first_prediction_year: int = 2018,
    last_prediction_year: int = 2026,
) -> dict[str, list[str]]:
    """Walk-forward 各折从 baseline 14/13 完整样本抽 80% 交易日。同一 date_seed 跨候选共用。"""
    work = training.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"]).dt.normalize()
    work[label_end_column] = pd.to_datetime(work[label_end_column])
    rng = np.random.default_rng(int(date_seed))
    out: dict[str, list[str]] = {}
    for predict_year in range(int(first_prediction_year), int(last_prediction_year) + 1):
        train_start, train_end = predict_year - 3, predict_year - 1
        train = work.loc[
            work["trade_date"].dt.year.between(train_start, train_end)
            & work[label_end_column].dt.year.le(train_end)
        ].copy()
        query_sizes = train.groupby("trade_date")["instrument_id"].transform("size")
        train = train.loc[query_sizes.ge(2)]
        keep_idx = sample_dates_stratified(train["trade_date"], frac=GROUP_DATE_SPEC["frac"], rng=rng)
        out[str(predict_year)] = [str(pd.Timestamp(x).date()) for x in keep_idx]
    return out


def ensure_keep_by_year(head: str, training: pd.DataFrame, *, label_end_column: str, date_seed: int) -> dict[str, list[str]]:
    MASK_ROOT.mkdir(parents=True, exist_ok=True)
    path = MASK_ROOT / f"{head}_gd{int(date_seed)}.json"
    if path.is_file():
        return load_json(path)
    payload = build_keep_by_year(training, label_end_column=label_end_column, date_seed=int(date_seed))
    dump_json(path, payload)
    return payload


def _rolling_group_date(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    label_end_column: str,
    rank_n_bins: int,
    eval_metric: str,
    date_seed: int,
    n_jobs: int,
    first_prediction_year: int = 2018,
    last_prediction_year: int = 2026,
    keep_by_year: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_stock_src_on_path()
    from first_board_hermes.baseline import daily_relevance, mean_daily_ndcg
    from xgboost import XGBRanker

    from ..round17.contract import model_parameters

    features = list(features)
    training = training.copy()
    training["trade_date"] = pd.to_datetime(training["trade_date"])
    training[label_end_column] = pd.to_datetime(training[label_end_column])
    candidates = candidates.copy()
    candidates["trade_date"] = pd.to_datetime(candidates["trade_date"])
    params = model_parameters(17, subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE, n_jobs=n_jobs)
    params["eval_metric"] = eval_metric
    params["rank_n_bins"] = rank_n_bins
    config = ranker_config_from_params(params)
    from dataclasses import asdict

    raw = asdict(config)
    keep = {
        "objective", "eval_metric", "ndcg_exp_gain", "n_estimators", "learning_rate", "max_depth",
        "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha",
        "tree_method", "random_state", "n_jobs",
    }
    fit_params = {k: v for k, v in raw.items() if k in keep}
    try:
        metric_k = int(str(config.eval_metric).rsplit("@", 1)[1])
    except (IndexError, ValueError):
        metric_k = 1
    preds = []
    folds = []
    kept_by_year: dict[str, int] = {}
    for predict_year in range(int(first_prediction_year), int(last_prediction_year) + 1):
        train_start, train_end = predict_year - 3, predict_year - 1
        train_mask = (
            training["trade_date"].dt.year.between(train_start, train_end)
            & training[label_end_column].dt.year.le(train_end)
        )
        train = training.loc[train_mask].copy()
        query_sizes = train.groupby("trade_date")["instrument_id"].transform("size")
        train = train.loc[query_sizes.ge(2)].sort_values(KEYS).reset_index(drop=True)
        train["trade_date"] = pd.to_datetime(train["trade_date"]).dt.normalize()
        if keep_by_year and str(predict_year) in keep_by_year:
            keep_idx = pd.to_datetime(keep_by_year[str(predict_year)])
        else:
            rng = np.random.default_rng(int(date_seed) + int(predict_year))
            keep_idx = sample_dates_stratified(train["trade_date"], frac=GROUP_DATE_SPEC["frac"], rng=rng)
        train = train.loc[train["trade_date"].isin(set(keep_idx))].sort_values(KEYS).reset_index(drop=True)
        query_sizes = train.groupby("trade_date")["instrument_id"].transform("size")
        train = train.loc[query_sizes.ge(2)].sort_values(KEYS).reset_index(drop=True)
        test = candidates.loc[candidates["trade_date"].dt.year.eq(predict_year)].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"empty group-date fold {train_start}-{train_end} -> {predict_year}")
        relevance = daily_relevance(train, target=target, n_bins=rank_n_bins)
        model = XGBRanker(**fit_params)
        model.fit(
            train[features].to_numpy(dtype=np.float32),
            relevance.to_numpy(dtype=np.int32),
            qid=train["trade_date"].factorize(sort=True)[0],
            verbose=False,
        )
        score = model.predict(test[features].to_numpy(dtype=np.float32))
        part = test[[*KEYS, target]].copy().rename(columns={"trade_date": "signal_date"})
        part["score"] = score
        part["predict_year"] = predict_year
        preds.append(part[["signal_date", "instrument_id", "score", "predict_year"]].copy())
        evaluable = part[target].notna()
        ndcg = None
        if int(evaluable.sum()):
            ev = part.loc[evaluable].copy()
            ev["trade_date"] = ev["signal_date"]
            ev["relevance"] = daily_relevance(ev, target=target, n_bins=rank_n_bins)
            ndcg = mean_daily_ndcg(ev, k=metric_k)
        folds.append({
            "predict_year": predict_year,
            "train_days_kept": int(train["trade_date"].nunique()),
            "mean_daily_ndcg": ndcg,
        })
        kept_by_year[str(predict_year)] = int(train["trade_date"].nunique())
    prediction = pd.concat(preds, ignore_index=True)
    prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
    prediction = prediction.loc[prediction["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)].copy()
    meta = {
        "model": "group_date_80pct",
        "frac": GROUP_DATE_SPEC["frac"],
        "replace": False,
        "stratify_by_year": True,
        "whole_day_query": True,
        "date_seed": int(date_seed),
        "xgb_random_state": 17,
        "paired_shared_mask": bool(keep_by_year),
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE,
        "folds": folds,
        "kept_by_year": kept_by_year,
    }
    return prediction, meta


def train_head_group_date(
    *,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    features: Sequence[str],
    target: str,
    label_end_column: str,
    rank_n_bins: int,
    eval_metric: str,
    output_dir: Path,
    n_seeds: int,
    resume: bool = True,
    n_jobs: int = 1,
    seed_offset: int = SEED_OFFSET,
    keep_by_seed: dict[int, dict[str, list[str]]] | None = None,
) -> dict[str, Any]:
    training_clean, _ = complete_cases(training, features, require_target=True, target=target)
    candidates_clean, _ = complete_cases(candidates, features, require_target=False, target=target)
    frames = []
    reports = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(output_dir)
    for i in range(int(n_seeds)):
        date_seed = int(seed_offset) + i
        seed_dir = output_dir / f"gd{date_seed}"
        seed_path = seed_dir / "predictions.parquet"
        if resume and seed_path.is_file():
            pred = pd.read_parquet(seed_path)
            pred["signal_date"] = pd.to_datetime(pred["signal_date"])
            frames.append(pred)
            reports.append({"date_seed": date_seed, "rows": int(len(pred)), "resumed": True})
            continue
        keep_by_year = None if not keep_by_seed else keep_by_seed.get(int(date_seed))
        pred, meta = _rolling_group_date(
            training_clean,
            candidates_clean,
            features=features,
            target=target,
            label_end_column=label_end_column,
            rank_n_bins=rank_n_bins,
            eval_metric=eval_metric,
            date_seed=date_seed,
            n_jobs=n_jobs,
            keep_by_year=keep_by_year,
        )
        seed_dir.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(seed_path, index=False)
        meta["prediction_sha256"] = sha256_file(seed_path)
        dump_json(seed_dir / "training_report.json", meta)
        frames.append(pred)
        reports.append({"date_seed": date_seed, "rows": int(len(pred)), "resumed": False, "kept_by_year": meta.get("kept_by_year")})
    diversity = assert_seed_diversity(frames) if len(frames) >= 2 else {"status": "lt2", "degenerate": None}
    ensemble = ensemble_rank_pct(frames)
    ens_path = output_dir / "predictions.parquet"
    ensemble.to_parquet(ens_path, index=False)
    payload = {
        "status": "trained_group_date",
        "n_seeds": int(n_seeds),
        "diversity": diversity,
        "seed_reports": reports,
        "rows": int(len(ensemble)),
        "ensemble": ensemble,
        "seed_frames": frames,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE,
    }
    dump_json(output_dir / "ensemble_report.json", {k: v for k, v in payload.items() if k not in {"ensemble", "seed_frames"}})
    return payload


def _native_audit() -> dict[str, Any]:
    """Read existing 1.0/1.0 seed artifacts. Do not invent 100 identical native models."""
    rows = []
    root = ROUND20_ROOT / "candidates"
    degenerate = 0
    checked = 0
    for path in sorted(root.glob("R20_*/summary.json")):
        payload = load_json(path)
        for head in ("five", "ten"):
            div = ((payload.get(head) or {}).get("seed_diversity") or {})
            if not div:
                continue
            checked += 1
            deg = bool(div.get("degenerate") or str(div.get("status") or "").endswith("DEGENERATE"))
            if deg:
                degenerate += 1
            rows.append({
                "code": payload.get("code") or path.parent.name,
                "head": head,
                "status": div.get("status"),
                "degenerate": deg,
                "mean_pairwise_corr": div.get("mean_pairwise_corr"),
                "top1_mean_agreement": div.get("top1_mean_agreement"),
                "unique_prediction_hashes": div.get("unique_prediction_hashes"),
            })
        if checked >= 20:
            break
    status = "DEGENERATE" if degenerate and degenerate == checked else ("MIXED" if degenerate else "DIVERSE")
    return {
        "audit": NATIVE_SEED_DIVERSITY_AUDIT,
        "status": status,
        "n_checked_heads": checked,
        "n_degenerate": degenerate,
        "fake_100_native_identical_models": False,
        "note": "subsample=colsample=1.0 → native seed rank-invariant. Must switch to group-date 80%. Do not run 100 native seeds.",
        "examples": rows[:8],
    }


def _specs_for_seed() -> list[dict[str, Any]]:
    from .combo import REG_CSV
    from .model import MODEL_JSON
    from .runner import COMBO_TRAIN_JSON, parse_extras

    combo = load_json(COMBO_TRAIN_JSON)
    model = load_json(MODEL_JSON)
    registry = pd.read_csv(REG_CSV) if REG_CSV.is_file() else pd.DataFrame()
    extras_of = {}
    for rec in registry.to_dict(orient="records") if len(registry) else []:
        extras_of[str(rec["code"])] = parse_extras(rec.get("extras"))
    codes: list[str] = ["R20_A01"]
    best = (combo.get("best") or {}).get("code")
    if best:
        codes.append(str(best))
    for f in model.get("finalists") or []:
        c = f.get("code")
        if c:
            codes.append(str(c))
    for c in ("R20_C01", "R20_C02", "R20_D01"):
        codes.append(c)
    # unique keep order
    seen = set()
    out = []
    for code in codes:
        if code in seen:
            continue
        seen.add(code)
        extras = extras_of.get(code)
        if extras is None and code.startswith("R20_") and code[4] in "SPTXR":
            extras = extras_of.get(code, ())
        if code == "R20_A01":
            extras = ()
            force = True
        elif code == "R20_D01":
            from .combo import REG_JSON
            extras = tuple(load_json(REG_JSON).get("selected_new_factors") or ())
            force = False
        else:
            # alias Cxx
            matrix = load_json(ROUND20_ROOT / "configs" / "model_matrix.json")
            for spec in matrix.get("specs") or []:
                if spec.get("code") == code:
                    extras = parse_extras(spec.get("extras"))
                    break
            force = False
        out.append({"code": code, "extras": tuple(extras or ()), "force_retrain": bool(code == "R20_A01")})
    return out[:8]


def _baseline_keep_maps(n_seeds: int, *, seed_offset: int = SEED_OFFSET) -> tuple[dict[int, dict[str, list[str]]], dict[int, dict[str, list[str]]]]:
    from .runner import _G

    five_base = slim_panel(_G["five_training"], [*KEYS, *list(FIVE_FEATURES), FIVE_HEAD.target, FIVE_HEAD.label_end_column])
    ten_base = slim_panel(_G["ten_training"], [*KEYS, *list(TEN_FEATURES), TEN_HEAD.target, TEN_HEAD.label_end_column])
    five_base, _ = complete_cases(five_base, list(FIVE_FEATURES), require_target=True, target=FIVE_HEAD.target)
    ten_base, _ = complete_cases(ten_base, list(TEN_FEATURES), require_target=True, target=TEN_HEAD.target)
    five_map: dict[int, dict[str, list[str]]] = {}
    ten_map: dict[int, dict[str, list[str]]] = {}
    for i in range(int(n_seeds)):
        date_seed = int(seed_offset) + i
        five_map[date_seed] = ensure_keep_by_year("five", five_base, label_end_column=FIVE_HEAD.label_end_column, date_seed=date_seed)
        ten_map[date_seed] = ensure_keep_by_year("ten", ten_base, label_end_column=TEN_HEAD.label_end_column, date_seed=date_seed)
    return five_map, ten_map


def _per_seed_backtests(
    dest: Path,
    five_frames: list[pd.DataFrame],
    ten_frames: list[pd.DataFrame],
    *,
    code: str,
    n_seeds: int,
    resume: bool,
    seed_offset: int = SEED_OFFSET,
) -> list[dict[str, Any]]:
    rows = []
    n = min(len(five_frames), len(ten_frames), int(n_seeds))
    for i in range(n):
        date_seed = int(seed_offset) + i
        pdir = dest / "per_seed" / f"{code}_gd{date_seed}"
        summary_path = pdir / "summary.json"
        if resume and summary_path.is_file() and (pdir / "nav.parquet").is_file():
            obs = load_json(summary_path)
            rows.append({"date_seed": date_seed, "sharpe": obs.get("sharpe"), "cagr": obs.get("cagr"), "resumed": True})
            continue
        extra = {
            "code": f"{code}_gd{date_seed}",
            "role": "group_date_per_seed",
            "date_seed": date_seed,
            "ensemble": "single-seed then 90/10",
        }
        obs = run_v5_fused_backtest(pdir, five_frames[i], ten_frames[i], extra=extra, resume=resume)
        dump_json(summary_path, slim_obs(obs))
        rows.append({"date_seed": date_seed, "sharpe": obs.get("sharpe"), "cagr": obs.get("cagr"), "resumed": False})
    return rows


def _run_group_date_code(spec: dict[str, Any], n_seeds: int, *, resume: bool, n_jobs: int) -> dict[str, Any]:
    from .runner import init_panels

    init_panels()
    from .runner import _G

    code = spec["code"]
    extras = list(spec.get("extras") or ())
    dest = ROUND20_ROOT / "seed" / code / f"n{n_seeds}"
    dest.mkdir(parents=True, exist_ok=True)
    summary_path = dest / "summary.json"
    if resume and summary_path.is_file() and (dest / "nav.parquet").is_file():
        payload = load_json(summary_path)
        payload["resumed"] = True
        return payload
    five_features = unique_keep(list(FIVE_FEATURES) + extras)
    ten_features = unique_keep(list(TEN_FEATURES) + extras)
    five_keep, ten_keep = _baseline_keep_maps(n_seeds)
    five = train_head_group_date(
        training=slim_panel(_G["five_training"], [*KEYS, *five_features, FIVE_HEAD.target, FIVE_HEAD.label_end_column]),
        candidates=slim_panel(_G["five_candidate"], [*KEYS, *five_features, FIVE_HEAD.target]),
        features=five_features,
        target=FIVE_HEAD.target,
        label_end_column=FIVE_HEAD.label_end_column,
        rank_n_bins=FIVE_HEAD.rank_n_bins,
        eval_metric=FIVE_HEAD.eval_metric,
        output_dir=dest / "five",
        n_seeds=n_seeds,
        resume=resume,
        n_jobs=n_jobs,
        keep_by_seed=five_keep,
    )
    ten = train_head_group_date(
        training=slim_panel(_G["ten_training"], [*KEYS, *ten_features, TEN_HEAD.target, TEN_HEAD.label_end_column]),
        candidates=slim_panel(_G["ten_candidate"], [*KEYS, *ten_features, TEN_HEAD.target]),
        features=ten_features,
        target=TEN_HEAD.target,
        label_end_column=TEN_HEAD.label_end_column,
        rank_n_bins=TEN_HEAD.rank_n_bins,
        eval_metric=TEN_HEAD.eval_metric,
        output_dir=dest / "ten",
        n_seeds=n_seeds,
        resume=resume,
        n_jobs=n_jobs,
        keep_by_seed=ten_keep,
    )
    extra = {
        "code": code,
        "role": f"group_date_n{n_seeds}",
        "n_seeds": n_seeds,
        "ensemble": "mean(seed_rank_pct) then 90/10",
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
    }
    obs = run_v5_fused_backtest(dest, five["ensemble"], ten["ensemble"], extra=extra, resume=resume)
    per_seed = _per_seed_backtests(
        dest,
        list(five.get("seed_frames") or []),
        list(ten.get("seed_frames") or []),
        code=code,
        n_seeds=n_seeds,
        resume=resume,
    )
    v5_sh = float(V5_ANCHOR["sharpe"])
    sh = finite_or_none(obs.get("sharpe"))
    pair_pos = None if sh is None else float(sh > v5_sh)
    per_seed_sharpes = [finite_or_none(r.get("sharpe")) for r in per_seed]
    payload = {
        **slim_obs(obs),
        "code": code,
        "n_seeds": n_seeds,
        "status": obs.get("status") or "ran",
        "cagr": obs.get("cagr"),
        "sharpe": obs.get("sharpe"),
        "max_drawdown": obs.get("max_drawdown"),
        "nav_end": obs.get("nav_end"),
        "completed_trades": obs.get("completed_trades"),
        "five_diversity": five.get("diversity"),
        "ten_diversity": ten.get("diversity"),
        "paired_sharpe_gt_v5_ensemble": pair_pos,
        "per_seed_sharpes": per_seed_sharpes,
        "per_seed": per_seed,
        "v5_sharpe": v5_sh,
        "ensemble": "mean(seed_rank_pct)+90/10+B03+O01+R7_5bp",
        "shared_date_mask": True,
        "out_dir": str(dest),
        "extras": extras,
    }
    dump_json(summary_path, payload)
    return payload


def _paired_from_rows(rows: list[dict[str, Any]], *, v5_row: dict[str, Any] | None = None) -> dict[str, Any]:
    v5 = float(V5_ANCHOR["sharpe"])
    cand = [r for r in rows if r.get("code") not in {"R20_A00", "R20_A01"} and r.get("sharpe") is not None]
    pos = [r for r in cand if float(r["sharpe"]) > v5]
    mean_sh = float(np.mean([float(r["sharpe"]) for r in cand])) if cand else None
    locked = {
        "n": len(cand),
        "n_sharpe_gt_v5": len(pos),
        "pos_ratio": (len(pos) / len(cand) if cand else None),
        "mean_sharpe": mean_sh,
        "v5_sharpe": v5,
        "note": "ensemble vs locked v5 Sharpe; exploratory, not OOS p.",
    }
    v5_seeds = list((v5_row or {}).get("per_seed_sharpes") or [])
    pairs = []
    for r in cand:
        cs = list(r.get("per_seed_sharpes") or [])
        n = min(len(cs), len(v5_seeds))
        for i in range(n):
            a = finite_or_none(cs[i])
            b = finite_or_none(v5_seeds[i])
            if a is None or b is None:
                continue
            pairs.append({"code": r.get("code"), "i": i, "cand": a, "v5": b, "delta": float(a) - float(b)})
    n_pos = sum(1 for p in pairs if p["delta"] > 0)
    same = {
        "n": len(pairs),
        "n_sharpe_gt_v5": n_pos,
        "pos_ratio": (n_pos / len(pairs) if pairs else None),
        "mean_delta": (float(np.mean([p["delta"] for p in pairs])) if pairs else None),
        "note": "same group-date 80% subset (baseline 14/13 mask) candidate vs A01; exploratory, not OOS p.",
    }
    return {
        "n": same["n"] or locked["n"],
        "n_sharpe_gt_v5": same["n_sharpe_gt_v5"] if pairs else locked["n_sharpe_gt_v5"],
        "pos_ratio": same["pos_ratio"] if pairs else locked["pos_ratio"],
        "mean_sharpe": mean_sh,
        "v5_sharpe": v5,
        "locked_v5_ensemble": locked,
        "same_date_seed_vs_A01": same,
        "note": "exploratory paired Δ; not independent OOS p-value. Do not write 80/100 as OOS p.",
    }


def run_seed(*, resume: bool = True, jobs: int = 4, **_: Any) -> dict[str, Any]:
    require_frozen()
    from .runner import prepare_panels

    prepare_panels(resume=True)
    audit = _native_audit()
    specs = _specs_for_seed()
    ladder = dict(SEED_LADDER)
    s1_rows: list[dict[str, Any]] = []
    xgb_jobs = 1
    if audit.get("status") != "DEGENERATE" and audit.get("n_degenerate") == 0:
        # still document; plan says if degenerate use group-date
        pass
    print(f"[round20/seed] native audit={audit.get('status')} n_deg={audit.get('n_degenerate')}", flush=True)
    print("[round20/seed] native 100 identical models FORBIDDEN; switching to group-date 80%", flush=True)
    n1 = int(ladder["S1_native"])
    for spec in specs:
        print(f"[round20/seed] S1 group-date n={n1} {spec['code']}", flush=True)
        try:
            s1_rows.append(_run_group_date_code(spec, n1, resume=resume, n_jobs=xgb_jobs))
        except Exception as exc:
            s1_rows.append({"code": spec["code"], "status": "SKIP_ERROR", "reason": f"{type(exc).__name__}: {exc}", "n_seeds": n1})
    v5_row = next((r for r in s1_rows if r.get("code") == "R20_A01"), None)
    paired_s1 = _paired_from_rows(s1_rows, v5_row=v5_row)
    # advance at most 6 by mean/ensemble sharpe; A01 is the paired baseline, not a new alpha
    ranked = sorted(
        [r for r in s1_rows if r.get("sharpe") is not None and r.get("code") not in {"R20_A00", "R20_A01"}],
        key=lambda r: -float(r["sharpe"]),
    )
    v5_sh = float(V5_ANCHOR["sharpe"])
    s2_pool = [r for r in ranked if float(r["sharpe"]) >= v5_sh - 0.15][: int(ladder["S2_max_advance"])]
    s2_rows: list[dict[str, Any]] = []
    s2_status = NOT_TRIGGERED
    if s2_pool:
        s2_status = "RAN"
        n2 = int(ladder["S2_group_date"])
        for spec0 in s2_pool:
            spec = next(s for s in specs if s["code"] == spec0["code"])
            print(f"[round20/seed] S2 group-date n={n2} {spec['code']}", flush=True)
            try:
                s2_rows.append(_run_group_date_code(spec, n2, resume=resume, n_jobs=xgb_jobs))
            except Exception as exc:
                s2_rows.append({"code": spec["code"], "status": "SKIP_ERROR", "reason": f"{type(exc).__name__}: {exc}", "n_seeds": n2})
    else:
        s2_status = NOT_TRIGGERED
    paired_s2 = _paired_from_rows(s2_rows, v5_row=v5_row) if s2_rows else {"status": s2_status}
    s3_rows: list[dict[str, Any]] = []
    s3_status = NOT_TRIGGERED
    if s2_rows:
        ranked2 = sorted([r for r in s2_rows if r.get("sharpe") is not None], key=lambda r: -float(r["sharpe"]))
        s3_pool = ranked2[: int(ladder["S3_max_advance"])]
        n3 = int(ladder["S3_group_date"])
        s3_status = "RAN"
        for spec0 in s3_pool:
            spec = next(s for s in specs if s["code"] == spec0["code"])
            print(f"[round20/seed] S3 group-date n={n3} {spec['code']}", flush=True)
            try:
                s3_rows.append(_run_group_date_code(spec, n3, resume=resume, n_jobs=xgb_jobs))
            except Exception as exc:
                s3_rows.append({"code": spec["code"], "status": "SKIP_ERROR", "reason": f"{type(exc).__name__}: {exc}", "n_seeds": n3})
    paired_s3 = _paired_from_rows(s3_rows, v5_row=v5_row) if s3_rows else {"status": s3_status}
    s4_status = NOT_TRIGGERED
    s4_reason = "not in pre-registered uncertain zone"
    if s3_rows and s2_rows:
        pos = paired_s3.get("pos_ratio")
        m2 = paired_s2.get("mean_sharpe")
        m3 = paired_s3.get("mean_sharpe")
        trigger_pos = pos is not None and float(ladder["S4_trigger_pos_ratio_lo"]) <= float(pos) <= float(ladder["S4_trigger_pos_ratio_hi"])
        trigger_d = m2 is not None and m3 is not None and abs(float(m3) - float(m2)) > float(ladder["S4_trigger_abs_dmean_sharpe"])
        if trigger_pos or trigger_d:
            s4_status = "TRIGGERED_BUT_BUDGET"
            s4_reason = f"trigger pos={pos} dmean={None if m2 is None or m3 is None else abs(m3-m2)}; 300 seeds not silently replaced by 10"
            # Run 300 only if triggered. This is expensive; still run because plan requires it when triggered.
            n4 = int(ladder["S4_group_date"])
            s4_status = "RAN"
            s4_rows = []
            for spec0 in s3_rows[:2]:
                spec = next(s for s in specs if s["code"] == spec0.get("code"))
                print(f"[round20/seed] S4 group-date n={n4} {spec['code']}", flush=True)
                s4_rows.append(_run_group_date_code(spec, n4, resume=resume, n_jobs=xgb_jobs))
        else:
            s4_status = NOT_TRIGGERED
            s4_reason = f"pos_ratio={pos} |Δmean Sharpe S2→S3|={None if m2 is None or m3 is None else abs(float(m3)-float(m2))}"
    payload = {
        "status": "DONE",
        "native_audit": audit,
        "group_date_spec": dict(GROUP_DATE_SPEC),
        "s1": {"n": n1, "rows": [{k: r.get(k) for k in ("code", "status", "sharpe", "cagr", "max_drawdown", "nav_end", "reason")} for r in s1_rows], "paired": paired_s1},
        "s2": {"n": int(ladder["S2_group_date"]), "status": s2_status, "rows": [{k: r.get(k) for k in ("code", "status", "sharpe", "cagr", "max_drawdown", "nav_end", "reason")} for r in s2_rows], "paired": paired_s2},
        "s3": {"n": int(ladder["S3_group_date"]), "status": s3_status, "rows": [{k: r.get(k) for k in ("code", "status", "sharpe", "cagr", "max_drawdown", "nav_end", "reason")} for r in s3_rows], "paired": paired_s3},
        "s4": {"n": int(ladder["S4_group_date"]), "status": s4_status, "reason": s4_reason},
        "fake_10_seeds_as_s4": False,
        "fake_100_native": False,
        "voting_threshold": "DIAGNOSTIC_ONLY",
        "ensemble": "mean(seed_rank_pct) then 90/10",
    }
    dump_json(SEED_JSON, payload)
    lines = [
        "# 10 Seed 稳健性",
        "",
        f"Native audit=`{audit.get('status')}` degenerate_heads=`{audit.get('n_degenerate')}/{audit.get('n_checked_heads')}`。",
        "**禁止**把 100 个 native seed 写成 100 个不同模型。改 group-date 80% 不放回、按年分层、整日 query。subsample=colsample=1.0。",
        f"S1 n=5 group-date paired pos_ratio=`{(paired_s1 or {}).get('pos_ratio')}` mean Sharpe=`{(paired_s1 or {}).get('mean_sharpe')}`",
        f"S2 n=30 status=`{s2_status}` S3 n=100 status=`{s3_status}` S4 n=300 status=`{s4_status}` `{s4_reason}`",
        "配对 Δ 是探索性门槛，不是独立 OOS p 值。投票门槛仅 DIAGNOSTIC_ONLY。",
        "",
    ]
    write_md(SEED_MD, "\n".join(lines))
    return payload


__all__ = ["SEED_JSON", "SEED_MD", "run_seed"]
