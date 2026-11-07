"""走步训练。复用 Round17 的 3 年/1 年切分与 XGBRanker 参数。subsample=colsample=1。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .paths import PREDICTION_START, SIGNAL_END, assert_project_result_path
from .research_io import dump_json
from .round17.train import model_parameters, ranker_config_from_params
from .round22_3.contract import COLSAMPLE, FIVE_HEAD, SEEDS, SUBSAMPLE, TEN_HEAD
from .round22_3.train import complete_cases_keep_extras
from .round7.train import ensemble_rank_pct
from .stock_readonly import sha256_file

KEYS = ["trade_date", "instrument_id"]
Mutator = Callable[[pd.DataFrame, pd.DataFrame, int, int], tuple[pd.DataFrame, pd.DataFrame]]


def _head(name: str):
    if name == "five":
        return FIVE_HEAD
    if name == "ten":
        return TEN_HEAD
    raise KeyError(name)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    return work


def train_ranker_head(
    *,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    features: Sequence[str],
    extras: Sequence[str],
    head_name: str,
    output_dir: Path,
    seeds: Sequence[int] = SEEDS,
    n_jobs: int = 2,
    resume: bool = True,
    mutator: Mutator | None = None,
    first_prediction_year: int = 2018,
    last_prediction_year: int = 2026,
    target: str | None = None,
    label_end_column: str | None = None,
) -> dict[str, Any]:
    from .round17.ensemble import assert_seed_diversity

    if abs(float(SUBSAMPLE) - 1.0) > 1e-12 or abs(float(COLSAMPLE) - 1.0) > 1e-12:
        raise RuntimeError("XGB subsample/colsample must stay 1.0")
    head = _head(head_name)
    target_name = target or head.target
    label_end_name = label_end_column or head.label_end_column
    features = list(features)
    extras = [c for c in extras if c in features]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(output_dir)
    training_clean, train_audit = complete_cases_keep_extras(
        _prepare(training), features, extras, require_target=True, target=target_name
    )
    candidates_clean, cand_audit = complete_cases_keep_extras(
        _prepare(candidates), features, extras, require_target=False, target=target_name
    )
    if label_end_name in training_clean.columns:
        training_clean[label_end_name] = pd.to_datetime(training_clean[label_end_name])
    frames = []
    seed_reports = []
    for seed in seeds:
        seed_dir = output_dir / f"seed{int(seed)}"
        seed_path = seed_dir / "predictions.parquet"
        if resume and seed_path.is_file() and (seed_dir / "training_report.json").is_file():
            prediction = pd.read_parquet(seed_path)
            prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
            frames.append(prediction)
            seed_reports.append({"seed": int(seed), "resumed": True, "rows": int(len(prediction))})
            continue
        prediction, meta = _roll(
            training_clean,
            candidates_clean,
            features=features,
            head=head,
            seed=int(seed),
            n_jobs=int(n_jobs),
            mutator=mutator,
            first_prediction_year=first_prediction_year,
            last_prediction_year=last_prediction_year,
            output_dir=seed_dir,
            target=target_name,
            label_end_column=label_end_name,
        )
        frames.append(prediction)
        seed_reports.append({"seed": int(seed), "resumed": False, "rows": int(len(prediction)), "folds": meta.get("folds")})
    diversity = assert_seed_diversity(frames) if len(frames) >= 2 else {"status": "SINGLE_SEED"}
    ensemble = ensemble_rank_pct(frames) if len(frames) >= 2 else frames[0][["signal_date", "instrument_id", "score"]]
    ensemble["signal_date"] = pd.to_datetime(ensemble["signal_date"])
    ensemble = ensemble.loc[ensemble["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)].copy()
    ens_path = output_dir / "predictions.parquet"
    ensemble.to_parquet(ens_path, index=False)
    median = _median_rank(frames)
    median_path = output_dir / "predictions_median.parquet"
    median.to_parquet(median_path, index=False)
    payload = {
        "status": "trained_oos",
        "head": head_name,
        "features": features,
        "extras": extras,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "seeds": [int(s) for s in seeds],
        "seed_diversity": diversity,
        "rows": int(len(ensemble)),
        "prediction_sha256": sha256_file(ens_path),
        "median_sha256": sha256_file(median_path),
        "filled_with_zero": False,
        "train_clean_audit": train_audit,
        "candidate_clean_audit": cand_audit,
        "seed_reports": seed_reports,
        "target": target_name,
    }
    (output_dir / "ensemble_report.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "seed_reports"}, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    dump_json(output_dir / "train_meta.json", payload)
    return payload


def _median_rank(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = None
    cols = []
    for index, frame in enumerate(frames):
        column = f"r{index}"
        part = frame[["signal_date", "instrument_id", "score"]].copy()
        part["signal_date"] = pd.to_datetime(part["signal_date"])
        part[column] = part.groupby("signal_date")["score"].rank(method="average", pct=True)
        part = part[["signal_date", "instrument_id", column]]
        merged = part if merged is None else merged.merge(part, on=["signal_date", "instrument_id"], how="inner")
        cols.append(column)
    if merged is None:
        raise RuntimeError("no frames for median")
    merged["score"] = merged[cols].median(axis=1)
    return merged[["signal_date", "instrument_id", "score"]].sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort"
    ).reset_index(drop=True)


def _roll(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    features: list[str],
    head,
    seed: int,
    n_jobs: int,
    mutator: Mutator | None,
    first_prediction_year: int,
    last_prediction_year: int,
    output_dir: Path,
    target: str,
    label_end_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from dataclasses import asdict

    from xgboost import XGBRanker

    from .round17.train import ensure_stock_src_on_path

    ensure_stock_src_on_path()
    from first_board_hermes.baseline import daily_relevance, mean_daily_ndcg

    params = model_parameters(seed, subsample=1.0, colsample_bytree=1.0, n_jobs=n_jobs)
    params["eval_metric"] = head.eval_metric
    params["rank_n_bins"] = head.rank_n_bins
    config = ranker_config_from_params(params)
    raw = asdict(config)
    rank_n_bins = int(raw.pop("rank_n_bins"))
    for key in (
        "absolute_return_thresholds",
        "hybrid_loss_floor",
        "hybrid_tail_threshold",
        "precomputed_relevance",
        "early_stopping_rounds",
        "validation_sessions",
    ):
        raw.pop(key, None)
    if abs(float(raw.get("subsample", 1.0)) - 1.0) > 1e-12 or abs(float(raw.get("colsample_bytree", 1.0)) - 1.0) > 1e-12:
        raise RuntimeError("refusing subsample/colsample below 1")
    label_end = label_end_column
    target_name = target
    predictions = []
    folds = []
    for predict_year in range(int(first_prediction_year), int(last_prediction_year) + 1):
        train_start, train_end = predict_year - 3, predict_year - 1
        train_mask = training["trade_date"].dt.year.between(train_start, train_end) & training[label_end].dt.year.le(train_end)
        train = training.loc[train_mask].copy()
        query_sizes = train.groupby("trade_date")["instrument_id"].transform("size")
        train = train.loc[query_sizes.ge(2)].sort_values(KEYS).reset_index(drop=True)
        test = candidates.loc[candidates["trade_date"].dt.year.eq(predict_year)].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"empty rolling fold {train_start}-{train_end} -> {predict_year}")
        if mutator is not None:
            train, test = mutator(train, test, predict_year, seed)
        relevance = daily_relevance(train, target=target_name, n_bins=rank_n_bins)
        model = XGBRanker(**raw)
        model.fit(
            train[features].to_numpy(dtype=np.float32),
            relevance.to_numpy(dtype=np.int32),
            qid=train["trade_date"].factorize(sort=True)[0],
            verbose=False,
        )
        score = model.predict(test[features].to_numpy(dtype=np.float32))
        prediction = test[["trade_date", "instrument_id"]].copy()
        prediction = prediction.rename(columns={"trade_date": "signal_date"})
        prediction["score"] = score
        prediction["predict_year"] = predict_year
        folds.append(
            {
                "predict_year": predict_year,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "extra_test_finite": {
                    col: float(pd.to_numeric(test[col], errors="coerce").notna().mean())
                    for col in features
                    if col.startswith("r22_") or col.startswith("r24_")
                },
            }
        )
        predictions.append(prediction)
        print(f"[fold] head={head.name} seed={seed} year={predict_year} train={len(train)} test={len(test)}", flush=True)
    result = pd.concat(predictions, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_dir / "predictions.parquet", index=False)
    meta = {
        "status": "trained_oos",
        "seed": seed,
        "subsample": raw.get("subsample"),
        "colsample_bytree": raw.get("colsample_bytree"),
        "folds": folds,
        "features": features,
    }
    (output_dir / "training_report.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result, meta
