"""Round17 折法包装：复制 Stock rolling_train_predict 日历年逻辑，允许课题侧新列。

不改 Stock src。未知列不走 ALL_FEATURES 拒绝。缺失不填 0，complete-case 删除。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..paths import PREDICTION_START, SIGNAL_END, assert_project_result_path
from ..stock_readonly import ensure_stock_src_on_path, sha256_file
from .contract import DISCOVERY_XGB_N_JOBS, SEEDS, model_parameters

KEYS = ["trade_date", "instrument_id"]


def complete_cases(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    require_target: bool,
    target: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    needed = [*features, *([target] if require_target else [])]
    missing = sorted({*KEYS, *needed} - set(frame.columns))
    if missing:
        raise ValueError(f"INVALID_FACTOR panel missing required columns: {missing}")
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work[list(features)] = work[list(features)].replace([np.inf, -np.inf], np.nan)
    mask = work[needed].notna().all(axis=1)
    audit = {
        "rows_before": len(work),
        "rows_dropped_in_memory": int((~mask).sum()),
        "rows_after": int(mask.sum()),
        "require_target": require_target,
        "feature_missing_rows": {name: int(work[name].isna().sum()) for name in features},
        "filled_with_zero": False,
    }
    clean = work.loc[mask].sort_values(KEYS).reset_index(drop=True)
    return clean, audit


def rolling_train_predict_open(
    training_panel: pd.DataFrame,
    candidate_panel: pd.DataFrame,
    output_dir: Path | None,
    *,
    features: Sequence[str],
    first_prediction_year: int,
    last_prediction_year: int,
    config,
    target: str,
    label_end_column: str,
    save_models: bool = False,
    pre_cleaned: bool = False,
    clean_audit: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exact 3-calendar-year train → next year predict. Allows columns outside ALL_FEATURES."""
    ensure_stock_src_on_path()
    from first_board_hermes.baseline import daily_relevance, mean_daily_ndcg
    from xgboost import XGBRanker

    features = list(features)
    if not features:
        raise ValueError("invalid feature subset; empty")
    if pre_cleaned:
        training = training_panel
        candidates = candidate_panel
        train_clean_audit, candidate_clean_audit = clean_audit or ({}, {})
    else:
        training, train_clean_audit = complete_cases(
            training_panel, features, require_target=True, target=target
        )
        candidates, candidate_clean_audit = complete_cases(
            candidate_panel, features, require_target=False, target=target
        )
    if label_end_column not in training:
        raise ValueError(f"training panel missing label end column {label_end_column}")
    training[label_end_column] = pd.to_datetime(training[label_end_column])
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        assert_project_result_path(output_dir)
        model_dir = output_dir / "models"
        if save_models:
            model_dir.mkdir(exist_ok=True)
    else:
        model_dir = None

    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    params = asdict(config)
    raw_rank_n_bins = params.pop("rank_n_bins")
    params.pop("absolute_return_thresholds")
    params.pop("hybrid_loss_floor")
    params.pop("hybrid_tail_threshold")
    params.pop("precomputed_relevance")
    params.pop("early_stopping_rounds")
    params.pop("validation_sessions")
    rank_n_bins = None if raw_rank_n_bins is None else int(raw_rank_n_bins)
    try:
        metric_k = int(str(config.eval_metric).rsplit("@", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError("eval_metric must have the form ndcg@K") from error

    for predict_year in range(int(first_prediction_year), int(last_prediction_year) + 1):
        train_start, train_end = predict_year - 3, predict_year - 1
        train_mask = (
            training["trade_date"].dt.year.between(train_start, train_end)
            & training[label_end_column].dt.year.le(train_end)
        )
        train = training.loc[train_mask].copy()
        query_sizes = train.groupby("trade_date")["instrument_id"].transform("size")
        train = train.loc[query_sizes.ge(2)].sort_values(KEYS).reset_index(drop=True)
        test = candidates.loc[candidates["trade_date"].dt.year.eq(predict_year)].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"empty rolling fold {train_start}-{train_end} -> {predict_year}")
        relevance = daily_relevance(train, target=target, n_bins=rank_n_bins)
        model = XGBRanker(**params)
        model.fit(
            train[features].to_numpy(dtype=np.float32),
            relevance.to_numpy(dtype=np.int32),
            qid=train["trade_date"].factorize(sort=True)[0],
            verbose=False,
        )
        score = model.predict(test[features].to_numpy(dtype=np.float32))
        prediction = test[["trade_date", "instrument_id", target]].copy()
        prediction = prediction.rename(columns={"trade_date": "signal_date"})
        prediction["score"] = score
        prediction["predict_year"] = predict_year
        evaluable = prediction[target].notna()
        eval_frame = prediction.loc[evaluable].copy()
        eval_frame["trade_date"] = eval_frame["signal_date"]
        eval_frame["relevance"] = daily_relevance(eval_frame, target=target, n_bins=rank_n_bins)
        folds.append({
            "predict_year": predict_year,
            "train_years": [train_start, train_end],
            "train_rows": int(len(train)),
            "train_days": int(train["trade_date"].nunique()),
            "candidate_rows_scored": int(len(prediction)),
            "evaluation_rows": int(evaluable.sum()),
            f"mean_daily_ndcg_at_{metric_k}": mean_daily_ndcg(eval_frame, k=metric_k),
        })
        if save_models and model_dir is not None:
            model.save_model(model_dir / f"xgbranker_{train_start}_{train_end}_to_{predict_year}.json")
        predictions.append(prediction.drop(columns=[target]))

    result = pd.concat(predictions, ignore_index=True).sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    if result.duplicated(["signal_date", "instrument_id"]).any():
        raise RuntimeError("duplicate out-of-sample predictions")
    metadata = {
        "status": "trained_oos",
        "walk_forward": "exact prior 3 calendar years train, next 1 calendar year predict",
        "target": target,
        "label_end_column": label_end_column,
        "features": features,
        "feature_contract_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
        "all_features_bypass": True,
        "filled_with_zero": False,
        "ranker": params | {"rank_n_bins": rank_n_bins},
        "models_persisted": bool(save_models),
        "training_clean_audit": train_clean_audit,
        "candidate_clean_audit": candidate_clean_audit,
        "folds": folds,
        "prediction_rows": int(len(result)),
    }
    if output_dir is not None:
        path = output_dir / "predictions.parquet"
        result.to_parquet(path, index=False)
        metadata["prediction_sha256"] = sha256_file(path)
        (output_dir / "training_report.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return result, metadata


def ranker_config_from_params(params: dict[str, Any]):
    ensure_stock_src_on_path()
    from first_board_hermes.baseline import RankerConfig

    allowed = {item.name for item in fields(RankerConfig)}
    return RankerConfig(**{key: value for key, value in params.items() if key in allowed})


def train_head_open(
    *,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    features: Sequence[str],
    target: str,
    label_end_column: str,
    rank_n_bins: int,
    eval_metric: str,
    subsample: float,
    colsample_bytree: float,
    first_prediction_year: int,
    last_prediction_year: int,
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = SEEDS,
    n_jobs: int = DISCOVERY_XGB_N_JOBS,
    save_models: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    from .ensemble import assert_seed_diversity, ensemble_rank_pct

    training_clean, train_audit = complete_cases(training, features, require_target=True, target=target)
    candidates_clean, cand_audit = complete_cases(candidates, features, require_target=False, target=target)
    frames = []
    seed_reports = []
    for seed in seeds:
        seed_dir = None
        seed_path = None
        if output_dir is not None:
            seed_dir = Path(output_dir) / f"seed{seed}"
            seed_path = seed_dir / "predictions.parquet"
            if resume and seed_path is not None and seed_path.is_file():
                prediction = pd.read_parquet(seed_path)
                prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
                prediction = prediction.loc[
                    prediction["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)
                ].copy()
                metadata = json.loads((seed_dir / "training_report.json").read_text(encoding="utf-8"))
                frames.append(prediction)
                seed_reports.append({
                    "seed": int(seed),
                    "rows": int(len(prediction)),
                    "resumed": True,
                    "folds": metadata.get("folds", []),
                    "prediction_sha256": sha256_file(seed_path) if seed_path.is_file() else None,
                })
                continue
        params = model_parameters(seed, subsample=subsample, colsample_bytree=colsample_bytree, n_jobs=n_jobs)
        params["eval_metric"] = eval_metric
        params["rank_n_bins"] = rank_n_bins
        config = ranker_config_from_params(params)
        prediction, metadata = rolling_train_predict_open(
            training_clean,
            candidates_clean,
            seed_dir,
            features=features,
            first_prediction_year=first_prediction_year,
            last_prediction_year=last_prediction_year,
            config=config,
            target=target,
            label_end_column=label_end_column,
            save_models=save_models,
            pre_cleaned=True,
            clean_audit=(train_audit, cand_audit),
        )
        prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
        lo, hi = PREDICTION_START, SIGNAL_END
        prediction = prediction.loc[prediction["signal_date"].dt.date.between(lo, hi)].copy()
        frames.append(prediction)
        seed_reports.append({
            "seed": int(seed),
            "rows": int(len(prediction)),
            "resumed": False,
            "folds": metadata.get("folds", []),
            "prediction_sha256": sha256_file(seed_path) if seed_path is not None and seed_path.is_file() else None,
        })
    diversity = assert_seed_diversity(frames)
    ensemble = ensemble_rank_pct(frames)
    ndcgs = []
    for report in seed_reports:
        for fold in report.get("folds") or []:
            for key, value in fold.items():
                if str(key).startswith("mean_daily_ndcg"):
                    try:
                        v = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(v):
                        ndcgs.append(v)
    payload = {
        "status": "trained_oos",
        "features": list(features),
        "target": target,
        "label_end_column": label_end_column,
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "seeds": list(seeds),
        "seed_diversity": diversity,
        "rows": int(len(ensemble)),
        "days": int(ensemble["signal_date"].nunique()) if len(ensemble) else 0,
        "mean_fold_ndcg": float(np.mean(ndcgs)) if ndcgs else None,
        "seed_reports": seed_reports,
        "ensemble": ensemble,
        "seed_frames": frames,
        "filled_with_zero": False,
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        assert_project_result_path(root)
        ens_path = root / "predictions.parquet"
        ensemble.to_parquet(ens_path, index=False)
        slim = {k: v for k, v in payload.items() if k not in {"ensemble", "seed_frames"}}
        slim["prediction_sha256"] = sha256_file(ens_path)
        (root / "ensemble_report.json").write_text(
            json.dumps(slim, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return payload
