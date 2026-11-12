"""Leakage-safe rolling XGBRanker baseline for the pure-Hermes panel."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker

from .dataset import (
    ADDITIONAL_HERMES_FEATURES,
    BASE_FEATURES,
    HERMES44_FEATURES,
    HERMES_FEATURES,
    HERMES_MONEY_FLOW_FEATURES,
    HERMES_PRIMITIVE_FEATURES,
    KEYS,
)
from .single_model import SINGLE_MODEL_FEATURES

FEATURES = [*BASE_FEATURES, *HERMES_FEATURES]
FULL29_FEATURES = [*FEATURES, *ADDITIONAL_HERMES_FEATURES]
FULL44_FEATURES = [*BASE_FEATURES, *HERMES44_FEATURES]
EXECUTION_TIME_FEATURES = ["d1_open_gap"]
ALL_FEATURES = list(dict.fromkeys([
    *FULL44_FEATURES, *SINGLE_MODEL_FEATURES, *EXECUTION_TIME_FEATURES,
]))
TARGET = "label_e2_net"
FEATURE_SETS = {
    "full23": FEATURES,
    "full29": FULL29_FEATURES,
    "full44": FULL44_FEATURES,
    "single_model44": list(SINGLE_MODEL_FEATURES),
    "full20": [*BASE_FEATURES, *HERMES_PRIMITIVE_FEATURES],
    "base17": list(BASE_FEATURES),
    "drop_money_flow": [
        name for name in FEATURES if name not in HERMES_MONEY_FLOW_FEATURES
    ],
    "drop_ttl": [
        name for name in FEATURES if name != HERMES_PRIMITIVE_FEATURES[0]
    ],
    "drop_pxvol": [
        name for name in FEATURES if name != HERMES_PRIMITIVE_FEATURES[1]
    ],
    "drop_path": [
        name for name in FEATURES if name != HERMES_PRIMITIVE_FEATURES[2]
    ],
}


@dataclass(frozen=True)
class RankerConfig:
    objective: str = "rank:ndcg"
    eval_metric: str = "ndcg@3"
    ndcg_exp_gain: bool = True
    rank_n_bins: int | None = 5
    absolute_return_thresholds: tuple[float, ...] | None = None
    hybrid_loss_floor: float | None = None
    hybrid_tail_threshold: float | None = None
    precomputed_relevance: bool = False
    n_estimators: int = 400
    learning_rate: float = 0.03
    max_depth: int = 6
    min_child_weight: float = 1.0
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    tree_method: str = "hist"
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int | None = None
    validation_sessions: int | None = None


RANKER_PRESETS = {
    "conservative": RankerConfig(
        n_estimators=200, max_depth=3, min_child_weight=10.0,
        reg_lambda=5.0, reg_alpha=0.1,
    ),
    "medium": RankerConfig(
        n_estimators=300, max_depth=4, min_child_weight=5.0,
        reg_lambda=3.0, reg_alpha=0.05,
    ),
    "baseline": RankerConfig(),
}


def daily_five_bin_relevance(
    frame: pd.DataFrame,
    target: str = TARGET,
    *,
    n_bins: int = 5,
) -> pd.Series:
    """Return deterministic within-day relevance grades in ``[0, n_bins - 1]``.

    The historical function name is retained for API compatibility.  Larger
    grades always represent larger realized returns.
    """
    if not 2 <= int(n_bins) <= 31:
        raise ValueError("n_bins must be between 2 and 31")
    n_bins = int(n_bins)
    result = pd.Series(index=frame.index, dtype="int8")
    work = frame[["trade_date", "instrument_id", target]].copy()
    work["instrument_id"] = work["instrument_id"].astype(str)
    for _, block in work.groupby("trade_date", sort=True):
        ordered = block.sort_values([target, "instrument_id"], kind="mergesort")
        n = len(ordered)
        if n <= 1 or ordered[target].nunique(dropna=True) <= 1:
            grade = np.zeros(n, dtype=np.int8)
        else:
            # Exact-count ordinal buckets avoid qcut edge failures on tied returns.
            ordinal = np.arange(n, dtype=np.int64)
            grade = np.minimum(n_bins - 1, (ordinal * n_bins // n)).astype(np.int8)
        result.loc[ordered.index] = grade
    return result.astype("int8")


def daily_ordinal_relevance(
    frame: pd.DataFrame,
    target: str = TARGET,
) -> pd.Series:
    """Return deterministic full within-day ranks, worst 0 and best n-1."""
    result = pd.Series(index=frame.index, dtype="int32")
    work = frame[["trade_date", "instrument_id", target]].copy()
    work["instrument_id"] = work["instrument_id"].astype(str)
    for _, block in work.groupby("trade_date", sort=True):
        ordered = block.sort_values([target, "instrument_id"], kind="mergesort")
        result.loc[ordered.index] = np.arange(len(ordered), dtype=np.int32)
    return result.astype("int32")


def daily_relevance(
    frame: pd.DataFrame,
    target: str = TARGET,
    *,
    n_bins: int | None = 5,
    absolute_return_thresholds: Sequence[float] | None = None,
    hybrid_loss_floor: float | None = None,
    hybrid_tail_threshold: float | None = None,
    precomputed: bool = False,
) -> pd.Series:
    """Dispatch to fixed bins or full ordinal ranks when ``n_bins`` is None."""
    hybrid_requested = hybrid_loss_floor is not None or hybrid_tail_threshold is not None
    if precomputed:
        if absolute_return_thresholds is not None or hybrid_requested:
            raise ValueError("precomputed relevance cannot use threshold overrides")
        values = pd.to_numeric(frame[target], errors="coerce")
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("precomputed relevance requires finite targets")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("precomputed relevance requires integer targets")
        if (numeric < 0).any() or (numeric > 31).any():
            raise ValueError("precomputed relevance must be within [0, 31]")
        return pd.Series(numeric.astype(np.int8), index=frame.index, dtype="int8")
    if absolute_return_thresholds is not None and hybrid_requested:
        raise ValueError("absolute and hybrid relevance modes are mutually exclusive")
    if hybrid_requested:
        if hybrid_loss_floor is None or hybrid_tail_threshold is None:
            raise ValueError("hybrid relevance requires both loss floor and tail threshold")
        if n_bins is None:
            raise ValueError("hybrid relevance requires a fixed base bin count")
        if not np.isfinite([hybrid_loss_floor, hybrid_tail_threshold]).all():
            raise ValueError("hybrid relevance thresholds must be finite")
        if float(hybrid_loss_floor) >= float(hybrid_tail_threshold):
            raise ValueError("hybrid loss floor must be below the tail threshold")
        values = pd.to_numeric(frame[target], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("hybrid relevance requires finite targets")
        grades = daily_five_bin_relevance(frame, target=target, n_bins=n_bins)
        grades.loc[values.le(float(hybrid_loss_floor))] = 0
        grades.loc[values.gt(float(hybrid_tail_threshold))] = int(n_bins)
        return grades.astype("int8")
    if absolute_return_thresholds is not None:
        thresholds = np.asarray(absolute_return_thresholds, dtype=float)
        if thresholds.ndim != 1 or len(thresholds) < 1:
            raise ValueError("absolute return thresholds must be a non-empty sequence")
        if not np.isfinite(thresholds).all() or not np.all(np.diff(thresholds) > 0):
            raise ValueError("absolute return thresholds must be finite and increasing")
        values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("absolute return relevance requires finite targets")
        grades = np.searchsorted(thresholds, values, side="left")
        return pd.Series(grades, index=frame.index, dtype="int8")
    if n_bins is None:
        return daily_ordinal_relevance(frame, target=target)
    return daily_five_bin_relevance(frame, target=target, n_bins=n_bins)


def in_memory_complete_cases(
    frame: pd.DataFrame,
    features: Sequence[str] = FEATURES,
    *,
    require_target: bool,
    target: str = TARGET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop non-finite model inputs on a copy; the source parquet is untouched."""
    needed = [*features, *([target] if require_target else [])]
    missing = sorted({*KEYS, *needed} - set(frame.columns))
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work[list(features)] = work[list(features)].replace([np.inf, -np.inf], np.nan)
    mask = work[needed].notna().all(axis=1)
    audit = {
        "rows_before": len(work),
        "rows_dropped_in_memory": int((~mask).sum()),
        "rows_after": int(mask.sum()),
        "require_target": require_target,
        "feature_missing_rows": {
            name: int(work[name].isna().sum()) for name in features
        },
    }
    clean = work.loc[mask].sort_values(KEYS).reset_index(drop=True)
    return clean, audit


def mean_daily_ndcg(frame: pd.DataFrame, *, k: int = 3) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    scores: list[float] = []
    for _, block in frame.groupby("signal_date", sort=True):
        if len(block) < 2:
            continue
        y_true = block["relevance"].to_numpy(dtype=float)[None, :]
        y_score = block["score"].to_numpy(dtype=float)[None, :]
        scores.append(float(ndcg_score(y_true, y_score, k=k, ignore_ties=False)))
    return float(np.mean(scores)) if scores else float("nan")


def mean_daily_ndcg_at_3(frame: pd.DataFrame) -> float:
    """Backward-compatible NDCG@3 helper."""
    return mean_daily_ndcg(frame, k=3)


def rolling_train_predict(
    training_panel: pd.DataFrame,
    candidate_panel: pd.DataFrame,
    output_dir: Path,
    *,
    features: Sequence[str] = FEATURES,
    first_prediction_year: int = 2021,
    last_prediction_year: int = 2026,
    config: RankerConfig | None = None,
    target: str = TARGET,
    label_end_column: str = "next2_date",
    save_models: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit exact three-calendar-year windows and predict the following year."""
    config = config or RankerConfig()
    features = list(features)
    unknown = sorted(set(features) - set(ALL_FEATURES))
    if unknown or not features:
        raise ValueError(f"invalid feature subset; unknown={unknown}, count={len(features)}")
    training, train_clean_audit = in_memory_complete_cases(
        training_panel, features, require_target=True, target=target
    )
    candidates, candidate_clean_audit = in_memory_complete_cases(
        candidate_panel, features, require_target=False, target=target
    )
    if label_end_column not in training:
        raise ValueError(f"training panel missing label end column {label_end_column}")
    training[label_end_column] = pd.to_datetime(training[label_end_column])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    if save_models:
        model_dir.mkdir(exist_ok=True)

    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    params = asdict(config)
    raw_rank_n_bins = params.pop("rank_n_bins")
    raw_absolute_return_thresholds = params.pop("absolute_return_thresholds")
    hybrid_loss_floor = params.pop("hybrid_loss_floor")
    hybrid_tail_threshold = params.pop("hybrid_tail_threshold")
    precomputed_relevance = bool(params.pop("precomputed_relevance"))
    early_stopping_rounds = params.pop("early_stopping_rounds")
    validation_sessions = params.pop("validation_sessions")
    if (early_stopping_rounds is None) != (validation_sessions is None):
        raise ValueError(
            "early_stopping_rounds and validation_sessions must be set together"
        )
    if early_stopping_rounds is not None:
        if int(early_stopping_rounds) <= 0 or int(validation_sessions) <= 0:
            raise ValueError("early stopping values must be positive")
        params["early_stopping_rounds"] = int(early_stopping_rounds)
    rank_n_bins = None if raw_rank_n_bins is None else int(raw_rank_n_bins)
    if rank_n_bins is not None and not 2 <= rank_n_bins <= 31:
        raise ValueError("rank_n_bins must be between 2 and 31 or None")
    absolute_return_thresholds = (
        None
        if raw_absolute_return_thresholds is None
        else tuple(float(value) for value in raw_absolute_return_thresholds)
    )
    try:
        metric_k = int(str(config.eval_metric).rsplit("@", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError("eval_metric must have the form ndcg@K") from error
    for predict_year in range(first_prediction_year, last_prediction_year + 1):
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
        train_available = train
        validation = None
        validation_start = None
        if validation_sessions is not None:
            dates = train_available["trade_date"].drop_duplicates().sort_values()
            if len(dates) <= int(validation_sessions) + 2:
                raise RuntimeError(
                    f"insufficient training dates for {validation_sessions} validation sessions"
                )
            validation_start = pd.Timestamp(dates.iloc[-int(validation_sessions)])
            # Purge fit samples whose realized exit reaches the validation window.
            fit_mask = (
                train_available["trade_date"].lt(validation_start)
                & train_available[label_end_column].lt(validation_start)
            )
            train = train_available.loc[fit_mask].copy()
            validation = train_available.loc[
                train_available["trade_date"].ge(validation_start)
            ].copy()
            train_sizes = train.groupby("trade_date")["instrument_id"].transform("size")
            validation_sizes = validation.groupby("trade_date")["instrument_id"].transform("size")
            train = train.loc[train_sizes.ge(2)].sort_values(KEYS).reset_index(drop=True)
            validation = validation.loc[validation_sizes.ge(2)].sort_values(KEYS).reset_index(drop=True)
            if train.empty or validation.empty:
                raise RuntimeError(
                    f"empty fit/validation split for {train_start}-{train_end} -> {predict_year}"
                )
        relevance = daily_relevance(
            train,
            target=target,
            n_bins=rank_n_bins,
            absolute_return_thresholds=absolute_return_thresholds,
            hybrid_loss_floor=hybrid_loss_floor,
            hybrid_tail_threshold=hybrid_tail_threshold,
            precomputed=precomputed_relevance,
        )
        model = XGBRanker(**params)
        fit_kwargs: dict[str, Any] = {}
        if validation is not None:
            validation_relevance = daily_relevance(
                validation,
                target=target,
                n_bins=rank_n_bins,
                absolute_return_thresholds=absolute_return_thresholds,
                hybrid_loss_floor=hybrid_loss_floor,
                hybrid_tail_threshold=hybrid_tail_threshold,
                precomputed=precomputed_relevance,
            )
            fit_kwargs.update({
                "eval_set": [(
                    validation[features].to_numpy(dtype=np.float32),
                    validation_relevance.to_numpy(dtype=np.int32),
                )],
                "eval_qid": [validation["trade_date"].factorize(sort=True)[0]],
            })
        model.fit(
            train[features].to_numpy(dtype=np.float32),
            relevance.to_numpy(dtype=np.int32),
            qid=train["trade_date"].factorize(sort=True)[0],
            verbose=False,
            **fit_kwargs,
        )
        score = model.predict(test[features].to_numpy(dtype=np.float32))
        prediction = test[["trade_date", "instrument_id", target]].copy()
        prediction = prediction.rename(columns={"trade_date": "signal_date"})
        prediction["score"] = score
        prediction["predict_year"] = predict_year
        prediction["fold_train_start"] = train_start
        prediction["fold_train_end"] = train_end

        evaluable = prediction[target].notna()
        eval_frame = prediction.loc[evaluable].copy()
        eval_frame["trade_date"] = eval_frame["signal_date"]
        eval_frame["relevance"] = daily_relevance(
            eval_frame,
            target=target,
            n_bins=rank_n_bins,
            absolute_return_thresholds=absolute_return_thresholds,
            hybrid_loss_floor=hybrid_loss_floor,
            hybrid_tail_threshold=hybrid_tail_threshold,
            precomputed=precomputed_relevance,
        )
        metric_top = eval_frame.sort_values(
            ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
        ).groupby("signal_date", sort=False).head(metric_k)
        folds.append({
            "predict_year": predict_year,
            "train_years": [train_start, train_end],
            "train_rows_available": len(train_available),
            "train_rows": len(train),
            "train_days": int(train["trade_date"].nunique()),
            "validation_start": (
                validation_start.isoformat() if validation_start is not None else None
            ),
            "validation_rows": len(validation) if validation is not None else 0,
            "validation_days": (
                int(validation["trade_date"].nunique()) if validation is not None else 0
            ),
            "best_iteration": (
                int(model.best_iteration) if validation is not None else None
            ),
            "best_score": float(model.best_score) if validation is not None else None,
            "candidate_rows_scored": len(prediction),
            "candidate_days": int(prediction["signal_date"].nunique()),
            "label_missing_candidates_scored": int((~evaluable).sum()),
            "evaluation_rows": int(evaluable.sum()),
            f"mean_daily_ndcg_at_{metric_k}": mean_daily_ndcg(eval_frame, k=metric_k),
            f"top{metric_k}_mean_target": float(metric_top[target].mean()),
        })
        if save_models:
            model.save_model(
                model_dir / f"xgbranker_{train_start}_{train_end}_to_{predict_year}.json"
            )
        predictions.append(prediction.drop(columns=[target]))

    result = pd.concat(predictions, ignore_index=True).sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    if result.duplicated(["signal_date", "instrument_id"]).any():
        raise RuntimeError("duplicate out-of-sample predictions")
    path = output_dir / "predictions.parquet"
    result.to_parquet(path, index=False)
    metadata = {
        "status": "trained_oos",
        "walk_forward": "exact prior 3 calendar years train, next 1 calendar year predict",
        "target": target,
        "label_end_column": label_end_column,
        "label_transform": (
            "precomputed integer relevance"
            if precomputed_relevance
            else
            (
                f"daily {rank_n_bins}-bin grades with loss <= {hybrid_loss_floor} "
                f"forced to 0 and return > {hybrid_tail_threshold} promoted to {rank_n_bins}"
            )
            if hybrid_loss_floor is not None
            else (
                f"absolute return thresholds {list(absolute_return_thresholds)}"
                if absolute_return_thresholds is not None
                else (
                    "daily deterministic full ordinal ranks, worst 0 and best n-1"
                    if rank_n_bins is None
                    else (
                        f"daily deterministic {rank_n_bins} equal-count ordinal grades "
                        f"0..{rank_n_bins - 1}"
                    )
                )
            )
        ),
        "metric": f"mean daily NDCG@{metric_k} on label-observable rows only",
        "prediction_universe": "all point-in-time feature-complete candidates; labels never filter scoring",
        "features": features,
        "feature_contract_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
        "ranker": params | {
            "rank_n_bins": rank_n_bins,
            "absolute_return_thresholds": absolute_return_thresholds,
            "hybrid_loss_floor": hybrid_loss_floor,
            "hybrid_tail_threshold": hybrid_tail_threshold,
            "precomputed_relevance": precomputed_relevance,
            "validation_sessions": validation_sessions,
        },
        "models_persisted": save_models,
        "training_clean_audit": train_clean_audit,
        "candidate_clean_audit": candidate_clean_audit,
        "folds": folds,
        "prediction_rows": len(result),
        "prediction_sha256": _sha256(path),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
