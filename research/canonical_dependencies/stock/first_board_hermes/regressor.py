"""Original-style daily rolling XGBRegressor for the pure-Hermes panel."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from .baseline import (
    ALL_FEATURES,
    FEATURES,
    TARGET,
    daily_five_bin_relevance,
    in_memory_complete_cases,
    mean_daily_ndcg_at_3,
)
from .single_model import SINGLE_MODEL_FEATURES, SINGLE_MODEL_REDUCED30_FEATURES


REGRESSOR_FEATURES = frozenset([
    *ALL_FEATURES,
    *SINGLE_MODEL_FEATURES,
    *SINGLE_MODEL_REDUCED30_FEATURES,
])


@dataclass(frozen=True)
class RegressorConfig:
    objective: str = "reg:squarederror"
    n_estimators: int = 100
    max_depth: int = 3
    min_child_weight: float = 1.0
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    random_state: int = 42
    n_jobs: int = 2
    tree_method: str = "hist"
    device: str = "cpu"


def rolling_session_train_predict(
    training_panel: pd.DataFrame,
    candidate_panel: pd.DataFrame,
    market_calendar: Sequence[Any],
    output_dir: Path,
    *,
    features: Sequence[str] = FEATURES,
    first_prediction_year: int = 2021,
    last_prediction_year: int = 2026,
    train_sessions: int = 240,
    config: RegressorConfig | None = None,
    target: str = TARGET,
    label_end_column: str = "next2_date",
    min_train_samples: int = 30,
    sample_start: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit daily on the prior market sessions and predict continuous net return."""
    config = config or RegressorConfig()
    features = list(features)
    unknown = sorted(set(features) - REGRESSOR_FEATURES)
    if unknown or not features:
        raise ValueError(f"invalid feature subset; unknown={unknown}, count={len(features)}")
    if train_sessions <= 0:
        raise ValueError("train_sessions must be positive")

    training, train_clean_audit = in_memory_complete_cases(
        training_panel, features, require_target=True, target=target
    )
    candidates, candidate_clean_audit = in_memory_complete_cases(
        candidate_panel, features, require_target=False, target=target
    )
    if label_end_column not in training:
        raise ValueError(f"training panel missing label end column {label_end_column}")
    training[label_end_column] = pd.to_datetime(training[label_end_column])
    candidates["trade_date"] = pd.to_datetime(candidates["trade_date"])
    calendar = pd.DatetimeIndex(pd.to_datetime(list(market_calendar))).drop_duplicates().sort_values()
    if calendar.empty:
        raise ValueError("market calendar is empty")
    rolling_prediction_start = None
    if sample_start is not None:
        sample_start = pd.Timestamp(sample_start)
        sample_calendar = calendar[calendar >= sample_start]
        if len(sample_calendar) <= train_sessions:
            raise ValueError(
                f"sample calendar has {len(sample_calendar)} sessions, "
                f"not enough for {train_sessions}-session cold start"
            )
        rolling_prediction_start = pd.Timestamp(sample_calendar[train_sessions])
        training = training.loc[training["trade_date"].ge(sample_start)].copy()
        candidates = candidates.loc[
            candidates["trade_date"].ge(rolling_prediction_start)
        ].copy()
    calendar_pos = {day: i for i, day in enumerate(calendar)}

    output_dir.mkdir(parents=True, exist_ok=True)
    params = asdict(config)
    prediction_blocks: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    signal_dates = pd.DatetimeIndex(candidates["trade_date"].drop_duplicates()).sort_values()
    signal_dates = signal_dates[
        (signal_dates.year >= first_prediction_year) & (signal_dates.year <= last_prediction_year)
    ]

    for signal_date in signal_dates:
        pos = calendar_pos.get(pd.Timestamp(signal_date))
        if pos is None or pos < train_sessions:
            continue
        window = calendar[pos - train_sessions : pos]
        train = training.loc[
            training["trade_date"].between(window[0], window[-1])
            & training[label_end_column].le(signal_date)
        ].copy()
        test = candidates.loc[candidates["trade_date"].eq(signal_date)].copy()
        if len(train) < min_train_samples or test.empty:
            audit_rows.append({
                "signal_date": signal_date,
                "train_start": window[0],
                "train_end": window[-1],
                "train_market_sessions": train_sessions,
                "train_rows": len(train),
                "candidate_rows": len(test),
                "status": "insufficient_train_samples" if len(train) < min_train_samples else "empty_test",
            })
            continue

        day_counts = train["trade_date"].value_counts()
        weights_by_day = 1.0 / day_counts
        sample_weight = train["trade_date"].map(weights_by_day).to_numpy()
        model = XGBRegressor(**params)
        model.fit(
            train[features].to_numpy(dtype=np.float32),
            train[target].to_numpy(dtype=np.float32),
            sample_weight=sample_weight,
            verbose=False,
        )
        scores = model.predict(test[features].to_numpy(dtype=np.float32))
        block = test[["trade_date", "instrument_id", target]].copy()
        block = block.rename(columns={"trade_date": "signal_date"})
        block["score"] = scores
        block["train_start"] = window[0]
        block["train_end"] = window[-1]
        prediction_blocks.append(block)
        audit_rows.append({
            "signal_date": signal_date,
            "train_start": window[0],
            "train_end": window[-1],
            "train_market_sessions": train_sessions,
            "train_rows": len(train),
            "train_days_with_candidates": int(train["trade_date"].nunique()),
            "candidate_rows": len(test),
            "latest_mature_label_end": train[label_end_column].max(),
            "status": "trained",
        })

    if not prediction_blocks:
        raise RuntimeError("no regressor predictions were produced")
    scored = pd.concat(prediction_blocks, ignore_index=True).sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    if scored.duplicated(["signal_date", "instrument_id"]).any():
        raise RuntimeError("duplicate out-of-sample predictions")

    evaluable = scored[target].notna()
    evaluation = scored.loc[evaluable].copy()
    evaluation["trade_date"] = evaluation["signal_date"]
    evaluation["relevance"] = daily_five_bin_relevance(evaluation, target=target)
    top3 = evaluation.sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    ).groupby("signal_date", sort=False).head(3)
    yearly = []
    for year, block in evaluation.groupby(evaluation["signal_date"].dt.year):
        selected = top3.loc[top3["signal_date"].dt.year.eq(year)]
        regression_metrics = _regression_metrics(block[target], block["score"])
        yearly.append({
            "year": int(year),
            "rows": len(block),
            "days": int(block["signal_date"].nunique()),
            **regression_metrics,
            "mean_daily_ndcg_at_3": mean_daily_ndcg_at_3(block),
            "top3_mean_target": float(selected[target].mean()),
        })

    prediction_path = output_dir / "predictions.parquet"
    audit_path = output_dir / "daily_training_audit.parquet"
    scored.drop(columns=[target]).to_parquet(prediction_path, index=False)
    pd.DataFrame(audit_rows).to_parquet(audit_path, index=False)
    oos_regression_metrics = _regression_metrics(evaluation[target], evaluation["score"])
    metadata = {
        "status": "trained_oos",
        "model": "XGBRegressor",
        "objective": config.objective,
        "walk_forward": f"daily refit on prior {train_sessions} exchange sessions",
        "sample_start": sample_start.isoformat() if sample_start is not None else None,
        "rolling_prediction_start": (
            rolling_prediction_start.isoformat()
            if rolling_prediction_start is not None else None
        ),
        "target": target,
        "label_end_column": label_end_column,
        "label_transform": "none; continuous execution-aligned net return",
        "sample_weight": "equal total weight per signal day",
        "features": features,
        "feature_contract_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
        "regressor": params,
        "xgboost_version": xgboost.__version__,
        "training_clean_audit": train_clean_audit,
        "candidate_clean_audit": candidate_clean_audit,
        "yearly": yearly,
        "primary_evaluation_metric": "oos_mse",
        "oos_regression_metrics": oos_regression_metrics,
        "ranking_diagnostics": {
            "mean_daily_ndcg_at_3": mean_daily_ndcg_at_3(evaluation),
            "top3_mean_target": float(top3[target].mean()),
        },
        # Backward-compatible aliases. These are diagnostics, not the primary
        # evaluation metric for XGBRegressor.
        "mean_daily_ndcg_at_3": mean_daily_ndcg_at_3(evaluation),
        "top3_mean_target": float(top3[target].mean()),
        "prediction_rows": len(scored),
        "prediction_sha256": _sha256(prediction_path),
        "daily_training_audit_sha256": _sha256(audit_path),
        "models_persisted": False,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return scored.drop(columns=[target]), metadata


def annual_three_year_train_one_year_predict(
    training_panel: pd.DataFrame,
    candidate_panel: pd.DataFrame,
    output_dir: Path,
    *,
    features: Sequence[str],
    prediction_years: Sequence[int],
    config: RegressorConfig | None = None,
    target: str = "label_exit_rule_net",
    label_end_column: str = "actual_exit_date",
    target_transform: str = "none",
    target_clip: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit once on the prior three calendar years and predict the next year."""
    config = config or RegressorConfig()
    features = list(features)
    unknown = sorted(set(features) - REGRESSOR_FEATURES)
    if unknown or not features:
        raise ValueError(f"invalid feature subset; unknown={unknown}, count={len(features)}")
    years = sorted(set(int(year) for year in prediction_years))
    if not years:
        raise ValueError("prediction_years must not be empty")
    if target_transform not in {"none", "daily_percentile_rank"}:
        raise ValueError(f"unsupported target transform: {target_transform}")
    if target_clip is not None:
        if target_transform != "none":
            raise ValueError("target clipping requires the untransformed continuous target")
        lower, upper = (float(value) for value in target_clip)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("target clip bounds must be finite and increasing")
        target_clip = (lower, upper)

    training, train_clean_audit = in_memory_complete_cases(
        training_panel, features, require_target=True, target=target
    )
    candidates, candidate_clean_audit = in_memory_complete_cases(
        candidate_panel, features, require_target=False, target=target
    )
    if label_end_column not in training:
        raise ValueError(f"training panel missing label end column {label_end_column}")
    training[label_end_column] = pd.to_datetime(training[label_end_column])
    candidates["trade_date"] = pd.to_datetime(candidates["trade_date"])
    output_dir.mkdir(parents=True, exist_ok=True)

    params = asdict(config)
    prediction_blocks: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    model_files: dict[str, dict[str, Any]] = {}
    for prediction_year in years:
        prediction_start = pd.Timestamp(f"{prediction_year}-01-01")
        prediction_end = pd.Timestamp(f"{prediction_year}-12-31")
        train_start = pd.Timestamp(f"{prediction_year - 3}-01-01")
        train_end = prediction_start - pd.Timedelta(days=1)
        train = training.loc[
            training["trade_date"].between(train_start, train_end)
            & training[label_end_column].lt(prediction_start)
        ].copy()
        test = candidates.loc[
            candidates["trade_date"].between(prediction_start, prediction_end)
        ].copy()
        if train.empty or test.empty:
            raise RuntimeError(
                f"empty annual fold {prediction_year}: train={len(train)}, test={len(test)}"
            )
        day_counts = train["trade_date"].value_counts()
        sample_weight = train["trade_date"].map(1.0 / day_counts).to_numpy()
        model_target = (
            _daily_percentile_target(train, target)
            if target_transform == "daily_percentile_rank"
            else (
                train[target].clip(*target_clip)
                if target_clip is not None
                else train[target]
            )
        )
        model = XGBRegressor(**params)
        model.fit(
            train[features].to_numpy(dtype=np.float32),
            model_target.to_numpy(dtype=np.float32),
            sample_weight=sample_weight,
            verbose=False,
        )
        scores = model.predict(test[features].to_numpy(dtype=np.float32))
        block = test[["trade_date", "instrument_id", target]].copy()
        block = block.rename(columns={"trade_date": "signal_date"})
        block["score"] = scores
        block["train_start"] = train_start
        block["train_end"] = train_end
        block["prediction_year"] = prediction_year
        prediction_blocks.append(block)

        model_path = output_dir / f"model_predict_{prediction_year}.json"
        model.save_model(model_path)
        model_files[str(prediction_year)] = {
            "path": model_path.name,
            "sha256": _sha256(model_path),
        }
        evaluation = block.loc[block[target].notna()]
        evaluation_target = (
            _daily_percentile_target(
                evaluation.rename(columns={"signal_date": "trade_date"}), target
            )
            if target_transform == "daily_percentile_rank"
            else (
                evaluation[target].clip(*target_clip)
                if target_clip is not None
                else evaluation[target]
            )
        )
        fold_rows.append({
            "prediction_year": prediction_year,
            "train_start": train_start,
            "train_end": train_end,
            "train_rows": int(len(train)),
            "train_candidate_days": int(train["trade_date"].nunique()),
            "latest_mature_label_end": train[label_end_column].max(),
            "prediction_rows": int(len(block)),
            **_regression_metrics(evaluation_target, evaluation["score"]),
            "raw_target_mean": float(evaluation[target].mean()),
        })

    scored = pd.concat(prediction_blocks, ignore_index=True).sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    if scored.duplicated(["signal_date", "instrument_id"]).any():
        raise RuntimeError("duplicate annual out-of-sample predictions")
    evaluation = scored.loc[scored[target].notna()]
    evaluation_target = (
        _daily_percentile_target(
            evaluation.rename(columns={"signal_date": "trade_date"}), target
        )
        if target_transform == "daily_percentile_rank"
        else (
            evaluation[target].clip(*target_clip)
            if target_clip is not None
            else evaluation[target]
        )
    )
    prediction_path = output_dir / "predictions.parquet"
    scored.drop(columns=[target]).to_parquet(prediction_path, index=False)
    metadata = {
        "status": "trained_annual_oos",
        "model": "XGBRegressor",
        "objective": config.objective,
        "primary_evaluation_metric": "oos_mse",
        "walk_forward": "fit prior three calendar years once; predict next calendar year",
        "target": target,
        "target_transform": target_transform,
        "target_clip": target_clip,
        "label_end_column": label_end_column,
        "maturity_rule": "label_end_column < prediction-year start",
        "sample_weight": "equal total weight per signal day",
        "features": features,
        "feature_contract_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
        "regressor": params,
        "xgboost_version": xgboost.__version__,
        "training_clean_audit": train_clean_audit,
        "candidate_clean_audit": candidate_clean_audit,
        "folds": fold_rows,
        "oos_regression_metrics": _regression_metrics(evaluation_target, evaluation["score"]),
        "raw_target_diagnostics": {
            "mean": float(evaluation[target].mean()),
            "positive_rate": float(evaluation[target].gt(0).mean()),
        },
        "prediction_rows": int(len(scored)),
        "prediction_sha256": _sha256(prediction_path),
        "model_files": model_files,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return scored.drop(columns=[target]), metadata


def _daily_percentile_target(frame: pd.DataFrame, target: str) -> pd.Series:
    """Map each day's continuous target to deterministic values in [0, 1]."""
    required = {"trade_date", "instrument_id", target}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"daily percentile target missing columns: {missing}")
    result = pd.Series(index=frame.index, dtype=np.float64)
    for _, block in frame.groupby("trade_date", sort=True):
        ordered = block.sort_values([target, "instrument_id"], kind="mergesort")
        if len(ordered) == 1:
            values = np.array([0.5], dtype=np.float64)
        else:
            values = np.arange(len(ordered), dtype=np.float64) / (len(ordered) - 1)
        result.loc[ordered.index] = values
    if result.isna().any() or not result.between(0.0, 1.0).all():
        raise RuntimeError("invalid daily percentile target")
    return result


def _regression_metrics(actual: Any, predicted: Any) -> dict[str, float | int]:
    """Return row-weighted OOS regression metrics for continuous targets."""
    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.any():
        raise ValueError("no finite observations available for regression evaluation")
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    residual = y_pred - y_true
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0 else float("nan")
    return {
        "oos_rows": int(len(y_true)),
        "oos_mse": mse,
        "oos_rmse": float(np.sqrt(mse)),
        "oos_mae": float(np.mean(np.abs(residual))),
        "oos_r2": r2,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
