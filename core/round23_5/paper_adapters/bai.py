"""Bai & Pukthuanthong 分类 stacking。paper_id=P04_BAI_2026。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

PAPER_ID = "P04_BAI_2026"


def tercile_codes(values: pd.Series, low: float, high: float) -> pd.Series:
    codes = pd.Series(np.ones(len(values), dtype=int), index=values.index)
    codes.loc[values <= low] = 0
    codes.loc[values > high] = 2
    return codes


def fit_rf(features: pd.DataFrame, labels: pd.Series, *, max_depth: int | None, seed: int, n_jobs: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(features.to_numpy(dtype=float), labels.to_numpy(dtype=int))
    return model


def fit_meta(meta_x: pd.DataFrame, labels: pd.Series) -> LogisticRegression:
    model = LogisticRegression(C=1.0, max_iter=400, solver="lbfgs")
    model.fit(meta_x.to_numpy(dtype=float), labels.to_numpy(dtype=int))
    return model


def expected_class(probabilities: np.ndarray) -> np.ndarray:
    classes = np.arange(probabilities.shape[1], dtype=float)
    return probabilities @ classes


def score_frame(keys: pd.DataFrame, score: np.ndarray, fallback: np.ndarray) -> pd.DataFrame:
    out = keys.copy()
    out["score"] = score
    out["fallback_v5"] = fallback
    return out
