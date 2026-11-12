"""独立大亏 / 连板确认模型。特征仅 D 收盘可见。不改 fs00263 score。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import FS00263_FEATURES, assert_project_result_path, round2_dir
from .data import ensemble_with_top1, load_candidate_features

FEATURE_COLS = list(FS00263_FEATURES)


def _year_folds(dates: pd.Series) -> list[tuple[int, pd.DatetimeIndex, pd.DatetimeIndex]]:
    years = sorted(pd.to_datetime(dates).dt.year.unique())
    out = []
    for year in years:
        train = dates[dates.dt.year < year]
        test = dates[dates.dt.year == year]
        if len(train) < 80 or len(test) == 0:
            continue
        out.append((int(year), pd.DatetimeIndex(train), pd.DatetimeIndex(test)))
    return out


def _fit_predict_logit(x_train, y_train, x_test) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
    ])
    pipe.fit(x_train, y_train)
    return pipe.predict_proba(x_test)[:, 1]


def _fit_predict_xgb(x_train, y_train, x_test) -> np.ndarray | None:
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    xtr = imputer.fit_transform(x_train)
    xte = imputer.transform(x_test)
    pos = max(float(y_train.mean()), 1e-6)
    clf = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        n_jobs=2,
        eval_metric="logloss",
        scale_pos_weight=(1 - pos) / pos,
    )
    clf.fit(xtr, y_train)
    return clf.predict_proba(xte)[:, 1]


def build_top1_feature_table() -> pd.DataFrame:
    ens = ensemble_with_top1()
    top1 = ens.loc[ens["is_top1"], ["signal_date", "instrument_id", "score"]].copy()
    panel = load_candidate_features()
    merged = top1.merge(
        panel,
        left_on=["signal_date", "instrument_id"],
        right_on=["trade_date", "instrument_id"],
        how="left",
    )
    return merged


def walk_forward_proba(
    frame: pd.DataFrame,
    *,
    label_col: str,
    threshold: float | None,
    model: str,
) -> pd.Series:
    work = frame.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    if threshold is not None:
        y = pd.to_numeric(work[label_col], errors="coerce") <= float(threshold)
    else:
        y = pd.to_numeric(work[label_col], errors="coerce").fillna(0).astype(int).eq(1)
    work["_y"] = y.astype(int)
    usable = work.dropna(subset=["_y"])
    proba = pd.Series(np.nan, index=work.index, dtype=float)
    for year, train_idx, test_idx in _year_folds(usable["signal_date"]):
        train = usable.loc[usable["signal_date"].isin(train_idx)]
        test = usable.loc[usable["signal_date"].isin(test_idx)]
        if train["_y"].nunique() < 2 or test.empty:
            continue
        x_train = train[FEATURE_COLS]
        x_test = test[FEATURE_COLS]
        if model == "logit":
            pred = _fit_predict_logit(x_train, train["_y"], x_test)
        elif model == "xgb":
            pred = _fit_predict_xgb(x_train, train["_y"], x_test)
            if pred is None:
                return pd.Series(dtype=float)
        else:
            raise ValueError(model)
        proba.loc[test.index] = pred
    return pd.Series(proba.values, index=pd.to_datetime(work["signal_date"]), name="proba")


def persist_model_note(code: str, payload: dict[str, Any]) -> None:
    dest = round2_dir("models")
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{code}.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
