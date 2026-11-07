"""走步模型。XGB 参数与既有 Ranker 一致；LightGBM 只用调用方传入的冻结参数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .r225241_io import dump_json
from .r225241_stats import ranking_groups_contiguous
from .round22_3.train import complete_cases_keep_extras

KEYS = ["trade_date", "instrument_id"]


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    return work


def fold_slices(year: int, label_end: pd.Series, trade_date: pd.Series) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    train_start, train_end = year - 3, year - 1
    dates = pd.to_datetime(trade_date)
    end = pd.to_datetime(label_end)
    train = dates.dt.year.between(train_start, train_end) & end.dt.year.le(train_end)
    inner_start = pd.Timestamp(year - 1, 7, 1)
    return train, dates.dt.year.eq(year), inner_start


def xgb_rank_fold_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    *,
    target: str,
    seed: int,
    n_bins: int,
    eval_metric: str,
    n_jobs: int = 2,
) -> np.ndarray:
    from xgboost import XGBRanker

    from .round17.train import ensure_stock_src_on_path

    ensure_stock_src_on_path()
    from first_board_hermes.baseline import daily_relevance

    work = train.sort_values(KEYS).reset_index(drop=True)
    if not ranking_groups_contiguous(work["trade_date"]):
        raise RuntimeError("XGB ranking groups are not one contiguous block per date")
    relevance = daily_relevance(work, target=target, n_bins=n_bins)
    model = XGBRanker(
        objective="rank:ndcg",
        ndcg_exp_gain=True,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=int(seed),
        n_jobs=int(n_jobs),
        eval_metric=eval_metric,
    )
    model.fit(
        work[list(features)].to_numpy(dtype=np.float32),
        relevance.to_numpy(dtype=np.int32),
        qid=work["trade_date"].factorize(sort=True)[0],
        verbose=False,
    )
    return model.predict(test[list(features)].to_numpy(dtype=np.float32))


def train_xgb_ranker(
    *,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    features: Sequence[str],
    extras: Sequence[str],
    target: str,
    label_end_column: str,
    n_bins: int,
    eval_metric: str,
    output_dir: Path,
    seed: int = 17,
    n_jobs: int = 2,
    first_year: int = 2018,
    last_year: int = 2026,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    pred_path = output_dir / "predictions.parquet"
    meta_path = output_dir / "train_meta.json"
    if resume and pred_path.is_file() and meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    features = list(features)
    training_clean, train_audit = complete_cases_keep_extras(
        _prepare(training), features, extras, require_target=True, target=target
    )
    candidates_clean, cand_audit = complete_cases_keep_extras(
        _prepare(candidates), features, extras, require_target=False, target=target
    )
    training_clean[label_end_column] = pd.to_datetime(training_clean[label_end_column])
    rows = []
    folds = []
    for year in range(first_year, last_year + 1):
        train_mask, _, _ = fold_slices(year, training_clean[label_end_column], training_clean["trade_date"])
        train = training_clean.loc[train_mask].copy()
        sizes = train.groupby("trade_date")["instrument_id"].transform("size")
        train = train.loc[sizes.ge(2)].sort_values(KEYS).reset_index(drop=True)
        test = candidates_clean.loc[candidates_clean["trade_date"].dt.year.eq(year)].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"empty rank fold {year}")
        score = xgb_rank_fold_predict(
            train, test, features, target=target, seed=seed, n_bins=n_bins, eval_metric=eval_metric, n_jobs=n_jobs
        )
        part = test[["trade_date", "instrument_id"]].copy()
        part = part.rename(columns={"trade_date": "signal_date"})
        part["score"] = score
        part["predict_year"] = year
        rows.append(part)
        folds.append({"predict_year": year, "train_rows": int(len(train)), "test_rows": int(len(test))})
        print(f"[xgb-rank] year={year} train={len(train)} test={len(test)}", flush=True)
    pred = pd.concat(rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(pred_path, index=False)
    payload = {
        "status": "trained_oos",
        "seed": int(seed),
        "features": features,
        "extras": list(extras),
        "target": target,
        "n_bins": int(n_bins),
        "eval_metric": eval_metric,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "rows": int(len(pred)),
        "folds": folds,
        "train_clean_audit": train_audit,
        "candidate_clean_audit": cand_audit,
        "filled_with_zero": False,
        "label_bins_fit_inside_training_fold": True,
    }
    dump_json(meta_path, payload)
    return payload


def train_lgbm_ranker(
    *,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    features: Sequence[str],
    target: str,
    label_end_column: str,
    n_bins: int,
    eval_at: list[int],
    output_dir: Path,
    seed: int = 17,
    first_year: int = 2018,
    last_year: int = 2026,
    resume: bool = True,
) -> dict[str, Any]:
    import lightgbm as lgb

    from .round17.train import ensure_stock_src_on_path

    ensure_stock_src_on_path()
    from first_board_hermes.baseline import daily_relevance

    output_dir = Path(output_dir)
    pred_path = output_dir / "predictions.parquet"
    meta_path = output_dir / "train_meta.json"
    if resume and pred_path.is_file() and meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    features = list(features)
    training_clean, train_audit = complete_cases_keep_extras(
        _prepare(training), features, [], require_target=True, target=target
    )
    candidates_clean, cand_audit = complete_cases_keep_extras(
        _prepare(candidates), features, [], require_target=False, target=target
    )
    training_clean[label_end_column] = pd.to_datetime(training_clean[label_end_column])
    rows = []
    folds = []
    params = {
        "objective": "lambdarank",
        "learning_rate": 0.03,
        "n_estimators": 200,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 50,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "verbosity": -1,
        "seed": int(seed),
        "eval_at": list(eval_at),
        "n_jobs": 2,
    }
    for year in range(first_year, last_year + 1):
        train_mask, _, _ = fold_slices(year, training_clean[label_end_column], training_clean["trade_date"])
        train = training_clean.loc[train_mask].copy()
        sizes = train.groupby("trade_date")["instrument_id"].transform("size")
        train = train.loc[sizes.ge(2)].sort_values(KEYS).reset_index(drop=True)
        test = candidates_clean.loc[candidates_clean["trade_date"].dt.year.eq(year)].sort_values(KEYS)
        if train.empty or test.empty:
            raise RuntimeError(f"empty lgbm fold {year}")
        if not ranking_groups_contiguous(train["trade_date"]):
            raise RuntimeError("LGBM groups are not contiguous")
        relevance = daily_relevance(train, target=target, n_bins=n_bins)
        group = train.groupby("trade_date", sort=False).size().to_numpy()
        model = lgb.LGBMRanker(**params)
        model.fit(
            train[features].to_numpy(dtype=np.float32),
            relevance.to_numpy(dtype=np.int32),
            group=group,
        )
        score = model.predict(test[features].to_numpy(dtype=np.float32))
        part = test[["trade_date", "instrument_id"]].copy()
        part = part.rename(columns={"trade_date": "signal_date"})
        part["score"] = score
        part["predict_year"] = year
        rows.append(part)
        folds.append({"predict_year": year, "train_rows": int(len(train)), "test_rows": int(len(test)), "n_groups": int(len(group))})
        print(f"[lgbm] year={year} train={len(train)} test={len(test)}", flush=True)
    pred = pd.concat(rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(pred_path, index=False)
    payload = {
        "status": "trained_oos",
        "model": "LGBMRanker",
        "params": params,
        "features": features,
        "target": target,
        "n_bins": int(n_bins),
        "rows": int(len(pred)),
        "folds": folds,
        "train_clean_audit": train_audit,
        "candidate_clean_audit": cand_audit,
        "filled_with_zero": False,
        "label_bins_fit_inside_training_fold": True,
        "one_group_per_trade_date": True,
    }
    dump_json(meta_path, payload)
    return payload


def _xgb_tree(kind: str, seed: int, n_jobs: int):
    common = dict(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=1,
        reg_lambda=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )
    if kind == "classifier":
        from xgboost import XGBClassifier

        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    from xgboost import XGBRegressor

    return XGBRegressor(objective="reg:squarederror", **common)


def train_walk_xgb_supervised(
    *,
    frame: pd.DataFrame,
    features: Sequence[str],
    label: str,
    label_end: str,
    tradable: str,
    kind: str,
    output_dir: Path,
    seed: int = 17,
    n_jobs: int = 2,
    first_year: int = 2018,
    last_year: int = 2026,
    resume: bool = True,
) -> dict[str, Any]:
    """Stage A/B。不可交易或标签未成熟不记成亏损 0。"""
    output_dir = Path(output_dir)
    pred_path = output_dir / "predictions.parquet"
    meta_path = output_dir / "train_meta.json"
    if resume and pred_path.is_file() and meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    work = _prepare(frame)
    work[label_end] = pd.to_datetime(work[label_end])
    rows = []
    folds = []
    features = list(features)
    for year in range(first_year, last_year + 1):
        train_mask, test_mask, _ = fold_slices(year, work[label_end], work["trade_date"])
        train = work.loc[train_mask].copy()
        mature = train[label].notna() & train[tradable].fillna(False).astype(bool)
        train = train.loc[mature]
        test = work.loc[test_mask].copy()
        if train.empty or test.empty:
            folds.append({"predict_year": year, "status": "EMPTY"})
            continue
        model = _xgb_tree(kind, seed, n_jobs)
        y = pd.to_numeric(train[label], errors="coerce")
        if kind == "classifier":
            y = (y > 0).astype(int)
        model.fit(train[features].to_numpy(dtype=np.float32), y.to_numpy())
        raw = model.predict_proba(test[features].to_numpy(dtype=np.float32))[:, 1] if kind == "classifier" else model.predict(test[features].to_numpy(dtype=np.float32))
        part = test[["trade_date", "instrument_id", tradable, label]].copy()
        part = part.rename(columns={"trade_date": "signal_date", label: "matured_label", tradable: "tradable"})
        part["score"] = raw
        part["predict_year"] = year
        part["kind"] = kind
        part["score_is_fill_probability"] = False
        rows.append(part)
        folds.append({"predict_year": year, "train_rows": int(len(train)), "test_rows": int(len(test)), "positive_rate": float((y > 0).mean()) if kind == "classifier" else None})
        print(f"[xgb-{kind}] year={year} train={len(train)} test={len(test)}", flush=True)
    if not rows:
        raise RuntimeError(f"no supervised folds for {kind}")
    pred = pd.concat(rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(pred_path, index=False)
    payload = {
        "status": "trained_oos",
        "kind": kind,
        "features": features,
        "label": label,
        "seed": int(seed),
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "untradable_coded_as_loss": False,
        "immature_coded_as_loss": False,
        "score_is_fill_probability": False,
        "folds": folds,
        "rows": int(len(pred)),
        "filled_with_zero": False,
    }
    dump_json(meta_path, payload)
    return payload
