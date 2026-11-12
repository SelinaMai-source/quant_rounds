"""内层模型：每个半年块单独训练，截止于该块开始之前，回看 36 个月成熟标签。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.r225241_models import KEYS, _prepare, xgb_rank_fold_predict
from core.r225241_stats import ranking_groups_contiguous
from core.research_integrity_v2.io_v2 import dump_json
from core.round22_3.train import complete_cases_keep_extras
from core.round23_4.contract import INNER_LOOKBACK_MONTHS
from core.round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD


def _fit_cutoff(block_start: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(block_start) - pd.Timedelta(days=1)


def _slice_train(frame: pd.DataFrame, label_end: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(frame["trade_date"])
    end = pd.to_datetime(frame[label_end])
    start = cutoff - pd.DateOffset(months=INNER_LOOKBACK_MONTHS)
    mask = dates.ge(start) & dates.le(cutoff) & end.le(cutoff)
    train = frame.loc[mask].copy()
    if train.empty:
        return train
    sizes = train.groupby("trade_date")["instrument_id"].transform("size")
    return train.loc[sizes.ge(2)].sort_values(KEYS).reset_index(drop=True)


def _predict_lgbm(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, n_bins: int, eval_at: list[int], truncation: int) -> np.ndarray:
    import lightgbm as lgb

    from core.round17.train import ensure_stock_src_on_path

    ensure_stock_src_on_path()
    from first_board_hermes.baseline import daily_relevance

    if not ranking_groups_contiguous(train["trade_date"]):
        raise RuntimeError("LGBM groups are not contiguous")
    relevance = daily_relevance(train, target=target, n_bins=n_bins)
    group = train.groupby("trade_date", sort=False).size().to_numpy()
    model = lgb.LGBMRanker(
        objective="lambdarank",
        learning_rate=0.03,
        n_estimators=200,
        num_leaves=7,
        max_depth=3,
        min_data_in_leaf=50,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        verbosity=-1,
        seed=17,
        eval_at=list(eval_at),
        lambdarank_truncation_level=int(truncation),
        n_jobs=2,
    )
    model.fit(train[features].to_numpy(dtype=np.float32), relevance.to_numpy(dtype=np.int32), group=group)
    return model.predict(test[features].to_numpy(dtype=np.float32))


def _predict_xgb(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, n_bins: int, eval_metric: str) -> np.ndarray:
    return xgb_rank_fold_predict(
        train,
        test,
        features,
        target=target,
        seed=17,
        n_bins=n_bins,
        eval_metric=eval_metric,
        n_jobs=2,
    )


def train_inner_block(
    *,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    block_start: pd.Timestamp,
    block_end: pd.Timestamp,
    head: str,
    output_dir: Path,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    pred_path = output_dir / "predictions.parquet"
    meta_path = output_dir / "train_meta.json"
    spec = FIVE_HEAD if head == "five" else TEN_HEAD
    features = list(FIVE_FEATURES if head == "five" else TEN_FEATURES)
    truncation = 4 if head == "five" else 5
    eval_at = [1] if head == "five" else [2]
    cutoff = _fit_cutoff(block_start)
    fingerprint = {
        "head": head,
        "features": features,
        "target": spec.target,
        "label_end": spec.label_end_column,
        "cutoff": str(cutoff.date()),
        "block_start": str(pd.Timestamp(block_start).date()),
        "block_end": str(pd.Timestamp(block_end).date()),
        "lookback_months": INNER_LOOKBACK_MONTHS,
        "seed": 17,
        "truncation": truncation,
    }
    if resume and pred_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint and meta.get("status") == "trained":
            return meta
    train_raw = _prepare(training)
    cand_raw = _prepare(candidates)
    train_clean, train_audit = complete_cases_keep_extras(train_raw, features, [], require_target=True, target=spec.target)
    cand_clean, cand_audit = complete_cases_keep_extras(cand_raw, features, [], require_target=False, target=spec.target)
    train = _slice_train(train_clean, spec.label_end_column, cutoff)
    test = cand_clean.loc[cand_clean["trade_date"].between(pd.Timestamp(block_start), pd.Timestamp(block_end))].sort_values(KEYS)
    payload: dict[str, Any] = {
        "fingerprint": fingerprint,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_audit": train_audit,
        "candidate_audit": cand_audit,
        "label_contract": "LEGACY_OPEN_INFO_MATURITY",
        "fit_cutoff": str(cutoff.date()),
        "saw_block_labels": False,
    }
    if train.empty or test.empty or train["trade_date"].nunique() < 5:
        payload["status"] = "EMPTY"
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["signal_date", "instrument_id", "score", "model"]).to_parquet(pred_path, index=False)
        dump_json(meta_path, payload)
        return payload
    xgb_score = _predict_xgb(train, test, features, spec.target, spec.rank_n_bins, spec.eval_metric)
    lgb_score = _predict_lgbm(train, test, features, spec.target, spec.rank_n_bins, eval_at, truncation)
    base = test[["trade_date", "instrument_id"]].copy().rename(columns={"trade_date": "signal_date"})
    xgb = base.copy()
    xgb["score"] = xgb_score
    xgb["model"] = "xgb"
    lgb = base.copy()
    lgb["score"] = lgb_score
    lgb["model"] = "lgbm"
    pred = pd.concat([xgb, lgb], ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(pred_path, index=False)
    payload["status"] = "trained"
    payload["max_label_end"] = str(pd.to_datetime(train[spec.label_end_column]).max().date())
    dump_json(meta_path, payload)
    print(
        f"[r234-inner] {head} {block_start.date()}..{block_end.date()} cutoff={cutoff.date()} train={len(train)} test={len(test)}",
        flush=True,
    )
    return payload
