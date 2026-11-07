"""Round23.2 LGBM：在新模块加入 lambdarank_truncation_level，不改旧 r225241_models。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from core.r225241_models import KEYS, _prepare, fold_slices
from core.r225241_stats import ranking_groups_contiguous
from core.research_integrity_v2.io_v2 import dump_json
from core.round22_3.train import complete_cases_keep_extras


def train_lgbm_ranker_trunc(
    *,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    features: Sequence[str],
    target: str,
    label_end_column: str,
    n_bins: int,
    eval_at: list[int],
    truncation_level: int,
    output_dir: Path,
    seed: int = 17,
    first_year: int = 2018,
    last_year: int = 2026,
    resume: bool = True,
) -> dict[str, Any]:
    import lightgbm as lgb

    from core.round17.train import ensure_stock_src_on_path

    ensure_stock_src_on_path()
    from first_board_hermes.baseline import daily_relevance

    output_dir = Path(output_dir)
    pred_path = output_dir / "predictions.parquet"
    meta_path = output_dir / "train_meta.json"
    if resume and pred_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("lambdarank_truncation_level") == int(truncation_level) and meta.get("eval_at") == list(eval_at):
            return meta
    features = list(features)
    training_clean, train_audit = complete_cases_keep_extras(_prepare(training), features, [], require_target=True, target=target)
    candidates_clean, cand_audit = complete_cases_keep_extras(_prepare(candidates), features, [], require_target=False, target=target)
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
        "lambdarank_truncation_level": int(truncation_level),
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
        model.fit(train[features].to_numpy(dtype=np.float32), relevance.to_numpy(dtype=np.int32), group=group)
        score = model.predict(test[features].to_numpy(dtype=np.float32))
        part = test[["trade_date", "instrument_id"]].copy().rename(columns={"trade_date": "signal_date"})
        part["score"] = score
        part["predict_year"] = year
        rows.append(part)
        folds.append({"predict_year": year, "train_rows": int(len(train)), "test_rows": int(len(test)), "n_groups": int(len(group))})
        print(f"[lgbm-trunc] year={year} trunc={truncation_level} train={len(train)} test={len(test)}", flush=True)
    pred = pd.concat(rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(pred_path, index=False)
    payload = {
        "status": "trained_oos",
        "model": "LGBMRanker",
        "params": params,
        "lambdarank_truncation_level": int(truncation_level),
        "eval_at": list(eval_at),
        "eval_at_is_not_truncation": True,
        "features": features,
        "target": target,
        "n_bins": int(n_bins),
        "rows": int(len(pred)),
        "folds": folds,
        "train_clean_audit": train_audit,
        "candidate_clean_audit": cand_audit,
        "filled_with_zero": False,
        "label_bins_fit_inside_training_fold": True,
    }
    dump_json(meta_path, payload)
    return payload
