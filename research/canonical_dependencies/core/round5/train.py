"""本地训练封装：只读 import Stock RankerConfig / rolling_train_predict。

新 XGB 强制 subsample=1.0、colsample_bytree=1.0。产物只写 result/round5/models|predictions/。
不改 Stock 已训完的 fs00263。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..paths import (
    FS00263_FEATURES,
    PREDICTION_START,
    SIGNAL_END,
    assert_project_result_path,
    ensure_round5_tree,
)
from ..stock_readonly import CANDIDATE_PANEL_RELATIVE, ensure_stock_src_on_path, stock_path
from .registry import LOCKED_FEATURES, TRAIN_SEEDS

LABEL_END = "actual_exit_date"
DEFAULT_TARGET = "label_e2_net"


def _ranker_mod():
    ensure_stock_src_on_path()
    from first_board_hermes.baseline import (
        ALL_FEATURES,
        RankerConfig,
        daily_relevance,
        in_memory_complete_cases,
        mean_daily_ndcg,
        rolling_train_predict,
    )
    return {
        "ALL_FEATURES": ALL_FEATURES,
        "RankerConfig": RankerConfig,
        "daily_relevance": daily_relevance,
        "in_memory_complete_cases": in_memory_complete_cases,
        "mean_daily_ndcg": mean_daily_ndcg,
        "rolling_train_predict": rolling_train_predict,
    }


def ranker_config_r5(**overrides):
    """Stock RankerConfig，强制关闭行/列采样。"""
    api = _ranker_mod()
    RankerConfig = api["RankerConfig"]
    params = dict(
        objective="rank:ndcg",
        eval_metric="ndcg@3",
        ndcg_exp_gain=True,
        rank_n_bins=5,
        n_estimators=50,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )
    params.update(overrides)
    params["subsample"] = 1.0
    params["colsample_bytree"] = 1.0
    allowed = set(RankerConfig.__dataclass_fields__)
    return RankerConfig(**{k: v for k, v in params.items() if k in allowed})


def pred_dir(key: str) -> Path:
    path = ensure_round5_tree() / "predictions" / str(key)
    path.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    return path


def model_dir(key: str) -> Path:
    path = ensure_round5_tree() / "models" / str(key)
    path.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    return path


def load_predictions(key: str) -> pd.DataFrame:
    path = pred_dir(key) / "predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing local predictions: {path}")
    frame = pd.read_parquet(path)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    if "instrument_id" in frame:
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    return frame.loc[frame["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)].copy()


def _load_panel(features: Sequence[str], target: str) -> pd.DataFrame:
    from .factors import load_training_panel

    return load_training_panel(features=list(features), target=target)


def _first_predict_year(panel: pd.DataFrame, features: Sequence[str]) -> int:
    """满 3 个完整年 → 下一年。新因子覆盖晚则推迟，禁止未来填 2018。"""
    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    needed = [c for c in features if c in work.columns]
    if not needed:
        return 2018
    cov = work.dropna(subset=needed)
    if cov.empty:
        return 2018
    years = sorted(cov["trade_date"].dt.year.unique())
    complete = [y for y in years if y < 2018 or True]
    # need 3 complete calendar years before first predict year
    first_data = int(min(years))
    first_pred = max(2018, first_data + 3)
    return int(first_pred)


def _huber_supported() -> bool:
    try:
        from xgboost import XGBRegressor

        XGBRegressor(objective="reg:pseudohubererror", n_estimators=1, max_depth=1)
        return True
    except Exception:
        return False


def ensemble_rank_pct(seed_frames: list[pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for i, frame in enumerate(seed_frames):
        tmp = frame[["signal_date", "instrument_id", "score"]].copy()
        tmp["signal_date"] = pd.to_datetime(tmp["signal_date"])
        tmp["instrument_id"] = tmp["instrument_id"].astype(str)
        tmp["rank_pct"] = tmp.groupby("signal_date")["score"].rank(pct=True)
        parts.append(tmp.rename(columns={"rank_pct": f"r{i}"})[["signal_date", "instrument_id", f"r{i}"]])
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["signal_date", "instrument_id"], how="outer")
    rank_cols = [c for c in out.columns if c.startswith("r")]
    out["score"] = out[rank_cols].mean(axis=1)
    return out[["signal_date", "instrument_id", "score"]].sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    ).reset_index(drop=True)


def _local_rolling(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    output_dir: Path,
    *,
    features: list[str],
    target: str,
    config,
    first_prediction_year: int,
    last_prediction_year: int = 2026,
    label_end_column: str = LABEL_END,
    mode: str = "ranker",
    invert_score: bool = False,
    save_models: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Walk-forward：满 3 个完整年训、下一年预测。允许新因子（不走 ALL_FEATURES 白名单）。"""
    api = _ranker_mod()
    daily_relevance = api["daily_relevance"]
    in_memory_complete_cases = api["in_memory_complete_cases"]
    mean_daily_ndcg = api["mean_daily_ndcg"]

    from xgboost import XGBClassifier, XGBRanker, XGBRegressor

    output_dir.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(output_dir)
    mdir = output_dir / "models"
    if save_models:
        mdir.mkdir(exist_ok=True)

    train_panel, train_audit = in_memory_complete_cases(
        training, features, require_target=True, target=target
    )
    cand_panel, cand_audit = in_memory_complete_cases(
        candidates, features, require_target=False, target=target
    )
    if label_end_column not in train_panel.columns:
        raise ValueError(f"training panel missing {label_end_column}")
    train_panel[label_end_column] = pd.to_datetime(train_panel[label_end_column])
    train_panel["trade_date"] = pd.to_datetime(train_panel["trade_date"])
    cand_panel["trade_date"] = pd.to_datetime(cand_panel["trade_date"])

    params = asdict(config)
    rank_n_bins = params.pop("rank_n_bins", 5)
    params.pop("absolute_return_thresholds", None)
    params.pop("hybrid_loss_floor", None)
    params.pop("hybrid_tail_threshold", None)
    params.pop("precomputed_relevance", None)
    params.pop("early_stopping_rounds", None)
    params.pop("validation_sessions", None)
    params.pop("ndcg_exp_gain", None)
    params["subsample"] = 1.0
    params["colsample_bytree"] = 1.0

    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for predict_year in range(first_prediction_year, last_prediction_year + 1):
        train_start, train_end = predict_year - 3, predict_year - 1
        cutoff = pd.Timestamp(predict_year, 1, 1)
        train_mask = (
            train_panel["trade_date"].dt.year.between(train_start, train_end)
            & train_panel[label_end_column].lt(cutoff)
        )
        train = train_panel.loc[train_mask].copy()
        if "trade_date" in train:
            sizes = train.groupby("trade_date")["instrument_id"].transform("size")
            train = train.loc[sizes.ge(2)].sort_values(["trade_date", "instrument_id"]).reset_index(drop=True)
        test = cand_panel.loc[cand_panel["trade_date"].dt.year.eq(predict_year)].copy()
        if train.empty or test.empty:
            folds.append({
                "predict_year": predict_year,
                "status": "empty_fold",
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            })
            continue
        x_train = train[features].to_numpy(dtype=np.float32)
        x_test = test[features].to_numpy(dtype=np.float32)
        if mode == "ranker":
            relevance = daily_relevance(train, target=target, n_bins=None if rank_n_bins is None else int(rank_n_bins))
            model = XGBRanker(**{k: v for k, v in params.items() if k in XGBRanker().get_params()})
            model.fit(
                x_train,
                relevance.to_numpy(dtype=np.int32),
                qid=train["trade_date"].factorize(sort=True)[0],
                verbose=False,
            )
            score = model.predict(x_test)
        elif mode == "classifier":
            y = pd.to_numeric(train[target], errors="coerce").fillna(0).astype(int)
            clf_params = {k: v for k, v in params.items() if k not in {"objective", "eval_metric", "rank_n_bins"}}
            clf_params["objective"] = "binary:logistic"
            clf_params["eval_metric"] = "logloss"
            clf_params["subsample"] = 1.0
            clf_params["colsample_bytree"] = 1.0
            model = XGBClassifier(**{k: v for k, v in clf_params.items() if k in XGBClassifier().get_params()})
            model.fit(x_train, y.to_numpy(), verbose=False)
            score = model.predict_proba(x_test)[:, 1]
        else:
            y = pd.to_numeric(train[target], errors="coerce")
            mask = y.notna()
            reg_params = {k: v for k, v in params.items() if k not in {"rank_n_bins", "eval_metric", "ndcg_exp_gain"}}
            reg_params["subsample"] = 1.0
            reg_params["colsample_bytree"] = 1.0
            model = XGBRegressor(**{k: v for k, v in reg_params.items() if k in XGBRegressor().get_params()})
            model.fit(x_train[mask.to_numpy()], y[mask].to_numpy(dtype=np.float32), verbose=False)
            score = model.predict(x_test)
        if invert_score:
            score = 1.0 - np.asarray(score, dtype=float)
        pred = test[["trade_date", "instrument_id"]].copy()
        pred = pred.rename(columns={"trade_date": "signal_date"})
        pred["score"] = score
        pred["predict_year"] = predict_year
        folds.append({
            "predict_year": predict_year,
            "train_years": [train_start, train_end],
            "train_rows": int(len(train)),
            "candidate_rows": int(len(pred)),
            "label_end_lt_predict_year": True,
        })
        if save_models:
            path = mdir / f"xgb_{train_start}_{train_end}_to_{predict_year}.json"
            model.save_model(path)
        predictions.append(pred)

    if not predictions:
        raise RuntimeError("no walk-forward predictions produced")
    result = pd.concat(predictions, ignore_index=True).sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    out_path = output_dir / "predictions.parquet"
    assert_project_result_path(out_path)
    result.to_parquet(out_path, index=False)
    metadata = {
        "status": "trained_oos",
        "walk_forward": "exact prior 3 calendar years; label_end < predict_year-01-01",
        "target": target,
        "label_end_column": label_end_column,
        "features": list(features),
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "mode": mode,
        "invert_score": invert_score,
        "folds": folds,
        "training_clean_audit": train_audit,
        "candidate_clean_audit": cand_audit,
        "prediction_rows": len(result),
        "written_at": datetime.now().astimezone().isoformat(),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _ = mean_daily_ndcg
    return result, metadata


def train_key(
    key: str,
    features: Sequence[str],
    *,
    target: str = DEFAULT_TARGET,
    objective: str = "rank:ndcg",
    eval_metric: str = "ndcg@3",
    n_estimators: int = 50,
    max_depth: int = 2,
    learning_rate: float = 0.05,
    seeds: Sequence[int] = TRAIN_SEEDS,
    resume: bool = True,
    invert_score: bool = False,
) -> dict[str, Any]:
    """5 种子 walk-forward + rank_pct 集成。"""
    dest = pred_dir(key)
    ens_path = dest / "predictions.parquet"
    meta_path = dest / "ensemble.json"
    if resume and ens_path.is_file() and meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    if objective == "reg:pseudohubererror" and not _huber_supported():
        payload = {
            "status": "SKIPPED_UNSUPPORTED",
            "key": key,
            "reason": "xgboost 无原生 Huber / reg:pseudohubererror",
        }
        dest.joinpath("ensemble.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return payload

    feats = list(features)
    panel = _load_panel(feats, target)
    first_year = _first_predict_year(panel, feats)
    api = _ranker_mod()
    ALL_FEATURES = set(api["ALL_FEATURES"])
    use_stock = (
        objective == "rank:ndcg"
        and set(feats).issubset(ALL_FEATURES)
        and target in panel.columns
    )
    mode = "ranker"
    if str(objective).startswith("reg:"):
        mode = "regressor"
        use_stock = False
    elif str(objective).startswith("binary:") or str(objective) == "binary:logistic":
        mode = "classifier"
        use_stock = False
    elif objective == "rank:pairwise":
        mode = "ranker"
        use_stock = False

    seed_frames = []
    seed_meta = []
    for seed in seeds:
        cfg = ranker_config_r5(
            objective=objective if mode == "ranker" or mode == "regressor" else "rank:ndcg",
            eval_metric=eval_metric if str(eval_metric).startswith("ndcg@") else "ndcg@3",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=int(seed),
        )
        if mode == "regressor":
            cfg = ranker_config_r5(
                objective=objective,
                eval_metric="ndcg@3",
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=int(seed),
            )
        out = model_dir(key) / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        assert_project_result_path(out)
        seed_pred_path = out / "predictions.parquet"
        if resume and seed_pred_path.is_file():
            seed_frames.append(pd.read_parquet(seed_pred_path))
            seed_meta.append({"seed": seed, "status": "resumed"})
            continue
        if use_stock:
            preds, meta = api["rolling_train_predict"](
                panel,
                panel,
                out,
                features=feats,
                first_prediction_year=first_year,
                last_prediction_year=2026,
                config=cfg,
                target=target,
                label_end_column=LABEL_END if LABEL_END in panel.columns else "next2_date",
                save_models=True,
            )
        else:
            preds, meta = _local_rolling(
                panel,
                panel,
                out,
                features=feats,
                target=target,
                config=cfg,
                first_prediction_year=first_year,
                last_prediction_year=2026,
                label_end_column=LABEL_END if LABEL_END in panel.columns else "next2_date",
                mode=mode,
                invert_score=invert_score,
            )
        seed_frames.append(preds)
        seed_meta.append({"seed": seed, "status": meta.get("status"), "folds": len(meta.get("folds") or [])})

    ensemble = ensemble_rank_pct(seed_frames)
    assert_project_result_path(ens_path)
    ensemble.to_parquet(ens_path, index=False)
    payload = {
        "status": "trained_oos",
        "key": key,
        "features": feats,
        "target": target,
        "objective": objective,
        "eval_metric": eval_metric,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "seeds": list(seeds),
        "first_prediction_year": first_year,
        "used_stock_rolling_train_predict": use_stock,
        "mode": mode,
        "invert_score": invert_score,
        "n_rows": int(len(ensemble)),
        "n_days": int(ensemble["signal_date"].nunique()),
        "seed_meta": seed_meta,
        "written_at": datetime.now().astimezone().isoformat(),
        "feature_sha256": hashlib.sha256(json.dumps(feats).encode()).hexdigest(),
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def train_spec(spec, *, resume: bool = True) -> dict[str, Any]:
    invert = spec.target == "label_bigloss_5"
    return train_key(
        spec.prediction_key or spec.code,
        spec.features or LOCKED_FEATURES,
        target=spec.target,
        objective=spec.objective,
        eval_metric=spec.eval_metric,
        n_estimators=spec.n_estimators,
        max_depth=spec.max_depth,
        learning_rate=spec.learning_rate,
        resume=resume,
        invert_score=invert,
    )


def apply_family_shortlist(spec, extra_features: Sequence[str]):
    from .registry import clone_spec

    feats = tuple(dict.fromkeys([*(spec.features or LOCKED_FEATURES), *extra_features]))
    return clone_spec(spec, features=feats)
