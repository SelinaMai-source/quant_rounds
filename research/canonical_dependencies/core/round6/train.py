"""本地训练：默认 target=label_exit_rule_net。B–K 强制 subsample=1.0。

A04 才允许 0.8（FORENSIC_ONLY）。canonical 走 Stock rolling_train_predict。
产物只写 result/round6/models|predictions/。不改 Stock。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..paths import (
    CANDIDATE_PANEL_RELATIVE,
    PREDICTION_START,
    SIGNAL_END,
    assert_project_result_path,
    ensure_round6_tree,
    stock_path,
)
from ..round5.factors import load_training_panel
from ..stock_readonly import ensure_stock_src_on_path, sha256_file
from .registry import (
    DEFAULT_TARGET,
    HYBRID_PCT_FEATURES,
    LOCKED_FEATURES,
    TRAIN_SEEDS,
    r5_pred_dir,
)

LABEL_END = "actual_exit_date"
TRAINING_PANEL_RELATIVE = Path(
    "data/first_board_hermes/"
    "aligned_original_candidate_full44_2014warmup_2018oos_v1/training_panel.parquet"
)


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


def xgb_ranker_weight_kind() -> str:
    """本机 XGB LTR：qid 存在时 sample_weight 必须按 query group，不能按行。

    近因只能挂在 trade_date 上；同一日组内权重相同，再收成一组一个权重。
    """
    from xgboost import XGBRanker

    _ = XGBRanker().get_params()
    return "group"


def ranker_config_r6(
    *,
    forensic_only: bool = False,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    **overrides,
):
    api = _ranker_mod()
    RankerConfig = api["RankerConfig"]
    if not forensic_only:
        subsample = 1.0
        colsample_bytree = 1.0
    params = dict(
        objective="rank:ndcg",
        eval_metric="ndcg@3",
        ndcg_exp_gain=True,
        rank_n_bins=5,
        n_estimators=50,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=1.0,
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )
    params.update(overrides)
    if not forensic_only:
        params["subsample"] = 1.0
        params["colsample_bytree"] = 1.0
    allowed = set(RankerConfig.__dataclass_fields__)
    return RankerConfig(**{k: v for k, v in params.items() if k in allowed})


def pred_dir(key: str) -> Path:
    path = ensure_round6_tree() / "predictions" / str(key)
    path.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    return path


def model_dir(key: str) -> Path:
    path = ensure_round6_tree() / "models" / str(key)
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


def training_panel_path() -> Path:
    return stock_path(TRAINING_PANEL_RELATIVE)


def training_panel_sha256() -> str:
    return sha256_file(training_panel_path())


def load_stock_training_panel(features: Sequence[str], target: str) -> pd.DataFrame:
    path = training_panel_path()
    frame = pd.read_parquet(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    extra = [
        "trade_date", "instrument_id", target, "actual_exit_date", "next2_date",
        "label_e2_net", "label_exit_rule_net",
    ]
    cols = []
    for c in list(features) + extra:
        if c in frame.columns and c not in cols:
            cols.append(c)
    return frame[cols].copy()


def load_panel(features: Sequence[str], target: str, *, source: str = "current") -> pd.DataFrame:
    if source == "frozen_training_panel":
        return load_stock_training_panel(features, target)
    return load_training_panel(features=list(features), target=target)


def apply_representation(
    panel: pd.DataFrame,
    features: Sequence[str],
    kind: str,
) -> pd.DataFrame:
    """只按同一 trade_date 变换。候选数 <3 用 deterministic fallback，不删日。"""
    if kind in (None, "", "raw"):
        return panel
    out = panel.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    gsize = out.groupby("trade_date")["instrument_id"].transform("size")
    small = gsize.lt(3)

    def _pct(col: str) -> pd.Series:
        x = pd.to_numeric(out[col], errors="coerce")
        pct = x.groupby(out["trade_date"]).rank(pct=True)
        return pct.where(~small, 0.5)

    def _robust(col: str) -> pd.Series:
        x = pd.to_numeric(out[col], errors="coerce")
        med = x.groupby(out["trade_date"]).transform("median")
        q1 = x.groupby(out["trade_date"]).transform(lambda s: s.quantile(0.25))
        q3 = x.groupby(out["trade_date"]).transform(lambda s: s.quantile(0.75))
        iqr = (q3 - q1).replace(0, np.nan)
        z = (x - med) / iqr
        return z.fillna(0.0).where(~small, 0.0)

    for col in features:
        if col not in out.columns:
            continue
        if kind == "pct":
            out[col] = _pct(col)
        elif kind == "robust_z":
            out[col] = _robust(col)
        elif kind == "hybrid":
            out[col] = _pct(col) if col in HYBRID_PCT_FEATURES else _robust(col)
        else:
            raise ValueError(f"unknown representation {kind!r}")
    return out


def apply_label_transform(panel: pd.DataFrame, *, kind: str, target: str) -> tuple[pd.DataFrame, str]:
    """训练标签变换。成熟时点仍走 actual_exit_date。"""
    out = panel.copy()
    if kind in (None, "", "none"):
        return out, target
    y = pd.to_numeric(out[target], errors="coerce")
    if kind == "ordinal04":
        api = _ranker_mod()
        out["label_ordinal_0_4"] = api["daily_relevance"](out, target=target, n_bins=5)
        return out, "label_ordinal_0_4"
    if kind == "clip_rank":
        clipped = y.clip(-0.10, 0.20)
        tmp = out.copy()
        tmp["_clip"] = clipped
        api = _ranker_mod()
        out["label_clip_rank"] = api["daily_relevance"](tmp, target="_clip", n_bins=5)
        return out, "label_clip_rank"
    if kind == "top1_binary":
        ranks = y.groupby(out["trade_date"]).rank(method="first", ascending=False)
        out["label_top1_binary"] = (ranks == 1).astype(float)
        out.loc[y.isna(), "label_top1_binary"] = np.nan
        return out, "label_top1_binary"
    raise ValueError(f"unknown label_transform {kind!r}")


def window_bounds(predict_year: int, window_kind: str, first_train_year: int = 2015) -> tuple[int, int]:
    train_end = predict_year - 1
    if window_kind == "2y":
        return predict_year - 2, train_end
    if window_kind == "5y":
        return predict_year - 5, train_end
    if window_kind == "expanding":
        return int(first_train_year), train_end
    return predict_year - 3, train_end  # 3y default


def fold_mask(
    panel: pd.DataFrame,
    *,
    train_start: int,
    train_end: int,
    predict_year: int,
    fold_rule: str,
    label_end_column: str = LABEL_END,
) -> pd.Series:
    dates = pd.to_datetime(panel["trade_date"])
    exits = pd.to_datetime(panel[label_end_column])
    year_ok = dates.dt.year.between(train_start, train_end)
    if fold_rule == "r5_local":
        label_ok = exits.lt(pd.Timestamp(predict_year, 1, 1))
    else:
        label_ok = exits.dt.year.le(train_end)
    # 硬规则：标签必须在预测日之前成熟
    pred_day = pd.Timestamp(predict_year, 1, 1)
    mature = exits.lt(pred_day)
    return year_ok & label_ok & mature


def recency_row_weights(train: pd.DataFrame, train_end: int, half_life_months: int) -> np.ndarray:
    """只按 trade_date 近因。XGB ranking 权重是 row/instance，不是 group。"""
    end = pd.Timestamp(train_end, 12, 31)
    days = (end - pd.to_datetime(train["trade_date"])).dt.days.clip(lower=0).to_numpy(dtype=float)
    half_days = float(half_life_months) * 30.4375
    return np.power(0.5, days / half_days)


def ensemble_rank_pct(seed_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """日内 pct 等权。对齐 Stock ensemble_report.json method。"""
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
    window_kind: str = "3y",
    fold_rule: str = "stock",
    half_life_months: int | None = None,
    forensic_only: bool = False,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    api = _ranker_mod()
    daily_relevance = api["daily_relevance"]
    in_memory_complete_cases = api["in_memory_complete_cases"]

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
    first_train_year = int(train_panel["trade_date"].dt.year.min())

    params = asdict(config)
    rank_n_bins = params.pop("rank_n_bins", 5)
    for drop_k in (
        "absolute_return_thresholds", "hybrid_loss_floor", "hybrid_tail_threshold",
        "precomputed_relevance", "early_stopping_rounds", "validation_sessions", "ndcg_exp_gain",
    ):
        params.pop(drop_k, None)
    if forensic_only:
        params["subsample"] = float(subsample)
        params["colsample_bytree"] = float(colsample_bytree)
    else:
        params["subsample"] = 1.0
        params["colsample_bytree"] = 1.0

    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for predict_year in range(first_prediction_year, last_prediction_year + 1):
        train_start, train_end = window_bounds(predict_year, window_kind, first_train_year)
        train_mask = fold_mask(
            train_panel,
            train_start=train_start,
            train_end=train_end,
            predict_year=predict_year,
            fold_rule=fold_rule,
            label_end_column=label_end_column,
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
        fit_w = None
        if half_life_months:
            fit_w = recency_row_weights(train, train_end, int(half_life_months))
        if mode == "ranker":
            relevance = daily_relevance(train, target=target, n_bins=None if rank_n_bins is None else int(rank_n_bins))
            model = XGBRanker(**{k: v for k, v in params.items() if k in XGBRanker().get_params()})
            qid = train["trade_date"].factorize(sort=True)[0]
            kw = {"qid": qid, "verbose": False}
            if fit_w is not None:
                # LTR 的 sample_weight = 每个 query group 一个权重；同日近因相同。
                group_w = (
                    pd.Series(fit_w)
                    .groupby(pd.Series(qid), sort=True)
                    .first()
                    .to_numpy(dtype=float)
                )
                kw["sample_weight"] = group_w
            model.fit(x_train, relevance.to_numpy(dtype=np.int32), **kw)
            score = model.predict(x_test)
        elif mode == "classifier":
            y = pd.to_numeric(train[target], errors="coerce").fillna(0).astype(int)
            clf_params = {k: v for k, v in params.items() if k not in {"objective", "eval_metric", "rank_n_bins"}}
            clf_params["objective"] = "binary:logistic"
            clf_params["eval_metric"] = "logloss"
            if not forensic_only:
                clf_params["subsample"] = 1.0
                clf_params["colsample_bytree"] = 1.0
            model = XGBClassifier(**{k: v for k, v in clf_params.items() if k in XGBClassifier().get_params()})
            kw = {"verbose": False}
            if fit_w is not None:
                kw["sample_weight"] = fit_w
            model.fit(x_train, y.to_numpy(), **kw)
            score = model.predict_proba(x_test)[:, 1]
        else:
            y = pd.to_numeric(train[target], errors="coerce")
            mask = y.notna()
            reg_params = {k: v for k, v in params.items() if k not in {"rank_n_bins", "eval_metric", "ndcg_exp_gain"}}
            if not forensic_only:
                reg_params["subsample"] = 1.0
                reg_params["colsample_bytree"] = 1.0
            model = XGBRegressor(**{k: v for k, v in reg_params.items() if k in XGBRegressor().get_params()})
            kw = {"verbose": False}
            if fit_w is not None:
                kw["sample_weight"] = fit_w[mask.to_numpy()]
            model.fit(x_train[mask.to_numpy()], y[mask].to_numpy(dtype=np.float32), **kw)
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
            "fold_rule": fold_rule,
            "window_kind": window_kind,
            "label_end_lt_predict_year": True,
            "half_life_months": half_life_months,
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
        "walk_forward": f"{window_kind}; fold={fold_rule}; label mature < predict_year-01-01",
        "target": target,
        "label_end_column": label_end_column,
        "features": list(features),
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "forensic_only": forensic_only,
        "mode": mode,
        "invert_score": invert_score,
        "window_kind": window_kind,
        "fold_rule": fold_rule,
        "half_life_months": half_life_months,
        "xgb_ranker_weight_kind": xgb_ranker_weight_kind(),
        "folds": folds,
        "training_clean_audit": train_audit,
        "candidate_clean_audit": cand_audit,
        "prediction_rows": len(result),
        "written_at": datetime.now().astimezone().isoformat(),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return result, metadata


def copy_round5_predictions(r5_key: str, dest_key: str) -> dict[str, Any]:
    src = r5_pred_dir(r5_key)
    src_pred = src / "predictions.parquet"
    if not src_pred.is_file():
        raise FileNotFoundError(f"missing R5 predictions to reuse: {src_pred}")
    dest = pred_dir(dest_key)
    shutil.copy2(src_pred, dest / "predictions.parquet")
    src_ens = src / "ensemble.json"
    payload = json.loads(src_ens.read_text(encoding="utf-8")) if src_ens.is_file() else {}
    payload = {
        **payload,
        "reused_from_round5": r5_key,
        "key": dest_key,
        "written_at": datetime.now().astimezone().isoformat(),
        "note": "同配方复用 R5 预测，回测仍在 round6 重跑。",
    }
    (dest / "ensemble.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return payload


def filter_frozen_universe(panel: pd.DataFrame) -> pd.DataFrame:
    from ..stock_readonly import load_fs00263_ensemble_predictions

    locked = load_fs00263_ensemble_predictions()
    keys = locked[["signal_date", "instrument_id"]].drop_duplicates().rename(
        columns={"signal_date": "trade_date"}
    )
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    keys["instrument_id"] = keys["instrument_id"].astype(str)
    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    # 训练年保留全样本；预测年只评冻结宇宙。walk-forward 里 candidates 再滤。
    return work


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
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    forensic_only: bool = False,
    window_kind: str = "3y",
    fold_rule: str = "stock",
    representation: str = "raw",
    half_life_months: int | None = None,
    label_transform: str = "none",
    use_stock_rolling: bool | None = None,
    panel_source: str = "current",
    freeze_universe: bool = False,
    reuse_round5_key: str | None = None,
    context_features: Sequence[str] = (),
) -> dict[str, Any]:
    dest = pred_dir(key)
    ens_path = dest / "predictions.parquet"
    meta_path = dest / "ensemble.json"
    if resume and ens_path.is_file() and meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    if reuse_round5_key and resume:
        r5 = r5_pred_dir(reuse_round5_key) / "predictions.parquet"
        if r5.is_file():
            return copy_round5_predictions(reuse_round5_key, key)

    if forensic_only and (float(subsample) != 1.0 or float(colsample_bytree) != 1.0):
        if float(subsample) != 0.8 or float(colsample_bytree) != 0.8:
            raise ValueError("FORENSIC_ONLY 只允许历史 0.8/0.8")
    elif float(subsample) != 1.0 or float(colsample_bytree) != 1.0:
        raise ValueError("B–K / 非 forensic 必须 subsample=colsample=1.0")

    feats = list(features) + [c for c in context_features if c not in features]
    panel = load_panel(feats, target, source=panel_source)
    panel, train_target = apply_label_transform(panel, kind=label_transform, target=target)
    panel = apply_representation(panel, feats, representation)
    if freeze_universe:
        locked = __import__("core.stock_readonly", fromlist=["load_fs00263_ensemble_predictions"])
        from ..stock_readonly import load_fs00263_ensemble_predictions

        keys = load_fs00263_ensemble_predictions()[["signal_date", "instrument_id"]].copy()
        keys["signal_date"] = pd.to_datetime(keys["signal_date"])
        # candidates later filtered; keep full history for training
        cand_keys = keys.rename(columns={"signal_date": "trade_date"})
        cand_keys["instrument_id"] = cand_keys["instrument_id"].astype(str)
    else:
        cand_keys = None

    first_year = 2018
    api = _ranker_mod()
    ALL_FEATURES = set(api["ALL_FEATURES"])
    can_stock = (
        use_stock_rolling is not False
        and objective == "rank:ndcg"
        and representation == "raw"
        and window_kind == "3y"
        and fold_rule == "stock"
        and not half_life_months
        and label_transform in (None, "", "none")
        and set(feats).issubset(ALL_FEATURES)
        and not context_features
    )
    if use_stock_rolling is True:
        can_stock = can_stock or (
            objective == "rank:ndcg"
            and representation == "raw"
            and window_kind == "3y"
            and fold_rule == "stock"
            and not half_life_months
            and label_transform in (None, "", "none")
            and set(feats).issubset(ALL_FEATURES)
        )
    if use_stock_rolling is False:
        can_stock = False
    mode = "ranker"
    if str(objective).startswith("reg:"):
        mode = "regressor"
        can_stock = False
    elif str(objective).startswith("binary:") or str(objective) == "binary:logistic":
        mode = "classifier"
        can_stock = False
    elif objective == "rank:pairwise":
        mode = "ranker"
        can_stock = False

    seed_frames = []
    seed_meta = []
    candidates = panel
    if cand_keys is not None:
        work = panel.copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"])
        work["instrument_id"] = work["instrument_id"].astype(str)
        oos = work.merge(cand_keys, on=["trade_date", "instrument_id"], how="inner")
        hist = work.loc[work["trade_date"].dt.year < 2018]
        candidates = pd.concat([hist, oos], ignore_index=True).drop_duplicates(
            ["trade_date", "instrument_id"]
        )

    for seed in seeds:
        cfg = ranker_config_r6(
            forensic_only=forensic_only,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective if mode in {"ranker", "regressor"} else "rank:ndcg",
            eval_metric=eval_metric if str(eval_metric).startswith("ndcg@") else "ndcg@3",
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
        if can_stock:
            preds, meta = api["rolling_train_predict"](
                panel,
                candidates,
                out,
                features=feats,
                first_prediction_year=first_year,
                last_prediction_year=2026,
                config=cfg,
                target=train_target,
                label_end_column=LABEL_END if LABEL_END in panel.columns else "next2_date",
                save_models=True,
            )
        else:
            preds, meta = _local_rolling(
                panel,
                candidates,
                out,
                features=feats,
                target=train_target,
                config=cfg,
                first_prediction_year=first_year,
                last_prediction_year=2026,
                label_end_column=LABEL_END if LABEL_END in panel.columns else "next2_date",
                mode=mode,
                invert_score=invert_score,
                window_kind=window_kind,
                fold_rule=fold_rule,
                half_life_months=half_life_months,
                forensic_only=forensic_only,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
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
        "target": train_target,
        "source_target": target,
        "objective": objective,
        "eval_metric": eval_metric,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8 if forensic_only else 1.0,
        "colsample_bytree": 0.8 if forensic_only else 1.0,
        "forensic_only": forensic_only,
        "seeds": list(seeds),
        "first_prediction_year": first_year,
        "used_stock_rolling_train_predict": can_stock,
        "mode": mode,
        "window_kind": window_kind,
        "fold_rule": fold_rule,
        "representation": representation,
        "half_life_months": half_life_months,
        "label_transform": label_transform,
        "panel_source": panel_source,
        "freeze_universe": freeze_universe,
        "xgb_ranker_weight_kind": xgb_ranker_weight_kind(),
        "invert_score": invert_score,
        "n_rows": int(len(ensemble)),
        "n_days": int(ensemble["signal_date"].nunique()),
        "seed_meta": seed_meta,
        "written_at": datetime.now().astimezone().isoformat(),
        "feature_sha256": hashlib.sha256(json.dumps(feats).encode()).hexdigest(),
        "training_panel_sha256": training_panel_sha256() if panel_source == "frozen_training_panel" else None,
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def train_spec(spec, *, resume: bool = True, **overrides) -> dict[str, Any]:
    invert = spec.target == "label_bigloss_5" or spec.label_transform == "top1_binary" and False
    kwargs = dict(
        target=spec.target,
        objective=spec.objective,
        eval_metric=spec.eval_metric,
        n_estimators=spec.n_estimators,
        max_depth=spec.max_depth,
        learning_rate=spec.learning_rate,
        seeds=spec.seeds or TRAIN_SEEDS,
        resume=resume,
        invert_score=invert,
        subsample=spec.subsample,
        colsample_bytree=spec.colsample_bytree,
        forensic_only=spec.forensic_only,
        window_kind=spec.window_kind,
        fold_rule=spec.fold_rule,
        representation=spec.representation,
        half_life_months=spec.half_life_months,
        label_transform=spec.label_transform,
        use_stock_rolling=spec.use_stock_rolling,
        reuse_round5_key=spec.reuse_round5_key,
        context_features=spec.context_features,
    )
    kwargs.update(overrides)
    return train_key(spec.prediction_key or spec.code, spec.features or LOCKED_FEATURES, **kwargs)
