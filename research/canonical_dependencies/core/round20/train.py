"""Round20 训练：train_head_open + force_retrain；M1 ElasticNet 排序代理；M2 LightGBM LambdaRank。1.0/1.0。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..paths import PREDICTION_START, SIGNAL_END, assert_project_result_path
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round7.train import assert_seed_rank_invariance
from ..round17.ensemble import ensemble_rank_pct
from ..round17.train import complete_cases, train_head_open
from ..stock_readonly import sha256_file
from .common import dump_json
from .contract import COLSAMPLE, SKIP_DEPENDENCY, SUBSAMPLE, XGB_N_JOBS

KEYS = ["trade_date", "instrument_id"]


def unique_keep(seq: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in seq:
        if name and name not in seen:
            seen.add(name)
            out.append(str(name))
    return out


def new_columns(base: Sequence[str], extras: Sequence[str]) -> list[str]:
    b = set(base)
    return [x for x in unique_keep(extras) if x not in b]


def slim_panel(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    keep = unique_keep([c for c in cols if c in frame.columns])
    return frame.loc[:, keep].copy()


def _invariance(payload: dict[str, Any]) -> dict[str, Any]:
    frames = payload.get("seed_frames") or []
    if len(frames) < 2:
        payload["rank_invariance"] = {"passed": None, "reason": "lt2_seeds"}
        return payload
    try:
        payload["rank_invariance"] = assert_seed_rank_invariance(frames)
    except RuntimeError as exc:
        payload["rank_invariance"] = {"passed": False, "error": str(exc)}
    return payload


def train_both_heads(
    *,
    five_training: pd.DataFrame,
    five_candidates: pd.DataFrame,
    ten_training: pd.DataFrame,
    ten_candidates: pd.DataFrame,
    five_extras: Sequence[str],
    ten_extras: Sequence[str],
    output_dir: Path,
    resume: bool = True,
    five_frozen: pd.DataFrame | None = None,
    ten_frozen: pd.DataFrame | None = None,
    force_retrain: bool = False,
    n_jobs: int = XGB_N_JOBS,
    five_drops: Sequence[str] = (),
    ten_drops: Sequence[str] = (),
) -> dict[str, Any]:
    """Round19 模式：空 extras 时 force_retrain 走 open path，避免 R18 frozen 短路。

    five_drops / ten_drops：从锁定 14/13 + extras 中真实删除。空 extras 且有 drop
    时必须重训，禁止走 frozen 短路。默认空元组，旧调用行为不变。
    """
    five_drop_set = {str(x) for x in five_drops if x}
    ten_drop_set = {str(x) for x in ten_drops if x}
    five_features = [x for x in unique_keep(list(FIVE_FEATURES) + list(five_extras)) if x not in five_drop_set]
    ten_features = [x for x in unique_keep(list(TEN_FEATURES) + list(ten_extras)) if x not in ten_drop_set]
    five_new = new_columns(FIVE_FEATURES, five_extras)
    ten_new = new_columns(TEN_FEATURES, ten_extras)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(output_dir)

    def _head(name: str, training, candidates, features, extras_new, frozen, head, drops):
        need_train = bool(extras_new) or force_retrain or bool(drops)
        if not need_train:
            if frozen is None:
                raise RuntimeError(f"{name} frozen predictions required when no extras and not force_retrain")
            return frozen, {
                "status": "FROZEN_10_10",
                "features": list(features) if name == "five" else list(TEN_FEATURES),
                "force_retrain": False,
                "extras": list(extras_new),
                "drops": list(drops),
                "filled_with_zero": False,
            }
        extra_cols = [head.target, head.label_end_column]
        tr = slim_panel(training, [*KEYS, *features, *extra_cols])
        ca = slim_panel(candidates, [*KEYS, *features, head.target])
        payload = train_head_open(
            training=tr,
            candidates=ca,
            features=features,
            target=head.target,
            label_end_column=head.label_end_column,
            rank_n_bins=head.rank_n_bins,
            eval_metric=head.eval_metric,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE,
            first_prediction_year=2018,
            last_prediction_year=2026,
            output_dir=output_dir / name,
            seeds=SEEDS,
            n_jobs=n_jobs,
            save_models=False,
            resume=resume,
        )
        payload["force_retrain"] = bool(force_retrain)
        payload["extras"] = list(extras_new)
        payload["drops"] = list(drops)
        payload["features"] = list(features)
        _invariance(payload)
        return payload["ensemble"], payload

    five_pred, five = _head(
        "five", five_training, five_candidates, five_features, five_new, five_frozen, FIVE_HEAD, list(five_drop_set)
    )
    ten_pred, ten = _head(
        "ten", ten_training, ten_candidates, ten_features, ten_new, ten_frozen, TEN_HEAD, list(ten_drop_set)
    )
    return {
        "five": {k: v for k, v in five.items() if k not in {"ensemble", "seed_frames"}},
        "ten": {k: v for k, v in ten.items() if k not in {"ensemble", "seed_frames"}},
        "five_pred": five_pred,
        "ten_pred": ten_pred,
        "five_new": five_new,
        "ten_new": ten_new,
        "five_features": five_features,
        "ten_features": ten_features,
        "five_drops": [x for x in unique_keep(list(five_drops)) if x],
        "ten_drops": [x for x in unique_keep(list(ten_drops)) if x],
        "model": "M0",
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE,
    }


def _rolling_linear(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    label_end_column: str,
    seed: int,
    first_prediction_year: int = 2018,
    last_prediction_year: int = 2026,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler

    features = list(features)
    training = training.copy()
    training["trade_date"] = pd.to_datetime(training["trade_date"])
    training[label_end_column] = pd.to_datetime(training[label_end_column])
    candidates = candidates.copy()
    candidates["trade_date"] = pd.to_datetime(candidates["trade_date"])
    preds: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
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
        y = train.groupby("trade_date")[target].rank(method="average", pct=True)
        scaler = StandardScaler()
        xt = scaler.fit_transform(train[features].to_numpy(dtype=np.float32))
        xe = scaler.transform(test[features].to_numpy(dtype=np.float32))
        model = ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            max_iter=8000,
            random_state=int(seed),
            fit_intercept=True,
        )
        model.fit(xt, y.to_numpy(dtype=np.float32))
        score = model.predict(xe)
        part = test[KEYS].copy()
        part = part.rename(columns={"trade_date": "signal_date"})
        part["score"] = score
        part["predict_year"] = predict_year
        preds.append(part)
        folds.append({
            "predict_year": predict_year,
            "train_years": [train_start, train_end],
            "train_rows": int(len(train)),
            "candidate_rows_scored": int(len(part)),
            "loss": "mse_on_within_day_label_rank_pct",
            "not_ndcg": True,
        })
    prediction = pd.concat(preds, ignore_index=True)
    prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
    prediction = prediction.loc[prediction["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)].copy()
    meta = {
        "model": "ElasticNet",
        "loss": "mse_on_within_day_label_rank_pct",
        "loss_ne_ndcg": True,
        "alpha": 1.0,
        "l1_ratio": 0.5,
        "folds": folds,
        "seed": int(seed),
        "features": features,
        "target": target,
    }
    return prediction, meta


def train_head_linear(
    *,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    features: Sequence[str],
    target: str,
    label_end_column: str,
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = SEEDS,
    resume: bool = True,
) -> dict[str, Any]:
    from ..round17.ensemble import assert_seed_diversity

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
            if resume and seed_path.is_file():
                prediction = pd.read_parquet(seed_path)
                prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
                frames.append(prediction)
                meta = json.loads((seed_dir / "training_report.json").read_text(encoding="utf-8")) if (seed_dir / "training_report.json").is_file() else {}
                seed_reports.append({"seed": int(seed), "rows": int(len(prediction)), "resumed": True, "folds": meta.get("folds", [])})
                continue
        prediction, metadata = _rolling_linear(
            training_clean,
            candidates_clean,
            features=features,
            target=target,
            label_end_column=label_end_column,
            seed=int(seed),
        )
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            assert_project_result_path(seed_dir)
            prediction.to_parquet(seed_path, index=False)
            metadata["prediction_sha256"] = sha256_file(seed_path)
            (seed_dir / "training_report.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        frames.append(prediction)
        seed_reports.append({"seed": int(seed), "rows": int(len(prediction)), "resumed": False, "folds": metadata.get("folds", [])})
    diversity = assert_seed_diversity(frames)
    ensemble = ensemble_rank_pct(frames)
    payload = {
        "status": "trained_oos",
        "model": "M1_ElasticNet_rank_proxy",
        "loss": "mse_on_within_day_label_rank_pct",
        "loss_ne_ndcg": True,
        "features": list(features),
        "target": target,
        "seeds": list(seeds),
        "seed_diversity": diversity,
        "rows": int(len(ensemble)),
        "days": int(ensemble["signal_date"].nunique()) if len(ensemble) else 0,
        "seed_reports": seed_reports,
        "ensemble": ensemble,
        "seed_frames": frames,
        "filled_with_zero": False,
        "train_audit": train_audit,
        "candidate_audit": cand_audit,
        "note": "Ridge/ElasticNet 排序代理。损失=日内 label rank-pct 的 MSE，不是 NDCG。",
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        assert_project_result_path(root)
        ens_path = root / "predictions.parquet"
        ensemble.to_parquet(ens_path, index=False)
        slim = {k: v for k, v in payload.items() if k not in {"ensemble", "seed_frames"}}
        slim["prediction_sha256"] = sha256_file(ens_path)
        dump_json(root / "ensemble_report.json", slim)
    return payload


def probe_lightgbm() -> dict[str, Any]:
    try:
        import lightgbm as lgb

        ver = getattr(lgb, "__version__", "unknown")
        if not hasattr(lgb, "LGBMRanker"):
            return {"available": False, "reason": "LGBMRanker missing", "version": ver}
        return {"available": True, "version": str(ver), "reason": None}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}", "version": None}


def _rolling_lgbm(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    label_end_column: str,
    rank_n_bins: int,
    seed: int,
    n_jobs: int,
    first_prediction_year: int = 2018,
    last_prediction_year: int = 2026,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import lightgbm as lgb

    from first_board_hermes.baseline import daily_relevance, mean_daily_ndcg

    features = list(features)
    training = training.copy()
    training["trade_date"] = pd.to_datetime(training["trade_date"])
    training[label_end_column] = pd.to_datetime(training[label_end_column])
    candidates = candidates.copy()
    candidates["trade_date"] = pd.to_datetime(candidates["trade_date"])
    preds: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
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
        group = train.groupby("trade_date", sort=False).size().to_numpy()
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[1],
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_child_samples=1,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=int(seed),
            n_jobs=int(n_jobs),
            verbosity=-1,
        )
        model.fit(
            train[features].to_numpy(dtype=np.float32),
            relevance.to_numpy(dtype=np.int32),
            group=group,
        )
        score = model.predict(test[features].to_numpy(dtype=np.float32))
        part = test[[*KEYS, target]].copy()
        part = part.rename(columns={"trade_date": "signal_date"})
        part["score"] = score
        part["predict_year"] = predict_year
        evaluable = part[target].notna()
        eval_frame = part.loc[evaluable].copy()
        ndcg = None
        if len(eval_frame):
            eval_frame["trade_date"] = eval_frame["signal_date"]
            eval_frame["relevance"] = daily_relevance(eval_frame, target=target, n_bins=rank_n_bins)
            ndcg = mean_daily_ndcg(eval_frame, k=1)
        preds.append(part[["signal_date", "instrument_id", "score", "predict_year"]].copy())
        folds.append({
            "predict_year": predict_year,
            "train_years": [train_start, train_end],
            "train_rows": int(len(train)),
            "candidate_rows_scored": int(len(part)),
            "mean_daily_ndcg_at_1": ndcg,
        })
    prediction = pd.concat(preds, ignore_index=True)
    if "signal_date" not in prediction.columns and "trade_date" in prediction.columns:
        prediction = prediction.rename(columns={"trade_date": "signal_date"})
    prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
    keep = ["signal_date", "instrument_id", "score"]
    if "predict_year" in prediction.columns:
        keep.append("predict_year")
    prediction = prediction[keep].copy()
    prediction = prediction.loc[prediction["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)].copy()
    meta = {"model": "LGBMRanker_lambdarank", "folds": folds, "seed": int(seed), "features": features, "target": target}
    return prediction, meta


def train_head_lgbm(
    *,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    features: Sequence[str],
    target: str,
    label_end_column: str,
    rank_n_bins: int,
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = SEEDS,
    resume: bool = True,
    n_jobs: int = XGB_N_JOBS,
) -> dict[str, Any]:
    from ..round17.ensemble import assert_seed_diversity

    probe = probe_lightgbm()
    if not probe.get("available"):
        return {
            "status": SKIP_DEPENDENCY,
            "model": "M2_LightGBM_LambdaRank",
            "reason": probe.get("reason"),
            "probe": probe,
            "ensemble": None,
        }
    training_clean, train_audit = complete_cases(training, features, require_target=True, target=target)
    candidates_clean, cand_audit = complete_cases(candidates, features, require_target=False, target=target)
    frames = []
    seed_reports = []
    ndcgs: list[float] = []
    for seed in seeds:
        seed_dir = None
        seed_path = None
        if output_dir is not None:
            seed_dir = Path(output_dir) / f"seed{seed}"
            seed_path = seed_dir / "predictions.parquet"
            if resume and seed_path.is_file():
                prediction = pd.read_parquet(seed_path)
                prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
                frames.append(prediction)
                meta = json.loads((seed_dir / "training_report.json").read_text(encoding="utf-8")) if (seed_dir / "training_report.json").is_file() else {}
                seed_reports.append({"seed": int(seed), "rows": int(len(prediction)), "resumed": True, "folds": meta.get("folds", [])})
                continue
        prediction, metadata = _rolling_lgbm(
            training_clean,
            candidates_clean,
            features=features,
            target=target,
            label_end_column=label_end_column,
            rank_n_bins=rank_n_bins,
            seed=int(seed),
            n_jobs=n_jobs,
        )
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            assert_project_result_path(seed_dir)
            prediction.to_parquet(seed_path, index=False)
            metadata["prediction_sha256"] = sha256_file(seed_path)
            (seed_dir / "training_report.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        frames.append(prediction)
        seed_reports.append({"seed": int(seed), "rows": int(len(prediction)), "resumed": False, "folds": metadata.get("folds", [])})
        for fold in metadata.get("folds") or []:
            v = fold.get("mean_daily_ndcg_at_1")
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                ndcgs.append(fv)
    diversity = assert_seed_diversity(frames)
    ensemble = ensemble_rank_pct(frames)
    payload = {
        "status": "trained_oos",
        "model": "M2_LightGBM_LambdaRank",
        "probe": probe,
        "features": list(features),
        "target": target,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE,
        "seeds": list(seeds),
        "seed_diversity": diversity,
        "rows": int(len(ensemble)),
        "days": int(ensemble["signal_date"].nunique()) if len(ensemble) else 0,
        "mean_fold_ndcg": float(np.mean(ndcgs)) if ndcgs else None,
        "seed_reports": seed_reports,
        "ensemble": ensemble,
        "seed_frames": frames,
        "filled_with_zero": False,
        "train_audit": train_audit,
        "candidate_audit": cand_audit,
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        assert_project_result_path(root)
        ens_path = root / "predictions.parquet"
        ensemble.to_parquet(ens_path, index=False)
        slim = {k: v for k, v in payload.items() if k not in {"ensemble", "seed_frames"}}
        slim["prediction_sha256"] = sha256_file(ens_path)
        dump_json(root / "ensemble_report.json", slim)
    return payload


def train_both_heads_family(
    *,
    family: str,
    five_training: pd.DataFrame,
    five_candidates: pd.DataFrame,
    ten_training: pd.DataFrame,
    ten_candidates: pd.DataFrame,
    five_extras: Sequence[str],
    ten_extras: Sequence[str],
    output_dir: Path,
    resume: bool = True,
    five_frozen: pd.DataFrame | None = None,
    ten_frozen: pd.DataFrame | None = None,
    force_retrain: bool = False,
    n_jobs: int = XGB_N_JOBS,
) -> dict[str, Any]:
    """family=M0 XGBRanker / M1 ElasticNet / M2 LightGBM。不可用时原样返回 SKIP_DEPENDENCY，不换模型。"""
    fam = str(family).upper()
    if fam in {"M0", "XGB", "XGBRANKER"}:
        out = train_both_heads(
            five_training=five_training,
            five_candidates=five_candidates,
            ten_training=ten_training,
            ten_candidates=ten_candidates,
            five_extras=five_extras,
            ten_extras=ten_extras,
            output_dir=output_dir,
            resume=resume,
            five_frozen=five_frozen,
            ten_frozen=ten_frozen,
            force_retrain=force_retrain,
            n_jobs=n_jobs,
        )
        out["model_family"] = "M0"
        return out
    five_features = unique_keep(list(FIVE_FEATURES) + list(five_extras))
    ten_features = unique_keep(list(TEN_FEATURES) + list(ten_extras))
    five_new = new_columns(FIVE_FEATURES, five_extras)
    ten_new = new_columns(TEN_FEATURES, ten_extras)
    output_dir = Path(output_dir)

    def _run(name, training, candidates, features, extras_new, frozen, head, trainer):
        need_train = bool(extras_new) or force_retrain
        if not need_train:
            if frozen is None:
                raise RuntimeError(f"{name} frozen predictions required")
            return frozen, {"status": "FROZEN_10_10", "features": list(head.features), "extras": []}
        extra_cols = [head.target, head.label_end_column]
        tr = slim_panel(training, [*KEYS, *features, *extra_cols])
        ca = slim_panel(candidates, [*KEYS, *features, head.target])
        kw = dict(
            training=tr,
            candidates=ca,
            features=features,
            target=head.target,
            label_end_column=head.label_end_column,
            output_dir=output_dir / name,
            seeds=SEEDS,
            resume=resume,
        )
        if trainer is train_head_lgbm:
            kw["rank_n_bins"] = head.rank_n_bins
            kw["n_jobs"] = n_jobs
        payload = trainer(**kw)
        if payload.get("status") == SKIP_DEPENDENCY:
            return None, payload
        return payload.get("ensemble"), {k: v for k, v in payload.items() if k not in {"ensemble", "seed_frames"}}

    if fam in {"M1", "RIDGE", "ELASTICNET"}:
        trainer = train_head_linear
        tag = "M1"
    elif fam in {"M2", "LGB", "LIGHTGBM"}:
        probe = probe_lightgbm()
        if not probe.get("available"):
            return {
                "status": SKIP_DEPENDENCY,
                "model_family": "M2",
                "reason": probe.get("reason"),
                "probe": probe,
                "five_pred": None,
                "ten_pred": None,
                "five": {"status": SKIP_DEPENDENCY},
                "ten": {"status": SKIP_DEPENDENCY},
            }
        trainer = train_head_lgbm
        tag = "M2"
    else:
        raise RuntimeError(f"unknown model family {family}")

    five_pred, five = _run("five", five_training, five_candidates, five_features, five_new, five_frozen, FIVE_HEAD, trainer)
    if isinstance(five, dict) and five.get("status") == SKIP_DEPENDENCY:
        return {"status": SKIP_DEPENDENCY, "model_family": tag, "five": five, "ten": {}, "five_pred": None, "ten_pred": None, "reason": five.get("reason")}
    ten_pred, ten = _run("ten", ten_training, ten_candidates, ten_features, ten_new, ten_frozen, TEN_HEAD, trainer)
    if isinstance(ten, dict) and ten.get("status") == SKIP_DEPENDENCY:
        return {"status": SKIP_DEPENDENCY, "model_family": tag, "five": five, "ten": ten, "five_pred": None, "ten_pred": None, "reason": ten.get("reason")}
    return {
        "status": "trained_oos",
        "model_family": tag,
        "five": five,
        "ten": ten,
        "five_pred": five_pred,
        "ten_pred": ten_pred,
        "five_new": five_new,
        "ten_new": ten_new,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE,
    }


__all__ = [
    "new_columns",
    "probe_lightgbm",
    "slim_panel",
    "train_both_heads",
    "train_both_heads_family",
    "train_head_lgbm",
    "train_head_linear",
    "unique_keep",
]
