"""Track B：去掉 d1_open_gap 后重训；以及买入不利成交。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..paths import (
    MARKET_END,
    PREDICTION_START,
    ROUND8_ROOT,
    SIGNAL_END,
    assert_project_result_path,
    ensure_round8_tree,
)
from ..replay_round7.contract import FIVE_FEATURES, SEEDS, TEN_FEATURES, TEN_PANEL, TEN_WEIGHT
from ..stock_readonly import ensure_stock_src_on_path, sha256_file
from .contract import FIVE_HEAD_PRED, TEN_HEAD_PRED

D_CLOSE_ROOT = ROUND8_ROOT / "models" / "d_close"
D_CLOSE_PRED = ROUND8_ROOT / "predictions" / "d_close" / "predictions.parquet"


def _model_parameters(seed: int, eval_metric: str) -> dict[str, Any]:
    return {
        "objective": "rank:ndcg",
        "eval_metric": eval_metric,
        "ndcg_exp_gain": True,
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "tree_method": "hist",
        "random_state": int(seed),
        "n_jobs": 2,
        "rank_n_bins": 5 if eval_metric == "ndcg@1" else 10,
    }


def _ensemble_rank_pct(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = None
    rank_columns = []
    for index, frame in enumerate(frames):
        column = f"seed_{index}_rank_pct"
        part = frame[["signal_date", "instrument_id", "score"]].copy()
        part["signal_date"] = pd.to_datetime(part["signal_date"])
        part[column] = part.groupby("signal_date")["score"].rank(method="average", pct=True)
        part = part[["signal_date", "instrument_id", column]]
        merged = part if merged is None else merged.merge(
            part, on=["signal_date", "instrument_id"], how="inner", validate="one_to_one"
        )
        rank_columns.append(column)
    if merged is None:
        raise ValueError("no seed predictions")
    merged["score"] = merged[rank_columns].mean(axis=1)
    return merged[["signal_date", "instrument_id", "score"]]


def train_d_close_head(*, name: str, panel_dir, features: tuple[str, ...],
                       target: str, label_end: str, eval_metric: str,
                       resume: bool = True) -> dict[str, Any]:
    ensure_stock_src_on_path()
    from first_board_hermes.baseline import RankerConfig, rolling_train_predict

    root = D_CLOSE_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(root / "dummy")
    ensemble_path = root / "predictions.parquet"
    report_path = root / "ensemble_report.json"
    if resume and ensemble_path.is_file() and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))

    training_path = panel_dir / "training_panel.parquet"
    candidate_path = panel_dir / "candidate_panel.parquet"
    required = ["trade_date", "instrument_id", *features, target, label_end]
    training = pd.read_parquet(training_path, columns=list(dict.fromkeys(required)))
    candidate = pd.read_parquet(
        candidate_path,
        columns=list(dict.fromkeys(["trade_date", "instrument_id", *features, target])),
    )
    for frame in (training, candidate):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    training[label_end] = pd.to_datetime(training[label_end])
    first_year = max(2018, int(candidate["trade_date"].dt.year.min()))
    last_year = int(SIGNAL_END.year)
    allowed = {item.name for item in fields(RankerConfig)}
    frames = []
    seed_reports = []
    for seed in SEEDS:
        seed_dir = root / f"seed{seed}"
        seed_path = seed_dir / "predictions.parquet"
        if resume and seed_path.is_file():
            prediction = pd.read_parquet(seed_path)
            metadata = json.loads((seed_dir / "training_report.json").read_text(encoding="utf-8"))
        else:
            params = _model_parameters(seed, eval_metric)
            config = RankerConfig(**{key: value for key, value in params.items() if key in allowed})
            prediction, metadata = rolling_train_predict(
                training,
                candidate,
                seed_dir,
                features=list(features),
                first_prediction_year=first_year,
                last_prediction_year=last_year,
                config=config,
                target=target,
                label_end_column=label_end,
                save_models=True,
            )
        prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
        prediction = prediction.loc[
            prediction["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)
        ].copy()
        frames.append(prediction)
        seed_reports.append({"seed": seed, "rows": int(len(prediction))})
    ensemble = _ensemble_rank_pct(frames)
    assert_project_result_path(ensemble_path)
    ensemble.to_parquet(ensemble_path, index=False)
    payload = {
        "status": "trained_oos",
        "head": name,
        "features": list(features),
        "dropped": ["d1_open_gap"],
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "seeds": list(SEEDS),
        "rows": int(len(ensemble)),
        "prediction_sha256": sha256_file(ensemble_path),
        "seed_reports": seed_reports,
        "written_at": datetime.now().astimezone().isoformat(),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    return payload


def fuse_d_close(*, resume: bool = True) -> dict[str, Any]:
    """无 gap 预滤：只要求两头都有分。融合权重仍 90/10。"""
    ensure_round8_tree()
    dest = D_CLOSE_PRED
    dest.parent.mkdir(parents=True, exist_ok=True)
    report_path = dest.parent / "report.json"
    if resume and dest.is_file() and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    five = pd.read_parquet(D_CLOSE_ROOT / "five" / "predictions.parquet")
    ten = pd.read_parquet(D_CLOSE_ROOT / "ten" / "predictions.parquet")
    keys = ["signal_date", "instrument_id"]
    five = five.rename(columns={"score": "score_five"})
    ten = ten.rename(columns={"score": "score_ten"})
    work = five.merge(ten, on=keys, how="inner", validate="one_to_one")
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    work["pct_five"] = work.groupby("signal_date")["score_five"].rank(method="average", pct=True)
    work["pct_ten"] = work.groupby("signal_date")["score_ten"].rank(method="average", pct=True)
    work["score"] = TEN_WEIGHT * work["pct_ten"] + (1.0 - TEN_WEIGHT) * work["pct_five"]
    out = work[keys + ["score"]].sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort"
    ).reset_index(drop=True)
    assert_project_result_path(dest)
    out.to_parquet(dest, index=False)
    payload = {
        "status": "complete",
        "rows": int(len(out)),
        "days": int(out["signal_date"].nunique()),
        "gap_prefilter": False,
        "fusion": "0.90 ten + 0.10 five after same-day percentile",
        "sha256": sha256_file(dest),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    return payload


def train_d_close(*, resume: bool = True) -> dict[str, Any]:
    from ..replay_round7.contract import FIVE_FEATURES as FF
    from pathlib import Path

    five_feats = tuple(x for x in FF if x != "d1_open_gap")
    ten_feats = tuple(x for x in TEN_FEATURES if x != "d1_open_gap")
    five_panel = Path(
        "/home/mxy/Stock/data/first_board_hermes/"
        "aligned_original_candidate_full44_plus_d1_open_gap_v1"
    )
    ten_panel = TEN_PANEL
    five = train_d_close_head(
        name="five",
        panel_dir=five_panel,
        features=five_feats,
        target="label_e2_net",
        label_end="next2_date",
        eval_metric="ndcg@1",
        resume=resume,
    )
    ten = train_d_close_head(
        name="ten",
        panel_dir=ten_panel,
        features=ten_feats,
        target="label_exit_rule_net",
        label_end="actual_exit_date",
        eval_metric="ndcg@2",
        resume=resume,
    )
    blend = fuse_d_close(resume=resume)
    return {"five": five, "ten": ten, "blend": blend}


__all__ = ["D_CLOSE_PRED", "train_d_close"]
