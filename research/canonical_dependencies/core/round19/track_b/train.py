"""Round19-only force_retrain wrapper. Does not change Round18 artifacts.

Round18 ``train_both_heads`` skips XGB when extras are empty and returns frozen
Round7 predictions. R19_B01 must retrain the original 14/13 columns through the
same ``train_head_open`` path, still without F1/F2/F3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ...round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ...round7.train import assert_seed_rank_invariance
from ...round17.train import train_head_open
from ..contract import COLSAMPLE, SUBSAMPLE, XGB_N_JOBS


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
) -> dict[str, Any]:
    """Same contract as Round18 ``train_both_heads`` plus ``force_retrain``.

    Default ``force_retrain=False`` preserves the Round18 frozen short-circuit.
    Only Round19 B01 / repaired runs pass ``force_retrain=True``.
    """
    five_features = list(FIVE_FEATURES) + list(five_extras)
    ten_features = list(TEN_FEATURES) + list(ten_extras)
    if list(five_extras) or force_retrain:
        five = train_head_open(
            training=five_training,
            candidates=five_candidates,
            features=five_features,
            target=FIVE_HEAD.target,
            label_end_column=FIVE_HEAD.label_end_column,
            rank_n_bins=FIVE_HEAD.rank_n_bins,
            eval_metric=FIVE_HEAD.eval_metric,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE,
            first_prediction_year=2018,
            last_prediction_year=2026,
            output_dir=output_dir / "five",
            seeds=SEEDS,
            n_jobs=XGB_N_JOBS,
            save_models=False,
            resume=resume,
        )
        five_pred = five["ensemble"]
        if five.get("seed_frames"):
            five["rank_invariance"] = assert_seed_rank_invariance(five["seed_frames"])
        five["force_retrain"] = bool(force_retrain)
        five["extras"] = list(five_extras)
    else:
        if five_frozen is None:
            raise RuntimeError("five frozen predictions required when no five extras and not force_retrain")
        five_pred = five_frozen
        five = {
            "status": "FROZEN_10_10",
            "features": list(FIVE_FEATURES),
            "force_retrain": False,
            "extras": [],
        }
    if list(ten_extras) or force_retrain:
        ten = train_head_open(
            training=ten_training,
            candidates=ten_candidates,
            features=ten_features,
            target=TEN_HEAD.target,
            label_end_column=TEN_HEAD.label_end_column,
            rank_n_bins=TEN_HEAD.rank_n_bins,
            eval_metric=TEN_HEAD.eval_metric,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE,
            first_prediction_year=2018,
            last_prediction_year=2026,
            output_dir=output_dir / "ten",
            seeds=SEEDS,
            n_jobs=XGB_N_JOBS,
            save_models=False,
            resume=resume,
        )
        ten_pred = ten["ensemble"]
        if ten.get("seed_frames"):
            ten["rank_invariance"] = assert_seed_rank_invariance(ten["seed_frames"])
        ten["force_retrain"] = bool(force_retrain)
        ten["extras"] = list(ten_extras)
    else:
        if ten_frozen is None:
            raise RuntimeError("ten frozen predictions required when no ten extras and not force_retrain")
        ten_pred = ten_frozen
        ten = {
            "status": "FROZEN_10_10",
            "features": list(TEN_FEATURES),
            "force_retrain": False,
            "extras": [],
        }
    return {"five": five, "ten": ten, "five_pred": five_pred, "ten_pred": ten_pred}


__all__ = ["train_both_heads"]
