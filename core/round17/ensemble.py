"""Round17 集成本：复制 round7 ensemble_rank_pct；多样性断言替代 rank invariance。"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..round7.train import ensemble_rank_pct
from .contract import SEED_DIVERSITY_DEGENERATE, SEED_DIVERSITY_PASS


def _rank_signature(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame[["signal_date", "instrument_id", "score"]].copy()
    ranked["signal_date"] = pd.to_datetime(ranked["signal_date"])
    ranked = ranked.sort_values(
        ["signal_date", "score", "instrument_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranked["rank"] = ranked.groupby("signal_date", sort=False).cumcount()
    return ranked


def _score_hash(frame: pd.DataFrame) -> str:
    work = frame[["signal_date", "instrument_id", "score"]].copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).astype("int64")
    work["instrument_id"] = work["instrument_id"].astype(str)
    work = work.sort_values(["signal_date", "instrument_id"])
    blob = work.to_numpy().tobytes()
    return hashlib.sha256(blob).hexdigest()


def assert_seed_diversity(frames: Iterable[pd.DataFrame]) -> dict[str, Any]:
    """Genuine diversity required. Never inject noise. Degenerate → flag, do not raise."""
    items = list(frames)
    if len(items) < 2:
        raise ValueError("need at least two seed predictions")
    hashes = [_score_hash(frame) for frame in items]
    unique_hashes = sorted(set(hashes))
    aligned = None
    cols = []
    for i, frame in enumerate(items):
        part = frame[["signal_date", "instrument_id", "score"]].copy()
        part["signal_date"] = pd.to_datetime(part["signal_date"])
        part = part.rename(columns={"score": f"s{i}"})
        aligned = part if aligned is None else aligned.merge(
            part, on=["signal_date", "instrument_id"], how="inner"
        )
        cols.append(f"s{i}")
    corr = aligned[cols].corr().to_numpy(dtype=float) if aligned is not None and len(aligned) else np.eye(len(items))
    pairwise = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairwise.append(float(corr[i, j]))
    top1 = []
    for frame in items:
        sig = _rank_signature(frame)
        day_top = sig.loc[sig["rank"].eq(0), ["signal_date", "instrument_id"]]
        top1.append(day_top.set_index("signal_date")["instrument_id"].astype(str))
    top1_agree = 1.0
    if len(top1) >= 2:
        ref = top1[0]
        agrees = []
        for other in top1[1:]:
            joined = pd.concat([ref, other], axis=1, join="inner")
            if joined.empty:
                agrees.append(0.0)
            else:
                agrees.append(float((joined.iloc[:, 0] == joined.iloc[:, 1]).mean()))
        top1_agree = float(np.mean(agrees)) if agrees else 1.0
    identical = len(unique_hashes) == 1
    mean_corr = float(np.nanmean(pairwise)) if pairwise else 1.0
    degenerate = bool(identical or (mean_corr >= 1.0 - 1e-12 and top1_agree >= 1.0 - 1e-12))
    status = SEED_DIVERSITY_DEGENERATE if degenerate else SEED_DIVERSITY_PASS
    return {
        "status": status,
        "n_seeds": len(items),
        "unique_prediction_hashes": len(unique_hashes),
        "prediction_hashes": hashes,
        "mean_pairwise_corr": mean_corr,
        "min_pairwise_corr": float(np.nanmin(pairwise)) if pairwise else None,
        "max_pairwise_corr": float(np.nanmax(pairwise)) if pairwise else None,
        "top1_mean_agreement": top1_agree,
        "degenerate": degenerate,
        "note": (
            "tree_method=hist + subsample=colsample=1.0 会使 seed 预测逐行相同；"
            "0.8/0.8 必须产生不同 hash / 非 1.0 相关 / Top1 不完全一致。"
            if degenerate else
            "genuine seed diversity under R17_STOCHASTIC_BAGGING_08_08"
        ),
    }


__all__ = ["assert_seed_diversity", "ensemble_rank_pct"]
