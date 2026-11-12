"""计划敞口与 R4 倍率。ratio 只来自内层计划订单，不看外层回撤。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.round23_4.contract import RANK_FRACTIONS


def planned_fraction_for_day(block: pd.DataFrame) -> float:
    """Top2 计划资金占净值比例。零倍率保持零，不把名额让给下一名。"""
    work = block.sort_values(["score", "instrument_id"], ascending=[False, True], kind="mergesort")
    if "allocation_multiplier" in work.columns:
        mult = pd.to_numeric(work["allocation_multiplier"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    else:
        mult = np.ones(len(work), dtype=float)
    total = 0.0
    for rank, fraction in enumerate(RANK_FRACTIONS):
        if rank >= len(mult):
            break
        total += float(fraction) * float(mult[rank])
    return float(total)


def mean_planned_exposure(scores: pd.DataFrame) -> float:
    if scores is None or len(scores) == 0:
        return float("nan")
    frame = scores.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    values = [planned_fraction_for_day(block) for _, block in frame.groupby("signal_date", sort=True)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def exposure_ratio(r3: float, r0: float) -> dict[str, float | str | bool]:
    if r0 is None or not np.isfinite(r0) or float(r0) == 0.0:
        return {"ratio": 1.0, "clipped": True, "identifiable": False, "reason": "DENOMINATOR_ZERO_DEFAULT_1"}
    raw = float(r3) / float(r0)
    clipped = float(np.clip(raw, 0.0, 1.0))
    return {
        "ratio": clipped,
        "raw_ratio": raw,
        "clipped": clipped != raw,
        "identifiable": True,
        "reason": "INNER_PLANNED_EXPOSURE",
    }


def scale_multipliers(scores: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """new = original * ratio。零仓位不被重新打开，也不把倍率改写成 1。"""
    out = scores.copy()
    if "allocation_multiplier" not in out.columns:
        out["allocation_multiplier"] = 1.0
    original = pd.to_numeric(out["allocation_multiplier"], errors="coerce").fillna(1.0)
    scaled = original * float(ratio)
    scaled = scaled.where(original > 0.0, 0.0)
    out["allocation_multiplier"] = scaled
    out["original_allocation_multiplier"] = original
    return out
