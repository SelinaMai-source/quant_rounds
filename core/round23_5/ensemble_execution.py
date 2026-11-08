"""正式袖套。不改 B03/O01、费用和退出。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.r225241_sleeve import run_round7


def run_sleeve(
    dest: Path,
    scores: pd.DataFrame,
    code: str,
    *,
    resume: bool = True,
    buy_extra_slippage_bps: float = 0.0,
    already_overlayed: bool = False,
) -> dict[str, Any]:
    hooks = None
    if buy_extra_slippage_bps:
        hooks = {"buy_extra_slippage_bps": float(buy_extra_slippage_bps)}
    return run_round7(
        Path(dest),
        scores,
        {
            "code": code,
            "round": "23.5",
            "scope": "FULL_HISTORY_FIRST_TO_SECOND",
            "information_lane": "LEGACY_OPEN_INFORMATION_COMPARISON",
        },
        resume=resume,
        validation_hooks=hooks,
        counts_as_alpha=buy_extra_slippage_bps == 0.0,
        scenario=buy_extra_slippage_bps > 0.0,
        already_overlayed=already_overlayed,
    )


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "score" not in out.columns and "v5" in out.columns:
        out = out.rename(columns={"v5": "score"})
    keep = ["signal_date", "instrument_id", "score"]
    if "allocation_multiplier" in out.columns:
        keep.append("allocation_multiplier")
    return out[keep]
