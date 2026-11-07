"""显式 merge 与排名唯一性。"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd


def explicit_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: Sequence[str],
    how: str = "left",
    validate: str | None = "one_to_one",
    suffixes: tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    keys = list(on)
    for frame, name in ((left, "left"), (right, "right")):
        missing = [k for k in keys if k not in frame.columns]
        if missing:
            raise KeyError(f"{name} missing merge keys: {missing}")
        dup = int(frame.duplicated(keys).sum())
        if dup:
            raise ValueError(f"{name} has {dup} duplicate keys on {keys}")
    return left.merge(right, on=keys, how=how, validate=validate, suffixes=suffixes)


def rank_unique(
    frame: pd.DataFrame,
    *,
    date_col: str = "signal_date",
    id_col: str = "instrument_id",
    score_col: str = "score",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[id_col] = work[id_col].astype(str)
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    n_before = int(len(work))
    dup = int(work.duplicated([date_col, id_col]).sum())
    if dup:
        raise ValueError(f"rank input not unique on {date_col}+{id_col}: {dup} dups")
    valid = work[score_col].notna()
    ranked = work.sort_values([date_col, score_col, id_col], ascending=[True, False, True], kind="mergesort")
    ranked["raw_rank"] = ranked.groupby(date_col, sort=False).cumcount() + 1
    audit = {
        "n_rows": n_before,
        "n_dates": int(ranked[date_col].nunique()),
        "n_valid_scores": int(valid.sum()),
        "n_missing_scores": int((~valid).sum()),
        "duplicate_keys": 0,
        "tie_break": "instrument_id ascending after score descending",
    }
    return ranked, audit
