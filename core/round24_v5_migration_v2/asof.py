"""可见版本连接：先限制 known_at，再取最新。禁止先 argMax(UPDATE_TIME)。"""

from __future__ import annotations

import pandas as pd


def latest_visible(frame: pd.DataFrame, decision_time: pd.Timestamp, *, known_col: str = "known_at") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    work[known_col] = pd.to_datetime(work[known_col])
    visible = work.loc[work[known_col].le(pd.Timestamp(decision_time))].copy()
    if visible.empty:
        return visible
    visible = visible.sort_values(known_col)
    keys = [c for c in ("instrument_id", "trade_date") if c in visible.columns]
    if not keys:
        return visible.tail(1)
    return visible.groupby(keys, as_index=False).tail(1)


def wrong_argmax_update_time(frame: pd.DataFrame) -> pd.DataFrame:
    """反例：先取今天最新回填，再假装历史可得。测试用，不得进入主 lane。"""
    work = frame.sort_values("update_time")
    keys = [c for c in ("instrument_id", "trade_date") if c in work.columns]
    return work.groupby(keys, as_index=False).tail(1)
