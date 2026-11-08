"""Jensen 等人的可实现效用原则，预注册成离散动作。paper_id=P03_JENSEN_2026。

完整 Portfolio-ML 不在这里实现。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PAPER_ID = "P03_JENSEN_2026"
ACTIONS = ("KEEP_V5", "TOP1_ONLY", "TOP1_TOP2", "SWAP_TOP2_TOP3", "ABSTAIN")


def apply_action(scores: pd.DataFrame, action: str) -> pd.DataFrame:
    """在已经排好序的全池上改计划。不把第三名递补成新的全池重排。"""
    work = scores.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    work["rank"] = work.groupby("signal_date")["score"].rank(ascending=False, method="first")
    if action in {"KEEP_V5", "TOP1_TOP2"}:
        return work.drop(columns=["rank"])
    if action == "TOP1_ONLY":
        return work.loc[work["rank"].eq(1)].drop(columns=["rank"])
    if action == "ABSTAIN":
        return work.iloc[0:0].drop(columns=["rank"])
    if action == "SWAP_TOP2_TOP3":
        parts = []
        for _, block in work.groupby("signal_date", sort=True):
            block = block.copy()
            second = block.loc[block["rank"].eq(2), "score"]
            third = block.loc[block["rank"].eq(3), "score"]
            if len(second) == 1 and len(third) == 1:
                block.loc[block["rank"].eq(2), "score"] = float(third.iloc[0])
                block.loc[block["rank"].eq(3), "score"] = float(second.iloc[0])
            parts.append(block)
        return pd.concat(parts, ignore_index=True).drop(columns=["rank"])
    raise ValueError(action)


def utility(net: pd.Series, turnover: float, ulcer: float, lambda_turnover: float, lambda_drawdown: float) -> float:
    if net.empty:
        return -1e9
    return float(net.mean()) - float(lambda_turnover) * float(turnover) - float(lambda_drawdown) * float(ulcer)


def ulcer_index(net: pd.Series) -> float:
    if net.empty:
        return 0.0
    wealth = (1.0 + net.astype(float)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(np.sqrt(np.mean(np.square(drawdown))))


def s3_fill(planned: list[str], blocked: set[str]) -> dict[str, list[str] | bool]:
    """保留原计划名单。涨停开盘的买单不成交，不把名单外的下一名递补进来。"""
    filled = [name for name in planned if name not in blocked]
    return {"planned": list(planned), "filled": filled, "refilled": False}
