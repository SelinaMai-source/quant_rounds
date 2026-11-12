"""Nardari & Schüssler 2024 式(7)。paper_id=P01_NARDARI_2024。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PAPER_ID = "P01_NARDARI_2024"
GAMMA = 3.0


def combination_weights(gross: np.ndarray, alpha: float, gamma: float = GAMMA) -> tuple[np.ndarray, bool]:
    """gross 为 T×M 的月度总收益（1+净收益）。只使用调用者已经截断的历史。"""
    gross = np.asarray(gross, dtype=float)
    if gross.ndim != 2 or gross.shape[0] < 2 or gross.shape[1] < 2:
        raise ValueError("need a return matrix")
    if np.any(~np.isfinite(gross)):
        raise ValueError("non-finite gross returns")
    age = np.arange(gross.shape[0] - 1, -1, -1, dtype=float)
    discount = np.power(float(alpha), age)

    def objective(weights: np.ndarray) -> float:
        portfolio = np.maximum(gross @ weights, 1e-8)
        utility = (np.power(portfolio, 1.0 - gamma) - 1.0) / (1.0 - gamma)
        return -float(np.dot(discount, utility))

    count = gross.shape[1]
    start = np.full(count, 1.0 / count)
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"ftol": 1e-12, "maxiter": 300},
    )
    weights = np.clip(result.x if result.success else start, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        return start, False
    return weights / total, bool(result.success)


def monthly_gross(daily: pd.Series) -> pd.Series:
    series = daily.astype(float).dropna().sort_index()
    gross = (1.0 + series).groupby(series.index.to_period("M")).prod()
    gross.index = gross.index.to_timestamp("M")
    return gross


def blend_rule_scores(frame: pd.DataFrame, weight_by_year: dict[int, dict[str, float]], columns: list[str]) -> pd.DataFrame:
    """把规则分数按当年权重凸组合。缺任何一个规则的日期回退 v5，不删股票。"""
    work = frame.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    out = work[["signal_date", "instrument_id"]].copy()
    scores = []
    fallback = []
    for day, block in work.groupby("signal_date", sort=True):
        year = int(pd.Timestamp(day).year)
        weights = weight_by_year.get(year)
        if not weights:
            scores.append(block["v5"].to_numpy())
            fallback.append(np.ones(len(block), dtype=bool))
            continue
        acc = np.zeros(len(block), dtype=float)
        ok = True
        for name in columns:
            acc = acc + float(weights[name]) * block[name].to_numpy(dtype=float)
        missing = ~np.isfinite(acc)
        base = block["v5"].to_numpy(dtype=float)
        acc = np.where(missing, base, acc)
        scores.append(acc)
        fallback.append(missing | np.array([weights.get("_fallback_year", False)] * len(block)))
    out["score"] = np.concatenate(scores)
    out["fallback_v5"] = np.concatenate(fallback)
    return out
