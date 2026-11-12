"""外层按年隔离。内层半年块只用于审计，不用来在看完外层后再挑参数。"""

from __future__ import annotations

import pandas as pd

EVAL_START = pd.Timestamp("2018-01-02")
EVAL_END = pd.Timestamp("2026-07-24")


def outer_years() -> list[int]:
    return list(range(2018, 2027))


def year_cutoff(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{int(year)}-01-01")


def inner_blocks(year: int) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Y-1 的两个半年。模型必须在每个块开始前已经拟合，不能用训到 Y-1 年末的模型回填 H1。"""
    previous = int(year) - 1
    return {
        "H1": (pd.Timestamp(f"{previous}-01-01"), pd.Timestamp(f"{previous}-06-30")),
        "H2": (pd.Timestamp(f"{previous}-07-01"), pd.Timestamp(f"{previous}-12-31")),
    }


def sufficient_history(year: int, first_ready: pd.Timestamp, minimum_days: int = 120) -> bool:
    if int(year) <= 2018:
        return False
    return (year_cutoff(year) - pd.Timestamp(first_ready)).days >= minimum_days


def assert_no_future_fit(predict_year: int, fit_cutoff: pd.Timestamp) -> None:
    if pd.Timestamp(fit_cutoff) > year_cutoff(predict_year):
        raise RuntimeError("fit cutoff leaks into the prediction year")
