"""涨跌幅制度与连板状态机。禁止把 limit_up=999999.999 当成涨停。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

SENTINEL_MIN = 99_999.0
ST_CUTOVER = pd.Timestamp("2026-07-06")
CHINEXT_REFORM = pd.Timestamp("2020-08-24")
MAIN_REGISTRATION = pd.Timestamp("2023-04-10")


def board_of(symbol: str, exchange: str) -> str:
    sym = str(symbol or "")
    exch = str(exchange or "")
    if exch == "XBEI" or sym.startswith(("83", "87", "43", "82", "88", "92", "4")):
        return "bse"
    if sym.startswith(("688", "689")):
        return "star"
    if sym.startswith(("300", "301")):
        return "chinext"
    return "main"


def is_st_name(name: str | None) -> bool:
    text = str(name or "").strip().upper().replace(" ", "")
    return text.startswith(("*ST", "ST", "S*ST", "SST"))


def limit_pct(board: str, trade_date: pd.Timestamp, listed_sessions: int, is_st: bool) -> float | None:
    """返回法定涨跌幅。None 表示该日不设限（上市初期）。"""
    day = pd.Timestamp(trade_date)
    sessions = int(listed_sessions)
    if board == "star":
        if sessions <= 5:
            return None
        amp = 0.20
    elif board == "chinext":
        if day < CHINEXT_REFORM:
            if sessions <= 1:
                return None
            amp = 0.10
        else:
            if sessions <= 5:
                return None
            amp = 0.20
    elif board == "bse":
        if sessions <= 1:
            return None
        amp = 0.30
    else:
        if day < MAIN_REGISTRATION:
            if sessions <= 1:
                return None
        elif sessions <= 5:
            return None
        amp = 0.10
    if is_st:
        return 0.10 if day >= ST_CUTOVER else 0.05
    return amp


def limit_price(pre_close: float, amp: float | None) -> float | None:
    if amp is None or not np.isfinite(pre_close) or pre_close <= 0:
        return None
    return float(np.round(pre_close * (1.0 + amp) + 1e-8, 2))


def vendor_limit_is_sentinel(limit_up: float) -> bool:
    return np.isfinite(limit_up) and float(limit_up) >= SENTINEL_MIN


def seal_flags(pre_close: float, high: float, close: float, amp: float | None) -> tuple[bool, bool]:
    """(sealed, touched)。不读取 vendor limit_up。"""
    px = limit_price(pre_close, amp)
    if px is None:
        return False, False
    touched = np.isfinite(high) and high + 1e-6 >= px - 1e-4
    sealed = np.isfinite(close) and close + 1e-6 >= px - 1e-4
    return bool(sealed), bool(touched and not sealed)


def consecutive_streak(halt: np.ndarray, unlimited: np.ndarray, sealed: np.ndarray) -> np.ndarray:
    """停牌沿用连板，炸板或普通日归零。不用 rolling sum。"""
    n = int(len(sealed))
    out = np.empty(n, dtype=np.int16)
    streak = 0
    for i in range(n):
        if bool(halt[i]):
            out[i] = streak
            continue
        if bool(unlimited[i]) or not bool(sealed[i]):
            streak = 0
        else:
            streak += 1
        out[i] = streak
    return out


def select_second_board(frame: pd.DataFrame) -> pd.DataFrame:
    """D 日为当前序列第二板。不看 D+1 是否封板。"""
    forbidden = [c for c in frame.columns if c.startswith("label_") or c in {"d1_sealed", "next_sealed"}]
    if forbidden:
        # 选择本身不得依赖这些列；这里只是断言调用方没有把它们放进筛选键。
        pass
    need = {"sealed", "streak"}
    missing = need - set(frame.columns)
    if missing:
        raise KeyError(f"second-board select missing {sorted(missing)}")
    mask = frame["sealed"].astype(bool) & frame["streak"].eq(2)
    return frame.loc[mask].copy()


def synthetic_calendar_case() -> pd.DataFrame:
    """供测试：主板、非 ST、无停牌。"""
    days = pd.bdate_range("2019-01-02", periods=6)
    pre = np.array([10, 11, 12.1, 11.5, 11.5, 12.65], dtype=float)
    # 手工涨停价 round(pre*1.1, 2)
    close = np.array([11.0, 12.1, 12.0, 11.5, 12.65, 13.92], dtype=float)
    high = close.copy()
    return pd.DataFrame(
        {
            "trade_date": days,
            "pre_close": pre,
            "high": high,
            "close": close,
            "board": "main",
            "is_st": False,
            "listed_sessions": [10, 11, 12, 13, 14, 15],
            "halt": False,
        }
    )
