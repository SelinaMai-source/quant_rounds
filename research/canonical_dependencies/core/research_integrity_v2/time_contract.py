"""决策时点 vs 执行时点；Asia/Shanghai。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")
LEGACY_OPEN_INFORMATION_LANE = "LEGACY_OPEN_INFORMATION_LANE"


def shanghai_now() -> datetime:
    return datetime.now(SHANGHAI)


def shanghai_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(SHANGHAI)
    return ts.tz_convert(SHANGHAI)


def decision_vs_execution_tags(*, uses_d1_open_gap: bool) -> dict[str, Any]:
    if uses_d1_open_gap:
        return {
            "lane": LEGACY_OPEN_INFORMATION_LANE,
            "pure_d_close": False,
            "claim_open_executable_verified": False,
            "note": "含 d1_open_gap 的历史契约可比较，但不能声称真实开盘可执行性已验证",
        }
    return {
        "lane": "D_CLOSE_DECISION",
        "pure_d_close": True,
        "claim_open_executable_verified": False,
    }
