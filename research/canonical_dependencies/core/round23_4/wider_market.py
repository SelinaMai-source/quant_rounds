"""内层窗口用的行情。正式 2018–2026 袖套仍用原快照窗口，不改生产加载。

内层 H1/H2 对 2018 外层年落在 2017。原 load_prepared_market 从 2018-01-01 往前只热身约 30 天，
2017 上半年会话数为 0。同一 Hermes v7 快照含更早日线，这里只为内层回放多读一段。
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from core.paths import MARKET_END, SNAPSHOT_RELATIVE

_PREPARED: Any = None
INNER_MARKET_START = date(2016, 11, 1)


def inner_prepared_market() -> Any:
    global _PREPARED
    if _PREPARED is not None:
        return _PREPARED
    from core.stock_readonly import ensure_stock_src_on_path, stock_path

    ensure_stock_src_on_path()
    from first_board_hermes.fixed_sleeve_backtest import prepare_fixed_sleeve_market
    from zyquant.data import DataSnapshot

    snapshot = DataSnapshot(stock_path(SNAPSHOT_RELATIVE), verify_hashes=False)
    bars = snapshot.table(
        "daily_raw",
        start=INNER_MARKET_START,
        end=MARKET_END,
        cutoff=snapshot.metadata.as_of_date,
        fields=[
            "trade_date",
            "instrument_id",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "volume",
            "paused",
            "limit_up",
            "limit_down",
        ],
    )
    _PREPARED = prepare_fixed_sleeve_market(bars, ma_window=5)
    return _PREPARED


def sessions_between(prepared: Any, start, end) -> int:
    calendar = pd.to_datetime(pd.Series(prepared.calendar))
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return int(((calendar >= start_ts) & (calendar <= end_ts)).sum())
