"""候选面板、交易日历、Hermes 列加载。缺失不填 0。"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ..paths import SIGNAL_END
from ..round7.contract import FIVE_HEAD
from .contract import HERMES_V7, IC_START, LABELS

KEYS = ["trade_date", "instrument_id"]
LABEL_END = {
    "label_e2_net": "next2_date",
    "label_exit_rule_net": "actual_exit_date",
}
EXTRA_COLS = [
    *LABELS,
    "next2_date",
    "actual_exit_date",
    "is_first_board_hard",
]
_EXCHANGE = {"SH": "XSHG", "SZ": "XSHE", "BJ": "XBEI"}


def market_to_instrument(code: Any) -> str | None:
    if code is None or (isinstance(code, float) and not np.isfinite(code)):
        return None
    text = str(code).strip().upper()
    if not text or text == "NAN":
        return None
    if "." in text:
        sym, suf = text.split(".", 1)
        exch = _EXCHANGE.get(suf, suf)
        return f"{sym}.{exch}"
    return text


def parse_yyyymmdd(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.replace(r"[^0-9]", "", regex=True)
    out = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    return out


def load_candidate_keys() -> pd.DataFrame:
    path = FIVE_HEAD.panel_dir / "candidate_panel.parquet"
    if not path.is_file():
        raise RuntimeError(f"MISSING_ARTIFACT candidate_panel: {path}")
    names = set(pq.read_schema(path).names)
    cols = [c for c in [*KEYS, *EXTRA_COLS] if c in names]
    frame = pd.read_parquet(path, columns=cols)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    for col in ("next2_date", "actual_exit_date"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    for col in LABELS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.drop_duplicates(KEYS).sort_values(KEYS).reset_index(drop=True)
    return frame


def load_trade_calendar() -> pd.DatetimeIndex:
    root = HERMES_V7 / "trade_calendar"
    table = ds.dataset(str(root), format="parquet").to_table(columns=["trade_date", "exchange"])
    frame = table.to_pandas()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    # 沪深交易日并集；一进二候选覆盖两市。
    days = pd.DatetimeIndex(sorted(frame["trade_date"].dropna().unique()))
    return days


def next_trading_day(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    cal = pd.DatetimeIndex(calendar).sort_values()
    arr = pd.to_datetime(dates, errors="coerce")
    pos = cal.searchsorted(arr, side="right")
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    ok = arr.notna() & (pos < len(cal))
    out.loc[ok] = cal[pos[ok]]
    return out


def load_hermes_table(
    name: str,
    columns: Iterable[str],
    *,
    instruments: Iterable[str] | None = None,
) -> pd.DataFrame:
    root = HERMES_V7 / name
    if not root.exists():
        raise RuntimeError(f"BLOCKED_DATA missing Hermes table {name}: {root}")
    cols = list(dict.fromkeys(columns))
    try:
        dataset = ds.dataset(str(root), format="parquet")
        table = dataset.to_table(columns=cols)
        frame = table.to_pandas()
    except Exception:
        files = sorted(root.rglob("*.parquet"))
        parts = []
        for path in files:
            try:
                part = pd.read_parquet(path, columns=[c for c in cols if c in pq.read_schema(path).names])
            except Exception:
                continue
            parts.append(part)
        if not parts:
            raise RuntimeError(f"BLOCKED_DATA cannot read Hermes table {name}: {root}")
        frame = pd.concat(parts, ignore_index=True)
    if "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if "available_at" in frame.columns:
        frame["available_at"] = pd.to_datetime(frame["available_at"])
    if "instrument_id" in frame.columns:
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        if instruments is not None:
            want = set(str(x) for x in instruments)
            frame = frame.loc[frame["instrument_id"].isin(want)].copy()
    return frame


def ic_window(panel: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(IC_START)
    end = pd.Timestamp(str(SIGNAL_END))
    return panel.loc[panel["trade_date"].between(start, end)].copy()


def asof_by_instrument(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    by: str = "instrument_id",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    keep_right = [by, right_on] + [c for c in (columns or []) if c not in {by, right_on}]
    r = right.loc[:, keep_right].dropna(subset=[by, right_on]).copy()
    l = left.copy()
    if by in l.columns:
        l[by] = l[by].astype(str)
    r[by] = r[by].astype(str)
    if np.issubdtype(pd.Series(l[left_on]).dtype, np.datetime64) or str(l[left_on].dtype).startswith("datetime"):
        l[left_on] = pd.to_datetime(l[left_on], errors="coerce")
        r[right_on] = pd.to_datetime(r[right_on], errors="coerce")
    l = l.dropna(subset=[by, left_on])
    r = r.dropna(subset=[by, right_on])
    pieces: list[pd.DataFrame] = []
    r_groups = {k: v.sort_values(right_on, kind="mergesort") for k, v in r.groupby(by, sort=False)}
    for iid, lgrp in l.groupby(by, sort=False):
        left_g = lgrp.sort_values(left_on, kind="mergesort")
        right_g = r_groups.get(iid)
        if right_g is None or right_g.empty:
            pieces.append(left_g)
            continue
        pieces.append(
            pd.merge_asof(
                left_g,
                right_g.drop(columns=[by]),
                left_on=left_on,
                right_on=right_on,
                direction="backward",
            )
        )
    if not pieces:
        return l
    return pd.concat(pieces, ignore_index=True)
