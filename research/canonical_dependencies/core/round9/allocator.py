"""Top2 replacement allocator：只输出 0/5/10% NAV，不再乘 E02 的 1.00/0.75/0.50。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..paths import ROUND9_ROOT, assert_project_result_path, ensure_round9_tree
from ..round8.overlays import MIN_KEEP
from .contract import (
    AGREE_SCORE,
    B01_CELL,
    B03_CELL,
    SEP_LOW,
    SEP_MID,
    SEP_SCORE,
    TOP2_LADDER,
    TOP2_NAV,
)


def sep_bucket(percentile: pd.Series) -> pd.Series:
    p = pd.to_numeric(percentile, errors="coerce")
    out = pd.Series("MID", index=p.index, dtype=object)
    out = out.where(~p.le(SEP_LOW), "LOW")
    out = out.where(~p.gt(SEP_MID), "HIGH")
    return out.where(p.notna(), "MID")


def snap_nav(value: float) -> float:
    choices = np.asarray(TOP2_LADDER, dtype=float)
    return float(choices[int(np.argmin(np.abs(choices - float(value))))])


def nav_to_mult(nav_frac: pd.Series) -> pd.Series:
    x = pd.to_numeric(nav_frac, errors="coerce").fillna(0.0).clip(0.0, TOP2_NAV)
    mult = x / float(TOP2_NAV)
    return mult.where(mult > 0, MIN_KEEP).clip(MIN_KEEP, 1.0)


def lookup_cell(agree: pd.Series, sep: pd.Series, table: Mapping[tuple[str, str], float]) -> pd.Series:
    keys = list(zip(agree.astype(str), sep.astype(str)))
    vals = [float(table.get((a, s), 0.0)) for a, s in keys]
    return pd.Series(vals, index=agree.index, dtype=float)


def b01_nav(agree: pd.Series, sep: pd.Series) -> pd.Series:
    return lookup_cell(agree, sep, B01_CELL)


def b03_nav(agree: pd.Series, sep: pd.Series) -> pd.Series:
    return lookup_cell(agree, sep, B03_CELL)


def b02_nav(agree: pd.Series, sep: pd.Series) -> pd.Series:
    a = agree.astype(str).map(AGREE_SCORE).astype(float).fillna(0.0)
    s = sep.astype(str).map(SEP_SCORE).astype(float).fillna(0.0)
    c2 = 0.5 * a + 0.5 * s
    out = pd.Series(0.0, index=agree.index, dtype=float)
    out = out.where(~c2.ge(0.25), 0.05)
    out = out.where(~c2.ge(0.75), 0.10)
    return out


def write_alloc(code: str, dates: pd.DatetimeIndex | pd.Series, nav_frac: pd.Series) -> Any:
    ensure_round9_tree()
    frame = pd.DataFrame({
        "signal_date": pd.to_datetime(pd.Index(dates)).normalize(),
        "top2_nav_frac": pd.to_numeric(nav_frac, errors="coerce").to_numpy(),
    })
    frame["top2_nav_frac"] = frame["top2_nav_frac"].map(snap_nav)
    path = ROUND9_ROOT / "ledger" / f"{code}_top2_nav.parquet"
    assert_project_result_path(path)
    frame.to_parquet(path, index=False)
    return path


def load_alloc(code: str) -> pd.Series:
    path = ROUND9_ROOT / "ledger" / f"{code}_top2_nav.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    return frame.drop_duplicates("signal_date").set_index("signal_date")["top2_nav_frac"].astype(float)


def activity_stats(nav_frac: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(nav_frac, errors="coerce").dropna()
    if x.empty:
        return {"n": 0, "zero_share": None, "mean_nav": None, "flag": "EMPTY"}
    zero_share = float((x <= 1e-12).mean())
    from .contract import LOW_ACTIVITY_ZERO_SHARE

    flag = "LOW_ACTIVITY" if zero_share >= LOW_ACTIVITY_ZERO_SHARE else "OK"
    return {
        "n": int(len(x)),
        "zero_share": zero_share,
        "share_0": float((x <= 1e-12).mean()),
        "share_5": float(np.isclose(x, 0.05).mean()),
        "share_10": float(np.isclose(x, 0.10).mean()),
        "mean_nav": float(x.mean()),
        "flag": flag,
    }


__all__ = [
    "activity_stats",
    "b01_nav",
    "b02_nav",
    "b03_nav",
    "load_alloc",
    "nav_to_mult",
    "sep_bucket",
    "snap_nav",
    "write_alloc",
]
