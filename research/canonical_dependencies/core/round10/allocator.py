"""Round10 Top2 allocator：B03 cell / 自定义九格 / 日度文件。默认仍只输出 0/5/10。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..paths import ROUND10_ROOT, assert_project_result_path, ensure_round10_tree
from ..round8.overlays import MIN_KEEP
from ..round9.allocator import lookup_cell, nav_to_mult
from ..round9.contract import B03_CELL, TOP2_NAV
from .contract import SEP_LOW, SEP_MID


def sep_bucket(percentile: pd.Series, *, low: float = SEP_LOW, mid: float = SEP_MID) -> pd.Series:
    p = pd.to_numeric(percentile, errors="coerce")
    out = pd.Series("MID", index=p.index, dtype=object)
    out = out.where(~p.le(float(low)), "LOW")
    out = out.where(~p.gt(float(mid)), "HIGH")
    return out.where(p.notna(), "MID")


def snap_nav(value: float, ladder: tuple[float, ...] = (0.0, 0.05, 0.10)) -> float:
    choices = np.asarray(ladder, dtype=float)
    return float(choices[int(np.argmin(np.abs(choices - float(value))))])


def nav_to_mult_custom(nav_frac: pd.Series, top2_nav: float = TOP2_NAV) -> pd.Series:
    x = pd.to_numeric(nav_frac, errors="coerce").fillna(0.0).clip(0.0, float(top2_nav))
    denom = float(top2_nav) if float(top2_nav) > 0 else 0.10
    mult = x / denom
    return mult.where(mult > 0, MIN_KEEP).clip(MIN_KEEP, 1.0)


def b03_nav(agree: pd.Series, sep: pd.Series) -> pd.Series:
    return lookup_cell(agree, sep, B03_CELL)


def cell_nav(agree: pd.Series, sep: pd.Series, table: Mapping[tuple[str, str], float]) -> pd.Series:
    return lookup_cell(agree, sep, table)


def write_alloc(code: str, dates, nav_frac: pd.Series, *, ladder: tuple[float, ...] | None = None) -> Any:
    ensure_round10_tree()
    frame = pd.DataFrame({
        "signal_date": pd.to_datetime(pd.Index(dates)).normalize(),
        "top2_nav_frac": pd.to_numeric(nav_frac, errors="coerce").to_numpy(),
    })
    if ladder is not None:
        frame["top2_nav_frac"] = frame["top2_nav_frac"].map(lambda x: snap_nav(x, ladder))
    path = ROUND10_ROOT / "ledger" / f"{code}_top2_nav.parquet"
    assert_project_result_path(path)
    frame.to_parquet(path, index=False)
    return path


def load_alloc(code: str) -> pd.Series:
    path = ROUND10_ROOT / "ledger" / f"{code}_top2_nav.parquet"
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
    return {
        "n": int(len(x)),
        "zero_share": zero_share,
        "share_0": float((x <= 1e-12).mean()),
        "share_5": float(np.isclose(x, 0.05).mean()),
        "share_10": float(np.isclose(x, 0.10).mean()),
        "mean_nav": float(x.mean()),
        "flag": "LOW_ACTIVITY" if zero_share >= 0.80 else "OK",
    }


__all__ = [
    "activity_stats",
    "b03_nav",
    "cell_nav",
    "load_alloc",
    "nav_to_mult",
    "nav_to_mult_custom",
    "sep_bucket",
    "snap_nav",
    "write_alloc",
]
