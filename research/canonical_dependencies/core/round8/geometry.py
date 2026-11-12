"""每日横截面几何。分数 = Round7 融合分（已是同日 percentile 后再 90/10）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import PREDICTION_START, SIGNAL_END, STOCK_ROUND7_ROOT
from ..replay_round7.contract import GAP_FLOOR, TEN_PANEL
from ..round2.overlays import rolling_percentile
from .contract import (
    FIVE_HEAD_PRED,
    ROLLING_WINDOW,
    TEN_HEAD_PRED,
    TEN_WEIGHT,
)

_CACHE: dict[str, Any] = {}
EPS = 1e-12


def _clip(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["instrument_id"] = out["instrument_id"].astype(str)
    return out.loc[out["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)].copy()


def load_fused_alpha() -> pd.DataFrame:
    from ..replay_round7.inject import load_locked_alpha

    return _CACHE.setdefault("alpha", load_locked_alpha())


def load_head_ensembles() -> tuple[pd.DataFrame, pd.DataFrame]:
    if "heads" not in _CACHE:
        five = _clip(pd.read_parquet(FIVE_HEAD_PRED, columns=["signal_date", "instrument_id", "score"]))
        ten = _clip(pd.read_parquet(TEN_HEAD_PRED, columns=["signal_date", "instrument_id", "score"]))
        _CACHE["heads"] = (five, ten)
    return _CACHE["heads"]


def load_gaps() -> pd.DataFrame:
    if "gaps" not in _CACHE:
        gaps = pd.read_parquet(
            TEN_PANEL / "candidate_panel.parquet",
            columns=["trade_date", "instrument_id", "d1_open_gap"],
        ).rename(columns={"trade_date": "signal_date"})
        gaps["signal_date"] = pd.to_datetime(gaps["signal_date"])
        gaps["instrument_id"] = gaps["instrument_id"].astype(str)
        _CACHE["gaps"] = gaps
    return _CACHE["gaps"]


def eligible_heads() -> pd.DataFrame:
    """与 blend 同一宇宙：finite d1_open_gap ≥ -0.01。"""
    if "eligible_heads" not in _CACHE:
        five, ten = load_head_ensembles()
        keys = ["signal_date", "instrument_id"]
        work = five.rename(columns={"score": "five_score"}).merge(
            ten.rename(columns={"score": "ten_score"}), on=keys, how="inner", validate="one_to_one"
        )
        work = work.merge(load_gaps(), on=keys, how="left")
        work["d1_open_gap"] = pd.to_numeric(work["d1_open_gap"], errors="coerce")
        work = work.loc[np.isfinite(work["d1_open_gap"]) & work["d1_open_gap"].ge(GAP_FLOOR)].copy()
        work["pct_five"] = work.groupby("signal_date")["five_score"].rank(method="average", pct=True)
        work["pct_ten"] = work.groupby("signal_date")["ten_score"].rank(method="average", pct=True)
        work["fused_score"] = TEN_WEIGHT * work["pct_ten"] + (1.0 - TEN_WEIGHT) * work["pct_five"]
        _CACHE["eligible_heads"] = work
    return _CACHE["eligible_heads"]


def _robust_z(top: float, med: float, values: np.ndarray) -> float:
    mad = float(np.median(np.abs(values - med)))
    return (top - med) / (mad + EPS)


def build_geometry(predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """每日几何 + rolling250 分位。index = signal_date。"""
    if "geometry" in _CACHE and predictions is None:
        return _CACHE["geometry"]

    fused = load_fused_alpha() if predictions is None else predictions.copy()
    fused["signal_date"] = pd.to_datetime(fused["signal_date"])
    fused["instrument_id"] = fused["instrument_id"].astype(str)
    fused["score"] = pd.to_numeric(fused["score"], errors="coerce")
    heads = eligible_heads()
    head_top = []
    for day, block in heads.groupby("signal_date", sort=True):
        five_ord = block.sort_values(["five_score", "instrument_id"], ascending=[False, True])
        ten_ord = block.sort_values(["ten_score", "instrument_id"], ascending=[False, True])
        five_top = five_ord["instrument_id"].head(2).tolist()
        ten_top = ten_ord["instrument_id"].head(2).tolist()
        f1 = five_top[0] if five_top else ""
        t1 = ten_top[0] if ten_top else ""
        f2 = set(five_top)
        t2 = set(ten_top)
        if f1 and f1 == t1:
            state = "A"
        elif f1 in t2 and t1 in f2:
            state = "B"
        else:
            state = "C"
        if len(block) >= 3:
            rho = float(block["five_score"].corr(block["ten_score"], method="spearman"))
        else:
            rho = np.nan
        five_rank = {iid: i for i, iid in enumerate(five_ord["instrument_id"].tolist(), start=1)}
        ten_rank = {iid: i for i, iid in enumerate(ten_ord["instrument_id"].tolist(), start=1)}
        top1_dist = abs(five_rank.get(f1, np.nan) - ten_rank.get(f1, np.nan)) if f1 else np.nan
        # Top1 rank distance between heads' own Top1 names
        head_top1_dist = abs(1 - ten_rank.get(f1, np.nan)) if f1 else np.nan
        head_top.append({
            "signal_date": pd.Timestamp(day),
            "five_top1": f1,
            "ten_top1": t1,
            "five_top2": five_top[1] if len(five_top) > 1 else "",
            "ten_top2": ten_top[1] if len(ten_top) > 1 else "",
            "head_state": state,
            "rank_corr": rho,
            "top1_rank_distance": float(head_top1_dist) if np.isfinite(head_top1_dist) else np.nan,
            "top2_rank_distance": float(abs(
                five_rank.get(five_top[1], np.nan) - ten_rank.get(five_top[1], np.nan)
            )) if len(five_top) > 1 else np.nan,
        })
    head_df = pd.DataFrame(head_top).set_index("signal_date") if head_top else pd.DataFrame()

    rows = []
    for day, block in fused.groupby("signal_date", sort=True):
        s = pd.to_numeric(block["score"], errors="coerce").dropna()
        if s.empty:
            continue
        vals = s.to_numpy(dtype=float)
        order = block.loc[s.index].sort_values(["score", "instrument_id"], ascending=[False, True])
        top1_id = str(order["instrument_id"].iloc[0])
        top2_id = str(order["instrument_id"].iloc[1]) if len(order) > 1 else ""
        top1 = float(s.max())
        top2 = float(order["score"].iloc[1]) if len(order) > 1 else np.nan
        med = float(s.median())
        p75 = float(s.quantile(0.75))
        p25 = float(s.quantile(0.25))
        iqr = p75 - p25
        mu = float(s.mean())
        sd = float(s.std(ddof=0)) if len(s) > 1 else np.nan
        mad = float(np.median(np.abs(vals - med)))
        rows.append({
            "signal_date": pd.Timestamp(day),
            "top1_id": top1_id,
            "top2_id": top2_id,
            "top1_score": top1,
            "top2_score": top2,
            "median": med,
            "p75": p75,
            "p25": p25,
            "iqr": iqr,
            "mad": mad,
            "mean": mu,
            "std": sd,
            "top1_minus_median": top1 - med,
            "top2_minus_median": (top2 - med) if np.isfinite(top2) else np.nan,
            "top1_minus_top2": (top1 - top2) if np.isfinite(top2) else np.nan,
            "top1_minus_p75": top1 - p75,
            "top1_z_robust": _robust_z(top1, med, vals),
            "top2_z_robust": _robust_z(top2, med, vals) if np.isfinite(top2) else np.nan,
            "sep_iqr": (top1 - med) / (iqr + EPS),
            "sep_mad": (top1 - med) / (mad + EPS),
            "candidate_count": int(len(s)),
            "small_candidate_day": int(len(s) <= 3),
        })
    geo = pd.DataFrame(rows).set_index("signal_date").sort_index()
    if not head_df.empty:
        geo = geo.join(head_df, how="left")

    pit_cols = (
        "top1_minus_median",
        "top2_minus_median",
        "top1_minus_top2",
        "top1_z_robust",
        "top2_z_robust",
        "iqr",
        "sep_iqr",
        "sep_mad",
        "candidate_count",
    )
    for col in pit_cols:
        geo[f"{col}_p250"] = rolling_percentile(pd.to_numeric(geo[col], errors="coerce"), ROLLING_WINDOW)

    # F03: same-bin historical PIT of top1_minus_median
    bins = pd.cut(
        geo["candidate_count"],
        bins=[0, 3, 6, np.inf],
        labels=["1-3", "4-6", "7+"],
        right=True,
    )
    geo["count_bin"] = bins.astype(str)
    cond = pd.Series(np.nan, index=geo.index, dtype=float)
    for _, idx in geo.groupby("count_bin").groups.items():
        cond.loc[idx] = rolling_percentile(geo.loc[idx, "top1_minus_median"], ROLLING_WINDOW)
    geo["top1_minus_median_p250_by_count"] = cond

    if predictions is None:
        _CACHE["geometry"] = geo
    return geo


def xs_hard_half(percentile: pd.Series, *, bottom: float = 0.20, low_mult: float = 0.50) -> pd.Series:
    p = pd.to_numeric(percentile, errors="coerce")
    out = pd.Series(1.0, index=p.index, dtype=float)
    out = out.where(~p.le(bottom), low_mult)
    return out.where(p.notna(), 1.0)


def xs_tiers(percentile: pd.Series, cuts: list[tuple[float, float]]) -> pd.Series:
    """cuts: [(upper_exclusive_or_equal, mult), ...] applied from high cut to low."""
    p = pd.to_numeric(percentile, errors="coerce")
    out = pd.Series(1.0, index=p.index, dtype=float)
    for upper, mult in sorted(cuts, key=lambda x: x[0], reverse=True):
        out = out.where(~p.le(upper), float(mult))
    return out.where(p.notna(), 1.0)


__all__ = [
    "build_geometry",
    "eligible_heads",
    "load_fused_alpha",
    "load_head_ensembles",
    "xs_hard_half",
    "xs_tiers",
]
