"""把 regime / confidence / risk 乘子合成 allocation_multiplier。不改 score。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MIN_MULT = 1e-12
SKIP_EPS = 1e-6


def clip_unit(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def combine_multipliers(*values: float) -> float:
    product = 1.0
    for value in values:
        product *= clip_unit(float(value))
        if product <= SKIP_EPS:
            return 0.0
    return clip_unit(product)


def apply_date_multiplier(
    predictions: pd.DataFrame,
    date_mult: pd.Series,
) -> pd.DataFrame:
    """date_mult: signal_date → [0, 1]. 0 = drop the whole day (keep Top1 identity)."""
    frame = predictions.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    if "allocation_multiplier" not in frame:
        frame["allocation_multiplier"] = 1.0
    mapped = frame["signal_date"].map(date_mult)
    mapped = pd.to_numeric(mapped, errors="coerce").fillna(1.0)
    skip = mapped <= SKIP_EPS
    frame.loc[~skip, "allocation_multiplier"] = (
        frame.loc[~skip, "allocation_multiplier"].astype(float) * mapped.loc[~skip]
    ).clip(MIN_MULT, 1.0)
    return frame.loc[~skip].copy()


def rolling_percentile(series: pd.Series, window: int = 250) -> pd.Series:
    ordered = series.sort_index()
    return ordered.rolling(window, min_periods=max(40, window // 5)).rank(pct=True)


def tertile_multiplier(
    percentile: pd.Series,
    *,
    low: float = 0.5,
    mid: float = 1.0,
    high: float = 0.5,
) -> pd.Series:
    out = pd.Series(mid, index=percentile.index, dtype=float)
    out = out.where(~percentile.lt(1 / 3), low)
    out = out.where(~percentile.gt(2 / 3), high)
    out = out.where(percentile.notna(), 1.0)
    return out


def breadth_tier_multiplier(
    up_ratio: pd.Series,
    t1: float,
    t2: float,
    t3: float,
    *,
    m1: float = 0.50,
    m2: float = 0.75,
    m3: float = 1.0,
) -> pd.Series:
    """<t1 → 0; [t1,t2) → m1; [t2,t3) → m2; ≥t3 → m3."""
    x = pd.to_numeric(up_ratio, errors="coerce")
    out = pd.Series(0.0, index=x.index, dtype=float)
    out = out.where(~x.ge(t1), m1)
    out = out.where(~x.ge(t2), m2)
    out = out.where(~x.ge(t3), m3)
    out = out.where(x.notna(), 0.0)
    return out


def fail_high_multiplier(fail_pct: pd.Series, threshold: float, high_mult: float) -> pd.Series:
    x = pd.to_numeric(fail_pct, errors="coerce")
    out = pd.Series(1.0, index=x.index, dtype=float)
    out = out.where(~x.ge(threshold), high_mult)
    out = out.where(x.notna(), 1.0)
    return out


def suc_con_weak_multiplier(
    suc_pct: pd.Series,
    con_pct: pd.Series,
    *,
    weak: float = 0.30,
    very_weak: float = 0.10,
) -> pd.Series:
    s = pd.to_numeric(suc_pct, errors="coerce")
    c = pd.to_numeric(con_pct, errors="coerce")
    out = pd.Series(1.0, index=s.index, dtype=float)
    both_weak = s.le(weak) & c.le(weak)
    both_very = s.le(very_weak) & c.le(very_weak)
    out = out.where(~both_weak, 0.5)
    out = out.where(~both_very, 0.0)
    return out


def down_high_multiplier(down_pct: pd.Series, threshold: float, high_mult: float) -> pd.Series:
    x = pd.to_numeric(down_pct, errors="coerce")
    out = pd.Series(1.0, index=x.index, dtype=float)
    out = out.where(~x.ge(threshold), high_mult)
    out = out.where(x.notna(), 1.0)
    return out


def regime_score_five(
    parts: dict[str, pd.Series],
    *,
    hard: bool,
) -> tuple[pd.Series, pd.Series]:
    """Transparent 0–1 parts → 5 buckets. Very Good does not exceed 1.0 (no >30% NAV)."""
    aligned = pd.concat(parts, axis=1).astype(float)
    score = aligned.mean(axis=1, skipna=True)
    # Very Bad / Bad / Neutral / Good / Very Good
    edges = (0.20, 0.40, 0.60, 0.80)
    bucket = pd.cut(
        score,
        bins=[-np.inf, edges[0], edges[1], edges[2], edges[3], np.inf],
        labels=["very_bad", "bad", "neutral", "good", "very_good"],
        right=False,
    )
    hard_map = bucket.map({
        "very_bad": 0.0,
        "bad": 0.33,
        "neutral": 0.67,
        "good": 1.0,
        "very_good": 1.0,
    }).astype(float)
    if hard:
        return score, hard_map.fillna(1.0)
    # soft: piecewise linear between bucket centers
    centers = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
    values = np.array([0.0, 0.33, 0.67, 1.0, 1.0])
    soft = pd.Series(np.interp(score.fillna(0.5), centers, values), index=score.index)
    return score, soft.clip(0.0, 1.0)


def confidence_agreement_multiplier(
    agree_n: pd.Series,
    *,
    mode: str,
) -> pd.Series:
    n = pd.to_numeric(agree_n, errors="coerce").fillna(0)
    if mode == "graded":
        out = pd.Series(0.0, index=n.index, dtype=float)
        out = out.where(~n.ge(3), 0.15 / 0.30)
        out = out.where(~n.ge(4), 0.25 / 0.30)
        out = out.where(~n.ge(5), 1.0)
        return out
    if mode == "ge4":
        return n.ge(4).astype(float)
    if mode == "unanimous":
        return n.ge(5).astype(float)
    raise ValueError(f"unknown confidence mode: {mode}")


def margin_multiplier(margin_pct: pd.Series, *, low_mult: float, cutoff: float = 0.30) -> pd.Series:
    x = pd.to_numeric(margin_pct, errors="coerce")
    out = pd.Series(1.0, index=x.index, dtype=float)
    out = out.where(~x.le(cutoff), low_mult)
    out = out.where(x.notna(), 1.0)
    return out


def probability_multiplier(
    proba: pd.Series,
    *,
    skip_above: float,
    half_above: float,
    invert: bool = False,
) -> pd.Series:
    """If invert=False: high p(loss) → cut. If invert=True: low p(continue) → cut."""
    x = pd.to_numeric(proba, errors="coerce")
    if invert:
        x = 1.0 - x
    out = pd.Series(1.0, index=x.index, dtype=float)
    out = out.where(~x.ge(half_above), 0.5)
    out = out.where(~x.ge(skip_above), 0.0)
    out = out.where(x.notna(), 1.0)
    return out


def overlay_summary(date_mult: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(date_mult, errors="coerce").dropna()
    if x.empty:
        return {"n_days": 0}
    return {
        "n_days": int(len(x)),
        "skip_days": int((x <= SKIP_EPS).sum()),
        "half_days": int(((x > SKIP_EPS) & (x < 0.99)).sum()),
        "full_days": int((x >= 0.99).sum()),
        "mean_mult": float(x.mean()),
        "min_mult": float(x.min()),
    }
