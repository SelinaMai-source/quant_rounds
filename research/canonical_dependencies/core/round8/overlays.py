"""Round8 overlay：父策略 = B01a × top1_minus_median。可拆 rank。不改 fused score。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..round2.overlays import apply_date_multiplier, combine_multipliers, overlay_summary
from ..round4.overlays import build_date_multiplier as r4_build
from ..round4.registry import R4Spec
from .contract import B01A, PARENT_CODE, RANK_NAV, V2_OVERLAY, XS_BOTTOM, XS_LOW_MULT
from .geometry import build_geometry, xs_hard_half, xs_tiers

SKIP_EPS = 1e-6
MIN_KEEP = 1e-12  # Stock 要求 allocation_multiplier ∈ (0, 1]；0 会报错，删行会让 Top3 补位


def _identity(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(1.0, index=idx, dtype=float)


def market_b01a(predictions: pd.DataFrame) -> pd.Series:
    spec = R4Spec(
        code="tmp_mkt",
        title_zh="B01a",
        description_zh="B01a",
        overlay={"kind": "breadth_tier", **B01A},
    )
    series, _, _ = r4_build(spec, predictions)
    return series


def confidence_xs_hard(geo: pd.DataFrame, *, bottom: float = XS_BOTTOM) -> pd.Series:
    return xs_hard_half(geo["top1_minus_median_p250"], bottom=bottom, low_mult=XS_LOW_MULT)


def confidence_from_kind(kind: str, geo: pd.DataFrame, *, bottom: float = XS_BOTTOM) -> pd.Series:
    p = geo["top1_minus_median_p250"]
    if kind in {"v2", "hard", "B04"}:
        return xs_hard_half(p, bottom=bottom, low_mult=0.50)
    if kind == "B01":
        return xs_tiers(p, [(0.50, 0.75), (0.20, 0.50)])
    if kind == "B02":
        return xs_tiers(p, [(0.70, 0.85), (0.40, 0.70), (0.20, 0.50)])
    if kind == "B03":
        x = pd.to_numeric(p, errors="coerce")
        out = (0.5 + 0.5 * x).clip(0.5, 1.0)
        return out.where(x.notna(), 1.0)
    if kind == "F01":
        return xs_hard_half(geo["sep_iqr_p250"], bottom=bottom, low_mult=0.50)
    if kind == "F02":
        return xs_hard_half(geo["sep_mad_p250"], bottom=bottom, low_mult=0.50)
    if kind == "F03":
        return xs_hard_half(geo["top1_minus_median_p250_by_count"], bottom=bottom, low_mult=0.50)
    raise ValueError(kind)


def product_date(*series_list: pd.Series) -> pd.Series:
    idx = series_list[0].index
    for s in series_list[1:]:
        idx = idx.union(s.index)
    out = pd.Series(1.0, index=idx, dtype=float)
    for s in series_list:
        a = s.reindex(idx).fillna(1.0)
        out = pd.Series([combine_multipliers(x, y) for x, y in zip(out, a)], index=idx)
    return out


def apply_rank_multipliers(
    predictions: pd.DataFrame,
    date_mult: pd.Series,
    *,
    top1_extra: pd.Series | None = None,
    top2_extra: pd.Series | None = None,
    drop_top2: pd.Series | None = None,
) -> pd.DataFrame:
    """date_mult 作用于全日；top1/top2 extra 只作用对应 rank。不改 score。

    关掉 Top2 时必须保留原 Top1/Top2 行，把乘子压到 MIN_KEEP。
    若删掉 Top2，Stock `head(top_k)` 会把第三名补上来。
    """
    frame = apply_date_multiplier(predictions, date_mult)
    if frame.empty:
        return frame
    work = frame.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(
        method="first", ascending=False
    )
    if top1_extra is not None:
        mapped = pd.to_numeric(work["signal_date"].map(top1_extra), errors="coerce").fillna(1.0)
        mask = work["_rank"].eq(1)
        work.loc[mask, "allocation_multiplier"] = (
            work.loc[mask, "allocation_multiplier"].astype(float) * mapped.loc[mask]
        ).clip(MIN_KEEP, 1.0)
    if top2_extra is not None:
        mapped = pd.to_numeric(work["signal_date"].map(top2_extra), errors="coerce").fillna(1.0)
        mask = work["_rank"].eq(2)
        scaled = work.loc[mask, "allocation_multiplier"].astype(float) * mapped.loc[mask]
        # 1e-12：保留 selection_rank=2，名义预算≈0，不会改买第三名
        work.loc[mask, "allocation_multiplier"] = scaled.clip(MIN_KEEP, 1.0)
    if drop_top2 is not None:
        flag = work["signal_date"].map(drop_top2.astype("boolean"))
        flag = pd.Series(flag, index=work.index).fillna(False).astype(bool)
        work.loc[work["_rank"].eq(2) & flag, "allocation_multiplier"] = MIN_KEEP
    # 不按乘子删行：date_mult=0 的日子已在 apply_date_multiplier 去掉
    return work.drop(columns=["_rank"], errors="ignore")


def exact_v2_date_mult(predictions: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    spec = R4Spec(
        code=PARENT_CODE,
        title_zh="baseline_v2",
        description_zh="B01a × top1_minus_median 底20%×0.5",
        overlay=dict(V2_OVERLAY),
    )
    series, meta, _ = r4_build(spec, predictions)
    return series, meta


def build_overlayed(
    spec: dict[str, Any],
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """返回 overlay 后的 predictions、meta、config overrides。"""
    geo = build_geometry(predictions)
    idx = geo.index
    kind = spec.get("kind") or "v2"
    bottom = float(spec.get("bottom", XS_BOTTOM))
    cfg: dict[str, Any] = {}
    meta: dict[str, Any] = {"kind": kind, "bottom": bottom}

    if kind == "exact_v2":
        date_mult, extra = exact_v2_date_mult(predictions)
        overlayed = apply_date_multiplier(predictions, date_mult)
        meta.update(extra)
        meta.update(overlay_summary(date_mult))
        return overlayed, meta, cfg

    if kind == "exact_p08":
        spec_r4 = R4Spec(
            code="R4_P_08",
            title_zh="生态+截面",
            description_zh="min(B01a, 连板率半仓) × top1_minus_median",
            overlay=spec["overlay"],
        )
        date_mult, extra, _ = r4_build(spec_r4, predictions)
        overlayed = apply_date_multiplier(predictions, date_mult)
        meta.update(extra)
        meta.update(overlay_summary(date_mult))
        return overlayed, meta, cfg

    market = market_b01a(predictions).reindex(idx).fillna(1.0)
    conf_kind = spec.get("conf_kind", "v2")
    conf = confidence_from_kind(conf_kind, geo, bottom=bottom).reindex(idx).fillna(1.0)
    extra = _identity(idx)

    if spec.get("ecology") == "always":
        eco = _ecology_relimit(predictions).reindex(idx).fillna(1.0)
        market = pd.Series([combine_multipliers(a, b) for a, b in zip(market, eco)], index=idx)
        meta["ecology"] = "min_b01a_relimit"
    elif spec.get("ecology") == "low_sep":
        eco = _ecology_relimit(predictions).reindex(idx).fillna(1.0)
        low = geo["top1_minus_median_p250"].le(0.20).reindex(idx).fillna(False)
        eco = eco.where(low, 1.0)
        market = pd.Series([combine_multipliers(a, b) for a, b in zip(market, eco)], index=idx)
        meta["ecology"] = "only_top1_bottom20"
    elif spec.get("ecology") == "both_low":
        eco = _ecology_relimit(predictions).reindex(idx).fillna(1.0)
        low = (
            geo["top1_minus_median_p250"].le(0.20)
            & geo["top2_minus_median_p250"].le(0.20)
        ).reindex(idx).fillna(False)
        eco = eco.where(low, 1.0)
        market = pd.Series([combine_multipliers(a, b) for a, b in zip(market, eco)], index=idx)
        meta["ecology"] = "both_bottom20"

    # disagreement extra
    e_mode = spec.get("disagree")
    if e_mode:
        extra = extra * _disagree_mult(geo, e_mode)

    if spec.get("h01"):
        low = geo["top1_minus_median_p250"].le(0.20)
        high_d = geo["head_state"].eq("C")
        hit = (low & high_d).reindex(idx).fillna(False)
        extra = extra.where(~hit, extra * 0.5)
        meta["h01"] = int(hit.sum())

    rank_mode = spec.get("rank_mode") or "shared"
    top1_extra = _identity(idx)
    top2_extra = _identity(idx)
    drop_top2 = pd.Series(False, index=idx)

    if rank_mode == "shared":
        date_mult = product_date(market, conf, extra)
    elif rank_mode == "top1_only":
        date_mult = product_date(market, extra)
        top1_extra = conf
    elif rank_mode == "top2_only":
        date_mult = product_date(market, extra)
        top2_extra = conf
    elif rank_mode == "separate":
        date_mult = product_date(market, extra)
        top1_extra = xs_hard_half(geo["top1_minus_median_p250"], bottom=bottom)
        top2_extra = xs_hard_half(geo["top2_minus_median_p250"], bottom=bottom)
        meta["rank_mode"] = "separate_geometry"
    else:
        date_mult = product_date(market, conf, extra)

    # C04
    if spec.get("c04"):
        strong = geo["top1_minus_median_p250"].gt(0.50)
        weak2 = geo["top2_minus_median_p250"].le(0.20)
        drop_top2 = (strong & weak2).reindex(idx).fillna(False)
        meta["c04_drop_days"] = int(drop_top2.sum())

    # D rules on Top2 budget (relative to 10% base unless d04)
    d_rule = spec.get("d_rule")
    if d_rule == "D01":
        drop_top2 = drop_top2 | geo["top2_minus_median_p250"].le(0.20).reindex(idx).fillna(False)
    elif d_rule == "D02":
        low2 = geo["top2_minus_median_p250"].le(0.20).reindex(idx).fillna(False)
        top2_extra = top2_extra.where(~low2, top2_extra * 0.5)
    elif d_rule == "D03":
        p2 = pd.to_numeric(geo["top2_minus_median_p250"], errors="coerce").reindex(idx)
        top2_extra = top2_extra.where(~p2.le(0.50), top2_extra * 0.5)
        drop_top2 = drop_top2 | p2.le(0.20).fillna(False)
    elif d_rule == "D04":
        cfg["rank_name_budget_nav_fractions"] = (0.3, 0.15)
        strong2 = geo["top2_minus_median_p250"].gt(0.50).reindex(idx).fillna(False)
        # base Top2=15%; non-strong days scale to 10%
        top2_extra = top2_extra.where(strong2, top2_extra * (0.10 / 0.15))
        meta["d04_strong_days"] = int(strong2.sum())

    if spec.get("h02"):
        low2 = geo["top2_minus_median_p250"].le(0.20).reindex(idx).fillna(False)
        drop_top2 = drop_top2 | low2
        meta["h02"] = "drop_top2_when_low_sep"

    if spec.get("ecology") == "top2_only":
        eco = _ecology_relimit(predictions).reindex(idx).fillna(1.0)
        top2_extra = pd.Series(
            [combine_multipliers(a, b) for a, b in zip(top2_extra.reindex(idx).fillna(1.0), eco)],
            index=idx,
        )
        meta["ecology"] = "top2_only"

    if spec.get("e02"):
        top2_extra = top2_extra * _disagree_mult(geo, "E01").reindex(idx).fillna(1.0)
        extra = _identity(idx)
        date_mult = product_date(market, conf)

    if spec.get("top_k") == 1:
        cfg["top_k"] = 1
        cfg["rank_name_budget_nav_fractions"] = (0.3,)

    overlayed = apply_rank_multipliers(
        predictions,
        date_mult,
        top1_extra=top1_extra,
        top2_extra=top2_extra,
        drop_top2=drop_top2,
    )
    meta.update(overlay_summary(date_mult))
    meta["conf_kind"] = conf_kind
    meta["rank_mode"] = rank_mode
    return overlayed, meta, cfg


def _ecology_relimit(predictions: pd.DataFrame) -> pd.Series:
    spec = R4Spec(
        code="tmp_eco",
        title_zh="relimit",
        description_zh="relimit",
        overlay={"kind": "eco_relimit_p250", "low": 0.20, "low_mult": 0.5},
    )
    series, _, _ = r4_build(spec, predictions)
    return series


def _disagree_mult(geo: pd.DataFrame, mode: str) -> pd.Series:
    state = geo["head_state"].astype(str)
    mapped = state.map({"A": 1.00, "B": 0.75, "C": 0.50}).astype(float)
    out = mapped.fillna(1.0)
    if mode == "E03":
        low = geo["top1_minus_median_p250"].le(0.20)
        hit = low & state.eq("C")
        extra = pd.Series(1.0, index=geo.index)
        extra = extra.where(~hit, 0.5)
        return extra
    return out


__all__ = [
    "apply_rank_multipliers",
    "build_overlayed",
    "confidence_from_kind",
    "exact_v2_date_mult",
    "market_b01a",
]
