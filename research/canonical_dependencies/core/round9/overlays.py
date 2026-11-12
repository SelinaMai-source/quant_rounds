"""Round9 overlay：复用 round8 apply_rank_multipliers / build_overlayed。不改 fused score。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..round8.overlays import MIN_KEEP, apply_rank_multipliers, build_overlayed as r8_build_overlayed
from ..round8.overlays import exact_v2_date_mult, market_b01a
from .contract import E02_OVERLAY

IMPLEMENTED_KINDS = {"exact_e02", "e02", "v3", "custom", "exact_v2", "exact_p08", "preregistered"}


def v3_date_mult(predictions: pd.DataFrame):
    """v2 底（B01a × top1-median），不含 E02 Top2 乘子。Top1 LOCK。"""
    from ..round8.overlays import confidence_from_kind, market_b01a, product_date
    from .geometry import build_geometry

    geo = build_geometry(predictions)
    idx = geo.index
    market = market_b01a(predictions).reindex(idx).fillna(1.0)
    conf = confidence_from_kind("v2", geo).reindex(idx).fillna(1.0)
    return product_date(market, conf), geo


def _preregistered(spec: dict[str, Any], predictions: pd.DataFrame):
    from ..round2.overlays import overlay_summary
    from ..round8.overlays import apply_rank_multipliers
    from .allocator import b01_nav, b02_nav, b03_nav, load_alloc, nav_to_mult, sep_bucket

    date_mult, geo = v3_date_mult(predictions)
    idx = date_mult.index
    agree = geo["head_state"].astype(str).reindex(idx)
    sep = sep_bucket(geo["top2_minus_median_p250"]).reindex(idx)
    name = str(spec.get("allocator") or "")
    alloc_code = spec.get("alloc_code") or spec.get("code")
    if name == "top2_zero":
        nav = pd.Series(0.0, index=idx, dtype=float)
    elif name == "B01_9cell":
        nav = b01_nav(agree, sep)
    elif name == "B02_blend":
        nav = b02_nav(agree, sep)
    elif name == "B03_conservative":
        nav = b03_nav(agree, sep)
    elif name in {"daily_file", "C01_net", "C02_win", "C03_tertile", "C04_isotonic"} or str(name).startswith(("E", "F", "D", "drawdown", "K")):
        key = str(alloc_code or spec.get("alloc_key") or "")
        if not key:
            raise RuntimeError(f"daily allocator missing alloc_code: {spec}")
        loaded = load_alloc(key).reindex(idx)
        nav = loaded.fillna(0.0)
    else:
        raise RuntimeError(f"unknown preregistered allocator: {name}")
    top2_extra = nav_to_mult(nav).reindex(idx).fillna(MIN_KEEP)
    overlayed = apply_rank_multipliers(predictions, date_mult, top2_extra=top2_extra)
    meta = overlay_summary(date_mult)
    meta.update({
        "round9_kind": "replacement_allocator",
        "allocator": name,
        "parent": "R8_E02",
        "top1_lock": True,
        "e02_multiplied": False,
        "top2_nav_mean": float(pd.to_numeric(nav, errors="coerce").mean()) if len(nav) else None,
        "top2_zero_days": int((pd.to_numeric(nav, errors="coerce") <= 1e-12).sum()) if len(nav) else 0,
    })
    return overlayed, meta, {}


def build_overlayed(
    spec: dict[str, Any],
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    kind = spec.get("kind") or "exact_e02"
    if kind == "preregistered":
        return _preregistered(spec, predictions)
    if kind in {"exact_e02", "e02", "v3"}:
        overlayed, meta, cfg = r8_build_overlayed(dict(E02_OVERLAY), predictions)
        meta["round9_kind"] = "exact_e02"
        meta["parent"] = "R8_E02"
        meta["top2_disagree"] = "A=1.00,B=0.75,C=0.50"
        meta["top1_lock"] = True
        return overlayed, meta, cfg
    if kind not in IMPLEMENTED_KINDS:
        raise RuntimeError(f"unknown overlay kind: {kind}")
    return r8_build_overlayed(spec, predictions)


__all__ = [
    "apply_rank_multipliers",
    "build_overlayed",
    "exact_v2_date_mult",
    "market_b01a",
    "v3_date_mult",
]
