"""Round 6 overlay：市场层锁 R4_B05；C03 按当前模型日内 rank/margin 重算。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..round2.overlays import apply_date_multiplier, combine_multipliers, margin_multiplier, overlay_summary
from ..round3.overlays import apply_skip_years
from ..round4.overlays import _leaf
from .registry import R6Spec, V3_CONFIDENCE, V3_MARKET


def add_intraday_rank(predictions: pd.DataFrame) -> pd.DataFrame:
    """日内 rank-pct。C03 不用绝对 pairwise score。"""
    frame = predictions.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    if "instrument_id" in frame:
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["rank_pct"] = frame.groupby("signal_date")["score"].rank(pct=True)
    return frame


def margin_frame_from_scores(
    predictions: pd.DataFrame,
    *,
    use_intraday_rank: bool = True,
) -> pd.DataFrame:
    """Top1−Top2 margin + rolling 250 PIT 分位。新模型用日内 rank；锁分用原始分以对齐 v3。"""
    frame = predictions.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    if "instrument_id" in frame:
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    value_col = "score"
    if use_intraday_rank:
        frame = add_intraday_rank(frame)
        value_col = "rank_pct"
    ranked = frame.sort_values(
        ["signal_date", value_col, "instrument_id"], ascending=[True, False, True]
    )
    top1 = (
        ranked.groupby("signal_date", sort=True)
        .head(1)[["signal_date", "instrument_id", value_col]]
        .rename(columns={"instrument_id": "top1_id", value_col: "top1_score"})
    )
    second = ranked.groupby("signal_date", sort=True).nth(1)
    if "signal_date" not in getattr(second, "columns", []):
        second = second.reset_index()
    top2 = second[["signal_date", "instrument_id", value_col]].rename(
        columns={"instrument_id": "top2_id", value_col: "top2_score"}
    )
    out = top1.merge(top2, on="signal_date", how="left")
    out["top1_top2_margin"] = out["top1_score"] - out["top2_score"]
    out = out.sort_values("signal_date")
    out["margin_p250"] = out["top1_top2_margin"].rolling(250, min_periods=40).rank(pct=True)
    return out.set_index("signal_date").sort_index()


def c03_from_predictions(
    predictions: pd.DataFrame,
    spec: dict[str, Any] | None = None,
    *,
    use_intraday_rank: bool = True,
) -> pd.Series:
    conf = spec or V3_CONFIDENCE
    agree = margin_frame_from_scores(predictions, use_intraday_rank=use_intraday_rank)
    return margin_multiplier(
        agree["margin_p250"],
        low_mult=float(conf.get("low_mult", 0.5)),
        cutoff=float(conf.get("cutoff", 0.30)),
    )


def build_date_multiplier(
    spec: R6Spec,
    predictions: pd.DataFrame,
) -> tuple[pd.Series, dict[str, Any], dict[str, Any] | None]:
    """市场层走 R4_B05 树；confidence 用当前模型 margin。锁分不用日内 rank，保证 parity。"""
    overlay = spec.overlay or {"kind": "stack", "market": V3_MARKET, "confidence": V3_CONFIDENCE}
    locked = spec.alpha_source == "locked_fs00263"
    use_rank = not locked
    meta: dict[str, Any] = {
        "kind": overlay.get("kind"),
        "c03_source": "locked_raw_margin" if locked else "current_model_intraday_rank_margin",
        "market_layer": "R4_B05",
    }
    if overlay.get("kind") == "stack":
        market, m_meta = _leaf(overlay.get("market") or V3_MARKET, predictions)
        conf = c03_from_predictions(
            predictions,
            overlay.get("confidence") or V3_CONFIDENCE,
            use_intraday_rank=use_rank,
        )
        c_meta = {
            "kind": "margin",
            "source": meta["c03_source"],
            "cutoff": 0.30,
            "low_mult": 0.5,
        }
        idx = market.index.union(conf.index)
        mm = market.reindex(idx).fillna(1.0)
        cc = conf.reindex(idx).fillna(1.0)
        series = pd.Series([combine_multipliers(a, b) for a, b in zip(mm, cc)], index=idx)
        extra = overlay.get("stage1")
        if extra is not None:
            gg = extra.reindex(idx).fillna(1.0)
            series = pd.Series([combine_multipliers(a, b) for a, b in zip(series, gg)], index=idx)
            meta["stage1"] = "applied"
        extra2 = overlay.get("meta_mult")
        if extra2 is not None:
            hh = extra2.reindex(idx).clip(upper=1.0).fillna(1.0)
            series = pd.Series([combine_multipliers(a, b) for a, b in zip(series, hh)], index=idx)
            meta["meta_mult"] = "applied_le_1"
        meta["market"] = m_meta
        meta["confidence"] = c_meta
    else:
        market, m_meta = _leaf(overlay, predictions)
        series = market
        meta.update(m_meta)

    series = apply_skip_years(series, spec.skip_years)
    series.index = pd.to_datetime(series.index)
    series = series.clip(0.0, 1.0)
    if spec.skip_years:
        meta["skip_years"] = list(spec.skip_years)
    meta.update(overlay_summary(series))
    return series, meta, None


def attach_stage1(overlay: dict[str, Any], stage1: pd.Series) -> dict[str, Any]:
    out = dict(overlay)
    out["stage1"] = stage1
    return out


def attach_meta_mult(overlay: dict[str, Any], meta_mult: pd.Series) -> dict[str, Any]:
    out = dict(overlay)
    out["meta_mult"] = meta_mult.clip(upper=1.0)
    return out


def position_cap(
    *,
    nav_frac: float = 0.3,
    market: float = 1.0,
    c03: float = 1.0,
    meta: float = 1.0,
) -> float:
    """NAV×30%×v3_market×当前C03×meta，meta≤1。"""
    m = min(1.0, max(0.0, float(meta)))
    return float(nav_frac) * float(market) * float(c03) * m


__all__ = [
    "add_intraday_rank",
    "apply_date_multiplier",
    "attach_meta_mult",
    "attach_stage1",
    "build_date_multiplier",
    "c03_from_predictions",
    "margin_frame_from_scores",
    "position_cap",
]
