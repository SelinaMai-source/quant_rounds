"""Round 5 overlay：复用 R4_B05 市场树；C03 必须按当前模型 margin 重算。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..round2.overlays import apply_date_multiplier, combine_multipliers, margin_multiplier, overlay_summary
from ..round3.overlays import apply_skip_years
from ..round4.overlays import _leaf
from .registry import R5Spec, V3_CONFIDENCE, V3_MARKET


def margin_frame_from_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Top1−Top2 score margin + rolling 250 PIT 分位。只用当前模型分数。"""
    frame = predictions.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    if "instrument_id" in frame:
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    ranked = frame.sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    )
    top1 = (
        ranked.groupby("signal_date", sort=True)
        .head(1)[["signal_date", "instrument_id", "score"]]
        .rename(columns={"instrument_id": "top1_id", "score": "top1_score"})
    )
    second = ranked.groupby("signal_date", sort=True).nth(1)
    if "signal_date" not in getattr(second, "columns", []):
        second = second.reset_index()
    top2 = second[["signal_date", "instrument_id", "score"]].rename(
        columns={"instrument_id": "top2_id", "score": "top2_score"}
    )
    out = top1.merge(top2, on="signal_date", how="left")
    out["top1_top2_margin"] = out["top1_score"] - out["top2_score"]
    out = out.sort_values("signal_date")
    out["margin_p250"] = out["top1_top2_margin"].rolling(250, min_periods=40).rank(pct=True)
    return out.set_index("signal_date").sort_index()


def c03_from_predictions(predictions: pd.DataFrame, spec: dict[str, Any] | None = None) -> pd.Series:
    conf = spec or V3_CONFIDENCE
    agree = margin_frame_from_scores(predictions)
    return margin_multiplier(
        agree["margin_p250"],
        low_mult=float(conf.get("low_mult", 0.5)),
        cutoff=float(conf.get("cutoff", 0.30)),
    )


def build_date_multiplier(
    spec: R5Spec,
    predictions: pd.DataFrame,
) -> tuple[pd.Series, dict[str, Any], dict[str, Any] | None]:
    """市场层走 R4_B05 树；confidence 用当前模型 margin（锁分时与 R4 一致）。"""
    overlay = spec.overlay or {"kind": "stack", "market": V3_MARKET, "confidence": V3_CONFIDENCE}
    meta: dict[str, Any] = {"kind": overlay.get("kind"), "c03_source": "current_model_margin"}
    if overlay.get("kind") == "stack":
        market, m_meta = _leaf(overlay.get("market") or V3_MARKET, predictions)
        conf, c_meta = _leaf(
            {"kind": "identity"},
            predictions,
        )
        # 覆盖 identity：用当前模型 C03
        conf = c03_from_predictions(predictions, overlay.get("confidence") or V3_CONFIDENCE)
        c_meta = {"kind": "margin", "source": "current_model_top1_top2", "cutoff": 0.30, "low_mult": 0.5}
        idx = market.index.union(conf.index)
        mm = market.reindex(idx).fillna(1.0)
        cc = conf.reindex(idx).fillna(1.0)
        series = pd.Series([combine_multipliers(a, b) for a, b in zip(mm, cc)], index=idx)
        extra = overlay.get("stage1")
        if extra is not None:
            gg = extra.reindex(idx).fillna(1.0)
            series = pd.Series([combine_multipliers(a, b) for a, b in zip(series, gg)], index=idx)
            meta["stage1"] = "applied"
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


__all__ = [
    "apply_date_multiplier",
    "attach_stage1",
    "build_date_multiplier",
    "c03_from_predictions",
    "margin_frame_from_scores",
]
