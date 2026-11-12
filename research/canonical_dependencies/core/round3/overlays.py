"""Round 3 overlay：复用 Round 2 乘子；G02 提供 rolling 250 PIT 版。不改 score。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..round2.data import SEEDS, load_all_seed_predictions, load_candidate_features, seed_top1_agreement
from ..round2.overlays import (
    apply_date_multiplier,
    combine_multipliers,
    confidence_agreement_multiplier,
    margin_multiplier,
    overlay_summary,
    rolling_percentile,
)
from ..round2.registry import R2Spec
from ..round2.runner import build_date_multiplier as r2_build_date_multiplier
from ..stock_readonly import load_fs00263_ensemble_predictions
from .registry import R2_CHILD_OVERLAYS, R3Spec

_CACHE: dict[str, Any] = {}


def _cached(key: str, factory):
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def agreement_frame(
    predictions: pd.DataFrame | None = None,
    *,
    seeds: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    use_seeds = tuple(seeds) if seeds else SEEDS
    cache_key = f"agree:{','.join(str(s) for s in use_seeds)}"

    def _build():
        ens = predictions if predictions is not None else load_fs00263_ensemble_predictions()
        all_seeds = load_all_seed_predictions()
        subset = all_seeds.loc[all_seeds["seed"].isin(use_seeds)].copy()
        return seed_top1_agreement(ens, subset)

    return _cached(cache_key, _build)


def confidence_ratio_multiplier(agree_n: pd.Series, n_seeds: int) -> pd.Series:
    """5/5=1.0 满仓；比例阈值与人数档相同，不因少一只 seed 改人数门槛。"""
    n = pd.to_numeric(agree_n, errors="coerce").fillna(0)
    ratio = n / float(max(n_seeds, 1))
    out = pd.Series(0.0, index=n.index, dtype=float)
    out = out.where(~ratio.ge(0.60), 0.15 / 0.30)
    out = out.where(~ratio.ge(0.80), 0.25 / 0.30)
    out = out.where(~ratio.ge(1.0 - 1e-12), 1.0)
    return out


def industry_crowd_series(
    predictions: pd.DataFrame,
    *,
    half: float,
    skip: float,
    pit: bool,
) -> tuple[pd.Series, dict[str, Any]]:
    ens = predictions.copy()
    ens["signal_date"] = pd.to_datetime(ens["signal_date"])
    top1 = (
        ens.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True])
        .groupby("signal_date", sort=True)
        .head(1)
    )
    panel = _cached("panel_ind", lambda: load_candidate_features(
        ["trade_date", "instrument_id", "ind_first_board_count_log"]
    ))
    merged = top1.merge(
        panel,
        left_on=["signal_date", "instrument_id"],
        right_on=["trade_date", "instrument_id"],
        how="left",
    ).sort_values("signal_date")
    raw = merged.set_index(pd.to_datetime(merged["signal_date"]))["ind_first_board_count_log"]
    if pit:
        ranked = rolling_percentile(raw, window=250)
        norm = "rolling_250_rank_include_today"
    else:
        ranked = raw.rank(pct=True)
        norm = "full_sample_rank_INVALID"
    series = pd.Series(1.0, index=ranked.index, dtype=float)
    series = series.where(~ranked.ge(half), 0.5)
    series = series.where(~ranked.ge(skip), 0.0)
    series = series.where(ranked.notna(), 1.0)
    meta = {
        "kind": "industry_crowd_pit" if pit else "industry_crowd",
        "half": half,
        "skip": skip,
        "normalization": norm,
        "uses_future": (not pit),
        "n_valid": int(ranked.notna().sum()),
    }
    return series, meta


def _r2_child_spec(code: str) -> R2Spec:
    overlay = dict(R2_CHILD_OVERLAYS.get(code) or {"kind": "identity"})
    market_min = 0.40 if code == "R2_B01b" else 0.45
    return R2Spec(
        code=code,
        module="R3",
        title_zh=code,
        description_zh=code,
        overlay=overlay,
        market_up_ratio_min=market_min,
    )


def _align_min_product(a: pd.Series, b: pd.Series) -> pd.Series:
    idx = a.index.union(b.index)
    aa = a.reindex(idx).fillna(1.0)
    bb = b.reindex(idx).fillna(1.0)
    return pd.Series([combine_multipliers(x, y) for x, y in zip(aa, bb)], index=idx)


def apply_skip_years(series: pd.Series, years: tuple[int, ...]) -> pd.Series:
    if not years:
        return series
    out = series.copy()
    out.index = pd.to_datetime(out.index)
    mask = out.index.year.isin(years)
    out.loc[mask] = 0.0
    return out


def _identity_series(predictions: pd.DataFrame) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(predictions["signal_date"].unique())).sort_values()
    return pd.Series(1.0, index=idx)


def build_date_multiplier(
    spec: R3Spec,
    predictions: pd.DataFrame,
) -> tuple[pd.Series, dict[str, Any], dict[str, Any] | None]:
    """Return (date_mult, overlay_meta, overlay_hooks)."""
    kind = (spec.overlay or {}).get("kind")
    hooks: dict[str, Any] | None = None
    meta: dict[str, Any] = {"kind": kind}

    if kind in (None, "identity"):
        series = _identity_series(predictions)
    elif kind == "industry_crowd":
        series, extra = industry_crowd_series(
            predictions,
            half=float(spec.overlay.get("half", 0.80)),
            skip=float(spec.overlay.get("skip", 0.90)),
            pit=bool(spec.pit_correct_g02 or spec.overlay.get("pit")),
        )
        meta.update(extra)
    elif kind == "industry_crowd_pit":
        series, extra = industry_crowd_series(
            predictions,
            half=float(spec.overlay.get("half", 0.80)),
            skip=float(spec.overlay.get("skip", 0.90)),
            pit=True,
        )
        meta.update(extra)
    elif kind == "confidence_agree_ratio":
        use_seeds = spec.seeds or SEEDS
        if spec.seed_leave_out is not None:
            use_seeds = tuple(s for s in use_seeds if s != spec.seed_leave_out)
        agree = agreement_frame(predictions, seeds=use_seeds)
        series = confidence_ratio_multiplier(agree["agree_n"], n_seeds=len(use_seeds))
        meta.update({"seeds": list(use_seeds), "n_seeds": len(use_seeds), "mode": "ratio"})
    elif kind == "dual_conf":
        agree = agreement_frame(predictions)
        a = confidence_agreement_multiplier(agree["agree_n"], mode="graded")
        m = margin_multiplier(agree["margin_p250"], low_mult=0.5, cutoff=0.30)
        idx = a.index.union(m.index)
        aa = a.reindex(idx).fillna(1.0)
        mm = m.reindex(idx).fillna(1.0)
        how = spec.overlay.get("combine", "min")
        if how == "intersect":
            series = pd.Series([combine_multipliers(x, y) for x, y in zip(aa, mm)], index=idx)
        elif how == "mean":
            series = (aa + mm) / 2.0
        else:
            series = pd.concat([aa, mm], axis=1).min(axis=1)
        meta["combine"] = how
    elif kind == "product":
        product = None
        hooks_acc: dict[str, Any] = {}
        for child_code in spec.overlay.get("components") or []:
            child_overlay = dict(R2_CHILD_OVERLAYS.get(child_code) or {"kind": "identity"})
            child = R3Spec(
                code=child_code,
                title_zh=child_code,
                description_zh=child_code,
                overlay=child_overlay,
                pit_correct_g02=spec.pit_correct_g02,
                market_up_ratio_min=0.40 if child_code == "R2_B01b" else 0.45,
            )
            child_series, child_meta, child_hooks = build_date_multiplier(child, predictions)
            product = child_series if product is None else _align_min_product(product, child_series)
            if child_hooks:
                hooks_acc.update(child_hooks)
            meta.setdefault("components", []).append({"code": child_code, **child_meta})
        series = product if product is not None else _identity_series(predictions)
        hooks = hooks_acc or None
    else:
        fake = _r2_child_spec(spec.code)
        fake.overlay = dict(spec.overlay)
        fake.market_up_ratio_min = spec.market_up_ratio_min
        series, r2_meta, hooks = r2_build_date_multiplier(fake, predictions)
        meta.update(r2_meta)

    series = apply_skip_years(series, spec.skip_years)
    series.index = pd.to_datetime(series.index)
    if spec.skip_years:
        meta["skip_years"] = list(spec.skip_years)
    meta.update(overlay_summary(series))
    return series, meta, hooks


__all__ = [
    "apply_date_multiplier",
    "agreement_frame",
    "build_date_multiplier",
    "confidence_ratio_multiplier",
    "industry_crowd_series",
]
