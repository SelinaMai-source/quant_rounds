"""Round 4 overlay：复用 R2/R3 乘子 + 生态 / 分差校准 / 二维 / CVaR。不改 score。禁止全样本分位。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND3_ROOT, ROUND4_ROOT
from ..round2.data import SEEDS, seed_top1_agreement
from ..round2.overlays import (
    apply_date_multiplier,
    breadth_tier_multiplier,
    combine_multipliers,
    confidence_agreement_multiplier,
    margin_multiplier,
    overlay_summary,
    rolling_percentile,
)
from ..round3.overlays import agreement_frame as r3_agreement_frame
from ..round3.overlays import apply_skip_years
from .ecology import (
    build_ecology_frame,
    eco_bigloss_multiplier,
    eco_five_multiplier,
    eco_oc_ewma_multiplier,
    eco_relimit_multiplier,
)
from .registry import R4Spec

_CACHE: dict[str, Any] = {}
MIN_GRID_N = 50


def _cached(key: str, factory):
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def agreement_frame(predictions: pd.DataFrame | None = None, *, seeds: tuple[int, ...] | None = None) -> pd.DataFrame:
    return r3_agreement_frame(predictions, seeds=seeds)


def _identity(predictions: pd.DataFrame) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(predictions["signal_date"].unique())).sort_values()
    return pd.Series(1.0, index=idx)


def _align_combine(series_list: list[pd.Series], how: str) -> pd.Series:
    idx = series_list[0].index
    for s in series_list[1:]:
        idx = idx.union(s.index)
    aligned = [s.reindex(idx).fillna(1.0) for s in series_list]
    if how == "min":
        out = aligned[0]
        for s in aligned[1:]:
            out = pd.concat([out, s], axis=1).min(axis=1)
        return out
    if how == "mean":
        return sum(aligned) / float(len(aligned))
    if how == "geom":
        prod = aligned[0].clip(lower=0)
        for s in aligned[1:]:
            prod = prod * s.clip(lower=0)
        return prod.pow(1.0 / len(aligned)).clip(0.0, 1.0)
    # product
    out = aligned[0]
    for s in aligned[1:]:
        out = pd.Series([combine_multipliers(a, b) for a, b in zip(out, s)], index=idx)
    return out


def _leaf(kind_spec: dict[str, Any], predictions: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    kind = (kind_spec or {}).get("kind") or "identity"
    meta: dict[str, Any] = {"kind": kind}
    eco = None

    def _eco():
        nonlocal eco
        if eco is None:
            eco = build_ecology_frame()
        return eco

    if kind in (None, "identity"):
        series = _identity(predictions)
    elif kind == "breadth_tier":
        from ..round2.runner import regime_frame

        series = breadth_tier_multiplier(
            regime_frame()["up_ratio"],
            float(kind_spec["t1"]), float(kind_spec["t2"]), float(kind_spec["t3"]),
        )
    elif kind == "margin":
        agree = agreement_frame(predictions)
        series = margin_multiplier(
            agree["margin_p250"],
            low_mult=float(kind_spec.get("low_mult", 0.5)),
            cutoff=float(kind_spec.get("cutoff", 0.30)),
        )
    elif kind == "confidence_agree":
        agree = agreement_frame(predictions)
        series = confidence_agreement_multiplier(agree["agree_n"], mode=kind_spec.get("mode", "graded"))
    elif kind == "eco_relimit_p250":
        series = eco_relimit_multiplier(
            _eco(), low=float(kind_spec.get("low", 0.20)), low_mult=float(kind_spec.get("low_mult", 0.5)),
        )
        meta["pit"] = "rolling_250"
    elif kind == "eco_oc_ewma":
        series = eco_oc_ewma_multiplier(
            _eco(), skip=float(kind_spec.get("skip", 0.20)), half=float(kind_spec.get("half", 0.40)),
        )
        meta["pit"] = "ewma20_then_rolling_250"
    elif kind == "eco_bigloss":
        series = eco_bigloss_multiplier(
            _eco(), skip=float(kind_spec.get("skip", 0.80)), half=float(kind_spec.get("half", 0.70)),
        )
        meta["pit"] = "rolling_250"
    elif kind == "eco_five":
        _score, series = eco_five_multiplier(_eco(), hard=bool(kind_spec.get("hard", True)))
        meta["pit"] = "rolling_250_five"
        meta["clip_max"] = 1.0
    elif kind == "margin_tiers":
        agree = agreement_frame(predictions)
        p = pd.to_numeric(agree["margin_p250"], errors="coerce")
        mults = tuple(kind_spec.get("mults") or (0.40, 0.55, 0.70, 0.85, 1.00))
        edges = (0.20, 0.40, 0.60, 0.80)
        out = pd.Series(float(mults[-1]), index=p.index, dtype=float)
        out = out.where(~p.lt(edges[3]), float(mults[3]))
        out = out.where(~p.lt(edges[2]), float(mults[2]))
        out = out.where(~p.lt(edges[1]), float(mults[1]))
        out = out.where(~p.lt(edges[0]), float(mults[0]))
        series = out.where(p.notna(), 1.0).clip(0.0, 1.0)
    elif kind == "margin_linear":
        agree = agreement_frame(predictions)
        p = pd.to_numeric(agree["margin_p250"], errors="coerce")
        lo = float(kind_spec.get("lo", 0.4))
        hi = float(kind_spec.get("hi", 1.0))
        series = (lo + (hi - lo) * p).clip(0.0, 1.0)
        series = series.where(p.notna(), 1.0)
    elif kind == "margin_isotonic":
        series, extra = _isotonic_margin(predictions, target=str(kind_spec.get("target", "win")))
        meta.update(extra)
    elif kind == "twodim_prereg":
        series = _twodim_prereg(predictions)
        meta["min_mult"] = 0.25
    elif kind == "twodim_hist":
        series, extra = _twodim_hist(predictions)
        meta.update(extra)
    elif kind == "cvar_rel":
        series, extra = _cvar_series(predictions, mode="rel")
        meta.update(extra)
    elif kind == "cvar_abs":
        series, extra = _cvar_series(predictions, mode="abs")
        meta.update(extra)
    elif kind == "cross_section":
        series, extra = _cross_section_mult(predictions, kind_spec)
        meta.update(extra)
    elif kind == "combine":
        parts = []
        for part in kind_spec.get("parts") or []:
            s, m = _leaf(part, predictions)
            parts.append(s)
            meta.setdefault("parts", []).append(m)
        series = _align_combine(parts, kind_spec.get("how", "min")) if parts else _identity(predictions)
        meta["how"] = kind_spec.get("how", "min")
    else:
        raise ValueError(f"unknown overlay kind: {kind}")

    series.index = pd.to_datetime(series.index)
    series = series.clip(0.0, 1.0)
    return series, meta


def _load_anchor_trades() -> pd.DataFrame:
    for path in (
        ROUND4_ROOT / "R4_00_baseline_v2" / "trades.parquet",
        ROUND3_ROOT / "R3_03" / "trades.parquet",
    ):
        if path.is_file():
            frame = pd.read_parquet(path)
            frame["signal_date"] = pd.to_datetime(frame["signal_date"])
            frame["actual_exit_date"] = pd.to_datetime(frame["actual_exit_date"])
            return frame
    return pd.DataFrame()


def _isotonic_margin(predictions: pd.DataFrame, *, target: str) -> tuple[pd.Series, dict[str, Any]]:
    from sklearn.isotonic import IsotonicRegression

    agree = agreement_frame(predictions)
    trades = _load_anchor_trades()
    if trades.empty or "ret" not in trades:
        return _identity(predictions), {"status": "ABANDONED", "reason": "no_anchor_trades"}

    joined = trades.merge(
        agree.reset_index().rename(columns={"index": "signal_date"}),
        on="signal_date",
        how="left",
    )
    joined = joined.dropna(subset=["top1_top2_margin", "actual_exit_date"])
    if target == "win":
        y_all = (pd.to_numeric(joined["ret"], errors="coerce") > 0).astype(float)
    else:
        y_all = pd.to_numeric(joined["ret"], errors="coerce")
    x_all = pd.to_numeric(joined["top1_top2_margin"], errors="coerce")
    years = sorted(pd.to_datetime(predictions["signal_date"]).dt.year.unique())
    pred_dates = pd.DatetimeIndex(pd.to_datetime(predictions["signal_date"].unique())).sort_values()
    out = pd.Series(1.0, index=pred_dates, dtype=float)
    used = 0
    abandoned = False
    reason = ""
    for year in years:
        cutoff = pd.Timestamp(year, 1, 1)
        train = joined.loc[joined["actual_exit_date"] < cutoff]
        if len(train) < 80:
            continue
        x = pd.to_numeric(train["top1_top2_margin"], errors="coerce")
        y = y_all.loc[train.index]
        mask = x.notna() & y.notna()
        if mask.sum() < 80:
            continue
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(x[mask].to_numpy(), y[mask].to_numpy())
        if target == "ret":
            grid = np.linspace(float(x[mask].min()), float(x[mask].max()), 21)
            pred_grid = iso.predict(grid)
            if np.nanmax(pred_grid) - np.nanmin(pred_grid) < 1e-4:
                abandoned = True
                reason = "flat_calibration"
                break
        year_idx = pred_dates[pred_dates.year == year]
        if len(year_idx) == 0:
            continue
        m = agree.reindex(year_idx)["top1_top2_margin"]
        p = pd.Series(iso.predict(m.fillna(m.median()).to_numpy()), index=year_idx)
        if target == "win":
            base = float(y[mask].mean())
            mult = (p / max(base, 0.30)).clip(0.40, 1.0)
        else:
            p_rank = p.rank(pct=True)
            mult = (0.40 + 0.60 * p_rank).clip(0.40, 1.0)
        out.loc[year_idx] = mult.reindex(year_idx).fillna(1.0)
        used += 1
    meta = {"target": target, "years_fit": used, "pit": "exit_before_signal_year"}
    if abandoned:
        meta.update({"status": "ABANDONED", "reason": reason})
        return _identity(predictions), meta
    if used == 0:
        return _identity(predictions), {"status": "ABANDONED", "reason": "insufficient_history", **meta}
    return out, meta


def _agree_bucket(n: pd.Series) -> pd.Series:
    x = pd.to_numeric(n, errors="coerce").fillna(0)
    out = pd.Series("le3", index=x.index, dtype=object)
    out = out.where(~x.ge(4), "eq4")
    out = out.where(~x.ge(5), "eq5")
    return out


def _margin_bucket(p: pd.Series) -> pd.Series:
    x = pd.to_numeric(p, errors="coerce")
    out = pd.Series("mid", index=x.index, dtype=object)
    out = out.where(~x.le(1.0 / 3.0), "low")
    out = out.where(~x.gt(2.0 / 3.0), "high")
    out = out.where(x.notna(), "mid")
    return out


def _twodim_prereg(predictions: pd.DataFrame) -> pd.Series:
    agree = agreement_frame(predictions)
    a = _agree_bucket(agree["agree_n"])
    m = _margin_bucket(agree["margin_p250"])
    table = {
        ("le3", "low"): 0.25,
        ("le3", "mid"): 0.50,
        ("le3", "high"): 0.75,
        ("eq4", "low"): 0.50,
        ("eq4", "mid"): 1.00,
        ("eq4", "high"): 1.00,
        ("eq5", "low"): 0.75,
        ("eq5", "mid"): 1.00,
        ("eq5", "high"): 1.00,
    }
    keys = list(zip(a.astype(str), m.astype(str)))
    vals = [table.get(k, 1.0) for k in keys]
    return pd.Series(vals, index=agree.index, dtype=float).clip(0.25, 1.0)


def _expanding_cell_stats(trades: pd.DataFrame, agree: pd.DataFrame) -> pd.DataFrame:
    joined = trades.merge(agree.reset_index().rename(columns={"index": "signal_date"}), on="signal_date", how="left")
    joined = joined.dropna(subset=["actual_exit_date", "ret"])
    joined["a"] = _agree_bucket(joined["agree_n"])
    joined["m"] = _margin_bucket(joined["margin_p250"])
    joined["cell"] = joined["a"].astype(str) + "|" + joined["m"].astype(str)
    joined = joined.sort_values("actual_exit_date")
    return joined


def _twodim_hist(predictions: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    agree = agreement_frame(predictions)
    trades = _load_anchor_trades()
    if trades.empty:
        return _identity(predictions), {"status": "ABANDONED", "reason": "no_anchor_trades"}
    hist = _expanding_cell_stats(trades, agree)
    dates = pd.DatetimeIndex(pd.to_datetime(predictions["signal_date"].unique())).sort_values()
    a = _agree_bucket(agree.reindex(dates)["agree_n"])
    m = _margin_bucket(agree.reindex(dates)["margin_p250"])
    cells = (a.astype(str) + "|" + m.astype(str)).reindex(dates)
    out = pd.Series(1.0, index=dates, dtype=float)
    used = 0
    exits = hist["actual_exit_date"].to_numpy()
    rets = pd.to_numeric(hist["ret"], errors="coerce").to_numpy()
    cell_ids = hist["cell"].to_numpy()
    for i, day in enumerate(dates):
        cell = cells.iloc[i]
        past = (exits < np.datetime64(day)) & (cell_ids == cell)
        n = int(past.sum())
        if n < MIN_GRID_N:
            continue
        mu = float(np.nanmean(rets[past]))
        if mu < 0:
            out.iloc[i] = 0.50
        elif mu < 0.01:
            out.iloc[i] = 0.75
        else:
            out.iloc[i] = 1.0
        used += 1
    return out.clip(0.25, 1.0), {"n_days_calibrated": used, "min_n": MIN_GRID_N, "pit": "exit_lt_signal"}


def _cvar_series(predictions: pd.DataFrame, *, mode: str) -> tuple[pd.Series, dict[str, Any]]:
    agree = agreement_frame(predictions)
    trades = _load_anchor_trades()
    if trades.empty:
        return _identity(predictions), {"status": "ABANDONED", "reason": "no_anchor_trades"}
    hist = _expanding_cell_stats(trades, agree)
    dates = pd.DatetimeIndex(pd.to_datetime(predictions["signal_date"].unique())).sort_values()
    a = _agree_bucket(agree.reindex(dates)["agree_n"])
    m = _margin_bucket(agree.reindex(dates)["margin_p250"])
    cells = (a.astype(str) + "|" + m.astype(str)).reindex(dates)
    cvar = pd.Series(np.nan, index=dates, dtype=float)
    exits = hist["actual_exit_date"].to_numpy()
    rets = pd.to_numeric(hist["ret"], errors="coerce").to_numpy()
    cell_ids = hist["cell"].to_numpy()
    for i, day in enumerate(dates):
        cell = cells.iloc[i]
        past = (exits < np.datetime64(day)) & (cell_ids == cell) & np.isfinite(rets)
        n = int(past.sum())
        if n < MIN_GRID_N:
            continue
        subset = rets[past]
        k = max(1, int(np.ceil(n * 0.10)))
        worst = np.sort(subset)[:k]
        cvar.iloc[i] = float(np.mean(worst))
    if mode == "abs":
        out = pd.Series(1.0, index=dates, dtype=float)
        out = out.where(~cvar.le(-0.03), 0.85)
        out = out.where(~cvar.le(-0.05), 0.67)
        out = out.where(~cvar.le(-0.08), 0.50)
        out = out.where(cvar.notna(), 1.0)
        return out.clip(0.25, 1.0), {"mode": "abs", "min_n": MIN_GRID_N, "no_skip": True}
    ranked = rolling_percentile((-cvar).dropna(), 250)
    p = ranked.reindex(dates)
    out = pd.Series(1.0, index=dates, dtype=float)
    out = out.where(~p.ge(0.60), 0.75)
    out = out.where(~p.ge(0.80), 0.50)
    out = out.where(p.notna(), 1.0)
    return out.clip(0.25, 1.0), {"mode": "rel", "min_n": MIN_GRID_N, "no_skip": True}


def _cross_section_mult(predictions: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    feat = spec.get("feature", "top1_minus_median")
    daily = cross_section_daily(predictions)
    col = f"{feat}_p250"
    if col not in daily.columns:
        daily[col] = rolling_percentile(pd.to_numeric(daily[feat], errors="coerce"), 250)
    p = daily[col]
    out = pd.Series(1.0, index=p.index, dtype=float)
    out = out.where(~p.le(0.20), 0.5)
    return out.where(p.notna(), 1.0), {"feature": feat, "pit": "rolling_250"}


def cross_section_daily(predictions: pd.DataFrame) -> pd.DataFrame:
    def _build():
        frame = predictions.copy()
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        rows = []
        for day, block in frame.groupby("signal_date", sort=True):
            s = pd.to_numeric(block["score"], errors="coerce").dropna()
            if s.empty:
                continue
            top = float(s.max())
            med = float(s.median())
            p75 = float(s.quantile(0.75))
            mu = float(s.mean())
            sd = float(s.std(ddof=0)) if len(s) > 1 else np.nan
            z = (top - mu) / sd if sd and sd > 1e-12 else np.nan
            w = np.exp(s.to_numpy() - np.nanmax(s.to_numpy()))
            w = w / w.sum() if w.sum() > 0 else w
            entropy = float(-(w * np.log(np.clip(w, 1e-12, 1.0))).sum())
            rows.append({
                "signal_date": pd.Timestamp(day),
                "top1_minus_median": top - med,
                "top1_minus_p75": top - p75,
                "z": z,
                "dispersion": sd,
                "entropy": entropy,
                "n": int(len(s)),
            })
        out = pd.DataFrame(rows).set_index("signal_date").sort_index()
        for col in ("top1_minus_median", "top1_minus_p75", "z", "dispersion", "entropy"):
            out[f"{col}_p250"] = rolling_percentile(out[col], 250)
        return out

    return _cached("xs:" + str(id(predictions)), _build)


def build_date_multiplier(
    spec: R4Spec,
    predictions: pd.DataFrame,
) -> tuple[pd.Series, dict[str, Any], dict[str, Any] | None]:
    overlay = spec.overlay or {"kind": "identity"}
    kind = overlay.get("kind")
    meta: dict[str, Any] = {"kind": kind}
    hooks = None

    if kind == "stack":
        market, m_meta = _leaf(overlay.get("market") or {"kind": "identity"}, predictions)
        conf, c_meta = _leaf(overlay.get("confidence") or {"kind": "identity"}, predictions)
        idx = market.index.union(conf.index)
        mm = market.reindex(idx).fillna(1.0)
        cc = conf.reindex(idx).fillna(1.0)
        product = pd.Series([combine_multipliers(a, b) for a, b in zip(mm, cc)], index=idx)
        cvar_spec = overlay.get("cvar")
        cvar_mode = overlay.get("cvar_mode") or "product"
        if cvar_spec:
            gv, g_meta = _leaf(cvar_spec, predictions)
            gg = gv.reindex(idx).fillna(1.0)
            if cvar_mode == "min":
                series = pd.concat([mm, cc, gg], axis=1).min(axis=1)
            elif cvar_mode == "geom_min":
                geom = (mm.clip(lower=0) * cc.clip(lower=0)).pow(0.5)
                series = pd.concat([geom, gg], axis=1).min(axis=1)
            else:
                series = pd.Series(
                    [combine_multipliers(a, b) for a, b in zip(product, gg)], index=idx,
                )
            meta["cvar"] = g_meta
            meta["cvar_mode"] = cvar_mode
        else:
            series = product
        meta["market"] = m_meta
        meta["confidence"] = c_meta
        meta["combine_default"] = "market_x_confidence"
    elif kind == "combine":
        series, extra = _leaf(overlay, predictions)
        meta.update(extra)
    else:
        series, extra = _leaf(overlay, predictions)
        meta.update(extra)

    series = apply_skip_years(series, spec.skip_years)
    series.index = pd.to_datetime(series.index)
    series = series.clip(0.0, 1.0)
    if spec.skip_years:
        meta["skip_years"] = list(spec.skip_years)
    meta.update(overlay_summary(series))
    return series, meta, hooks


__all__ = [
    "apply_date_multiplier",
    "agreement_frame",
    "build_date_multiplier",
    "cross_section_daily",
]
