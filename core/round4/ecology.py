"""一进二生态：D-1 首板队列在 D 日已实现的统计。滚动分位，禁止全样本回填。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..round2.data import load_candidate_features, load_regime_frame
from ..round2.overlays import regime_score_five, rolling_percentile

LIMIT_TOL = 0.005
BIG_LOSS = -0.05
_CACHE: dict[str, Any] = {}


def _cached(key: str, factory):
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def _next_day_map(dates: pd.Series) -> pd.Series:
    uniq = pd.DatetimeIndex(pd.to_datetime(dates.dropna().unique())).sort_values()
    nxt = pd.Series(uniq[1:], index=uniq[:-1])
    return pd.to_datetime(dates).map(nxt)


def build_ecology_frame(prepared=None) -> pd.DataFrame:
    """Index = D（信号日）。全部用 ≤D 已实现信息。"""
    return _cached("eco", lambda: _build(prepared))


def _bars_from_prepared(prepared) -> pd.DataFrame | None:
    if prepared is None:
        return None
    bars = getattr(prepared, "bars", None)
    if bars is None or bars.empty:
        return None
    cols = [c for c in ("trade_date", "instrument_id", "open", "close", "high", "pre_close", "limit_up") if c in bars.columns]
    out = bars[cols].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["instrument_id"] = out["instrument_id"].astype(str)
    return out


def _build(prepared) -> pd.DataFrame:
    panel = load_candidate_features([
        "trade_date", "instrument_id", "is_first_board_hard",
        "ind_first_board_count_log", "ind_first_board_share",
        "next1_date", "next1_open", "label_d1_touch",
    ])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    hard = panel["is_first_board_hard"]
    if hard.dtype == object:
        fb = panel.loc[hard.astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    else:
        fb = panel.loc[hard.fillna(False).astype(bool)].copy()

    supply = fb.groupby("trade_date", sort=True).agg(
        fb_count=("instrument_id", "nunique"),
        n_ind=("ind_first_board_count_log", "nunique"),
    )
    fb = fb.copy()
    fb["ind_key"] = pd.to_numeric(fb["ind_first_board_count_log"], errors="coerce").round(6)
    cnt = fb.groupby(["trade_date", "ind_key"], sort=True).size()
    tot = fb.groupby("trade_date", sort=True).size().replace(0, np.nan)
    share = cnt / tot
    hhi = (share ** 2).groupby(level=0).sum().rename("hhi")
    supply = supply.join(hhi, how="left")
    supply["ind_diffusion"] = supply["n_ind"] / supply["fb_count"].replace(0, np.nan)

    if "next1_date" in fb.columns:
        fb["d1"] = pd.to_datetime(fb["next1_date"])
    else:
        fb["d1"] = _next_day_map(fb["trade_date"])
    missing = fb["d1"].isna()
    if missing.any():
        fb.loc[missing, "d1"] = _next_day_map(fb.loc[missing, "trade_date"])

    bars = _bars_from_prepared(prepared)
    if bars is None:
        from ..stock_readonly import load_prepared_market

        loaded = load_prepared_market(warmup_ma5=True)
        bars = _bars_from_prepared(loaded["prepared"])
    realized = fb.merge(
        bars,
        left_on=["d1", "instrument_id"],
        right_on=["trade_date", "instrument_id"],
        how="left",
        suffixes=("", "_bar"),
    )
    close = pd.to_numeric(realized["close"], errors="coerce")
    open_ = pd.to_numeric(realized["open"], errors="coerce")
    high = pd.to_numeric(realized["high"], errors="coerce")
    pre = pd.to_numeric(realized["pre_close"], errors="coerce")
    up = pd.to_numeric(realized["limit_up"], errors="coerce")
    realized["cont"] = close >= (up - LIMIT_TOL)
    realized["touch"] = high >= (up - LIMIT_TOL)
    if "label_d1_touch" in realized.columns:
        lab = pd.to_numeric(realized["label_d1_touch"], errors="coerce")
        realized["touch"] = realized["touch"].fillna(False) | lab.fillna(0).astype(bool)
    realized["red"] = close > pre
    realized["oc"] = close / open_ - 1.0
    realized["gap"] = open_ / pre - 1.0
    realized["day_ret"] = close / pre - 1.0
    realized["big_loss"] = realized["day_ret"] <= BIG_LOSS

    d1 = realized.groupby("d1", sort=True).agg(
        continue_rate=("cont", "mean"),
        touch_rate=("touch", "mean"),
        red_rate=("red", "mean"),
        oc_median=("oc", "median"),
        gap_median=("gap", "median"),
        big_loss_rate=("big_loss", "mean"),
        n_realized=("instrument_id", "size"),
    )
    d1.index = pd.to_datetime(d1.index)
    d1.index.name = "trade_date"

    regime = load_regime_frame()
    frame = d1.join(supply, how="outer")
    frame = frame.join(regime[["up_ratio", "up_num", "fail_ratio", "suc_ratio", "con_ratio"]], how="left")
    if "prev_limit_relimit_rate" in regime.columns:
        frame["mkt_relimit"] = regime["prev_limit_relimit_rate"]
    if "prev_limit_red_rate" in regime.columns:
        frame["mkt_red"] = regime["prev_limit_red_rate"]
    frame["continue_rate"] = frame["continue_rate"].fillna(frame.get("mkt_relimit"))
    frame["red_rate"] = frame["red_rate"].fillna(frame.get("mkt_red"))
    frame = frame.sort_index()

    for col in ("continue_rate", "touch_rate", "red_rate", "oc_median", "gap_median",
                "big_loss_rate", "fb_count", "hhi", "ind_diffusion"):
        if col not in frame:
            frame[col] = np.nan
        s = pd.to_numeric(frame[col], errors="coerce")
        frame[f"{col}_ma5"] = s.rolling(5, min_periods=3).mean()
        frame[f"{col}_ma10"] = s.rolling(10, min_periods=5).mean()
        frame[f"{col}_ma20"] = s.rolling(20, min_periods=8).mean()
        frame[f"{col}_ewma20"] = s.ewm(span=20, min_periods=8, adjust=False).mean()
        frame[f"{col}_p250"] = rolling_percentile(s, 250)

    return frame


def eco_relimit_multiplier(eco: pd.DataFrame, *, low: float = 0.20, low_mult: float = 0.5) -> pd.Series:
    p = pd.to_numeric(eco["continue_rate_p250"], errors="coerce")
    out = pd.Series(1.0, index=eco.index, dtype=float)
    out = out.where(~p.le(low), float(low_mult))
    return out.where(p.notna(), 1.0)


def eco_oc_ewma_multiplier(eco: pd.DataFrame, *, skip: float = 0.20, half: float = 0.40) -> pd.Series:
    p = rolling_percentile(pd.to_numeric(eco["oc_median_ewma20"], errors="coerce"), 250)
    out = pd.Series(1.0, index=eco.index, dtype=float)
    out = out.where(~p.le(half), 0.5)
    out = out.where(~p.le(skip), 0.0)
    return out.where(p.notna(), 1.0)


def eco_bigloss_multiplier(eco: pd.DataFrame, *, skip: float = 0.80, half: float = 0.70) -> pd.Series:
    p = pd.to_numeric(eco["big_loss_rate_p250"], errors="coerce")
    out = pd.Series(1.0, index=eco.index, dtype=float)
    out = out.where(~p.ge(half), 0.5)
    out = out.where(~p.ge(skip), 0.0)
    return out.where(p.notna(), 1.0)


def eco_five_multiplier(eco: pd.DataFrame, *, hard: bool) -> tuple[pd.Series, pd.Series]:
    cont = pd.to_numeric(eco["continue_rate_p250"], errors="coerce")
    oc = rolling_percentile(pd.to_numeric(eco["oc_median_ewma20"], errors="coerce"), 250)
    inv_loss = 1.0 - pd.to_numeric(eco["big_loss_rate_p250"], errors="coerce")
    supply_p = pd.to_numeric(eco["fb_count_p250"], errors="coerce")
    supply_mid = 1.0 - (supply_p - 0.5).abs() * 2.0
    inv_hhi = 1.0 - pd.to_numeric(eco["hhi_p250"], errors="coerce")
    parts = {
        "continue": cont.clip(0, 1),
        "oc": oc.clip(0, 1),
        "inv_loss": inv_loss.clip(0, 1),
        "supply_mid": supply_mid.clip(0, 1),
        "inv_hhi": inv_hhi.clip(0, 1),
    }
    return regime_score_five(parts, hard=hard)
