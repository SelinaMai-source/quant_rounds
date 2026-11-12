"""实现 A–F 可计算公式。缺失不填 0。E 无公告时点 → 可算但 BLOCKED_PIT。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_ROOT, assert_project_result_path
from .common import dump_json, load_json, write_md
from .contract import (
    BLOCK_TRADING_PARQUET,
    BLOCKED_DATA,
    BLOCKED_PIT,
    DIAGNOSTIC_ONLY,
    HOLDER_NUM_PARQUET,
    PIT_ELIGIBLE_MODEL,
)
from .freeze import require_frozen
from .hypotheses import HYPOTHESES, require_hypotheses
from .min1 import AC_FACTORS, build_ac_factors
from .panel import (
    KEYS,
    asof_by_instrument,
    load_candidate_keys,
    load_hermes_table,
    load_trade_calendar,
    market_to_instrument,
    next_trading_day,
    parse_yyyymmdd,
)

PANEL_PATH = ROUND21_ROOT / "data" / "factor_panel.parquet"
PANEL_JSON = ROUND21_ROOT / "data" / "factor_panel.json"
CONSTRUCT_JSON = ROUND21_ROOT / "construct" / "construct.json"
CONSTRUCT_MD = ROUND21_ROOT / "02_construct.md"
FACTORS_DIR = ROUND21_ROOT / "factors"
B_CACHE = ROUND21_ROOT / "data" / "b_factors.parquet"
D_CACHE = ROUND21_ROOT / "data" / "d_factors.parquet"
E_CACHE = ROUND21_ROOT / "data" / "e_factors.parquet"
F_CACHE = ROUND21_ROOT / "data" / "f_factors.parquet"

UNITS: dict[str, dict[str, str]] = {
    "r21_a1_vw_limit_gap": {"unit": "CNY", "kind": "amount_weighted_limit_gap"},
    "r21_a2_minutes_to_reseal": {"unit": "minute", "kind": "duration"},
    "r21_a3_retouch_count": {"unit": "count", "kind": "event_count"},
    "r21_a4_tail_seal_share": {"unit": "share", "kind": "fraction_0_1"},
    "r21_b1_fb_amount_share": {"unit": "share", "kind": "pool_amount_share"},
    "r21_b2_seal_rel_strength": {"unit": "ratio", "kind": "vs_same_day_median"},
    "r21_b3_amt_surprise_vs_peer": {"unit": "ratio_gap", "kind": "self_minus_peer_median"},
    "r21_b4_activity_x_resilience": {"unit": "share_x_inv_minutes", "kind": "preregistered_interaction"},
    "r21_c1_min1_amihud": {"unit": "1/CNY", "kind": "abs_ret_per_amount"},
    "r21_c2_post_touch_amt_vs_impact": {"unit": "CNY/price", "kind": "amount_per_amplitude"},
    "r21_c3_post_touch_path_eff": {"unit": "ratio", "kind": "path_efficiency_0_1"},
    "r21_c4_downside_quad_var": {"unit": "share", "kind": "downside_quad_var_share"},
    "r21_d1_log_holder_ratio": {"unit": "log_ratio", "kind": "log_holder_change"},
    "r21_d2_holder_daily_chg": {"unit": "holders_per_day", "kind": "ann_interval_normalized"},
    "r21_d3_holder_per_circ": {"unit": "holders_per_share", "kind": "holder_over_circulating"},
    "r21_d4_holder_x_float": {"unit": "log1p_product", "kind": "preregistered_interaction"},
    "r21_e1_block_discount": {"unit": "fraction", "kind": "1_minus_price_over_close"},
    "r21_e2_block_vol20_to_circ": {"unit": "share", "kind": "block_volume_over_circ"},
    "r21_e3_discount_wtd_pressure": {"unit": "pressure", "kind": "discount_amount_over_circ_mcap"},
    "r21_e4_block_event_intensity_60": {"unit": "count", "kind": "sum_frequency_60d"},
    "r21_f1_fin_bal_chg": {"unit": "fraction", "kind": "balance_pct_change"},
    "r21_f2_fin_repay_pressure": {"unit": "fraction", "kind": "repay_over_prev_balance"},
    "r21_f3_fin_buy_intensity": {"unit": "fraction", "kind": "fin_buy_over_amount"},
    "r21_f4_fin_bal_accel": {"unit": "fraction_delta", "kind": "balance_chg_diff"},
}


def _finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _card_map() -> dict[str, dict[str, Any]]:
    return {c["factor_id"]: dict(c) for c in HYPOTHESES}


def _known_at_close(dates: pd.Series) -> pd.Series:
    ts = pd.to_datetime(dates, errors="coerce")
    return ts + pd.Timedelta(hours=15)


def build_b_factors(keys: pd.DataFrame, ac: pd.DataFrame) -> pd.DataFrame:
    ids = keys["instrument_id"].unique()
    daily = load_hermes_table("daily_raw", KEYS + ["amount"], instruments=ids)
    daily["amount"] = _finite(daily["amount"])
    ev = load_hermes_table(
        "daily_limit_events",
        KEYS + ["limit_type", "close_limit_value"],
        instruments=ids,
    )
    ev = ev.loc[ev["limit_type"].astype(str).eq("up"), KEYS + ["close_limit_value"]]
    ev["close_limit_value"] = _finite(ev["close_limit_value"])
    work = keys[KEYS].merge(daily, on=KEYS, how="left").merge(ev, on=KEYS, how="left")
    amt = _finite(work["amount"])
    day_sum = amt.groupby(work["trade_date"]).transform("sum")
    share = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = amt.notna() & day_sum.gt(0) & np.isfinite(amt) & np.isfinite(day_sum)
    share.loc[ok] = amt.loc[ok] / day_sum.loc[ok]
    work["r21_b1_fb_amount_share"] = share

    seal = _finite(work["close_limit_value"])
    med = seal.groupby(work["trade_date"]).transform("median")
    rel = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = seal.notna() & med.notna() & np.isfinite(seal) & np.isfinite(med) & med.ne(0)
    rel.loc[ok] = seal.loc[ok] / med.loc[ok]
    work["r21_b2_seal_rel_strength"] = rel

    hist = daily.sort_values(KEYS)
    hist["mean5"] = hist.groupby("instrument_id", sort=False)["amount"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    work = work.merge(hist[KEYS + ["mean5"]], on=KEYS, how="left")
    mean5 = _finite(work["mean5"])
    surprise = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = amt.notna() & mean5.gt(0) & np.isfinite(amt) & np.isfinite(mean5)
    surprise.loc[ok] = amt.loc[ok] / mean5.loc[ok]
    peer = surprise.groupby(work["trade_date"]).transform("median")
    b3 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = surprise.notna() & peer.notna() & np.isfinite(surprise) & np.isfinite(peer)
    b3.loc[ok] = surprise.loc[ok] - peer.loc[ok]
    work["r21_b3_amt_surprise_vs_peer"] = b3

    a2 = ac[KEYS + ["r21_a2_minutes_to_reseal"]]
    work = work.merge(a2, on=KEYS, how="left")
    a2v = _finite(work["r21_a2_minutes_to_reseal"])
    b1 = _finite(work["r21_b1_fb_amount_share"])
    b4 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = b1.notna() & a2v.notna() & np.isfinite(b1) & np.isfinite(a2v)
    b4.loc[ok] = b1.loc[ok] * (1.0 / (1.0 + a2v.loc[ok]))
    work["r21_b4_activity_x_resilience"] = b4
    return work[
        KEYS
        + [
            "r21_b1_fb_amount_share",
            "r21_b2_seal_rel_strength",
            "r21_b3_amt_surprise_vs_peer",
            "r21_b4_activity_x_resilience",
        ]
    ]


def _holder_snapshots(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    raw = pd.read_parquet(HOLDER_NUM_PARQUET)
    frame = pd.DataFrame(
        {
            "instrument_id": raw["MARKET_CODE"].map(market_to_instrument),
            "ann_dt": parse_yyyymmdd(raw["ANN_DT"]),
            "holder_num": _finite(raw["HOLDER_NUM"]),
        }
    )
    frame = frame.dropna(subset=["instrument_id", "ann_dt"])
    frame = frame.loc[frame["holder_num"].gt(0)].copy()
    frame = frame.sort_values(["instrument_id", "ann_dt"])
    frame = frame.drop_duplicates(["instrument_id", "ann_dt"], keep="last")
    g = frame.groupby("instrument_id", sort=False)
    prev_h = g["holder_num"].shift(1)
    prev_ann = g["ann_dt"].shift(1)
    ratio = frame["holder_num"] / prev_h
    ok = prev_h.gt(0) & frame["holder_num"].gt(0) & np.isfinite(ratio) & ratio.gt(0)
    frame["r21_d1_log_holder_ratio"] = np.where(ok, np.log(ratio), np.nan)
    days = (frame["ann_dt"] - prev_ann).dt.days
    days = days.where(days.notna(), np.nan)
    denom = np.maximum(pd.to_numeric(days, errors="coerce"), 1.0)
    chg = pd.Series(np.nan, index=frame.index, dtype="float64")
    ok = prev_h.notna() & frame["holder_num"].notna() & denom.notna()
    chg.loc[ok] = (frame.loc[ok, "holder_num"] - prev_h.loc[ok]) / denom.loc[ok]
    frame["r21_d2_holder_daily_chg"] = chg
    frame["known_at"] = next_trading_day(frame["ann_dt"], calendar)
    frame = frame.dropna(subset=["known_at"])
    return frame


def build_d_factors(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    calendar = load_trade_calendar()
    snaps = _holder_snapshots(calendar)
    ids = keys["instrument_id"].unique()
    snaps = snaps.loc[snaps["instrument_id"].isin(ids)].copy()
    val = load_hermes_table(
        "daily_valuation",
        KEYS + ["circulating_shares", "free_float_shares", "available_at"],
        instruments=ids,
    )
    val["circulating_shares"] = _finite(val["circulating_shares"])
    val["free_float_shares"] = _finite(val["free_float_shares"])
    work = asof_by_instrument(
        keys[KEYS],
        snaps,
        left_on="trade_date",
        right_on="known_at",
        columns=["holder_num", "r21_d1_log_holder_ratio", "r21_d2_holder_daily_chg", "known_at", "ann_dt"],
    )
    work = asof_by_instrument(
        work,
        val,
        left_on="trade_date",
        right_on="available_at",
        columns=["circulating_shares", "free_float_shares", "available_at"],
    )
    circ = _finite(work["circulating_shares"])
    hold = _finite(work["holder_num"])
    d3 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = hold.gt(0) & circ.gt(0) & np.isfinite(hold) & np.isfinite(circ)
    d3.loc[ok] = hold.loc[ok] / circ.loc[ok]
    work["r21_d3_holder_per_circ"] = d3
    ff = _finite(work["free_float_shares"])
    d4 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = hold.gt(0) & ff.gt(0) & np.isfinite(hold) & np.isfinite(ff)
    d4.loc[ok] = np.log1p(hold.loc[ok]) * np.log1p(ff.loc[ok])
    work["r21_d4_holder_x_float"] = d4
    holder_known = pd.to_datetime(work["known_at"], errors="coerce")
    val_at = pd.to_datetime(work["available_at"], errors="coerce")
    work["d_known_at"] = holder_known
    work["d3_known_at"] = pd.concat([holder_known, val_at], axis=1).max(axis=1)
    work["d4_known_at"] = work["d3_known_at"]
    meta = {
        "n_holder_snaps": int(len(snaps)),
        "pit_rule": "ANN_DT next trading day; valuation.available_at",
        "holder_enddate_used": False,
        "filled_with_zero": False,
    }
    return work, meta


def _event_window_sum(
    keys: pd.DataFrame,
    events: pd.DataFrame,
    value_col: str,
    window: int,
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    cal = pd.DatetimeIndex(calendar).sort_values()
    cmap = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    ev = events.dropna(subset=["instrument_id", "trade_date", value_col]).copy()
    ev["di"] = ev["trade_date"].map(cmap)
    ev = ev.dropna(subset=["di"])
    ev["di"] = ev["di"].astype(int)
    ev = ev.sort_values(["instrument_id", "di"])
    ev["csum"] = ev.groupby("instrument_id", sort=False)[value_col].cumsum()
    left = keys[KEYS].copy()
    left["di"] = left["trade_date"].map(cmap)
    left = left.dropna(subset=["di"]).copy()
    left["di"] = left["di"].astype(int)
    left["di0"] = left["di"] - int(window)
    left["_rid"] = np.arange(len(left))
    end = asof_by_instrument(left, ev, left_on="di", right_on="di", columns=["csum"]).sort_values("_rid")
    start = asof_by_instrument(left, ev, left_on="di0", right_on="di", columns=["csum"]).sort_values("_rid")
    end_s = _finite(end["csum"]).to_numpy()
    start_s = _finite(start["csum"]).to_numpy()
    val = end_s - np.where(np.isfinite(start_s), start_s, 0.0)
    keep = np.isfinite(end_s) & np.isfinite(val) & (val != 0)
    left = left.sort_values("_rid")
    left["_win"] = np.where(keep, val, np.nan)
    merged = keys[KEYS].merge(left[KEYS + ["_win"]], on=KEYS, how="left")
    return _finite(merged["_win"]).set_axis(keys.index)


def build_e_factors(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(BLOCK_TRADING_PARQUET)
    ev = pd.DataFrame(
        {
            "instrument_id": raw["MARKET_CODE"].map(market_to_instrument),
            "trade_date": parse_yyyymmdd(raw["TRADE_DATE"]),
            "price": _finite(raw["B_SHARE_PRICE"]),
            "volume": _finite(raw["B_SHARE_VOLUME"]),
            "amount": _finite(raw["B_SHARE_AMOUNT"]),
            "freq": _finite(raw["B_FREQUENCY"]),
        }
    )
    ev = ev.dropna(subset=["instrument_id", "trade_date"])
    w = ev["amount"].where(ev["amount"].gt(0))
    ev["wpx_num"] = ev["price"] * w
    day = ev.groupby(["instrument_id", "trade_date"], as_index=False).agg(
        wpx_num=("wpx_num", "sum"),
        wpx_den=("amount", "sum"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
        freq=("freq", "sum"),
    )
    wpx = pd.Series(np.nan, index=day.index, dtype="float64")
    ok = day["wpx_den"].gt(0) & day["wpx_num"].notna()
    wpx.loc[ok] = day.loc[ok, "wpx_num"] / day.loc[ok, "wpx_den"]
    day["wpx"] = wpx
    ids = keys["instrument_id"].unique()
    daily = load_hermes_table("daily_raw", KEYS + ["close"], instruments=ids)
    daily["close"] = _finite(daily["close"])
    work = keys[KEYS].merge(day[KEYS + ["wpx", "volume", "amount", "freq"]], on=KEYS, how="left")
    work = work.merge(daily, on=KEYS, how="left")
    close = _finite(work["close"])
    wpx_s = _finite(work["wpx"])
    e1 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = wpx_s.notna() & close.gt(0) & np.isfinite(wpx_s) & np.isfinite(close)
    e1.loc[ok] = 1.0 - wpx_s.loc[ok] / close.loc[ok]
    work["r21_e1_block_discount"] = e1

    ev2 = ev.loc[ev["instrument_id"].isin(ids)].copy()
    ev2["discount"] = np.nan
    ev2 = ev2.merge(daily, on=KEYS, how="left")
    ok = ev2["price"].notna() & ev2["close"].gt(0) & ev2["amount"].gt(0)
    ev2.loc[ok, "discount"] = (1.0 - ev2.loc[ok, "price"] / ev2.loc[ok, "close"]) * ev2.loc[ok, "amount"]
    calendar = load_trade_calendar()
    work["e2_vol20"] = _event_window_sum(work, ev2.rename(columns={"volume": "value"}), "value", 20, calendar)
    work["e3_press20"] = _event_window_sum(work, ev2.rename(columns={"discount": "value"}), "value", 20, calendar)
    work["r21_e4_block_event_intensity_60"] = _event_window_sum(
        work, ev2.rename(columns={"freq": "value"}), "value", 60, calendar
    )
    val = load_hermes_table(
        "daily_valuation",
        KEYS + ["circulating_shares", "circulating_market_cap", "available_at"],
        instruments=ids,
    )
    val["circulating_shares"] = _finite(val["circulating_shares"])
    val["circulating_market_cap"] = _finite(val["circulating_market_cap"])
    work = asof_by_instrument(
        work,
        val,
        left_on="trade_date",
        right_on="available_at",
        columns=["circulating_shares", "circulating_market_cap"],
    )
    circ = _finite(work["circulating_shares"])
    mcap = _finite(work["circulating_market_cap"])
    e2 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = work["e2_vol20"].notna() & circ.gt(0)
    e2.loc[ok] = _finite(work.loc[ok, "e2_vol20"]) / circ.loc[ok]
    work["r21_e2_block_vol20_to_circ"] = e2
    e3 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = work["e3_press20"].notna() & mcap.gt(0)
    e3.loc[ok] = _finite(work.loc[ok, "e3_press20"]) / mcap.loc[ok]
    work["r21_e3_discount_wtd_pressure"] = e3
    meta = {
        "blocked": BLOCKED_PIT,
        "reason": "block_trading has TRADE_DATE only; no ANN/available_at/publish timestamp",
        "model_eligible": DIAGNOSTIC_ONLY,
        "filled_with_zero": False,
        "no_event_stays_nan": True,
    }
    return work[KEYS + [c["factor_id"] for c in HYPOTHESES if c["family"] == "E"]], meta


def build_f_factors(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ids = keys["instrument_id"].unique()
    margin = load_hermes_table(
        "daily_margin",
        KEYS + ["financing_balance", "financing_buy_value", "financing_repayment_value", "available_at"],
        instruments=ids,
    )
    daily = load_hermes_table("daily_raw", KEYS + ["amount"], instruments=ids)
    daily["amount"] = _finite(daily["amount"])
    margin = margin.merge(daily, on=KEYS, how="left")
    margin = margin.dropna(subset=["instrument_id", "available_at"])
    margin = margin.sort_values(["instrument_id", "available_at", "trade_date"])
    g = margin.groupby("instrument_id", sort=False)
    bal = _finite(margin["financing_balance"])
    prev = g["financing_balance"].shift(1)
    prev = _finite(prev)
    f1 = pd.Series(np.nan, index=margin.index, dtype="float64")
    ok = bal.notna() & prev.gt(0) & np.isfinite(bal) & np.isfinite(prev)
    f1.loc[ok] = bal.loc[ok] / prev.loc[ok] - 1.0
    margin["r21_f1_fin_bal_chg"] = f1
    repay = _finite(margin["financing_repayment_value"])
    f2 = pd.Series(np.nan, index=margin.index, dtype="float64")
    ok = repay.notna() & prev.gt(0) & np.isfinite(repay)
    f2.loc[ok] = repay.loc[ok] / prev.loc[ok]
    margin["r21_f2_fin_repay_pressure"] = f2
    buy = _finite(margin["financing_buy_value"])
    amt = _finite(margin["amount"])
    f3 = pd.Series(np.nan, index=margin.index, dtype="float64")
    ok = buy.notna() & amt.gt(0) & np.isfinite(buy)
    f3.loc[ok] = buy.loc[ok] / amt.loc[ok]
    margin["r21_f3_fin_buy_intensity"] = f3
    prev_f1 = g["r21_f1_fin_bal_chg"].shift(1)
    f4 = pd.Series(np.nan, index=margin.index, dtype="float64")
    ok = f1.notna() & prev_f1.notna() & np.isfinite(f1) & np.isfinite(prev_f1)
    f4.loc[ok] = f1.loc[ok] - prev_f1.loc[ok]
    margin["r21_f4_fin_bal_accel"] = f4
    work = asof_by_instrument(
        keys[KEYS],
        margin,
        left_on="trade_date",
        right_on="available_at",
        columns=[
            "r21_f1_fin_bal_chg",
            "r21_f2_fin_repay_pressure",
            "r21_f3_fin_buy_intensity",
            "r21_f4_fin_bal_accel",
            "available_at",
        ],
    )
    work["f_known_at"] = pd.to_datetime(work["available_at"], errors="coerce")
    meta = {
        "pit_rule": "known_at=available_at; D close default not used as published",
        "non_margin_filled_zero": False,
        "filled_with_zero": False,
        "n_margin_rows": int(len(margin)),
    }
    return work, meta


def _write_factor_values(
    panel: pd.DataFrame,
    known_map: dict[str, pd.Series],
    cards: dict[str, dict[str, Any]],
    family_status: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    written = []
    for fid, card in cards.items():
        dest = FACTORS_DIR / fid
        dest.mkdir(parents=True, exist_ok=True)
        assert_project_result_path(dest)
        values = panel[KEYS].copy()
        values["value"] = _finite(panel[fid]) if fid in panel.columns else np.nan
        known = known_map.get(fid)
        if known is None:
            values["known_at"] = _known_at_close(values["trade_date"])
        else:
            values["known_at"] = pd.to_datetime(known, errors="coerce")
            miss = values["known_at"].isna() & values["value"].notna()
            values.loc[miss, "known_at"] = _known_at_close(values.loc[miss, "trade_date"])
        path = dest / "values.parquet"
        assert_project_result_path(path)
        values.to_parquet(path, index=False)
        n = int(len(values))
        v = int(values["value"].notna().sum())
        family = card["family"]
        blocked = card.get("blocked")
        if family == "E":
            status = BLOCKED_PIT
            eligible = DIAGNOSTIC_ONLY
            reason = "no announcement timestamp on block_trading"
        elif fid not in panel.columns:
            status = BLOCKED_DATA
            eligible = DIAGNOSTIC_ONLY
            reason = "column not constructed"
        elif v == 0:
            status = BLOCKED_DATA
            eligible = DIAGNOSTIC_ONLY
            reason = "all values missing; not filled with 0"
        else:
            status = "COMPUTED"
            eligible = card.get("model_eligible_default") or PIT_ELIGIBLE_MODEL
            reason = None
        rec = {
            "factor_id": fid,
            "family": family,
            "formula": card.get("formula"),
            "n": n,
            "n_valid": v,
            "coverage": (v / n) if n else None,
            "filled_with_zero": False,
            "construct_status": status,
            "model_eligible": eligible,
            "blocked": blocked or (status if status in {BLOCKED_DATA, BLOCKED_PIT} else None),
            "reason": reason,
            "units": UNITS.get(fid),
            "values_path": str(path),
            "a1_is_not_r18_f1": fid != "r21_a1_vw_limit_gap" or True,
        }
        if fid == "r21_a1_vw_limit_gap":
            rec["a1_is_not_r18_f1"] = True
            rec["r18_f1"] = "log1p(V_seal / (A_below + 0.01*A_day))"
        dump_json(dest / "construct.json", rec)
        written.append(rec)
    return written


def run_construct(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    require_frozen()
    require_hypotheses()
    if resume and PANEL_PATH.is_file() and CONSTRUCT_JSON.is_file():
        existing = load_json(CONSTRUCT_JSON)
        if existing.get("status") == "DONE":
            return existing
    keys = load_candidate_keys()
    print(f"[round21/construct] keys={len(keys)} dates={keys['trade_date'].nunique()}", flush=True)
    print("[round21/construct] A/C minute formulas", flush=True)
    ac, ac_meta = build_ac_factors(keys, jobs=jobs, resume=resume)
    print("[round21/construct] B daily competition", flush=True)
    if resume and B_CACHE.is_file():
        b = pd.read_parquet(B_CACHE)
        b["trade_date"] = pd.to_datetime(b["trade_date"])
        b["instrument_id"] = b["instrument_id"].astype(str)
    else:
        b = build_b_factors(keys, ac)
        B_CACHE.parent.mkdir(parents=True, exist_ok=True)
        b.to_parquet(B_CACHE, index=False)
    print("[round21/construct] D holder ANN_DT next session", flush=True)
    if resume and D_CACHE.is_file():
        d = pd.read_parquet(D_CACHE)
        d["trade_date"] = pd.to_datetime(d["trade_date"])
        d["instrument_id"] = d["instrument_id"].astype(str)
        d_meta = {"resumed": True}
    else:
        d, d_meta = build_d_factors(keys)
        d.to_parquet(D_CACHE, index=False)
    print("[round21/construct] E block diagnostic (BLOCKED_PIT)", flush=True)
    if resume and E_CACHE.is_file():
        e = pd.read_parquet(E_CACHE)
        e["trade_date"] = pd.to_datetime(e["trade_date"])
        e["instrument_id"] = e["instrument_id"].astype(str)
        e_meta = {
            "blocked": BLOCKED_PIT,
            "reason": "block_trading has TRADE_DATE only; no ANN/available_at/publish timestamp",
            "resumed": True,
        }
    else:
        e, e_meta = build_e_factors(keys)
        e.to_parquet(E_CACHE, index=False)
    print("[round21/construct] F margin available_at", flush=True)
    if resume and F_CACHE.is_file():
        f = pd.read_parquet(F_CACHE)
        f["trade_date"] = pd.to_datetime(f["trade_date"])
        f["instrument_id"] = f["instrument_id"].astype(str)
        f_meta = {"resumed": True}
    else:
        f, f_meta = build_f_factors(keys)
        f.to_parquet(F_CACHE, index=False)
    panel = keys.copy()
    for frame, extra in (
        (ac, AC_FACTORS),
        (
            b,
            [
                "r21_b1_fb_amount_share",
                "r21_b2_seal_rel_strength",
                "r21_b3_amt_surprise_vs_peer",
                "r21_b4_activity_x_resilience",
            ],
        ),
        (
            d,
            [
                "r21_d1_log_holder_ratio",
                "r21_d2_holder_daily_chg",
                "r21_d3_holder_per_circ",
                "r21_d4_holder_x_float",
            ],
        ),
        (e, [c["factor_id"] for c in HYPOTHESES if c["family"] == "E"]),
        (
            f,
            [
                "r21_f1_fin_bal_chg",
                "r21_f2_fin_repay_pressure",
                "r21_f3_fin_buy_intensity",
                "r21_f4_fin_bal_accel",
            ],
        ),
    ):
        keep = KEYS + [c for c in extra if c in frame.columns]
        panel = panel.merge(frame[keep], on=KEYS, how="left")
    for col in [c["factor_id"] for c in HYPOTHESES]:
        if col in panel.columns:
            panel[col] = _finite(panel[col]).replace([np.inf, -np.inf], np.nan)
    if panel.duplicated(KEYS).any():
        raise RuntimeError("construct panel duplicate keys")
    def _align_known(frame: pd.DataFrame, col: str) -> pd.Series:
        s = pd.Series(pd.NaT, index=panel.index)
        if col not in frame.columns:
            return s
        tmp = panel[KEYS].merge(frame[KEYS + [col]], on=KEYS, how="left")
        return pd.to_datetime(tmp[col], errors="coerce")

    known_map: dict[str, pd.Series] = {}
    close_known = _known_at_close(panel["trade_date"])
    for fid in [c["factor_id"] for c in HYPOTHESES if c["family"] in {"A", "B", "C"}]:
        known_map[fid] = close_known
    known_map["r21_d1_log_holder_ratio"] = _align_known(d, "d_known_at")
    known_map["r21_d2_holder_daily_chg"] = _align_known(d, "d_known_at")
    known_map["r21_d3_holder_per_circ"] = _align_known(d, "d3_known_at")
    known_map["r21_d4_holder_x_float"] = _align_known(d, "d4_known_at")
    for fid in [c["factor_id"] for c in HYPOTHESES if c["family"] == "E"]:
        known_map[fid] = close_known
    for fid in [c["factor_id"] for c in HYPOTHESES if c["family"] == "F"]:
        known_map[fid] = _align_known(f, "f_known_at")

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(PANEL_PATH)
    panel.to_parquet(PANEL_PATH, index=False)
    cards = _card_map()
    written = _write_factor_values(panel, known_map, cards, {"E": e_meta})
    payload = {
        "status": "DONE",
        "round": 21,
        "n_rows": int(len(panel)),
        "n_dates": int(panel["trade_date"].nunique()),
        "date_min": str(panel["trade_date"].min().date()),
        "date_max": str(panel["trade_date"].max().date()),
        "filled_with_zero": False,
        "a1_is_not_r18_f1": True,
        "panel": str(PANEL_PATH),
        "ac_meta": ac_meta,
        "d_meta": d_meta,
        "e_meta": e_meta,
        "f_meta": f_meta,
        "factors": written,
        "proposed": [c["factor_id"] for c in HYPOTHESES],
        "computed": [r["factor_id"] for r in written if r["n_valid"] > 0],
        "blocked_pit": [r["factor_id"] for r in written if r["blocked"] == BLOCKED_PIT],
        "blocked_data": [r["factor_id"] for r in written if r["blocked"] == BLOCKED_DATA],
    }
    dump_json(PANEL_JSON, {k: payload[k] for k in ("status", "n_rows", "n_dates", "date_min", "date_max", "panel")})
    dump_json(CONSTRUCT_JSON, payload)
    lines = [
        "# Round21 · 因子构造",
        "",
        f"行=`{payload['n_rows']}` 日=`{payload['n_dates']}` `{payload['date_min']}`→`{payload['date_max']}`。缺失不填 0。A1 ≠ R18 F1。",
        "",
        "## 公式",
        "",
    ]
    for rec in written:
        lines.append(
            f"- `{rec['factor_id']}` valid=`{rec['n_valid']}/{rec['n']}` "
            f"status=`{rec['construct_status']}` blocked=`{rec['blocked']}` eligible=`{rec['model_eligible']}`"
        )
    lines += [
        "",
        f"E 家族 `{payload['blocked_pit']}`：无公告时点，诊断可算，禁 `PIT_ELIGIBLE_MODEL`。",
        f"面板 `{PANEL_PATH}`。",
        "",
    ]
    write_md(CONSTRUCT_MD, "\n".join(lines))
    return payload
