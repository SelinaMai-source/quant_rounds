"""实现 G–N 32 个公式。缺失不填 0。无数据记 BLOCKED_DATA。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..paths import ROUND21_2_ROOT
from ..round18.f1 import _bar_seconds, _min_root
from ..round21.contract import HERMES_REBUILD_RAW, HERMES_V7
from ..round21.panel import KEYS, asof_by_instrument, load_candidate_keys, load_hermes_table, load_trade_calendar
from .common import dump_json, write_md
from .contract import BLOCKED_DATA, EPS, MD, NEW_FACTOR_IDS
from .hypotheses import require_hypotheses

PANEL_PATH = ROUND21_2_ROOT / "data" / "factor_panel.parquet"
PANEL_JSON = ROUND21_2_ROOT / "data" / "factor_panel.json"
CONSTRUCT_JSON = ROUND21_2_ROOT / "construct" / "construct.json"
MASK_PATH = ROUND21_2_ROOT / "data" / "coverage_masks.parquet"


def _finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _known_close(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates, errors="coerce") + pd.Timedelta(hours=15)


def _map_raw_id(ticker: pd.Series, exch: pd.Series) -> pd.Series:
    t = ticker.astype(str).str.zfill(6)
    e = exch.astype(str)
    return t + "." + e


def _load_raw_table(name: str, columns: list[str], instruments: set[str] | None) -> pd.DataFrame:
    root = HERMES_REBUILD_RAW / name
    if not root.exists():
        raise RuntimeError(f"{BLOCKED_DATA} missing raw {name}: {root}")
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"{BLOCKED_DATA} empty raw {name}")
    parts = []
    want = set(instruments or [])
    for path in files:
        names = set(pq.read_schema(path).names)
        cols = [c for c in columns if c in names]
        if "TICKER_SYMBOL" not in cols:
            cols = ["TICKER_SYMBOL", "EXCHANGE_CD", *cols]
        try:
            part = pd.read_parquet(path, columns=[c for c in cols if c in names])
        except Exception:
            continue
        if part.empty:
            continue
        part["instrument_id"] = _map_raw_id(part["TICKER_SYMBOL"], part["EXCHANGE_CD"])
        if want:
            part = part.loc[part["instrument_id"].isin(want)]
        if not part.empty:
            parts.append(part)
    if not parts:
        raise RuntimeError(f"{BLOCKED_DATA} no overlapping instruments in raw {name}")
    frame = pd.concat(parts, ignore_index=True)
    if "TRADE_DATE" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["TRADE_DATE"])
    if "UPDATE_TIME" in frame.columns:
        frame["update_time"] = pd.to_datetime(frame["UPDATE_TIME"])
    return frame


def _cache(name: str) -> Path:
    return ROUND21_2_ROOT / "data" / f"{name}.parquet"


def _trading_days_between(prev: pd.Series, cur: pd.Series, cal: pd.DatetimeIndex) -> pd.Series:
    cal = pd.DatetimeIndex(cal).sort_values()
    a = pd.to_datetime(prev, errors="coerce")
    b = pd.to_datetime(cur, errors="coerce")
    ia = cal.searchsorted(a, side="right")
    ib = cal.searchsorted(b, side="left")
    out = pd.Series(np.nan, index=prev.index, dtype="float64")
    ok = a.notna() & b.notna()
    out.loc[ok] = (ib[ok] - ia[ok]).astype("float64")
    return out


def build_g(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache("g_factors")
    if path.is_file():
        return pd.read_parquet(path), {"resumed": True, "path": str(path)}
    ids = set(keys["instrument_id"].astype(str))
    equd = _load_raw_table("mkt_equd", ["TICKER_SYMBOL", "EXCHANGE_CD", "TRADE_DATE", "TURNOVER_VALUE", "DEAL_AMOUNT", "UPDATE_TIME"], ids)
    mf = _load_raw_table(
        "mkt_equ_mf_new",
        [
            "TICKER_SYMBOL",
            "EXCHANGE_CD",
            "TRADE_DATE",
            "BUY_ORD_S",
            "BUY_ORD_M",
            "BUY_ORD_L",
            "BUY_ORD_XL",
            "SELL_ORD_S",
            "SELL_ORD_M",
            "SELL_ORD_L",
            "SELL_ORD_XL",
            "BUY_VOL_S",
            "BUY_VOL_M",
            "BUY_VOL_L",
            "BUY_VOL_XL",
            "SELL_VOL_S",
            "SELL_VOL_M",
            "SELL_VOL_L",
            "SELL_VOL_XL",
            "UPDATE_TIME",
        ],
        ids,
    )
    daily = load_hermes_table("daily_raw", KEYS + ["amount"], instruments=ids)
    work = keys[KEYS].merge(daily, on=KEYS, how="left")
    e = equd[KEYS + ["DEAL_AMOUNT", "TURNOVER_VALUE", "update_time"]].drop_duplicates(KEYS)
    work = work.merge(e, on=KEYS, how="left")
    amt = _finite(work["amount"])
    deal = _finite(work["DEAL_AMOUNT"])
    g1 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = amt.notna() & deal.gt(0) & np.isfinite(amt) & np.isfinite(deal)
    g1.loc[ok] = amt.loc[ok] / deal.loc[ok]
    work["r21_g1_avg_ticket_value"] = g1
    same = pd.to_datetime(work["update_time"]).dt.normalize() == pd.to_datetime(work["trade_date"]).dt.normalize()
    work["r21_g1_strict"] = (ok & same.fillna(False)).astype(np.float64)

    m = mf.drop_duplicates(KEYS)
    work = work.merge(m.drop(columns=["TICKER_SYMBOL", "EXCHANGE_CD", "TRADE_DATE"], errors="ignore"), on=KEYS, how="left", suffixes=("", "_mf"))
    buy_ord = _finite(work.get("BUY_ORD_S")).fillna(0) + _finite(work.get("BUY_ORD_M")).fillna(0) + _finite(work.get("BUY_ORD_L")).fillna(0) + _finite(work.get("BUY_ORD_XL")).fillna(0)
    sell_ord = _finite(work.get("SELL_ORD_S")).fillna(0) + _finite(work.get("SELL_ORD_M")).fillna(0) + _finite(work.get("SELL_ORD_L")).fillna(0) + _finite(work.get("SELL_ORD_XL")).fillna(0)
    # 只有原始字段存在时才算；全 0 视为无记录
    has_ord = work["BUY_ORD_XL"].notna() | work["SELL_ORD_XL"].notna()
    g3 = pd.Series(np.nan, index=work.index, dtype="float64")
    den = buy_ord + sell_ord
    ok3 = has_ord & den.gt(0)
    g3.loc[ok3] = (buy_ord.loc[ok3] - sell_ord.loc[ok3]) / (den.loc[ok3] + EPS)
    work["r21_g3_order_count_imbalance"] = g3

    bvx = _finite(work.get("BUY_VOL_XL"))
    box = _finite(work.get("BUY_ORD_XL"))
    svx = _finite(work.get("SELL_VOL_XL"))
    sox = _finite(work.get("SELL_ORD_XL"))
    g2 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok2 = bvx.gt(0) & box.gt(0) & svx.gt(0) & sox.gt(0)
    g2.loc[ok2] = np.log((bvx.loc[ok2] / box.loc[ok2]) / (svx.loc[ok2] / sox.loc[ok2]))
    work["r21_g2_large_ticket_asym"] = g2

    buy_vol = _finite(work.get("BUY_VOL_S")).fillna(0) + _finite(work.get("BUY_VOL_M")).fillna(0) + _finite(work.get("BUY_VOL_L")).fillna(0) + bvx.fillna(0)
    g4 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok4 = has_ord & buy_vol.gt(0) & buy_ord.gt(0) & bvx.notna() & box.notna()
    g4.loc[ok4] = (bvx.loc[ok4] / buy_vol.loc[ok4]) - (box.loc[ok4] / buy_ord.loc[ok4])
    work["r21_g4_large_ticket_conc_gap"] = g4
    ut = pd.to_datetime(work.get("update_time_mf", work.get("update_time")), errors="coerce")
    same_mf = ut.dt.normalize() == pd.to_datetime(work["trade_date"]).dt.normalize()
    work["r21_g2_strict"] = (ok2 & same_mf.fillna(False)).astype(np.float64)
    work["r21_g3_strict"] = (ok3 & same_mf.fillna(False)).astype(np.float64)
    work["r21_g4_strict"] = (ok4 & same_mf.fillna(False)).astype(np.float64)
    keep = KEYS + [c for c in work.columns if c.startswith("r21_g")]
    out = work[keep].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out, {"status": "DONE", "n": int(len(out)), "source": "raw mkt_equd + mkt_equ_mf_new"}


def build_h(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache("h_factors")
    if path.is_file():
        return pd.read_parquet(path), {"resumed": True}
    ids = set(keys["instrument_id"].astype(str))
    mf = load_hermes_table(
        "daily_money_flow",
        KEYS
        + [
            "net_in_open",
            "net_in_close",
            "net_inflow_s",
            "net_inflow_m",
            "net_inflow_l",
            "net_inflow_xl",
            "main_net_inflow",
            "available_at",
        ],
        instruments=ids,
    )
    daily = load_hermes_table("daily_raw", KEYS + ["amount"], instruments=ids)
    left = keys[KEYS].copy()
    left["known_at"] = _known_close(left["trade_date"])
    joined = asof_by_instrument(
        left,
        mf,
        left_on="known_at",
        right_on="available_at",
        columns=[
            "net_in_open",
            "net_in_close",
            "net_inflow_s",
            "net_inflow_m",
            "net_inflow_l",
            "net_inflow_xl",
            "main_net_inflow",
            "trade_date",
        ],
    )
    # asof 可能带入 mf.trade_date；改名
    if "trade_date_y" in joined.columns:
        joined = joined.rename(columns={"trade_date_y": "mf_trade_date", "trade_date_x": "trade_date"})
    joined = joined.merge(daily, on=KEYS, how="left")
    amt = _finite(joined["amount"])
    opn = _finite(joined["net_in_open"])
    cls = _finite(joined["net_in_close"])
    s = _finite(joined["net_inflow_s"])
    m = _finite(joined["net_inflow_m"])
    l = _finite(joined["net_inflow_l"])
    xl = _finite(joined["net_inflow_xl"])
    main = _finite(joined["main_net_inflow"])
    avail = pd.to_datetime(joined.get("available_at"), errors="coerce")
    known = pd.to_datetime(joined["known_at"], errors="coerce")
    # 2016–2020 money_flow available_at 为整表回填；STRICT 仅 2021+ 且 available_at<=known_at
    strict = avail.notna() & known.notna() & (avail <= known) & (avail.dt.year >= 2021)
    h1 = pd.Series(np.nan, index=joined.index, dtype="float64")
    ok = cls.notna() & opn.notna() & amt.notna() & np.isfinite(amt)
    h1.loc[ok] = (cls.loc[ok] - opn.loc[ok]) / (amt.loc[ok] + EPS)
    h2 = pd.Series(np.nan, index=joined.index, dtype="float64")
    ok2 = l.notna() & xl.notna() & s.notna() & m.notna() & amt.notna()
    h2.loc[ok2] = (l.loc[ok2] + xl.loc[ok2] - s.loc[ok2] - m.loc[ok2]) / (amt.loc[ok2] + EPS)
    main_ratio = pd.Series(np.nan, index=joined.index, dtype="float64")
    okm = main.notna() & amt.notna()
    main_ratio.loc[okm] = main.loc[okm] / (amt.loc[okm] + EPS)
    dup = None
    both = h2.notna() & main_ratio.notna()
    if int(both.sum()) > 100:
        rho = h2.loc[both].corr(main_ratio.loc[both])
        dup = float(rho) if rho is not None and np.isfinite(rho) else None
    h3 = pd.Series(np.nan, index=joined.index, dtype="float64")
    ok3 = xl.notna() & s.notna() & (xl != 0) & (s != 0)
    h3.loc[ok3] = np.sign(xl.loc[ok3]) * np.sign(s.loc[ok3])
    large = l.fillna(0) + xl.fillna(0)
    hist = joined[KEYS + ["available_at"]].copy()
    hist["large_net"] = large
    hist = hist.sort_values(KEYS)
    # 用 mf 原始日序列做滞后持续性：对每只股票 shift 1..5
    raw = mf[KEYS + ["net_inflow_l", "net_inflow_xl", "available_at"]].copy()
    raw["large"] = _finite(raw["net_inflow_l"]).fillna(0) + _finite(raw["net_inflow_xl"]).fillna(0)
    raw = raw.sort_values(KEYS)
    raw["sgn"] = np.sign(raw["large"].replace(0, np.nan))
    for k in range(1, 6):
        raw[f"sgn_l{k}"] = raw.groupby("instrument_id", sort=False)["sgn"].shift(k)
    raw["persist"] = raw[[f"sgn_l{k}" for k in range(1, 6)]].mean(axis=1)
    persist_map = raw[KEYS + ["persist"]].drop_duplicates(KEYS)
    joined = joined.merge(persist_map, on=KEYS, how="left")
    h4 = pd.Series(np.nan, index=joined.index, dtype="float64")
    pers = _finite(joined["persist"])
    intensity = pd.Series(np.nan, index=joined.index, dtype="float64")
    oki = large.notna() & amt.notna()
    intensity.loc[oki] = large.loc[oki].abs() / (amt.loc[oki] + EPS)
    ok4 = pers.notna() & intensity.notna()
    h4.loc[ok4] = pers.loc[ok4] * intensity.loc[ok4]
    out = joined[KEYS].copy()
    out["r21_h1_late_flow_dominance"] = np.where(strict, h1, np.nan)
    out["r21_h2_large_small_flow_div"] = np.where(strict, h2, np.nan)
    out["r21_h3_flow_dir_disagree"] = np.where(strict, h3, np.nan)
    out["r21_h4_large_flow_persist"] = np.where(strict, h4, np.nan)
    out["r21_h1_research"] = h1
    out["r21_h2_research"] = h2
    out["r21_h1_strict"] = strict.astype(np.float64)
    out["r21_h2_duplicate_rho_vs_main"] = dup
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out, {"status": "DONE", "h2_vs_main_rho": dup, "duplicate_if_abs_rho_ge_0.999": bool(dup is not None and abs(dup) >= 0.999)}


def _failed_flag(ev: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    evu = ev.loc[ev["limit_type"].astype(str).eq("up"), KEYS + ["first_limit_time_seconds", "close_limit_value", "limit_price"]].copy()
    work = daily.merge(evu, on=KEYS, how="left")
    close = _finite(work["close"])
    lim = _finite(work["limit_up"])
    miss = lim.isna()
    lim.loc[miss] = _finite(work.loc[miss, "limit_price"])
    touch = _finite(work["first_limit_time_seconds"]).notna() | (close.notna() & lim.notna() & close.ge(lim * 0.995))
    sealed = close.notna() & lim.notna() & close.ge(lim * 0.999)
    failed = touch & ~sealed
    work["touched"] = touch.astype(np.float64)
    work["failed"] = failed.astype(np.float64)
    work["sealed"] = sealed.astype(np.float64)
    return work[KEYS + ["touched", "failed", "sealed"]]


def build_i(keys: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache("i_factors")
    if path.is_file():
        return pd.read_parquet(path), {"resumed": True}
    ids = set(keys["instrument_id"].astype(str))
    ev = load_hermes_table("daily_limit_events", KEYS + ["limit_type", "limit_price", "first_limit_time_seconds", "close_limit_value"], instruments=ids)
    daily = load_hermes_table("daily_raw", KEYS + ["close", "limit_up"], instruments=ids)
    flags = _failed_flag(ev, daily).sort_values(KEYS).reset_index(drop=True)
    cal = pd.DatetimeIndex(calendar).sort_values()
    g = flags.groupby("instrument_id", sort=False)
    failed_l1 = g["failed"].shift(1)
    touched_l1 = g["touched"].shift(1)
    sealed_l1 = g["sealed"].shift(1)
    fsum = failed_l1.groupby(flags["instrument_id"], sort=False).transform(lambda s: s.rolling(20, min_periods=1).sum())
    tsum = touched_l1.groupby(flags["instrument_id"], sort=False).transform(lambda s: s.rolling(20, min_periods=1).sum())
    i1 = pd.Series(np.nan, index=flags.index, dtype="float64")
    ok1 = (tsum.fillna(0) > 0) | (fsum.fillna(0) > 0)
    i1.loc[ok1] = fsum.loc[ok1] / tsum.loc[ok1].clip(lower=1.0)
    i3 = pd.Series(0.0, index=flags.index, dtype="float64")
    has_hist = pd.Series(False, index=flags.index)
    for k in range(1, 21):
        shifted = g["failed"].shift(k)
        hit = shifted.notna()
        i3 = i3 + (0.9 ** (k - 1)) * shifted.fillna(0.0)
        has_hist = has_hist | hit
    i3 = i3.where(has_hist)
    cum_t = touched_l1.groupby(flags["instrument_id"], sort=False).cumsum()
    cum_s = sealed_l1.groupby(flags["instrument_id"], sort=False).cumsum()
    i4 = pd.Series(np.nan, index=flags.index, dtype="float64")
    ok4 = cum_t.gt(0)
    i4.loc[ok4] = cum_s.loc[ok4] / cum_t.loc[ok4]
    fail_dates = flags["trade_date"].where(flags["failed"].eq(1.0))
    last_incl = fail_dates.groupby(flags["instrument_id"], sort=False).ffill()
    last_before = last_incl.where(flags["failed"].ne(1.0), last_incl.groupby(flags["instrument_id"], sort=False).shift(1))
    i2 = pd.Series(np.nan, index=flags.index, dtype="float64")
    a = pd.to_datetime(last_before, errors="coerce")
    b = pd.to_datetime(flags["trade_date"], errors="coerce")
    ok2 = a.notna() & b.notna()
    if bool(ok2.any()):
        ia = cal.searchsorted(a.loc[ok2], side="right")
        ib = cal.searchsorted(b.loc[ok2], side="left")
        i2.loc[ok2] = (ib - ia).astype("float64")
    hist = flags[KEYS].copy()
    hist["r21_i1_failed_board_ratio_20"] = i1
    hist["r21_i2_days_since_failed"] = i2
    hist["r21_i3_failed_recurrence_w20"] = i3
    hist["r21_i4_hit_to_close_conversion"] = i4
    out = keys[KEYS].merge(hist, on=KEYS, how="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out, {"status": "DONE", "n": int(len(out))}


def build_j(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache("j_factors")
    if path.is_file():
        return pd.read_parquet(path), {"resumed": True}
    ids = set(keys["instrument_id"].astype(str))
    daily = load_hermes_table("daily_raw", KEYS + ["open", "high", "low", "close", "volume", "amount"], instruments=ids)
    val = load_hermes_table("daily_valuation", KEYS + ["turnover_rate", "available_at"], instruments=ids)
    daily = daily.sort_values(KEYS)
    daily["close"] = _finite(daily["close"])
    daily["high"] = _finite(daily["high"])
    daily["volume"] = _finite(daily["volume"])
    daily["amount"] = _finite(daily["amount"])
    daily["ret1"] = daily.groupby("instrument_id", sort=False)["close"].pct_change()
    # 滞后：D 日因子只用 D-1 及更早
    g = daily.groupby("instrument_id", sort=False)
    daily["amt_20"] = g["amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).sum())
    daily["vol_20"] = g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).sum())
    daily["high_20"] = g["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
    daily["close_l1"] = g["close"].shift(1)
    daily["ret_5"] = g["close"].transform(lambda s: s.shift(1) / s.shift(6) - 1.0)
    v = val.sort_values(KEYS)
    v["turnover_rate"] = _finite(v["turnover_rate"])
    v["to_l1"] = v.groupby("instrument_id", sort=False)["turnover_rate"].shift(1)
    v["to_sum5"] = v.groupby("instrument_id", sort=False)["to_l1"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    work = keys[KEYS].merge(
        daily[KEYS + ["close_l1", "amt_20", "vol_20", "high_20", "ret_5"]],
        on=KEYS,
        how="left",
    )
    work = work.merge(v[KEYS + ["to_sum5", "turnover_rate"]], on=KEYS, how="left")
    j1 = _finite(work["to_sum5"])
    j2 = pd.Series(np.nan, index=work.index, dtype="float64")
    vwap = _finite(work["amt_20"]) / _finite(work["vol_20"]).replace(0, np.nan)
    cl = _finite(work["close_l1"])
    ok = cl.notna() & vwap.notna() & vwap.gt(0)
    j2.loc[ok] = cl.loc[ok] / vwap.loc[ok] - 1.0
    j3 = pd.Series(np.nan, index=work.index, dtype="float64")
    hx = _finite(work["high_20"])
    ok3 = cl.notna() & hx.notna() & hx.gt(0)
    j3.loc[ok3] = cl.loc[ok3] / hx.loc[ok3] - 1.0
    j4 = pd.Series(np.nan, index=work.index, dtype="float64")
    r5 = _finite(work["ret_5"])
    ok4 = r5.notna() & j1.notna()
    j4.loc[ok4] = r5.loc[ok4] / (j1.loc[ok4] + EPS)
    out = work[KEYS].copy()
    out["r21_j1_preboard_turnover_5"] = j1
    out["r21_j2_preboard_px_vwap20"] = j2
    out["r21_j3_preboard_high_prox"] = j3
    out["r21_j4_prior_ret_per_turnover"] = j4
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out, {"status": "DONE"}


def _k_one_day(minutes: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    out = part[KEYS].copy()
    for c in ("r21_k1_pre_hit_vol_frac", "r21_k2_vol_time_dispersion", "r21_k3_minute_vol_entropy", "r21_k4_signed_path_vol_pressure", "r21_k4_pre_hit"):
        out[c] = np.nan
    if minutes.empty:
        return out
    m = minutes.copy()
    m["instrument_id"] = m["instrument_id"].astype(str)
    if "bar_time" in m.columns:
        m["bar_sec"] = _bar_seconds(m["bar_time"])
    vol = pd.to_numeric(m.get("volume", m.get("vol")), errors="coerce")
    if vol is None or vol.isna().all():
        amt = pd.to_numeric(m.get("amount"), errors="coerce")
        vol = amt
    m["volume"] = vol
    m["close"] = pd.to_numeric(m.get("close"), errors="coerce")
    recs = []
    lim = part.copy()
    lim["instrument_id"] = lim["instrument_id"].astype(str)
    joined = m.merge(lim[["instrument_id", "touch_seconds"]], on="instrument_id", how="inner")
    for iid, g in joined.groupby("instrument_id", sort=False):
        v = pd.to_numeric(g["volume"], errors="coerce").to_numpy(dtype=float)
        sec = pd.to_numeric(g["bar_sec"], errors="coerce").to_numpy(dtype=float)
        cl = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=float)
        touch = float(pd.to_numeric(g["touch_seconds"].iloc[0], errors="coerce"))
        rec = {"instrument_id": str(iid)}
        ok = np.isfinite(v) & (v > 0) & np.isfinite(sec)
        if int(ok.sum()) == 0:
            recs.append(rec)
            continue
        vv, ss, cc = v[ok], sec[ok], cl[ok]
        tot = float(vv.sum())
        if tot <= 0:
            recs.append(rec)
            continue
        if np.isfinite(touch):
            pre = ss < touch
            if int(pre.sum()) > 0:
                rec["r21_k1_pre_hit_vol_frac"] = float(vv[pre].sum() / tot)
                dcl = np.diff(cc, prepend=cc[0])
                sgn = np.sign(dcl)
                sgn[dcl == 0] = 0
                rec["r21_k4_pre_hit"] = float(np.dot(vv, sgn) / tot)  # will overwrite below for full
        mu = float(np.dot(vv, ss) / tot)
        rec["r21_k2_vol_time_dispersion"] = float(np.dot(vv, (ss - mu) ** 2) / tot)
        p = vv / tot
        p = p[p > 0]
        n = max(int(len(p)), 2)
        rec["r21_k3_minute_vol_entropy"] = float(-np.sum(p * np.log(p)) / np.log(n))
        dcl = np.diff(cc, prepend=cc[0])
        sgn = np.sign(dcl)
        sgn[~np.isfinite(dcl)] = 0
        sgn[dcl == 0] = 0
        rec["r21_k4_signed_path_vol_pressure"] = float(np.dot(vv, sgn) / tot)
        recs.append(rec)
    if recs:
        got = pd.DataFrame(recs)
        out = out.drop(columns=[c for c in out.columns if c.startswith("r21_k")]).merge(got, on="instrument_id", how="left")
        out["trade_date"] = part["trade_date"].iloc[0]
    return out


def _k_proc(root: str, part: pd.DataFrame) -> pd.DataFrame:
    files = list(Path(root).glob("*.parquet"))
    minutes = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    return _k_one_day(minutes, part)


def build_k(keys: pd.DataFrame, *, jobs: int = 8) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache("k_factors")
    if path.is_file():
        return pd.read_parquet(path), {"resumed": True}
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from ..round21.min1 import _load_limit_px

    ac_path = Path(__file__).resolve().parents[2] / "result" / "round21" / "data" / "ac_factors.parquet"
    base = _load_limit_px(keys)
    if ac_path.is_file():
        ac = pd.read_parquet(ac_path, columns=KEYS + ["touch_seconds"])
        ac["trade_date"] = pd.to_datetime(ac["trade_date"])
        ac["instrument_id"] = ac["instrument_id"].astype(str)
        base = base.merge(ac, on=KEYS, how="left", suffixes=("", "_ac"))
        if "touch_seconds" not in base.columns and "touch_seconds_ac" in base.columns:
            base["touch_seconds"] = base["touch_seconds_ac"]
        elif "touch_seconds_ac" in base.columns:
            base["touch_seconds"] = _finite(base["touch_seconds"]).fillna(_finite(base["touch_seconds_ac"]))
    else:
        base["touch_seconds"] = _finite(base.get("first_limit_time_seconds"))
    tasks = []
    parts = []
    for day, block in base.groupby("trade_date", sort=True):
        root = _min_root(pd.Timestamp(day))
        part = block[KEYS + ["touch_seconds"]].copy()
        if root is None:
            empty = part[KEYS].copy()
            for c in ("r21_k1_pre_hit_vol_frac", "r21_k2_vol_time_dispersion", "r21_k3_minute_vol_entropy", "r21_k4_signed_path_vol_pressure"):
                empty[c] = np.nan
            parts.append(empty)
            continue
        tasks.append((str(root), part))

    if tasks:
        workers = min(max(1, int(jobs)), 8)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_k_proc, root, part) for root, part in tasks]
            done = 0
            for fut in as_completed(futs):
                parts.append(fut.result())
                done += 1
                if done % 400 == 0:
                    print(f"[round21.2/k] days {done}/{len(tasks)}", flush=True)
    k = pd.concat(parts, ignore_index=True) if parts else keys[KEYS].copy()
    out = keys[KEYS].merge(k, on=KEYS, how="left")
    # entropy vs vendor
    try:
        intra = load_hermes_table("daily_intraday_factors", KEYS + ["volume_weighted_entropy"], instruments=set(keys["instrument_id"]))
        tmp = out.merge(intra, on=KEYS, how="left")
        a = _finite(tmp["r21_k3_minute_vol_entropy"])
        b = _finite(tmp["volume_weighted_entropy"])
        both = a.notna() & b.notna()
        rho = float(a.loc[both].corr(b.loc[both], method="spearman")) if int(both.sum()) > 50 else None
    except Exception:
        rho = None
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    dump_json(ROUND21_2_ROOT / "data" / "k3_vs_vendor.json", {"spearman_vs_volume_weighted_entropy": rho})
    return out, {"status": "DONE", "k3_vs_vendor_spearman": rho}


def build_l(keys: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache("l_factors")
    if path.is_file():
        return pd.read_parquet(path), {"resumed": True}
    sc = load_hermes_table("share_capital", ["instrument_id", "available_at", "announced_at", "tradable_shares", "effective_from"])
    sc["available_at"] = pd.to_datetime(sc["available_at"], errors="coerce")
    sc["tradable_shares"] = _finite(sc["tradable_shares"])
    sc = sc.dropna(subset=["instrument_id", "available_at"]).sort_values(["instrument_id", "available_at"])
    sc["prev_float"] = sc.groupby("instrument_id", sort=False)["tradable_shares"].shift(1)
    sc["float_chg"] = sc["tradable_shares"] / sc["prev_float"] - 1.0
    sc["log_chg"] = np.log(sc["tradable_shares"] / sc["prev_float"])
    sc["is_increase"] = sc["float_chg"] > 0
    left = keys[KEYS].copy()
    left["known_at"] = _known_close(left["trade_date"])
    joined = asof_by_instrument(left, sc, left_on="known_at", right_on="available_at", columns=["tradable_shares", "prev_float", "float_chg", "log_chg", "is_increase", "announced_at"])
    ids = set(keys["instrument_id"].astype(str))
    daily = load_hermes_table("daily_raw", KEYS + ["volume"], instruments=ids)
    daily = daily.sort_values(KEYS)
    daily["volume"] = _finite(daily["volume"])
    daily["vol20"] = daily.groupby("instrument_id", sort=False)["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    val = load_hermes_table("daily_valuation", KEYS + ["turnover_rate"], instruments=ids)
    work = joined.merge(daily[KEYS + ["vol20"]], on=KEYS, how="left").merge(val, on=KEYS, how="left")
    l1 = _finite(work.get("log_chg"))
    # 要求 prev 当时已知：prev_float 非空
    l1 = l1.where(_finite(work.get("prev_float")).notna())
    l2 = pd.Series(np.nan, index=work.index, dtype="float64")
    chg_abs = _finite(work.get("tradable_shares")) - _finite(work.get("prev_float"))
    v20 = _finite(work["vol20"])
    ok = chg_abs.notna() & v20.gt(0)
    l2.loc[ok] = chg_abs.loc[ok] / v20.loc[ok]
    # days since last increase announcement
    inc = sc.loc[sc["is_increase"].fillna(False), ["instrument_id", "available_at"]].copy()
    l3 = pd.Series(np.nan, index=work.index, dtype="float64")
    if not inc.empty:
        tmp = asof_by_instrument(left, inc, left_on="known_at", right_on="available_at", columns=[])
        last = pd.to_datetime(tmp.get("available_at"), errors="coerce")
        l3 = _trading_days_between(last, work["trade_date"], calendar)
    l4 = pd.Series(np.nan, index=work.index, dtype="float64")
    to = _finite(work.get("turnover_rate"))
    ratio = _finite(work.get("float_chg"))
    ok4 = to.notna() & ratio.notna()
    l4.loc[ok4] = to.loc[ok4] * ratio.loc[ok4]
    out = work[KEYS].copy()
    out["r21_l1_known_float_chg"] = l1
    out["r21_l2_float_shock_vs_liq"] = l2
    out["r21_l3_days_since_float_up_ann"] = l3
    out["r21_l4_turnover_x_float_chg"] = l4
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out, {"status": "DONE"}


def build_m(keys: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache("m_factors")
    if path.is_file():
        return pd.read_parquet(path), {"resumed": True}
    fm = load_hermes_table(
        "fundamental_metrics",
        ["instrument_id", "metric_code", "value", "available_at", "fiscal_period_end"],
    )
    fm["available_at"] = pd.to_datetime(fm["available_at"], errors="coerce")
    fm["value"] = _finite(fm["value"])
    fm = fm.dropna(subset=["instrument_id", "available_at", "metric_code"])
    codes = {
        "rev": ["hermes_total_revenue_yoy_q_pct", "revenue_yoy"],
        "np": ["net_profit_parent_yoy", "net_profit"],
        "ocf": ["operating_cash_flow"],
        "np_lvl": ["net_profit", "net_profit_parent"],
    }

    def _latest(code_list: list[str]) -> pd.DataFrame:
        sub = fm.loc[fm["metric_code"].astype(str).isin(code_list)].copy()
        if sub.empty:
            return pd.DataFrame(columns=["instrument_id", "available_at", "value", "prev"])
        sub = sub.sort_values(["instrument_id", "available_at", "fiscal_period_end"])
        sub = sub.drop_duplicates(["instrument_id", "available_at", "metric_code"], keep="last")
        # 每个 instrument 按 available_at 取该日最新 code 优先级
        for c in code_list:
            one = sub.loc[sub["metric_code"].astype(str).eq(c), ["instrument_id", "available_at", "value"]].copy()
            if not one.empty:
                one["prev"] = one.groupby("instrument_id", sort=False)["value"].shift(1)
                return one
        return pd.DataFrame(columns=["instrument_id", "available_at", "value", "prev"])

    left = keys[KEYS].copy()
    left["known_at"] = _known_close(left["trade_date"])
    rev = _latest(codes["rev"])
    npy = _latest(["net_profit_parent_yoy"])
    ocf = _latest(codes["ocf"])
    npl = _latest(codes["np_lvl"])
    jrev = asof_by_instrument(left, rev, left_on="known_at", right_on="available_at", columns=["value", "prev"])
    jnpy = asof_by_instrument(left, npy, left_on="known_at", right_on="available_at", columns=["value"])
    jocf = asof_by_instrument(left, ocf, left_on="known_at", right_on="available_at", columns=["value"])
    jnpl = asof_by_instrument(left, npl, left_on="known_at", right_on="available_at", columns=["value"])
    m1 = _finite(jrev.get("value")) - _finite(jrev.get("prev"))
    m2 = pd.Series(np.nan, index=left.index, dtype="float64")
    o = _finite(jocf.get("value"))
    n = _finite(jnpl.get("value"))
    ok = o.notna() & n.notna() & (n.abs() > EPS)
    m2.loc[ok] = o.loc[ok] / n.loc[ok].abs()
    last_disc = pd.to_datetime(jrev.get("available_at"), errors="coerce")
    m3 = _trading_days_between(last_disc, left["trade_date"], calendar)
    m4 = _finite(jrev.get("value")) - _finite(jnpy.get("value"))
    out = left[KEYS].copy()
    out["r21_m1_rev_yoy_change"] = m1
    out["r21_m2_ocf_over_abs_np"] = m2
    out["r21_m3_earnings_info_age"] = m3
    out["r21_m4_growth_quality_div"] = m4
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out, {"status": "DONE", "rev_code_used": codes["rev"]}


def build_n(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache("n_factors")
    if path.is_file():
        return pd.read_parquet(path), {"resumed": True}
    ind = pd.read_parquet(HERMES_V7 / "industry_membership")
    ind = ind.loc[ind["classification"].astype(str).eq("SW_L1:SW")].copy()
    ind["instrument_id"] = ind["instrument_id"].astype(str)
    ind["known_at"] = pd.to_datetime(ind["known_at"], errors="coerce")
    ind["effective_from"] = pd.to_datetime(ind.get("effective_from"), errors="coerce")
    ind["industry_id"] = ind["industry_id"].astype(str)
    left = keys[KEYS].copy()
    left["known_at"] = _known_close(left["trade_date"])
    # asof industry
    jind = asof_by_instrument(left, ind, left_on="known_at", right_on="known_at", columns=["industry_id"])
    ids_all = set(ind["instrument_id"].astype(str))
    daily = load_hermes_table("daily_raw", KEYS + ["close", "amount", "limit_up"], instruments=ids_all)
    ev = load_hermes_table("daily_limit_events", KEYS + ["limit_type", "first_limit_time_seconds", "close_limit_value", "limit_price"], instruments=ids_all)
    flags = _failed_flag(ev, daily)
    daily = daily.sort_values(KEYS)
    daily["close"] = _finite(daily["close"])
    daily["amount"] = _finite(daily["amount"])
    daily["ret5"] = daily.groupby("instrument_id", sort=False)["close"].transform(lambda s: s.shift(1) / s.shift(6) - 1.0)
    daily["limit_up_close"] = (_finite(daily["close"]) >= _finite(daily["limit_up"]) * 0.999).astype(float)
    hist = daily[KEYS + ["ret5", "amount", "limit_up_close"]].merge(flags, on=KEYS, how="left")
    hist = hist.merge(jind[KEYS + ["industry_id"]], on=KEYS, how="left")
    hist["industry_id"] = hist["industry_id"].astype(str)
    hist = hist.sort_values(["industry_id", "trade_date"])
    # 日×行业聚合（排除在 merge 后按自身减）
    day_ind = (
        hist.groupby(["trade_date", "industry_id"], sort=False)
        .agg(
            peer_n=("instrument_id", "size"),
            peer_limit=("limit_up_close", "mean"),
            peer_failed=("failed", "mean"),
            peer_ret5=("ret5", "mean"),
            peer_amt=("amount", "mean"),
        )
        .reset_index()
    )
    # 滞后窗口
    day_ind = day_ind.sort_values(["industry_id", "trade_date"])
    g = day_ind.groupby("industry_id", sort=False)
    day_ind["peer_limit_5"] = g["peer_limit"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    day_ind["peer_failed_10"] = g["peer_failed"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    work = jind.merge(hist[KEYS + ["ret5", "amount", "limit_up_close"]], on=KEYS, how="left")
    work = work.merge(day_ind, on=["trade_date", "industry_id"], how="left")
    # 自身修正：peer mean 含自身时用 (n*mean - self)/(n-1)
    n = _finite(work["peer_n"])
    own_lim = _finite(work["limit_up_close"])
    own_fail = _finite(work.get("failed"))
    own_ret = _finite(work["ret5"])
    own_amt = _finite(work["amount"])
    adj = n.gt(1)
    peer_lim = _finite(work["peer_limit"])
    peer_fail = _finite(work["peer_failed"])
    peer_ret = _finite(work["peer_ret5"])
    peer_amt = _finite(work["peer_amt"])
    peer_lim_ex = peer_lim.where(~adj, (peer_lim * n - own_lim.fillna(0)) / (n - 1))
    peer_ret_ex = peer_ret.where(~adj, (peer_ret * n - own_ret.fillna(0)) / (n - 1))
    peer_amt_ex = peer_amt.where(~adj, (peer_amt * n - own_amt.fillna(0)) / (n - 1))
    out = work[KEYS].copy()
    out["r21_n1_lagged_peer_limit_up"] = _finite(work["peer_limit_5"])
    out["r21_n2_peer_return_div"] = own_ret - peer_ret_ex
    out["r21_n3_peer_failed_spillover"] = _finite(work["peer_failed_10"])
    intensity = np.log1p(own_amt)
    peer_int = np.log1p(peer_amt_ex)
    out["r21_n4_peer_adj_fb_intensity"] = intensity - peer_int
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out, {"status": "DONE", "peer": "SW_L1:SW equal-weight exclude-self where possible"}


def run_construct(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    if resume and PANEL_PATH.is_file() and CONSTRUCT_JSON.is_file() and MD["04"].is_file():
        from .common import load_json

        existing = load_json(CONSTRUCT_JSON)
        if existing.get("status") == "DONE":
            return existing
    require_hypotheses()
    keys = load_candidate_keys()
    cal = load_trade_calendar()
    blocked: list[str] = []
    frames = [keys[KEYS].copy()]
    metas: dict[str, Any] = {}
    builders = [
        ("G", lambda: build_g(keys)),
        ("H", lambda: build_h(keys)),
        ("I", lambda: build_i(keys, cal)),
        ("J", lambda: build_j(keys)),
        ("K", lambda: build_k(keys, jobs=jobs)),
        ("L", lambda: build_l(keys, cal)),
        ("M", lambda: build_m(keys, cal)),
        ("N", lambda: build_n(keys)),
    ]
    for fam, fn in builders:
        print(f"[round21.2] construct family {fam}", flush=True)
        try:
            frame, meta = fn()
            metas[fam] = meta
            extra = [c for c in frame.columns if c not in KEYS]
            frames.append(frame[KEYS + extra])
        except Exception as exc:
            metas[fam] = {"status": BLOCKED_DATA, "error": str(exc)}
            blocked.append(fam)
            print(f"[round21.2] family {fam} BLOCKED_DATA: {exc}", flush=True)
    panel = frames[0]
    for fr in frames[1:]:
        extra = [c for c in fr.columns if c not in panel.columns]
        if extra:
            panel = panel.merge(fr[KEYS + extra], on=KEYS, how="left")
    # 主值列
    present = [c for c in NEW_FACTOR_IDS if c in panel.columns]
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    dest = ROUND21_2_ROOT / "factors"
    for fid in present:
        d = dest / fid
        d.mkdir(parents=True, exist_ok=True)
        panel[KEYS + [fid]].to_parquet(d / "values.parquet", index=False)
    payload = {
        "status": "DONE",
        "n": int(len(panel)),
        "computed": present,
        "n_computed": len(present),
        "n_proposed": len(NEW_FACTOR_IDS),
        "blocked_families": blocked,
        "filled_with_zero": False,
        "families": metas,
        "path": str(PANEL_PATH),
    }
    dump_json(CONSTRUCT_JSON, payload)
    dump_json(PANEL_JSON, {"n": int(len(panel)), "columns": list(panel.columns)})
    write_md(
        MD["04"],
        "\n".join(
            [
                "# Round21.2 · 04 因子构造",
                "",
                f"proposed=`32` computed=`{len(present)}` blocked_families=`{blocked}` filled_with_zero=False",
                f"ids=`{present}`",
                "",
            ]
        ),
    )
    return payload
