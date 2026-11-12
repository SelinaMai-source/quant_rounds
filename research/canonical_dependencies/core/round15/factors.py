"""期权链 / IV / O_F1–F4。无数据 SKIP，不把 PC ratio 叫 RN kurtosis。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND15_ROOT
from .common import dump_json, load_json, write_md
from .contract import IV_MIN_CHAIN, PCT_MIN_OBS, PCT_WINDOW
from .freeze import load_skips, require_frozen
from .sdk import DATA_ROOT

FACTOR_JSON = ROUND15_ROOT / "factors" / "factors.json"
FACTOR_MD = ROUND15_ROOT / "04_option_factors.md"
FACTOR_PARQUET = ROUND15_ROOT / "factors" / "option_factors_daily.parquet"
TEST_JSON = ROUND15_ROOT / "tests" / "iv_unit_tests.json"


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(spot: float, strike: float, t_years: float, rate: float, sigma: float, *, call: bool) -> float:
    if t_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0.0) if call else max(strike - spot, 0.0)
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    df = math.exp(-rate * t_years)
    if call:
        return spot * _norm_cdf(d1) - strike * df * _norm_cdf(d2)
    return strike * df * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    *,
    call: bool,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 80,
) -> float | None:
    if not all(map(np.isfinite, [price, spot, strike, t_years, rate])) or price <= 0 or t_years <= 0:
        return None
    intrinsic = max(spot - strike, 0.0) if call else max(strike - spot, 0.0)
    if price < intrinsic * 0.5 and price < 1e-8:
        return None
    a, b = lo, hi
    fa = bs_price(spot, strike, t_years, rate, a, call=call) - price
    fb = bs_price(spot, strike, t_years, rate, b, call=call) - price
    if fa * fb > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = bs_price(spot, strike, t_years, rate, mid, call=call) - price
        if abs(fm) < tol:
            return float(mid)
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return float(0.5 * (a + b))


def bs_delta(spot: float, strike: float, t_years: float, rate: float, sigma: float, *, call: bool) -> float | None:
    if t_years <= 0 or sigma <= 0 or spot <= 0:
        return None
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt_t)
    return float(_norm_cdf(d1) if call else _norm_cdf(d1) - 1.0)


def pc_parity_gap(call: float, put: float, spot: float, strike: float, t_years: float, rate: float) -> float:
    df = math.exp(-rate * t_years)
    return float((call - put) - (spot - strike * df))


def run_iv_unit_tests() -> dict[str, Any]:
    spot, strike, t, r, sig = 3.0, 3.0, 30 / 365.0, 0.02, 0.25
    call = bs_price(spot, strike, t, r, sig, call=True)
    put = bs_price(spot, strike, t, r, sig, call=False)
    iv_c = implied_vol(call, spot, strike, t, r, call=True)
    iv_p = implied_vol(put, spot, strike, t, r, call=False)
    gap = pc_parity_gap(call, put, spot, strike, t, r)
    invalid_zero = implied_vol(0.0, spot, strike, t, r, call=True)
    tests = {
        "synthetic_call_roundtrip": bool(iv_c is not None and abs(iv_c - sig) < 1e-3),
        "synthetic_put_roundtrip": bool(iv_p is not None and abs(iv_p - sig) < 1e-3),
        "pc_parity_near_zero": bool(abs(gap) < 1e-8),
        "invalid_quote_not_filled_with_zero": invalid_zero is None,
        "numerical_stable_atm": bool(np.isfinite(call) and call > 0),
    }
    return {
        "status": "PASS" if all(tests.values()) else "FAIL",
        "tests": tests,
        "call": call,
        "put": put,
        "iv_c": iv_c,
        "iv_p": iv_p,
        "parity_gap": gap,
    }


def _load_option_frames() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    basic = DATA_ROOT / "option_basic_info.parquet"
    if basic.is_file():
        out["basic"] = pd.read_parquet(basic)
    kdir = DATA_ROOT / "option_kline_day"
    parts = sorted(kdir.glob("part-*.parquet")) if kdir.is_dir() else []
    if parts:
        frames = [pd.read_parquet(p) for p in parts]
        out["kline"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sample_k = DATA_ROOT / "sample" / "option_kline_sample.parquet"
    sample_b = DATA_ROOT / "sample" / "option_basic_info_sample.parquet"
    if "kline" not in out and sample_k.is_file():
        out["kline"] = pd.read_parquet(sample_k)
    if "basic" not in out and sample_b.is_file():
        out["basic"] = pd.read_parquet(sample_b)
    etf = DATA_ROOT / "etf_daily.parquet"
    if etf.is_file():
        out["etf"] = pd.read_parquet(etf)
    return out


def _pick_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for c in df.columns:
        cl = str(c).lower()
        if any(n.lower() in cl for n in names):
            return c
    return None


def _try_build_factors(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {
        "stable_chain": False,
        "bilateral_25d": False,
        "has_10d_put": False,
        "chain_dense": False,
        "available_at_recorded": False,
        "pit_ok": False,
        "rf_source": None,
        "invalid_quotes_filled_with_zero": False,
    }
    kline = frames.get("kline")
    basic = frames.get("basic")
    if kline is None or kline.empty:
        meta["reason"] = "NO_OPTION_KLINE"
        return pd.DataFrame(), meta
    date_col = _pick_col(kline, ("trade_date", "date", "kline_time", "datetime", "time"))
    close_col = _pick_col(kline, ("close", "Close", "px_close"))
    if date_col is None or close_col is None:
        meta["reason"] = "KLINE_MISSING_DATE_OR_CLOSE"
        meta["kline_cols"] = list(kline.columns)
        return pd.DataFrame(), meta
    work = kline.copy()
    work["_date"] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work["_px"] = pd.to_numeric(work[close_col], errors="coerce")
    work = work.loc[work["_px"].gt(0) & work["_date"].notna()]
    if basic is not None and not basic.empty:
        meta["basic_cols"] = list(basic.columns)
    n_days = int(work["_date"].nunique())
    n_codes = int(work.get("_key", work.index.to_series()).nunique()) if "_key" in work.columns else int(len(work))
    meta["n_option_days"] = n_days
    meta["n_quote_rows"] = int(len(work))
    meta["n_codes_in_kline"] = n_codes
    # 没有 strike/cp/expiry 列则无法反推 IV 面。
    strike_col = None
    cp_col = None
    if basic is not None:
        strike_col = _pick_col(basic, ("strike", "k", "exercise_price", "exe_price"))
        cp_col = _pick_col(basic, ("call_put", "cp", "option_type", "type"))
    meta["has_strike"] = strike_col is not None
    meta["has_cp"] = cp_col is not None
    if not meta["has_strike"] or not meta["has_cp"]:
        meta["reason"] = "NO_STRIKE_OR_CP_FOR_IV"
        meta["stable_chain"] = n_days >= 20 and n_codes >= IV_MIN_CHAIN
        return pd.DataFrame(), meta
    meta["reason"] = "CHAIN_PRESENT_BUT_IV_SURFACE_NOT_BUILT_WITHOUT_PIT_FIELDS"
    meta["stable_chain"] = False
    return pd.DataFrame(), meta


def run_factors(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and FACTOR_JSON.is_file():
        existing = load_json(FACTOR_JSON)
        if existing.get("status") in {"DONE", "SKIP_IV_SURFACE"}:
            return existing
    tests = run_iv_unit_tests()
    dump_json(TEST_JSON, tests)
    skips = load_skips()
    frames = _load_option_frames()
    daily, meta = _try_build_factors(frames)
    skip_iv = bool(skips.get("SKIP_IV_SURFACE", True)) or daily.empty
    if skip_iv:
        payload = {
            "status": "SKIP_IV_SURFACE",
            "skip": True,
            "meta": meta,
            "unit_tests": tests,
            "n_factor_days": 0,
            "O_F1": None,
            "O_F2": None,
            "O_F3": "SKIP_F3",
            "O_F4": None,
            "note": "无稳定 PIT 期权链/IV。不伪造。不把 PC ratio 叫 RN kurtosis。",
        }
        dump_json(FACTOR_JSON, payload)
        write_md(
            FACTOR_MD,
            "\n".join([
                "# Round 15 · 期权因子",
                "",
                "**SKIP_IV_SURFACE**。无可用 IV 面。",
                f"unit tests = `{tests.get('status')}`。",
                f"meta={meta}",
                "",
            ]),
        )
        return payload
    daily.to_parquet(FACTOR_PARQUET, index=False)
    payload = {
        "status": "DONE",
        "skip": False,
        "meta": meta,
        "unit_tests": tests,
        "n_factor_days": int(len(daily)),
        "path": str(FACTOR_PARQUET),
        "percentile_window": PCT_WINDOW,
        "percentile_min": PCT_MIN_OBS,
    }
    dump_json(FACTOR_JSON, payload)
    write_md(FACTOR_MD, f"# Round 15 · 期权因子\n\nstatus=`DONE` n_days={len(daily)}\n")
    return payload


def load_factors() -> dict[str, Any]:
    return load_json(FACTOR_JSON)


def load_factor_daily() -> pd.DataFrame:
    if not FACTOR_PARQUET.is_file():
        return pd.DataFrame()
    return pd.read_parquet(FACTOR_PARQUET)


__all__ = [
    "bs_delta",
    "bs_price",
    "implied_vol",
    "load_factor_daily",
    "load_factors",
    "pc_parity_gap",
    "run_factors",
    "run_iv_unit_tests",
]
