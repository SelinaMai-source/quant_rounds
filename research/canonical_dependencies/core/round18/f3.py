"""F3 = V_seal / FFMV。FFMV = as-of-D free_float_shares * D 未复权收盘。禁止总市值顶替。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ..paths import ROUND18_ROOT, assert_project_result_path
from .common import dump_json, load_json
from .contract import F3_NAME, HERMES_V7

KEYS = ["trade_date", "instrument_id"]
F3_PARQUET = ROUND18_ROOT / "factors" / "f3.parquet"
F3_META = ROUND18_ROOT / "factors" / "f3_meta.json"


def build_f3(keys: pd.DataFrame, *, v_seal: pd.DataFrame | None = None, resume: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    if resume and F3_PARQUET.is_file() and F3_META.is_file():
        return pd.read_parquet(F3_PARQUET), load_json(F3_META)
    work = keys[KEYS].drop_duplicates().copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)

    val = ds.dataset(str(HERMES_V7 / "daily_valuation"), format="parquet").to_table(
        columns=["trade_date", "instrument_id", "free_float_shares", "free_float_market_cap", "market_cap", "available_at"]
    ).to_pandas()
    val["trade_date"] = pd.to_datetime(val["trade_date"])
    val["instrument_id"] = val["instrument_id"].astype(str)
    raw = ds.dataset(str(HERMES_V7 / "daily_raw"), format="parquet").to_table(
        columns=["trade_date", "instrument_id", "close"]
    ).to_pandas()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["instrument_id"] = raw["instrument_id"].astype(str)

    if v_seal is None:
        ev = ds.dataset(str(HERMES_V7 / "daily_limit_events"), format="parquet").to_table(
            columns=["trade_date", "instrument_id", "limit_type", "close_limit_value"]
        ).to_pandas()
        ev["trade_date"] = pd.to_datetime(ev["trade_date"])
        ev["instrument_id"] = ev["instrument_id"].astype(str)
        ev = ev.loc[ev["limit_type"].astype(str).eq("up"), KEYS + ["close_limit_value"]]
        v_seal = ev.rename(columns={"close_limit_value": "V_seal"})
    else:
        v_seal = v_seal[KEYS + ["V_seal"]].copy()
        v_seal["trade_date"] = pd.to_datetime(v_seal["trade_date"])
        v_seal["instrument_id"] = v_seal["instrument_id"].astype(str)

    merged = (
        work.merge(val, on=KEYS, how="left")
        .merge(raw, on=KEYS, how="left")
        .merge(v_seal, on=KEYS, how="left")
    )
    shares = pd.to_numeric(merged["free_float_shares"], errors="coerce")
    close = pd.to_numeric(merged["close"], errors="coerce")
    ffmv = shares * close
    vendor_ffmv = pd.to_numeric(merged["free_float_market_cap"], errors="coerce")
    v = pd.to_numeric(merged["V_seal"], errors="coerce")
    avail = pd.to_datetime(merged["available_at"], errors="coerce")
    same = avail.dt.normalize() == merged["trade_date"].dt.normalize()
    ok = v.notna() & ffmv.notna() & ffmv.gt(0) & np.isfinite(v) & np.isfinite(ffmv)
    merged[F3_NAME] = np.where(ok, v / ffmv, np.nan)
    merged["ffmv_asof_d"] = ffmv
    frame = merged[KEYS + [F3_NAME, "V_seal", "ffmv_asof_d"]].copy()
    meta = {
        "name": F3_NAME,
        "formula": "V_seal / (free_float_shares * unadj_close)",
        "status": "built",
        "n": int(len(frame)),
        "n_valid": int(frame[F3_NAME].notna().sum()),
        "filled_with_zero": False,
        "used_total_mv": False,
        "used_vendor_ffmv": False,
        "available_at_eq_trade_date": float(same.mean()) if len(merged) else None,
        "vendor_ffmv_corr_with_asof": (
            float(pd.Series(ffmv).corr(vendor_ffmv)) if pd.Series(ffmv).notna().any() else None
        ),
        "revision_risk": "available_at==trade_date，股本可能含事后修订；已用 as-of-D 已列示股本×当日未复权收盘，不用总市值。",
        "pit_note": "不能无条件当严格 PIT；正式晋升须披露修订风险。",
    }
    assert_project_result_path(F3_PARQUET)
    F3_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(F3_PARQUET, index=False)
    dump_json(F3_META, meta)
    return frame, meta
