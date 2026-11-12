"""F2 = log(AvgBuySize_LXL / AvgSellSize_LXL)。单位不一致则 SKIP_SCHEMA。禁止分母补 1。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND18_ROOT, assert_project_result_path
from .common import dump_json, load_json
from .contract import F2_NAME, LAGGED_DIAGNOSTIC, MF_RAW, PIT_TIMESTAMP_UNVERIFIED
from .inventory import INV_JSON

KEYS = ["trade_date", "instrument_id"]
F2_PARQUET = ROUND18_ROOT / "factors" / "f2.parquet"
F2_META = ROUND18_ROOT / "factors" / "f2_meta.json"


def _to_inst(ticker: pd.Series, exchange: pd.Series) -> pd.Series:
    t = ticker.astype(str).str.zfill(6)
    e = exchange.astype(str)
    return t + "." + e


def build_f2(keys: pd.DataFrame, *, resume: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    if resume and F2_PARQUET.is_file() and F2_META.is_file():
        return pd.read_parquet(F2_PARQUET), load_json(F2_META)
    inv = load_json(INV_JSON)
    units = ((inv.get("f2_raw_mf") or {}).get("units") or {})
    consistent = bool(units.get("schema_units_consistent"))
    if not MF_RAW.is_dir():
        meta = {"name": F2_NAME, "status": "SKIP_DATA", "reason": "raw mkt_equ_mf_new missing", "n_valid": 0}
        dump_json(F2_META, meta)
        empty = keys[KEYS].drop_duplicates().assign(**{F2_NAME: np.nan})
        return empty, meta
    if not consistent:
        meta = {
            "name": F2_NAME,
            "status": "SKIP_SCHEMA",
            "reason": "VOL/ORD/金额口径不一致，禁止 VOL÷不对应委托笔数",
            "n_valid": 0,
            "units": units,
        }
        dump_json(F2_META, meta)
        empty = keys[KEYS].drop_duplicates().assign(**{F2_NAME: np.nan})
        return empty, meta

    cols = [
        "TICKER_SYMBOL", "EXCHANGE_CD", "TRADE_DATE",
        "BUY_VOL_L", "BUY_VOL_XL", "SELL_VOL_L", "SELL_VOL_XL",
        "BUY_ORD_L", "BUY_ORD_XL", "SELL_ORD_L", "SELL_ORD_XL",
        "UPDATE_TIME",
    ]
    want = keys[KEYS].drop_duplicates().copy()
    want["trade_date"] = pd.to_datetime(want["trade_date"])
    want["instrument_id"] = want["instrument_id"].astype(str)
    inst_set = set(want["instrument_id"])
    frames = []
    for path in sorted(MF_RAW.rglob("*.parquet")):
        part = pd.read_parquet(path, columns=cols)
        part["instrument_id"] = _to_inst(part["TICKER_SYMBOL"], part["EXCHANGE_CD"])
        part = part.loc[part["instrument_id"].isin(inst_set)]
        if part.empty:
            continue
        frames.append(part)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols + ["instrument_id"])
    raw["trade_date"] = pd.to_datetime(raw["TRADE_DATE"])
    raw["update_time"] = pd.to_datetime(raw.get("UPDATE_TIME"), errors="coerce")
    merged = want.merge(raw, on=KEYS, how="left")
    buy_vol = pd.to_numeric(merged["BUY_VOL_L"], errors="coerce") + pd.to_numeric(merged["BUY_VOL_XL"], errors="coerce")
    sell_vol = pd.to_numeric(merged["SELL_VOL_L"], errors="coerce") + pd.to_numeric(merged["SELL_VOL_XL"], errors="coerce")
    buy_ord = pd.to_numeric(merged["BUY_ORD_L"], errors="coerce") + pd.to_numeric(merged["BUY_ORD_XL"], errors="coerce")
    sell_ord = pd.to_numeric(merged["SELL_ORD_L"], errors="coerce") + pd.to_numeric(merged["SELL_ORD_XL"], errors="coerce")
    buy_ok = buy_ord.gt(0) & buy_vol.notna() & np.isfinite(buy_vol)
    sell_ok = sell_ord.gt(0) & sell_vol.notna() & np.isfinite(sell_vol)
    avg_buy = np.where(buy_ok, buy_vol / buy_ord, np.nan)
    avg_sell = np.where(sell_ok, sell_vol / sell_ord, np.nan)
    ratio_ok = (
        np.isfinite(avg_buy) & np.isfinite(avg_sell)
        & (np.asarray(avg_buy) > 0) & (np.asarray(avg_sell) > 0)
    )
    merged[F2_NAME] = np.where(ratio_ok, np.log(avg_buy / avg_sell), np.nan)
    same_day = merged["update_time"].dt.normalize() == merged["trade_date"].dt.normalize()
    merged["f2_update_eq_trade"] = same_day
    year = merged["trade_date"].dt.year
    pit_by_year = {}
    for y, block in merged.groupby(year):
        pit_by_year[str(int(y))] = {
            "n": int(len(block)),
            "update_eq_trade": float(block["f2_update_eq_trade"].mean()) if len(block) else None,
            "n_valid": int(block[F2_NAME].notna().sum()),
        }
    # 2021 前 UPDATE_TIME 几乎全部 ≠ trade_date → 不得把当日资金流当已验证 PIT
    pre2021 = year.lt(2021)
    verified_share = float(same_day.loc[merged[F2_NAME].notna()].mean()) if merged[F2_NAME].notna().any() else 0.0
    pit_status = PIT_TIMESTAMP_UNVERIFIED if verified_share < 0.95 else "PIT_OK"
    frame = merged[KEYS + [F2_NAME, "f2_update_eq_trade"]].copy()
    meta = {
        "name": F2_NAME,
        "formula": "log((BUY_VOL_L+XL)/(BUY_ORD_L+XL) / (SELL_VOL_L+XL)/(SELL_ORD_L+XL))",
        "status": "built",
        "schema_units_consistent": True,
        "vol_unit": "shares",
        "ord_unit": "order_count",
        "inflow_unit": "yuan",
        "n": int(len(frame)),
        "n_valid": int(frame[F2_NAME].notna().sum()),
        "filled_with_zero": False,
        "no_denominator_plus_one": True,
        "pit_status": pit_status,
        "pit_verified_share_on_valid": verified_share,
        "pre2021_update_eq_trade": float(same_day.loc[pre2021].mean()) if pre2021.any() else None,
        "pit_by_year": pit_by_year,
        "note": "2021前 UPDATE_TIME 回填；正式晋升不得把 F2 当已验证 PIT。不得静默换成滞后公式。",
        "lagged_diagnostic_allowed": LAGGED_DIAGNOSTIC,
    }
    assert_project_result_path(F2_PARQUET)
    F2_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(F2_PARQUET, index=False)
    dump_json(F2_META, meta)
    return frame, meta
