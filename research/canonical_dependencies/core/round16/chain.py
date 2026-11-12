"""PIT 期权链：仅当日真实存在的合约。Asia/Shanghai。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_ROOT
from .common import dump_json, load_json, to_ymd, write_md
from .coverage import underlying_from_exchange
from .download import load_all_basic, load_all_kline, load_etf

CHAIN_PARQUET = ROUND16_ROOT / "data" / "processed" / "option_chain.parquet"
CHAIN_JSON = ROUND16_ROOT / "chain" / "chain.json"
CHAIN_MD = ROUND16_ROOT / "02_contract_chain_reconstruction.md"


def _ymd_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8), format="%Y%m%d", errors="coerce")


def run_chain(**_: Any) -> dict[str, Any]:
    kline = load_all_kline()
    basic = load_all_basic()
    etf = load_etf()
    if kline.empty or basic.empty:
        payload = {"status": "EMPTY", "n": 0, "reason": "NO_KLINE_OR_BASIC"}
        dump_json(CHAIN_JSON, payload)
        write_md(CHAIN_MD, "# Round16 · 02 合约链\n\nEMPTY。\n")
        return payload
    k = kline.copy()
    k["contract_code"] = k["code"].astype(str) if "code" in k.columns else k["_key"].astype(str)
    k["trade_date"] = pd.to_datetime(k["kline_time"], errors="coerce").dt.normalize()
    k["last"] = pd.to_numeric(k.get("close"), errors="coerce")
    k["volume"] = pd.to_numeric(k.get("volume"), errors="coerce")
    k["open"] = pd.to_numeric(k.get("open"), errors="coerce")
    k["high"] = pd.to_numeric(k.get("high"), errors="coerce")
    k["low"] = pd.to_numeric(k.get("low"), errors="coerce")
    k = k.loc[k["trade_date"].notna()]
    b = basic.copy()
    b["contract_code"] = b["MARKET_CODE"].astype(str)
    b["call_put"] = b["CONTRACT_TYPE"].astype(str).str.upper().str[0]
    b["strike"] = pd.to_numeric(b["EXERCISE_PRICE"], errors="coerce")
    b["expiry_date"] = _ymd_series(b["EXPIRY_DATE"])
    b["start_trade"] = _ymd_series(b["START_TRADE_DATE"])
    b["last_trade"] = _ymd_series(b["LAST_TRADE_DATE"] if "LAST_TRADE_DATE" in b.columns else b["EXPIRY_DATE"])
    b["contract_multiplier"] = pd.to_numeric(b.get("CONTRACT_UNIT"), errors="coerce")
    b["adjustment_flag"] = b.get("CONTRACT_ADJUST_FLAG")
    b["underlying_code"] = b["EXCHANGE_CODE"].map(underlying_from_exchange)
    keep = [
        "contract_code", "call_put", "strike", "expiry_date", "start_trade", "last_trade",
        "contract_multiplier", "adjustment_flag", "underlying_code",
    ]
    b = b[keep].drop_duplicates("contract_code")
    m = k.merge(b, on="contract_code", how="left")
    # 仅当日真实存在：start <= D <= last_trade，且未到期
    alive = (
        m["start_trade"].notna()
        & m["last_trade"].notna()
        & (m["trade_date"] >= m["start_trade"])
        & (m["trade_date"] <= m["last_trade"])
        & (m["expiry_date"].isna() | (m["trade_date"] < m["expiry_date"]))
    )
    m = m.loc[alive].copy()
    m["days_to_expiry"] = (m["expiry_date"] - m["trade_date"]).dt.days
    m["quote_timestamp"] = m["trade_date"] + pd.Timedelta(hours=15)
    m["available_at"] = m["quote_timestamp"]
    m["source"] = "AmazingData.query_kline.day"
    m["bid"] = np.nan
    m["ask"] = np.nan
    m["mid"] = m["last"]
    m["open_interest"] = np.nan
    etf_px = pd.DataFrame()
    if not etf.empty:
        e = etf.copy()
        e["underlying_code"] = e["code"].astype(str) if "code" in e.columns else e.get("_key").astype(str)
        e["trade_date"] = pd.to_datetime(e["kline_time"], errors="coerce").dt.normalize()
        e["underlying_price"] = pd.to_numeric(e["close"], errors="coerce")
        etf_px = e[["underlying_code", "trade_date", "underlying_price"]].drop_duplicates(["underlying_code", "trade_date"])
        m = m.merge(etf_px, on=["underlying_code", "trade_date"], how="left")
    else:
        m["underlying_price"] = np.nan
    m["valid_quote"] = (
        m["last"].gt(0)
        & m["strike"].gt(0)
        & m["days_to_expiry"].gt(0)
        & m["underlying_price"].gt(0)
        & np.isfinite(m["last"])
    )
    # 去重：同日同合约保留最后一条
    m = m.sort_values(["trade_date", "contract_code"]).drop_duplicates(["trade_date", "contract_code"], keep="last")
    cols = [
        "trade_date", "quote_timestamp", "available_at", "contract_code", "underlying_code",
        "call_put", "strike", "expiry_date", "days_to_expiry", "contract_multiplier",
        "adjustment_flag", "underlying_price", "bid", "ask", "mid", "last", "volume",
        "open_interest", "source", "valid_quote",
    ]
    out = m[cols].copy()
    out["tz"] = "Asia/Shanghai"
    CHAIN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CHAIN_PARQUET, index=False)
    payload = {
        "status": "DONE",
        "n": int(len(out)),
        "n_valid": int(out["valid_quote"].sum()),
        "n_days": int(out["trade_date"].nunique()),
        "n_contracts": int(out["contract_code"].nunique()),
        "n_underlyings": int(out["underlying_code"].nunique(dropna=True)),
        "min": str(out["trade_date"].min().date()) if len(out) else None,
        "max": str(out["trade_date"].max().date()) if len(out) else None,
        "path": str(CHAIN_PARQUET),
        "no_future_contract_join": True,
        "bid_ask_present": False,
        "quote_field": "last/mid_from_close",
        "pit_tz": "Asia/Shanghai",
    }
    dump_json(CHAIN_JSON, payload)
    write_md(
        CHAIN_MD,
        "\n".join([
            "# Round 16 · 02 合约链重建",
            "",
            f"n=`{payload['n']}` valid=`{payload['n_valid']}` days=`{payload['n_days']}` contracts=`{payload['n_contracts']}`。",
            f"区间 `{payload['min']}` → `{payload['max']}`。仅 start_trade≤D≤last_trade 且未到期。",
            "日 K 无买卖报价，mid=close，质量见 03。不用未来合约属性构造过去曲面。",
            f"path=`{CHAIN_PARQUET}`。",
            "",
        ]),
    )
    return payload


def load_chain() -> pd.DataFrame:
    if not CHAIN_PARQUET.is_file():
        return pd.DataFrame()
    return pd.read_parquet(CHAIN_PARQUET)


def load_chain_status() -> dict[str, Any]:
    return load_json(CHAIN_JSON)
