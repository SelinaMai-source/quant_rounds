"""现场盘点分钟 / raw 资金流 / 估值股本。不编造，不填 0。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND18_ROOT
from ..round7.contract import FIVE_HEAD, TEN_HEAD
from .common import dump_json, file_sha256, write_md
from .contract import HERMES_SCHEMA, HERMES_V7, MF_RAW, MIN1_2013, MIN1_2022

INV_JSON = ROUND18_ROOT / "inventory" / "inventory.json"
INV_MD = ROUND18_ROOT / "01_data_inventory.md"
SCHEMA_JSON = ROUND18_ROOT / "inventory" / "f2_schema.json"


def _min_days(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    bars = root / "minute_bars"
    if not bars.is_dir():
        return []
    return sorted(p.name.replace("trade_date=", "") for p in bars.glob("trade_date=*") if p.is_dir())


def _mf_schema_rows() -> list[dict[str, Any]]:
    if not HERMES_SCHEMA.is_file():
        return []
    rows = json.loads(HERMES_SCHEMA.read_text(encoding="utf-8"))
    return [r for r in rows if r.get("TABLE_NAME") == "mkt_equ_mf_new"]


def _probe_mf_units() -> dict[str, Any]:
    samples = {}
    years = ("2014", "2018", "2021", "2022", "2025")
    unit_ok = True
    notes = []
    for year in years:
        ydir = MF_RAW / f"year={year}"
        files = sorted(ydir.rglob("*.parquet")) if ydir.is_dir() else []
        if not files:
            samples[year] = {"status": "MISSING"}
            continue
        df = pd.read_parquet(files[0])
        need = [
            "TURNOVER_VOL", "TURNOVER_VALUE", "BUY_VOL_L", "BUY_VOL_XL", "SELL_VOL_L",
            "SELL_VOL_XL", "BUY_ORD_L", "BUY_ORD_XL", "SELL_ORD_L", "SELL_ORD_XL",
            "INFLOW_L", "INFLOW_XL", "UPDATE_TIME", "TRADE_DATE",
        ]
        miss = [c for c in need if c not in df.columns]
        if miss:
            samples[year] = {"status": "SKIP_SCHEMA", "missing": miss}
            unit_ok = False
            continue
        tot_vol = (
            df[["BUY_VOL_S", "BUY_VOL_M", "BUY_VOL_L", "BUY_VOL_XL",
                "SELL_VOL_S", "SELL_VOL_M", "SELL_VOL_L", "SELL_VOL_XL"]]
            .sum(axis=1, min_count=1)
            if set(["BUY_VOL_S", "SELL_VOL_S"]).issubset(df.columns)
            else np.nan
        )
        vol_ratio = pd.to_numeric(tot_vol, errors="coerce") / pd.to_numeric(df["TURNOVER_VOL"], errors="coerce").replace(0, np.nan)
        px = pd.to_numeric(df["INFLOW_L"], errors="coerce") / pd.to_numeric(df["BUY_VOL_L"], errors="coerce").replace(0, np.nan)
        upd = pd.to_datetime(df["UPDATE_TIME"], errors="coerce")
        td = pd.to_datetime(df["TRADE_DATE"], errors="coerce")
        same_day = float((upd.dt.normalize() == td.dt.normalize()).mean()) if len(df) else None
        rec = {
            "status": "ok",
            "n": int(len(df)),
            "vol_sum_to_turnover_median": float(vol_ratio.median()) if hasattr(vol_ratio, "median") else None,
            "inflowL_over_buyVolL_median": float(px.median()) if px.notna().any() else None,
            "buy_ord_l_median": float(pd.to_numeric(df["BUY_ORD_L"], errors="coerce").median()),
            "update_eq_trade_date": same_day,
        }
        samples[year] = rec
        if rec["vol_sum_to_turnover_median"] is None or abs(rec["vol_sum_to_turnover_median"] - 1.0) > 0.05:
            unit_ok = False
            notes.append(f"{year}: VOL vs TURNOVER_VOL not ~1")
        if rec["inflowL_over_buyVolL_median"] is None or not (1.0 <= rec["inflowL_over_buyVolL_median"] <= 200):
            unit_ok = False
            notes.append(f"{year}: INFLOW/VOL not price-like")
        if rec["buy_ord_l_median"] is not None and rec["buy_ord_l_median"] <= 0:
            unit_ok = False
            notes.append(f"{year}: BUY_ORD_L non-positive")
    return {"samples": samples, "schema_units_consistent": unit_ok, "notes": notes}


def run_inventory(**_: Any) -> dict[str, Any]:
    five_path = FIVE_HEAD.panel_dir / "candidate_panel.parquet"
    ten_path = TEN_HEAD.panel_dir / "candidate_panel.parquet"
    five = pd.read_parquet(five_path, columns=["trade_date", "instrument_id"])
    five["trade_date"] = pd.to_datetime(five["trade_date"])
    days_2013 = _min_days(MIN1_2013)
    days_2022 = _min_days(MIN1_2022)
    mf_schema = _mf_schema_rows()
    mf_probe = _probe_mf_units()
    val_dir = HERMES_V7 / "daily_valuation"
    lim_dir = HERMES_V7 / "daily_limit_events"
    raw_dir = HERMES_V7 / "daily_raw"
    mf_v7 = HERMES_V7 / "daily_money_flow"
    val_cols = []
    if val_dir.is_dir():
        parts = list(val_dir.rglob("*.parquet"))
        if parts:
            val_cols = list(pd.read_parquet(parts[0]).columns)
    schema_cols = {r["COLUMN_NAME"]: r.get("COLUMN_COMMENT") for r in mf_schema}
    dump_json(SCHEMA_JSON, {"columns": mf_schema, "probe": mf_probe})
    payload = {
        "status": "done",
        "five_panel": str(five_path),
        "ten_panel": str(ten_path),
        "five_rows": int(len(five)),
        "five_days": int(five["trade_date"].nunique()),
        "five_min": str(five["trade_date"].min().date()) if len(five) else None,
        "five_max": str(five["trade_date"].max().date()) if len(five) else None,
        "f1_minutes": {
            "path_2013": str(MIN1_2013),
            "path_2022": str(MIN1_2022),
            "exists_2013": MIN1_2013.is_dir(),
            "exists_2022": MIN1_2022.is_dir(),
            "n_days_2013": len(days_2013),
            "n_days_2022": len(days_2022),
            "first": days_2013[0] if days_2013 else None,
            "last": days_2022[-1] if days_2022 else (days_2013[-1] if days_2013 else None),
            "subset": "close-limit only",
            "readable": bool(MIN1_2013.is_dir() and MIN1_2022.is_dir()),
        },
        "f2_raw_mf": {
            "path": str(MF_RAW),
            "exists": MF_RAW.is_dir(),
            "n_parts": len(list(MF_RAW.rglob("*.parquet"))) if MF_RAW.is_dir() else 0,
            "schema_path": str(HERMES_SCHEMA),
            "has_buy_ord": "BUY_ORD_L" in schema_cols,
            "has_buy_vol": "BUY_VOL_L" in schema_cols,
            "comments": {k: schema_cols.get(k) for k in (
                "BUY_VOL_L", "BUY_VOL_XL", "SELL_VOL_L", "SELL_VOL_XL",
                "BUY_ORD_L", "BUY_ORD_XL", "SELL_ORD_L", "SELL_ORD_XL",
            )},
            "units": mf_probe,
        },
        "f3_valuation": {
            "path": str(val_dir),
            "exists": val_dir.is_dir(),
            "has_free_float_shares": "free_float_shares" in val_cols,
            "has_free_float_market_cap": "free_float_market_cap" in val_cols,
            "columns": val_cols,
            "available_at_note": "v7 available_at==trade_date; prefer shares * unadj close as-of-D",
        },
        "limit_events": {"path": str(lim_dir), "exists": lim_dir.is_dir()},
        "daily_raw": {"path": str(raw_dir), "exists": raw_dir.is_dir()},
        "daily_money_flow_v7": {"path": str(mf_v7), "exists": mf_v7.is_dir(), "dropped_ord_vol": True},
        "permissions": {
            "minutes": "ok" if MIN1_2013.is_dir() else "PERMISSION_REQUIRED",
            "mf_raw": "ok" if MF_RAW.is_dir() else "PERMISSION_REQUIRED",
            "valuation": "ok" if val_dir.is_dir() else "PERMISSION_REQUIRED",
        },
        "schema_sha256": file_sha256(HERMES_SCHEMA),
    }
    dump_json(INV_JSON, payload)
    f1 = payload["f1_minutes"]
    f2 = payload["f2_raw_mf"]
    lines = [
        "# 01 数据盘点",
        "",
        f"- 候选 five 行=`{payload['five_rows']}` 日=`{payload['five_days']}` `{payload['five_min']}`..`{payload['five_max']}`",
        f"- F1 分钟：2013 段 {f1['n_days_2013']} 日 / 2022 段 {f1['n_days_2022']} 日，仅收盘涨停子集。",
        f"- F2 raw `mkt_equ_mf_new` parts=`{f2['n_parts']}` schema_units_consistent=`{mf_probe.get('schema_units_consistent')}`",
        f"- F2 BUY_VOL 注释=`{schema_cols.get('BUY_VOL_L')}` BUY_ORD 注释=`{schema_cols.get('BUY_ORD_L')}`",
        f"- F3 daily_valuation free_float_shares=`{'free_float_shares' in val_cols}`",
        "",
        "缺失不填 0。单位不一致则 F2 不得进正式回测。",
    ]
    write_md(INV_MD, "\n".join(lines))
    return payload
