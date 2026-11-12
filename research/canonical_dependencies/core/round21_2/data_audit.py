"""原始数据现场核验。缺表记 BLOCKED_DATA，不编造。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ..paths import ROUND21_2_ROOT
from ..round21.contract import HERMES_REBUILD_RAW, HERMES_V7, MIN1_2013, MIN1_2022
from .common import dump_json, write_md
from .contract import BLOCKED_DATA, MD

AUDIT_JSON = ROUND21_2_ROOT / "source_audit" / "source_audit.json"


def _schema(root, n_files: int = 1) -> dict[str, Any]:
    from pathlib import Path

    p = Path(root)
    if not p.exists():
        return {"exists": False, "status": BLOCKED_DATA, "columns": []}
    files = list(p.rglob("*.parquet"))
    if not files:
        return {"exists": True, "n_files": 0, "status": BLOCKED_DATA, "columns": []}
    try:
        cols = list(pq.read_schema(files[0]).names)
    except Exception as exc:
        return {"exists": True, "n_files": len(files), "status": BLOCKED_DATA, "error": str(exc), "columns": []}
    return {"exists": True, "n_files": len(files), "columns": cols, "status": "OK"}


def _mf_lag() -> dict[str, Any]:
    rows = []
    for y in (2016, 2018, 2020, 2021, 2022, 2023, 2024, 2025, 2026):
        d = HERMES_V7 / "daily_money_flow" / f"year={y}"
        files = list(d.rglob("*.parquet"))
        if not files:
            rows.append({"year": y, "status": "NO_FILES"})
            continue
        mf = pd.read_parquet(files[0], columns=["trade_date", "available_at"])
        mf["trade_date"] = pd.to_datetime(mf["trade_date"])
        mf["available_at"] = pd.to_datetime(mf["available_at"])
        lag = (mf["available_at"] - mf["trade_date"]).dt.days
        rows.append(
            {
                "year": y,
                "n": int(len(mf)),
                "median_lag_days": float(lag.median()) if lag.notna().any() else None,
                "p90_lag_days": float(lag.quantile(0.9)) if lag.notna().any() else None,
                "strict_same_day_share": float((lag.fillna(99) <= 0).mean()),
                "backfill_warning": bool(lag.median() is not None and float(lag.median()) > 7),
            }
        )
    return {"rows": rows, "strict_pit_from": 2021, "note": "2016–2020 money_flow available_at 为整表回填，STRICT_PIT 禁用这些年同日值"}


def run_data_audit(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and AUDIT_JSON.is_file() and MD["02"].is_file():
        from .common import load_json

        existing = load_json(AUDIT_JSON)
        if existing.get("status") == "DONE":
            return existing
    hermes = {}
    for name in (
        "daily_raw",
        "daily_money_flow",
        "daily_limit_events",
        "daily_valuation",
        "daily_intraday_factors",
        "share_capital",
        "corporate_actions",
        "financial_reports",
        "fundamental_metrics",
        "financial_facts",
        "industry_membership",
        "trade_calendar",
    ):
        hermes[name] = _schema(HERMES_V7 / name)
    raw = {}
    for name in (
        "mkt_equd",
        "mkt_equ_mf_new",
        "equ_shares_change",
        "equ_free_shares",
        "equ_div_pit",
        "fdmt_main_data_q_pit",
        "fdmt_md_n_ttmp",
        "mkt_limit",
    ):
        raw[name] = _schema(HERMES_REBUILD_RAW / name)
    min1 = {
        "2013": {"exists": MIN1_2013.exists(), "path": str(MIN1_2013)},
        "2022": {"exists": MIN1_2022.exists(), "path": str(MIN1_2022)},
    }
    mf = _mf_lag()
    buy_ord_in_v7 = "BUY_ORD_S" in (hermes.get("daily_money_flow") or {}).get("columns", [])
    payload = {
        "status": "DONE",
        "filled_with_zero": False,
        "hermes_v7": hermes,
        "hermes_raw": raw,
        "min1": min1,
        "money_flow_lag": mf,
        "buy_ord_in_v7_money_flow": buy_ord_in_v7,
        "g_order_fields_source": "RAW mkt_equ_mf_new (BUY_ORD_*/SELL_ORD_*/BUY_VOL_*/SELL_VOL_*)",
        "g_deal_amount_source": "RAW mkt_equd.DEAL_AMOUNT",
        "blocked_tables": [k for k, v in {**hermes, **raw}.items() if v.get("status") == BLOCKED_DATA],
    }
    dump_json(AUDIT_JSON, payload)
    lines = [
        "# Round21.2 · 02 新原始数据盘点",
        "",
        "现场核验。缺表 = BLOCKED_DATA，不编造。",
        "",
        f"Hermes v7 表：`{ {k: v.get('status') for k, v in hermes.items()} }`",
        f"Raw rebuild：`{ {k: v.get('status') for k, v in raw.items()} }`",
        f"BUY_ORD_* 在 v7 daily_money_flow=`{buy_ord_in_v7}`。G 族订单笔数走 raw `mkt_equ_mf_new`。",
        f"资金流 available_at：2021 前整表回填，STRICT_PIT 自 `{mf.get('strict_pit_from')}`。",
        f"min1 2013 exists=`{min1['2013']['exists']}` 2022 exists=`{min1['2022']['exists']}`。仅收盘涨停子集。",
        "ZyQuant 350 只作 Novelty 对照。",
        "",
    ]
    write_md(MD["02"], "\n".join(lines))
    return payload
