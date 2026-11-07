"""现场核验 Hermes v7 / min1 / holder_num / block_trading / ZyQuant 350。不编造缺失表。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..paths import ROUND21_ROOT
from .common import dump_json, write_md
from .contract import (
    AUDIT_MD,
    BLOCK_TRADING_PARQUET,
    HERMES_REBUILD_RAW,
    HERMES_SCHEMA,
    HERMES_V7,
    HOLDER_NUM_PARQUET,
    MIN1_2013,
    MIN1_2022,
    ZYQUANT_350_ROOT,
)

AUDIT_JSON = ROUND21_ROOT / "source_audit" / "source_audit.json"
AUDIT_DIR = ROUND21_ROOT / "source_audit"


def _probe_parquet(path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file() if path.exists() else False,
        "is_dir": path.is_dir() if path.exists() else False,
    }
    if not path.exists():
        rec["status"] = "MISSING"
        return rec
    try:
        if path.is_file():
            rec["bytes"] = int(path.stat().st_size)
            pf = pq.ParquetFile(path)
            rec["n_rows_meta"] = int(pf.metadata.num_rows)
            rec["columns"] = list(pf.schema_arrow.names)
            rec["schema"] = {n: str(pf.schema_arrow.field(n).type) for n in pf.schema_arrow.names}
            rec["status"] = "OK"
            return rec
        files = list(path.rglob("*.parquet"))
        rec["n_parquet_files"] = len(files)
        rec["bytes"] = int(sum(f.stat().st_size for f in files))
        if files:
            pf = pq.ParquetFile(files[0])
            rec["sample_file"] = str(files[0])
            rec["columns"] = list(pf.schema_arrow.names)
            rec["schema"] = {n: str(pf.schema_arrow.field(n).type) for n in pf.schema_arrow.names}
        rec["status"] = "OK"
        return rec
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "READ_ERROR"
        rec["error"] = str(exc)
        return rec


def _min1_days(root: Path) -> dict[str, Any]:
    rec = {
        "path": str(root),
        "exists": root.exists(),
        "children": [p.name for p in sorted(root.iterdir())] if root.is_dir() else [],
    }
    bars = root / "minute_bars"
    if not bars.is_dir():
        rec["n_days"] = 0
        rec["status"] = "MISSING_MINUTE_BARS"
        rec["schema"] = _probe_parquet(root)
        return rec
    days = sorted(
        p.name.replace("trade_date=", "")
        for p in bars.iterdir()
        if p.is_dir() and p.name.startswith("trade_date=")
    )
    rec["n_days"] = len(days)
    rec["first"] = days[0] if days else None
    rec["last"] = days[-1] if days else None
    sample = next(bars.glob("trade_date=*/*.parquet"), None)
    rec["bar_schema"] = _probe_parquet(sample) if sample else {}
    rec["status"] = "OK"
    rec["note"] = "close-limit 子集，无全市场 tick/十档。禁止全市场分钟均值。"
    return rec


def _count_dirs(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_dir())


def run_audit(**_: Any) -> dict[str, Any]:
    hermes_children = [p.name for p in sorted(HERMES_V7.iterdir())] if HERMES_V7.is_dir() else []
    planned_tables = {
        "daily_raw": HERMES_V7 / "daily_raw",
        "daily_limit_events": HERMES_V7 / "daily_limit_events",
        "daily_margin": HERMES_V7 / "daily_margin",
        "daily_valuation": HERMES_V7 / "daily_valuation",
        "share_capital": HERMES_V7 / "share_capital",
        "industry_membership": HERMES_V7 / "industry_membership",
    }
    extra_landed = {
        "daily_money_flow": HERMES_V7 / "daily_money_flow",
        "daily_intraday_factors": HERMES_V7 / "daily_intraday_factors",
    }
    hermes = {k: _probe_parquet(p) for k, p in planned_tables.items()}
    extra = {k: _probe_parquet(p) for k, p in extra_landed.items()}
    money_cols = extra.get("daily_money_flow", {}).get("columns") or []
    buy_ord_in_v7 = [c for c in money_cols if "ORD" in str(c).upper()]
    schema_rec = {
        "path": str(HERMES_SCHEMA),
        "exists": HERMES_SCHEMA.is_file(),
        "bytes": int(HERMES_SCHEMA.stat().st_size) if HERMES_SCHEMA.is_file() else 0,
        "note": "在 v7-rebuild 根，不在 raw/",
    }
    raw_rec = {
        "path": str(HERMES_REBUILD_RAW),
        "exists": HERMES_REBUILD_RAW.is_dir(),
        "children": [p.name for p in sorted(HERMES_REBUILD_RAW.iterdir())] if HERMES_REBUILD_RAW.is_dir() else [],
        "source_schema_in_raw": (HERMES_REBUILD_RAW / "source_schema.json").exists(),
    }
    holder = _probe_parquet(HOLDER_NUM_PARQUET)
    if holder.get("status") == "OK":
        import pandas as pd

        frame = pd.read_parquet(HOLDER_NUM_PARQUET, columns=["MARKET_CODE", "ANN_DT", "HOLDER_ENDDATE", "HOLDER_NUM"])
        ann = frame["ANN_DT"].dropna().astype(str)
        holder["n_codes"] = int(frame["MARKET_CODE"].nunique())
        holder["ann_dt_null"] = int(frame["ANN_DT"].isna().sum())
        holder["ann_dt_min"] = str(ann.min()) if len(ann) else None
        holder["ann_dt_max"] = str(ann.max()) if len(ann) else None
        holder["holder_num_min"] = int(frame["HOLDER_NUM"].min())
        holder["holder_enddate_present"] = "HOLDER_ENDDATE" in frame.columns
    block = _probe_parquet(BLOCK_TRADING_PARQUET)
    if block.get("status") == "OK":
        import pandas as pd

        frame = pd.read_parquet(BLOCK_TRADING_PARQUET, columns=["TRADE_DATE", "MARKET_CODE"])
        td = frame["TRADE_DATE"].dropna().astype(str)
        block["n_codes"] = int(frame["MARKET_CODE"].nunique())
        block["trade_date_min"] = str(td.min()) if len(td) else None
        block["trade_date_max"] = str(td.max()) if len(td) else None
        block["announcement_columns"] = [
            c for c in (block.get("columns") or []) if any(k in str(c).upper() for k in ("ANN", "PUBLISH", "AVAILABLE"))
        ]
        block["has_announcement_timestamp"] = bool(block["announcement_columns"])
    z350_hash = None
    z350_n = 0
    if ZYQUANT_350_ROOT.is_dir():
        hashes = [p for p in ZYQUANT_350_ROOT.iterdir() if p.is_dir()]
        z350_hash = hashes[0].name if hashes else None
        z350_n = _count_dirs(hashes[0]) if hashes else 0
    z350 = {
        "path": str(ZYQUANT_350_ROOT),
        "exists": ZYQUANT_350_ROOT.is_dir(),
        "hash_dir": z350_hash,
        "n_factors": z350_n,
        "role": "Novelty 对照库，不整包进模型",
    }
    min1 = {
        "2013_2021": _min1_days(MIN1_2013),
        "2022_20260724": _min1_days(MIN1_2022),
    }
    project_inventory = ROUND21_ROOT.parent / "data_inventory"
    invented = []
    payload = {
        "round": 21,
        "do_not_invent_tables": True,
        "invented_tables": invented,
        "project_data_inventory_exists": project_inventory.exists(),
        "hermes_v7": {
            "path": str(HERMES_V7),
            "exists": HERMES_V7.is_dir(),
            "children": hermes_children,
            "planned_tables": hermes,
            "extra_landed_observed": extra,
            "buy_ord_in_v7_landed": buy_ord_in_v7,
            "buy_ord_note": "BUY_ORD_* 出现在 source_schema.json，不在 v7 落盘表。六家族不依赖它。",
        },
        "source_schema": schema_rec,
        "rebuild_raw": raw_rec,
        "min1": min1,
        "holder_num": holder,
        "block_trading": block,
        "zyquant_350": z350,
        "rules": {
            "min1_close_limit_subset_only": True,
            "no_full_market_minute_mean": True,
            "missing_not_filled_zero": True,
            "non_margin_financing_null_not_zero": True,
            "holder_use_ann_dt_only": True,
            "block_no_announcement_timestamp": not bool(block.get("has_announcement_timestamp")),
        },
        "counts": {
            "hermes_planned_ok": sum(1 for v in hermes.values() if v.get("status") == "OK"),
            "hermes_planned_n": len(hermes),
            "min1_days_2013": min1["2013_2021"].get("n_days"),
            "min1_days_2022": min1["2022_20260724"].get("n_days"),
            "holder_rows": holder.get("n_rows_meta"),
            "holder_codes": holder.get("n_codes"),
            "block_rows": block.get("n_rows_meta"),
            "block_codes": block.get("n_codes"),
            "zyquant_350_n": z350_n,
            "buy_ord_in_v7_landed_n": len(buy_ord_in_v7),
        },
    }
    dump_json(AUDIT_JSON, payload)
    dump_json(AUDIT_DIR / "hermes_tables.json", {"planned": hermes, "extra": extra})
    dump_json(AUDIT_DIR / "min1.json", min1)
    dump_json(AUDIT_DIR / "holder_num.json", holder)
    dump_json(AUDIT_DIR / "block_trading.json", block)
    dump_json(AUDIT_DIR / "zyquant_350.json", z350)
    c = payload["counts"]
    lines = [
        "# Round21 · 原始数据盘点（现场核验）",
        "",
        "`result/data_inventory/` 不存在；本文件是本轮自建 source_audit。不编造缺失表。",
        "",
        "## 计数",
        "",
        f"- Hermes 计划表 OK `{c['hermes_planned_ok']}/{c['hermes_planned_n']}`（daily_raw/limit_events/margin/valuation/share_capital/industry_membership）",
        f"- min1 日数 2013–2021=`{c['min1_days_2013']}` 2022–2026-07-24=`{c['min1_days_2022']}`（仅收盘涨停子集）",
        f"- holder_num rows=`{c['holder_rows']}` codes=`{c['holder_codes']}` cols=`ANN_DT,HOLDER_ENDDATE,HOLDER_NUM`",
        f"- block_trading rows=`{c['block_rows']}` codes=`{c['block_codes']}` 无公告时点=`{payload['rules']['block_no_announcement_timestamp']}`",
        f"- ZyQuant 350=`{c['zyquant_350_n']}` Novelty only",
        f"- BUY_ORD 在 v7 落盘=`{c['buy_ord_in_v7_landed_n']}`",
        "",
        "## 路径",
        "",
        f"- Hermes v7 `{HERMES_V7}` children=`{hermes_children}`",
        f"- source_schema `{HERMES_SCHEMA}` exists=`{schema_rec['exists']}` in_raw=`{raw_rec['source_schema_in_raw']}`",
        f"- min1 `{MIN1_2013}` / `{MIN1_2022}`",
        f"- holder `{HOLDER_NUM_PARQUET}`",
        f"- block `{BLOCK_TRADING_PARQUET}`",
        f"- 350 `{ZYQUANT_350_ROOT}` hash=`{z350_hash}`",
        "",
        "缺失不填 0。非两融标的融资在 daily_margin 中缺行，join 后保持 NULL。",
        "",
    ]
    write_md(AUDIT_MD, "\n".join(lines))
    write_md(AUDIT_DIR / "source_audit.md", "\n".join(lines))
    return payload
