"""数据审计与授权内恢复。新文件只写 result/round22/data_recovery。不覆盖 Hermes v7。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..round18.contract import MIN1_2013, MIN1_2022
from ..round21_4.forward import _max_date
from .common import dump_json, load_json, write_csv, write_md
from .contract import (
    BLOCKED_ACCESS,
    BLOCKED_DATA,
    BLOCKED_PIT,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PORT,
    DATA_LIVE_RELEASES,
    FORWARD_OOS_NOT_AVAILABLE,
    HERMES_REBUILD_RAW,
    HERMES_V7,
    MARKET_CUTOFF,
    MD,
)
from ..paths import ROUND22_ROOT

REC = ROUND22_ROOT / "data_recovery"
REC_JSON = REC / "recovery.json"


def _listdir(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {"path": str(path), "exists": False, "status": "MISSING", "entries": []}
        entries = sorted(p.name for p in path.iterdir())
        return {"path": str(path), "exists": True, "status": "OK", "n": len(entries), "entries": entries}
    except PermissionError as exc:
        return {"path": str(path), "exists": True, "status": BLOCKED_ACCESS, "error": type(exc).__name__}
    except OSError as exc:
        return {"path": str(path), "exists": True, "status": "ERROR", "error": type(exc).__name__}


def _schema(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        info["status"] = "MISSING"
        return info
    try:
        if path.is_file():
            info["columns"] = list(pq.read_schema(path).names)
            info["status"] = "OK"
            return info
        found = None
        for p in path.rglob("*.parquet"):
            found = p
            break
        if found is None:
            info["status"] = "NO_PARQUET"
            return info
        info["sample"] = str(found)
        info["columns"] = list(pq.read_schema(found).names)
        info["status"] = "OK"
        return info
    except PermissionError:
        info["status"] = BLOCKED_ACCESS
        return info
    except Exception as exc:
        info["status"] = "ERROR"
        info["error"] = type(exc).__name__
        return info


def _clickhouse() -> dict[str, Any]:
    user = os.environ.get("CLICKHOUSE_USER") or os.environ.get("CH_USER")
    password = os.environ.get("CLICKHOUSE_PASSWORD") or os.environ.get("CH_PASSWORD")
    out: dict[str, Any] = {
        "host": CLICKHOUSE_HOST,
        "port": CLICKHOUSE_PORT,
        "credentials_present": bool(user and password),
        "status": BLOCKED_ACCESS,
    }
    if not user or not password:
        out["reason"] = "环境里没有现成的 ClickHouse 用户名和密码。不猜测、不绕过。"
        return out
    try:
        import urllib.parse
        import urllib.request

        query = urllib.parse.quote("SHOW TABLES")
        url = f"http://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/?query={query}"
        req = urllib.request.Request(url)
        token = f"{user}:{password}".encode()
        import base64

        req.add_header("Authorization", "Basic " + base64.b64encode(token).decode())
        with urllib.request.urlopen(req, timeout=8) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        names = [line.strip() for line in text.splitlines() if line.strip()]
        interesting = [n for n in names if any(k in n.lower() for k in ("lhb", "dragon", "billboard", "concept", "theme", "halt", "buyback", "holder", "reduction"))]
        out["status"] = "OK"
        out["n_tables"] = len(names)
        out["interesting"] = interesting[:40]
        out.pop("reason", None)
    except Exception as exc:
        out["status"] = BLOCKED_ACCESS
        out["reason"] = f"连接 127.0.0.1:18123 失败：{type(exc).__name__}"
    return out


def _classify_sources(hermes_tables: list[str], raw_tables: list[str], schemas: dict[str, Any]) -> dict[str, Any]:
    names = {n.lower(): n for n in hermes_tables + raw_tables}

    def has(*keys: str) -> str | None:
        for key in keys:
            for low, orig in names.items():
                if key in low:
                    return orig
        return None

    theme = has("concept", "theme", "block_member", "ths")
    theme_cols = set((schemas.get(theme) or {}).get("columns") or []) if theme else set()
    theme_pit = any(c.lower() in {"into_date", "out_date", "in_date", "entry_date"} or "INTO" in c for c in theme_cols)
    halt = has("md_sec_halt", "halt", "suspend")
    lhb = has("lhb", "dragon", "billboard", "top_list", "longhu")
    ann = has("announcement", "buyback", "reduction", "holder_trade")
    ann_cols = set((schemas.get(ann) or {}).get("columns") or []) if ann else set()
    ann_pit = any("announce" in c.lower() or c.lower() in {"ann_date", "publish_date", "declare_date"} for c in ann_cols)
    return {
        "dragon_tiger_table": lhb,
        "dragon_tiger_status": "LOCAL_FOUND" if lhb else BLOCKED_DATA,
        "theme_table": theme,
        "theme_has_into_out": bool(theme_pit),
        "theme_status": "PIT_FIELDS" if theme_pit else (BLOCKED_PIT if theme else BLOCKED_DATA),
        "halt_table": halt,
        "halt_status": "LOCAL_FOUND" if halt else "USE_DAILY_PAUSED_IF_PRESENT",
        "announcement_table": ann,
        "announcement_has_announce_date": bool(ann_pit),
        "announcement_status": "PIT_FIELDS" if ann_pit else (BLOCKED_PIT if ann else BLOCKED_DATA),
        "money_flow_raw": "mkt_equ_mf_new" if (HERMES_REBUILD_RAW / "mkt_equ_mf_new").exists() else None,
        "equd_raw": "mkt_equd" if (HERMES_REBUILD_RAW / "mkt_equd").exists() else None,
    }


def run_recover(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and REC_JSON.is_file() and MD["06"].is_file():
        existing = load_json(REC_JSON)
        if existing.get("status") in {"DONE", "PARTIAL", BLOCKED_ACCESS}:
            return existing
    REC.mkdir(parents=True, exist_ok=True)
    hermes = _listdir(HERMES_V7)
    raw = _listdir(HERMES_REBUILD_RAW)
    live = _listdir(DATA_LIVE_RELEASES)
    want = (
        "daily_raw",
        "daily_limit_events",
        "daily_money_flow",
        "share_capital",
        "announcement",
        "corporate_actions",
    )
    schemas: dict[str, Any] = {}
    for name in want:
        schemas[name] = _schema(HERMES_V7 / name)
    for name in ("mkt_equd", "mkt_equ_mf_new"):
        schemas[name] = _schema(HERMES_REBUILD_RAW / name)
    tables = list(hermes.get("entries") or [])
    raw_tables = list(raw.get("entries") or [])
    for name in tables + raw_tables:
        low = name.lower()
        if any(k in low for k in ("lhb", "dragon", "concept", "theme", "halt", "buyback", "holder", "reduction", "billboard")):
            root = HERMES_V7 / name if name in tables else HERMES_REBUILD_RAW / name
            schemas[name] = _schema(root)
    classified = _classify_sources(tables, raw_tables, schemas)
    ch = _clickhouse()
    if ch.get("status") != "OK" and classified["dragon_tiger_status"] != "LOCAL_FOUND":
        classified["dragon_tiger_status"] = BLOCKED_ACCESS if ch.get("status") == BLOCKED_ACCESS and not classified["dragon_tiger_table"] else classified["dragon_tiger_status"]
        if not classified["dragon_tiger_table"]:
            classified["dragon_tiger_status"] = BLOCKED_ACCESS if ch.get("credentials_present") is False or ch.get("status") == BLOCKED_ACCESS else BLOCKED_DATA
            if not classified["dragon_tiger_table"] and ch.get("status") == BLOCKED_ACCESS:
                classified["dragon_tiger_access"] = BLOCKED_ACCESS

    forward_fields = {
        "eod_daily_raw": _max_date(HERMES_V7 / "daily_raw"),
        "minute_2013": _max_date(MIN1_2013 / "minute_bars") if (MIN1_2013 / "minute_bars").exists() else {"exists": False, "max_date": None},
        "minute_2022": _max_date(MIN1_2022 / "minute_bars") if (MIN1_2022 / "minute_bars").exists() else {"exists": False, "max_date": None},
        "money_flow": _max_date(HERMES_V7 / "daily_money_flow") if (HERMES_V7 / "daily_money_flow").exists() else {"exists": False, "max_date": None},
        "candidate_note": "21.4 修复后 EOD 截止 2026-07-24，其后批次不完整",
    }
    post = []
    for field, info in forward_fields.items():
        if not isinstance(info, dict):
            continue
        mx = info.get("max_date")
        post.append({"field": field, "max_date": mx, "after_cutoff": bool(mx and str(mx) > MARKET_CUTOFF)})
    eod_max = str((forward_fields.get("eod_daily_raw") or {}).get("max_date") or "")
    minute_max = str((forward_fields.get("minute_2022") or {}).get("max_date") or "")
    complete_after = bool(eod_max > MARKET_CUTOFF and minute_max > MARKET_CUTOFF)
    # 即便文件日期越过 cutoff，候选面板 / 退出标签 / O01 仍未形成同日同池同引擎的独立 Forward。
    forward_status = FORWARD_OOS_NOT_AVAILABLE
    payload_note = (
        "EOD 与分钟日期越过 cutoff，但候选、标签与 O01 仍不构成独立 Forward。"
        if complete_after
        else "2026-07-24 之后的 EOD 或分钟仍不完整。"
    )
    payload = {
        "status": "PARTIAL" if classified["dragon_tiger_status"].startswith("BLOCKED") or classified["theme_status"].startswith("BLOCKED") else "DONE",
        "hermes_v7_readonly": str(HERMES_V7),
        "hermes_tables": tables,
        "raw_tables": raw_tables,
        "schemas_status": {k: {"status": v.get("status"), "n_columns": len(v.get("columns") or []), "columns_head": (v.get("columns") or [])[:12]} for k, v in schemas.items()},
        "classified": classified,
        "clickhouse": {k: v for k, v in ch.items() if k != "password"},
        "data_live": {k: v for k, v in live.items() if k != "entries"} | {"n": live.get("n"), "sample": (live.get("entries") or [])[:8]},
        "forward_fields": {k: ({kk: vv for kk, vv in v.items() if kk in {"max_date", "exists", "n_files", "status"}} if isinstance(v, dict) else v) for k, v in forward_fields.items()},
        "forward_rows": post,
        "forward_status": forward_status,
        "forward_note": payload_note,
        "cutoff": MARKET_CUTOFF,
        "writes": "result/round22/data_recovery only",
        "hermes_v7_overwritten": False,
    }
    if payload["status"] == "DONE" and forward_status == FORWARD_OOS_NOT_AVAILABLE:
        payload["status"] = "PARTIAL"
    dump_json(REC_JSON, payload)
    dump_json(REC / "source_classification.json", classified)
    write_csv(ROUND22_ROOT / "coverage" / "forward_matrix.csv", post)
    write_md(
        MD["05"],
        "\n".join(
            [
                "# Round22 · 05 数据审计",
                "",
                f"Hermes v7 只读。表数量={len(tables)}。raw 表数量={len(raw_tables)}。",
                f"data_live/releases status=`{live.get('status')}` n=`{live.get('n')}`。",
                f"ClickHouse {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT} status=`{ch.get('status')}` credentials_present=`{ch.get('credentials_present')}`。",
                f"龙虎榜=`{classified['dragon_tiger_status']}` 题材=`{classified['theme_status']}` 公告=`{classified['announcement_status']}` 停牌表=`{classified['halt_status']}`。",
                "",
                "无权限记 BLOCKED_ACCESS。不绕过，不把缺失写成 0。",
                "",
            ]
        ),
    )
    write_md(
        MD["06"],
        "\n".join(
            [
                "# Round22 · 06 数据恢复",
                "",
                "流程 DISCOVER→RECOVER→VALIDATE→INTEGRATE→RETEST。本轮没有把新分区写回 Hermes v7。",
                f"恢复目录只在 `{REC}`。",
                f"分类：`{classified}`。",
                "",
                "UPDATE_TIME 回填不能当历史 PIT。",
                "",
            ]
        ),
    )
    write_md(
        MD["07"],
        "\n".join(
            [
                "# Round22 · 07 Forward 覆盖",
                "",
                "沿用 Round21.4 已修好的全分区 `_max_date`（旧 bug 会把 daily_raw 锁在 2010-05-31）。",
                f"cutoff=`{MARKET_CUTOFF}`。status=`{forward_status}`。",
                "2026-07-24 之后的 EOD 批次在 21.4 已判断为不完整。本轮复扫后仍不把不完整批次当成独立 Forward。",
                "主研究不因 Forward 停止。已用于选因子的日期不会再标成未触碰 Forward。",
                "无独立 Forward：不得写成 live-ready。",
                "",
                f"字段={post}",
                "",
            ]
        ),
    )
    return payload
