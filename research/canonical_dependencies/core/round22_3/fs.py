"""Track D：财务报表 PIT Join。facts→reports 用 report_id；利润与现金流按 instrument+period+statement+available_at+revision。"""

from __future__ import annotations

from typing import Any

import pyarrow.parquet as pq

from ..round21.contract import HERMES_REBUILD_RAW, HERMES_V7
from .common import dump_json, load_json, write_md
from .contract import (
    BLOCKED_DATA,
    BLOCKED_PERIOD_MISMATCH,
    BLOCKED_PIT,
    MD,
    ROUND22_2_ROOT,
    ROUND22_3_ROOT,
)
from .freeze import require_frozen_gate

FS_JSON = ROUND22_3_ROOT / "fs" / "fs.json"


def _schema(root, name: str) -> dict[str, Any]:
    path = root / name
    out: dict[str, Any] = {"name": name, "path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    sample = next(path.rglob("*.parquet"), None) if path.is_dir() else (path if path.suffix == ".parquet" else None)
    if sample is None:
        out["n_parquet"] = 0
        return out
    try:
        schema = pq.read_schema(sample)
        cols = list(schema.names)
    except Exception as exc:
        out["error"] = type(exc).__name__
        return out
    out["sample"] = str(sample)
    out["columns"] = cols
    upper = {c.upper() for c in cols}
    out["has_report_id"] = "REPORT_ID" in upper or "REPORTID" in upper
    out["has_statement"] = any(x in upper for x in ("STATEMENT", "STATEMENT_TYPE", "REPORT_TYPE", "SHEET"))
    out["has_period"] = any(x in upper for x in ("PERIOD", "END_DATE", "REPORT_PERIOD", "ENDDATE", "FISCAL_PERIOD"))
    out["has_available"] = any(x in upper for x in ("AVAILABLE_AT", "PUBLISHED_AT", "ANN_DT", "PUBLISH_DATE"))
    out["has_revision"] = any(x in upper for x in ("REVISION", "REV", "UPDATE_TIME"))
    return out


def run_fs(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and FS_JSON.is_file() and MD["11"].is_file():
        existing = load_json(FS_JSON)
        if existing.get("status") in {"DONE", "PARTIAL", BLOCKED_DATA, BLOCKED_PIT, BLOCKED_PERIOD_MISMATCH}:
            return existing
    helper = load_json(ROUND22_3_ROOT / "fs" / "fs_join.json") or load_json(ROUND22_3_ROOT / "data_recovery" / "fs.json")
    if helper and helper.get("status"):
        dump_json(FS_JSON, helper)
        write_md(MD["11"], f"# Round22.3 · 11 FS Join\n\n`{helper.get('status')}` {helper.get('reason')}\n\n只做 FS1/FS2。不同 statement 的 report_id 不得当同一 ID。累计/单季不对齐则 BLOCKED_PERIOD_MISMATCH。\n")
        return helper
    prev = load_json(ROUND22_2_ROOT / "data_recovery" / "fs.json")
    schemas = {
        "financial_facts": _schema(HERMES_V7, "financial_facts"),
        "financial_reports": _schema(HERMES_V7, "financial_reports"),
        "fundamental_metrics": _schema(HERMES_V7, "fundamental_metrics"),
        "vw_fdmt_is_new": _schema(HERMES_REBUILD_RAW, "vw_fdmt_is_new"),
        "vw_fdmt_cf_new": _schema(HERMES_REBUILD_RAW, "vw_fdmt_cf_new"),
        "vw_fdmt_bs_new": _schema(HERMES_REBUILD_RAW, "vw_fdmt_bs_new"),
        "fdmt_main_data_q_pit": _schema(HERMES_REBUILD_RAW, "fdmt_main_data_q_pit"),
    }
    dump_json(ROUND22_3_ROOT / "fs" / "schema_audit.json", schemas)
    facts = schemas["financial_facts"]
    reports = schemas["financial_reports"]
    income = schemas["vw_fdmt_is_new"]
    cash = schemas["vw_fdmt_cf_new"]
    reason = None
    status = BLOCKED_DATA
    if not facts.get("exists") or not reports.get("exists"):
        reason = "financial_facts / financial_reports 不可用。22.2 仅命中表名未完成 join。"
    elif not facts.get("has_report_id") or not reports.get("has_report_id"):
        reason = "facts→reports 缺 report_id，不得用别的键硬连。"
        status = BLOCKED_PIT
    elif not (income.get("has_period") and cash.get("has_period") and income.get("has_available") and cash.get("has_available")):
        reason = "利润与现金流未同时具备 period + available_at/published_at。"
        status = BLOCKED_PIT
    else:
        reason = (
            "表与时点列已命中。完整 instrument+period+statement+available_at+revision 对齐由 helper 完成；"
            "本模块在 helper 未返回前不硬造应计。不同 statement 的 report_id 不得当同一 ID。"
        )
        status = "PENDING_HELPER_OR_BLOCKED"
    payload = {
        "status": status if not helper else helper.get("status"),
        "reason": reason,
        "schemas": {k: {kk: vv for kk, vv in v.items() if kk != "columns"} for k, v in schemas.items()},
        "fs1": {"factor_id": "r22_3_fs1_accrual_minus_ocf", "formula": "(NI-OCF)/avg_assets", "constructed": False, "model_eligible": False},
        "fs2": {"factor_id": "r22_3_fs2_ocf_over_assets", "formula": "OCF/avg_assets", "constructed": False, "model_eligible": False},
        "filled_with_zero": False,
        "prior_22_2": {"status": prev.get("status"), "reason": prev.get("reason")},
        "join_keys": {
            "facts_to_reports": "report_id",
            "income_cash": "instrument+period+statement+available_at+revision",
            "forbid_cross_statement_report_id": True,
        },
    }
    dump_json(FS_JSON, payload)
    dump_json(ROUND22_3_ROOT / "data_recovery" / "fs.json", payload)
    dump_json(ROUND22_3_ROOT / "factors" / "r22_3_fs1_accrual_minus_ocf" / "pit.json", {**payload["fs1"], "status": payload["status"]})
    dump_json(ROUND22_3_ROOT / "factors" / "r22_3_fs2_ocf_over_assets" / "pit.json", {**payload["fs2"], "status": payload["status"]})
    write_md(MD["11"], f"# Round22.3 · 11 FS Join\n\n`{payload['status']}` {reason}\n\n22.2 为 BLOCKED_DATA（join 未做成）。本轮要求 report_id + period/statement 对齐。不填 0。只做 FS1/FS2。\n")
    return payload
