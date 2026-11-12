"""恢复/核验 D 日涨停分钟、raw 订单档、as-of 股本。不覆盖 Hermes 原始。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND18_ROOT
from .common import dump_json, load_json, write_md
from .contract import HERMES_SCHEMA, HERMES_V7, MF_RAW, MIN1_2013, MIN1_2022
from .inventory import run_inventory

DL_JSON = ROUND18_ROOT / "download" / "recover.json"
DL_MD = ROUND18_ROOT / "04_data_recovery.md"


def run_recover(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and DL_JSON.is_file():
        existing = load_json(DL_JSON)
        if existing.get("status") in {"RECOVERED_LOCAL", "ALREADY_PRESENT"}:
            return existing
    inv = run_inventory()
    missing = []
    if not MIN1_2013.is_dir() or not MIN1_2022.is_dir():
        missing.append("F1_minutes")
    if not MF_RAW.is_dir():
        missing.append("F2_raw_mf")
    if not (HERMES_V7 / "daily_valuation").is_dir():
        missing.append("F3_valuation")
    if not (HERMES_V7 / "daily_limit_events").is_dir():
        missing.append("limit_events")
    if not (HERMES_V7 / "daily_raw").is_dir():
        missing.append("daily_raw")
    if missing:
        payload = {
            "status": "PERMISSION_REQUIRED" if missing else "SKIP_DATA",
            "missing": missing,
            "downloaded": False,
            "overwrote_hermes": False,
            "note": "本地缺失且本轮不覆盖/不重拉 Hermes 原始。无权限标 PERMISSION_REQUIRED。",
        }
    else:
        payload = {
            "status": "ALREADY_PRESENT",
            "missing": [],
            "downloaded": False,
            "overwrote_hermes": False,
            "sources": {
                "min1_2013": str(MIN1_2013),
                "min1_2022": str(MIN1_2022),
                "mf_raw": str(MF_RAW),
                "schema": str(HERMES_SCHEMA),
                "v7": str(HERMES_V7),
            },
            "inventory": {
                "f1_days": (inv.get("f1_minutes") or {}).get("n_days_2013"),
                "f1_days_2022": (inv.get("f1_minutes") or {}).get("n_days_2022"),
                "f2_parts": (inv.get("f2_raw_mf") or {}).get("n_parts"),
            },
        }
    dump_json(DL_JSON, payload)
    write_md(
        DL_MD,
        "# 04 数据恢复\n\n"
        f"status=`{payload['status']}` downloaded=`{payload['downloaded']}` "
        f"overwrote_hermes=`False` missing=`{payload['missing']}`\n",
    )
    return payload
