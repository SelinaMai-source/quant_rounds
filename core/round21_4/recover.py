"""DISCOVER → RECOVER → VALIDATE → INTEGRATE → RETEST。授权、有界、限速、版本化新目录。"""

from __future__ import annotations

from typing import Any

from ..paths import MARKET_END, ROUND21_4_ROOT
from .common import dump_json, load_json, write_md
from .contract import BLOCKED_PERMISSION, DATA_LIVE_RELEASES, INCOMPLETE, MD
from .forward import COV_JSON, run_coverage

REC_JSON = ROUND21_4_ROOT / "data_recovery" / "recovery.json"


def run_recover(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and REC_JSON.is_file() and MD["04"].is_file():
        existing = load_json(REC_JSON)
        if existing.get("status") in {"DONE", "PARTIAL", "BLOCKED"}:
            return existing
    cov = run_coverage(resume=True)
    live = cov.get("data_live_releases") or {}
    missing = list(cov.get("missing") or [])
    steps = [
        {"step": "DISCOVER", "status": "DONE", "finding": f"coverage missing={len(missing)} live={live.get('status')}"},
    ]
    recovered = []
    blocked = []
    if live.get("status") == BLOCKED_PERMISSION:
        steps.append({"step": "RECOVER", "status": BLOCKED_PERMISSION, "reason": "data_live/releases 权限受限，尊重授权不越权读取"})
        blocked.append({"source": str(DATA_LIVE_RELEASES), "reason": BLOCKED_PERMISSION})
    elif live.get("status") == "OK":
        steps.append({"step": "RECOVER", "status": "PARTIAL", "reason": "releases 可见但未并入 Hermes v7 版本化快照；禁止原地覆盖 v7"})
        recovered.append({"source": str(DATA_LIVE_RELEASES), "batches": live.get("entries") or [], "integrated": False})
    else:
        steps.append({"step": "RECOVER", "status": "MISSING", "reason": "releases 目录不存在或不可列"})
    steps.append(
        {
            "step": "VALIDATE",
            "status": INCOMPLETE,
            "reason": "无独立 2026-07-25 后同日同池同引擎同标签成熟快照，不能 VALIDATE 为 Forward",
        }
    )
    steps.append({"step": "INTEGRATE", "status": "SKIPPED", "reason": "禁止覆盖 Hermes v7；新目录未授权写入 /data1/stock-mxy"})
    steps.append({"step": "RETEST", "status": "SKIPPED", "reason": "INTEGRATE 未完成"})
    payload = {
        "status": "PARTIAL",
        "cutoff": str(MARKET_END),
        "steps": steps,
        "missing": missing,
        "recovered": recovered,
        "blocked": blocked,
        "versioned_snapshot": None,
        "do_not_overwrite_hermes_v7": True,
        "independent_forward": False,
        "remaining": missing,
    }
    dump_json(REC_JSON, payload)
    write_md(
        MD["04"],
        "\n".join(
            [
                "# Round21.4 · 04 数据恢复协议",
                "",
                "覆盖不足走 DISCOVER→RECOVER→VALIDATE→INTEGRATE→RETEST。授权、有界、限速、版本化新目录。禁止覆盖 Hermes v7。",
                "",
                f"status=`{payload['status']}` live=`{live.get('status')}` remaining=`{len(missing)}`",
                f"步骤=`{steps}`",
                "",
            ]
        ),
    )
    return payload
