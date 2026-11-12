"""c-forward：冻结 hash 之后才探 live inbox。缺字段 → FORWARD_OOS_NOT_AVAILABLE。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ...paths import MARKET_END
from ..contract import FORWARD_OOS_NOT_AVAILABLE, LIVE_VALIDATED, SAME_OPEN_EXECUTION_OPTIMISM
from ..freeze import require_frozen
from .support import (
    FIRST_BOARD_NEEDLES,
    FWD_JSON,
    FWD_MD,
    INBOX_BATCHES,
    INBOX_RELAY,
    dump_json,
    resume_ok,
    write_md,
)


def _list_dir(path: Path, *, limit: int = 40) -> dict[str, Any]:
    rec: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return rec
    rec["is_dir"] = path.is_dir()
    try:
        kids = sorted(path.iterdir(), key=lambda p: p.name)
    except PermissionError:
        rec["readable"] = False
        rec["error"] = "Permission denied"
        return rec
    rec["readable"] = True
    rec["n_children"] = len(kids)
    rec["names_head"] = [p.name for p in kids[:limit]]
    return rec


def _batch_dates(root: Path) -> list[str]:
    if not root.exists():
        return []
    out: list[str] = []
    try:
        kids = list(root.iterdir())
    except PermissionError:
        return []
    for child in kids:
        parts = child.name.split("-")
        if len(parts) >= 3 and parts[2].isdigit() and len(parts[2]) == 8:
            out.append(parts[2])
    return sorted(set(out))


def _weekday_span(start: date, end: date) -> list[str]:
    out: list[str] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


def _probe_tables(root: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"path": str(root), "exists": root.exists(), "raw_tables": [], "first_board_hits": []}
    if not root.exists():
        return rec
    try:
        samples = sorted(root.iterdir(), key=lambda p: p.name)
    except PermissionError:
        rec["error"] = "Permission denied listing batches"
        return rec
    if not samples:
        return rec
    sample = samples[0]
    raw = sample / "raw"
    if raw.is_dir():
        try:
            rec["raw_tables"] = sorted(p.name for p in raw.iterdir())
        except PermissionError:
            rec["raw_tables_error"] = "Permission denied"
    rec["sample_batch"] = sample.name
    hits: list[str] = []
    try:
        for child in sample.rglob("*"):
            name = child.name.lower()
            if any(k in name for k in FIRST_BOARD_NEEDLES):
                hits.append(str(child.relative_to(sample)))
            if len(hits) >= 20:
                break
    except PermissionError:
        rec["rglob_error"] = "Permission denied"
    rec["first_board_hits"] = hits
    required_missing = [
        "first_board candidate_panel",
        "daily_limit_events / minute first-board tables",
        "option / IV chain for O01",
        "v7 merged unused panel after 2026-07-24",
    ]
    rec["required_missing"] = required_missing
    rec["manifest_readable"] = False
    man = sample / "manifest.json"
    if man.is_file():
        try:
            man.read_text(encoding="utf-8")
            rec["manifest_readable"] = True
        except PermissionError:
            rec["manifest_error"] = "Permission denied"
    return rec


def _probe_relay(root: Path) -> dict[str, Any]:
    rec = _list_dir(root)
    if not root.exists():
        rec["partial_or_failed"] = True
        rec["note"] = "server_relay missing"
        return rec
    failed = 0
    partial = 0
    dates: list[str] = []
    try:
        for child in root.rglob("*"):
            name = child.name
            if name == "FAILED.json":
                failed += 1
            if ".partial" in str(child):
                partial += 1
            if child.is_dir() and len(child.name) == 10 and child.name[4] == "-":
                dates.append(child.name)
    except PermissionError:
        rec["rglob_error"] = "Permission denied"
    rec["failed_json"] = failed
    rec["partial_markers"] = partial
    rec["relay_dates_head"] = sorted(set(dates))[:30]
    rec["usable_complete"] = False
    return rec


def run_forward(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    existing = resume_ok(FWD_JSON) if resume else None
    if existing is not None:
        return existing
    frozen = require_frozen()
    batches = _list_dir(INBOX_BATCHES)
    relay = _probe_relay(INBOX_RELAY)
    tables = _probe_tables(INBOX_BATCHES)
    present = _batch_dates(INBOX_BATCHES)
    latest = date.fromisoformat(present[-1][:4] + "-" + present[-1][4:6] + "-" + present[-1][6:8]) if present else MARKET_END
    weekdays = _weekday_span(MARKET_END + timedelta(days=1), max(latest, MARKET_END + timedelta(days=1)))
    missing_weekdays = [d for d in weekdays if d not in present]
    first_board_complete = bool(tables.get("first_board_hits")) and not tables.get("required_missing")
    usable = bool(
        batches.get("exists")
        and batches.get("readable")
        and present
        and not missing_weekdays
        and first_board_complete
        and tables.get("manifest_readable")
        and relay.get("usable_complete")
    )
    reasons = [
        f"研究快照截止 {MARKET_END}。冻结 hash 之后才探 inbox。",
        f"batches 存在={batches.get('exists')} readable={batches.get('readable')} n={batches.get('n_children')} dates={present[:3]}..{present[-3:] if present else []}.",
        f"2026-07-25 之后工作日缺口 {len(missing_weekdays)} 天（含 2026-07-27..08-28）。",
        "inbox 仅为 hermes-eod 通用表（mkt_equd / mkt_limit 等），无 first_board / candidate_panel / option / minute 首板可跑字段。",
        f"manifest.json 可读={tables.get('manifest_readable')}；server_relay failed={relay.get('failed_json')} partial={relay.get('partial_markers')}。",
        "未并入 v7，不得把增量 EOD 当完整未选参 forward panel。",
        f"{SAME_OPEN_EXECUTION_OPTIMISM} 仍在。禁止写 {LIVE_VALIDATED}。",
    ]
    payload = {
        "status": FORWARD_OOS_NOT_AVAILABLE,
        "forward_validation_supported": False,
        "live_validated": False,
        "same_open_execution_optimism": SAME_OPEN_EXECUTION_OPTIMISM,
        "cutoff": str(MARKET_END),
        "checked_after_freeze": True,
        "freeze_hashes": {
            "track_a": (frozen.get("track_a") or {}).get("sha256"),
            "track_b": (frozen.get("track_b") or {}).get("sha256"),
        },
        "inbox_batches": batches,
        "server_relay": relay,
        "tables": tables,
        "present_dates": present,
        "n_present_dates": len(present),
        "missing_weekdays": missing_weekdays,
        "n_missing_weekdays": len(missing_weekdays),
        "usable_complete_unused_first_board": usable,
        "reasons": reasons,
        "note": " ".join(reasons),
    }
    dump_json(FWD_JSON, payload)
    write_md(
        FWD_MD,
        "\n".join(
            [
                "# Round19 Track C · Forward OOS",
                "",
                f"**{FORWARD_OOS_NOT_AVAILABLE}**。不得写 `{LIVE_VALIDATED}`。",
                "",
                f"冻结后再查 `{INBOX_BATCHES}` 与 `{INBOX_RELAY}`。",
                f"Track A sha256=`{(frozen.get('track_a') or {}).get('sha256')}`",
                f"Track B sha256=`{(frozen.get('track_b') or {}).get('sha256')}`",
                "",
                *([f"- {r}" for r in reasons]),
                "",
                f"`{SAME_OPEN_EXECUTION_OPTIMISM}` 仍在。C3/C4 支持 ≠ 独立 OOS。",
                "",
            ]
        ),
    )
    return payload
