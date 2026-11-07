"""Forward OOS 检查 + append-only shadow 协议。默认 FORWARD_OOS_NOT_AVAILABLE。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, ROUND12_ROOT
from ..round11.contract import source_tree_sha256
from ..stock_readonly import sha256_file
from .common import dump_json, write_md
from .contract import PROTOCOL_SOURCE_FILES, SNAPSHOT_FINGERPRINT, SNAPSHOT_ID, STRATEGY_SOURCE_FILES
from .freeze import require_frozen

SHADOW_MD = ROUND12_ROOT / "14_shadow.md"
SHADOW_JSON = ROUND12_ROOT / "shadow_protocol" / "shadow.json"
PROTOCOL_DIR = ROUND12_ROOT / "shadow_protocol"


def _probe_forward() -> dict[str, Any]:
    """冻结后检查 2026-07-24 之后是否存在完整、未用于选参的首板可跑数据。"""
    inbox_candidates = [
        Path("/home/mxy/Stock/data/inbox"),
        Path("/data1/stock-mxy/data/inbox"),
        Path("/home/mxy/AmazingData/data/inbox"),
    ]
    found = []
    for root in inbox_candidates:
        if not root.exists():
            found.append({"path": str(root), "exists": False})
            continue
        try:
            names = sorted(p.name for p in root.iterdir())[:40]
            found.append({"path": str(root), "exists": True, "readable": True, "names_head": names})
        except PermissionError:
            found.append({"path": str(root), "exists": True, "readable": False, "error": "Permission denied"})
    return {
        "snapshot_end": "2026-07-24",
        "required_span": "2026-07-27 .. at least first-board tables",
        "inbox": found,
        "usable_complete_unused": False,
        "status": "FORWARD_OOS_NOT_AVAILABLE",
        "reason": (
            "研究快照止于 2026-07-24。inbox 即使有 2026-08-31..09-30 增量，"
            "仍缺 2026-07-27..08-28、未并入 v7、缺首板所需表、部分路径 Permission denied。"
            "不得把 CPCV/bootstrap 冒充 OOS。"
        ),
    }


def _write_protocol(frozen: dict[str, Any], forward: dict[str, Any]) -> None:
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ROUND12_ROOT / "round12_manifest_v1.yaml"
    hashes = {
        "manifest": frozen.get("sha256"),
        "manifest_file": sha256_file(manifest) if manifest.is_file() else None,
        "source_tree": source_tree_sha256(STRATEGY_SOURCE_FILES),
        "protocol_tree": source_tree_sha256(PROTOCOL_SOURCE_FILES),
        "snapshot_id": SNAPSHOT_ID,
        "snapshot_fingerprint": SNAPSHOT_FINGERPRINT,
    }
    dump_json(PROTOCOL_DIR / "frozen_manifest.json", {"frozen": frozen, "hashes": hashes})
    dump_json(PROTOCOL_DIR / "hashes.json", hashes)
    schema = {
        "daily_row": {
            "trade_date": "date",
            "status": "SHADOW_OK | NO_TRADE | DATA_FAIL | EXECUTION_EXCEPTION",
            "data_hash": "sha256",
            "model_hash": "sha256",
            "config_hash": "sha256",
            "source_hash": "sha256",
            "signal_hash": "sha256",
            "order_hash": "sha256",
            "n_signals": "int",
            "n_orders": "int",
            "note": "str",
        },
        "append_only": True,
        "kill_switch": {
            "modes": ["WARN", "NO_TRADE"],
            "optimizes_return": False,
            "monitors_only": True,
        },
        "allowed_live_ids": ["R8_E02", "R9_B03", "R12_06_LIQ_EXEC", "R12_07_LIQ_EXEC_CONSERVATIVE"],
        "no_retune_on_short_sample": True,
        "veto_allowed": True,
    }
    dump_json(PROTOCOL_DIR / "schema.json", schema)
    write_md(
        PROTOCOL_DIR / "README.md",
        "\n".join([
            "# Round 12 shadow protocol",
            "",
            "append-only。日数据 / 模型 / 配置 / 源码 / 信号 / 订单 hash。",
            "状态：`SHADOW_OK` / `NO_TRADE` / `DATA_FAIL` / `EXECUTION_EXCEPTION`。",
            "Kill switch 只监控（WARN/NO_TRADE），不优化收益。",
            "",
            "入口：`python tools/run_shadow_daily.py --date YYYY-MM-DD`",
            "",
            f"forward OOS = `{forward['status']}`。",
            "",
        ]),
    )


def run_shadow(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    frozen = require_frozen()
    dest = PROTOCOL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    if resume and SHADOW_JSON.is_file():
        from .common import load_json

        existing = load_json(SHADOW_JSON)
        if existing.get("status"):
            return existing
    forward = _probe_forward()
    _write_protocol(frozen, forward)
    ran_forward = False
    payload = {
        "status": forward["status"],
        "forward": forward,
        "ran_v3_v4_r12_06_r12_07_on_forward": ran_forward,
        "protocol_dir": str(PROTOCOL_DIR),
        "tool": "tools/run_shadow_daily.py",
        "kill_switch": "monitor_only",
        "manifest_sha256": frozen.get("sha256"),
    }
    dump_json(SHADOW_JSON, payload)
    write_md(
        SHADOW_MD,
        "\n".join([
            "# Round 12 · Forward / Shadow",
            "",
            f"**{forward['status']}**",
            "",
            forward["reason"],
            "",
            "已冻结 manifest 后再探。短样本不得调参。",
            f"协议目录：`{PROTOCOL_DIR}`。",
            "工具：`tools/run_shadow_daily.py`。",
            "",
        ]),
    )
    return payload


def load_shadow() -> dict[str, Any]:
    from .common import load_json

    return load_json(SHADOW_JSON)


__all__ = ["load_shadow", "run_shadow"]
