"""Track C 路径。产物只写 result/round19。不覆盖 Track A/B 原件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...paths import ROUND19_ROOT
from ..common import dump_json, load_json, write_md

TC = ROUND19_ROOT / "track_c"
FWD_JSON = TC / "forward" / "forward.json"
FWD_MD = TC / "forward" / "forward.md"
BIAS_JSON = TC / "decision" / "selection_bias.json"
BIAS_MD = TC / "decision" / "selection_bias.md"
DEC_JSON = TC / "decision" / "decision.json"
DEC_MD = TC / "decision" / "decision.md"
DOSSIER_JSON = ROUND19_ROOT / "stats" / "dossier.json"

INBOX_BATCHES = Path("/data1/Stock/data_live/inbox/batches")
INBOX_RELAY = Path("/data1/Stock/data_live/inbox/server_relay")

FIRST_BOARD_NEEDLES = (
    "first_board",
    "candidate_panel",
    "daily_limit_events",
    "option",
    "minute",
    "fb_",
)


def resume_ok(path: Path, *, status: str = "DONE") -> dict[str, Any] | None:
    if not path.is_file():
        return None
    existing = load_json(path)
    if existing.get("status") == status:
        return existing
    return None


__all__ = [
    "BIAS_JSON",
    "BIAS_MD",
    "DEC_JSON",
    "DEC_MD",
    "DOSSIER_JSON",
    "FIRST_BOARD_NEEDLES",
    "FWD_JSON",
    "FWD_MD",
    "INBOX_BATCHES",
    "INBOX_RELAY",
    "TC",
    "dump_json",
    "load_json",
    "resume_ok",
    "write_md",
]
