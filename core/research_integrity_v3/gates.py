"""机器可读 GateResult。PASS 必须引用产物路径与 sha256。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED = frozenset({"PASS", "FAIL", "BLOCKED", "NOT_EVALUATED"})


def fingerprint(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class GateResult:
    gate_id: str
    status: str
    artifact: str | None = None
    artifact_sha256: str | None = None
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ALLOWED:
            raise ValueError(f"illegal gate status: {self.status}")
        if self.status == "PASS":
            if not self.artifact or not self.artifact_sha256:
                raise ValueError(f"{self.gate_id} PASS requires artifact and sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "status": self.status,
            "artifact": self.artifact,
            "artifact_sha256": self.artifact_sha256,
            "reason": self.reason,
            "detail": self.detail,
        }


def gate_from_file(
    gate_id: str,
    path: Path,
    *,
    ok: bool,
    reason: str,
    detail: dict[str, Any] | None = None,
) -> GateResult:
    path = Path(path)
    if not path.is_file():
        return GateResult(gate_id, "BLOCKED", reason=f"missing artifact {path}", detail=detail or {})
    digest = sha256_file(path)
    return GateResult(
        gate_id,
        "PASS" if ok else "FAIL",
        artifact=str(path),
        artifact_sha256=digest,
        reason=reason,
        detail=detail or {},
    )
