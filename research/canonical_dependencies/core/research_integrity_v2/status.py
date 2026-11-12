"""三态证据状态。缺证据不得默认 True / PASS。"""

from __future__ import annotations

from enum import Enum
from typing import Any


class EvidenceStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"
    BLOCKED = "BLOCKED"


def as_bool(value: Any) -> bool | None:
    """只接受真正的 bool；字符串 'False'/'True' 返回 None，禁止被当成真。"""
    if isinstance(value, bool):
        return value
    return None


def require_bool(value: Any, *, field: str) -> bool:
    parsed = as_bool(value)
    if parsed is None:
        raise TypeError(f"{field} must be real bool, got {type(value).__name__}={value!r}")
    return parsed


def status_from_bool(value: bool | None, *, missing: EvidenceStatus = EvidenceStatus.NOT_EVALUATED) -> EvidenceStatus:
    if value is None:
        return missing
    return EvidenceStatus.PASS if value else EvidenceStatus.FAIL
