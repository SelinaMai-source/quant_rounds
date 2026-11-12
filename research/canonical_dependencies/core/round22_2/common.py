"""Round 22.2 落盘。只写 result/round22_2。禁止写 result/round22*、result/round21* 与 CURRENT。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import (
    RESULT_ROOT,
    ROUND21_1_ROOT,
    ROUND21_2_ROOT,
    ROUND21_3_ROOT,
    ROUND21_4_ROOT,
    ROUND21_ROOT,
    ROUND22_1_ROOT,
    ROUND22_ROOT,
    assert_project_result_path,
)
from ..stock_readonly import sha256_file
from .contract import ROUND22_2_ROOT, ROUND22_2_SHARED_DIRS

_FORBIDDEN_ROOTS = (
    ROUND21_ROOT.resolve(),
    ROUND21_1_ROOT.resolve(),
    ROUND21_2_ROOT.resolve(),
    ROUND21_3_ROOT.resolve(),
    ROUND21_4_ROOT.resolve(),
    ROUND22_ROOT.resolve(),
    ROUND22_1_ROOT.resolve(),
)


def ensure_round22_2_tree() -> Path:
    ROUND22_2_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND22_2_SHARED_DIRS:
        (ROUND22_2_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND22_2_ROOT


def _forbid_legacy(path: Path) -> Path:
    resolved = Path(path).resolve()
    for root in _FORBIDDEN_ROOTS:
        if resolved == root or root in resolved.parents:
            raise RuntimeError(f"refusing to write into legacy ledger: {resolved}")
    result = RESULT_ROOT.resolve()
    if resolved != result and result not in resolved.parents:
        # tracker / STANDARD / zoo / evaluate 由各自模块断言
        pass
    if resolved.name in {"CURRENT", "CURRENT_ALPHA_BASELINE"} or resolved.parent.name == "CURRENT":
        raise RuntimeError(f"refusing to touch CURRENT: {resolved}")
    name = resolved.name
    if name.startswith("round22") and name != "round22_2" and ROUND22_2_ROOT.resolve() not in resolved.parents:
        if resolved.parent == result and name != "round22_2":
            raise RuntimeError(f"refusing to write sibling round ledger: {resolved}")
    return resolved


def dump_json(path: Path, payload: Any) -> Path:
    ensure_round22_2_tree()
    _forbid_legacy(path)
    if str(path).startswith(str(RESULT_ROOT)):
        assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def write_md(path: Path, body: str) -> Path:
    ensure_round22_2_tree()
    _forbid_legacy(path)
    if str(path).startswith(str(RESULT_ROOT)):
        assert_project_result_path(path)
    text = body if body.endswith("\n") else body + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_state(stage: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    path = ROUND22_2_ROOT / "_state.json"
    payload = load_json(path) or {"round": "22.2"}
    payload["round"] = "22.2"
    payload["last_successful_stage"] = stage
    if extra:
        payload[stage] = extra
    dump_json(path, payload)
    return payload


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256_file(path)


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def feature_set_hash(features: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(list(features), ensure_ascii=False).encode("utf-8")).hexdigest()


def finite_or_none(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def close_enough(a: Any, b: Any, *, abs_tol: float) -> bool:
    fa, fb = finite_or_none(a), finite_or_none(b)
    if fa is None or fb is None:
        return False
    return abs(fa - fb) < abs_tol


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> Path:
    import pandas as pd

    ensure_round22_2_tree()
    _forbid_legacy(path)
    if str(path).startswith(str(RESULT_ROOT)):
        assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if columns:
        for col in columns:
            if col not in frame.columns:
                frame[col] = np.nan
        frame = frame[columns]
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path
