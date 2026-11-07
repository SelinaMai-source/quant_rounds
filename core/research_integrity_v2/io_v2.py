"""本轮结果落盘。只允许新目录，禁止写 CURRENT / 历史 round。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.paths import RESULT_ROOT, assert_project_result_path

_ALLOWED = (
    "round22_6",
    "round23_2",
    "round24_2",
    "round22_6_24_2_master",
)


def assert_new_result(path: Path) -> Path:
    resolved = Path(path).resolve()
    root = RESULT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside result: {resolved}")
    top = resolved.relative_to(root).parts[0]
    if top == "CURRENT" or top.startswith("CURRENT"):
        raise RuntimeError(f"refusing to touch CURRENT: {resolved}")
    if top not in _ALLOWED:
        raise RuntimeError(f"refusing to write historical or foreign result: {resolved}")
    return assert_project_result_path(resolved)


def _atomic_text(path: Path, text: str) -> Path:
    path = assert_new_result(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def dump_json(path: Path, payload: Any) -> Path:
    return _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def write_text(path: Path, body: str) -> Path:
    return _atomic_text(path, body)


def load_json(path: Path) -> dict[str, Any]:
    file = Path(path)
    if not file.is_file():
        return {}
    return json.loads(file.read_text(encoding="utf-8"))


def write_df(path: Path, frame: pd.DataFrame) -> Path:
    path = assert_new_result(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def write_parquet(path: Path, frame: pd.DataFrame) -> Path:
    path = assert_new_result(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def stage_path(root: Path, name: str) -> Path:
    return root / "_state" / f"{name}.json"


def save_stage(root: Path, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = stage_path(root, name)
    dump_json(path, payload)
    return payload


def load_stage(root: Path, name: str) -> dict[str, Any] | None:
    path = stage_path(root, name)
    if not path.is_file():
        return None
    return load_json(path)
