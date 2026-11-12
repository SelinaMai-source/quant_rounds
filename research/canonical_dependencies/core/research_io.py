"""三轮共用落盘。只写本轮新目录，不覆盖历史 round，不碰 CURRENT。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import PROJECT_ROOT, RESULT_ROOT, assert_project_result_path

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
ZOO_ROOT = PROJECT_ROOT / "research" / "factor_zoo"
MODEL_REGISTRY_ROOT = PROJECT_ROOT / "research" / "model_registry"

_ALLOWED_RESULT = (
    "round22_4",
    "round23",
    "round24",
    "round22_24_master",
)


def _result_ok(path: Path) -> bool:
    resolved = path.resolve()
    root = RESULT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        return False
    rel = resolved.relative_to(root)
    top = rel.parts[0]
    if top == "CURRENT" or top.startswith("CURRENT"):
        raise RuntimeError(f"refusing to touch CURRENT: {resolved}")
    if top not in _ALLOWED_RESULT:
        raise RuntimeError(f"refusing to write historical or foreign result: {resolved}")
    return True


def assert_new_result(path: Path) -> Path:
    resolved = path.resolve()
    if not _result_ok(resolved):
        raise RuntimeError(f"refusing path: {resolved}")
    return assert_project_result_path(resolved)


def assert_research_sidecar(path: Path) -> Path:
    resolved = path.resolve()
    research = (PROJECT_ROOT / "research").resolve()
    if resolved == research or research in resolved.parents:
        return resolved
    raise RuntimeError(f"refusing to write outside research sidecars: {resolved}")


def dump_json(path: Path, payload: Any) -> Path:
    path = assert_new_result(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: Path, body: str) -> Path:
    path = assert_new_result(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = body if body.endswith("\n") else body + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def write_df(path: Path, frame: pd.DataFrame) -> Path:
    path = assert_new_result(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


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


def append_tracker(section: str) -> None:
    """只在文末追加。不改旧行。"""
    path = TRACKER_PATH
    if not path.is_file():
        raise RuntimeError(f"missing tracker: {path}")
    text = section if section.endswith("\n") else section + "\n"
    if not text.startswith("\n"):
        text = "\n" + text
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def append_zoo_note(note: str) -> None:
    path = ZOO_ROOT / "registry.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and note.strip() in path.read_text(encoding="utf-8"):
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + note.strip() + "\n")
