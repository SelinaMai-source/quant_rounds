"""Round17 共用落盘。禁止写 Stock。原子写。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND17_ROOT, assert_project_result_path, ensure_round17_tree
from ..round11.common import (
    COMPLETE_YEARS,
    daily_from_nav,
    metrics_from_nav,
    observed_from_report,
    pct,
)
from ..round16.common import cagr_of, to_ymd


def dump_json(path: Path, payload: Any) -> Path:
    ensure_round17_tree()
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def update_state(stage: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    path = ROUND17_ROOT / "_state.json"
    payload = load_json(path) or {"round": 17}
    payload["round"] = 17
    payload["last_successful_stage"] = stage
    if extra:
        payload[stage] = extra
    dump_json(path, payload)
    return payload


def write_md(path: Path, body: str) -> Path:
    ensure_round17_tree()
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


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return sha256_bytes(blob)


def clip_nav_window(nav: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    work = nav.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    return work.loc[work["trade_date"].between(lo, hi)].sort_values("trade_date").reset_index(drop=True)


def yearly_from_nav(nav: pd.DataFrame) -> dict[str, float]:
    work = nav.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work = work.sort_values("trade_date")
    if work.empty or "total_asset" not in work.columns:
        return {}
    out: dict[str, float] = {}
    for year, block in work.groupby(work["trade_date"].dt.year, sort=True):
        first = float(block["total_asset"].iloc[0])
        last = float(block["total_asset"].iloc[-1])
        if first > 0:
            out[str(int(year))] = last / first - 1.0
    return out


def finite_or_none(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


__all__ = [
    "COMPLETE_YEARS",
    "cagr_of",
    "clip_nav_window",
    "daily_from_nav",
    "dump_json",
    "file_sha256",
    "finite_or_none",
    "load_json",
    "metrics_from_nav",
    "observed_from_report",
    "pct",
    "sha256_bytes",
    "sha256_json",
    "to_ymd",
    "update_state",
    "write_md",
    "yearly_from_nav",
]
