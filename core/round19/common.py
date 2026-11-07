"""Round19 共用落盘。禁止写 Stock。原子写。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND19_ROOT, assert_project_result_path, ensure_round19_tree
from ..round11.common import (
    COMPLETE_YEARS,
    daily_from_nav,
    metrics_from_nav,
    observed_from_report,
    pct,
)
from ..round16.common import cagr_of, to_ymd
from ..stock_readonly import sha256_file


def dump_json(path: Path, payload: Any) -> Path:
    ensure_round19_tree()
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def update_state(stage: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    path = ROUND19_ROOT / "_state.json"
    payload = load_json(path) or {"round": 19}
    payload["round"] = 19
    payload["last_successful_stage"] = stage
    if extra:
        payload[stage] = extra
    dump_json(path, payload)
    return payload


def write_md(path: Path, body: str) -> Path:
    ensure_round19_tree()
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
    return sha256_file(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_or_none(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def slim_obs(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "cagr": finite_or_none(obs.get("cagr")),
        "sharpe": finite_or_none(obs.get("sharpe")),
        "max_drawdown": finite_or_none(obs.get("max_drawdown")),
        "completed_trades": obs.get("completed_trades"),
        "n_trading_observations": obs.get("n_trading_observations") or obs.get("n_days"),
        "total_return": finite_or_none(obs.get("total_return")),
        "yearly_returns": obs.get("yearly_returns") or {},
        "status": obs.get("status"),
    }


def not_stage0(name: str) -> dict[str, Any]:
    raise RuntimeError(
        f"STAGE_NOT_IMPLEMENTED:{name} 由后续 Track 代理实现。Stage 0 只做 read/parity/freeze。"
    )


__all__ = [
    "COMPLETE_YEARS",
    "cagr_of",
    "daily_from_nav",
    "dump_json",
    "file_sha256",
    "finite_or_none",
    "load_json",
    "metrics_from_nav",
    "not_stage0",
    "observed_from_report",
    "pct",
    "sha256_bytes",
    "slim_obs",
    "to_ymd",
    "update_state",
    "write_md",
]
