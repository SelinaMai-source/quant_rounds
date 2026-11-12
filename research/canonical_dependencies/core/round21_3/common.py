"""Round21.3 共用落盘。只写 result/round21_3。禁止写 result/round21 / 21.1 / 21.2 账本。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import (
    ROUND21_1_ROOT,
    ROUND21_2_ROOT,
    ROUND21_3_ROOT,
    ROUND21_ROOT,
    assert_project_result_path,
    ensure_round21_3_tree,
)
from ..round11.common import daily_from_nav, metrics_from_nav, observed_from_report
from ..round16.common import cagr_of, to_ymd
from ..stock_readonly import sha256_file


def _forbid_legacy(path: Path) -> Path:
    resolved = Path(path).resolve()
    for root in (ROUND21_ROOT.resolve(), ROUND21_1_ROOT.resolve(), ROUND21_2_ROOT.resolve()):
        if resolved == root or root in resolved.parents:
            raise RuntimeError(f"refusing to write into legacy ledger: {resolved}")
    return resolved


def dump_json(path: Path, payload: Any) -> Path:
    ensure_round21_3_tree()
    _forbid_legacy(path)
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def write_md(path: Path, body: str) -> Path:
    ensure_round21_3_tree()
    _forbid_legacy(path)
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
    path = ROUND21_3_ROOT / "_state.json"
    payload = load_json(path) or {"round": "21.3"}
    payload["round"] = "21.3"
    payload["last_successful_stage"] = stage
    if extra:
        payload[stage] = extra
    dump_json(path, payload)
    return payload


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256_file(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return sha256_bytes(blob)


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


def slim_obs(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "cagr": finite_or_none(obs.get("cagr")),
        "sharpe": finite_or_none(obs.get("sharpe")),
        "max_drawdown": finite_or_none(obs.get("max_drawdown")),
        "completed_trades": obs.get("completed_trades"),
        "n_trading_observations": obs.get("n_trading_observations") or obs.get("n_obs"),
        "nav_end": finite_or_none(obs.get("nav_end")),
        "total_return": finite_or_none(obs.get("total_return")),
        "yearly_returns": obs.get("yearly_returns") or {},
        "status": obs.get("status"),
    }


def close_enough(a: Any, b: Any, *, abs_tol: float) -> bool:
    fa, fb = finite_or_none(a), finite_or_none(b)
    if fa is None or fb is None:
        return False
    return abs(fa - fb) < abs_tol


__all__ = [
    "cagr_of",
    "close_enough",
    "daily_from_nav",
    "dump_json",
    "feature_set_hash",
    "file_sha256",
    "finite_or_none",
    "load_json",
    "metrics_from_nav",
    "observed_from_report",
    "sha256_bytes",
    "sha256_json",
    "slim_obs",
    "to_ymd",
    "update_state",
    "write_md",
]
