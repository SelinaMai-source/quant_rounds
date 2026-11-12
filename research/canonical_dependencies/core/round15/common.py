"""Round15 共用：落盘、日收益。不改策略。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND15_ROOT, assert_project_result_path, ensure_round15_tree
from ..round11.common import (
    COMPLETE_YEARS,
    daily_from_nav,
    max_drawdown_of,
    metrics_from_nav,
    observed_from_report,
    pct,
    rolling_cagr,
    sharpe_of,
    worst_window_return,
)


def dump_json(path: Path, payload: Any) -> Path:
    ensure_round15_tree()
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def write_md(path: Path, body: str) -> Path:
    ensure_round15_tree()
    assert_project_result_path(path)
    text = body if body.endswith("\n") else body + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def stage_dir(name: str) -> Path:
    path = ROUND15_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cagr_of(daily: pd.Series) -> float | None:
    x = daily.dropna().astype(float)
    if x.empty:
        return None
    wealth = float((1.0 + x).prod())
    years = max(len(x) / 252.0, 1e-9)
    return wealth ** (1.0 / years) - 1.0


def pack_daily(daily: pd.Series) -> dict[str, Any]:
    w12 = worst_window_return(daily, 252)
    return {
        "cagr": cagr_of(daily),
        "sharpe": sharpe_of(daily),
        "max_drawdown": max_drawdown_of(daily),
        "worst12m": w12.get("return"),
        "worst12m_end": w12.get("end"),
        "n_days": int(daily.dropna().shape[0]),
    }


def redact(obj: Any) -> Any:
    """递归打码 password/token，禁止落盘或打印密钥。"""
    secret = {"password", "passwd", "token", "secret", "api_key", "apikey"}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in secret or "password" in str(k).lower() or "token" in str(k).lower():
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(x) for x in obj]
    return obj


__all__ = [
    "COMPLETE_YEARS",
    "cagr_of",
    "daily_from_nav",
    "dump_json",
    "load_json",
    "max_drawdown_of",
    "metrics_from_nav",
    "observed_from_report",
    "pack_daily",
    "pct",
    "redact",
    "rolling_cagr",
    "sharpe_of",
    "stage_dir",
    "worst_window_return",
    "write_md",
]
