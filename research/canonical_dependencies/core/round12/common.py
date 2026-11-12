"""Round12 共用：落盘、NAV、阶段目录。不改策略。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND12_ROOT, assert_project_result_path, ensure_round12_tree
from ..round11.common import (
    COMPLETE_YEARS,
    daily_from_nav,
    max_drawdown_of,
    metrics_from_nav,
    moments,
    observed_from_report,
    pct,
    rolling_cagr,
    sharpe_of,
    worst_window_return,
)


def dump_json(path: Path, payload: Any) -> Path:
    ensure_round12_tree()
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def write_md(path: Path, body: str) -> Path:
    ensure_round12_tree()
    assert_project_result_path(path)
    text = body if body.endswith("\n") else body + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def stage_dir(name: str) -> Path:
    path = ROUND12_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def daily_sr(daily: pd.Series) -> float | None:
    x = daily.dropna().astype(float)
    if len(x) < 3:
        return None
    vol = float(x.std(ddof=1))
    if vol <= 0 or not np.isfinite(vol):
        return None
    return float(x.mean() / vol)


def worst12m(daily: pd.Series) -> dict[str, Any]:
    return worst_window_return(daily, 252)


__all__ = [
    "COMPLETE_YEARS",
    "daily_from_nav",
    "daily_sr",
    "dump_json",
    "load_json",
    "max_drawdown_of",
    "metrics_from_nav",
    "moments",
    "observed_from_report",
    "pct",
    "rolling_cagr",
    "sharpe_of",
    "stage_dir",
    "worst12m",
    "worst_window_return",
    "write_md",
]
