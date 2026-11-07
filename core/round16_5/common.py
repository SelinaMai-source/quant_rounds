"""Round16.5 共用：落盘、哈希。禁止写 Stock。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_5_ROOT, assert_project_result_path, ensure_round16_5_tree
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
from ..round16.common import cagr_of, pack_daily, pick_col, redact, to_ymd


def dump_json(path: Path, payload: Any) -> Path:
    ensure_round16_5_tree()
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def write_md(path: Path, body: str) -> Path:
    ensure_round16_5_tree()
    assert_project_result_path(path)
    text = body if body.endswith("\n") else body + "\n"
    path.write_text(text, encoding="utf-8")
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


def empirical_p(n_beat_or_tie: int, n: int) -> float | None:
    """(1 + #placebos beating or tying) / (1 + N). Never 0."""
    if n <= 0:
        return None
    return float(1 + int(n_beat_or_tie)) / float(1 + int(n))


def holm_adjust(pvals: dict[str, float | None]) -> dict[str, float | None]:
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    m = len(items)
    if m == 0:
        return {k: None for k in pvals}
    items.sort(key=lambda kv: kv[1])
    adj: dict[str, float | None] = {k: None for k in pvals}
    running = 0.0
    for i, (k, p) in enumerate(items, start=1):
        val = min(1.0, p * (m - i + 1))
        running = max(running, val)
        adj[k] = float(min(1.0, running))
    return adj


def nav_end(nav: pd.DataFrame) -> float | None:
    if nav is None or nav.empty or "total_asset" not in nav.columns:
        return None
    ordered = nav.copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"]).dt.normalize()
    return float(ordered.sort_values("trade_date")["total_asset"].iloc[-1])


__all__ = [
    "COMPLETE_YEARS",
    "cagr_of",
    "daily_from_nav",
    "dump_json",
    "empirical_p",
    "file_sha256",
    "holm_adjust",
    "load_json",
    "max_drawdown_of",
    "metrics_from_nav",
    "nav_end",
    "observed_from_report",
    "pack_daily",
    "pct",
    "pick_col",
    "redact",
    "rolling_cagr",
    "sharpe_of",
    "to_ymd",
    "worst_window_return",
    "write_md",
]
