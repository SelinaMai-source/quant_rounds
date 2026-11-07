"""对照权威 ROUND7_FLAT_5BP 产物做数字 parity。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contract import V5_ANCHOR

ROOT_CANDIDATES = [
    Path(__file__).resolve().parents[2],
    Path("/data1/stock-mxy/课题/一进二策略"),
]


def _find(rel: str) -> Path | None:
    for root in ROOT_CANDIDATES:
        p = root / rel
        if p.is_file():
            return p
    return None


def load_replay_summary() -> dict[str, Any]:
    path = _find("result/replay_r7_caliber/v5_R16_O01/replay_summary.json")
    if path is None:
        path = _find("result/round22_3/parity/R223_V5_R16_O01/replay_summary.json")
    if path is None:
        raise FileNotFoundError("missing authoritative replay_summary.json")
    return json.loads(path.read_text(encoding="utf-8"))


def check_parity(tol: float = 1e-9) -> dict[str, Any]:
    obs = load_replay_summary()
    checks = {}
    for key in ("sharpe", "cagr", "max_drawdown", "nav_end", "completed_trades"):
        a = V5_ANCHOR[key]
        b = obs.get(key)
        if key == "completed_trades":
            checks[key] = int(a) == int(b)
        else:
            checks[key] = abs(float(a) - float(b)) <= tol
    return {
        "status": "MATCH" if all(checks.values()) else "MISMATCH",
        "checks": checks,
        "anchor": {k: V5_ANCHOR[k] for k in checks},
        "observed": {k: obs.get(k) for k in checks},
        "fee_schedule": obs.get("fee_schedule"),
    }
