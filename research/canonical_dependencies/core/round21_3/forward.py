"""Forward：检查 2026-07-24 之后 EOD/分钟/资金流/股本/标签/O01。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import MARKET_END, ROUND21_3_ROOT
from ..round21.contract import HERMES_V7
from .common import dump_json, write_md
from .contract import FORWARD_OOS_NOT_AVAILABLE, MD

FWD_JSON = ROUND21_3_ROOT / "forward" / "forward.json"


def _max_date(path: Path, col_candidates: tuple[str, ...] = ("trade_date", "date", "datetime")) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "max_date": None}
    try:
        import pyarrow.parquet as pq

        if path.is_dir():
            files = sorted(path.rglob("*.parquet"))[:5]
            if not files:
                return {"path": str(path), "exists": True, "max_date": None, "note": "empty dir"}
            path = files[-1]
        schema = set(pq.read_schema(path).names)
        col = next((c for c in col_candidates if c in schema), None)
        if col is None:
            return {"path": str(path), "exists": True, "max_date": None, "cols": list(schema)[:12]}
        import pandas as pd

        s = pd.read_parquet(path, columns=[col])[col]
        mx = pd.to_datetime(s, errors="coerce").max()
        return {"path": str(path), "exists": True, "max_date": None if pd.isna(mx) else str(pd.Timestamp(mx).date()), "col": col}
    except Exception as exc:
        return {"path": str(path), "exists": True, "error": str(exc)}


def run_forward(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and FWD_JSON.is_file() and MD["23"].is_file():
        from .common import load_json

        existing = load_json(FWD_JSON)
        if existing.get("status") in {FORWARD_OOS_NOT_AVAILABLE, "AVAILABLE"}:
            return existing
    cutoff = str(MARKET_END)
    checks = {
        "eod_daily_raw": _max_date(HERMES_V7 / "daily_raw"),
        "minute": _max_date(Path("/data1/Stock/data/factor_cache")),
        "money_flow": _max_date(HERMES_V7 / "daily_money_flow"),
        "share_capital": _max_date(HERMES_V7 / "share_capital", ("available_at", "effective_from", "trade_date")),
        "labels": _max_date(Path("/data1/Stock/data/first_board_hermes")),
        "o01": {"note": "O01 overlay uses cached v4 through MARKET_END", "market_end": cutoff},
    }
    missing = []
    for k, v in checks.items():
        mx = v.get("max_date") if isinstance(v, dict) else None
        if mx is None or mx <= cutoff:
            missing.append({"field": k, "max_date": mx, "need_after": cutoff})
    payload = {
        "status": FORWARD_OOS_NOT_AVAILABLE,
        "cutoff": cutoff,
        "checks": checks,
        "missing": missing,
        "attempted": True,
        "versioned_snapshot": None,
        "note": "2026-07-24 之后无独立 EOD/分钟/资金流/股本/标签/O01 Forward 快照",
    }
    dump_json(FWD_JSON, payload)
    write_md(
        MD["23"],
        "\n".join(
            [
                "# Round21.3 · 23 Forward 可用性",
                "",
                f"status=`{FORWARD_OOS_NOT_AVAILABLE}` cutoff=`{cutoff}`",
                f"缺失=`{missing}`",
                "无独立 Forward 不得 live-ready，最多 V6_RESEARCH_CANDIDATE。",
                "",
            ]
        ),
    )
    return payload
