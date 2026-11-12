"""a-pit：区分 2018 历史发布时间 vs 2026 再下载。无法证明 available_at ≤ decision → 不得 PIT_PASS。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...round16.chain import load_chain
from ...round16.factors import load_factors
from ..common import dump_json, write_md
from ..contract import PIT_TIMESTAMP_UNVERIFIED
from .support import PIT_JSON, resume_ok, skip_result, ta_md


def run_pit(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    existing = resume_ok(PIT_JSON) if resume else None
    if existing is not None and existing.get("status") not in {None, "SKIP"}:
        return existing
    try:
        fac = load_factors()
        chain = load_chain()
    except Exception as exc:
        payload = skip_result("a-pit", f"{type(exc).__name__}:{exc}")
        dump_json(PIT_JSON, payload)
        return payload
    pit16 = fac.get("pit") or {}
    unverified = True
    detail: dict[str, Any] = {
        "round16_pit": pit16,
        "historical_quote_time": "decision-time 2018–2026 日K last_close 代理",
        "redownload_time": "2026 research cache / AmazingData 再拉",
        "must_not_confuse": True,
    }
    if chain.empty or "available_at" not in chain.columns:
        detail["missing_available_at"] = True
        detail["cannot_prove_available_at_le_decision"] = True
    else:
        close = pd.to_datetime(chain["trade_date"]) + pd.Timedelta(hours=15)
        av = pd.to_datetime(chain["available_at"])
        detail["available_at_le_close"] = bool((av <= close + pd.Timedelta(minutes=1)).all())
        detail["decision_vs_order"] = "decision=D_close; order=D+1_open; snapshot_intraday_unverified"
    payload = {
        "status": PIT_TIMESTAMP_UNVERIFIED,
        "pit_pass": False,
        "live_ready": False,
        "rule": "available_at <= decision_timestamp < order",
        "tz": "Asia/Shanghai",
        "entry_unchanged": True,
        "same_open_execution_optimism": True,
        "detail": detail,
        "note": "无法证明 available_at <= decision_timestamp，不得写 PIT_PASS / LIVE_READY。",
    }
    dump_json(PIT_JSON, payload)
    write_md(
        ta_md("pit.md"),
        "\n".join(
            [
                "# Round19 Track A · a-pit",
                "",
                f"status=`{PIT_TIMESTAMP_UNVERIFIED}`。pit_pass=False live_ready=False。",
                "2018 历史发布时间 ≠ 2026 再下载时间。日 K 无法核实盘中 snapshot。",
                "",
            ]
        ),
    )
    return payload
