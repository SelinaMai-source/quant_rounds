"""a-match：C3 活动匹配诊断。未过 ACTIVITY_MATCHING_PASS 不得比 Sharpe。"""

from __future__ import annotations

from typing import Any

from ..common import dump_json, load_json, write_md
from ..contract import ACTIVITY_MATCHING_PASS, EX_POST_DIAGNOSTIC_ONLY
from .support import EQ_JSON, EVENTS_JSON, MATCH_JSON, resume_ok, skip_result, ta_md


def run_match(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    existing = resume_ok(MATCH_JSON) if resume else None
    if existing is not None:
        return existing
    eq = load_json(EQ_JSON)
    events = load_json(EVENTS_JSON)
    if not eq:
        payload = skip_result("a-match", "EQUITY_MISSING", status="BLOCKED")
        dump_json(MATCH_JSON, payload)
        return payload
    c3 = eq.get("c3") or {}
    c2 = eq.get("c2") or {}
    status = c3.get("matching_status")
    passed = bool(c3.get("activity_matched") and status == ACTIVITY_MATCHING_PASS)
    fill_n = len(events.get("fill_change_dates") or [])
    payload = {
        "status": "DONE",
        "activity_matching_pass": passed,
        "matching_status": status,
        "c3_n_veto": c3.get("n_veto"),
        "c3_n_o01_eligible": c3.get("n_o01_eligible"),
        "year_match": c3.get("year_match"),
        "cell_match": c3.get("cell_match"),
        "budget_ok": c3.get("budget_ok"),
        "layer": c3.get("layer"),
        "by_year": c3.get("by_year"),
        "by_cell": c3.get("by_cell"),
        "c2_flag": c2.get("activity_flag"),
        "c2_jaccard": c2.get("jaccard"),
        "sharpe_comparison_allowed": passed,
        "ex_post_fill_n": fill_n,
        "ex_post_fill_match_label": EX_POST_DIAGNOSTIC_ONLY,
        "no_future_pnl_or_fill_selection": True,
        "c3_is_control_not_candidate": True,
        "note": "匹配对象=决策时已知 eligible Top2 + 年频处理次数 + year×B01A + 计划 Top2 veto 次数 + 计划风险预算。事后 fill 只标 EX_POST_DIAGNOSTIC_ONLY。",
    }
    dump_json(MATCH_JSON, payload)
    write_md(
        ta_md("match.md"),
        "\n".join(
            [
                "# Round19 Track A · a-match",
                "",
                f"C3 `{status}` PASS=`{passed}` year=`{c3.get('year_match')}` cell=`{c3.get('cell_match')}` budget_ok=`{c3.get('budget_ok')}`。",
                f"Sharpe 比较允许=`{passed}`。C2 flag=`{c2.get('activity_flag')}` Jaccard=`{c2.get('jaccard')}`。",
                f"事后 fill N=`{fill_n}` 标签=`{EX_POST_DIAGNOSTIC_ONLY}`。C3 不得晋级。",
                "",
            ]
        ),
    )
    return payload
