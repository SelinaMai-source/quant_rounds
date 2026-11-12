"""c-decision：两条线独立综合。禁止 Option × New Factors。不改 CURRENT。"""

from __future__ import annotations

from typing import Any

from ..common import load_json
from ..contract import (
    A_BLOCKED_BY_PIT_OR_DATA,
    A_KEEP_O01_AS_UNCONFIRMED_CANDIDATE,
    A_RECOMMEND_LOCK_O01_AS_DEVELOPMENT_V5,
    AWAITING_USER_LOCK,
    C_BLOCKED_INSUFFICIENT_EVIDENCE,
    C_KEEP_BASELINE_V4_KEEP_O01_CANDIDATE,
    C_KEEP_BASELINE_V4_WITHDRAW_O01,
    C_RECOMMEND_O01_AS_DEVELOPMENT_V5,
    CLOSE_ROUND18_FACTOR_HYPOTHESES,
    CURRENT_BASELINE,
    FORWARD_OOS_NOT_AVAILABLE,
    OPTION_CANDIDATE,
    PARENT_V4,
    PRIOR_RESEARCH_SELECTION_EXISTS,
    R16_SPA_P,
    RECOMMENDED_BASELINE_V5,
    REPAIRED_ROUND18_CANDIDATE,
    REPEATED_HISTORICAL_TESTING,
    SAME_OPEN_EXECUTION_OPTIMISM,
    WINNERS_CURSE,
)
from ..freeze import require_frozen
from ..track_a.support import DECISION_JSON as A_DEC
from ..track_b.support import VERDICT_JSON as B_DEC
from .forward import run_forward
from .support import BIAS_JSON, BIAS_MD, DEC_JSON, DEC_MD, FWD_JSON, dump_json, resume_ok, write_md


def _selection_bias(a: dict[str, Any], fwd: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "status": "DONE",
        "label": REPEATED_HISTORICAL_TESTING,
        "winners_curse": WINNERS_CURSE,
        "prior_research_selection_exists": PRIOR_RESEARCH_SELECTION_EXISTS,
        "same_history_test_index": 3,
        "tests_of_same_history": [
            {"round": 16, "role": "family search / first selection of O01", "spa_p": R16_SPA_P},
            {"round": "16.5", "role": "confirmatory / method repair, not independent OOS", "spa_p": R16_SPA_P},
            {"round": 19, "role": "third test of the same 2018–2026 history", "spa_p": a.get("spa_this_round") or R16_SPA_P},
        ],
        "spa_p_retained": R16_SPA_P,
        "c3_c4_not_independent_oos": True,
        "c3_c4_support_is_not_forward": True,
        "forward": fwd.get("status") or FORWARD_OOS_NOT_AVAILABLE,
        "same_open_execution_optimism": SAME_OPEN_EXECUTION_OPTIMISM,
        "note": (
            "R16 在同一 2018–2026 窗上选出 O01；R16.5 与 R19 是同一历史的第 2、第 3 次检验。"
            "C3/C4 公平对照支持 ≠ 独立 forward OOS。winner's curse：过门统计仍可能高估开发窗增量。"
            "不得把 p_sharpe≈0.031 或 SPA=0.496 写成独立确认。"
        ),
    }
    dump_json(BIAS_JSON, payload)
    write_md(
        BIAS_MD,
        "\n".join(
            [
                "# Round19 Track C · 选择偏差披露",
                "",
                f"**{REPEATED_HISTORICAL_TESTING}** / **{WINNERS_CURSE}**。",
                "",
                "R19 是同一 2018–2026 历史的**第三次**检验。",
                f"Round16 family SPA p=`{R16_SPA_P}` 必须保留。",
                "C3/C4 支持不等于独立 OOS。",
                f"forward=`{payload['forward']}`。{SAME_OPEN_EXECUTION_OPTIMISM} 仍在。",
                "",
                payload["note"],
                "",
            ]
        ),
    )
    return payload


def run_decision(*, resume: bool = True, **kwargs: Any) -> dict[str, Any]:
    existing = resume_ok(DEC_JSON) if resume else None
    if existing is not None:
        return existing
    require_frozen()
    fwd = load_json(FWD_JSON) or run_forward(resume=resume, **kwargs)
    a = load_json(A_DEC)
    b = load_json(B_DEC)
    if not a:
        raise RuntimeError("TRACK_A_DECISION missing; refuse to invent")
    if not b:
        raise RuntimeError("TRACK_B_DECISION missing; refuse to invent")
    bias = _selection_bias(a, fwd)
    a_dec = a.get("TRACK_A_DECISION")
    b_dec = b.get("TRACK_B_DECISION")
    repaired = b.get("repaired_best")
    notes = [
        "Track A 与 Track B 独立：R18 bug 不使 O01 更真；O01 历史增量不叠加任何 R18 模型。",
        "禁止 Option × New Factors 联合优化。",
        "修复后的 R18 候选（本轮无）不得与 O01 竞争 v5。",
        "不得自动替换 CURRENT_ALPHA_BASELINE。",
    ]
    if repaired:
        notes.append(f"{REPAIRED_ROUND18_CANDIDATE}={repaired} 不得竞争 v5。")
    if a_dec == A_BLOCKED_BY_PIT_OR_DATA:
        decision = C_BLOCKED_INSUFFICIENT_EVIDENCE
        rec = None
        lock = None
        reason = "Track A 被 PIT/数据阻断。综合证据不足。"
    elif a_dec == A_RECOMMEND_LOCK_O01_AS_DEVELOPMENT_V5 and not a.get("failed_gates"):
        decision = C_RECOMMEND_O01_AS_DEVELOPMENT_V5
        rec = OPTION_CANDIDATE
        lock = AWAITING_USER_LOCK
        reason = (
            "Track A 预注册门均过，建议把 R16_O01 锁定为 development v5；"
            "Track B 关闭当前 F1/F2/F3 假设且无修复候选，不改变 O01 证据。"
            "CURRENT 仍是 baseline_v4，等用户锁。"
        )
    elif a_dec == A_KEEP_O01_AS_UNCONFIRMED_CANDIDATE:
        decision = C_KEEP_BASELINE_V4_KEEP_O01_CANDIDATE
        rec = OPTION_CANDIDATE
        lock = None
        reason = "O01 保持未确认研究候选。CURRENT=baseline_v4。"
    else:
        decision = C_KEEP_BASELINE_V4_WITHDRAW_O01
        rec = None
        lock = None
        reason = f"Track A 不足以推荐锁定。a={a_dec}。"
    payload = {
        "status": "DONE",
        "ROUND19_STATUS": "COMPLETE",
        "TRACK_C_DECISION": decision,
        "TRACK_A_DECISION": a_dec,
        "TRACK_B_DECISION": b_dec,
        "PIPELINE_BUG_VERDICT": b.get("PIPELINE_BUG_VERDICT"),
        "current": CURRENT_BASELINE,
        "parent": PARENT_V4,
        "auto_promote": False,
        "recommended_baseline_v5": rec if decision == C_RECOMMEND_O01_AS_DEVELOPMENT_V5 else None,
        "recommended_label": f"{RECOMMENDED_BASELINE_V5}={rec}" if rec and decision == C_RECOMMEND_O01_AS_DEVELOPMENT_V5 else None,
        "lock_flag": lock,
        "repaired_round18_candidate": repaired,
        "repaired_cannot_compete_for_v5": True,
        "no_fuse_option_x_factors": True,
        "forward": fwd.get("status") or FORWARD_OOS_NOT_AVAILABLE,
        "live_validated": False,
        "same_open_execution_optimism": SAME_OPEN_EXECUTION_OPTIMISM,
        "selection_bias": bias.get("label"),
        "spa_p_retained": R16_SPA_P,
        "reason": reason,
        "notes": notes,
        "failed_gates_a": a.get("failed_gates") or [],
        "c3_c4_not_independent_oos": True,
    }
    dump_json(DEC_JSON, payload)
    write_md(
        DEC_MD,
        "\n".join(
            [
                "# Round19 Track C · 综合决策",
                "",
                f"**{decision}** CURRENT=`{CURRENT_BASELINE}` AUTO_PROMOTE=False lock=`{lock}`。",
                f"RECOMMENDED_BASELINE_V5=`{rec}`。" if rec else "无 v5 推荐。",
                reason,
                "",
                f"Track A=`{a_dec}` Track B=`{b_dec}` repaired=`{repaired}`。",
                f"forward=`{payload['forward']}` LIVE_VALIDATED=False。",
                "",
                *([f"- {n}" for n in notes]),
                "",
            ]
        ),
    )
    return payload
