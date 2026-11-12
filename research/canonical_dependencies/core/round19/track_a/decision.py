"""a-decision：Track A 预注册门。不自动升 CURRENT。C3/C4 不得晋级。"""

from __future__ import annotations

from typing import Any

from ..common import dump_json, load_json, write_md
from ..contract import (
    A_BLOCKED_BY_PIT_OR_DATA,
    A_KEEP_O01_AS_UNCONFIRMED_CANDIDATE,
    A_RECOMMEND_LOCK_O01_AS_DEVELOPMENT_V5,
    ACTIVITY_MATCHING_PASS,
    AWAITING_USER_LOCK,
    CURRENT_BASELINE,
    LAST_PRICE_PROXY_ONLY,
    PIT_TIMESTAMP_UNVERIFIED,
    R16_SPA_P,
)
from .support import (
    ATTR_JSON,
    CONC_JSON,
    DECISION_JSON,
    EQ_JSON,
    EVENTS_JSON,
    MATCH_JSON,
    PIT_JSON,
    PLACEBO_JSON,
    PRED_JSON,
    QUOTE_JSON,
    REPLAY_JSON,
    STAT_JSON,
    resume_ok,
    ta_md,
)


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    existing = resume_ok(DECISION_JSON) if resume else None
    if existing is not None:
        return existing
    replay = load_json(REPLAY_JSON)
    events = load_json(EVENTS_JSON)
    attr = load_json(ATTR_JSON)
    conc = load_json(CONC_JSON)
    eq = load_json(EQ_JSON)
    match = load_json(MATCH_JSON)
    placebo = load_json(PLACEBO_JSON)
    pit = load_json(PIT_JSON)
    quote = load_json(QUOTE_JSON)
    pred = load_json(PRED_JSON)
    stats = load_json(STAT_JSON)

    failed: list[str] = []
    notes: list[str] = []
    blockers: list[str] = []

    c0_ok = bool((replay.get("c0") or {}).get("passed"))
    c1_ok = bool((replay.get("c1") or {}).get("passed"))
    if not replay:
        blockers.append("a-replay missing")
    if not c0_ok:
        failed.append("c0_parity")
    if not c1_ok:
        failed.append("c1_exact_parity")
    if not events.get("passed"):
        failed.append("event_layers")
        notes.append(f"event counts={events.get('counts')}")
    if events.get("layer_mix_fail"):
        failed.append("layer_mix")

    c3 = eq.get("c3") or {}
    match_pass = bool(match.get("activity_matching_pass") or c3.get("matching_status") == ACTIVITY_MATCHING_PASS)
    if not match_pass:
        failed.append("c3_ACTIVITY_MATCHING_NOT_PASS")
        notes.append("C3 未过 ACTIVITY_MATCHING_PASS，不得用其 Sharpe 作晋级门。")

    c1_obs = (replay.get("c1") or {}).get("observed") or attr.get("c1") or {}
    c0_obs = (replay.get("c0") or {}).get("observed") or attr.get("c0") or {}
    c3_obs = c3.get("observed") or {}
    c1_sh = c1_obs.get("sharpe")
    c3_sh = c3_obs.get("sharpe")
    beats_c3 = False
    if match_pass and c1_sh is not None and c3_sh is not None:
        beats_c3 = float(c1_sh) > float(c3_sh) + 1e-12
        if not beats_c3:
            failed.append("o01_not_gt_matched_c3")
    elif match_pass:
        failed.append("c3_sharpe_unavailable")

    p = placebo.get("p_empirical_sharpe")
    n = int(placebo.get("n") or 0)
    p_supportive = p is not None and n >= 1000 and float(p) < 0.10
    if n < 1000:
        failed.append("c4_incomplete")
        if placebo.get("status") in {"SKIP", "BLOCKED", None, ""}:
            blockers.append("C4 placebo missing/incomplete")
    if not p_supportive:
        failed.append("c4_p_not_supportive")
        notes.append("不得因历史 p≈0.031 单独宣称稳定 Alpha。")

    if conc.get("depends_on_few_extremes"):
        failed.append("loo_depends_few_extremes")
    if conc.get("status") not in {"DONE"}:
        failed.append("concentration_incomplete")

    sign_flip = stats.get("stress_sign_flip") or {}
    if sign_flip.get("plus20") or sign_flip.get("plus30"):
        failed.append("cost_sign_flip")

    pit_status = pit.get("status")
    if pit_status == PIT_TIMESTAMP_UNVERIFIED or pit.get("pit_pass") is False:
        notes.append("PIT 未证明 available_at<=decision，不得写 PIT_PASS / LIVE_READY。")
    if pit.get("status") in {"SKIP", "BLOCKED"}:
        blockers.append("PIT data")

    quote_v = quote.get("verdict") or LAST_PRICE_PROXY_ONLY
    if quote_v == "QUOTE_QUALITY_FAIL":
        failed.append("quote_quality_fail")
    notes.append(f"quote={quote_v}；无完整 IV 曲面。")
    notes.append(f"Round16 family SPA p={R16_SPA_P} 保留。R19 是同一历史第三次检验。PRIOR_SELECTION_EXISTS。")
    notes.append("C3/C4 不得晋级。不与 Round18 因子融合。不自动升 CURRENT。")

    core_missing = any(x.get("status") in {"SKIP", "BLOCKED"} for x in (replay, events, attr) if x)
    if core_missing and (not c1_ok):
        decision = A_BLOCKED_BY_PIT_OR_DATA
        reason = "核心回放/事件数据不可得或 C1 未能精确重放。"
    elif (
        c0_ok
        and c1_ok
        and events.get("passed")
        and match_pass
        and beats_c3
        and p_supportive
        and not conc.get("depends_on_few_extremes")
        and not (sign_flip.get("plus20") or sign_flip.get("plus30"))
        and quote_v != "QUOTE_QUALITY_FAIL"
    ):
        decision = A_RECOMMEND_LOCK_O01_AS_DEVELOPMENT_V5
        reason = (
            "预注册门均满足：C0/C1 parity、事件账可重建、C3 ACTIVITY_MATCHING_PASS 且 O01 更优、"
            "C4 p 具支持性、LOO 不靠少数极端日、+20/+30bp 不翻转。仍须披露 SPA 0.496 与三次检验选择偏差。"
            "只建议锁定为 development v5，不改 CURRENT。"
        )
    else:
        decision = A_KEEP_O01_AS_UNCONFIRMED_CANDIDATE
        reason = "证据不够预注册晋级门。O01 保持未确认研究候选。CURRENT=baseline_v4。"
        if failed:
            reason += f" 未过门：{failed}。"

    lock = AWAITING_USER_LOCK if decision == A_RECOMMEND_LOCK_O01_AS_DEVELOPMENT_V5 else None
    payload = {
        "status": "DONE",
        "TRACK_A_DECISION": decision,
        "lock_flag": lock,
        "current": CURRENT_BASELINE,
        "auto_promote": False,
        "reason": reason,
        "failed_gates": failed,
        "blockers": blockers,
        "notes": notes,
        "c0_passed": c0_ok,
        "c1_passed": c1_ok,
        "c1_observed": c1_obs,
        "c0_observed": c0_obs,
        "event_counts": events.get("counts"),
        "nav_end_difference": attr.get("nav_end_difference"),
        "cagr_increment": attr.get("cagr_increment"),
        "sharpe_increment": attr.get("sharpe_increment"),
        "concentration_depends_few": conc.get("depends_on_few_extremes"),
        "leave_top3": (conc.get("leave_top3") or {}),
        "c3_matching_status": c3.get("matching_status"),
        "c3_sharpe": c3_sh,
        "c1_sharpe": c1_sh,
        "beats_c3": beats_c3,
        "c4_p_sharpe": p,
        "c4_n": n,
        "pit": pit_status,
        "quote": quote_v,
        "prediction_increment": (pred.get("increment") if pred else None),
        "spa_this_round": ((stats.get("spa") or {}).get("p_spa") if stats else None),
        "spa_round16_retained": R16_SPA_P,
        "sign_flip": sign_flip,
        "c3_c4_not_promotion_eligible": True,
    }
    dump_json(DECISION_JSON, payload)
    write_md(
        ta_md("decision.md"),
        "\n".join(
            [
                "# Round19 Track A · a-decision",
                "",
                f"**{decision}** CURRENT=`{CURRENT_BASELINE}` AUTO_PROMOTE=False lock=`{lock}`。",
                reason,
                f"C4 p=`{p}` SPA16=`{R16_SPA_P}` C3 match=`{c3.get('matching_status')}` NAVΔ=`{attr.get('nav_end_difference')}`。",
                "",
            ]
        ),
    )
    _print_block(payload)
    return payload


def _print_block(p: dict[str, Any]) -> None:
    print("", flush=True)
    print("===== ROUND19 TRACK A =====", flush=True)
    print(f"TRACK_A_DECISION={p.get('TRACK_A_DECISION')} CURRENT={p.get('current')} AUTO_PROMOTE=False", flush=True)
    c0 = p.get("c0_observed") or {}
    c1 = p.get("c1_observed") or {}
    print(f"C0 parity={p.get('c0_passed')} CAGR={c0.get('cagr')} Sharpe={c0.get('sharpe')} trades={c0.get('completed_trades')}", flush=True)
    print(f"C1 parity={p.get('c1_passed')} CAGR={c1.get('cagr')} Sharpe={c1.get('sharpe')} MaxDD={c1.get('max_drawdown')} trades={c1.get('completed_trades')}", flush=True)
    print(f"event_counts={p.get('event_counts')}", flush=True)
    print(f"NAV_delta={p.get('nav_end_difference')} CAGR_inc={p.get('cagr_increment')} Sharpe_inc={p.get('sharpe_increment')}", flush=True)
    print(f"concentration_depends_few={p.get('concentration_depends_few')} leave_top3={p.get('leave_top3')}", flush=True)
    print(f"C3 matching={p.get('c3_matching_status')} C3_Sharpe={p.get('c3_sharpe')} beats_c3={p.get('beats_c3')}", flush=True)
    print(f"C4 p_sharpe={p.get('c4_p_sharpe')} n={p.get('c4_n')}", flush=True)
    print(f"PIT={p.get('pit')} quote={p.get('quote')} pred_inc={p.get('prediction_increment')}", flush=True)
    print(f"SPA_this={p.get('spa_this_round')} SPA16={p.get('spa_round16_retained')} sign_flip={p.get('sign_flip')}", flush=True)
    print(f"blockers={p.get('blockers')} failed_gates={p.get('failed_gates')}", flush=True)
    print(f"key_paths decision={DECISION_JSON}", flush=True)
    print("===== END TRACK A =====", flush=True)
