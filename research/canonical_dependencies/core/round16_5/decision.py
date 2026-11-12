"""四选一裁决。CURRENT 默认 KEEP_BASELINE_V4。禁止为出 v5 降门。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND16_5_ROOT
from .common import dump_json, load_json, write_md
from .contract import (
    AWAITING_USER_LOCK,
    DE_RISKING_ONLY,
    KEEP_BASELINE_V4,
    LIVE_NOT_VALIDATED,
    O01_ROUND16,
    PARENT_V4,
    PRIOR_SELECTION_EXISTS,
    PROMISING_BUT_UNCONFIRMED,
    PROMOTION_GATES,
    R16_SPA_P,
    V00,
    V01,
    V04,
)
from .attribution import load_attribution
from .equity import load_equity
from .freeze import require_frozen
from .orthogonal import load_orthogonal
from .placebo import load_placebo
from .quote import load_pit, load_quote
from .stats import load_bootstrap, load_forward, load_spa
from .stress import load_stress

PROMO_JSON = ROUND16_5_ROOT / "promotion" / "promotion.json"
PROMO_MD = ROUND16_5_ROOT / "15_baseline_decision.md"
LIM_MD = ROUND16_5_ROOT / "16_limitations.md"


def _de_risking_only(attr: dict[str, Any], eq: dict[str, Any]) -> bool:
    """增量若只是更低平均暴露且 V03 已解释，则 DE_RISKING_ONLY。"""
    inc_cagr = float(attr.get("cagr_increment") or 0)
    n_avoided = int(attr.get("n_v00_trades") or 0) - int(attr.get("n_v01_trades") or 0)
    v03_sh = ((eq.get("observed") or {}).get("sharpe"))
    v01_sh = ((attr.get("v01") or {}).get("sharpe"))
    v00_sh = ((attr.get("v00") or {}).get("sharpe"))
    if v01_sh is None or v00_sh is None:
        return False
    # V01 Sharpe 增益很小且 V03 Sharpe 不低于 V01 → 更像降风险而非期权增量
    if v03_sh is not None and float(v01_sh) <= float(v03_sh) + 1e-6 and float(v01_sh) > float(v00_sh):
        return True
    if inc_cagr <= 0:
        return False
    return False


def run_decision() -> dict[str, Any]:
    require_frozen()
    attr = load_attribution()
    placebo = load_placebo()
    eq = load_equity()
    ort = load_orthogonal()
    quote = load_quote()
    pit = load_pit()
    stress = load_stress()
    boot = load_bootstrap()
    spa = load_spa()
    fwd = load_forward()

    failed = []
    notes = []
    p = placebo.get("p_empirical_sharpe")
    n = int(placebo.get("n") or 0)
    if n <= 0:
        failed.append("placebo_n_zero")
    quotes_fail = quote.get("verdict") == "QUOTE_QUALITY_FAIL"
    if quotes_fail:
        failed.append("quote_quality_fail")
    sign_flip = (stress.get("sign_flip") or {})
    if sign_flip.get("plus20") or sign_flip.get("plus30"):
        failed.append("cost_sign_flip")

    v00 = attr.get("v00") or {}
    v01 = attr.get("v01") or {}
    v03 = eq.get("observed") or {}
    v04 = ort.get("observed") or {}
    v01_sh = v01.get("sharpe")
    v00_sh = v00.get("sharpe")
    v03_sh = v03.get("sharpe")
    v01_mdd = v01.get("max_drawdown")
    v00_mdd = v00.get("max_drawdown")
    v03_mdd = v03.get("max_drawdown")

    de_risk = _de_risking_only(attr, eq)
    if de_risk:
        failed.append(DE_RISKING_ONLY)
        notes.append("V01 相对 V00 的改善更接近降暴露/股票风险，而非期权增量。")

    beats_v03 = False
    if not eq.get("activity_matched"):
        failed.append("v03_ACTIVITY_NOT_FULLY_MATCHED")
        notes.append("V03 活动未匹配，不能把 V01 vs V03 当成已通过的活动匹配门。")
    if v01_sh is not None and v03_sh is not None:
        beats_v03 = float(v01_sh) > float(v03_sh) + 1e-12
        if not beats_v03 and v01_mdd is not None and v03_mdd is not None:
            beats_v03 = abs(float(v01_mdd)) + 1e-12 < abs(float(v03_mdd))
    if not beats_v03:
        failed.append("v01_not_gt_activity_matched_v03")
    elif not eq.get("activity_matched"):
        beats_v03 = False

    inc_not_only_exposure = True
    if int(attr.get("n_v01_trades") or 0) < int(attr.get("n_v00_trades") or 0) and (float(attr.get("cagr_increment") or 0) <= 0):
        inc_not_only_exposure = False
        failed.append("increment_only_or_negative")

    p_lt = float(PROMOTION_GATES["strengthen_requires_placebo_p_lt"])
    placebo_sig = p is not None and float(p) < p_lt
    quote_ok = quote.get("verdict") not in {None, "QUOTE_QUALITY_FAIL"}
    no_right_tail = True
    if v01.get("cagr") is not None and v00.get("cagr") is not None:
        # 粗检：V01 CAGR 不低于 V00 的 90%
        if float(v01["cagr"]) < 0.9 * float(v00["cagr"]) and float(v01["cagr"]) < float(v00["cagr"]):
            no_right_tail = False
            failed.append("right_tail_damage")

    strengthen_ok = (
        placebo_sig
        and beats_v03
        and bool(eq.get("activity_matched"))
        and (not de_risk)
        and quote_ok
        and (not quotes_fail)
        and not (sign_flip.get("plus20") or sign_flip.get("plus30"))
        and inc_not_only_exposure
        and no_right_tail
    )
    v04_better = False
    if v04.get("sharpe") is not None and v01_sh is not None:
        v04_better = float(v04["sharpe"]) > float(v01_sh) + 1e-12 and float(v04.get("cagr") or 0) > float(v01.get("cagr") or 0)

    if strengthen_ok and not v04_better:
        four = "STRENGTHEN_CANDIDATE_RECOMMENDATION"
        reason = "全引擎 placebo 经验 p 过加强门，且 V01 优于活动匹配 V03，报价未 FAIL，成本符号未翻。"
        status = "CONFIRMED_HISTORICAL_DEVELOPMENT"
        candidate = O01_ROUND16
    elif v04_better and (v04.get("sharpe") is not None) and (v00_sh is not None) and float(v04["sharpe"]) > float(v00_sh):
        four = "RECOMMEND_V04_AS_ALTERNATIVE"
        reason = "V04 正交残差在同窗 Sharpe 与 CAGR 上优于 V01。V03 不能升 v5。CURRENT 仍 v4。"
        status = PROMISING_BUT_UNCONFIRMED
        candidate = V04
    elif (v01.get("cagr") is None) or (float(attr.get("cagr_increment") or 0) <= 0 and (v01_sh is None or v00_sh is None or float(v01_sh) <= float(v00_sh))):
        four = "WITHDRAW_CANDIDATE_RECOMMENDATION"
        reason = "V01 相对 V00 无稳健增量，撤回候选推荐。"
        status = "WITHDRAWN"
        candidate = None
    else:
        four = "KEEP_CANDIDATE_RECOMMENDATION"
        reason = (
            "证据不够加强门（默认）。CURRENT=baseline_v4，CANDIDATE=R16_O01，"
            f"{PROMISING_BUT_UNCONFIRMED}。PRIOR_SELECTION_EXISTS=TRUE，Round16 SPA p={R16_SPA_P}。"
        )
        status = PROMISING_BUT_UNCONFIRMED
        candidate = O01_ROUND16
        if failed:
            reason += f" 未过门：{failed}。"

    payload = {
        "four_way": four,
        "decision": KEEP_BASELINE_V4,
        "final_current": KEEP_BASELINE_V4,
        "parent": PARENT_V4,
        "candidate": candidate,
        "lock_status": AWAITING_USER_LOCK if candidate else None,
        "status": status,
        "live": LIVE_NOT_VALIDATED,
        "auto_promote_v5": False,
        "reason": reason,
        "failed_gates": failed,
        "notes": notes,
        "PRIOR_SELECTION_EXISTS": PRIOR_SELECTION_EXISTS,
        "round16_spa_p": R16_SPA_P,
        "placebo_n": n,
        "placebo_p_sharpe": p,
        "quote_verdict": quote.get("verdict"),
        "pit": pit.get("status"),
        "forward": fwd.get("status"),
        "de_risking_only": de_risk,
        "beats_v03": beats_v03,
        "v04_better_than_v01": v04_better,
        "sign_flip": sign_flip,
        "gates_frozen": dict(PROMOTION_GATES),
        "spa_this_round": (spa.get("spa") or {}).get("p_spa"),
    }
    dump_json(PROMO_JSON, payload)
    write_md(
        PROMO_MD,
        "\n".join([
            "# Round 16.5 · 15 裁决",
            "",
            f"四选一：**{four}**",
            f"CURRENT=**{KEEP_BASELINE_V4}** CANDIDATE=`{candidate}` LIVE=`{LIVE_NOT_VALIDATED}` status=`{status}`",
            f"{reason}",
            "",
            "禁止为出 v5 降门。不自动升 CURRENT。",
            "",
        ]),
    )
    write_md(
        LIM_MD,
        "\n".join([
            "# Round 16.5 · 16 限制",
            "",
            f"- PRIOR_SELECTION_EXISTS=`{PRIOR_SELECTION_EXISTS}`；2018–2026 重复检验不是独立确认。",
            f"- Round16 SPA p=`{R16_SPA_P}` 保留。",
            f"- IV=`last_close_only`；bid/ask=`{quote.get('bid_ask')}`。",
            f"- PIT=`{pit.get('status')}`。",
            f"- Forward=`{fwd.get('status')}` LIVE=`{LIVE_NOT_VALIDATED}`。",
            "- V02/V03 不是候选。",
            "- 权威增量=FULL_ENGINE_NAV_DIFFERENCE。",
            "",
        ]),
    )
    return payload


def load_decision() -> dict[str, Any]:
    return load_json(PROMO_JSON)
