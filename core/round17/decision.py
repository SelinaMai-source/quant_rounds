"""KEEP_BASELINE_V4 或 RECOMMENDED_BASELINE_V5 + AWAITING_USER_LOCK。不自动改 CURRENT。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND17_ROOT
from .common import dump_json, load_json, write_md
from .contract import (
    AWAITING_USER_LOCK,
    CURRENT_BASELINE,
    KEEP_BASELINE_V4,
    LIVE_NOT_VALIDATED,
    MAXDD_WORSEN_CAP,
    OPTION_CANDIDATE,
    PARENT_V4,
    RECOMMENDED_BASELINE_V5,
    SEED_DIVERSITY_DEGENERATE,
)
from .freeze import require_frozen

PROMO_JSON = ROUND17_ROOT / "promotion" / "promotion.json"
PROMO_MD = ROUND17_ROOT / "20_baseline_decision.md"
LIM_MD = ROUND17_ROOT / "16_limitations.md"


def run_decision() -> dict[str, Any]:
    require_frozen()
    four = load_json(ROUND17_ROOT / "fourway" / "fourway.json")
    pre = load_json(ROUND17_ROOT / "preflight" / "preflight.json")
    stress = load_json(ROUND17_ROOT / "stress" / "stress.json")
    spa = load_json(ROUND17_ROOT / "spa" / "spa.json")
    attr = load_json(ROUND17_ROOT / "attribution" / "increments.json")
    fwd = load_json(ROUND17_ROOT / "forward" / "forward.json")
    a00 = four.get("A00") or {}
    a01 = four.get("A01") or {}
    failed = []
    notes = []
    best_b02 = None
    best_sh = None
    for item in four.get("finalists") or []:
        sh = (item.get("B02") or {}).get("sharpe")
        if sh is None:
            continue
        if best_sh is None or float(sh) > best_sh:
            best_sh = float(sh)
            best_b02 = item
    if pre.get("status") == SEED_DIVERSITY_DEGENERATE:
        failed.append("seed_diversity_degenerate")
        notes.append("0.8/0.8 仍退化，不得把 A01 当 bagging alpha。")
    if stress.get("sign_flip"):
        failed.append("plus20_plus30_sign_flip")
    a00_sh = a00.get("sharpe")
    a00_dd = a00.get("max_drawdown")
    competitive = False
    dd_ok = False
    if best_b02 and a00_sh is not None and best_sh is not None:
        competitive = float(best_sh) >= float(a00_sh) - 1e-6
        bdd = (best_b02.get("B02") or {}).get("max_drawdown")
        if a00_dd is not None and bdd is not None:
            dd_ok = float(bdd) >= float(a00_dd) - MAXDD_WORSEN_CAP
        if not competitive:
            failed.append("sharpe_not_competitive")
        if not dd_ok:
            failed.append("maxdd_worse_than_cap")
    else:
        failed.append("no_best_b02")
    yearly = (best_b02.get("B02") or {}).get("yearly_returns") if best_b02 else {}
    pos_years = [y for y, v in (yearly or {}).items() if v is not None and float(v) > 0]
    if pos_years and set(pos_years) <= {"2024", "2025", "2026"}:
        failed.append("only_2024_2025")
        notes.append("PERFORMANCE_CONCENTRATION_RISK")
    notes.append("选择偏差已披露（全部 SUCCESS 进入 SPA/PBO）。")
    notes.append("R16_O01 未融合。")
    notes.append("A01 运气不是晋升理由。")
    rec_id = (best_b02 or {}).get("B02_id") if best_b02 else None
    if failed:
        four_way = KEEP_BASELINE_V4
        recommended = None
        lock = None
    else:
        four_way = f"{RECOMMENDED_BASELINE_V5}={rec_id}"
        recommended = rec_id
        lock = AWAITING_USER_LOCK
    payload = {
        "four_way": four_way,
        "recommended": recommended,
        "lock": lock,
        "final_current": CURRENT_BASELINE,
        "auto_promote": False,
        "parent": PARENT_V4,
        "option_candidate_untouched": OPTION_CANDIDATE,
        "live": LIVE_NOT_VALIDATED,
        "failed_gates": failed,
        "notes": notes,
        "a00": a00,
        "a01": a01,
        "best_b02": best_b02,
        "forward": (fwd or {}).get("status"),
        "spa": spa,
        "competitive": competitive,
        "dd_ok": dd_ok,
    }
    dump_json(PROMO_JSON, payload)
    write_md(PROMO_MD, "\n".join([
        "# 20 Baseline decision",
        "",
        f"四选一=**{four_way}** CURRENT=**{CURRENT_BASELINE}** lock=`{lock}`",
        f"failed_gates=`{failed}`",
        "未自动升 CURRENT。未修改 R16_O01。",
    ]))
    write_md(LIM_MD, "\n".join([
        "# Limitations",
        "",
        "- 发现窗 2018–2021 用于排序，全历史为 HISTORICAL_RETROSPECTIVE_AUDIT。",
        "- 2022–2023 不是独立 OOS。",
        "- Forward OOS 不可得。",
        "- 孤立 0.8/0.8 不是生产默认。",
        "- 未把期权 overlay 并入新因子模型。",
        "- 全部 5000 进入选择偏差会计，但 Top12 仍是多重检验后代。",
    ]))
    return payload
