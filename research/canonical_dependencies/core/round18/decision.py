"""KEEP_BASELINE_V4 或 RECOMMENDED_BASELINE_V5 + AWAITING_USER_LOCK。不自动改 CURRENT。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND18_ROOT
from .common import dump_json, load_json, write_md
from .contract import (
    A00,
    AWAITING_USER_LOCK,
    CURRENT_BASELINE,
    F2_NAME,
    KEEP_BASELINE_V4,
    MAXDD_WORSEN_CAP,
    OPTION_CANDIDATE,
    PARENT_V4,
    PERFORMANCE_CONCENTRATION_RISK,
    RECOMMENDED_BASELINE_V5,
)
from .freeze import require_frozen

PROMO_JSON = ROUND18_ROOT / "promotion" / "promotion.json"
PROMO_MD = ROUND18_ROOT / "15_baseline_decision.md"


def run_decision() -> dict[str, Any]:
    require_frozen()
    matrix = load_json(ROUND18_ROOT / "metrics" / "matrix.json")
    stress = load_json(ROUND18_ROOT / "stress" / "stress.json")
    cap = load_json(ROUND18_ROOT / "capacity" / "capacity.json")
    spa = load_json(ROUND18_ROOT / "spa" / "spa.json")
    pit = load_json(ROUND18_ROOT / "pit" / "pit.json")
    fwd = load_json(ROUND18_ROOT / "forward" / "forward.json")
    rows = [r for r in (matrix.get("rows") or []) if r.get("formal")]
    a00 = next((r for r in rows if r.get("code") == A00), {}) or {}
    failed: list[str] = []
    notes = ["R16_O01 未融合。", "D 系列不得晋级。", "CURRENT 不自动升 v5。"]
    candidates = [
        r for r in rows
        if r.get("code") != A00
        and r.get("sharpe") is not None
        and r.get("promotion_eligible")
        and r.get("status") not in {"SKIP_SCHEMA", "SKIP_PIT", "SKIP_DATA", "SKIP_COVERAGE", "EXECUTION_FAILED"}
    ]
    best = None
    if candidates:
        best = max(candidates, key=lambda r: float(r.get("sharpe") or -9))
    a00_sh = a00.get("sharpe")
    a00_dd = a00.get("max_drawdown")
    if best is None:
        failed.append("no_eligible_candidate")
    else:
        extras = set(best.get("five_extras") or []) | set(best.get("ten_extras") or [])
        if F2_NAME in extras and not (pit.get("factors") or {}).get(F2_NAME, {}).get("promote_ok"):
            failed.append("f2_pit_blocks_promotion")
        for name in extras:
            frec = (pit.get("factors") or {}).get(name) or {}
            if not frec.get("promote_ok"):
                failed.append(f"pit_or_coverage_{name}")
        if a00_sh is not None and float(best["sharpe"]) < float(a00_sh) - 1e-6:
            failed.append("sharpe_not_competitive")
        if a00_dd is not None and best.get("max_drawdown") is not None:
            if float(best["max_drawdown"]) < float(a00_dd) - MAXDD_WORSEN_CAP:
                failed.append("maxdd_worse_than_cap")
        yearly = best.get("yearly_returns") or {}
        pos = [y for y, v in yearly.items() if v is not None and float(v) > 0]
        if pos and set(pos) <= {"2024", "2025", "2026"}:
            failed.append("only_2024_2025")
            notes.append(PERFORMANCE_CONCENTRATION_RISK)
    if stress.get("sign_flip"):
        failed.append("plus20_plus30_sign_flip")
    if cap.get("sign_flip"):
        failed.append("capacity_sign_flip")
    spa_rec = (spa.get("spa") or {})
    spa_p = spa_rec.get("p_value") or spa_rec.get("spa_pvalue")
    if spa_p is not None:
        try:
            if float(spa_p) > 0.10:
                failed.append("spa_not_significant")
        except (TypeError, ValueError):
            pass
    if failed:
        four_way = KEEP_BASELINE_V4
        recommended = None
        lock = None
        final_current = CURRENT_BASELINE
    else:
        four_way = f"{RECOMMENDED_BASELINE_V5}={best.get('code')}"
        recommended = best.get("code")
        lock = AWAITING_USER_LOCK
        final_current = CURRENT_BASELINE
    payload = {
        "four_way": four_way,
        "recommended": recommended,
        "lock": lock,
        "final_current": final_current,
        "failed_gates": failed,
        "notes": notes,
        "best": best,
        "a00": a00,
        "option_untouched": OPTION_CANDIDATE,
        "parent": PARENT_V4,
        "forward": fwd.get("status"),
    }
    dump_json(PROMO_JSON, payload)
    write_md(
        PROMO_MD,
        "# 15 基线决策\n\n"
        f"**{four_way}** CURRENT=`{final_current}` lock=`{lock}`\n\n"
        f"failed=`{failed}`\n\n"
        f"best=`{(best or {}).get('code')}` Sharpe=`{(best or {}).get('sharpe')}`\n",
    )
    return payload
