"""Round 22.3 evaluate_factor。未知因子不伪造 PASS。"""

from __future__ import annotations

from typing import Any

from .common import dump_json, load_json
from .contract import (
    AUTO_PROMOTE_CURRENT,
    C1_LAG1_ID,
    C4_ID,
    FACTORY_STEPS_EXTRA,
    HN2_ID,
    KEEP_BASELINE_V5,
    NEW_FACTOR_SPECS,
    NOT_RUN,
    ROUND22_3_ROOT,
    V5_ANCHOR,
)

FACTORY_CORE = (
    "FactorDefinition",
    "DataAudit",
    "PIT",
    "IC",
    "Rolling",
    "Shape",
    "TopTail",
    "Redundancy",
    "ConditionalIC",
    "Combination",
    "ModelIncrementality",
    "PortfolioEvidence",
    "PromotionGate",
)
MANDATORY_NO_SKIP = frozenset({"Rolling", "Shape", "TopTail", "ConditionalIC", "IC", "PIT", "DataAudit"})
KNOWN = {C1_LAG1_ID, C4_ID, HN2_ID, "r22_c1_xl_net_imbalance", "r22_3_fs1_accrual_minus_ocf", "r22_3_fs2_ocf_over_assets"}


def load_definition(factor_id: str) -> dict[str, Any]:
    for spec in NEW_FACTOR_SPECS:
        if spec["factor_id"] == factor_id:
            return dict(spec)
    if factor_id in KNOWN or factor_id.startswith("r22_"):
        return {"factor_id": factor_id, "round": "22.3"}
    raise KeyError(factor_id)


def evaluate_factor(factor_id: str, *, through: str = "PromotionGate", persist: bool = False) -> dict[str, Any]:
    definition = load_definition(factor_id)
    dest = ROUND22_3_ROOT / "factors" / factor_id
    payload: dict[str, Any] = {
        "factor_id": factor_id,
        "step": through,
        "factory_steps": list(FACTORY_CORE) + list(FACTORY_STEPS_EXTRA),
        "definition": definition,
        "copied_global_result": False,
        "round": "22.3",
        "auto_promote": AUTO_PROMOTE_CURRENT,
        "keep_if_fail": KEEP_BASELINE_V5,
        "v5_anchor": dict(V5_ANCHOR),
        "silent_skip_forbidden": list(MANDATORY_NO_SKIP),
        "compare_to": "R16_O01",
    }
    pit = load_json(dest / "pit.json")
    ic = load_json(dest / "ic.json")
    payload["PIT"] = {"status": pit.get("status") or pit.get("pit_status") or ("MISSING" if not pit else "OK"), "body": pit}
    payload["IC"] = {"status": ic.get("status") or ("MISSING" if not ic else "OK")}
    payload["DataAudit"] = {"status": "OK" if (ROUND22_3_ROOT / "data_recovery").exists() else "MISSING"}
    for step in ("Rolling", "Shape", "TopTail", "ConditionalIC"):
        path = dest / f"{step.lower()}.json"
        if step == "ConditionalIC":
            path = dest / "conditional.json"
        rec = load_json(path)
        if rec:
            payload[step] = {"status": rec.get("status") or "OK"}
        else:
            payload[step] = {"status": "MISSING" if step in MANDATORY_NO_SKIP else NOT_RUN}
    promo = {"passed": False, "status": KEEP_BASELINE_V5, "reason": "default keep v5; no live-ready; compare full-history vs R16_O01 only"}
    payload["PromotionGate"] = promo
    payload["status"] = promo["status"]
    if persist:
        dump_json(dest / "evaluate.json", payload)
        dump_json(dest / "promotion.json", promo)
    return payload
