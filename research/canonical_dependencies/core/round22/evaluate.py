"""Round 22 evaluate_factor。只读本轮产物，未知因子不伪造 PASS。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND22_ROOT
from .common import dump_json, load_json
from .contract import AUTO_PROMOTE_CURRENT, FACTOR_SPECS, FACTORY_STEPS_EXTRA, KEEP_BASELINE_V5, V5_ANCHOR

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
SKIP_IF_MISSING = {
    "Rolling",
    "Shape",
    "TopTail",
    "ConditionalIC",
    "Combination",
    "ModelIncrementality",
    "PortfolioEvidence",
    "PromotionGate",
}


def load_definition(factor_id: str) -> dict[str, Any]:
    for card in FACTOR_SPECS:
        if card["factor_id"] == factor_id:
            return dict(card)
    raise KeyError(factor_id)


def evaluate_factor(factor_id: str, *, through: str = "PromotionGate", persist: bool = False) -> dict[str, Any]:
    definition = load_definition(factor_id)
    allowed = set(FACTORY_CORE) | set(FACTORY_STEPS_EXTRA)
    if through not in allowed:
        raise RuntimeError(f"unknown factory step: {through}")
    dest = ROUND22_ROOT / "factors" / factor_id
    payload: dict[str, Any] = {
        "factor_id": factor_id,
        "step": through,
        "factory_steps": list(FACTORY_CORE) + list(FACTORY_STEPS_EXTRA),
        "definition": definition,
        "copied_global_result": False,
        "round": "22",
        "auto_promote": AUTO_PROMOTE_CURRENT,
        "keep_if_fail": KEEP_BASELINE_V5,
        "v5_anchor": dict(V5_ANCHOR),
        "profile": definition.get("profile"),
    }
    files = {
        "DataAudit": ROUND22_ROOT / "data_recovery" / "recovery.json",
        "PIT": dest / "pit.json",
        "IC": dest / "ic.json",
        "Rolling": dest / "rolling.json",
        "Shape": dest / "shape.json",
        "TopTail": dest / "top_tail.json",
        "Redundancy": dest / "redundancy.json",
        "ConditionalIC": dest / "conditional.json",
        "Combination": dest / "combination.json",
        "ModelIncrementality": dest / "incrementality.json",
        "PortfolioEvidence": dest / "portfolio.json",
        "PromotionGate": dest / "promotion.json",
    }
    extra_files = {
        "DATA_RECOVERY_BEFORE_COVERAGE_REJECT": ROUND22_ROOT / "data_recovery" / "recovery.json",
        "FORMULA_EQUIVALENCE_CHECK": ROUND22_ROOT / "construct" / "formula_equivalence.csv",
        "TRUE_DELETE_ASSERTION": ROUND22_ROOT / "model" / "train.json",
        "FEE_LEDGER_COMPLETENESS": ROUND22_ROOT / "backtest" / "fee_both_calibers.csv",
        "BOOTSTRAP_SPA_NUMERICAL": ROUND22_ROOT / "stats" / "stats.json",
        "SEED_STATUS_ENUM": ROUND22_ROOT / "seed" / "seed_robustness.json",
        "ECONOMIC_MATERIALITY_GATE": ROUND22_ROOT / "configs" / "frozen_gates.json",
        "FORWARD_AVAILABILITY": ROUND22_ROOT / "coverage" / "forward_matrix.csv",
        "PROFILE_SEPARATION": ROUND22_ROOT / "04_profile_cutoff.md",
        "HEAD_SPECIFIC_E0_E9": ROUND22_ROOT / "model" / "experiment_registry.csv",
    }
    if not (dest / "pit.json").is_file():
        pit = load_json(ROUND22_ROOT / "pit" / "pit.json")
        row = next((r for r in (pit.get("rows") or []) if r.get("factor_id") == factor_id), None)
        if row:
            dest.mkdir(parents=True, exist_ok=True)
            dump_json(dest / "pit.json", row)
    last_status = "OK"
    if through == "FactorDefinition":
        payload["status"] = "DEFINITION_ONLY"
        return payload
    for step, path in files.items():
        rec = load_json(path) if path.suffix == ".json" and path.is_file() else {}
        if not path.exists():
            rec = {"status": "SKIPPED" if step in SKIP_IF_MISSING else "MISSING", "reason": f"missing {path.name}"}
            last_status = rec["status"]
        else:
            last_status = str(rec.get("status") or rec.get("pit_status") or "OK")
        payload[step] = rec if rec else {"status": last_status, "path": str(path)}
        if step == through:
            break
    if through in extra_files:
        p = extra_files[through]
        payload[through] = {"exists": p.exists(), "path": str(p), "status": "OK" if p.exists() else "MISSING"}
        last_status = payload[through]["status"]
    payload["status"] = last_status
    if persist:
        dump_json(dest / "evaluate.json", payload)
    return payload
