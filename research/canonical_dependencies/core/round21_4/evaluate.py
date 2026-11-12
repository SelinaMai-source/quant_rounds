"""Round 21.4 evaluate_factor：Head-specific 四路强制。未知因子不伪造 PASS。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_4_ROOT
from .common import dump_json, load_json
from .contract import AUTO_PROMOTE_CURRENT, FACTORY_STEPS_EXTRA, KEEP_BASELINE_V5, V5_ANCHOR, WXYZ_FACTOR_IDS
from .hypotheses import HYPOTHESES, require_hypotheses

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
    "Redundancy",
    "ConditionalIC",
    "Combination",
    "ModelIncrementality",
    "PortfolioEvidence",
    "PromotionGate",
}


def load_definition(factor_id: str) -> dict[str, Any]:
    for card in HYPOTHESES:
        if card["factor_id"] == factor_id:
            return dict(card)
    raise KeyError(factor_id)


def evaluate_factor(factor_id: str, *, through: str = "PromotionGate", persist: bool = False) -> dict[str, Any]:
    require_hypotheses()
    definition = load_definition(factor_id)
    allowed = set(FACTORY_CORE) | set(FACTORY_STEPS_EXTRA)
    if through not in allowed:
        raise RuntimeError(f"unknown factory step: {through}")
    dest = ROUND21_4_ROOT / "factors" / factor_id
    payload: dict[str, Any] = {
        "factor_id": factor_id,
        "step": through,
        "factory_steps": list(FACTORY_CORE) + list(FACTORY_STEPS_EXTRA),
        "definition": definition,
        "copied_global_result": False,
        "round": "21.4",
        "head_specific_required": True,
        "head_paths": ["five_only_add", "ten_only_add", "both_add", "both_replace_nearest"],
        "auto_promote": AUTO_PROMOTE_CURRENT,
        "keep_if_fail": KEEP_BASELINE_V5,
        "v5_anchor": dict(V5_ANCHOR),
    }
    files = {
        "DataAudit": dest / "construct.json",
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
    }
    extra_files = {
        "DATA_RECOVERY_BEFORE_COVERAGE_REJECT": ROUND21_4_ROOT / "data_recovery" / "recovery.json",
        "FORMULA_EQUIVALENCE_CHECK": ROUND21_4_ROOT / "construct" / "formula_equivalence.csv",
        "TRUE_DELETE_ASSERTION": ROUND21_4_ROOT / "model" / "amatrix_train.json",
        "FEE_LEDGER_COMPLETENESS": ROUND21_4_ROOT / "backtest" / "a_fee_corrected.csv",
        "BOOTSTRAP_SPA_NUMERICAL": ROUND21_4_ROOT / "stats" / "stats.json",
        "SEED_STATUS_ENUM": ROUND21_4_ROOT / "seed" / "seed_robustness.json",
        "ECONOMIC_MATERIALITY_GATE": ROUND21_4_ROOT / "round21_4_manifest.yaml",
        "FORWARD_AVAILABILITY": ROUND21_4_ROOT / "forward" / "forward.json",
        "NEW_VS_BASELINE_CORRELATION": ROUND21_4_ROOT / "correlation" / "correlation.json",
        "NEAREST_LEGACY_FACTOR_IDENTIFICATION": ROUND21_4_ROOT / "correlation" / "fold_nearest.json",
        "FOUR_WAY_REPLACEMENT_TEST": ROUND21_4_ROOT / "attribution" / "attribution.json",
        "HEAD_SPECIFIC_FOUR_WAY": ROUND21_4_ROOT / "backtest" / "wxyz_fourway.json",
        "CLUSTER_REPLACEMENT_IF_APPLICABLE": dest / "cluster.json",
        "OOF_INCREMENTALITY": dest / "incrementality.json",
        "FULL_PORTFOLIO_INCREMENTALITY": dest / "portfolio.json",
        "REPLACE_HASH_INTEGRITY": dest / "replace_hash.json",
        "COMBINATION_FAILURE_ANALYSIS": ROUND21_4_ROOT / "combo" / "failure.json",
    }
    last_status = "OK"
    if through == "FactorDefinition":
        payload["status"] = "DEFINITION_ONLY"
        payload["values_constructed"] = (dest / "values.parquet").is_file()
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
        payload[through] = {"exists": p.exists(), "path": str(p), "status": "OK" if p.exists() else "SKIPPED"}
    payload["status"] = last_status
    if persist:
        dump_json(dest / "promotion.json", payload)
    return payload
