"""Round 21.3 evaluate_factor：读 result/round21_3/factors/<id>。未知因子不伪造 PASS。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_3_ROOT
from .common import dump_json, load_json
from .contract import (
    AUTO_PROMOTE_CURRENT,
    FACTORY_STEPS_EXTRA,
    KEEP_BASELINE_V5,
    NEW_FACTOR_IDS,
    V5_ANCHOR,
)
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
    dest = ROUND21_3_ROOT / "factors" / factor_id
    payload: dict[str, Any] = {
        "factor_id": factor_id,
        "step": through,
        "factory_steps": list(FACTORY_CORE) + list(FACTORY_STEPS_EXTRA),
        "definition": definition,
        "copied_global_result": False,
        "round": "21.3",
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
        "DATA_RECOVERY_BEFORE_COVERAGE_REJECT": ROUND21_3_ROOT / "data_recovery" / "recovery.json",
        "FORMULA_EQUIVALENCE_CHECK": ROUND21_3_ROOT / "construct" / "formula_equivalence.csv",
        "TRUE_DELETE_ASSERTION": ROUND21_3_ROOT / "model" / "l1_train.json",
        "FEE_LEDGER_COMPLETENESS": ROUND21_3_ROOT / "backtest" / "l1_fee_corrected.csv",
        "BOOTSTRAP_SPA_NUMERICAL": ROUND21_3_ROOT / "stats" / "stats.json",
        "SEED_STATUS_ENUM": ROUND21_3_ROOT / "seed" / "seed_robustness.json",
        "ECONOMIC_MATERIALITY_GATE": ROUND21_3_ROOT / "round21_3_manifest.yaml",
        "FORWARD_AVAILABILITY": ROUND21_3_ROOT / "forward" / "forward.json",
        "NEW_VS_BASELINE_CORRELATION": ROUND21_3_ROOT / "correlation" / "correlation.json",
        "NEAREST_LEGACY_FACTOR_IDENTIFICATION": ROUND21_3_ROOT / "correlation" / "fold_nearest.json",
        "FOUR_WAY_REPLACEMENT_TEST": ROUND21_3_ROOT / "attribution" / "attribution.json",
        "CLUSTER_REPLACEMENT_IF_APPLICABLE": dest / "cluster.json",
        "OOF_INCREMENTALITY": dest / "incrementality.json",
        "FULL_PORTFOLIO_INCREMENTALITY": dest / "portfolio.json",
        "REPLACE_HASH_INTEGRITY": dest / "replace_hash.json",
    }
    last_status = "OK"
    if through == "FactorDefinition":
        payload["status"] = "DEFINITION_ONLY"
        payload["values_constructed"] = (dest / "values.parquet").is_file()
        return payload
    for step, path in files.items():
        if path.is_file():
            payload[step] = load_json(path)
        else:
            kind = "SKIPPED" if step in SKIP_IF_MISSING else "MISSING"
            payload[step] = {
                "status": kind,
                "reason": f"{step} artifact not written for {factor_id}; refuse fake PASS",
            }
        if step == through:
            last_status = "OK" if path.is_file() else payload[step]["status"]
            break
    extras = {}
    for step, path in extra_files.items():
        if path.is_file():
            extras[step] = load_json(path) if path.suffix == ".json" else {"status": "OK", "path": str(path)}
        else:
            extras[step] = {"status": "SKIPPED", "reason": f"21.3 {step} not written; refuse fake PASS"}
    payload["round21_3"] = extras
    if through == "PromotionGate":
        pit = payload.get("PIT") or {}
        pit_status = pit.get("pit_status") or pit.get("status")
        port = payload.get("PortfolioEvidence") or {}
        rows = port.get("rows") or []
        scored = [r for r in rows if r.get("sharpe") is not None]
        best = max(scored, key=lambda r: float(r["sharpe"])) if scored else None
        v5_sh = float(V5_ANCHOR["sharpe"])
        better = bool(best and best.get("sharpe") is not None and float(best["sharpe"]) > v5_sh + 1e-12)
        if pit_status != "STRICT_PIT":
            status, reason, passed = "SKIPPED", f"PIT={pit_status}; refuse fake PASS", False
        elif not rows:
            status, reason, passed = "SKIPPED", "no full-engine row; refuse fake PASS", False
        elif not better:
            status, reason, passed = "FAIL", f"not better than v5 Sharpe={v5_sh}", False
        else:
            status, reason, passed = KEEP_BASELINE_V5, "stat better but economic gate pending report", False
        gate = {
            "factor_id": factor_id,
            "status": status,
            "reason": reason,
            "passed": passed,
            "decision": KEEP_BASELINE_V5,
            "auto_promote": AUTO_PROMOTE_CURRENT,
            "best": None if best is None else {"code": best.get("code"), "sharpe": best.get("sharpe")},
            "pit_status": pit_status,
        }
        payload["PromotionGate"] = gate
        last_status = status
        if persist:
            dump_json(dest / "promotion.json", gate)
    if through in extras:
        last_status = extras[through].get("status") or last_status
    payload["status"] = last_status
    payload["in_new_factor_ids"] = factor_id in NEW_FACTOR_IDS
    return payload
