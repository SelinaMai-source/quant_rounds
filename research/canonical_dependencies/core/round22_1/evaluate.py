"""Round 22.1 evaluate_factor。mandatory 阶段禁止 SKIP_IF_MISSING。未知因子不伪造 PASS。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND22_1_ROOT, ROUND22_ROOT
from .common import dump_json, load_json
from .contract import (
    AUTO_PROMOTE_CURRENT,
    FACTOR_SPECS,
    FACTORY_STEPS_EXTRA,
    FULL_EVIDENCE_COMPLETE,
    KEEP_BASELINE_V5,
    NOT_RUN,
    PIT_PASS_IDS,
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
# 禁止对 mandatory 证据阶段静默 skip。缺文件必须 MISSING / NOT_RUN，不得 SKIPPED。
MANDATORY_NO_SKIP = frozenset({"Rolling", "Shape", "TopTail", "ConditionalIC", "IC", "PIT", "DataAudit"})
SKIP_IF_MISSING = {
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


def _status_of(path, rec: dict[str, Any], step: str) -> str:
    if path.exists():
        return str(rec.get("status") or rec.get("pit_status") or "OK")
    if step in MANDATORY_NO_SKIP:
        return "MISSING"
    if step in SKIP_IF_MISSING:
        return "SKIPPED"
    return "MISSING"


def evaluate_factor(factor_id: str, *, through: str = "PromotionGate", persist: bool = False) -> dict[str, Any]:
    definition = load_definition(factor_id)
    allowed = set(FACTORY_CORE) | set(FACTORY_STEPS_EXTRA)
    if through not in allowed:
        raise RuntimeError(f"unknown factory step: {through}")
    dest = ROUND22_1_ROOT / "factors" / factor_id
    payload: dict[str, Any] = {
        "factor_id": factor_id,
        "step": through,
        "factory_steps": list(FACTORY_CORE) + list(FACTORY_STEPS_EXTRA),
        "definition": definition,
        "copied_global_result": False,
        "round": "22.1",
        "auto_promote": AUTO_PROMOTE_CURRENT,
        "keep_if_fail": KEEP_BASELINE_V5,
        "v5_anchor": dict(V5_ANCHOR),
        "profile": definition.get("profile"),
        "silent_skip_forbidden": list(MANDATORY_NO_SKIP),
    }
    files = {
        "DataAudit": ROUND22_1_ROOT / "data_recovery" / "recovery.json",
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
        "IC_UNIVERSE_AUDIT": ROUND22_1_ROOT / "ic" / "ic_universe_audit.csv",
        "MANDATORY_ROLLING_CONDITIONAL_SHAPE": dest / "rolling.json",
        "TOP1_TOP2_LEDGER_TRACE": ROUND22_1_ROOT / "ledger" / "ledger.json",
        "DATA_RECOVERY_BEFORE_COVERAGE_REJECT": ROUND22_1_ROOT / "data_recovery" / "recovery.json",
        "FORMULA_IDENTITY_PROVENANCE": ROUND22_1_ROOT / "diagnostics" / "formula_provenance.json",
        "FEE_LEDGER_COMPLETENESS": ROUND22_1_ROOT / "ledger" / "fee_both_calibers.csv",
        "BOOTSTRAP_SPA_NUMERICAL": ROUND22_1_ROOT / "stats" / "stats.json",
        "SEED_STATUS_ENUM": ROUND22_1_ROOT / "seed" / "seed_robustness.json",
        "ECONOMIC_MATERIALITY_GATE": ROUND22_1_ROOT / "configs" / "frozen_gates.json",
        "FORWARD_AVAILABILITY": ROUND22_1_ROOT / "coverage" / "forward.json",
        "NO_SILENT_SKIP": dest / "evaluate.json",
    }
    if not (dest / "pit.json").is_file():
        pit = load_json(ROUND22_1_ROOT / "pit" / "pit.json") or load_json(ROUND22_ROOT / "pit" / "pit.json")
        row = next((r for r in (pit.get("rows") or []) if r.get("factor_id") == factor_id), None)
        if row:
            dest.mkdir(parents=True, exist_ok=True)
            dump_json(dest / "pit.json", row)
    last_status = "OK"
    if through == "FactorDefinition":
        payload["status"] = "DEFINITION_ONLY"
        return payload
    missing_mandatory: list[str] = []
    for step, path in files.items():
        rec = load_json(path) if path.suffix == ".json" and path.is_file() else {}
        if not path.exists():
            rec = {"status": _status_of(path, {}, step), "reason": f"missing {path.name}"}
            last_status = rec["status"]
            if step in MANDATORY_NO_SKIP:
                missing_mandatory.append(step)
        else:
            last_status = str(rec.get("status") or rec.get("pit_status") or "OK")
        payload[step] = rec if rec else {"status": last_status, "path": str(path)}
        if step == through:
            break
    if through in extra_files:
        p = extra_files[through]
        payload[through] = {"exists": p.exists(), "path": str(p), "status": "OK" if p.exists() else "MISSING"}
        last_status = payload[through]["status"]
    if factor_id in PIT_PASS_IDS and missing_mandatory:
        payload["evidence_status"] = "NOT_RUN" if last_status in {NOT_RUN, "MISSING"} else "FAILED"
        payload["full_evidence"] = False
        payload["missing_mandatory"] = missing_mandatory
        if payload.get("status") == FULL_EVIDENCE_COMPLETE:
            payload["status"] = "FAILED"
            payload["reason"] = "缺 mandatory JSON，不得标 FULL_EVIDENCE_COMPLETE"
    elif factor_id in PIT_PASS_IDS and not missing_mandatory and through in {"PromotionGate", "ConditionalIC", "TopTail"}:
        payload["full_evidence"] = True
        payload["evidence_status"] = FULL_EVIDENCE_COMPLETE
    payload["status"] = last_status
    if persist:
        dump_json(dest / "evaluate.json", payload)
    return payload
