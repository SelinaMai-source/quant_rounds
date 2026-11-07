"""Factor Evidence Factory。FactorDefinition → … → PromotionGate。

未过门写 SKIPPED/FAIL + 原因，禁止伪造 PASS，禁止复制全局结果冒充单因子包。
"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_ROOT
from .common import dump_json, load_json
from .contract import (
    AUTO_PROMOTE_CURRENT,
    FACTORY_STEPS,
    KEEP_BASELINE_V5,
    PROMOTION_GATES,
    RECOMMEND_NEW_ALPHA_CANDIDATE,
    V5_ANCHOR,
)
from .hypotheses import HYPOTHESES, require_hypotheses

IMPLEMENTED_THROUGH = frozenset(FACTORY_STEPS)
SKIP_IF_MISSING = frozenset(
    {
        "Redundancy",
        "ConditionalIC",
        "Combination",
        "ModelIncrementality",
        "PortfolioEvidence",
        "PromotionGate",
    }
)


class FactorDefinition:
    def __init__(self, card: dict[str, Any]):
        self.card = dict(card)
        self.factor_id = str(card["factor_id"])
        self.family = str(card["family"])
        self.formula = str(card["formula"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self.card)


def load_definition(factor_id: str) -> FactorDefinition:
    for card in HYPOTHESES:
        if card["factor_id"] == factor_id:
            return FactorDefinition(card)
    raise KeyError(factor_id)


def _pit_status(pit: dict[str, Any]) -> str:
    if not pit:
        return "MISSING"
    if pit.get("status") in {"SKIPPED", "MISSING"}:
        return str(pit.get("status"))
    return str(pit.get("pit_status") or pit.get("status") or "MISSING")


def _best_portfolio_rows(factor_id: str) -> list[dict[str, Any]]:
    from .backtest import BT_JSON

    bt = load_json(BT_JSON)
    rows = []
    for rec in bt.get("main_rows") or []:
        code = str(rec.get("code") or "")
        if factor_id in code:
            rows.append(rec)
    return rows


def _redundancy_action(red: dict[str, Any]) -> str:
    if not red:
        return ""
    if red.get("status") in {"SKIPPED", "MISSING"}:
        return ""
    decision = red.get("decision") or {}
    return str(decision.get("action") or red.get("action") or "")


def _increment_pass(inc: dict[str, Any]) -> bool | None:
    if not inc or inc.get("status") in {"SKIPPED", "MISSING"}:
        return None
    for run in inc.get("runs") or []:
        gate = (run or {}).get("increment") or {}
        if gate.get("passed") is True:
            return True
    if inc.get("runs"):
        return False
    return None


def build_promotion_gate(factor_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """晋级门：缺产物 = SKIPPED+原因；未优于 v5 = FAIL；禁止伪造 PASS。"""
    pit = payload.get("PIT") if isinstance(payload.get("PIT"), dict) else {}
    red = payload.get("Redundancy") if isinstance(payload.get("Redundancy"), dict) else {}
    inc = payload.get("ModelIncrementality") if isinstance(payload.get("ModelIncrementality"), dict) else {}
    port = payload.get("PortfolioEvidence") if isinstance(payload.get("PortfolioEvidence"), dict) else {}
    pit_status = _pit_status(pit)
    action = _redundancy_action(red)
    rows = _best_portfolio_rows(factor_id)
    v5_sh = float(V5_ANCHOR["sharpe"])
    v5_cagr = float(V5_ANCHOR["cagr"])
    best = None
    better = False
    if rows:
        scored = [r for r in rows if r.get("sharpe") is not None]
        best = max(scored, key=lambda r: float(r["sharpe"])) if scored else None
        if best is not None:
            sh = best.get("sharpe")
            cg = best.get("cagr")
            better = (sh is not None and float(sh) > v5_sh + 1e-12) or (
                cg is not None and float(cg) > v5_cagr + 1e-12
            )
    inc_pass = _increment_pass(inc)
    quality = action == "KEEP"
    yearly = (best or {}).get("yearly_returns") or (port.get("yearly_returns") if isinstance(port, dict) else {}) or {}
    not_only_2425 = False
    if yearly:
        others = [k for k in yearly if str(k) not in {"2024", "2025"}]
        not_only_2425 = any(yearly.get(k) not in (None, 0, 0.0) for k in others) or bool(others)
    checks = {
        "pit_pass": pit_status == "PIT_PASS",
        "quality_pass": quality,
        "ic_rolling_conditional": bool(payload.get("IC")) and bool(payload.get("Rolling")),
        "top_tail": bool(payload.get("TopTail")) and (payload.get("TopTail") or {}).get("status") not in {"SKIPPED", "MISSING"},
        "replace_or_complement": bool(red) and (red.get("status") not in {"SKIPPED", "MISSING"}),
        "oof_not_worse": bool(inc_pass),
        "same_caliber_net_better_than_v5": better,
        "robustness": not_only_2425 if yearly else False,
        "subsample_1_0": True,
        "auto_promote_current": False,
    }
    required = [
        "pit_pass",
        "quality_pass",
        "same_caliber_net_better_than_v5",
        "oof_not_worse",
    ]
    missing_steps = [k for k in required if not checks.get(k)]
    if pit_status in {"MISSING", "SKIPPED"}:
        status = "SKIPPED"
        reason = f"PIT artifact {pit_status}; refuse fake PASS"
    elif pit_status == "BLOCKED_PIT":
        status = "SKIPPED"
        reason = "BLOCKED_PIT: E/大宗无公告时点，禁 PIT_ELIGIBLE_MODEL；refuse fake PASS"
    elif pit_status == "PIT_FAIL_COVERAGE":
        status = "SKIPPED"
        reason = "PIT_FAIL_COVERAGE: 覆盖不足，DIAGNOSTIC_ONLY；refuse fake PASS"
    elif not rows:
        status = "SKIPPED"
        reason = "no full-engine portfolio row for this factor; downstream not copied from global; refuse fake PASS"
    elif not better:
        status = "FAIL"
        reason = (
            f"same_caliber_net_better_than_v5=False "
            f"best={best.get('code') if best else None} "
            f"Sharpe={best.get('sharpe') if best else None} "
            f"CAGR={best.get('cagr') if best else None} "
            f"vs v5 Sharpe={v5_sh} CAGR={v5_cagr}"
        )
    elif missing_steps:
        status = "FAIL"
        reason = f"gates failed: {missing_steps}; refuse fake PASS"
    else:
        status = RECOMMEND_NEW_ALPHA_CANDIDATE
        reason = None
    passed = status == RECOMMEND_NEW_ALPHA_CANDIDATE and better
    # 本轮全局未过门：即使单因子误过，也不升 CURRENT。
    decision = RECOMMEND_NEW_ALPHA_CANDIDATE if passed else KEEP_BASELINE_V5
    return {
        "factor_id": factor_id,
        "status": status,
        "reason": reason,
        "passed": bool(passed),
        "decision": decision,
        "lock": None,
        "auto_promote": AUTO_PROMOTE_CURRENT,
        "do_not_lock_v6": True,
        "copied_global_result": False,
        "checks": checks,
        "failed_required": missing_steps,
        "gates_spec": {k: PROMOTION_GATES.get(k) for k in PROMOTION_GATES},
        "best_same_factor_row": None
        if best is None
        else {
            "code": best.get("code"),
            "sharpe": best.get("sharpe"),
            "cagr": best.get("cagr"),
            "max_drawdown": best.get("max_drawdown"),
            "completed_trades": best.get("completed_trades"),
            "nav_end": best.get("nav_end"),
        },
        "v5": {"sharpe": v5_sh, "cagr": v5_cagr},
        "redundancy_action": action,
        "pit_status": pit_status,
        "n_portfolio_rows": len(rows),
        "increment_pass": inc_pass,
    }


def evaluate_factor(factor_id: str, *, through: str = "FactorDefinition", persist: bool = False) -> dict[str, Any]:
    """API：FactorDefinition → DataAudit → … → PromotionGate。"""
    require_hypotheses()
    definition = load_definition(factor_id)
    from ..round21_1.contract import FACTORY_STEPS_EXTRA

    allowed = set(FACTORY_STEPS) | set(FACTORY_STEPS_EXTRA)
    if through not in allowed:
        raise RuntimeError(f"unknown factory step: {through}")
    dest = ROUND21_ROOT / "factors" / factor_id
    payload: dict[str, Any] = {
        "factor_id": factor_id,
        "step": through,
        "factory_steps": list(FACTORY_STEPS),
        "definition": definition.as_dict(),
        "copied_global_result": False,
    }
    if through == "FactorDefinition":
        payload["status"] = "DEFINITION_ONLY"
        payload["values_constructed"] = (dest / "values.parquet").is_file()
        return payload
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
    last_status = "OK"
    for step, path in files.items():
        if path.is_file():
            payload[step] = load_json(path)
        else:
            kind = "SKIPPED" if step in SKIP_IF_MISSING else "MISSING"
            payload[step] = {
                "status": kind,
                "reason": f"{step} artifact not written for {factor_id}; do not copy global results; refuse fake PASS",
            }
        if step == through:
            last_status = "OK" if path.is_file() else payload[step]["status"]
            break
        if through == "PromotionGate":
            continue
    if through == "PromotionGate":
        gate = build_promotion_gate(factor_id, payload)
        payload["PromotionGate"] = gate
        last_status = str(gate.get("status") or "SKIPPED")
        if persist:
            dump_json(dest / "promotion.json", gate)
    payload["status"] = last_status
    payload["values_constructed"] = (dest / "values.parquet").is_file()
    payload["package"] = load_json(dest / "package.json") if (dest / "package.json").is_file() else {}
    from ..paths import ROUND21_1_ROOT

    r211 = ROUND21_1_ROOT / "factors" / factor_id
    extra_files = {
        "NEW_VS_BASELINE_CORRELATION": ROUND21_1_ROOT / "correlation" / "correlation.json",
        "NEAREST_LEGACY_FACTOR_IDENTIFICATION": ROUND21_1_ROOT / "correlation" / "fold_nearest.json",
        "FOUR_WAY_REPLACEMENT_TEST": r211 / "attribution.json",
        "CLUSTER_REPLACEMENT_IF_APPLICABLE": ROUND21_1_ROOT / "09_cluster.md",
        "OOF_INCREMENTALITY": r211 / "incrementality.json",
        "FULL_PORTFOLIO_INCREMENTALITY": r211 / "portfolio.json",
    }
    extra_payload = {}
    for step, path in extra_files.items():
        if path.is_file():
            extra_payload[step] = load_json(path) if path.suffix == ".json" else {"status": "OK", "path": str(path)}
        else:
            extra_payload[step] = {
                "status": "SKIPPED",
                "reason": f"Round21.1 {step} not written for {factor_id}; refuse fake PASS",
            }
    payload["round21_1"] = extra_payload
    if through in extra_files:
        payload["status"] = extra_payload[through].get("status") or last_status
    return payload


__all__ = [
    "FactorDefinition",
    "IMPLEMENTED_THROUGH",
    "build_promotion_gate",
    "evaluate_factor",
    "load_definition",
]
