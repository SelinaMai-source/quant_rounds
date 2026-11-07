"""经济门：LEGACY（含旧缺陷）与 CORRECTED（有符号 NAV + 三态证据）。"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from .nav_delta import signed_nav_metrics
from .status import EvidenceStatus, as_bool
from .time_contract import shanghai_now

GATE_KEYS = (
    "id",
    "delta_cagr_min_abs",
    "delta_sharpe_min",
    "nav_rel_min",
    "maxdd_worse_max_abs",
    "nav_abs_min",
)


def _sha(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_gate() -> dict[str, Any]:
    from core.round22_3.contract import ECONOMIC_MATERIALITY

    return {k: ECONOMIC_MATERIALITY[k] for k in GATE_KEYS if k in ECONOMIC_MATERIALITY}


def freeze_economic_gate(dest: Path | None = None) -> dict[str, Any]:
    gate = _load_gate()
    raw_path = Path(__file__).resolve().parents[1] / "round21_4" / "contract.py"
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.is_file() else None
    code_hash = hashlib.sha256(inspect.getsource(evaluate_economic_gate).encode("utf-8")).hexdigest()
    payload = {
        "gate_id": gate["id"],
        "thresholds": gate,
        "canonical_threshold_hash": _sha(gate),
        "raw_file": str(raw_path),
        "raw_file_hash": raw_hash,
        "gate_code_hash": code_hash,
        "freeze_timestamp": shanghai_now().isoformat(),
        "implementation": {
            "LEGACY": "core.round22_3.contract.material_vs_v5（绝对值 nav_rel；缺证据默认真）",
            "CORRECTED": "research_integrity_v2.economic_gate.evaluate_economic_gate",
        },
        "promotion_uses": "CORRECTED",
    }
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _legacy(obs: dict[str, Any], *, top_changed: bool, fee_same_direction: bool, not_only_year: bool) -> dict[str, Any]:
    from core.round22_3.contract import material_vs_v5

    out = material_vs_v5(obs, top_changed=top_changed, fee_same_direction=fee_same_direction, not_only_year=not_only_year)
    out["implementation"] = "LEGACY"
    return out


def evaluate_economic_gate(
    obs: dict[str, Any],
    *,
    baseline: dict[str, Any],
    top_changed: bool | None = None,
    fee_same_direction: bool | None = None,
    not_only_year: bool | None = None,
) -> dict[str, Any]:
    gate = _load_gate()
    legacy = _legacy(
        obs,
        top_changed=bool(top_changed) if top_changed is not None else False,
        fee_same_direction=True if fee_same_direction is None else bool(fee_same_direction),
        not_only_year=True if not_only_year is None else bool(not_only_year),
    )

    dc = None if obs.get("cagr") is None else float(obs["cagr"]) - float(baseline["cagr"])
    ds = None if obs.get("sharpe") is None else float(obs["sharpe"]) - float(baseline["sharpe"])
    dd = None if obs.get("max_drawdown") is None else float(obs["max_drawdown"]) - float(baseline["max_drawdown"])
    nav_m = signed_nav_metrics(obs.get("nav_end"), baseline.get("nav_end"), min_rel=float(gate["nav_rel_min"]))

    def _triple(value: bool | None, computed_ok: bool | None) -> dict[str, Any]:
        if value is None:
            return {"value": None, "status": EvidenceStatus.NOT_EVALUATED.value, "ok": None}
        real = as_bool(value)
        if real is None:
            return {"value": value, "status": EvidenceStatus.BLOCKED.value, "ok": None, "reason": "non_bool"}
        return {"value": real, "status": EvidenceStatus.PASS.value if real else EvidenceStatus.FAIL.value, "ok": real}

    top_ev = _triple(top_changed, None)
    fee_ev = _triple(fee_same_direction, None)
    year_ev = _triple(not_only_year, None)

    flags = {
        "delta_cagr_ok": None if dc is None else bool(dc >= float(gate["delta_cagr_min_abs"])),
        "delta_sharpe_ok": None if ds is None else bool(ds >= float(gate["delta_sharpe_min"])),
        "nav_improvement_ok": nav_m["nav_improvement_ok"],
        "top_changed_ok": top_ev["ok"],
        "maxdd_ok": None if dd is None else bool(dd >= -float(gate["maxdd_worse_max_abs"])),
        "fee_direction_ok": fee_ev["ok"],
        "not_only_2024_or_2025": year_ev["ok"],
    }
    statuses = {
        "delta_cagr": EvidenceStatus.NOT_EVALUATED.value if flags["delta_cagr_ok"] is None else (EvidenceStatus.PASS.value if flags["delta_cagr_ok"] else EvidenceStatus.FAIL.value),
        "delta_sharpe": EvidenceStatus.NOT_EVALUATED.value if flags["delta_sharpe_ok"] is None else (EvidenceStatus.PASS.value if flags["delta_sharpe_ok"] else EvidenceStatus.FAIL.value),
        "nav_improvement": nav_m["status"],
        "top_changed": top_ev["status"],
        "maxdd": EvidenceStatus.NOT_EVALUATED.value if flags["maxdd_ok"] is None else (EvidenceStatus.PASS.value if flags["maxdd_ok"] else EvidenceStatus.FAIL.value),
        "fee_direction": fee_ev["status"],
        "not_only_year": year_ev["status"],
    }
    evaluated = [v for v in flags.values() if v is not None]
    material = bool(evaluated) and all(evaluated) and all(v is not None for v in flags.values())
    any_missing = any(v is None for v in flags.values())
    corrected = {
        "implementation": "CORRECTED",
        "gate_id": gate["id"],
        "compare_to": baseline.get("code") or "R16_O01",
        "delta_cagr": dc,
        "delta_sharpe": ds,
        "delta_maxdd": dd,
        **{k: nav_m[k] for k in ("nav_improvement", "delta_terminal_nav_rmb", "relative_terminal_nav_change", "nav_improvement_ok")},
        "flags": flags,
        "statuses": statuses,
        "economically_material": False if any_missing else material,
        "promotion_blocked_missing_evidence": any_missing,
        "evidence": {"top_changed": top_ev, "fee_same_direction": fee_ev, "not_only_year": year_ev},
    }
    diffs = []
    if legacy.get("nav_rel") is not None and nav_m.get("nav_improvement") is not None:
        if abs(float(legacy["nav_rel"]) - abs(float(nav_m["nav_improvement"]))) > 1e-12:
            diffs.append("nav_abs_vs_signed_mismatch")
        if (legacy.get("flags") or {}).get("nav_rel_ok") and nav_m["nav_improvement"] is not None and nav_m["nav_improvement"] < 0:
            diffs.append("legacy_nav_rel_ok_on_nav_decline")
    if fee_same_direction is None and (legacy.get("flags") or {}).get("fee_direction_ok") is True:
        diffs.append("legacy_defaulted_fee_direction_true")
    if not_only_year is None and (legacy.get("flags") or {}).get("not_only_2024_or_2025") is True:
        diffs.append("legacy_defaulted_not_only_year_true")
    if top_changed is None and (legacy.get("flags") or {}).get("top_changed_ok") is False:
        diffs.append("legacy_defaulted_top_changed_false_when_arg_omitted")
    return {
        "LEGACY": legacy,
        "CORRECTED": corrected,
        "diffs": diffs,
        "promotion_decision_source": "CORRECTED",
        "economically_material": corrected["economically_material"],
    }
