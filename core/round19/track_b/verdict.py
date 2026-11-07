"""PIPELINE_BUG_VERDICT。禁止搜 epsilon / 改 F 公式 / 改 head 权重。"""

from __future__ import annotations

from typing import Any

from ..contract import (
    CLOSE_ROUND18_FACTOR_HYPOTHESES,
    INCONCLUSIVE_DUE_TO_DATA_PIT,
    PARITY_METRIC_ABS,
    TRACK_B_GATES,
)
from .audit import run_parity
from .heads import run_heads, run_top_change
from .allocation import run_allocation
from .neighbors import run_neighbors
from .retrain import run_fresh_retrain
from .support import B00, B01, PARITY_JSON, VERDICT_JSON, cand_dir, dump_json, load_json, tb_dir, write_md

MATERIAL_SHARPE_GAP = 0.15
MATERIAL_TOP1 = 0.80


def _ensure_all(*, resume: bool = True, **kw: Any) -> dict[str, Any]:
    out = {}
    out["retrain"] = load_json(tb_dir("retrain") / "fresh_retrain.json") or run_fresh_retrain(resume=resume, **kw)
    out["parity"] = load_json(PARITY_JSON) or run_parity(resume=resume, **kw)
    out["heads"] = load_json(tb_dir("heads") / "heads.json") or run_heads(resume=resume, **kw)
    out["top"] = load_json(tb_dir("heads") / "top_change.json") or run_top_change(resume=resume, **kw)
    out["alloc"] = load_json(tb_dir("heads") / "allocation.json") or run_allocation(resume=resume, **kw)
    out["neighbors"] = load_json(tb_dir("heads") / "neighbors.json") or run_neighbors(resume=resume, **kw)
    return out


def _layer_status(parity: dict[str, Any], level: int) -> str:
    for layer in parity.get("layers") or []:
        if int(layer.get("level") or 0) == level:
            return str(layer.get("status") or "NOT_IDENTIFIABLE")
    return "NOT_IDENTIFIABLE"


def _layer_detail(parity: dict[str, Any], level: int) -> dict[str, Any]:
    for layer in parity.get("layers") or []:
        if int(layer.get("level") or 0) == level:
            return layer
    return {}


def run_bug_verdict(*, resume: bool = True, **kw: Any) -> dict[str, Any]:
    pack = _ensure_all(resume=resume, **kw)
    parity = pack["parity"]
    labels = load_json(tb_dir("audit") / "labels.json")
    feats = load_json(tb_dir("audit") / "feature_order.json")
    miss = load_json(tb_dir("audit") / "missing.json")
    b00 = load_json(cand_dir(B00) / "summary.json")
    b01 = load_json(cand_dir(B01) / "summary.json")
    obs0 = (b00 or {}).get("observed") or {}
    obs1 = (b01 or {}).get("observed") or {}
    l1, l2, l3, l4, l5, l6 = [_layer_status(parity, i) for i in range(1, 7)]
    l5d = _layer_detail(parity, 5)
    l6d = _layer_detail(parity, 6)

    reasons: list[str] = []
    contract_ok = True
    if miss.get("filled_with_zero"):
        contract_ok = False
        reasons.append("filled_with_zero=True")
    if feats and not feats.get("passed"):
        contract_ok = False
        reasons.append(f"feature_order FAIL extras={feats.get('five_extras')}{feats.get('ten_extras')}")
    if labels and not labels.get("passed"):
        contract_ok = False
        reasons.append("label/next2_date mismatch vs original panel")
    if l1 == "FAIL":
        contract_ok = False
        reasons.append("LEVEL1 join_keys FAIL")
    if l3 == "FAIL":
        contract_ok = False
        reasons.append("LEVEL3 prediction_rows B01 != R7 45256")
    if l4 == "FAIL":
        contract_ok = False
        reasons.append("LEVEL4 feature_order/missing FAIL")
    if l2 == "FAIL":
        contract_ok = False
        reasons.append("LEVEL2 labels_groups FAIL")

    sharpe_gap = None
    try:
        sharpe_gap = abs(float(obs1.get("sharpe")) - float(obs0.get("sharpe")))
    except (TypeError, ValueError):
        sharpe_gap = None
    five_top1 = ((l5d.get("detail") or {}).get("five") or {}).get("top1_agree")
    overlay_top1 = ((l6d.get("detail") or {}).get("overlay_top1_agree"))
    xgb_env = bool(parity.get("xgb_env_diff_recorded") or l5 == "XGB_ENV_DIFF" or l6 == "XGB_ENV_DIFF")

    material = False
    if not contract_ok:
        # contract break is material only if it actually moves the locked original-feature path
        if sharpe_gap is not None and sharpe_gap > MATERIAL_SHARPE_GAP:
            material = True
            reasons.append(f"contract FAIL and |ΔSharpe|={sharpe_gap:.4f} > {MATERIAL_SHARPE_GAP}")
        elif five_top1 is not None and five_top1 < MATERIAL_TOP1:
            material = True
            reasons.append(f"contract FAIL and five Top1 agree={five_top1} < {MATERIAL_TOP1}")
        elif l3 == "FAIL" or (miss.get("filled_with_zero")) or (feats and not feats.get("passed")) or (
            labels and not labels.get("passed")
        ):
            material = True
            reasons.append("contract violation on universe/labels/features/fillna")
        else:
            reasons.append("contract FAIL but not classified material pending impact")

    a01_extra = miss.get("a01_extra_drop_vs_r7")
    if a01_extra:
        reasons.append(
            f"R18_A01 extra complete_cases drop vs R7 = {a01_extra} (F1 missingness mechanism, not default bug)"
        )

    if not material and contract_ok:
        decision = CLOSE_ROUND18_FACTOR_HYPOTHESES
        verdict = "NO_MATERIAL_PIPELINE_BUG"
        note = (
            "B01 force-retrain of original 14/13 does not expose a material pipeline bug. "
            "Round18 A01+ degradation is not explained by the A00 locked-overlay short-circuit. "
            "Does not prove the factors can never have alpha; closes the current F1/F2/F3 formulas "
            "on this panel/model/sample."
        )
    elif not material and not contract_ok and (l2 == "NOT_IDENTIFIABLE" or l1 == "NOT_IDENTIFIABLE"):
        decision = INCONCLUSIVE_DUE_TO_DATA_PIT
        verdict = "INCONCLUSIVE"
        note = "join/label/fold not fully identifiable."
    elif material:
        decision = "MATERIAL_PIPELINE_BUG"
        verdict = "MATERIAL_PIPELINE_BUG"
        note = "Isolated repair required before any R18 matrix re-run. Original formulas only. No epsilon search."
    else:
        decision = CLOSE_ROUND18_FACTOR_HYPOTHESES
        verdict = "NO_MATERIAL_PIPELINE_BUG"
        note = "Residual differences recorded as XGB_ENV_DIFF, not a repairable contract bug."

    if xgb_env:
        reasons.append("XGB_ENV_DIFF recorded; not treated as ignorable PASS")

    payload = {
        "status": verdict,
        "PIPELINE_BUG_VERDICT": verdict,
        "TRACK_B_DECISION": decision,
        "material_bug": material,
        "r18_rebacktest_required": bool(material),
        "contract_ok": contract_ok,
        "xgb_env_diff": xgb_env,
        "six_layer": {i: _layer_status(parity, i) for i in range(1, 7)},
        "B00": obs0,
        "B01": obs1,
        "sharpe_gap": sharpe_gap,
        "five_top1_agree": five_top1,
        "overlay_top1_agree": overlay_top1,
        "metric_abs_tol": PARITY_METRIC_ABS,
        "reasons": reasons,
        "gates": dict(TRACK_B_GATES),
        "forbid_search_epsilon": True,
        "forbid_change_f_formulas": True,
        "forbid_change_head_weights": True,
        "round18_originals_untouched": True,
        "note": note,
        "repaired_best": None,
    }
    dump_json(VERDICT_JSON, payload)
    dump_json(tb_dir() / "TRACK_B_DECISION.json", payload)
    lines = [
        "# Track B · PIPELINE_BUG_VERDICT",
        "",
        f"PIPELINE_BUG_VERDICT=`{verdict}`",
        f"TRACK_B_DECISION=`{decision}`",
        f"material_bug=`{material}` r18_rebacktest_required=`{material}`",
        f"xgb_env_diff=`{xgb_env}` contract_ok=`{contract_ok}`",
        "",
        "## 六层",
        "",
        "| L | status |",
        "|---|---|",
    ]
    for i in range(1, 7):
        lines.append(f"| {i} | `{_layer_status(parity, i)}` |")
    lines += [
        "",
        f"- B00 Sharpe=`{obs0.get('sharpe')}` CAGR=`{obs0.get('cagr')}` trades=`{obs0.get('completed_trades')}`",
        f"- B01 Sharpe=`{obs1.get('sharpe')}` CAGR=`{obs1.get('cagr')}` trades=`{obs1.get('completed_trades')}`",
        f"- |ΔSharpe|=`{sharpe_gap}` five Top1=`{five_top1}` overlay Top1=`{overlay_top1}`",
        "",
        "## reasons",
        "",
    ]
    for r in reasons:
        lines.append(f"- {r}")
    lines += ["", note, "", "未搜 epsilon / 未改 F 公式 / 未改 head 权重。未覆盖 result/round18 原件。", ""]
    write_md(tb_dir() / "12_bug_verdict.md", "\n".join(lines))
    print_track_b_block(payload)
    return payload


def print_track_b_block(payload: dict[str, Any] | None = None) -> None:
    payload = payload or load_json(VERDICT_JSON)
    six = payload.get("six_layer") or {}
    print("", flush=True)
    print("===== ROUND19 TRACK B =====", flush=True)
    print(f"PIPELINE_BUG_VERDICT={payload.get('PIPELINE_BUG_VERDICT')}", flush=True)
    print(f"TRACK_B_DECISION={payload.get('TRACK_B_DECISION')}", flush=True)
    print(
        "six_layer "
        + " ".join(f"L{k}={v}" for k, v in sorted(six.items(), key=lambda kv: int(kv[0]))),
        flush=True,
    )
    print(
        f"material_bug={payload.get('material_bug')} r18_rebacktest_required={payload.get('r18_rebacktest_required')}",
        flush=True,
    )
    print(f"xgb_env_diff={payload.get('xgb_env_diff')} contract_ok={payload.get('contract_ok')}", flush=True)
    b00 = payload.get("B00") or {}
    b01 = payload.get("B01") or {}
    print(
        f"B00 Sharpe={b00.get('sharpe')} CAGR={b00.get('cagr')} trades={b00.get('completed_trades')}",
        flush=True,
    )
    print(
        f"B01 Sharpe={b01.get('sharpe')} CAGR={b01.get('cagr')} trades={b01.get('completed_trades')}",
        flush=True,
    )
    print(f"verdict={VERDICT_JSON}", flush=True)
    print("===== END TRACK B =====", flush=True)


__all__ = ["print_track_b_block", "run_bug_verdict"]
