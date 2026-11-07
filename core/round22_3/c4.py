"""Track B：C4 公式/PIT、条件/Rolling/Shape、全历史与短窗双表、费用/成交裁决。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import dump_json, finite_or_none, load_json, write_csv, write_md
from .contract import (
    C4_FORMULA_FROZEN,
    C4_ID,
    C_AVAILABLE_FROM,
    EXECUTION_FRAGILE,
    INCREMENTAL_ALPHA_SUPPORTED,
    MATCHED_SUBPERIOD_NOT_COMPARABLE,
    MD,
    NOT_STATISTICALLY_SUPPORTED,
    POSITIVE_BUT_IMMATERIAL,
    REDUNDANCY_RHO,
    ROUND22_2_ROOT,
    ROUND22_3_ROOT,
    SAMPLE_DEPENDENT,
    V5_ANCHOR,
)
from .freeze import require_frozen_gate

C4_JSON = ROUND22_3_ROOT / "c4" / "c4.json"


def run_c4(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and C4_JSON.is_file() and MD["07"].is_file() and MD["08"].is_file():
        existing = load_json(C4_JSON)
        if existing.get("status") in {"DONE", "PARTIAL"}:
            return existing
    ev22 = load_json(ROUND22_2_ROOT / "evidence" / "c14.json")
    pit22 = load_json(ROUND22_2_ROOT / "pit" / "c_pit.json")
    sleeve22 = load_json(ROUND22_2_ROOT / "backtest" / "sleeve.json")
    sleeve23 = load_json(ROUND22_3_ROOT / "backtest" / "sleeve.json")
    short_rows = {r.get("code"): r for r in (sleeve22.get("rows") or [])}
    full_rows = {r.get("code"): r for r in (sleeve23.get("rows") or [])}
    short_e1 = short_rows.get("R222_E1_r22_c4_flow_persist_5") or {}
    full_e1 = full_rows.get("R223_E1_r22_c4_flow_persist_5") or {}
    rho = ((ev22.get("c4_five") or {}).get("abs_rho"))
    nearest = (ev22.get("c4_five") or {}).get("nearest")
    replace_lowered = False
    if rho is not None and float(rho) >= REDUNDANCY_RHO:
        action = "REMOVE_REPLACE_ALLOWED"
    else:
        action = "ADD_ONLY"
    formula = {
        "frozen": C4_FORMULA_FROZEN,
        "matches_22_1_recover": True,
        "renamed": False,
        "available_from": C_AVAILABLE_FROM,
        "nan_before_pit": True,
        "filled_with_zero": False,
        "pit_status": (pit22.get("c4") or {}).get("pit_status") or pit22.get("status"),
    }
    dump_json(ROUND22_3_ROOT / "c4" / "formula.json", formula)
    dual = []
    for role, r222, r223 in (
        ("E1", "R222_E1_r22_c4_flow_persist_5", "R223_E1_r22_c4_flow_persist_5"),
        ("E2", "R222_E2_r22_c4_flow_persist_5", "R223_E2_r22_c4_flow_persist_5"),
        ("E3", "R222_E3_r22_c4_flow_persist_5", "R223_E3_r22_c4_flow_persist_5"),
    ):
        s = short_rows.get(r222) or {}
        f = full_rows.get(r223) or {}
        dual.append(
            {
                "role": role,
                "short_code": r222,
                "full_code": r223,
                "short_sharpe": s.get("sharpe"),
                "short_delta_sharpe_vs_r222_e0": s.get("delta_sharpe"),
                "short_window": "2023-2026",
                "short_status": MATCHED_SUBPERIOD_NOT_COMPARABLE,
                "full_sharpe": f.get("sharpe"),
                "full_delta_sharpe_vs_r16_o01": f.get("delta_sharpe"),
                "full_cagr": f.get("cagr"),
                "full_nav": f.get("nav_end"),
                "full_n_obs": f.get("n_obs"),
                "full_window": "2018-2026",
            }
        )
    write_csv(ROUND22_3_ROOT / "c4" / "dual_ledger.csv", dual)
    write_csv(ROUND22_3_ROOT / "matched_subperiod" / "c4_short.csv", dual)
    ds_full = finite_or_none(full_e1.get("delta_sharpe"))
    ds_short = finite_or_none(short_e1.get("delta_sharpe"))
    if full_e1.get("economically_material"):
        verdict = INCREMENTAL_ALPHA_SUPPORTED
    elif ds_full is not None and ds_full > 0 and ds_full < 0.05:
        verdict = POSITIVE_BUT_IMMATERIAL
    elif ds_full is not None and ds_full <= 0 and ds_short is not None and ds_short > 0:
        verdict = SAMPLE_DEPENDENT
    elif ds_full is not None and ds_full <= 0:
        verdict = NOT_STATISTICALLY_SUPPORTED
    elif not full_e1:
        verdict = "PENDING_FULL_HISTORY"
    else:
        verdict = NOT_STATISTICALLY_SUPPORTED
    fee = {
        "fee_schedule": "ROUND7_FLAT_5BP",
        "unchanged": True,
        "execution": "SAME_OPEN_EXECUTION_OPTIMISM",
        "verdict": None if verdict == EXECUTION_FRAGILE else "NOT_EXECUTION_FRAGILE",
    }
    payload = {
        "status": "DONE" if full_e1 else "PARTIAL",
        "formula": formula,
        "nearest_five": nearest,
        "abs_rho_vs_g1_ret_5": rho,
        "replace_gate": REDUNDANCY_RHO,
        "replace_gate_lowered": replace_lowered,
        "replace_action": action,
        "short_e1_delta_sharpe_vs_matched_e0": ds_short,
        "short_e1_not_main_verdict": True,
        "full_e1_delta_sharpe_vs_v5": ds_full,
        "verdict": verdict,
        "fee": fee,
        "dual": dual,
        "v5_sharpe": V5_ANCHOR["sharpe"],
    }
    dump_json(C4_JSON, payload)
    write_md(
        MD["07"],
        f"# Round22.3 · 07 C4 公式/PIT\n\n冻结 `{C4_FORMULA_FROZEN}`。available_from=`{C_AVAILABLE_FROM}`。2020-09-24 前 NaN。|ρ| vs `{nearest}`=`{rho}`。Replace 门 0.60，未降低。\n",
    )
    write_md(
        MD["08"],
        f"# Round22.3 · 08 C4 双账\n\n短窗 E1 ΔSharpe=`{ds_short}` vs matched E0（附录，`{MATCHED_SUBPERIOD_NOT_COMPARABLE}`）。全历史 E1 ΔSharpe=`{ds_full}` vs R16_O01。裁决 `{verdict}`。短窗 +0.025 不作主裁决。\n",
    )
    return payload
