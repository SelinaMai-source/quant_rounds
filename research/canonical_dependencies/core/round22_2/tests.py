"""Round 22.2 单元门。失败不得把 NOT_RUN 写成 COMPLETE。"""

from __future__ import annotations

from typing import Any

from .common import dump_json, load_json, sha256_json
from .contract import (
    C1_FORMULA_FROZEN,
    C4_FORMULA_FROZEN,
    ECONOMIC_MATERIALITY,
    FIVE_FEATURES,
    FUSION_WEIGHTS,
    ROUND22_2_ROOT,
    TEN_FEATURES,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
)

TEST_JSON = ROUND22_2_ROOT / "tests" / "unit_tests.json"


def run_tests() -> dict[str, Any]:
    rows = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        rows.append({"name": name, "passed": bool(ok), "detail": detail})

    parity = load_json(ROUND22_2_ROOT / "parity" / "parity.json")
    add("parity_pass", parity.get("passed") is True, str(parity.get("status")))
    obs = parity.get("observed") or {}
    add("parity_cagr", abs(float(obs.get("cagr") or 0) - V5_ANCHOR["cagr"]) < 1e-8, str(obs.get("cagr")))
    add("parity_sharpe", abs(float(obs.get("sharpe") or 0) - V5_ANCHOR["sharpe"]) < 1e-8, str(obs.get("sharpe")))
    add("parity_nav", abs(float(obs.get("nav_end") or 0) - V5_NAV_END_EXACT) < 0.01, str(obs.get("nav_end")))
    add("parity_trades", int(obs.get("completed_trades") or 0) == 1107, str(obs.get("completed_trades")))
    freeze = load_json(ROUND22_2_ROOT / "configs" / "frozen_gates.json")
    body = freeze.get("body") or {}
    add("gate_frozen", freeze.get("status") == "FROZEN" and freeze.get("do_not_retune") is True)
    add("gate_id", (body.get("economic_materiality") or {}).get("id") == "R21_3_ECONOMIC_MATERIALITY_V1")
    add("gate_hash", sha256_json(body) == freeze.get("body_sha256"), str(freeze.get("body_sha256")))
    add("five14", len(FIVE_FEATURES) == 14 and body.get("five_n") == 14)
    add("ten13", len(TEN_FEATURES) == 13 and body.get("ten_n") == 13)
    add("fusion", abs(FUSION_WEIGHTS["ten"] - 0.90) < 1e-12 and abs(FUSION_WEIGHTS["five"] - 0.10) < 1e-12)
    add("c1_formula", body.get("c1_formula_frozen") == C1_FORMULA_FROZEN)
    add("c4_formula", body.get("c4_formula_frozen") == C4_FORMULA_FROZEN)
    add("no_buy_value_xl", body.get("c1_no_buy_value_xl") is True)
    add("current_unchanged", body.get("current_baseline_changed") is False)
    add("economic_thresholds", ECONOMIC_MATERIALITY["delta_cagr_min_abs"] == 0.01 and ECONOMIC_MATERIALITY["delta_sharpe_min"] == 0.05)
    pit = load_json(ROUND22_2_ROOT / "pit" / "c_pit.json")
    add("c_formula_match", (pit.get("identity") or {}).get("status") in {None, "MATCH", ""} or (pit.get("identity") or {}).get("formula_changed") is False)
    inter = load_json(ROUND22_2_ROOT / "evidence" / "existing11.json")
    add("existing11_extracted", inter.get("n") == 11 and inter.get("recomputed") is False)
    train = load_json(ROUND22_2_ROOT / "model" / "train.json")
    add("no_alias_add", train.get("alias_add") is not True)
    add("no_mix_full_v5", train.get("compare_mixed_full_v5") is not True)
    n_failed = int(sum(not r["passed"] for r in rows))
    payload = {"status": "PASS" if n_failed == 0 else "FAIL", "n": len(rows), "n_failed": n_failed, "rows": rows}
    dump_json(TEST_JSON, payload)
    return payload
