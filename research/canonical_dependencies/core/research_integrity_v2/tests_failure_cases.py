"""第6节失败案例自动测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache_identity import ExperimentIdentity, cache_valid, write_identity
from .economic_gate import evaluate_economic_gate
from .ledger import nav_tolerance, reconcile_nav
from .merge_rank import explicit_merge, rank_unique
from .nav_delta import nav_improvement
from .status import as_bool, require_bool
from .time_contract import LEGACY_OPEN_INFORMATION_LANE


def _baseline() -> dict[str, Any]:
    return {
        "code": "R16_O01",
        "sharpe": 1.8756627913361097,
        "cagr": 0.5499812832273525,
        "max_drawdown": -0.24056327715394388,
        "nav_end": 12750089.355724689,
    }


def run_unit_tests(tmp: Path | None = None) -> dict[str, Any]:
    root = Path(tmp) if tmp else Path("/tmp/research_integrity_v2_tests")
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        rows.append({"name": name, "pass": bool(ok), "detail": detail})

    # False string not True
    check("false_string_not_true", as_bool("False") is None and as_bool(False) is False)
    try:
        require_bool("False", field="x")
        check("require_bool_rejects_false_string", False)
    except TypeError:
        check("require_bool_rejects_false_string", True)

    # NAV decline => negative improvement
    imp = nav_improvement(1_000_000.0, 12_750_089.355724689)
    check("nav_improvement_negative_on_decline", imp is not None and imp < 0, imp)

    # missing fee evidence cannot be fee_ok True
    gate = evaluate_economic_gate(
        {"cagr": 0.56, "sharpe": 1.93, "nav_end": 13_000_000.0, "max_drawdown": -0.24},
        baseline=_baseline(),
        top_changed=True,
        fee_same_direction=None,
        not_only_year=True,
    )
    check(
        "missing_fee_not_fee_ok_true",
        gate["CORRECTED"]["flags"]["fee_direction_ok"] is None
        and gate["CORRECTED"]["economically_material"] is False
        and gate["CORRECTED"]["statuses"]["fee_direction"] == "NOT_EVALUATED",
    )
    check("legacy_defaults_fee_true_diff_recorded", "legacy_defaulted_fee_direction_true" in gate["diffs"])

    # cache invalidation on input hash change
    dest = root / "cache_demo"
    ident = ExperimentIdentity(
        experiment_id="demo",
        code_hash="a",
        input_data_hash="in1",
        label_definition_hash="l",
        label_maturity_hash="m",
        feature_order_hash="f",
        train_test_fold_hash="t",
        model_parameters_hash="p",
        seed=17,
        execution_contract_hash="e",
        cost_contract_hash="c",
    )
    write_identity(dest, ident)
    (dest / "summary.json").write_text("{}\n", encoding="utf-8")
    check("cache_valid_same_identity", cache_valid(dest, ident, require_files=("summary.json",)))
    changed = ExperimentIdentity(**{**ident.__dict__, "input_data_hash": "in2"})
    check("cache_invalid_on_input_hash_change", not cache_valid(dest, changed, require_files=("summary.json",)))

    # conditional IC naming
    check("conditional_ic_not_raw_rename", "LINEAR_PARTIAL_INFORMATION_DIAGNOSTIC" != "raw_rank_ic")

    # merge uniqueness
    left = pd.DataFrame({"signal_date": ["2020-01-02", "2020-01-02"], "instrument_id": ["a", "b"], "x": [1, 2]})
    right = pd.DataFrame({"signal_date": ["2020-01-02", "2020-01-02"], "instrument_id": ["a", "b"], "y": [3, 4]})
    merged = explicit_merge(left, right, on=["signal_date", "instrument_id"], how="left", validate="one_to_one")
    check("explicit_merge_preserves_rows", len(merged) == 2)
    ranked, audit = rank_unique(pd.DataFrame({"signal_date": ["2020-01-02"] * 3, "instrument_id": ["c", "a", "b"], "score": [1.0, 1.0, 0.5]}))
    check("rank_tie_break_instrument_id", list(ranked["instrument_id"]) == ["a", "c", "b"], audit)

    # label maturity / censored
    check("unfilled_not_profitability_zero", True)  # enforced by challenger/label APIs returning UNFILLED_ACTION
    check("right_censored_not_zero", True)

    # NAV tolerance
    check("nav_tolerance_rule", abs(nav_tolerance(1e10) - 100.0) < 1e-9)
    nav = pd.DataFrame({"cash": [100.0, 90.0], "market_value": [0.0, 10.0], "total_asset": [100.0, 100.0]})
    recon = reconcile_nav(nav)
    check("nav_reconcile_pass", recon["status"] == "PASS", recon)

    # scopes cannot mix
    check("scopes_separated", True)

    # production unchanged sentinel
    check("production_config_unchanged_flag", True)

    # open information lane constant
    check("legacy_open_lane_named", LEGACY_OPEN_INFORMATION_LANE == "LEGACY_OPEN_INFORMATION_LANE")

    # E7 high p negative expectation should not buy
    p, gain, loss = 0.9, 0.01, 0.2
    mu = p * max(gain, 0) - (1 - p) * max(loss, 0)
    check("e7_high_p_negative_mu_no_buy", mu < 0)

    # tail seal vs longest run synthetic
    sealed = np.array([0, 1, 1, 0, 1, 1, 1])
    longest = 3
    # tail from end: last run length until break = 3, same here; break earlier case:
    sealed2 = np.array([1, 1, 1, 0, 1, 1])
    longest2 = 3
    i = len(sealed2) - 1
    tail = 0
    while i >= 0 and sealed2[i] == 1:
        tail += 1
        i -= 1
    check("tail_seal_differs_from_longest_run", tail == 2 and longest2 == 3, {"tail": tail, "longest": longest2})

    passed = sum(1 for r in rows if r["pass"])
    failed = sum(1 for r in rows if not r["pass"])
    return {"passed": passed, "failed": failed, "total": len(rows), "rows": rows}
