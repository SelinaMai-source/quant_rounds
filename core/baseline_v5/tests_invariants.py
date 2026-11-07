"""锁定 baseline_v5 不变量。"""

from __future__ import annotations

from .contract import (
    ALIAS,
    FEE_COMPONENTS,
    FEE_SCHEDULE,
    FIVE_FEATURES,
    FUSION_WEIGHTS,
    IDENTITY,
    O01_FORMULA,
    TEN_FEATURES,
    V5_ANCHOR,
    XGB_PARAMS,
    frozen_feature_hashes,
)


def test_identity():
    assert IDENTITY == "baseline_v5"
    assert ALIAS == "R16_O01"


def test_anchor_numbers():
    assert abs(V5_ANCHOR["sharpe"] - 1.8756627913361097) < 1e-12
    assert abs(V5_ANCHOR["cagr"] - 0.5499812832273525) < 1e-12
    assert abs(V5_ANCHOR["max_drawdown"] - (-0.24056327715394388)) < 1e-12
    assert abs(V5_ANCHOR["nav_end"] - 12750089.355724689) < 1e-6
    assert V5_ANCHOR["completed_trades"] == 1107
    assert V5_ANCHOR["initial_capital"] == 300000.0


def test_features():
    assert len(FIVE_FEATURES) == 14
    assert len(TEN_FEATURES) == 13
    assert "g0_amount_log" in FIVE_FEATURES
    assert "g0_amount_log" not in TEN_FEATURES
    assert set(TEN_FEATURES).issubset(set(FIVE_FEATURES))
    assert FIVE_FEATURES[-1] == "d1_open_gap"
    h = frozen_feature_hashes()
    assert len(h["five14"]) == 64
    assert len(h["ten13"]) == 64


def test_blend_and_fees():
    assert FUSION_WEIGHTS == {"ten": 0.90, "five": 0.10}
    assert FEE_SCHEDULE == "ROUND7_FLAT_5BP"
    assert FEE_COMPONENTS["slippage_bps"] == 12.3
    assert FEE_COMPONENTS["commission_bps"] == 1.0
    assert FEE_COMPONENTS["transfer_fee_bps"] == 0.1
    assert FEE_COMPONENTS["sell_tax_bps"] == 5.0


def test_xgb_no_sampling():
    assert XGB_PARAMS["subsample"] == 1.0
    assert XGB_PARAMS["colsample_bytree"] == 1.0
    assert XGB_PARAMS["n_estimators"] == 100
    assert XGB_PARAMS["max_depth"] == 2
    assert XGB_PARAMS["learning_rate"] == 0.05


def test_o01_formula():
    assert "PUT_SKEW25" in O01_FORMULA
    assert "apply_top2_veto" in O01_FORMULA


def run_all() -> dict:
    tests = [
        test_identity,
        test_anchor_numbers,
        test_features,
        test_blend_and_fees,
        test_xgb_no_sampling,
        test_o01_formula,
    ]
    failed = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed.append({"test": fn.__name__, "error": str(exc)})
    return {"passed": len(tests) - len(failed), "failed": failed, "total": len(tests)}


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), ensure_ascii=False, indent=2))
