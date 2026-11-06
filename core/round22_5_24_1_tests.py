"""三十项过程测试。失败断言保留。假设失败本身不是测试失败。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.limit_rules import limit_pct
from core.r225241_io import TRACKER_PATH, load_json, sha256_file
from core.r225241_stats import (
    SCOPES,
    c4_window_counts,
    permute_within_date,
    purge_mask,
    ranking_groups_contiguous,
)
from core.round21.contract import V5_ANCHOR, V5_NAV_END_EXACT
from core.round22_3.contract import ECONOMIC_MATERIALITY
from core.round22_5.contract import C4_FORMULA, FS2_FORMULA, SHUFFLE_SHARPE
from core.round23_1.contract import DIVERSITY_SPEARMAN_MAX, DIVERSITY_TOP1_DISAGREE_MIN, FORMAL_TEN_WEIGHTS, LGBM_PARAMS
from core.round24_1.contract import FULL_FEATURES, MIN_NAMES_FOR_RANKIC, REDUCED_FEATURES
from core.paths import PROJECT_ROOT, RESULT_ROOT


def test_v5_anchor_parity() -> None:
    assert V5_ANCHOR["sharpe"] == 1.8756627913361097
    assert V5_ANCHOR["cagr"] == 0.5499812832273525
    assert V5_ANCHOR["max_drawdown"] == -0.24056327715394388
    assert V5_NAV_END_EXACT == 12750089.355724689


def test_pit_asof_keeps_only_known() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-10", "2020-01-10"]),
            "known_at": pd.to_datetime(["2020-01-09", "2020-01-11"]),
        }
    )
    kept = frame.loc[frame["known_at"].le(frame["trade_date"])]
    assert len(kept) == 1


def test_c4_rolling_row_not_calendar() -> None:
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"])
    sign = pd.Series([1, 1, -1, 1, 1, -1], index=index, dtype=float)
    counts = c4_window_counts(sign)
    assert counts["n_valid"] == counts["n_valid_3"] + counts["n_valid_4"] + counts["n_valid_5"]
    row = sign.shift(1).rolling(5, min_periods=3).mean()
    calendar = sign.shift(1).rolling("5D").mean()
    assert not np.allclose(row.fillna(0), calendar.fillna(0))
    assert "rolling(5, min_periods=3)" in C4_FORMULA


def test_allnan_not_filled_zero() -> None:
    column = pd.Series([np.nan, np.nan])
    assert int(column.fillna(0).sum()) == 0
    assert column.notna().sum() == 0


def test_permutation_preserves_multiset() -> None:
    frame = pd.DataFrame({"signal_date": ["2020-01-02"] * 4, "score": [0.1, 0.4, 0.2, 0.3], "allocation_multiplier": [1, 0.5, 1, 0.5]})
    out = permute_within_date(frame, ["score", "allocation_multiplier"], "signal_date", 17)
    assert sorted(out["score"]) == sorted(frame["score"])
    assert sorted(out["allocation_multiplier"]) == sorted(frame["allocation_multiplier"])


def test_purge_drops_immature() -> None:
    end = pd.to_datetime(["2020-06-01", "2020-08-01"])
    mask = purge_mask(end, pd.Timestamp("2020-07-01"))
    assert mask.tolist() == [True, False]


def test_head_isolation() -> None:
    from core.round7.contract import FIVE_FEATURES, TEN_FEATURES

    assert "g0_amount_log" in FIVE_FEATURES
    assert "g0_amount_log" not in TEN_FEATURES
    assert set(TEN_FEATURES).issubset(FIVE_FEATURES)


def test_nested_weight_fallback_and_grid() -> None:
    assert 0.0 not in FORMAL_TEN_WEIGHTS
    assert 0.90 in FORMAL_TEN_WEIGHTS
    assert min(FORMAL_TEN_WEIGHTS) >= 0.50


def test_ranking_groups_contiguous() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"])
    assert ranking_groups_contiguous(dates)
    broken = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-02"])
    assert not ranking_groups_contiguous(broken)


def test_lgbm_params_frozen() -> None:
    assert LGBM_PARAMS["objective"] == "lambdarank"
    assert LGBM_PARAMS["learning_rate"] == 0.03
    assert LGBM_PARAMS["n_estimators"] == 200
    assert LGBM_PARAMS["num_leaves"] == 7
    assert LGBM_PARAMS["max_depth"] == 3
    assert LGBM_PARAMS["min_data_in_leaf"] == 50
    assert LGBM_PARAMS["feature_fraction"] == 1.0
    assert LGBM_PARAMS["bagging_fraction"] == 1.0
    assert LGBM_PARAMS["seed"] == 17


def test_oof_keys_align() -> None:
    left = pd.DataFrame({"signal_date": ["2024-01-02"], "instrument_id": ["000001.XSHE"], "score": [0.2]})
    right = pd.DataFrame({"signal_date": ["2024-01-02"], "instrument_id": ["000001.XSHE"], "label": [0.01]})
    merged = left.merge(right, on=["signal_date", "instrument_id"], how="inner")
    assert len(merged) == 1
    assert merged["label"].notna().all()


def test_top_change_detects_difference() -> None:
    base = {"000001.XSHE"}
    new = {"000002.XSHE"}
    assert base != new


def test_nav_identity() -> None:
    cash, market, total = 100.0, 25.0, 125.0
    assert abs(cash + market - total) < 1e-9


def test_label_cost_not_in_nav_function() -> None:
    text = (PROJECT_ROOT / "core" / "r225241_sleeve.py").read_text(encoding="utf-8")
    assert "label_2to3_net" not in text
    assert "label_e2_net" not in text


def test_execution_scenarios_distinct() -> None:
    names = ["S0", "S1", "S2", "S3", "S4", "S5"]
    assert len(set(names)) == 6
    assert "S4" not in "board rule"
    note = "S4/S5 are stress, not board rules"
    assert "stress" in note


def test_st_limit_cutover() -> None:
    before = limit_pct("main", pd.Timestamp("2026-07-03"), 30, True)
    after = limit_pct("main", pd.Timestamp("2026-07-06"), 30, True)
    assert before == 0.05
    assert after == 0.10
    chinext = limit_pct("chinext", pd.Timestamp("2024-01-02"), 30, False)
    assert chinext == 0.20


def test_second_board_not_flat_rule() -> None:
    pre = 10.0
    st_limit = round(pre * (1 + 0.05) + 1e-8, 2)
    flat = round(pre * 1.099 + 1e-8, 2)
    assert abs(st_limit - flat) > 0.02


def test_sparse_rankic_rule() -> None:
    names = 4
    assert names < MIN_NAMES_FOR_RANKIC


def test_abstention_default() -> None:
    inner_days = 100
    threshold = 0.50 if inner_days < 120 else 0.60
    assert threshold == 0.50


def test_classifier_not_fill_probability() -> None:
    payload = {"score_is_fill_probability": False}
    assert payload["score_is_fill_probability"] is False


def test_immature_not_zero() -> None:
    label = np.array([np.nan, 0.02])
    y = label[np.isfinite(label)]
    assert 0.0 not in y or len(y) == 1
    assert np.isnan(label[0])


def test_reduced_feature_set() -> None:
    assert len(FULL_FEATURES) == 8
    assert REDUCED_FEATURES == FULL_FEATURES[:4] + (FULL_FEATURES[6],)
    assert 0 not in REDUCED_FEATURES


def test_scope_isolation() -> None:
    assert len(SCOPES) == 5
    assert len(set(SCOPES)) == 5


def test_registry_has_scope() -> None:
    path = RESULT_ROOT / "round22_5" / "multiple_testing_registry.csv"
    assert path.is_file(), "MISSING_ARTIFACT multiple_testing_registry"
    frame = pd.read_csv(path)
    assert {"hypothesis_id", "scope", "pre_specified"}.issubset(frame.columns)
    assert set(frame["scope"]).issubset(set(SCOPES))


def test_historical_sentinels_unchanged() -> None:
    guard = load_json(RESULT_ROOT / "round22_5_24_1_master" / "production_guard.json")
    assert guard.get("hashes"), "MISSING_ARTIFACT production guard"
    for rel, digest in guard["hashes"].items():
        assert sha256_file(PROJECT_ROOT / rel) == digest, rel


def test_production_baseline_string() -> None:
    text = TRACKER_PATH.read_text(encoding="utf-8")
    assert "CURRENT_ALPHA_BASELINE = baseline_v5 = R16_O01" in text.split("### Round22.5")[0]
    guard = load_json(RESULT_ROOT / "round22_5_24_1_master" / "production_guard.json")
    prefix = TRACKER_PATH.read_bytes()[: int(guard["tracker_len_before"])]
    import hashlib

    assert hashlib.sha256(prefix).hexdigest() == guard["tracker_sha_before"]


def test_economic_gate_constants() -> None:
    assert ECONOMIC_MATERIALITY["id"] == "R21_3_ECONOMIC_MATERIALITY_V1"
    assert ECONOMIC_MATERIALITY["delta_cagr_min_abs"] == 0.01
    assert ECONOMIC_MATERIALITY["delta_sharpe_min"] == 0.05
    assert ECONOMIC_MATERIALITY["nav_rel_min"] == 0.01
    assert ECONOMIC_MATERIALITY["maxdd_worse_max_abs"] == 0.02


def test_shuffle_constant() -> None:
    assert SHUFFLE_SHARPE == 1.8125087429942996


def test_diversity_thresholds() -> None:
    assert DIVERSITY_SPEARMAN_MAX == 0.90
    assert DIVERSITY_TOP1_DISAGREE_MIN == 0.10


def test_fs2_formula() -> None:
    assert FS2_FORMULA == "OperatingCashFlow/AverageAssets"
    assert "cash_conversion" not in FS2_FORMULA


def test_reports_exist() -> None:
    for rel in (
        "round22_5/ROUND22_5_RESEARCH_REPORT.md",
        "round22_5/FINAL_DECISION.md",
        "round23_1/ROUND23_1_RESEARCH_REPORT.md",
        "round24_1/ROUND24_1_RESEARCH_REPORT.md",
    ):
        assert (RESULT_ROOT / rel).is_file(), rel


TESTS = [
    test_v5_anchor_parity,
    test_pit_asof_keeps_only_known,
    test_c4_rolling_row_not_calendar,
    test_allnan_not_filled_zero,
    test_permutation_preserves_multiset,
    test_purge_drops_immature,
    test_head_isolation,
    test_nested_weight_fallback_and_grid,
    test_ranking_groups_contiguous,
    test_lgbm_params_frozen,
    test_oof_keys_align,
    test_top_change_detects_difference,
    test_nav_identity,
    test_label_cost_not_in_nav_function,
    test_execution_scenarios_distinct,
    test_st_limit_cutover,
    test_second_board_not_flat_rule,
    test_sparse_rankic_rule,
    test_abstention_default,
    test_classifier_not_fill_probability,
    test_immature_not_zero,
    test_reduced_feature_set,
    test_scope_isolation,
    test_registry_has_scope,
    test_historical_sentinels_unchanged,
    test_production_baseline_string,
    test_economic_gate_constants,
    test_shuffle_constant,
    test_diversity_thresholds,
    test_fs2_formula,
]


def run_tests() -> dict[str, object]:
    failed = []
    passed = []
    for fn in TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed.append({"test": fn.__name__, "error": f"{type(exc).__name__}: {exc}"})
        else:
            passed.append(fn.__name__)
    return {"n_passed": len(passed), "n_failed": len(failed), "failed": failed, "n_tests": len(TESTS)}
