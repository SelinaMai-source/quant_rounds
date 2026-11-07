"""Round 7 contract and causality tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import ROUND7_ROOT, assert_project_result_path
from .blend import fuse_predictions
from .contract import FIVE_HEAD, TEN_HEAD, model_parameters
from .risk import VALID_MULTIPLIERS, compute_daily_allocations
from .train import assert_seed_rank_invariance, ensemble_rank_pct


def _pred(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "signal_date": pd.to_datetime(["2020-01-02"] * len(scores)),
        "instrument_id": [f"s{i}" for i in range(len(scores))],
        "score": scores,
    })


def test_contract() -> None:
    assert FIVE_HEAD.label_end_column == "next2_date"
    assert TEN_HEAD.label_end_column == "actual_exit_date"
    assert FIVE_HEAD.rank_n_bins == 5 and FIVE_HEAD.eval_metric == "ndcg@1"
    assert TEN_HEAD.rank_n_bins == 10 and TEN_HEAD.eval_metric == "ndcg@2"
    params = model_parameters(17)
    assert params["subsample"] == 1.0 and params["colsample_bytree"] == 1.0
    assert params["n_estimators"] == 100 and params["max_depth"] == 2


def test_label_maturity_boundary() -> None:
    label_end = pd.Series(pd.to_datetime(["2019-12-31", "2020-01-01"]))
    eligible = label_end.dt.year.le(2019)
    assert eligible.tolist() == [True, False]


def test_seed_invariance_and_ensemble() -> None:
    a = _pred([3.0, 2.0, 1.0])
    b = _pred([30.0, 20.0, 10.0])
    assert assert_seed_rank_invariance([a, b])["passed"]
    ensemble = ensemble_rank_pct([a, b])
    assert ensemble.iloc[0]["instrument_id"] == "s0"
    try:
        assert_seed_rank_invariance([a, _pred([1.0, 2.0, 3.0])])
    except RuntimeError:
        pass
    else:
        raise AssertionError("rank mismatch must stop the pipeline")


def test_gap_before_fusion_and_stable_tie() -> None:
    five = _pred([3.0, 2.0, 1.0]).rename(columns={"score": "score_five"})
    ten = _pred([1.0, 2.0, 3.0]).rename(columns={"score": "score_ten"})
    gaps = five[["signal_date", "instrument_id"]].copy()
    gaps["d1_open_gap"] = [-0.02, 0.0, 0.0]
    output = fuse_predictions(five, ten, gaps)
    assert set(output["instrument_id"]) == {"s1", "s2"}
    assert output.iloc[0]["instrument_id"] == "s2"
    assert np.isclose(output.iloc[0]["score"], 0.9 * 1.0 + 0.1 * 0.5)


def _daily_scores(include_future: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2018-01-02", "2021-12-31", freq="B")
    frame = pd.DataFrame({
        "signal_date": dates,
        "auction_score": np.linspace(0.0, 1.0, len(dates)),
        "style_score": np.linspace(1.0, 0.0, len(dates)),
    })
    if include_future:
        frame = pd.concat([frame, pd.DataFrame({
            "signal_date": pd.to_datetime(["2025-01-02"]),
            "auction_score": [999.0],
            "style_score": [-999.0],
        })], ignore_index=True)
    return frame


def _market(include_future: bool = False) -> pd.DataFrame:
    scores = _daily_scores(include_future)
    return pd.DataFrame({
        "trade_date": scores["signal_date"],
        "market_up_ratio": np.where(np.arange(len(scores)) % 2 == 0, 0.50, 0.40),
        "market_limit_count": 20 + np.arange(len(scores)) % 7,
    })


def test_thresholds_ignore_future() -> None:
    _, before = compute_daily_allocations(_daily_scores(False), _market(False))
    state, after = compute_daily_allocations(_daily_scores(True), _market(True))
    assert before["style_low_threshold"] == after["style_low_threshold"]
    assert before["auction_high_threshold"] == after["auction_high_threshold"]
    observed = set(state.loc[state["allocation_multiplier"].gt(0), "allocation_multiplier"].unique())
    assert observed <= VALID_MULTIPLIERS


def test_round7_write_guard() -> None:
    assert assert_project_result_path(ROUND7_ROOT / "unit-test" / "x.json")
    assert "/root/Stock/runs" not in str(FIVE_HEAD.panel_dir)
    assert "/root/Stock/runs" not in str(TEN_HEAD.panel_dir)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("ALL_ROUND7_TESTS_OK")
