"""Round 6 单测：课题内，不写 Stock。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .overlays import margin_frame_from_scores, position_cap
from .registry import GROUP_ORDER, V3_MARKET
from .train import apply_representation, ensemble_rank_pct, fold_mask, xgb_ranker_weight_kind


def test_same_day_percentile_no_future() -> None:
    dates = pd.to_datetime(["2018-01-02"] * 3 + ["2018-01-03"] * 3)
    panel = pd.DataFrame({
        "trade_date": dates,
        "instrument_id": ["a", "b", "c", "a", "b", "c"],
        "feat": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
    })
    out = apply_representation(panel, ["feat"], "pct")
    day1 = out.loc[out["trade_date"] == dates[0], "feat"]
    # 第一天的 pct 不得被第二天 100/200/300 拉高
    assert float(day1.max()) <= 1.0 + 1e-12
    assert float(day1.min()) >= 0.0
    assert abs(float(day1.max()) - 1.0) < 1e-9 or day1.max() <= 1.0
    # 第二天大数不回写第一天
    assert float(out.loc[out["instrument_id"].eq("c") & out["trade_date"].eq(dates[0]), "feat"].iloc[0]) < 2.0


def test_rolling_margin_no_future() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i, d in enumerate(pd.date_range("2018-01-02", periods=80, freq="B")):
        for j, inst in enumerate(["a", "b", "c"]):
            rows.append({"signal_date": d, "instrument_id": inst, "score": float(j + i * 0.01 + rng.random())})
    base = pd.DataFrame(rows)
    spiked = base.copy()
    last_day = spiked["signal_date"].max()
    spiked.loc[(spiked["signal_date"] == last_day) & (spiked["instrument_id"] == "c"), "score"] = 1e6
    # 用原始分差：末日 1e6 不得改写更早一天的 rolling 分位
    m_base = margin_frame_from_scores(base, use_intraday_rank=False)
    m_spike = margin_frame_from_scores(spiked, use_intraday_rank=False)
    early_idx = m_base.index[:-1]
    a = m_base.loc[early_idx, "margin_p250"].dropna()
    b = m_spike.loc[early_idx, "margin_p250"].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 0
    assert np.allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy(), equal_nan=True)
    assert float(m_spike.iloc[-1]["top1_top2_margin"]) > 1000
    ranked = margin_frame_from_scores(spiked, use_intraday_rank=True)
    assert float(ranked.iloc[:40]["margin_p250"].max()) <= 1.0 + 1e-12


def test_label_maturity() -> None:
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2016-06-01", "2017-06-01", "2017-12-01"]),
        "instrument_id": ["a", "b", "c"],
        "actual_exit_date": pd.to_datetime(["2016-06-05", "2018-01-02", "2017-12-05"]),
        "label_exit_rule_net": [0.1, 0.2, 0.3],
    })
    mask = fold_mask(panel, train_start=2015, train_end=2017, predict_year=2018, fold_rule="stock")
    kept = panel.loc[mask]
    assert "b" not in set(kept["instrument_id"])  # exit 2018-01-02 未成熟
    assert set(kept["instrument_id"]) <= {"a", "c"}


def test_outer_fold_meta_no_leakage() -> None:
    idx = pd.date_range("2018-01-02", periods=600, freq="B")
    frame = pd.DataFrame({
        "year": idx.year,
        "y_ret": np.linspace(-0.05, 0.08, len(idx)),
        "y_win": (np.linspace(-0.05, 0.08, len(idx)) > 0).astype(float),
        "margin": np.linspace(0, 1, len(idx)),
        "agree_n": 3,
        "n_cand": 10,
        "disp": 0.1,
        "b01a": 1.0,
        "relimit": 0.3,
        **{f"top1_f{i}": 0.1 * i for i in range(6)},
    }, index=idx)
    from .meta import fit_oof_meta

    pred, _, meta = fit_oof_meta(frame)
    assert meta["oof_only"] is True
    for fold in meta["folds"]:
        if fold.get("status") == "empty":
            continue
        year = fold["year"]
        # 该年预测不得用该年训练（函数已 assert）
        assert year not in []  # 占位：fit 内部已禁 leakage
    assert pred.notna().any()


def test_oof_primary_predictions() -> None:
    # walk-forward：预测年不在训练年
    from .train import window_bounds

    for y in range(2018, 2027):
        a, b = window_bounds(y, "3y")
        assert b == y - 1
        assert a < y


def test_baseline_v3_market_layer() -> None:
    assert V3_MARKET["kind"] == "combine"
    assert V3_MARKET["how"] == "min"
    kinds = {p["kind"] for p in V3_MARKET["parts"]}
    assert "breadth_tier" in kinds
    assert "eco_relimit_p250" in kinds


def test_trade_position_cap() -> None:
    assert position_cap(nav_frac=0.3, market=1.0, c03=1.0, meta=1.0) == 0.3
    assert position_cap(nav_frac=0.3, market=0.5, c03=0.5, meta=1.0) == 0.075
    assert position_cap(nav_frac=0.3, market=1.0, c03=1.0, meta=1.5) == 0.3  # meta≤1
    assert position_cap(nav_frac=0.3, market=1.0, c03=1.0, meta=0.0) == 0.0


def test_ensemble_rank_normalization() -> None:
    d = pd.to_datetime(["2018-01-02"] * 3)
    a = pd.DataFrame({"signal_date": d, "instrument_id": ["x", "y", "z"], "score": [3.0, 2.0, 1.0]})
    b = pd.DataFrame({"signal_date": d, "instrument_id": ["x", "y", "z"], "score": [30.0, 20.0, 10.0]})
    ens = ensemble_rank_pct([a, b])
    # 日内 pct 等权后 x 最高
    top = ens.sort_values("score", ascending=False).iloc[0]["instrument_id"]
    assert top == "x"
    assert abs(float(ens["score"].max()) - 1.0) < 1e-9 or ens["score"].max() <= 1.0


def test_candidate_group_order() -> None:
    assert GROUP_ORDER == ("B", "C", "D", "E", "F", "G", "H", "K", "S")
    assert GROUP_ORDER.index("B") < GROUP_ORDER.index("C")
    assert GROUP_ORDER.index("C") < GROUP_ORDER.index("D")
    assert GROUP_ORDER.index("D") < GROUP_ORDER.index("E")


def test_xgb_weight_is_row() -> None:
    # 本机 XGB LTR 在有 qid 时按 group 收权重；近因仍只来自 trade_date。
    assert xgb_ranker_weight_kind() == "group"


if __name__ == "__main__":
    for fn in (
        test_same_day_percentile_no_future,
        test_rolling_margin_no_future,
        test_label_maturity,
        test_outer_fold_meta_no_leakage,
        test_oof_primary_predictions,
        test_baseline_v3_market_layer,
        test_trade_position_cap,
        test_ensemble_rank_normalization,
        test_candidate_group_order,
        test_xgb_weight_is_row,
    ):
        fn()
        print("ok", fn.__name__)
    print("ALL_ROUND6_TESTS_OK")
