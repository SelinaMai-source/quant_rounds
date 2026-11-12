"""R23.4 选择器与门禁测试。不依赖全历史回测。"""

from __future__ import annotations

import numpy as np

from core.research_integrity_v3.gates import GateResult
from core.round23_4.exposure import exposure_ratio, scale_multipliers
from core.round23_4.select_risk import choose_r2, choose_r3
import pandas as pd


def test_pass_requires_hash():
    try:
        GateResult("g", "PASS")
    except ValueError:
        return
    raise AssertionError("PASS without artifact must fail")


def test_default_not_in_distance_sort():
    records = [
        {"key": "l_0.00", "w_lgbm": 0.0, "n_days": 200, "n_trades": 80, "mean_log_growth": 0.001, "window_fail": False, "block_maxdd": [-0.1, -0.1]},
        {"key": "l_0.25", "w_lgbm": 0.25, "n_days": 200, "n_trades": 80, "mean_log_growth": 0.002, "window_fail": False, "block_mean_deltas": [0.001, 0.001], "boot_lb": 0.0001, "block_maxdd": [-0.1, -0.1]},
        {"key": "l_0.10", "w_lgbm": 0.10, "n_days": 200, "n_trades": 80, "mean_log_growth": 0.0015, "window_fail": False, "block_mean_deltas": [0.001, 0.001], "boot_lb": 0.0001, "block_maxdd": [-0.1, -0.1]},
    ]
    chosen = choose_r2(records)
    assert chosen["chosen"] == "l_0.10"


def test_r3_log_growth_is_not_one_pct_cagr():
    records = [
        {
            "key": "l_0.00",
            "w_lgbm": 0.0,
            "n_days": 200,
            "n_trades": 80,
            "mean_ann_log_growth": 0.40,
            "mean_ui": 0.10,
            "block_ui": [0.10, 0.10],
            "block_maxdd": [-0.20, -0.20],
            "window_fail": False,
        },
        {
            "key": "l_0.10",
            "w_lgbm": 0.10,
            "n_days": 200,
            "n_trades": 80,
            "mean_ann_log_growth": 0.385,
            "mean_ui": 0.09,
            "block_ui": [0.09, 0.09],
            "block_maxdd": [-0.20, -0.20],
            "window_fail": False,
            "no_leverage": True,
        },
    ]
    # 年化 log 损失 0.015 > 0.01，不能因为小于 1 个百分点 CAGR 就通过。
    assert choose_r3(records)["chosen"] == "l_0.00"


def test_r3_accepts_ui_improvement_within_log_tolerance():
    records = [
        {
            "key": "l_0.00",
            "w_lgbm": 0.0,
            "n_days": 200,
            "n_trades": 80,
            "mean_ann_log_growth": 0.40,
            "mean_ui": 0.10,
            "block_ui": [0.10, 0.10],
            "block_maxdd": [-0.20, -0.20],
            "window_fail": False,
        },
        {
            "key": "l_0.10",
            "w_lgbm": 0.10,
            "n_days": 200,
            "n_trades": 80,
            "mean_ann_log_growth": 0.395,
            "mean_ui": 0.094,
            "block_ui": [0.094, 0.094],
            "block_maxdd": [-0.19, -0.19],
            "window_fail": False,
            "no_leverage": True,
        },
    ]
    assert choose_r3(records)["chosen"] == "l_0.10"


def test_zero_position_not_reopened():
    frame = pd.DataFrame({"signal_date": ["2018-01-02"], "instrument_id": ["000001.SZ"], "score": [1.0], "allocation_multiplier": [0.0]})
    out = scale_multipliers(frame, 0.5)
    assert float(out["allocation_multiplier"].iloc[0]) == 0.0


def test_ratio_clip_and_zero_denominator():
    assert exposure_ratio(0.2, 0.4)["ratio"] == 0.5
    assert exposure_ratio(0.5, 0.4)["ratio"] == 1.0
    bad = exposure_ratio(0.2, 0.0)
    assert bad["ratio"] == 1.0 and bad["identifiable"] is False


def test_drawdown_not_stitched():
    from core.research_integrity_v3.risk_metrics import ulcer_index
    import pandas as pd

    # 两段重置净值不能当成一条价格：第二段从 1 重新开始，拼接后的峰值会被第一段污染。
    a = pd.Series([0.0, 0.1])
    b = pd.Series([0.0, -0.05])
    wealth_a = (1 + a).cumprod()
    wealth_b = (1 + b).cumprod()
    stitched = pd.concat([wealth_a, wealth_b * wealth_a.iloc[-1]], ignore_index=True)
    separate_dd = 1 - wealth_b / wealth_b.cummax()
    stitched_dd = 1 - stitched / stitched.cummax()
    assert abs(float(ulcer_index(separate_dd)) - float(ulcer_index(stitched_dd))) > 1e-12
    assert float(wealth_b.iloc[0]) == 1.0


if __name__ == "__main__":
    test_pass_requires_hash()
    test_default_not_in_distance_sort()
    test_r3_log_growth_is_not_one_pct_cagr()
    test_r3_accepts_ui_improvement_within_log_tolerance()
    test_zero_position_not_reopened()
    test_ratio_clip_and_zero_denominator()
    test_drawdown_not_stitched()
    print("round23_4 tests passed")
