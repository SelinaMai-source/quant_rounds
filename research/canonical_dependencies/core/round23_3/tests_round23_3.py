"""Round23.3 单元/契约测试（≥部分关键门槛）。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.round23_3.blend import blend_xgb_lgbm
from core.round23_3.selection import CandidateRecord, choose_weight


class TestDefaultNotForcedWinner(unittest.TestCase):
    def test_non_default_can_win(self):
        default = CandidateRecord(
            key="t_0.90",
            is_default=True,
            inner_utility=0.001,
            distance_to_default=0.0,
            n_days=200,
            n_trades=80,
            maxdd=-0.1,
            mean_delta=None,
            block_deltas=[],
            boot_lb=None,
            all_gates_pass=True,
            window_fail=False,
            details={},
        )
        better = CandidateRecord(
            key="t_0.75",
            is_default=False,
            inner_utility=0.002,
            distance_to_default=0.15,
            n_days=200,
            n_trades=80,
            maxdd=-0.09,
            mean_delta=0.0005,
            block_deltas=[0.0004, 0.0006],
            boot_lb=0.0001,
            all_gates_pass=True,
            window_fail=False,
            details={},
        )
        worse_far = CandidateRecord(
            key="t_1.00",
            is_default=False,
            inner_utility=0.003,
            distance_to_default=0.10,
            n_days=200,
            n_trades=80,
            maxdd=-0.09,
            mean_delta=0.0004,
            block_deltas=[0.0003, 0.0005],
            boot_lb=0.0001,
            all_gates_pass=True,
            window_fail=False,
            details={},
        )
        out = choose_weight([default, better, worse_far], default_key="t_0.90")
        self.assertEqual(out["chosen"], "t_1.00")  # 更近默认且过门
        self.assertEqual(out["reason"], "QUALIFIED_NON_DEFAULT")

    def test_fallback_when_none_pass(self):
        default = CandidateRecord(
            key="w_0.00",
            is_default=True,
            inner_utility=0.001,
            distance_to_default=0.0,
            n_days=200,
            n_trades=80,
            maxdd=-0.1,
            mean_delta=None,
            block_deltas=[],
            boot_lb=None,
            all_gates_pass=True,
            window_fail=False,
            details={},
        )
        bad = CandidateRecord(
            key="w_0.25",
            is_default=False,
            inner_utility=0.002,
            distance_to_default=0.25,
            n_days=50,
            n_trades=10,
            maxdd=-0.3,
            mean_delta=-0.001,
            block_deltas=[-0.001, 0.0],
            boot_lb=-0.001,
            all_gates_pass=False,
            window_fail=False,
            details={},
        )
        out = choose_weight([default, bad], default_key="w_0.00")
        self.assertEqual(out["chosen"], "w_0.00")
        self.assertIn("FALLBACK_DEFAULT", out["reason"])


class TestZeroLgbmIdentity(unittest.TestCase):
    def test_zero_weight_identity(self):
        xgb = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
                "instrument_id": ["a", "b"],
                "score": [0.9, 0.1],
            }
        )
        lgbm = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
                "instrument_id": ["a", "b"],
                "score": [0.2, 0.8],
            }
        )
        out, fb = blend_xgb_lgbm(xgb, lgbm, 0.0)
        self.assertTrue(out.equals(xgb))
        self.assertEqual(len(fb), 0)


class TestMissingLgbmNoShrink(unittest.TestCase):
    def test_fallback_preserves_universe(self):
        xgb = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2020-01-02"] * 3),
                "instrument_id": ["a", "b", "c"],
                "score": [0.9, 0.5, 0.1],
            }
        )
        lgbm = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2020-01-02"] * 2),
                "instrument_id": ["a", "b"],
                "score": [0.2, 0.8],
            }
        )
        out, fb = blend_xgb_lgbm(xgb, lgbm, 0.25)
        self.assertEqual(set(out["instrument_id"]), {"a", "b", "c"})
        self.assertEqual(len(out), 3)
        self.assertGreaterEqual(len(fb), 1)


if __name__ == "__main__":
    unittest.main()
