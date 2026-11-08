"""Round 23.5 测试。失败就保持失败，不写成 PASS。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.round23_5.literature_registry import PAPERS, papers
from core.round23_5.nested_validation import assert_no_future_fit, year_cutoff
from core.round23_5.oof_validation import mature_mask
from core.round23_5.paper_adapters.jensen import apply_action, s3_fill
from core.round23_5.paper_adapters.nardari import combination_weights
from core.round23_5.protocol import EXPERIMENTS, P1_ALPHAS, P3_LAMBDAS
from core.round23_5.statistics import block_indices


class LiteratureTests(unittest.TestCase):
    def test_exactly_ten_and_no_stat_papers(self):
        rows = papers()
        self.assertEqual(len(rows), 10)
        blob = " ".join(row["title"] + row["paper_id"] for row in rows)
        self.assertNotIn("Reality Check", blob)
        self.assertNotIn("Hansen", blob)

    def test_grades_frozen_before_pnl(self):
        grades = {row["paper_id"]: row["grade"] for row in PAPERS}
        self.assertEqual(grades["P01_NARDARI_2024"], "B")
        self.assertEqual(grades["P07_DEEPTRADER_2021"], "D")
        self.assertTrue(all(row["grade"] in {"A", "B", "C", "D"} for row in PAPERS))

    def test_selected_methods_match_real_ids(self):
        selected = {row["paper_id"] for row in EXPERIMENTS["experiments"]}
        self.assertEqual(selected, {"P01_NARDARI_2024", "P04_BAI_2026", "P03_JENSEN_2026"})


class MethodTests(unittest.TestCase):
    def test_nardari_simplex(self):
        rng = np.random.default_rng(17)
        gross = 1.0 + rng.normal(0.01, 0.04, size=(36, 3))
        weights, ok = combination_weights(gross, 0.99)
        self.assertTrue(ok)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue(np.all(weights >= -1e-8))

    def test_alpha_is_not_a_weight_grid(self):
        self.assertEqual(set(P1_ALPHAS), {"P1_BASE_A099", "P1_SENS_A100", "P1_SENS_A090"})
        self.assertEqual(len(P3_LAMBDAS), 3)

    def test_label_maturity_keeps_prediction_pool(self):
        frame = pd.DataFrame({
            "actual_exit_date": ["2018-01-10", "2019-02-01", None],
            "next2_date": ["2018-01-05", "2018-01-05", "2018-01-05"],
        })
        mask = mature_mask(frame, pd.Timestamp("2018-12-31"))
        self.assertListEqual(mask.tolist(), [True, False, False])

    def test_inner_outer_cutoff(self):
        assert_no_future_fit(2020, year_cutoff(2020))
        with self.assertRaises(RuntimeError):
            assert_no_future_fit(2020, pd.Timestamp("2020-06-01"))

    def test_s3_does_not_refill(self):
        result = s3_fill(["a", "b"], {"a"})
        self.assertEqual(result["planned"], ["a", "b"])
        self.assertEqual(result["filled"], ["b"])
        self.assertFalse(result["refilled"])

    def test_swap_does_not_drop_names(self):
        scores = pd.DataFrame({
            "signal_date": ["2020-01-02"] * 4,
            "instrument_id": ["a", "b", "c", "d"],
            "score": [4, 3, 2, 1],
        })
        swapped = apply_action(scores, "SWAP_TOP2_TOP3")
        self.assertEqual(len(swapped), 4)
        top1 = apply_action(scores, "TOP1_ONLY")
        self.assertEqual(top1["instrument_id"].tolist(), ["a"])

    def test_bootstrap_shared_index(self):
        idx = block_indices(50, 10, 4, 17)
        self.assertEqual(idx.shape, (4, 50))

    def test_production_flags(self):
        self.assertFalse(EXPERIMENTS["do_not_edit_after_outer_nav"] is False)


if __name__ == "__main__":
    unittest.main()
