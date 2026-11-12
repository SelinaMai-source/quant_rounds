"""Round23.3 + Round24 共享完整性测试清单（Part D）。"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from core.round23_3.blend import blend_xgb_lgbm
from core.round23_3.contract import ROOT as R233
from core.round23_3.selection import CandidateRecord, choose_weight
from core.round24_v5_migration.contract import ROOT as R24M
from core.round7.contract import FIVE_FEATURES, TEN_FEATURES


class TestPartDIntegrity(unittest.TestCase):
    def test_feature_order_ten13(self):
        self.assertEqual(len(FIVE_FEATURES), 14)
        self.assertEqual(len(TEN_FEATURES), 13)
        self.assertNotIn("g0_amount_log", TEN_FEATURES)

    def test_zero_lgbm_identity(self):
        xgb = pd.DataFrame({"signal_date": pd.to_datetime(["2020-01-02"]), "instrument_id": ["a"], "score": [1.0]})
        out, _ = blend_xgb_lgbm(xgb, xgb, 0.0)
        self.assertTrue(out.equals(xgb))

    def test_default_not_forced_winner(self):
        default = CandidateRecord("t_0.90", True, 0.001, 0.0, 200, 80, -0.1, None, [], None, True, False, {})
        alt = CandidateRecord("t_0.75", False, 0.002, 0.15, 200, 80, -0.09, 0.001, [0.001, 0.001], 0.0001, True, False, {})
        out = choose_weight([default, alt], default_key="t_0.90")
        self.assertEqual(out["chosen"], "t_0.75")

    def test_production_flags(self):
        for root in (R233, R24M):
            dec = root / "decision.json"
            if dec.is_file():
                import json

                body = json.loads(dec.read_text())
                self.assertFalse(body.get("current_baseline_changed", True))
                self.assertFalse(body.get("auto_promote", True))

    def test_no_d1_in_v5_migrated_manifest(self):
        path = R24M / "feature_set_manifest.json"
        if path.is_file():
            import json

            man = json.loads(path.read_text())
            self.assertNotIn("d1_open_gap", man.get("V5_MIGRATED") or [])

    def test_inner_window_not_full_history(self):
        path = R233 / "inner_fix.json"
        if path.is_file():
            import json

            body = json.loads(path.read_text())
            self.assertEqual(body.get("status"), "PASS")
            n = (body.get("smoke") or {}).get("n_nav_days")
            if n is not None:
                self.assertLessEqual(int(n), 200)


if __name__ == "__main__":
    unittest.main()
