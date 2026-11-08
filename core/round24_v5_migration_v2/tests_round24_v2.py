"""R24 V2 时点连接测试。"""

from __future__ import annotations

import pandas as pd

from core.round24_v5_migration_v2.asof import latest_visible, wrong_argmax_update_time
from core.round24_v5_migration_v2.contract import FIVE_THEORY, TEN_THEORY


def test_asof_ignores_future_version():
    frame = pd.DataFrame(
        {
            "trade_date": ["2018-01-02", "2018-01-02"],
            "instrument_id": ["000001.SZ", "000001.SZ"],
            "known_at": ["2018-01-02 22:00", "2026-01-01 22:00"],
            "update_time": ["2018-01-03", "2026-01-02"],
            "value": [1.0, 9.0],
        }
    )
    visible = latest_visible(frame, pd.Timestamp("2018-01-02 22:00"))
    leaked = wrong_argmax_update_time(frame)
    assert float(visible["value"].iloc[0]) == 1.0
    assert float(leaked["value"].iloc[0]) == 9.0


def test_ten_excludes_amount_log():
    assert "g0_amount_log" in FIVE_THEORY
    assert "g0_amount_log" not in TEN_THEORY
    assert len(TEN_THEORY) == len(FIVE_THEORY) - 1


if __name__ == "__main__":
    test_asof_ignores_future_version()
    test_ten_excludes_amount_log()
    print("round24 v2 tests passed")
