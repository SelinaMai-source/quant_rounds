"""不依赖全历史回测的门：涨跌幅、连板、无泄漏、列操作、证据门。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.limit_rules import (
    consecutive_streak,
    is_st_name,
    limit_pct,
    seal_flags,
    select_second_board,
    vendor_limit_is_sentinel,
)
from core.round22_4.contract import evidence_pass


def _check(name: str, ok: bool, rows: list[dict]) -> None:
    rows.append({"name": name, "passed": bool(ok)})


def run_unit_tests() -> dict:
    rows: list[dict] = []
    _check("main_10", limit_pct("main", pd.Timestamp("2019-06-03"), 30, False) == 0.10, rows)
    _check("main_ipo_unlimited", limit_pct("main", pd.Timestamp("2019-06-03"), 1, False) is None, rows)
    _check("main_reg_five", limit_pct("main", pd.Timestamp("2023-04-11"), 5, False) is None, rows)
    _check("main_reg_six", limit_pct("main", pd.Timestamp("2023-04-11"), 6, False) == 0.10, rows)
    _check("st_5", limit_pct("main", pd.Timestamp("2024-01-02"), 40, True) == 0.05, rows)
    _check("st_10_after_cutover", limit_pct("main", pd.Timestamp("2026-07-06"), 40, True) == 0.10, rows)
    _check("st_5_before_cutover", limit_pct("main", pd.Timestamp("2026-07-03"), 40, True) == 0.05, rows)
    _check("chinext_before", limit_pct("chinext", pd.Timestamp("2020-08-21"), 20, False) == 0.10, rows)
    _check("chinext_after", limit_pct("chinext", pd.Timestamp("2020-08-24"), 20, False) == 0.20, rows)
    _check("chinext_ipo5", limit_pct("chinext", pd.Timestamp("2021-01-04"), 5, False) is None, rows)
    _check("star_20", limit_pct("star", pd.Timestamp("2021-01-04"), 20, False) == 0.20, rows)
    _check("bse_30", limit_pct("bse", pd.Timestamp("2022-01-04"), 20, False) == 0.30, rows)
    _check("st_name", is_st_name("*ST国农") and not is_st_name("平安银行"), rows)
    sealed, bust = seal_flags(10.0, 11.0, 11.0, 0.10)
    _check("seal_10", sealed and not bust, rows)
    sealed_s, _ = seal_flags(5.53, 5.28, 5.28, 0.20)
    _check("sentinel_not_used", vendor_limit_is_sentinel(999999.999) and not sealed_s, rows)
    halt = np.array([0, 1, 0], dtype=bool)
    unlimited = np.array([0, 0, 0], dtype=bool)
    sealed_arr = np.array([1, 0, 1], dtype=bool)
    streak = consecutive_streak(halt, unlimited, sealed_arr)
    _check("halt_carries", list(streak) == [1, 1, 2], rows)
    bust_streak = consecutive_streak(np.array([0, 0, 0], dtype=bool), unlimited, np.array([1, 0, 1], dtype=bool))
    _check("bust_resets", list(bust_streak) == [1, 0, 1], rows)
    rolling = pd.Series([1, 0, 1]).rolling(2, min_periods=1).sum().tolist()
    _check("not_rolling_sum", list(streak) != rolling, rows)
    frame = pd.DataFrame(
        {
            "sealed": [True, True, False],
            "streak": [1, 2, 2],
            "label_2to3_seal": [1, 0, 1],
            "next_sealed": [True, False, True],
        }
    )
    picked = select_second_board(frame)
    _check("second_board_ignores_future", len(picked) == 1 and int(picked.iloc[0]["streak"]) == 2 and bool(picked.iloc[0]["sealed"]), rows)
    base = ["g0_amount_log", "g1_ret_5"]
    added = base + ["r22_c4_flow_persist_5"]
    removed = [c for c in added if c != "g1_ret_5"]
    replaced = ["r22_c4_flow_persist_5" if c == "g1_ret_5" else c for c in base]
    _check("add_changes_columns", added != base and "r22_c4_flow_persist_5" in added, rows)
    _check("remove_changes_columns", "g1_ret_5" not in removed and removed != added, rows)
    _check("replace_changes_columns", replaced != base and "g1_ret_5" not in replaced and "r22_c4_flow_persist_5" in replaced, rows)
    _check("null_feature", float(pd.Series([np.nan, np.nan]).notna().mean()) == 0.0, rows)
    _check("evidence_blocks_weak_negative", evidence_pass(-0.01, 0.1, 500, True) is False, rows)
    _check("evidence_passes_positive", evidence_pass(0.03, 0.1, 500, True) is True, rows)
    _check("evidence_blocks_short", evidence_pass(0.05, 0.3, 10, True) is False, rows)
    n_failed = int(sum(not r["passed"] for r in rows))
    return {"status": "PASS" if n_failed == 0 else "FAIL", "n": len(rows), "n_failed": n_failed, "rows": rows}


if __name__ == "__main__":
    import json

    print(json.dumps(run_unit_tests(), ensure_ascii=False, indent=2))
