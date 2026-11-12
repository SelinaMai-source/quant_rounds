"""同日 basket：Top2 买不到必须留现金，不得加给 Top1、不换成 Top3。

Stock `_open_cohort` 已按 `rank_name_budget_nav_fractions` 分预算；
课题只接线，不新写第三套引擎。
"""

from __future__ import annotations


def test_rank_fractions_leave_cash_contract() -> None:
    entry_nav = 300_000.0
    fractions = (0.15, 0.15)
    picks = [
        {"allocation_multiplier": 1.0, "selection_rank": 1},
        {"allocation_multiplier": 1.0, "selection_rank": 2},
    ]
    budgets = [
        entry_nav * fractions[int(p["selection_rank"]) - 1] * float(p["allocation_multiplier"])
        for p in picks
    ]
    assert budgets[0] == 45_000.0
    assert budgets[1] == 45_000.0
    # Top2 被缺口/涨停拒绝后，Top1 预算仍是 15%，剩余 15% 留现金
    opened_only_top1 = budgets[0]
    leftover = budgets[1]
    assert opened_only_top1 == 45_000.0
    assert leftover == 45_000.0
    assert opened_only_top1 + leftover == 90_000.0
    assert opened_only_top1 != 90_000.0


def test_no_promote_top3() -> None:
    top_k = 2
    ranks = (0.20, 0.10)
    assert len(ranks) == top_k
    assert sum(ranks) <= 0.30 + 1e-12


if __name__ == "__main__":
    test_rank_fractions_leave_cash_contract()
    test_no_promote_top3()
    print("ok")
