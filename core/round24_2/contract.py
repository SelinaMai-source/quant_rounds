"""Round 24.2 冻结契约。"""

from __future__ import annotations

from core.paths import RESULT_ROOT

ROOT = RESULT_ROOT / "round24_2"
ROUND24 = RESULT_ROOT / "round24"
ROUND24_1 = RESULT_ROOT / "round24_1"

PERF_START = "2018-01-02"
PERF_END = "2026-07-24"
CAND_START = "2015-01-01"
MARKET_WARMUP_START = "2013-01-01"

CORE3 = (
    "amount_log",
    "return_5",
    "amount_vs_5",
)
TARGET7 = (
    "second_board_amount_log",
    "second_board_turnover_surprise",
    "second_board_return_5",
    "first_to_second_amount_ratio",
    "second_board_market_breadth",
    "second_board_first_touch_time",
    "second_board_tail_seal_fraction",
)

HYPOTHESES = (
    {"id": "H242_WARMUP", "claim": "2018 折可用 2015–2017 成熟样本，无 EMPTY 后与旧基线假比较", "pass_if": "training_history_repair 显示有效训练"},
    {"id": "H242_LABEL", "claim": "label_available_at 来自真实退出，RIGHT_CENSORED 不作 0", "pass_if": "label_execution_parity PASS"},
    {"id": "H242_MINUTE", "claim": "缺行业不删分钟列；TARGET7 含 F5/F6", "pass_if": "minute features 保留"},
    {"id": "H242_MATRIX", "claim": "相对 CASH/T0C/K3 区分少亏与正可实现收益", "pass_if": "主表完整"},
)
