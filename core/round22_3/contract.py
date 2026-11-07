"""Round 22.3 冻结契约。两本账不得混表。看结果前冻结权威经济门。不改交易引擎，不升 v6。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND22_1_ROOT, ROUND22_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round21.contract import (
    AUTO_PROMOTE_CURRENT,
    COLSAMPLE,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    FEE_SCHEDULE,
    HERMES_REBUILD_RAW,
    HERMES_V7,
    NAV_END_ABS_TOL,
    O01_FORMULA,
    PARITY_METRIC_ABS,
    SUBSAMPLE,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    XGB_N_JOBS,
)
from ..round21_4.contract import ECONOMIC_MATERIALITY as R21_3_GATE
from ..round22.contract import FACTOR_SPECS as R22_FACTOR_SPECS
from ..round22.contract import OOF_INCREMENT_GATE as R22_OOF_GATE

ROUND22_2_ROOT = RESULT_ROOT / "round22_2"
ROUND22_3_ROOT = RESULT_ROOT / "round22_3"

KEEP_BASELINE_V5 = "KEEP_BASELINE_V5"
RECOMMEND_V6_RESEARCH_CANDIDATE = "RECOMMEND_V6_RESEARCH_CANDIDATE"
V6_RESEARCH_CANDIDATE = "V6_RESEARCH_CANDIDATE"
SUBPERIOD_ALPHA_RESEARCH_CANDIDATE = "SUBPERIOD_ALPHA_RESEARCH_CANDIDATE"
AUTO_PROMOTE_CURRENT = AUTO_PROMOTE_CURRENT
CURRENT_BASELINE = CURRENT_BASELINE
CURRENT_ALIAS = CURRENT_ALIAS
NOT_TRIGGERED = "NOT_TRIGGERED"
BLOCKED_PIT = "BLOCKED_PIT"
BLOCKED_DATA = "BLOCKED_DATA"
BLOCKED_ACCESS = "BLOCKED_ACCESS"
BLOCKED_SEMANTICS = "BLOCKED_SEMANTICS"
BLOCKED_PERIOD_MISMATCH = "BLOCKED_PERIOD_MISMATCH"
BLOCKED = "BLOCKED"
FAILED = "FAILED"
AUDIT_FAILURE = "AUDIT_FAILURE"
PARTIAL = "PARTIAL"
COMPLETE = "COMPLETE"
NOT_RUN = "NOT_RUN"
PIT_PASS = "PIT_PASS"
DUPLICATE = "DUPLICATE"
DUPLICATE_EXPRESSION = "DUPLICATE_EXPRESSION"
CROSS_SECTIONALLY_DEGENERATE = "CROSS_SECTIONALLY_DEGENERATE"
COMPOSITE_OF_KNOWN_FACTORS = "COMPOSITE_OF_KNOWN_FACTORS"
EXPLORATORY_DIAGNOSTIC = "EXPLORATORY_DIAGNOSTIC"
NOT_APPLICABLE = "NOT_APPLICABLE"
NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
IMPORTED_BY_HASH = "IMPORTED_BY_HASH"
TRAINED_REAL = "TRAINED_REAL"
REPEATED_HISTORICAL_TESTING = "REPEATED_HISTORICAL_TESTING"
FULL_HISTORY_STRICT_PIT = "FULL_HISTORY_STRICT_PIT"
SUBPERIOD_STRICT_PIT = "SUBPERIOD_STRICT_PIT"
LAGGED_SAFE_RESEARCH = "LAGGED_SAFE_RESEARCH"
RESEARCH_ONLY_BACKFILL = "RESEARCH_ONLY_BACKFILL"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
NON_CANONICAL_EXPLORATORY = "NON_CANONICAL_EXPLORATORY"
SAME_OPEN_EXECUTION_OPTIMISM = "SAME_OPEN_EXECUTION_OPTIMISM"
HISTORICAL_FEE_CORRECTED = "HISTORICAL_FEE_CORRECTED"
MATCHED_SUBPERIOD_NOT_COMPARABLE = "MATCHED_SUBPERIOD_NOT_COMPARABLE"
SEMANTICALLY_IDENTICAL_HASH_MISMATCH = "SEMANTICALLY_IDENTICAL_HASH_MISMATCH"
ECONOMIC_GATE_SEMANTIC_DRIFT = "ECONOMIC_GATE_SEMANTIC_DRIFT"
POSITIVE_BUT_IMMATERIAL = "POSITIVE_BUT_IMMATERIAL"
INCREMENTAL_ALPHA_SUPPORTED = "INCREMENTAL_ALPHA_SUPPORTED"
NOT_STATISTICALLY_SUPPORTED = "NOT_STATISTICALLY_SUPPORTED"
SAMPLE_DEPENDENT = "SAMPLE_DEPENDENT"
EXECUTION_FRAGILE = "EXECUTION_FRAGILE"
NOT_EVALUATED = "NOT_EVALUATED"

FIVE_FEATURES = tuple(FIVE_FEATURES)
TEN_FEATURES = tuple(TEN_FEATURES)
FIVE_ONLY_FEATURE = "g0_amount_log"
FIVE_HEAD = FIVE_HEAD
TEN_HEAD = TEN_HEAD
SEEDS = tuple(SEEDS)
SUBSAMPLE = SUBSAMPLE
COLSAMPLE = COLSAMPLE
XGB_N_JOBS = XGB_N_JOBS
FEE_SCHEDULE = FEE_SCHEDULE
V5_ANCHOR = dict(V5_ANCHOR)
V5_NAV_END_EXACT = V5_NAV_END_EXACT
NAV_END_ABS_TOL = NAV_END_ABS_TOL
PARITY_METRIC_ABS = PARITY_METRIC_ABS
O01_FORMULA = O01_FORMULA
FUSION_WEIGHTS = {"ten": 0.90, "five": 0.10}
HERMES_V7 = HERMES_V7
HERMES_REBUILD_RAW = HERMES_REBUILD_RAW
MARKET_CUTOFF = "2026-07-24"
DECISION_AUDIT = "decision=D_close; order=D+1_open; same_open_execution_optimism=True"
HOLDER_NUM_PARQUET = Path("/home/mxy/AmazingData/data/amazingdata_raw/holder_num/all.parquet")
C_PANEL_22_2 = ROUND22_2_ROOT / "data" / "c_factor_panel.parquet"
HN2_PANEL_22_2 = ROUND22_2_ROOT / "data_recovery" / "hn2_panel.parquet"

PIT_MIN_FINITE = 0.80
REDUNDANCY_RHO = 0.60
KEYS = ["trade_date", "instrument_id"]
EPS = 1e-12
LABELS = ("label_e2_net", "label_exit_rule_net")
E0_CODE = "R223_E0"
IC_DAILY_MIN = 5
IC_AGG_MIN = 60
BLOCK_DAYS = 20
BOOT_N = 5000
SPA_BOOT = 5000
RIDGE_ALPHA = 1.0
MIN_TRAIN_YEARS = 3
FIRST_PREDICTION_YEAR = 2018
LAST_PREDICTION_YEAR = 2026
MAIN_WINDOW_START = "2018-01-02"
MAIN_WINDOW_END = "2026-07-24"
MAIN_N_OBS = 2075
SHORT_WINDOW_START = "2023-01-03"
SHORT_WINDOW_END = "2026-07-24"
SHORT_N_OBS = 860
C_AVAILABLE_FROM = "2020-09-24"
C_AVAILABLE_N_DATES = 1394

C1_ID = "r22_c1_xl_net_imbalance"
C1_LAG1_ID = "r22_2_c1_xl_net_imbalance_lag1"
C4_ID = "r22_c4_flow_persist_5"
HN2_ID = "r22_2_hn2_avg_holding_chg"
C1_FORMULA_FROZEN = "NET_FLOW_XL / TURNOVER_VALUE"
C4_FORMULA_FROZEN = "mean(sign(NET_FLOW)_{D-5..D-1})"
HN2_FORMULA_FROZEN = "Δlog(then_known_circ_shares / HOLDER_NUM); known_at=ANN_DT next trading day"
C1_NO_BUY_VALUE_XL = True

MAIN_LEDGER = {
    "id": "MAIN_2018_2026",
    "window_start": MAIN_WINDOW_START,
    "window_end": MAIN_WINDOW_END,
    "n_obs": MAIN_N_OBS,
    "n_trades": 1107,
    "compare_to": "R16_O01",
    "sharpe": V5_ANCHOR["sharpe"],
    "cagr": V5_ANCHOR["cagr"],
    "max_drawdown": V5_ANCHOR["max_drawdown"],
    "nav_end": V5_NAV_END_EXACT,
    "official": True,
}
SHORT_LEDGER = {
    "id": "R222_MATCHED_2023_2026",
    "window_start": SHORT_WINDOW_START,
    "window_end": SHORT_WINDOW_END,
    "n_obs": SHORT_N_OBS,
    "n_trades": 473,
    "compare_to": "R222_E0",
    "sharpe": 2.734502736378759,
    "cagr": 1.1523509512291934,
    "nav_end": 4573035.628716091,
    "official": False,
    "status": MATCHED_SUBPERIOD_NOT_COMPARABLE,
    "note": "High Sharpe is the window, not a better strategy. Do not mix with R16_O01.",
}

# 权威阈值对象：只含 21.3/21.4 数字门槛，不含 round 字符串。
THRESHOLD_KEYS = (
    "id",
    "delta_cagr_min_abs",
    "delta_sharpe_min",
    "nav_rel_min",
    "v5_nav",
    "nav_abs_min",
    "top1_or_top2_change_gt_0",
    "maxdd_worse_max_abs",
    "fee_corrected_direction_holds",
    "not_only_2024_or_2025",
    "predeclared_21_2_immaterial",
)


def canonical_threshold(gate: dict[str, Any] | None = None) -> dict[str, Any]:
    src = dict(R21_3_GATE if gate is None else gate)
    out: dict[str, Any] = {}
    for key in THRESHOLD_KEYS:
        if key in src:
            out[key] = src[key]
    return out


ECONOMIC_MATERIALITY = {
    **canonical_threshold(),
    "carried_from": "round21_3",
    "frozen_before_formal_nav": True,
    "do_not_retune": True,
    "note": "ΔCAGR≥+1.0pct、ΔSharpe≥+0.05、|ΔNAV|/NAV≥1%、Top1/Top2 变化>0、MaxDD 恶化≤2pct、费用方向不翻、不只靠 2024 或 2025。看候选回测前冻结。",
}

DECISION_RULE = {
    "allowed": [KEEP_BASELINE_V5, RECOMMEND_V6_RESEARCH_CANDIDATE, V6_RESEARCH_CANDIDATE, SUBPERIOD_ALPHA_RESEARCH_CANDIDATE],
    "v6_requires": [
        "strict_pit",
        "comparable_sample",
        "economic_gate",
        "fee_direction_holds",
        "top1_top2_net_not_worse",
        "increment_from_new_factor_not_remove_alone",
        "cross_year",
    ],
    "short_pit_max": SUBPERIOD_ALPHA_RESEARCH_CANDIDATE,
    "no_independent_forward_not_live_ready": True,
    "forbid_live_ready_v6": True,
    "current_baseline_changed": False,
    "auto_promote": False,
    "compare_to": "R16_O01",
    "forbid_compare_to_r222_e0": True,
}

OOF_INCREMENT_GATE = dict(R22_OOF_GATE)
OOF_INCREMENT_GATE["compare_to"] = E0_CODE

FACTORY_STEPS_EXTRA = (
    "TWO_LEDGER_ISOLATION",
    "CANONICAL_THRESHOLD_HASH",
    "FULL_HISTORY_REPLAY",
    "NAN_BEFORE_PIT_KEEP_POOL",
    "MATCHED_SUBPERIOD_NOT_COMPARABLE",
    "C4_DUAL_LEDGER_ATTRIBUTION",
    "HN2_IDENTITY_DECOMPOSITION",
    "FS_REPORT_ID_STATEMENT_ALIGN",
    "ECONOMIC_MATERIALITY_GATE",
    "FORWARD_AVAILABILITY",
    "NO_SILENT_SKIP",
    "NO_ALIAS_ADD",
)

E0_IMPORT_CANDIDATES = (
    ROUND22_ROOT / "candidates" / "R22_E0",
    ROUND22_1_ROOT / "candidates" / "R22_E0",
    RESULT_ROOT / "round21_4" / "candidates" / "R21_4_A0",
    ROUND22_2_ROOT / "parity" / "R222_V5_R16_O01",
)

REPLAY_GROUPS: tuple[dict[str, Any], ...] = (
    {"code": "R223_E0", "role": "E0_CONTROL", "factor_id": None, "r222": "R222_E0", "five_extras": (), "ten_extras": ()},
    {"code": "R223_E1_r22_2_c1_xl_net_imbalance_lag1", "role": "E1_FIVE_ADD", "factor_id": C1_LAG1_ID, "r222": "R222_E1_r22_2_c1_xl_net_imbalance_lag1", "five_extras": (C1_LAG1_ID,), "ten_extras": ()},
    {"code": "R223_E2_r22_2_c1_xl_net_imbalance_lag1", "role": "E2_TEN_ADD", "factor_id": C1_LAG1_ID, "r222": "R222_E2_r22_2_c1_xl_net_imbalance_lag1", "five_extras": (), "ten_extras": (C1_LAG1_ID,)},
    {"code": "R223_E3_r22_2_c1_xl_net_imbalance_lag1", "role": "E3_BOTH_ADD", "factor_id": C1_LAG1_ID, "r222": "R222_E3_r22_2_c1_xl_net_imbalance_lag1", "five_extras": (C1_LAG1_ID,), "ten_extras": (C1_LAG1_ID,)},
    {"code": "R223_E1_r22_c4_flow_persist_5", "role": "E1_FIVE_ADD", "factor_id": C4_ID, "r222": "R222_E1_r22_c4_flow_persist_5", "five_extras": (C4_ID,), "ten_extras": ()},
    {"code": "R223_E2_r22_c4_flow_persist_5", "role": "E2_TEN_ADD", "factor_id": C4_ID, "r222": "R222_E2_r22_c4_flow_persist_5", "five_extras": (), "ten_extras": (C4_ID,)},
    {"code": "R223_E3_r22_c4_flow_persist_5", "role": "E3_BOTH_ADD", "factor_id": C4_ID, "r222": "R222_E3_r22_c4_flow_persist_5", "five_extras": (C4_ID,), "ten_extras": (C4_ID,)},
    {"code": "R223_E1_r22_2_hn2_avg_holding_chg", "role": "E1_FIVE_ADD", "factor_id": HN2_ID, "r222": "R222_E1_r22_2_hn2_avg_holding_chg", "five_extras": (HN2_ID,), "ten_extras": ()},
    {"code": "R223_E2_r22_2_hn2_avg_holding_chg", "role": "E2_TEN_ADD", "factor_id": HN2_ID, "r222": None, "five_extras": (), "ten_extras": (HN2_ID,), "new_this_round": True},
    {"code": "R223_E3_r22_2_hn2_avg_holding_chg", "role": "E3_BOTH_ADD", "factor_id": HN2_ID, "r222": None, "five_extras": (HN2_ID,), "ten_extras": (HN2_ID,), "new_this_round": True},
    {"code": "R223_E1_c1_c4_joint", "role": "E1_FIVE_ADD_JOINT", "factor_id": "c1_c4", "r222": "R222_E1_c1_c4_joint", "five_extras": (C1_LAG1_ID, C4_ID), "ten_extras": ()},
    {"code": "R223_E2_c1_c4_joint", "role": "E2_TEN_ADD_JOINT", "factor_id": "c1_c4", "r222": "R222_E2_c1_c4_joint", "five_extras": (), "ten_extras": (C1_LAG1_ID, C4_ID)},
    {"code": "R223_E3_c1_c4_joint", "role": "E3_BOTH_ADD_JOINT", "factor_id": "c1_c4", "r222": "R222_E3_c1_c4_joint", "five_extras": (C1_LAG1_ID, C4_ID), "ten_extras": (C1_LAG1_ID, C4_ID)},
)

NEW_FACTOR_SPECS = (
    {
        "factor_id": HN2_ID,
        "family": "HN",
        "short": "户均持股变化",
        "formula": HN2_FORMULA_FROZEN,
        "lineage": "lock 22.2; test Δlog(circ)-Δlog(holders)",
        "priority": 1,
    },
    {
        "factor_id": "r22_3_fs1_accrual_minus_ocf",
        "family": "FS",
        "short": "应计减经营现金流",
        "formula": "(NI - OCF) / avg_assets  same report period, published_at/available_at",
        "lineage": "22.2 FS1 join repair",
        "priority": 2,
    },
    {
        "factor_id": "r22_3_fs2_ocf_over_assets",
        "family": "FS",
        "short": "经营现金流/均资产",
        "formula": "OCF / avg_assets  same report period, published_at/available_at",
        "lineage": "FS2 companion; same PIT join as FS1",
        "priority": 2,
    },
)

FACTOR_SPECS_R22 = tuple(dict(s) for s in R22_FACTOR_SPECS)

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FACTOR_ZOO_ROOT = PROJECT_ROOT / "research" / "factor_zoo"
STANDARD_MD = PROJECT_ROOT / "docs" / "ALPHA_FACTOR_RESEARCH_STANDARD.md"
EVALUATE_PY = PROJECT_ROOT / "tools" / "evaluate_factor.py"
FINAL_REPORT = ROUND22_3_ROOT / "ROUND22_3_RESEARCH_REPORT.md"
MANIFEST_PATH = ROUND22_3_ROOT / "round22_3_manifest.yaml"

ROUND22_3_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "configs",
    "candidates",
    "weights",
    "data",
    "data_recovery",
    "ic",
    "evidence",
    "factors",
    "ledger",
    "nav",
    "construct",
    "pit",
    "correlation",
    "backtest",
    "stats",
    "bootstrap",
    "spa",
    "seed",
    "zoo",
    "tests",
    "report",
    "logs",
    "diagnostics",
    "rejected",
    "tracker",
    "coverage",
    "forward",
    "hypotheses",
    "model",
    "matched_subperiod",
    "audit",
    "c4",
    "hn2",
    "fs",
)

MD = {
    "00": ROUND22_3_ROOT / "00_research_contract.md",
    "parity": ROUND22_3_ROOT / "00_v5_parity.md",
    "01": ROUND22_3_ROOT / "01_two_ledger.md",
    "02": ROUND22_3_ROOT / "02_gate_sha.md",
    "03": ROUND22_3_ROOT / "03_matched_folds.md",
    "04": ROUND22_3_ROOT / "04_spa_audit.md",
    "05": ROUND22_3_ROOT / "05_fullhist_train.md",
    "06": ROUND22_3_ROOT / "06_fullhist_sleeve.md",
    "07": ROUND22_3_ROOT / "07_c4_formula_pit.md",
    "08": ROUND22_3_ROOT / "08_c4_dual_ledger.md",
    "09": ROUND22_3_ROOT / "09_hn2_identity.md",
    "10": ROUND22_3_ROOT / "10_hn2_heads.md",
    "11": ROUND22_3_ROOT / "11_fs_join.md",
    "12": ROUND22_3_ROOT / "12_factory.md",
    "13": ROUND22_3_ROOT / "13_e0_e9.md",
    "14": ROUND22_3_ROOT / "14_oof.md",
    "15": ROUND22_3_ROOT / "15_sleeve.md",
    "16": ROUND22_3_ROOT / "16_top1_top2.md",
    "17": ROUND22_3_ROOT / "17_yearly_fee.md",
    "18": ROUND22_3_ROOT / "18_bootstrap_spa.md",
    "19": ROUND22_3_ROOT / "19_zoo.md",
    "20": ROUND22_3_ROOT / "20_tracker.md",
    "21": ROUND22_3_ROOT / "21_standard.md",
    "22": ROUND22_3_ROOT / "22_final_decision.md",
}

STAGES = (
    "parity",
    "freeze",
    "audit",
    "train",
    "backtest",
    "ledger",
    "c4",
    "hn2",
    "fs",
    "stats",
    "factory",
    "zoo",
    "report",
)

Q_ITEMS: tuple[tuple[str, str], ...] = (
    ("Q1", "Stage0 全历史 v5 parity 的 CAGR / Sharpe / MaxDD / trades / NAV / n_obs 分别是多少？是否 PASS？"),
    ("Q2", "经济门 ID、body_sha256、threshold_sha256 是什么？是否在任何候选回测前冻结？是否回调？"),
    ("Q3", "22.1 / 22.2 / 21.4 门槛对象 canonical hash 是 SEMANTICALLY_IDENTICAL_HASH_MISMATCH 还是 ECONOMIC_GATE_SEMANTIC_DRIFT？"),
    ("Q4", "两本账的窗口、n_obs、笔数、对照各是什么？为何不得混表？"),
    ("Q5", "22.2 短窗 E0 Sharpe≈2.735 能否与 R16_O01 Sharpe=1.876 比较？追踪表如何标注？"),
    ("Q6", "Matched 折 Train/Test 重建结果？第一折是否满 3 年？2022 是否 OOF？"),
    ("Q7", "C1 冻结公式是什么？表上有没有 BUY_VALUE_XL？有没有为收益改公式？"),
    ("Q8", "C4 冻结公式是什么？是否仍是 mean(sign(NET_FLOW)_{D-5..D-1})？"),
    ("Q9", "C1/C4 的 PIT 状态、feature_available_from、覆盖？2020-09-24 前是否保持 NaN？"),
    ("Q10", "全历史训练折是否规范 3y/1y 从 2018 起？有没有把 2023+ 预测贴到 2018 NAV？有没有填 0 或删股票？"),
    ("Q11", "全历史 E0 是导入还是真训？Sharpe / CAGR / NAV / n_obs 是否对齐 R16_O01？"),
    ("Q12", "C1 lag1 E1/E2/E3：短窗数字是什么、全历史重跑后是什么、二者为何不可互换？"),
    ("Q13", "C4 E1/E2/E3：短窗数字是什么、全历史重跑后是什么、二者为何不可互换？"),
    ("Q14", "HN2 E1（及本轮 Ten/Both）：短窗数字是什么、全历史重跑后是什么、二者为何不可互换？"),
    ("Q15", "c1_c4_joint E1/E2/E3：短窗数字是什么、全历史重跑后是什么、二者为何不可互换？"),
    ("Q16", "主表 Δ 对照的是 R16_O01 还是 R222_E0？有没有把短窗 Sharpe 2.7 当正式改善？"),
    ("Q17", "C4 对 g1_ret_5 的 |ρ|？有没有因此降低 Replace 门？"),
    ("Q18", "C4 全历史条件 IC / Rolling / Shape / Top1Top2 / NAV 桥的裁决是什么？"),
    ("Q19", "C4 费用与成交敏感性？是否 EXECUTION_FRAGILE？"),
    ("Q20", "HN2 是否恒等于 Δlog(circ)−Δlog(holders)？与 r21_d1 的关系？"),
    ("Q21", "HN2 完整 Evidence（PIT / IC / Rolling / Shape / 冗余）？有没有填 0？"),
    ("Q22", "HN2 H1–H3 全历史真训与合格袖套结果？达冗余门才 Replace 了吗？"),
    ("Q23", "财务报表 PIT Join 如何修？report_id 与 period/statement 如何对齐？"),
    ("Q24", "FS1/FS2 是否构造、口径是否对齐、是否合格才训？状态？"),
    ("Q25", "短窗 SPA Original vs Corrected：n_alternatives、是否误算 E0、p？"),
    ("Q26", "全历史重跑后同窗口 SPA：n_alternatives / p / 状态？"),
    ("Q27", "C4-E1 在全历史对齐日上的配对 bootstrap？"),
    ("Q28", "Top1/Top2 净收益如何算？缺列有没有写成已算不合格？"),
    ("Q29", "分年、2022、Ex2024/Ex2025、费用校正、执行乐观性结果？"),
    ("Q30", "有没有任何全历史候选通过冻结经济门（对照 R16_O01）？"),
    ("Q31", "本轮最佳全历史候选 vs v5 的 ΔSharpe / ΔCAGR / ΔNAV？"),
    ("Q32", "最终决策是 KEEP_BASELINE_V5 / V6_RESEARCH_CANDIDATE / SUBPERIOD_ALPHA_RESEARCH_CANDIDATE？依据？"),
    ("Q33", "CURRENT_BASELINE_CHANGED 与 AUTO_PROMOTE 是什么？有没有改 CURRENT / 升 v6？"),
    ("Q34", "有没有独立 Forward？能否 live-ready？LIVE_READY_V6 是否被禁止？"),
    ("Q35", "追踪表：22.2 短窗行是否标 MATCHED_SUBPERIOD_NOT_COMPARABLE？本轮是否只追加全历史新回测？有无写入时间？"),
    ("Q36", "zoo 追加了哪些 22.3 行？DUPLICATE/失败是否进 rejected archive？"),
    ("Q37", "STANDARD 与 evaluate_factor 本轮更新了什么？"),
    ("Q38", "单元测试条数与失败数？Parity FAIL 是否停止晋级？"),
    ("Q39", "n_formal / n_trained_real / n_imported / n_oof？"),
    ("Q40", "有没有覆盖 result/round22* 或 result/round21* 或 Hermes v7？"),
    ("Q41", "C4 短窗 E1 ΔSharpe=+0.025 vs matched E0：主裁决用了吗？全历史 Δ 才进主裁决了吗？"),
    ("Q42", "HN2 短窗 Five Add ΔSharpe=−0.627 vs matched E0：全历史 H1–H3 是否改变结论？"),
    ("Q43", "C1 短窗全部为负：全历史是否仍为负？"),
    ("Q44", "联合 Add 短窗为负：全历史是否仍为负？有没有失败后无限组合？"),
    ("Q45", "主表是否同时披露 feature_available_from 与覆盖？这是不是全历史严格 PIT？"),
    ("Q46", "折内 |ρ|≥0.60 是否触发 Remove/Replace？Q(Replace)−Q(Remove)？"),
    ("Q47", "ROUND22_3_STATUS 是 COMPLETE、PARTIAL 还是 BLOCKED？未跑完的项是什么？有没有把 NOT_RUN 写成 COMPLETE？"),
    ("Q48", "下一轮最有希望的信息来源与约束？"),
)
assert len(Q_ITEMS) == 48


def spec_by_id(factor_id: str) -> dict[str, Any]:
    for spec in NEW_FACTOR_SPECS:
        if spec["factor_id"] == factor_id:
            return dict(spec)
    for spec in FACTOR_SPECS_R22:
        if spec["factor_id"] == factor_id:
            return dict(spec)
    raise KeyError(factor_id)


def exp_dir(code: str) -> Path:
    return ROUND22_3_ROOT / "candidates" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND22_3_ROOT / "weights" / f"overlayed_{code}.parquet"


def material_vs_v5(obs: dict[str, Any], *, top_changed: bool = False, fee_same_direction: bool = True, not_only_year: bool = True) -> dict[str, Any]:
    gate = ECONOMIC_MATERIALITY
    dc = None if obs.get("cagr") is None else float(obs["cagr"]) - float(V5_ANCHOR["cagr"])
    ds = None if obs.get("sharpe") is None else float(obs["sharpe"]) - float(V5_ANCHOR["sharpe"])
    dn = None if obs.get("nav_end") is None else abs(float(obs["nav_end"]) - float(V5_NAV_END_EXACT)) / float(V5_NAV_END_EXACT)
    dd = None if obs.get("max_drawdown") is None else float(obs["max_drawdown"]) - float(V5_ANCHOR["max_drawdown"])
    flags = {
        "delta_cagr_ok": dc is not None and dc >= float(gate["delta_cagr_min_abs"]),
        "delta_sharpe_ok": ds is not None and ds >= float(gate["delta_sharpe_min"]),
        "nav_rel_ok": dn is not None and dn >= float(gate["nav_rel_min"]),
        "top_changed_ok": bool(top_changed),
        "maxdd_ok": dd is None or dd >= -float(gate["maxdd_worse_max_abs"]),
        "fee_direction_ok": bool(fee_same_direction),
        "not_only_2024_or_2025": bool(not_only_year),
    }
    return {
        "delta_cagr": dc,
        "delta_sharpe": ds,
        "nav_rel": dn,
        "delta_maxdd": dd,
        "flags": flags,
        "economically_material": all(flags.values()),
        "gate_id": gate["id"],
        "compare_to": "R16_O01",
    }
