"""Round 22.2 冻结契约。看结果前写入 manifest，禁止回调。不改交易引擎，不升 v6。"""

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
FULL_EVIDENCE_COMPLETE = "FULL_EVIDENCE_COMPLETE"
FULL_RESEARCH_COMPLETE = "FULL_RESEARCH_COMPLETE"
MONOTONICALLY_EQUIVALENT_FOR_RANKING = "MONOTONICALLY_EQUIVALENT_FOR_RANKING"

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
DATA_LIVE_RELEASES = Path("/data1/Stock/data_live/releases")
MARKET_CUTOFF = "2026-07-24"
DECISION_AUDIT = "decision=D_close; order=D+1_open; same_open_execution_optimism=True"
HOLDER_NUM_PARQUET = Path("/home/mxy/AmazingData/data/amazingdata_raw/holder_num/all.parquet")
BLOCK_TRADING_PARQUET = Path("/home/mxy/AmazingData/data/amazingdata_raw/block_trading/all.parquet")
C_PANEL_22_1 = ROUND22_1_ROOT / "data_recovery" / "c_factor_panel.parquet"

PIT_MIN_FINITE = 0.80
REDUNDANCY_RHO = 0.60
KEYS = ["trade_date", "instrument_id"]
EPS = 1e-12
LABELS = ("label_e2_net", "label_exit_rule_net")
E0_CODE = "R222_E0"
IC_DAILY_MIN = 5
IC_AGG_MIN = 60
IC_THRESHOLDS = (5, 10, 20, 60)
ROLLING_WINDOWS = (60, 120, 252)
BLOCK_DAYS = 20
BOOT_N = 5000
SPA_BOOT = 5000
RIDGE_ALPHA = 1.0
MIN_TRAIN_YEARS = 3
FORMAL_IC_YEARS = tuple(range(2018, 2027))
DIAG_IC_YEARS = tuple(range(2013, 2018))

C1_ID = "r22_c1_xl_net_imbalance"
C4_ID = "r22_c4_flow_persist_5"
C1_FORMULA_FROZEN = "NET_FLOW_XL / TURNOVER_VALUE"
C4_FORMULA_FROZEN = "mean(sign(NET_FLOW)_{D-5..D-1})"
C1_NO_BUY_VALUE_XL = True

EXISTING_11 = (
    "r22_a2_own_touch_limit_excess_20",
    "r22_a3_ret20_mkt_beta_resid",
    "r22_a4_touch5_over_mkt_breadth",
    "r22_b1_dyn_peer_ret_d1",
    "r22_b2_dyn_peer_ret_disp",
    "r22_b3_dyn_peer_to_innov",
    "r22_b4_dyn_peer_ret_over_amp",
    "r22_d1_minute_high_touch_rank",
    "r22_d2_minute_high_lead_min",
    "r22_d3_lead_resid_past_only",
    "r22_d4_lead_x_seal_streak",
)
assert len(EXISTING_11) == 11

GROUPS = (
    "GROUP1_ABSORBED",
    "GROUP2_CONDITIONAL_BUT_ADD_NEGATIVE",
    "GROUP3_WEAK_OR_UNSTABLE",
    "GROUP4_NONMONOTONE_OR_TAIL_ONLY",
    "GROUP5_COMBO_TRANSLATION_FAILED",
)

NEW_FACTOR_SPECS = (
    {
        "factor_id": "r22_2_hn1_neg_log_holder_ratio",
        "family": "HN",
        "short": "户数对数比符号翻转",
        "formula": "-log(HOLDER_NUM_t / HOLDER_NUM_{t-1})  ordered by ANN_DT",
        "lineage": "DUPLICATE of -r21_d1_log_holder_ratio",
        "priority": 2,
    },
    {
        "factor_id": "r22_2_hn2_avg_holding_chg",
        "family": "HN",
        "short": "户均持股变化",
        "formula": "Δ log(then_known_circ_shares / HOLDER_NUM) asof ANN_DT next trading day",
        "lineage": "new vs r21_d1/d3",
        "priority": 1,
    },
    {
        "factor_id": "r22_2_bt1_block_discount",
        "family": "BT",
        "short": "大宗折价",
        "formula": "1 - B_SHARE_PRICE / close on TRADE_DATE",
        "lineage": "DUPLICATE of r21_e1_block_discount; BLOCKED_PIT unless announcement timestamp recovered",
        "priority": 3,
    },
    {
        "factor_id": "r22_2_mg1_fin_buy_intensity",
        "family": "MG",
        "short": "融资买入强度",
        "formula": "financing_buy_value / amount with available_at ≤ cutoff",
        "lineage": "DUPLICATE of r21_f3_fin_buy_intensity",
        "priority": 3,
    },
    {
        "factor_id": "r22_2_fs1_accrual_minus_ocf",
        "family": "FS",
        "short": "应计减经营现金流",
        "formula": "(NI - OCF) / avg_assets  same report period, published_at/available_at",
        "lineage": "relatively new; priority below C and HN",
        "priority": 4,
    },
)

FACTOR_SPECS_R22 = tuple(dict(s) for s in R22_FACTOR_SPECS)
FACTOR_IDS_R22 = tuple(s["factor_id"] for s in FACTOR_SPECS_R22)

ECONOMIC_MATERIALITY = {
    **{k: v for k, v in R21_3_GATE.items() if k != "note"},
    "id": "R21_3_ECONOMIC_MATERIALITY_V1",
    "carried_from": "round21_3",
    "frozen_before_formal_nav": True,
    "do_not_retune": True,
    "round": "22.2",
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
}

OOF_INCREMENT_GATE = dict(R22_OOF_GATE)
OOF_INCREMENT_GATE["compare_to"] = E0_CODE

FACTORY_STEPS_EXTRA = (
    "IC_UNIVERSE_AUDIT",
    "MANDATORY_ROLLING_CONDITIONAL_SHAPE",
    "TOP1_TOP2_LEDGER_TRACE",
    "DATA_RECOVERY_BEFORE_COVERAGE_REJECT",
    "FORMULA_IDENTITY_PROVENANCE",
    "FEE_LEDGER_COMPLETENESS",
    "BOOTSTRAP_SPA_NUMERICAL",
    "SEED_STATUS_ENUM",
    "ECONOMIC_MATERIALITY_GATE",
    "FORWARD_AVAILABILITY",
    "NO_SILENT_SKIP",
    "STRICT_PIT_TIMELINE",
    "MATCHED_SAMPLE_RETRAIN",
    "NO_ALIAS_ADD",
    "DUPLICATE_LINEAGE_CHECK",
)

E0_IMPORT_CANDIDATES = (
    ROUND22_ROOT / "candidates" / "R22_E0",
    ROUND22_1_ROOT / "candidates" / "R22_E0",
    RESULT_ROOT / "round21_4" / "candidates" / "R21_4_A0",
)

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FACTOR_ZOO_ROOT = PROJECT_ROOT / "research" / "factor_zoo"
STANDARD_MD = PROJECT_ROOT / "docs" / "ALPHA_FACTOR_RESEARCH_STANDARD.md"
EVALUATE_PY = PROJECT_ROOT / "tools" / "evaluate_factor.py"
FINAL_REPORT = ROUND22_2_ROOT / "ROUND22_2_RESEARCH_REPORT.md"
MANIFEST_PATH = ROUND22_2_ROOT / "round22_2_manifest.yaml"
ROUND22_READONLY = ROUND22_ROOT
ROUND22_1_READONLY = ROUND22_1_ROOT

ROUND22_2_SHARED_DIRS: tuple[str, ...] = (
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
    "matched",
)

MD = {
    "00": ROUND22_2_ROOT / "00_research_contract.md",
    "parity": ROUND22_2_ROOT / "00_v5_parity.md",
    "01": ROUND22_2_ROOT / "01_c_formula_identity.md",
    "02": ROUND22_2_ROOT / "02_c_pit_timeline.md",
    "03": ROUND22_2_ROOT / "03_matched_sample.md",
    "04": ROUND22_2_ROOT / "04_c14_evidence.md",
    "05": ROUND22_2_ROOT / "05_c14_train.md",
    "06": ROUND22_2_ROOT / "06_existing11_extract.md",
    "07": ROUND22_2_ROOT / "07_group_diagnosis.md",
    "08": ROUND22_2_ROOT / "08_hn.md",
    "09": ROUND22_2_ROOT / "09_bt.md",
    "10": ROUND22_2_ROOT / "10_mg.md",
    "11": ROUND22_2_ROOT / "11_fs.md",
    "12": ROUND22_2_ROOT / "12_factory.md",
    "13": ROUND22_2_ROOT / "13_e0_e9.md",
    "14": ROUND22_2_ROOT / "14_oof.md",
    "15": ROUND22_2_ROOT / "15_sleeve.md",
    "16": ROUND22_2_ROOT / "16_top1_top2.md",
    "17": ROUND22_2_ROOT / "17_yearly_fee.md",
    "18": ROUND22_2_ROOT / "18_bootstrap_spa.md",
    "19": ROUND22_2_ROOT / "19_zoo.md",
    "20": ROUND22_2_ROOT / "20_tracker.md",
    "21": ROUND22_2_ROOT / "21_standard.md",
    "22": ROUND22_2_ROOT / "22_replace_remove.md",
    "23": ROUND22_2_ROOT / "23_final_decision.md",
}

STAGES = (
    "parity",
    "freeze",
    "interpret",
    "pit_c",
    "discover",
    "evidence_c",
    "train",
    "backtest",
    "ledger",
    "stats",
    "factory",
    "zoo",
    "report",
)

Q_ITEMS: tuple[tuple[str, str], ...] = (
    ("Q1", "Stage0 v5 parity 的 CAGR / Sharpe / MaxDD / trades / NAV 分别是多少？是否 PASS？"),
    ("Q2", "经济门 ID 与 body_sha256 是什么？是否在任何候选回测前冻结？是否回调？"),
    ("Q3", "C1 冻结公式是什么？表上有没有 BUY_VALUE_XL？有没有为收益改公式？"),
    ("Q4", "C4 冻结公式是什么？是否与 22.1 recover.py 一致？"),
    ("Q5", "C_FACTOR_PIT_TIMELINE 逐年 UPDATE_TIME / available_at / 回填 / 决策截止的结论？"),
    ("Q6", "C1 严格 PIT 状态是 FULL_HISTORY_STRICT_PIT / SUBPERIOD_STRICT_PIT / LAGGED_SAFE_RESEARCH / RESEARCH_ONLY_BACKFILL / BLOCKED_PIT 哪一个？"),
    ("Q7", "C4 严格 PIT 状态是什么？C4 已滞后 D-5..D-1，available_at 能否在 D_close 前到达？"),
    ("Q8", "若必须 D-1 滞后，新版本号是什么？有没有同名悄悄改期？"),
    ("Q9", "Matched PIT Sample 的日期范围、交易日数、训练年数？不足 3 年如何标记？"),
    ("Q10", "短窗对照是重训 matched v5 还是截断全历史 NAV？有没有和 2018–2026 v5 Sharpe 混表？"),
    ("Q11", "C1 双标签 RankIC / ICIR：2018–2026 正式列与 2013–2017 诊断列各是多少？"),
    ("Q12", "C4 双标签 RankIC / ICIR：2018–2026 与 2013–2017？"),
    ("Q13", "Corr(C1, C4) 是多少？"),
    ("Q14", "C1/C4 对 Five14 / Ten13 的折内 |ρ|？是否达到 0.60 冗余门？"),
    ("Q15", "ConditionalIC(C4 | Legacy, C1) 诊断结果？是否当独立 Alpha？"),
    ("Q16", "E0 matched baseline 的 Sharpe / CAGR / NAV / trades？对照层级？"),
    ("Q17", "E1–E3 Add C1 的 OOF 与正式袖套结果？"),
    ("Q18", "Add C4（E1–E3 或计划中的 E4–E6）的 OOF 与正式袖套结果？"),
    ("Q19", "折内 |ρ|≥0.60 是否触发 Remove/Replace？Q(Replace)−Q(Remove) 是多少？"),
    ("Q20", "C1 与 C4 是否都过增量门？有没有预注册联合 Add？有没有失败后无限组合？"),
    ("Q21", "11 个已有因子的 GROUP1–5 归属分别是什么？"),
    ("Q22", "正式评价列是否只用 2018–2026？2013–2017 是否仅诊断？"),
    ("Q23", "D1 与 D2 是否仍 MONOTONICALLY_EQUIVALENT_FOR_RANKING？能否当两条独立 Alpha？"),
    ("Q24", "哪些因子 Add 为负？最可能解释是吸收、条件信息还是组合翻译失败？"),
    ("Q25", "本轮有没有重开组合搜索 / PCA / 择时 / 仓位 / 退出？"),
    ("Q26", "HN1 与 r21_d1_log_holder_ratio 的关系？是否 DUPLICATE？"),
    ("Q27", "HN2 公式、PIT 规则、多年未更新如何处理？有没有填 0？"),
    ("Q28", "BT 是否恢复到公告时点？未恢复是否 BLOCKED_PIT？无大宗有没有填 0？"),
    ("Q29", "MG1 与 r21_f3_fin_buy_intensity 的关系？覆盖？有没有改 v5 全池？"),
    ("Q30", "FS 是否构造？利润与经营现金流是否同报告期同口径？published_at/available_at？"),
    ("Q31", "新数据只写到哪个目录？有没有覆盖 Hermes v7 或 result/round22* / result/round21*？"),
    ("Q32", "正式袖套代号、OOF 门、同级对照是否齐全（有 E3 必有 E1）？"),
    ("Q33", "Top1/Top2 净收益如何算？有没有因缺列导致 top_net_ok=False？"),
    ("Q34", "分年、2022、Ex2024/Ex2025、费用校正、执行乐观性结果？"),
    ("Q35", "Bootstrap N/block 与 SPA 的 observed / n_alternatives / p / 状态？"),
    ("Q36", "zoo 追加了哪些 22.2 行？DUPLICATE/失败是否进 rejected archive？"),
    ("Q37", "最终决策是 KEEP_BASELINE_V5 / V6_RESEARCH_CANDIDATE / SUBPERIOD_ALPHA_RESEARCH_CANDIDATE？依据？"),
    ("Q38", "CURRENT_BASELINE_CHANGED 与 AUTO_PROMOTE 是什么？有没有改 CURRENT / 升 v6？"),
    ("Q39", "有没有独立 Forward？能否 live-ready？LIVE_READY_V6 是否被禁止？"),
    ("Q40", "ROUND22_2_STATUS 是 COMPLETE 还是 PARTIAL？未跑完的项是什么？有没有把 NOT_RUN 写成 COMPLETE？"),
)
assert len(Q_ITEMS) == 40


def spec_by_id(factor_id: str) -> dict[str, Any]:
    for spec in NEW_FACTOR_SPECS:
        if spec["factor_id"] == factor_id:
            return dict(spec)
    for spec in FACTOR_SPECS_R22:
        if spec["factor_id"] == factor_id:
            return dict(spec)
    raise KeyError(factor_id)


def exp_dir(code: str) -> Path:
    return ROUND22_2_ROOT / "candidates" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND22_2_ROOT / "weights" / f"overlayed_{code}.parquet"
