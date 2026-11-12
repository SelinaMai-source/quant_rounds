"""Round 22.1 冻结契约。看结果前写入 manifest，禁止回调。不改交易引擎，不升 v6。"""

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

KEEP_BASELINE_V5 = "KEEP_BASELINE_V5"
RECOMMEND_V6_RESEARCH_CANDIDATE = "RECOMMEND_V6_RESEARCH_CANDIDATE"
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
PARTIAL = "PARTIAL"
COMPLETE = "COMPLETE"
NOT_RUN = "NOT_RUN"
PIT_PASS = "PIT_PASS"
DUPLICATE_EXPRESSION = "DUPLICATE_EXPRESSION"
CROSS_SECTIONALLY_DEGENERATE = "CROSS_SECTIONALLY_DEGENERATE"
COMPOSITE_OF_KNOWN_FACTORS = "COMPOSITE_OF_KNOWN_FACTORS"
EXPLORATORY_DIAGNOSTIC = "EXPLORATORY_DIAGNOSTIC"
NOT_APPLICABLE = "NOT_APPLICABLE"
NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
IMPORTED_BY_HASH = "IMPORTED_BY_HASH"
REPEATED_HISTORICAL_TESTING = "REPEATED_HISTORICAL_TESTING"
PROFILE_A = "BASELINE_SIGNAL_CUTOFF"
PROFILE_B = "POST_CLOSE_TO_NEXT_PREOPEN"
FULL_EVIDENCE_COMPLETE = "FULL_EVIDENCE_COMPLETE"
FULL_RESEARCH_COMPLETE = "FULL_RESEARCH_COMPLETE"
MONOTONICALLY_EQUIVALENT_FOR_RANKING = "MONOTONICALLY_EQUIVALENT_FOR_RANKING"
PRE_FREEZE_REVISION = "PRE-FREEZE_REVISION"
POST_FREEZE_DEVIATION = "POST-FREEZE_DEVIATION"
HYPOTHESIS_NOT_CONFIRMED = "HYPOTHESIS_NOT_CONFIRMED"
AUDIT_FAILURE = "AUDIT_FAILURE"
RECOVERY_COMPLETE = "RECOVERY_COMPLETE"

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
CLICKHOUSE_HOST = "127.0.0.1"
CLICKHOUSE_PORT = 18123
MARKET_CUTOFF = "2026-07-24"
DECISION_AUDIT = "decision=D_close; order=D+1_open; same_open_execution_optimism=True"

PIT_MIN_FINITE = 0.80
REDUNDANCY_RHO = 0.60
KEYS = ["trade_date", "instrument_id"]
EPS = 1e-12
LABELS = ("label_e2_net", "label_exit_rule_net")
E0_CODE = "R22_E0"
IC_DAILY_MIN = 5
IC_AGG_MIN = 60
IC_THRESHOLDS = (5, 10, 20, 60)
ROLLING_WINDOWS = (60, 120, 252)
BLOCK_DAYS = 20
BOOT_N = 5000
SPA_BOOT = 5000
RIDGE_ALPHA = 1.0

PIT_PASS_IDS = (
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
assert len(PIT_PASS_IDS) == 11

BLOCKED_IDS = (
    "r22_c1_xl_net_imbalance",
    "r22_c4_flow_persist_5",
    "r22_f1_lhb_net_buy",
    "r22_f2_lhb_inst_share",
    "r22_f3_lhb_appear_5d",
    "r22_f4_lhb_same_day_amt_ratio",
    "r22_g1_theme_limit_breadth",
    "r22_g2_theme_count",
    "r22_g3_theme_ret_d1",
    "r22_g4_theme_peer_overlap",
    "r22_h1_days_since_reduction_ann",
    "r22_h2_reduction_shares_to_float",
    "r22_h3_buyback_announced_not_done",
    "r22_h4_supply_event_20d",
)
assert len(BLOCKED_IDS) == 14

FACTOR_SPECS = tuple(dict(s) for s in R22_FACTOR_SPECS)
FACTOR_IDS = tuple(s["factor_id"] for s in FACTOR_SPECS)

# 沿用 R21_3_ECONOMIC_MATERIALITY_V1，禁止回调。
ECONOMIC_MATERIALITY = {
    **{k: v for k, v in R21_3_GATE.items() if k != "note"},
    "id": "R21_3_ECONOMIC_MATERIALITY_V1",
    "carried_from": "round21_3",
    "frozen_before_formal_nav": True,
    "do_not_retune": True,
    "round": "22.1",
    "note": "ΔCAGR≥+1.0pct、ΔSharpe≥+0.05、|ΔNAV|/NAV≥1%、Top1/Top2 变化>0、MaxDD 恶化≤2pct、费用方向不翻、不只靠 2024 或 2025。看候选回测前冻结。",
}

DECISION_RULE = {
    "allowed": [KEEP_BASELINE_V5, RECOMMEND_V6_RESEARCH_CANDIDATE],
    "v6_requires": [
        "economic_gate",
        "fee_direction_holds",
        "top1_top2_net_not_worse",
        "increment_from_new_factor_not_remove_alone",
    ],
    "no_independent_forward_not_live_ready": True,
    "current_baseline_changed": False,
    "auto_promote": False,
}

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
)

FORMAL_SLEEVE_CODES = (
    "R22_E0",
    "R22_E1_r22_a2_own_touch_limit_excess_20",
    "R22_E2_r22_a2_own_touch_limit_excess_20",
    "R22_E3_r22_a2_own_touch_limit_excess_20",
    "R22_E1_r22_b1_dyn_peer_ret_d1",
    "R22_E2_r22_b1_dyn_peer_ret_d1",
    "R22_E3_r22_b1_dyn_peer_ret_d1",
    "R22_E1_r22_d1_minute_high_touch_rank",
    "R22_E2_r22_d1_minute_high_touch_rank",
    "R22_E3_r22_d1_minute_high_touch_rank",
    "R22_E4_r22_d1_minute_high_touch_rank",
    "R22_E5_r22_d1_minute_high_touch_rank",
    "R22_E1_r22_d2_minute_high_lead_min",
    "R22_E2_r22_d2_minute_high_lead_min",
    "R22_E3_r22_d2_minute_high_lead_min",
    "R22_E4_r22_d2_minute_high_lead_min",
    "R22_E5_r22_d2_minute_high_lead_min",
    "R22_E1_r22_d3_lead_resid_past_only",
    "R22_E2_r22_d3_lead_resid_past_only",
    "R22_E3_r22_d3_lead_resid_past_only",
    "R22_E4_r22_d3_lead_resid_past_only",
    "R22_E5_r22_d3_lead_resid_past_only",
    "R22_E1_r22_d4_lead_x_seal_streak",
    "R22_E2_r22_d4_lead_x_seal_streak",
    "R22_E3_r22_d4_lead_x_seal_streak",
)
assert len(FORMAL_SLEEVE_CODES) == 25

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FACTOR_ZOO_ROOT = PROJECT_ROOT / "research" / "factor_zoo"
STANDARD_MD = PROJECT_ROOT / "docs" / "ALPHA_FACTOR_RESEARCH_STANDARD.md"
EVALUATE_PY = PROJECT_ROOT / "tools" / "evaluate_factor.py"
FINAL_REPORT = ROUND22_1_ROOT / "ROUND22_1_RESEARCH_REPORT.md"
MANIFEST_PATH = ROUND22_1_ROOT / "round22_1_manifest.yaml"
ROUND22_READONLY = ROUND22_ROOT

MD = {
    "00": ROUND22_1_ROOT / "00_research_contract.md",
    "parity": ROUND22_1_ROOT / "00_v5_parity.md",
    "01": ROUND22_1_ROOT / "01_ic_methodology_audit.md",
    "02": ROUND22_1_ROOT / "02_evidence_rolling_conditional.md",
    "03": ROUND22_1_ROOT / "03_top1_top2_ledger.md",
    "04": ROUND22_1_ROOT / "04_c_semantics.md",
    "05": ROUND22_1_ROOT / "05_fgh_recovery.md",
    "06": ROUND22_1_ROOT / "06_d_family.md",
    "07": ROUND22_1_ROOT / "07_formula_provenance.md",
    "08": ROUND22_1_ROOT / "08_nav_bridge.md",
    "09": ROUND22_1_ROOT / "09_factory_tests.md",
    "10": ROUND22_1_ROOT / "10_yearly_fee.md",
    "11": ROUND22_1_ROOT / "11_bootstrap_spa.md",
    "12": ROUND22_1_ROOT / "12_zoo.md",
    "13": ROUND22_1_ROOT / "13_tracker.md",
    "14": ROUND22_1_ROOT / "14_pit_pass_matrix.md",
    "15": ROUND22_1_ROOT / "15_blocked_matrix.md",
    "16": ROUND22_1_ROOT / "16_label_maturity.md",
    "17": ROUND22_1_ROOT / "17_replace_remove.md",
    "18": ROUND22_1_ROOT / "18_standard.md",
    "19": ROUND22_1_ROOT / "19_final_decision.md",
}

STAGES = (
    "parity",
    "freeze",
    "ic_audit",
    "evidence",
    "ledger",
    "recover",
    "dfamily",
    "factory",
    "zoo",
    "report",
)

Q_ITEMS: tuple[tuple[str, str], ...] = (
    ("Q1", "Stage0 v5 parity 的 CAGR / Sharpe / MaxDD / trades / NAV 分别是多少？是否 PASS？"),
    ("Q2", "经济门 ID 与 body_sha256 是什么？是否在任何候选回测前冻结？是否回调？"),
    ("Q3", "主 IC 宇宙是全市场还是一进二 first_board candidate_panel？对照诊断是否混进主列？"),
    ("Q4", "日级 IC 跳过规则是每日 n<5 还是每日 n<60？诊断矩阵原先怎么写错的？"),
    ("Q5", "聚合 NOT_IDENTIFIABLE 门槛是累计有效 IC 日 <60 还是每日 n<60？"),
    ("Q6", "ICIR 公式是什么？是否年化？ddof 是多少？"),
    ("Q7", "by_year 是否裁到 2018+？2013–2017 是否进入 IC？"),
    ("Q8", "预注册门槛 {5,10,20,60} 的敏感性结论是什么？有没有事后选最佳？"),
    ("Q9", "label_e2_net / label_exit_rule_net 在训练 cutoff 前是否已 mature？"),
    ("Q10", "11 个 PIT_PASS 因子的 Rolling 60/120/252 是否都写出 rolling.json？"),
    ("Q11", "Conditional IC 是折内 Ridge 还是全历史一次拟合？Five 控几列、Ten 控几列？"),
    ("Q12", "Shape / Top-tail 是因子值上分位还是排名前几？不足是否强行五分位？"),
    ("Q13", "evaluate.py 是否去掉 Rolling/Shape/TopTail/ConditionalIC 的 SKIP_IF_MISSING？"),
    ("Q14", "缺任一 JSON 能否标 FULL_EVIDENCE_COMPLETE？本轮实际状态是什么？"),
    ("Q15", "Top1/Top2 如何从已有 five_pred/ten_pred 重建？是否重训 51 模型？"),
    ("Q16", "新增账本字段有没有改变成交价、仓位、退出、费率或 NAV？"),
    ("Q17", "E0 Same Orders / Same NAV / Same Sharpe 对照结果？"),
    ("Q18", "此前 top_net_ok=False 的原因是缺列还是已算但不合格？修复后呢？"),
    ("Q19", "NAV 桥：期初 + 已实现 + 分红 − 费用 ± 未实现 = 期末，对得上吗？"),
    ("Q20", "mkt_equ_mf_new 全表分档加总 vs TURNOVER_VOL / TURNOVER_VALUE 的结论？"),
    ("Q21", "C1/C4 解封了还是仍 BLOCKED_SEMANTICS？有没有换高收益公式？"),
    ("Q22", "F 族 ClickHouse 凭证、download_attempts、最终状态？"),
    ("Q23", "G 族是否有 INTO_DATE/OUT_DATE/known_at？有没有用申万或 2026 成分回填？"),
    ("Q24", "H 族是否有公告日？share_capital 的晚到 PUBLISH_DATE 有没有倒推？"),
    ("Q25", "新数据只写到哪个目录？有没有覆盖 Hermes v7 或 result/round22？"),
    ("Q26", "D1 与 D2 的 RankIC 是否相同？日 Spearman / 排序一致率？"),
    ("Q27", "D1 与 D2 是否 MONOTONICALLY_EQUIVALENT_FOR_RANKING？这是否等于两条独立 Alpha？"),
    ("Q28", "Q(E4)-Q(E0)、Q(E5)-Q(E4)、Q(E5)-Q(E0)、Q(E1)-Q(E0) 的 Sharpe/CAGR/NAV 是多少？"),
    ("Q29", "D2 Replace−Remove ≈ +0.140 能否单独当新 Alpha？"),
    ("Q30", "D4 标签是什么？能否当新原始信息升级？"),
    ("Q31", "A2/A4/B1/C1/D4 的 Original Prompt vs Frozen vs 实际代码 provenance？"),
    ("Q32", "Bootstrap N/block 与 SPA 的 observed / n_alternatives / p / 状态？"),
    ("Q33", "最终决策是 KEEP_BASELINE_V5 还是 RECOMMEND_V6_RESEARCH_CANDIDATE？依据？"),
    ("Q34", "CURRENT_BASELINE_CHANGED 与 AUTO_PROMOTE 是什么？有没有改 CURRENT？"),
    ("Q35", "有没有独立 Forward？能否 live-ready？下一轮优先解封什么？"),
)
assert len(Q_ITEMS) == 35


def spec_by_id(factor_id: str) -> dict[str, Any]:
    for spec in FACTOR_SPECS:
        if spec["factor_id"] == factor_id:
            return dict(spec)
    raise KeyError(factor_id)


def exp_dir(code: str) -> Path:
    return ROUND22_1_ROOT / "candidates" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND22_1_ROOT / "weights" / f"overlayed_{code}.parquet"


def r22_exp_dir(code: str) -> Path:
    return ROUND22_ROOT / "candidates" / str(code)
