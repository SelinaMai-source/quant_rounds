"""Round 21.2 冻结契约。不改 14/13。不自动升 v6。不覆盖 R21/R21.1 账本。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND21_1_ROOT, ROUND21_2_ROOT, ROUND21_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round21.contract import (
    AUTO_PROMOTE_CURRENT,
    COLSAMPLE,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    FEE_SCHEDULE,
    GITHUB_CLONE_DEST,
    GITHUB_REPO,
    GITHUB_SPARSE_DIR,
    HERMES_REBUILD_RAW,
    HERMES_V7,
    KEEP_BASELINE_V5,
    NAV_END_ABS_TOL,
    NOT_TRIGGERED,
    O01_FORMULA,
    PARITY_METRIC_ABS,
    SUBSAMPLE,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    XGB_N_JOBS,
)

KEEP_BASELINE_V5 = KEEP_BASELINE_V5
RECOMMEND_ALPHA_CANDIDATE = "RECOMMEND_ALPHA_CANDIDATE"
RECOMMEND_NEW_ALPHA_CANDIDATE = RECOMMEND_ALPHA_CANDIDATE
AWAITING_USER_REVIEW = "AWAITING_USER_REVIEW"
AUTO_PROMOTE_CURRENT = AUTO_PROMOTE_CURRENT
CURRENT_BASELINE = CURRENT_BASELINE
CURRENT_ALIAS = CURRENT_ALIAS
NOT_TRIGGERED = NOT_TRIGGERED
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
BLOCKED_PIT = "BLOCKED_PIT"
BLOCKED_DATA = "BLOCKED_DATA"
BLOCKED_SEMANTICS = "BLOCKED_SEMANTICS"
PIT_ELIGIBLE_MODEL = "PIT_ELIGIBLE_MODEL"
STRICT_PIT_VALID = "STRICT_PIT_VALID"
RESEARCH_ONLY = "RESEARCH_ONLY"
FORMULA_INVALID = "FORMULA_INVALID"
NEAREST_LEGACY_NOT_IDENTIFIABLE = "NEAREST_LEGACY_NOT_IDENTIFIABLE"
IDENTICAL_BY_DESIGN = "IDENTICAL_BY_DESIGN"
REPLACE_NOT_APPLIED = "REPLACE_NOT_APPLIED"
DUPLICATE_EXPRESSION = "DUPLICATE_EXPRESSION"
NO_OP = "NO_OP"
REPEATED_HISTORICAL_TESTING = "REPEATED_HISTORICAL_TESTING"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
HISTORICAL_FEE_CORRECTED = "HISTORICAL_FEE_CORRECTED"
EXPLORATORY_ONLY = "EXPLORATORY_ONLY"
NEVER_OBSERVED = "NEVER_OBSERVED"
PARTIAL = "PARTIAL"
COMPLETE = "COMPLETE"

FIVE_FEATURES = tuple(FIVE_FEATURES)
TEN_FEATURES = tuple(TEN_FEATURES)
SHARED_13 = tuple(TEN_FEATURES)
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
GITHUB_REPO = GITHUB_REPO
GITHUB_SPARSE_DIR = GITHUB_SPARSE_DIR
GITHUB_CLONE_DEST = GITHUB_CLONE_DEST
GITHUB_PINNED_SHA = "7316e40b9967b724e5e6e85c967e21f42acb7439"

PIT_MIN_FINITE = 0.80
CLUSTER_SPEARMAN_ABS = 0.85
RIDGE_ALPHA = 1.0
BLOCK_DAYS = 20
BOOT_N = 5000
SPA_BOOT = 2000
IC_YEARS = tuple(range(2018, 2027))
PREDICT_YEARS = tuple(range(2018, 2027))
LABELS = ("label_e2_net", "label_exit_rule_net")
KEYS = ["trade_date", "instrument_id"]
EPS = 1e-12
INTERACTION_CAP = 120

E0_CODE = "R21_2_E0"
A00 = "R21_2_A00"

COVERAGE_CLASSES = (
    "STRUCTURAL_NOT_APPLICABLE",
    "EVENT_CONDITIONAL",
    "TRUE_DATA_MISSING",
    "SOURCE_ALIGNMENT_FAILURE",
    "FORMULA_UNDEFINED",
    "TIMESTAMP_UNVERIFIED",
)

NEW_FACTOR_IDS = (
    "r21_g1_avg_ticket_value",
    "r21_g2_large_ticket_asym",
    "r21_g3_order_count_imbalance",
    "r21_g4_large_ticket_conc_gap",
    "r21_h1_late_flow_dominance",
    "r21_h2_large_small_flow_div",
    "r21_h3_flow_dir_disagree",
    "r21_h4_large_flow_persist",
    "r21_i1_failed_board_ratio_20",
    "r21_i2_days_since_failed",
    "r21_i3_failed_recurrence_w20",
    "r21_i4_hit_to_close_conversion",
    "r21_j1_preboard_turnover_5",
    "r21_j2_preboard_px_vwap20",
    "r21_j3_preboard_high_prox",
    "r21_j4_prior_ret_per_turnover",
    "r21_k1_pre_hit_vol_frac",
    "r21_k2_vol_time_dispersion",
    "r21_k3_minute_vol_entropy",
    "r21_k4_signed_path_vol_pressure",
    "r21_l1_known_float_chg",
    "r21_l2_float_shock_vs_liq",
    "r21_l3_days_since_float_up_ann",
    "r21_l4_turnover_x_float_chg",
    "r21_m1_rev_yoy_change",
    "r21_m2_ocf_over_abs_np",
    "r21_m3_earnings_info_age",
    "r21_m4_growth_quality_div",
    "r21_n1_lagged_peer_limit_up",
    "r21_n2_peer_return_div",
    "r21_n3_peer_failed_spillover",
    "r21_n4_peer_adj_fb_intensity",
)

PREREGISTERED_INTERACTIONS = (
    ("G", "H"),
    ("G", "K"),
    ("H", "I"),
    ("H", "J"),
    ("I", "K"),
    ("J", "L"),
    ("J", "M"),
    ("K", "N"),
    ("L", "M"),
    ("G", "N"),
)

FACTORY_STEPS_EXTRA = (
    "NEW_VS_BASELINE_CORRELATION",
    "NEAREST_LEGACY_FACTOR_IDENTIFICATION",
    "FOUR_WAY_REPLACEMENT_TEST",
    "CLUSTER_REPLACEMENT_IF_APPLICABLE",
    "OOF_INCREMENTALITY",
    "FULL_PORTFOLIO_INCREMENTALITY",
    "FORMULA_EQUIVALENCE_CHECK",
    "REPLACE_HASH_INTEGRITY",
)

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FACTOR_ZOO_ROOT = PROJECT_ROOT / "research" / "factor_zoo"
STANDARD_MD = PROJECT_ROOT / "docs" / "ALPHA_FACTOR_RESEARCH_STANDARD.md"
FINAL_REPORT = ROUND21_2_ROOT / "ROUND21_2_ALPHA_MINING_REPORT.md"
MANIFEST_NAME = "round21_2_manifest.yaml"

MD = {
    "00": ROUND21_2_ROOT / "00_research_contract.md",
    "01": ROUND21_2_ROOT / "01_round21_1_audit.md",
    "02": ROUND21_2_ROOT / "02_new_raw_data_audit.md",
    "03": ROUND21_2_ROOT / "03_new_factor_hypotheses.md",
    "04": ROUND21_2_ROOT / "04_factor_construction.md",
    "05": ROUND21_2_ROOT / "05_PIT_and_coverage.md",
    "06": ROUND21_2_ROOT / "06_individual_factor_evidence.md",
    "07": ROUND21_2_ROOT / "07_github_rolling_evidence.md",
    "08": ROUND21_2_ROOT / "08_nonlinearity_top_tail.md",
    "09": ROUND21_2_ROOT / "09_correlation_and_redundancy.md",
    "10": ROUND21_2_ROOT / "10_conditional_IC.md",
    "11": ROUND21_2_ROOT / "11_nearest_factor_replacement.md",
    "12": ROUND21_2_ROOT / "12_factor_interactions.md",
    "13": ROUND21_2_ROOT / "13_nested_walk_forward.md",
    "14": ROUND21_2_ROOT / "14_model_incrementality.md",
    "15": ROUND21_2_ROOT / "15_full_backtest.md",
    "16": ROUND21_2_ROOT / "16_top1_top2_attribution.md",
    "17": ROUND21_2_ROOT / "17_yearly_robustness.md",
    "18": ROUND21_2_ROOT / "18_cost_sensitivity.md",
    "19": ROUND21_2_ROOT / "19_statistical_validation.md",
    "20": ROUND21_2_ROOT / "20_seed_robustness.md",
    "21": ROUND21_2_ROOT / "21_final_decision.md",
    "parity": ROUND21_2_ROOT / "00_v5_parity.md",
}

STAGES = (
    "parity",
    "audit",
    "lineage",
    "data_audit",
    "hypotheses",
    "construct",
    "pit",
    "evidence",
    "github",
    "correlate",
    "train",
    "interact",
    "backtest",
    "attribution",
    "stats",
    "seed",
    "zoo",
    "report",
)

INCREMENT_GATE = {
    "id": "R21_2_OOF_INCREMENT_V1",
    "compare_to": E0_CODE,
    "metric": "oof_rankic_mean",
    "not_worse_both_heads": True,
    "strict_better_any_head": True,
    "min_oos_days": 200,
    "min_delta": 0.0,
    "eps": 1e-12,
    "seed_only_if_gate_hit": True,
}

Q_ITEMS = (
    ("Q1", "Round21.1 的 Replace 实现是否真实正确？"),
    ("Q2", "旧相关性报告中数值不一致的问题是什么？"),
    ("Q3", "本轮实际挖掘了哪些新的信息源？"),
    ("Q4", "32 个新公式有多少真正成功计算？"),
    ("Q5", "有多少因子通过严格 PIT？"),
    ("Q6", "哪些因子因为供应商历史回填无法使用？"),
    ("Q7", "G 家族订单笔数是否提供独立信息？"),
    ("Q8", "H 家族资金流分歧是否优于旧大单因子？"),
    ("Q9", "历史炸板失败记忆是否具有预测力？"),
    ("Q10", "首板前筹码换手是否具有额外 Alpha？"),
    ("Q11", "日内成交量时间分布是否有效？"),
    ("Q12", "流通股本供给变化是否有效？"),
    ("Q13", "财务公告信息是否影响一进二质量？"),
    ("Q14", "Peer 流动性替代是否包含独立信号？"),
    ("Q15", "哪些因子的 Rolling IC 最稳定？"),
    ("Q16", "哪些因子是非单调或 Top-tail Only？"),
    ("Q17", "哪些因子与旧 14/13 高度相关？"),
    ("Q18", "哪些旧因子可以被新因子更好替换？"),
    ("Q19", "哪些交互确实改善 OOF Ranking？"),
    ("Q20", "哪些新因子改善 Top1 或 Top2？"),
    ("Q21", "哪些候选通过真实组合回测？"),
    ("Q22", "最佳候选比 v5 提高多少 Sharpe？"),
    ("Q23", "CAGR 增加还是降低？"),
    ("Q24", "MaxDD 有没有恶化？"),
    ("Q25", "2022 年表现是否稳定？"),
    ("Q26", "历史费用校正后增量是否仍存在？"),
    ("Q27", "结果有无统计支持？"),
    ("Q28", "是否具有独立 Forward OOS？"),
    ("Q29", "是否值得推荐新的 Alpha Candidate？"),
    ("Q30", "本轮应把哪些因子永久加入研究库？"),
    ("Q31", "下一轮最有希望的信息来源是什么？"),
)


def exp_dir(code: str) -> Path:
    return ROUND21_2_ROOT / "candidates" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND21_2_ROOT / "weights" / f"overlayed_{code}.parquet"


def model_parameters(seed: int, *, n_jobs: int = XGB_N_JOBS) -> dict[str, Any]:
    return {
        "objective": "rank:ndcg",
        "eval_metric": None,
        "ndcg_exp_gain": True,
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 1.0,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "tree_method": "hist",
        "random_state": int(seed),
        "n_jobs": int(n_jobs),
    }


def family_of(fid: str) -> str:
    parts = str(fid).split("_")
    if len(parts) >= 2 and len(parts[1]) >= 1:
        return parts[1][0].upper()
    return "?"
