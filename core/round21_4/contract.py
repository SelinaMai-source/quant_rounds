"""Round 21.4 冻结契约。不改 14/13。不自动升 v6。不覆盖 R21/R21.1/R21.2/R21.3 账本。

冻结经济意义门沿用 21.3（看结果前写 manifest，禁止回调）：
ΔCAGR≥+1.0pct 且 ΔSharpe≥+0.05 且 |ΔNAV|/NAV≥1% 且 Top1/Top2 变化>0
且 MaxDD 恶化≤2pct 且费用校正方向不翻转。+0.003/+43 元预先不达标。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND21_2_ROOT, ROUND21_3_ROOT, ROUND21_4_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round21.contract import (
    AUTO_PROMOTE_CURRENT,
    COLSAMPLE,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    FEE_SCHEDULE,
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
V6_RESEARCH_CANDIDATE = "V6_RESEARCH_CANDIDATE"
AWAITING_USER_REVIEW = "AWAITING_USER_REVIEW"
AUTO_PROMOTE_CURRENT = AUTO_PROMOTE_CURRENT
CURRENT_BASELINE = CURRENT_BASELINE
CURRENT_ALIAS = CURRENT_ALIAS
NOT_TRIGGERED = NOT_TRIGGERED
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
BLOCKED_PIT = "BLOCKED_PIT"
BLOCKED_DATA = "BLOCKED_DATA"
BLOCKED_PERMISSION = "BLOCKED_PERMISSION"
BLOCKED_SEMANTICS = "BLOCKED_SEMANTICS"
BLOCKED = "BLOCKED"
FAILED = "FAILED"
PIT_ELIGIBLE_MODEL = "PIT_ELIGIBLE_MODEL"
STRICT_PIT_VALID = "STRICT_PIT_VALID"
STRICT_PIT = "STRICT_PIT"
RESEARCH_ONLY = "RESEARCH_ONLY"
FORMULA_INVALID = "FORMULA_INVALID"
DUPLICATE = "DUPLICATE"
DUPLICATE_EXPRESSION = "DUPLICATE_EXPRESSION"
NO_OP = "NO_OP"
REPEATED_HISTORICAL_TESTING = "REPEATED_HISTORICAL_TESTING"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
HISTORICAL_FEE_CORRECTED = "HISTORICAL_FEE_CORRECTED"
EXPLORATORY_ONLY = "EXPLORATORY_ONLY"
PARTIAL = "PARTIAL"
COMPLETE = "COMPLETE"
CROSS_SECTIONALLY_DEGENERATE = "CROSS_SECTIONALLY_DEGENERATE"
SAME_OPEN_EXECUTION_OPTIMISM = "SAME_OPEN_EXECUTION_OPTIMISM"
STRICT_PIT_SUBPERIOD_EVIDENCE = "STRICT_PIT_SUBPERIOD_EVIDENCE"
STRICT_PIT_SUBPERIOD_RESEARCH = "STRICT_PIT_SUBPERIOD_RESEARCH"
FEATURE_RECOMPOSITION = "FEATURE_RECOMPOSITION"
ECONOMICALLY_IMMATERIAL = "ECONOMICALLY_IMMATERIAL"
BLOCKED_DATA_UNAVAILABLE = "BLOCKED_DATA_UNAVAILABLE"
UNSEEN_FORWARD_EVALUATION = "UNSEEN_FORWARD_EVALUATION"
PREVIOUSLY_INSPECTED = "PREVIOUSLY_INSPECTED"
INCOMPLETE = "INCOMPLETE"
RAW_BASELINE = "RAW_BASELINE"
IDENTITY_NO_OP = "IDENTITY_NO_OP"
IMPORTED_BY_HASH = "IMPORTED_BY_HASH"
TRAINED_REAL = "TRAINED_REAL"

FIVE_FEATURES = tuple(FIVE_FEATURES)
TEN_FEATURES = tuple(TEN_FEATURES)
SHARED_13 = tuple(TEN_FEATURES)
FIVE_ONLY_FEATURE = "g0_amount_log"
L1_FACTOR = "r21_l1_known_float_chg"
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

# data_live 只读发现；releases root-only，尊重授权，禁止越权读取。
DATA_LIVE_ROOT = Path("/data1/Stock/data_live")
DATA_LIVE_RELEASES = DATA_LIVE_ROOT / "releases"

PIT_MIN_FINITE = 0.80
CLUSTER_SPEARMAN_ABS = 0.85
RIDGE_ALPHA = 1.0
BLOCK_DAYS = 20
BOOT_N = 5000
SPA_BOOT = 2000
IC_YEARS = tuple(range(2018, 2027))
LABELS = ("label_e2_net", "label_exit_rule_net")
KEYS = ["trade_date", "instrument_id"]
EPS = 1e-12

E0_CODE = "R21_4_E0"
A00 = "R21_4_A00"

# ---------------------------------------------------------------------------
# TRACK A — L1 Head-Specific A0–A9（看结果前冻结）
# g0_amount_log 仅存在于 FIVE_FEATURES；对 Ten 的 g0 drop 是 NO-OP。
# lineage_import：特征列表与历史候选完全一致时按 hash 导入（21.3 既有模式，非 alias ADD）。
# ---------------------------------------------------------------------------
G0 = FIVE_ONLY_FEATURE
L1 = L1_FACTOR

A_SPECS: tuple[dict[str, Any], ...] = (
    {
        "code": "R21_4_A0",
        "role": "A0_CONTROL",
        "five_extras": (), "ten_extras": (), "five_drops": (), "ten_drops": (),
        "expect_five": "F0", "expect_ten": "T0",
        "lineage_import": ("round21_3", "R21_3_E0"),
        "formal": True, "head": "both_90_10", "alias_from": None, "force_retrain": True,
    },
    {
        "code": "R21_4_A1",
        "role": "A1_FIVE_DROP_G0",
        "five_extras": (), "ten_extras": (), "five_drops": (G0,), "ten_drops": (),
        "expect_five": "F0-g0", "expect_ten": "T0",
        "lineage_import": ("round21_3", "R21_3_E1_r21_l1_known_float_chg"),
        "formal": True, "head": "both_90_10", "alias_from": None, "force_retrain": True,
    },
    {
        "code": "R21_4_A2",
        "role": "A2_FIVE_ADD_L1",
        "five_extras": (L1,), "ten_extras": (), "five_drops": (), "ten_drops": (),
        "expect_five": "F0+L1", "expect_ten": "T0",
        "lineage_import": None,
        "formal": True, "head": "both_90_10", "alias_from": None, "force_retrain": True,
    },
    {
        "code": "R21_4_A3",
        "role": "A3_REPLICATE_21_2",
        "five_extras": (L1,), "ten_extras": (), "five_drops": (G0,), "ten_drops": (),
        "expect_five": "F0-g0+L1", "expect_ten": "T0",
        "lineage_import": ("round21_2", "R21_2_E3_r21_l1_known_float_chg"),
        "replicate_of": "R21_2_E3_r21_l1_known_float_chg",
        "formal": True, "head": "both_90_10", "alias_from": None, "force_retrain": True,
    },
    {
        "code": "R21_4_A4",
        "role": "A4_TEN_ADD_L1",
        "five_extras": (), "ten_extras": (L1,), "five_drops": (), "ten_drops": (),
        "expect_five": "F0", "expect_ten": "T0+L1",
        "lineage_import": None,
        "formal": True, "head": "both_90_10", "alias_from": None, "force_retrain": True,
    },
    {
        "code": "R21_4_A5",
        "role": "A5_BOTH_REPLACE",
        "five_extras": (L1,), "ten_extras": (L1,), "five_drops": (G0,), "ten_drops": (),
        "expect_five": "F0-g0+L1", "expect_ten": "T0+L1",
        "lineage_import": ("round21_3", "R21_3_E3_r21_l1_known_float_chg"),
        "formal": True, "head": "both_90_10", "alias_from": None, "force_retrain": True,
    },
    {
        "code": "R21_4_A6",
        "role": "A6_REPLICATE_21_3",
        "five_extras": (L1,), "ten_extras": (L1,), "five_drops": (G0,), "ten_drops": (G0,),
        "expect_five": "F0-g0+L1", "expect_ten": "T0-g0+L1==T0+L1",
        "lineage_import": ("round21_3", "R21_3_E3_r21_l1_known_float_chg"),
        "replicate_of": "R21_3_E3_r21_l1_known_float_chg",
        "identity_with": "R21_4_A5",
        "formal": True, "head": "both_90_10", "alias_from": None, "force_retrain": True,
    },
    {
        "code": "R21_4_A7",
        "role": "A7_TEN_DROP_G0_NOOP",
        "five_extras": (), "ten_extras": (), "five_drops": (), "ten_drops": (G0,),
        "expect_five": "F0", "expect_ten": "T0-g0==T0",
        "lineage_import": ("round21_3", "R21_3_E0"),
        "identity_with": "R21_4_A0",
        "formal": True, "head": "both_90_10", "alias_from": None, "force_retrain": True,
    },
    {
        "code": "R21_4_A8",
        "role": "A8_TEN_DROP_G0_ADD_L1",
        "five_extras": (), "ten_extras": (L1,), "five_drops": (), "ten_drops": (G0,),
        "expect_five": "F0", "expect_ten": "T0-g0+L1==T0+L1",
        "lineage_import": None,
        "identity_with": "R21_4_A4",
        "formal": True, "head": "both_90_10", "alias_from": None, "force_retrain": True,
    },
    {
        "code": "R21_4_A9",
        "role": "A9_BOTH_ADD_L1",
        "five_extras": (L1,), "ten_extras": (L1,), "five_drops": (), "ten_drops": (),
        "expect_five": "F0+L1", "expect_ten": "T0+L1",
        "lineage_import": ("round21_3", "R21_3_E2_r21_l1_known_float_chg"),
        "formal": True, "head": "both_90_10", "alias_from": None, "force_retrain": True,
    },
)

# A 矩阵归因对照（看结果前冻结）
A_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("R21_4_A1", "R21_4_A0", "A1-A0_drop_g0_five"),
    ("R21_4_A2", "R21_4_A0", "A2-A0_add_l1_five"),
    ("R21_4_A3", "R21_4_A1", "A3-A1_l1_after_drop"),
    ("R21_4_A3", "R21_4_A0", "A3-A0_43rmb"),
    ("R21_4_A4", "R21_4_A0", "A4-A0_add_l1_ten"),
    ("R21_4_A5", "R21_4_A3", "A5-A3_add_l1_ten_after"),
    ("R21_4_A6", "R21_4_A3", "A6-A3_full_21_3"),
)

A_IDENTITY_ASSERTS: tuple[tuple[str, str], ...] = (
    ("R21_4_A6", "R21_4_A5"),
    ("R21_4_A7", "R21_4_A0"),
    ("R21_4_A8", "R21_4_A4"),
)

# 21.2 复现锚点（来自 21.2 账本 summary.json，只读引用）
REPLICATION_ANCHORS = {
    "R21_4_A3": {
        "source": "R21_2_E3_r21_l1_known_float_chg",
        "sharpe": 1.8787184416808305,
        "cagr": 0.5499818928219966,
        "max_drawdown": -0.22913610780166738,
        "nav_end": 12750132.258793645,
        "completed_trades": 1107,
        "delta_nav_vs_v5": 42.90306895664497,
    },
    "R21_4_A6": {
        "source": "R21_3_E3_r21_l1_known_float_chg",
        "sharpe": 1.3126163950237428,
        "cagr": 0.3255151043857125,
        "max_drawdown": -0.30009060122128184,
        "nav_end": 3343629.2374206143,
        "completed_trades": 1095,
        "delta_nav_vs_v5": -9406460.118304074,
    },
}

# ---------------------------------------------------------------------------
# 冻结经济意义门（沿用 21.3，禁止回调）
# ---------------------------------------------------------------------------
ECONOMIC_MATERIALITY = {
    "id": "R21_3_ECONOMIC_MATERIALITY_V1",
    "carried_from": "round21_3",
    "frozen_before_formal_nav": True,
    "do_not_retune": True,
    "delta_cagr_min_abs": 0.01,
    "delta_sharpe_min": 0.05,
    "nav_rel_min": 0.01,
    "v5_nav": V5_NAV_END_EXACT,
    "nav_abs_min": V5_NAV_END_EXACT * 0.01,
    "top1_or_top2_change_gt_0": True,
    "maxdd_worse_max_abs": 0.02,
    "fee_corrected_direction_holds": True,
    "not_only_2024_or_2025": True,
    "predeclared_21_2_immaterial": {
        "code": "R21_2_E3_r21_l1_known_float_chg",
        "delta_sharpe": 0.003055650344720773,
        "delta_nav_rmb": 42.90,
        "verdict": "IMMATERIAL",
    },
    "note": "统计改善单独报告，不够晋级。21.2 +0.003/+43 预先不达标。",
}

# ---------------------------------------------------------------------------
# TRACK E — W/X/Y/Z 预注册（16 个；等价记 DUPLICATE；至少 4 种机制）
# ---------------------------------------------------------------------------
WXYZ_FACTOR_IDS: tuple[str, ...] = (
    "r21_w1_updown_vol_ratio",
    "r21_w2_impact_instability",
    "r21_w3_volume_innov_residual",
    "r21_w4_response_asymmetry",
    "r21_x1_near_limit_unsealed_freq20",
    "r21_x2_close_to_limit_gap",
    "r21_x3_near_limit_persist",
    "r21_x4_touch_day_vol_share",
    "r21_y1_turnover_innov_minus_peer",
    "r21_y2_turnover_accel_diff",
    "r21_y3_peer_quantile_exposure",
    "r21_y4_impact_diff_vs_peer",
    "r21_z1_vol5_vol20_ratio",
    "r21_z2_downside_semivar_ratio",
    "r21_z3_vol_per_turnover",
    "r21_z4_range_position",
)

WXYZ_MECHANISMS = {
    "W": "首板前价量响应（≠资金流；日级量价弹性/创新/不对称）",
    "X": "近涨停结构（触板未封/缺口/持续/量占比；区分 10%/20%/ST）",
    "Y": "Peer 流动性（PIT-safe 截面 peer 相对量）",
    "Z": "波动压缩（波动比/半方差/波动换手比/区间位置）",
}

# ---------------------------------------------------------------------------
# TRACK D — C0–C11 基线矩阵（看结果前冻结；不预设任何方法优于 Raw）
# ---------------------------------------------------------------------------
COMBO_MATRIX: tuple[dict[str, Any], ...] = (
    {"code": "C0", "family": "raw", "method": "RAW_BASELINE", "params": {}},
    {"code": "C1", "family": "feature_selection", "method": "corr_cluster", "params": {"rho": 0.85}},
    {"code": "C2", "family": "feature_selection", "method": "mrmr", "params": {"k": 10}},
    {"code": "C3", "family": "feature_selection", "method": "stability", "params": {"n_boot": 50, "freq": 0.6}},
    {"code": "C4", "family": "feature_selection", "method": "group_lofo", "params": {"drop": 1}},
    {"code": "C5", "family": "dimension_reduction", "method": "global_pca", "params": {"var": 0.95}},
    {"code": "C6", "family": "dimension_reduction", "method": "family_pca", "params": {"pc": 1}},
    {"code": "C7", "family": "orthogonalization", "method": "ridge_residual", "params": {"alpha": 1.0}},
    {"code": "C8", "family": "supervised", "method": "shrunk_ic", "params": {"shrink": 0.5}},
    {"code": "C9", "family": "supervised", "method": "linear_inner_wf", "params": {"models": ("ridge", "elasticnet", "pls")}},
    {"code": "C10", "family": "nonlinear", "method": "preregistered_interactions", "params": {"cap": 6}},
    {"code": "C11", "family": "hierarchical", "method": "family_composite", "params": {"agg": "mean_z"}},
)

# 预注册交互（非线性家族；禁止无限乘积；沿用 21.3 家族对）
PREREGISTERED_INTERACTIONS: tuple[tuple[str, str], ...] = (
    ("T", "U"),
    ("T", "V"),
    ("U", "V"),
    ("T", "L"),
    ("U", "K"),
    ("V", "L"),
)

OOF_INCREMENT_GATE = {
    "id": "R21_4_OOF_INCREMENT_V1",
    "compare_to": "C0/A0",
    "metric": "oof_rankic_mean",
    "not_worse_both_heads": True,
    "strict_better_any_head": True,
    "min_oos_days": 200,
    "min_delta": 0.0,
    "eps": 1e-12,
}

FACTORY_STEPS_EXTRA = (
    "DATA_RECOVERY_BEFORE_COVERAGE_REJECT",
    "FORMULA_EQUIVALENCE_CHECK",
    "TRUE_DELETE_ASSERTION",
    "FEE_LEDGER_COMPLETENESS",
    "BOOTSTRAP_SPA_NUMERICAL",
    "SEED_STATUS_ENUM",
    "ECONOMIC_MATERIALITY_GATE",
    "FORWARD_AVAILABILITY",
    "NEW_VS_BASELINE_CORRELATION",
    "NEAREST_LEGACY_FACTOR_IDENTIFICATION",
    "FOUR_WAY_REPLACEMENT_TEST",
    "HEAD_SPECIFIC_FOUR_WAY",
    "CLUSTER_REPLACEMENT_IF_APPLICABLE",
    "OOF_INCREMENTALITY",
    "FULL_PORTFOLIO_INCREMENTALITY",
    "REPLACE_HASH_INTEGRITY",
    "COMBINATION_FAILURE_ANALYSIS",
)

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FACTOR_ZOO_ROOT = PROJECT_ROOT / "research" / "factor_zoo"
COMBO_LIB_ROOT = PROJECT_ROOT / "research" / "factor_combination"
STANDARD_MD = PROJECT_ROOT / "docs" / "ALPHA_FACTOR_RESEARCH_STANDARD.md"
FINAL_REPORT = ROUND21_4_ROOT / "ROUND21_4_RESEARCH_REPORT.md"
MANIFEST_NAME = "round21_4_manifest.yaml"

MD = {
    "00": ROUND21_4_ROOT / "00_research_contract.md",
    "parity": ROUND21_4_ROOT / "00_v5_parity.md",
    "01": ROUND21_4_ROOT / "01_round21_3_handoff.md",
    "02": ROUND21_4_ROOT / "02_forward_maxdate_fix.md",
    "03": ROUND21_4_ROOT / "03_forward_coverage_matrix.md",
    "04": ROUND21_4_ROOT / "04_data_recovery.md",
    "05": ROUND21_4_ROOT / "05_l1_amatrix_design.md",
    "06": ROUND21_4_ROOT / "06_l1_amatrix_train.md",
    "07": ROUND21_4_ROOT / "07_l1_amatrix_backtest.md",
    "08": ROUND21_4_ROOT / "08_l1_attribution.md",
    "09": ROUND21_4_ROOT / "09_43rmb_reconciliation.md",
    "10": ROUND21_4_ROOT / "10_combo_method_registry.md",
    "11": ROUND21_4_ROOT / "11_combo_data_protocol.md",
    "12": ROUND21_4_ROOT / "12_combo_feature_selection.md",
    "13": ROUND21_4_ROOT / "13_combo_dimension_reduction.md",
    "14": ROUND21_4_ROOT / "14_combo_orthogonalization.md",
    "15": ROUND21_4_ROOT / "15_combo_supervised.md",
    "16": ROUND21_4_ROOT / "16_combo_nonlinear.md",
    "17": ROUND21_4_ROOT / "17_combo_hierarchical.md",
    "18": ROUND21_4_ROOT / "18_combination_failure_analyzer.md",
    "19": ROUND21_4_ROOT / "19_combo_c0_c11_results.md",
    "20": ROUND21_4_ROOT / "20_wxyz_preregistration.md",
    "21": ROUND21_4_ROOT / "21_wxyz_construction.md",
    "22": ROUND21_4_ROOT / "22_wxyz_pit_coverage.md",
    "23": ROUND21_4_ROOT / "23_wxyz_evidence.md",
    "24": ROUND21_4_ROOT / "24_wxyz_head_fourway.md",
    "25": ROUND21_4_ROOT / "25_wxyz_combination.md",
    "26": ROUND21_4_ROOT / "26_statistical_validation.md",
    "27": ROUND21_4_ROOT / "27_final_decision.md",
}

STAGES = (
    "parity",
    "contract",
    "forward_fix",
    "coverage",
    "recover",
    "amatrix_train",
    "amatrix_backtest",
    "attribution",
    "recon43",
    "combo_build",
    "combo_run",
    "failure",
    "wxyz_construct",
    "wxyz_pit",
    "wxyz_evidence",
    "wxyz_fourway",
    "wxyz_combo",
    "factory",
    "zoo",
    "report",
)

TRACK_A_STAGES = ("amatrix_train", "amatrix_backtest", "attribution", "recon43")
TRACK_B_STAGES = ("forward_fix", "coverage", "recover")
TRACK_C_STAGES = ("factory",)
TRACK_D_STAGES = ("combo_build", "combo_run", "failure")
TRACK_E_STAGES = ("wxyz_construct", "wxyz_pit", "wxyz_evidence", "wxyz_fourway", "wxyz_combo")
POST_STAGES = ("zoo", "report")

Q_ITEMS: tuple[tuple[str, str], ...] = (
    ("Q1", "A0–A9 十个 head-specific 配置的真实全引擎结果分别是什么？"),
    ("Q2", "A3 是否复现 21.2 E3（特征/预测 hash 对齐，NAV +42.90）？"),
    ("Q3", "A6 是否复现 21.3 E3（特征/预测 hash 对齐，ΔNAV −9,406,460）？"),
    ("Q4", "A5≡A6、A0≡A7、A4≡A8 恒等断言是否成立（g0_amount_log 不在 Ten）？"),
    ("Q5", "43 元来源分解：A1−A0、A3−A1、A3−A0 各是多少？"),
    ("Q6", "L1 加进 Five 且保留 g0（A2−A0）的边际效应？"),
    ("Q7", "L1 加进 Ten（A4−A0）的边际效应？−940 万是否由 Ten 头解释？"),
    ("Q8", "哪个 head 对 L1 有毒？证据链是什么？"),
    ("Q9", "43 元逐交易对账的首个分叉日与桥接是什么？"),
    ("Q10", "A 矩阵中是否有任何配置通过冻结经济意义门？"),
    ("Q11", "HISTORICAL_FEE_CORRECTED 下 A 矩阵方向是否保持？"),
    ("Q12", "g0_amount_log 删除的独立效应（A1−A0）是多少？"),
    ("Q13", "forward.py _max_date bug 根因是什么？如何修复？"),
    ("Q14", "修复后 daily_raw / money_flow / share_capital 的真实最大日期？"),
    ("Q15", "FORWARD_DATA_COVERAGE_MATRIX 各项覆盖结论？"),
    ("Q16", "2026-07-25 之后 EOD release 是否存在？权限如何？"),
    ("Q17", "DISCOVER→RECOVER→VALIDATE→INTEGRATE→RETEST 走到哪一步？BLOCKED 原因？"),
    ("Q18", "是否存在独立 Forward OOS？UNSEEN/PREVIOUSLY_INSPECTED/INCOMPLETE 分类？"),
    ("Q19", "money_flow 返回 null 是扫描 bug 还是真缺数据？"),
    ("Q20", "Forward 不可用对最终晋级决策的约束？"),
    ("Q21", "evaluate_factor 升级了哪些 head-specific 强制？"),
    ("Q22", "数据缺失先恢复（DATA_RECOVERY_BEFORE_COVERAGE_REJECT）是否接入？"),
    ("Q23", "zoo 与 STANDARD 本轮更新内容？"),
    ("Q24", "Rejected Archive 本轮新增哪些条目？"),
    ("Q25", "六方法家族各自由哪些方法组成？注册表在哪？"),
    ("Q26", "C0–C11 基线矩阵定义与冻结时间？"),
    ("Q27", "特征选择家族（C1–C4）OOF 结果？"),
    ("Q28", "降维家族（C5–C6）OOF 结果？"),
    ("Q29", "正交化家族（C7）OOF 结果？"),
    ("Q30", "监督组合家族（C8–C9）OOF 结果？"),
    ("Q31", "非线性家族（C10）OOF 结果？"),
    ("Q32", "分层家族（C11）OOF 结果？"),
    ("Q33", "是否有任何组合方法优于 Raw？否则是否 RAW_BASELINE？"),
    ("Q34", "CombinationFailureAnalyzer 对「单因子 IC 有效但组合变差」的归因？"),
    ("Q35", "Nested Chronological WF 如何防止方法选择泄漏？"),
    ("Q36", "组合方法有没有进入全袖套？结果如何？"),
    ("Q37", "W/X/Y/Z 预注册多少因子？几个真实计算成功？几个 BLOCKED？"),
    ("Q38", "几个通过 PIT？几个 DUPLICATE / RESEARCH_ONLY？"),
    ("Q39", "W 族首板前价量响应的证据？"),
    ("Q40", "X 族近涨停结构的证据？与 I 族是否去重？"),
    ("Q41", "Y 族 Peer 流动性的证据？peer 定义是否 PIT-safe？"),
    ("Q42", "Z 族波动压缩的证据？"),
    ("Q43", "W/X/Y/Z head-specific 四路有没有因子通过 OOF 增量门？"),
    ("Q44", "本轮最佳候选 vs v5 的 ΔSharpe / ΔCAGR / ΔNAV？"),
    ("Q45", "是否达到冻结经济意义门？MaxDD 恶化是否 ≤2pct？"),
    ("Q46", "最终决策 KEEP_BASELINE_V5 还是 V6_RESEARCH_CANDIDATE？CURRENT_BASELINE_CHANGED？"),
    ("Q47", "下一轮最有希望的信息来源与方法论方向？"),
)


def exp_dir(code: str) -> Path:
    return ROUND21_4_ROOT / "candidates" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND21_4_ROOT / "weights" / f"overlayed_{code}.parquet"


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


def lineage_root(round_name: str) -> Path:
    if round_name == "round21_2":
        return ROUND21_2_ROOT
    if round_name == "round21_3":
        return ROUND21_3_ROOT
    raise RuntimeError(f"no lineage from {round_name}")


def material_vs_v5(obs: dict[str, Any], *, top_changed: bool = False) -> dict[str, Any]:
    gate = ECONOMIC_MATERIALITY
    dc = None if obs.get("cagr") is None else float(obs["cagr"]) - float(V5_ANCHOR["cagr"])
    ds = None if obs.get("sharpe") is None else float(obs["sharpe"]) - float(V5_ANCHOR["sharpe"])
    dn = None if obs.get("nav_end") is None else abs(float(obs["nav_end"]) - float(V5_NAV_END_EXACT)) / float(V5_NAV_END_EXACT)
    dd = None if obs.get("max_drawdown") is None else float(obs["max_drawdown"]) - float(V5_ANCHOR["max_drawdown"])
    dd_ok = dd is None or dd >= -float(gate["maxdd_worse_max_abs"])
    flags = {
        "delta_cagr_ok": dc is not None and dc >= float(gate["delta_cagr_min_abs"]),
        "delta_sharpe_ok": ds is not None and ds >= float(gate["delta_sharpe_min"]),
        "nav_rel_ok": dn is not None and dn >= float(gate["nav_rel_min"]),
        "top_changed_ok": bool(top_changed),
        "maxdd_ok": bool(dd_ok),
    }
    return {
        "delta_cagr": dc,
        "delta_sharpe": ds,
        "nav_rel": dn,
        "delta_maxdd": dd,
        "flags": flags,
        "economically_material": all(flags.values()),
        "gate_id": gate["id"],
    }
