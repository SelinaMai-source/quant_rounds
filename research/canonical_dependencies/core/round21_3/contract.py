"""Round 21.3 冻结契约。不改 14/13。不自动升 v6。不覆盖 R21/R21.1/R21.2 账本。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND21_1_ROOT, ROUND21_2_ROOT, ROUND21_3_ROOT, ROUND21_ROOT
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
RECOMMEND_BASELINE_V6_CANDIDATE = "RECOMMEND_BASELINE_V6_CANDIDATE"
V6_RESEARCH_CANDIDATE = "V6_RESEARCH_CANDIDATE"
RECOMMEND_ALPHA_CANDIDATE = "RECOMMEND_ALPHA_CANDIDATE"
AWAITING_USER_REVIEW = "AWAITING_USER_REVIEW"
AUTO_PROMOTE_CURRENT = AUTO_PROMOTE_CURRENT
CURRENT_BASELINE = CURRENT_BASELINE
CURRENT_ALIAS = CURRENT_ALIAS
NOT_TRIGGERED = NOT_TRIGGERED
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
BLOCKED_PIT = "BLOCKED_PIT"
BLOCKED_DATA = "BLOCKED_DATA"
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
AUDIT_FAILURE = "AUDIT_FAILURE"
SAME_OPEN_EXECUTION_OPTIMISM = "SAME_OPEN_EXECUTION_OPTIMISM"
STRICT_PIT_SUBPERIOD_EVIDENCE = "STRICT_PIT_SUBPERIOD_EVIDENCE"
FEATURE_RECOMPOSITION = "FEATURE_RECOMPOSITION"

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
INTERACTION_CAP = 6

E0_CODE = "R21_3_E0"
A00 = "R21_3_A00"
L1_E0 = "R21_3_E0"
L1_E1 = "R21_3_E1_r21_l1_known_float_chg"
L1_E2 = "R21_3_E2_r21_l1_known_float_chg"
L1_E3 = "R21_3_E3_r21_l1_known_float_chg"

# 冻结：看正式回测前写入 manifest，禁止按结果回调。
ECONOMIC_MATERIALITY = {
    "id": "R21_3_ECONOMIC_MATERIALITY_V1",
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

# T/U/V 12–18，看收益前冻结。代数≡G/H/K/L 记 DUPLICATE。
NEW_FACTOR_IDS = (
    "r21_t1_ticket_vs_med20",
    "r21_t2_bucket_js_div",
    "r21_t3_large_ticket_asym_chg",
    "r21_t4_dlog_count_minus_dlog_amt",
    "r21_t5_deal_count_vs_med20",
    "r21_u1_pre_post_hit_impact",
    "r21_u2_limit_vs_pullback_vol",
    "r21_u3_state_persist_switch",
    "r21_u4_pre_post_vol_conc",
    "r21_u5_open_recover_eff",
    "r21_v1_ann_to_effective_days",
    "r21_v2_classified_float_shock",
    "r21_v3_float_shock_vs_liq",
    "r21_v4_future_event_overlap",
)

PREREGISTERED_INTERACTIONS = (
    ("T", "U"),
    ("T", "V"),
    ("U", "V"),
    ("T", "L"),
    ("U", "K"),
    ("V", "L"),
)

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
    "CLUSTER_REPLACEMENT_IF_APPLICABLE",
    "OOF_INCREMENTALITY",
    "FULL_PORTFOLIO_INCREMENTALITY",
    "REPLACE_HASH_INTEGRITY",
)

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FACTOR_ZOO_ROOT = PROJECT_ROOT / "research" / "factor_zoo"
STANDARD_MD = PROJECT_ROOT / "docs" / "ALPHA_FACTOR_RESEARCH_STANDARD.md"
FINAL_REPORT = ROUND21_3_ROOT / "ROUND21_3_RESEARCH_REPORT.md"
MANIFEST_NAME = "round21_3_manifest.yaml"

MD = {
    "00": ROUND21_3_ROOT / "00_research_contract.md",
    "01": ROUND21_3_ROOT / "01_round21_2_forensics.md",
    "02": ROUND21_3_ROOT / "02_data_recovery.md",
    "03": ROUND21_3_ROOT / "03_new_factor_hypotheses.md",
    "04": ROUND21_3_ROOT / "04_factor_construction.md",
    "05": ROUND21_3_ROOT / "05_PIT_and_coverage.md",
    "06": ROUND21_3_ROOT / "06_individual_factor_evidence.md",
    "07": ROUND21_3_ROOT / "07_github_rolling_evidence.md",
    "08": ROUND21_3_ROOT / "08_nonlinearity_top_tail.md",
    "09": ROUND21_3_ROOT / "09_correlation_and_redundancy.md",
    "10": ROUND21_3_ROOT / "10_conditional_IC.md",
    "11": ROUND21_3_ROOT / "11_nearest_factor_replacement.md",
    "12": ROUND21_3_ROOT / "12_factor_interactions.md",
    "13": ROUND21_3_ROOT / "13_nested_walk_forward.md",
    "14": ROUND21_3_ROOT / "14_model_incrementality.md",
    "15": ROUND21_3_ROOT / "15_full_backtest.md",
    "16": ROUND21_3_ROOT / "16_top1_top2_attribution.md",
    "17": ROUND21_3_ROOT / "17_yearly_robustness.md",
    "18": ROUND21_3_ROOT / "18_cost_sensitivity.md",
    "19": ROUND21_3_ROOT / "19_statistical_validation.md",
    "20": ROUND21_3_ROOT / "20_seed_robustness.md",
    "21": ROUND21_3_ROOT / "21_n1n3_cross_section.md",
    "22": ROUND21_3_ROOT / "22_l1_nav_reconciliation.md",
    "23": ROUND21_3_ROOT / "23_forward_availability.md",
    "24": ROUND21_3_ROOT / "24_final_decision.md",
    "parity": ROUND21_3_ROOT / "00_v5_parity.md",
}

STAGES = (
    "parity",
    "contract",
    "l1_train",
    "l1_backtest",
    "nav_recon",
    "stats",
    "seed",
    "n1n3",
    "recover",
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
    "forward",
    "zoo",
    "report",
)

TRACK_A_STAGES = ("l1_train", "l1_backtest", "nav_recon", "stats", "seed", "n1n3")
TRACK_B_STAGES = (
    "recover",
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
    "forward",
)
POST_STAGES = ("zoo", "report")

INCREMENT_GATE = {
    "id": "R21_3_OOF_INCREMENT_V1",
    "compare_to": E0_CODE,
    "metric": "oof_rankic_mean",
    "not_worse_both_heads": True,
    "strict_better_any_head": True,
    "min_oos_days": 200,
    "min_delta": 0.0,
    "eps": 1e-12,
    "seed_only_if_gate_hit": True,
}

L1_SPECS = (
    {
        "code": L1_E0,
        "role": "E0_CONTROL",
        "factor_id": None,
        "five_extras": (),
        "ten_extras": (),
        "five_drops": (),
        "ten_drops": (),
        "formal": True,
        "head": "both_90_10",
        "alias_from": None,
        "force_retrain": True,
    },
    {
        "code": L1_E1,
        "role": "E1_REMOVE",
        "factor_id": L1_FACTOR,
        "five_extras": (),
        "ten_extras": (),
        "five_drops": (FIVE_ONLY_FEATURE,),
        "ten_drops": (),
        "formal": True,
        "head": "both_90_10",
        "alias_from": None,
        "force_retrain": True,
        "note": "g0_amount_log Five-only drop; Ten stays 13",
    },
    {
        "code": L1_E2,
        "role": "E2_ADD",
        "factor_id": L1_FACTOR,
        "five_extras": (L1_FACTOR,),
        "ten_extras": (L1_FACTOR,),
        "five_drops": (),
        "ten_drops": (),
        "formal": True,
        "head": "both_90_10",
        "alias_from": None,
        "force_retrain": True,
    },
    {
        "code": L1_E3,
        "role": "E3_REPLACE",
        "factor_id": L1_FACTOR,
        "five_extras": (L1_FACTOR,),
        "ten_extras": (L1_FACTOR,),
        "five_drops": (FIVE_ONLY_FEATURE,),
        "ten_drops": (),
        "formal": True,
        "head": "both_90_10",
        "alias_from": None,
        "force_retrain": True,
        "assert": "E3 = E0 - {g0_amount_log} + {L1}",
    },
)

Q_ITEMS = (
    ("Q1", "21.2 最佳 L1 E3 的 +43 元 / ΔSharpe+0.003 来自 L1 还是删成交额对数？"),
    ("Q2", "E1（只删 g0_amount_log）全引擎结果是什么？"),
    ("Q3", "E3 = E0 − {g0_amount_log} + {L1} 断言是否成立？"),
    ("Q4", "L1 vs g0_amount_log 折内 ρ≈−0.073 是否构成去冗余？"),
    ("Q5", "NAV 逐日对账首个分叉日及此后解释是什么？"),
    ("Q6", "为何 NAV≈不变但 MaxDD 改善约 1.14pct？"),
    ("Q7", "HISTORICAL_FEE_CORRECTED 下 E0–E3 的真实 Sharpe 是多少？"),
    ("Q8", "费用校正后方向是否翻转？"),
    ("Q9", "Bootstrap N=5000 block=20 的配对日收益结果是什么？"),
    ("Q10", "SPA 观测统计量 / 备择数 / 数值 p 是多少？"),
    ("Q11", "Seed 状态枚举（COMPLETE/NOT_TRIGGERED/BLOCKED/FAILED）是什么？"),
    ("Q12", "N1/N3 是否截面退化？"),
    ("Q13", "N1/N3 若构造 bug 是否已修？确认无截面变化则是否进 Rejected Archive？"),
    ("Q14", "数据恢复协议对 G/H 短历史做了哪些尝试？"),
    ("Q15", "H/G 短历史 STRICT_PIT_SUBPERIOD_EVIDENCE 是否与全历史 v5 直接比？"),
    ("Q16", "本轮冻结了哪些 T/U/V 公式？几个 DUPLICATE？"),
    ("Q17", "T 族订单票是否提供独立于 G 的信息？"),
    ("Q18", "U 族触板前后状态是否有效？"),
    ("Q19", "V 族公告供给是否有效？V3 是否沿用 L2？"),
    ("Q20", "evaluate_factor Gate0–13 有多少因子走完？"),
    ("Q21", "哪些新因子通过真实组合回测？"),
    ("Q22", "最佳候选比 v5 提高多少 Sharpe / CAGR / NAV？"),
    ("Q23", "是否达到冻结经济意义门？"),
    ("Q24", "MaxDD 有没有恶化超过 2pct？"),
    ("Q25", "2022 / Ex2024 / Ex2025 是否稳健？"),
    ("Q26", "历史费用校正后增量是否仍存在？"),
    ("Q27", "结果有无统计支持（数值 p）？"),
    ("Q28", "是否具有独立 Forward OOS？"),
    ("Q29", "最终决策 KEEP_BASELINE_V5 还是 V6 候选？"),
)


def exp_dir(code: str) -> Path:
    return ROUND21_3_ROOT / "candidates" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND21_3_ROOT / "weights" / f"overlayed_{code}.parquet"


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


def material_vs_v5(obs: dict[str, Any], *, top_changed: bool = False) -> dict[str, Any]:
    gate = ECONOMIC_MATERIALITY
    dc = None if obs.get("cagr") is None else float(obs["cagr"]) - float(V5_ANCHOR["cagr"])
    ds = None if obs.get("sharpe") is None else float(obs["sharpe"]) - float(V5_ANCHOR["sharpe"])
    dn = None if obs.get("nav_end") is None else abs(float(obs["nav_end"]) - float(V5_NAV_END_EXACT)) / float(V5_NAV_END_EXACT)
    dd = None if obs.get("max_drawdown") is None else float(obs["max_drawdown"]) - float(V5_ANCHOR["max_drawdown"])
    # MaxDD 更负 = 恶化。dd < -0.02 则恶化超过 2pct。
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
