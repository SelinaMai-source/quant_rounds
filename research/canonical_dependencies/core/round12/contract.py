"""Round 12 冻结契约。Risk Remediation，不是 Alpha 搜索。默认 KEEP_BASELINE_V4。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND9_ROOT, ROUND11_ROOT, ROUND12_ROOT
from ..round11.contract import (
    BASELINE_V1,
    BASELINE_V2,
    BASELINE_V3,
    BASELINE_V4,
    BASELINES,
    BLEND_PRED,
    PARITY_ABS_NAV,
    PARITY_METRIC_ABS,
    PROTOCOL_SOURCE_FILES as R11_PROTOCOL,
    RISK_PRED,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    STRATEGY_SOURCE_FILES,
    V4_TARGETS,
)

KEEP_BASELINE_V4 = "KEEP_BASELINE_V4"
AUTO_PROMOTE_V5 = False
PARENT_V4 = "R9_B03"
R11_V4_CORRECTED = "R11_V4_CORRECTED"
R11_V4_CORRECTED_DIR = ROUND11_ROOT / "cost" / R11_V4_CORRECTED
V4_DIR = ROUND9_ROOT / BASELINE_V4

MANIFEST_NAME = "round12_manifest_v1.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round12_risk_remediation_a21f0fc3.plan.md")
PLAN_COPY = ROUND12_ROOT / "ROUND12_PLAN.md"
TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"

R12_00 = "R12_00_LEGACY_V4"
R12_01 = "R12_01_CORRECTED_V4"
R12_02 = "R12_02_PIT_UNIVERSE"
R12_03 = "R12_03_EXEC_MEDIAN"
R12_04 = "R12_04_EXEC_CONSERVATIVE"
R12_05 = "R12_05_LIQUIDITY_PRIORITY"
R12_06 = "R12_06_LIQ_EXEC"
R12_07 = "R12_07_LIQ_EXEC_CONSERVATIVE"

PRE_REGISTERED_IDS = (R12_00, R12_01, R12_02, R12_03, R12_04, R12_05, R12_06, R12_07)
ANCHOR_IDS = (R12_00, R12_01)
VARIANT_IDS = (R12_02, R12_03, R12_04, R12_05, R12_06, R12_07)

R12_01_TARGETS = {
    "cagr": 0.5253672918108749,
    "sharpe": 1.8120914675763191,
    "max_drawdown": -0.25455988069271596,
    "completed_trades": 1149,
    "total_return": 36.06055834433112,
    "loyo_median": 0.4634117772829208,
    "loyo_worst": 0.324869449262855,
}

PARTICIPATION = 0.03
CAPITALS = (300_000.0, 1_000_000.0, 3_000_000.0)
DIAGNOSTIC_CAPITAL = 10_000_000.0
CONTRACT_CASH = 300_000.0
RANK_NAV = (0.3, 0.1)
TOP_K = 2
FALLBACK_P50_BPS = 20.0
FALLBACK_P75_BPS = 30.0
FALLBACK_P90_BPS = 50.0

MINUTE_ROOT = Path("/home/mxy/AmazingData/data/amazingdata_raw/kline_min1")
HERMES_SNAPSHOT = PROJECT_ROOT.parents[1] / "data" / "datasets" / "hermes-cn-a-2010-20260724-v7-all-features"
if not HERMES_SNAPSHOT.is_dir():
    HERMES_SNAPSHOT = Path("/home/mxy/Stock/data/datasets/hermes-cn-a-2010-20260724-v7-all-features")

CHINEXT_20PCT_FROM = "2020-08-24"
T_DECISION = "D+1 09:25"
FILL_SEMANTICS = "SAME_OPEN_EXECUTION_OPTIMISM"

PROMOTION_MAX_CAGR_DROP_PP = 1.0
PROMOTION_SHARPE_SLACK = 0.10
PROMOTION_MDD_SLACK = 0.02
PROMOTION_LOYO_SLACK = 0.02

VERDICTS = (
    "ROBUST_BUT_FORWARD_OOS_REQUIRED",
    "PROMISING_BUT_SELECTION_RISK",
    "EXECUTION_RISK_TOO_HIGH",
    "SURVIVORSHIP_RISK_TOO_HIGH",
    "BACKTEST_FRAGILE",
    "INVALID_BACKTEST",
)

RISK_LEVELS = ("LOW", "MODERATE", "HIGH", "CRITICAL", "UNQUANTIFIED")

STAGES = (
    "intake",
    "parity",
    "universe",
    "execution",
    "deploy",
    "trials",
    "dsr",
    "pbo",
    "alphaforge",
    "placebo",
    "timing-placebo",
    "staircase",
    "capacity",
    "shadow",
    "controls",
    "stats",
    "report",
)

PROTOCOL_SOURCE_FILES = (
    "core/round12/contract.py",
    "core/round12/freeze.py",
    "core/round12/common.py",
    "core/round12/pipeline.py",
    "core/round12/intake.py",
    "core/round12/parity.py",
    "core/round12/universe.py",
    "core/round12/execution.py",
    "core/round12/deploy.py",
    "core/round12/runner.py",
    "core/round12/trials.py",
    "core/round12/dsr.py",
    "core/round12/stats.py",
    "core/round12/placebo.py",
    "core/round12/timing_placebo.py",
    "core/round12/alphaforge.py",
    "core/round12/shadow.py",
    "core/round12/docs.py",
    "tools/run_round12.py",
    "scripts/run_round12.py",
    "tools/run_shadow_daily.py",
)

PRE_REGISTERED_QUESTIONS: dict[str, str] = {
    "Q1": "Manifest 是否在任何新策略绩效之前冻结？sha256？",
    "Q2": "R12_00 是否与 legacy v4（R9_B03）NAV/交易/CAGR/Sharpe/MaxDD 精确 parity？",
    "Q3": "R12_01 是否与 R11_V4_CORRECTED 精确 parity？它是不是 v5？",
    "Q4": "本轮有没有发明额外策略变体、搜 cap、或因结果改预注册规则？",
    "Q5": "PIT 宇宙是否用 daily_raw spine + known_at≤day 的 ST，且未用当前证券列表 / membership.effective_to？",
    "Q6": "是否存在独立退市公告 PIT？缺证据能否标 SURVIVORSHIP_PASS？",
    "Q7": "逐信号日 vs 候选面板的 n / only 集合 / Top1-Top2 影响？终审？",
    "Q8": "R12_02 是否跑了？质量门槛是什么？",
    "Q9": "成交因果链是否钉死 information≤T_DECISION<order≤fill？官方 open 成交如何标记？",
    "Q10": "全市场 1 分钟覆盖？有无 09:25 竞价？2022+ 是否可识别？",
    "Q11": "EXEC_P50/P75/P90 取值、校准/验证切、是否 FIXED_STRESS_FALLBACK？",
    "Q12": "Execution 四分类：SUPPORTED / OPTIMISTIC / HIGH_RISK / NOT_IDENTIFIABLE？",
    "Q13": "R12_03 / R12_04 成绩？相对 R12_01 的 alpha 留存？",
    "Q14": "流动性规则是否为预注册 3% + Top1 先分完？有没有搜 cap？",
    "Q15": "300k / 1M / 3M 容量评级？10M 是否仅 diagnostic？",
    "Q16": "R12_06（主 DEPLOYMENT_CANDIDATE）成绩、fill rate、Top1/Top2 fill？",
    "Q17": "R12_07 conservative 成绩？能否作为更保守部署上界？",
    "Q18": "相对 legacy v4 与 R12_01 的 CAGR/Sharpe/MaxDD/LOYO 留存？",
    "Q19": "会计分解（非因果）：legacy→印花税→P50/P75→流动性？",
    "Q20": "registry v2 与 COMPARABLE_UNIVERSES 的 TRUE PBO 2.0（取消 max_cols=80）？",
    "Q21": "全宇宙 TRUE_GLOBAL_PBO 是否可识别？可否把可恢复子集外推到 1339/2678？",
    "Q22": "CANONICAL_DSR（日频 SR + 日 N）值、提名 N、单测是否通过？",
    "Q23": "AlphaForge 对指定仓库是否只重试一次？失败是否阻断？",
    "Q24": "全引擎 matched placebo ≥1000 的 CAGR/Sharpe/MDD/Worst12M 百分位？",
    "Q25": "State-timing placebo（v2 confidence / v3 disagreement / v4 B03）真实点与经验 p？",
    "Q26": "v1→v2→v3→v4 staircase：STRUCTURAL_INCREMENT / WEAK / SELECTION_SENSITIVE / NOT_IDENTIFIABLE？",
    "Q27": "2026-07-24 之后是否存在完整未选参 forward OOS？否则 shadow 协议是否落地？",
    "Q28": "升 v5 门槛是否满足？正式裁决 KEEP_BASELINE_V4 还是升 v5？六选一 verdict？",
}

REPORT_SECTIONS = (
    "01_mission_and_freeze",
    "02_round11_risk_intake",
    "03_anchor_parity",
    "04_pit_universe",
    "05_survivorship_audit",
    "06_execution_inventory",
    "07_minute_calibration",
    "08_deploy_variants",
    "09_liquidity_capacity",
    "10_alpha_retention",
    "11_trial_registry_v2",
    "12_true_pbo_2",
    "13_canonical_dsr",
    "14_alphaforge",
    "15_full_engine_placebo",
    "16_state_timing_placebo",
    "17_staircase_spa",
    "18_forward_shadow",
    "19_performance_ladder",
    "20_promotion_gate",
    "21_six_way_verdict",
    "22_risk_dimensions",
    "23_questions_and_limits",
)


def experiment_dir(code: str) -> Path:
    return ROUND12_ROOT / "deploy" / str(code)


def parity_dir(code: str) -> Path:
    return ROUND12_ROOT / "parity" / str(code)


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round12_tree

    ensure_round12_tree()
    assert_project_result_path(PLAN_COPY)
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    if not text.startswith("<!-- 本文件是开工计划"):
        PLAN_COPY.write_text(header + text, encoding="utf-8")
    else:
        PLAN_COPY.write_text(text, encoding="utf-8")
    return PLAN_COPY


def pre_registered_spec(code: str) -> dict[str, Any]:
    table = {
        R12_00: {
            "parent": PARENT_V4,
            "fee": "LEGACY_FLAT_5BP",
            "role": "legacy_parity_anchor",
            "is_v5": False,
        },
        R12_01: {
            "parent": PARENT_V4,
            "fee": "HISTORICAL_FEE_CORRECTED",
            "role": "REALISM_REFERENCE_NOT_V5",
            "is_v5": False,
        },
        R12_02: {
            "parent": R12_01,
            "fee": "HISTORICAL_FEE_CORRECTED",
            "role": "pit_filter_only_if_quality_ok",
            "conditional": True,
            "is_v5": False,
        },
        R12_03: {
            "parent": R12_01,
            "fee": "HISTORICAL_FEE_CORRECTED",
            "exec": "EXEC_P50_OR_FALLBACK_20BP",
            "role": "execution_median",
            "is_v5": False,
        },
        R12_04: {
            "parent": R12_01,
            "fee": "HISTORICAL_FEE_CORRECTED",
            "exec": "EXEC_P75_OR_FALLBACK_30BP",
            "role": "execution_conservative",
            "is_v5": False,
        },
        R12_05: {
            "parent": R12_01,
            "fee": "HISTORICAL_FEE_CORRECTED",
            "liquidity": "PRIORITY_3PCT_TOP1_FIRST",
            "role": "liquidity_priority",
            "is_v5": False,
        },
        R12_06: {
            "parent": R12_05,
            "fee": "HISTORICAL_FEE_CORRECTED",
            "exec": "EXEC_P50_OR_FALLBACK_20BP",
            "liquidity": "PRIORITY_3PCT_TOP1_FIRST",
            "role": "DEPLOYMENT_CANDIDATE",
            "is_v5": False,
        },
        R12_07: {
            "parent": R12_05,
            "fee": "HISTORICAL_FEE_CORRECTED",
            "exec": "EXEC_P75_OR_FALLBACK_30BP",
            "liquidity": "PRIORITY_3PCT_TOP1_FIRST",
            "role": "conservative_deployment",
            "is_v5": False,
        },
    }
    if code not in table:
        raise KeyError(code)
    return {"code": code, **table[code]}


__all__ = [
    "ANCHOR_IDS",
    "AUTO_PROMOTE_V5",
    "BASELINE_V1",
    "BASELINE_V2",
    "BASELINE_V3",
    "BASELINE_V4",
    "BASELINES",
    "BLEND_PRED",
    "CAPITALS",
    "CHINEXT_20PCT_FROM",
    "CONTRACT_CASH",
    "COPY_PLAN",
    "CURSOR_PLAN",
    "DIAGNOSTIC_CAPITAL",
    "FALLBACK_P50_BPS",
    "FALLBACK_P75_BPS",
    "FALLBACK_P90_BPS",
    "FILL_SEMANTICS",
    "HERMES_SNAPSHOT",
    "KEEP_BASELINE_V4",
    "MANIFEST_NAME",
    "MINUTE_ROOT",
    "PARITY_ABS_NAV",
    "PARITY_METRIC_ABS",
    "PARENT_V4",
    "PARTICIPATION",
    "PLAN_COPY",
    "PRE_REGISTERED_IDS",
    "PRE_REGISTERED_QUESTIONS",
    "PROMOTION_LOYO_SLACK",
    "PROMOTION_MAX_CAGR_DROP_PP",
    "PROMOTION_MDD_SLACK",
    "PROMOTION_SHARPE_SLACK",
    "PROTOCOL_SOURCE_FILES",
    "R11_PROTOCOL",
    "R11_V4_CORRECTED",
    "R11_V4_CORRECTED_DIR",
    "R12_00",
    "R12_01",
    "R12_01_TARGETS",
    "R12_02",
    "R12_03",
    "R12_04",
    "R12_05",
    "R12_06",
    "R12_07",
    "RANK_NAV",
    "REPORT_SECTIONS",
    "RISK_LEVELS",
    "RISK_PRED",
    "SNAPSHOT_FINGERPRINT",
    "SNAPSHOT_ID",
    "STAGES",
    "STRATEGY_SOURCE_FILES",
    "T_DECISION",
    "TOP_K",
    "TRACKER_PATH",
    "V4_DIR",
    "V4_TARGETS",
    "VARIANT_IDS",
    "VERDICTS",
    "copy_plan",
    "experiment_dir",
    "parity_dir",
    "pre_registered_spec",
]
