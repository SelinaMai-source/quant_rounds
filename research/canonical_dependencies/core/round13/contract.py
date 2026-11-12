"""Round 13 冻结契约。MODEL_SELECTION_VARIANCE：只对 v2/v3/v4 做 target-weight 聚合。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND11_ROOT, ROUND12_ROOT, ROUND13_ROOT
from ..round11.contract import (
    BASELINE_V2,
    BASELINE_V3,
    BASELINE_V4,
    BASELINES,
    BLEND_PRED,
    PARITY_ABS_NAV,
    PARITY_METRIC_ABS,
    RISK_PRED,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    STRATEGY_SOURCE_FILES,
)
from ..round12.contract import R12_01, R12_01_TARGETS, RANK_NAV, TOP_K

KEEP_BASELINE_V4 = "KEEP_BASELINE_V4"
RECOMMENDED_BASELINE_V5 = "RECOMMENDED_BASELINE_V5"
AWAITING_USER_LOCK = "AWAITING_USER_LOCK"
AUTO_PROMOTE_V5 = False
PARENT_V4 = "R9_B03"

MANIFEST_NAME = "round13_manifest_v1.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round13_baseline_consolidation_a29eb352.plan.md")
PLAN_COPY = ROUND13_ROOT / "ROUND13_PLAN.md"
TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"

R13_00 = "R13_00_BASELINE_V4"
R13_CTRL_V2 = "R13_CTRL_V2"
R13_CTRL_V3 = "R13_CTRL_V3"
A01 = "A01"
A02 = "A02"
A03 = "A03"
A04 = "A04"
A05 = "A05"

CONTROL_IDS = (R13_00, R13_CTRL_V2, R13_CTRL_V3)
CANDIDATE_IDS = (A01, A02, A03, A04, A05)
PRE_REGISTERED_IDS = CONTROL_IDS + CANDIDATE_IDS
STAT_IDS = ("v2", "v3", "v4", A01, A02, A03, A04, A05)
COMPLEXITY_ORDER = (A01, A03, A02, A04, A05)

R11_V2_CORRECTED_DIR = ROUND11_ROOT / "cost" / "R11_V2_CORRECTED"
R11_V3_CORRECTED_DIR = ROUND11_ROOT / "cost" / "R11_V3_CORRECTED"
R12_01_DIR = ROUND12_ROOT / "parity" / R12_01

CTRL_V2_TARGETS = {
    "cagr": 0.4892276960723958,
    "sharpe": 1.669435528345692,
    "max_drawdown": -0.2575106398530568,
    "completed_trades": 1787,
    "total_return": 29.18685022505568,
    "loyo_median": 0.4252289853301098,
    "loyo_worst": 0.2996739097021752,
}
CTRL_V3_TARGETS = {
    "cagr": 0.5161023969543328,
    "sharpe": 1.7611698014453812,
    "max_drawdown": -0.2509299072080845,
    "completed_trades": 1783,
    "total_return": 34.17825155826645,
    "loyo_median": 0.45077162444806773,
    "loyo_worst": 0.31835326433139244,
}

PARTICIPATION = 0.03
CAPITALS = (300_000.0, 1_000_000.0, 3_000_000.0)
DIAGNOSTIC_CAPITAL = 10_000_000.0
CONTRACT_CASH = 300_000.0
WANT_TOP2_EPS = 1e-6
NEAR_TIE_CAGR_PP = 0.005
NEAR_TIE_SHARPE = 0.03

BLOCK_DAYS = 20
BOOT_N = 5000
BOOT_N_MIN = 2000
OOB_N = 2000
PLACEBO_N = 1000
TIMING_N = 500
SPA_BOOT = 2000
EXEC_BPS = (0.0, 10.0, 20.0, 30.0, 50.0)
FALLBACK_P50_BPS = 20.0
FALLBACK_P75_BPS = 30.0

S9_STATUS = "SCENARIO_NOT_PREDEFINED"
S10_UP_RATIO = 0.45
B01A_T1 = 0.45

PROMOTION_MAX_CAGR_DROP_PP = 1.0
PROMOTION_SHARPE_SLACK = 0.03
PROMOTION_MDD_SLACK_PP = 1.0
PROMOTION_LOYO_SLACK_PP = 0.5
SPA_SUPPORT_P = 0.10
OOB_P_THRESHOLD = 0.60

STAGES = (
    "freeze",
    "parity",
    "weights",
    "candidates",
    "scenarios",
    "bootstrap",
    "oob",
    "cpcv",
    "spa",
    "placebo",
    "timing",
    "execution",
    "capacity",
    "forward",
    "promotion",
    "report",
)

PROTOCOL_SOURCE_FILES = (
    "core/round13/__init__.py",
    "core/round13/contract.py",
    "core/round13/common.py",
    "core/round13/freeze.py",
    "core/round13/runner.py",
    "core/round13/weights.py",
    "core/round13/candidates.py",
    "core/round13/scenarios.py",
    "core/round13/bootstrap.py",
    "core/round13/oob.py",
    "core/round13/pipeline.py",
    "core/round13/docs.py",
    "tools/run_round13.py",
)

CANDIDATE_FORMULAS = {
    A01: "w = 0.50 w_v3 + 0.50 w_v4",
    A02: "w = (w_v2 + w_v3 + w_v4) / 3",
    A03: "w = median(w_v2, w_v3, w_v4) per (date, stock); no extra renormalize except engine clip",
    A04: "Top1=v4; Top2 only if v3 and v4 both want Top2; same name → min(w3,w4); else no Top2",
    A05: "Top1=v4; same Top2 → mean; only one wants Top2 → 50% of that side; different Top2 → no Top2",
}

PRE_REGISTERED_QUESTIONS: dict[str, str] = {
    "Q1": "Manifest 是否在任何候选绩效之前冻结？sha256？S9 是否锁为 SCENARIO_NOT_PREDEFINED？",
    "Q2": "R13_00 是否与 R12_01_CORRECTED_V4 精确 parity？",
    "Q3": "R13_CTRL_V2 / R13_CTRL_V3 是否与 R11 corrected 账本精确 parity？失败是否 STOP？",
    "Q4": "本轮有没有发明额外候选、60/40 网格、A04b、或因结果改预注册公式？",
    "Q5": "日度目标权重是否来自 locked alpha + exact overlay（v2/v3/v4），而不是两条 NAV 相加？",
    "Q6": "是否统一单组合回测？有没有对两条 NAV 做平均？",
    "Q7": "引擎公式是否 RANK_NAV=(0.3,0.1)、TOP_K=2，并用 allocation_multiplier 注入？",
    "Q8": "某日 union 是否意外出现第 3 名？是否 STOP 而非改 top_k？",
    "Q9": "A01–A05 公式是否与 freeze 清单一致？v3/v4 Top2 分歧频率（身份 vs 仓位）？",
    "Q10": "近平局是否按 A01>A03>A02>A04>A05 留更简单者？",
    "Q11": "主评价合同是否 HISTORICAL_FEE_CORRECTED？锚点是否 R12_01？",
    "Q12": "A01–A05 corrected CAGR/Sharpe/MaxDD/LOYO 相对 v4？",
    "Q13": "情景 S0–S10 是否同一套切点、未调参？",
    "Q14": "S9 high-vol 是否 SCENARIO_NOT_PREDEFINED，且不进 promotion worst/median？",
    "Q15": "S10 是否复用已冻结 B01A up_ratio<0.45，而不是新阈值？",
    "Q16": "Pareto / minimax / max-regret 结论？median/worst scenario CAGR/Sharpe/MaxDD？",
    "Q17": "Block bootstrap 20 日、N≥2000 的 rank mean/std、p10/p90、top1/top3 频率？",
    "Q18": "Winner-vs-ensemble OOB：P(Axx>winner)、median delta、5/95 CI？",
    "Q19": "CPCV（embargo=12、块≥60）median/mean/p10/worst、Sharpe>0/>1 比例？",
    "Q20": "Family SPA：H0=A01–A05 相对 corrected v4 无更优；raw p 与 adjusted p？",
    "Q21": "全引擎 matched placebo ≥1000 的 CAGR/Sharpe/MDD/Worst12M 百分位？",
    "Q22": "Timing placebo（年×候选数桶×风险桶）≥500？PnL/单位暴露？",
    "Q23": "执行弹性 0/10/20/30/50bp 斜率与 break-even？有没有优化切点？",
    "Q24": "容量 300k/1M/3M？10M 是否仅 diagnostic？",
    "Q25": "2026-07-24 之后是否存在完整未选参 forward OOS？否则 FORWARD_OOS_NOT_AVAILABLE？",
    "Q26": "Promotion gate（Core+Robustness+Scenario 全过，且 Model-selection≥2）是否满足？失败了哪一扇门？",
    "Q27": "ALPHA_BASELINE 与 DEPLOYMENT_REFERENCE 是否分开？部署假设有没有进入 Alpha lineage？",
    "Q28": "正式裁决 KEEP_BASELINE_V4 还是 RECOMMENDED_BASELINE_V5 + AWAITING_USER_LOCK？有没有把 v5 写成 CURRENT？",
}


def experiment_dir(code: str) -> Path:
    return ROUND13_ROOT / "candidates" / str(code)


def parity_dir(code: str) -> Path:
    return ROUND13_ROOT / "parity" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND13_ROOT / "weights" / f"overlayed_{code}.parquet"


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round13_tree

    ensure_round13_tree()
    assert_project_result_path(PLAN_COPY)
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    if not text.startswith("<!-- 本文件是开工计划"):
        PLAN_COPY.write_text(header + text, encoding="utf-8")
    else:
        PLAN_COPY.write_text(text, encoding="utf-8")
    return PLAN_COPY


def pre_registered_spec(code: str) -> dict[str, Any]:
    table: dict[str, dict[str, Any]] = {
        R13_00: {
            "role": "corrected_v4_parity_anchor",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "parent": R12_01,
            "is_v5": False,
        },
        R13_CTRL_V2: {
            "role": "corrected_v2_parity_control",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "overlay": {"kind": "exact_v2"},
            "is_v5": False,
        },
        R13_CTRL_V3: {
            "role": "corrected_v3_parity_control",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "overlay": {"kind": "exact_e02"},
            "is_v5": False,
        },
    }
    for cid in CANDIDATE_IDS:
        table[cid] = {
            "role": "aggregation_candidate",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "formula": CANDIDATE_FORMULAS[cid],
            "top_k": TOP_K,
            "is_v5": False,
            "search": False,
        }
    if code not in table:
        raise KeyError(code)
    return {"code": code, **table[code]}


__all__ = [
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "AWAITING_USER_LOCK",
    "AUTO_PROMOTE_V5",
    "B01A_T1",
    "BASELINE_V2",
    "BASELINE_V3",
    "BASELINE_V4",
    "BASELINES",
    "BLEND_PRED",
    "BLOCK_DAYS",
    "BOOT_N",
    "BOOT_N_MIN",
    "CANDIDATE_FORMULAS",
    "CANDIDATE_IDS",
    "CAPITALS",
    "COMPLEXITY_ORDER",
    "CONTRACT_CASH",
    "CONTROL_IDS",
    "CTRL_V2_TARGETS",
    "CTRL_V3_TARGETS",
    "CURSOR_PLAN",
    "DIAGNOSTIC_CAPITAL",
    "EXEC_BPS",
    "FALLBACK_P50_BPS",
    "FALLBACK_P75_BPS",
    "KEEP_BASELINE_V4",
    "MANIFEST_NAME",
    "NEAR_TIE_CAGR_PP",
    "NEAR_TIE_SHARPE",
    "OOB_N",
    "OOB_P_THRESHOLD",
    "PARITY_ABS_NAV",
    "PARITY_METRIC_ABS",
    "PARENT_V4",
    "PARTICIPATION",
    "PLACEBO_N",
    "PLAN_COPY",
    "PRE_REGISTERED_IDS",
    "PRE_REGISTERED_QUESTIONS",
    "PROMOTION_LOYO_SLACK_PP",
    "PROMOTION_MAX_CAGR_DROP_PP",
    "PROMOTION_MDD_SLACK_PP",
    "PROMOTION_SHARPE_SLACK",
    "PROTOCOL_SOURCE_FILES",
    "R11_V2_CORRECTED_DIR",
    "R11_V3_CORRECTED_DIR",
    "R12_01",
    "R12_01_DIR",
    "R12_01_TARGETS",
    "R13_00",
    "R13_CTRL_V2",
    "R13_CTRL_V3",
    "RANK_NAV",
    "RECOMMENDED_BASELINE_V5",
    "RISK_PRED",
    "S10_UP_RATIO",
    "S9_STATUS",
    "SNAPSHOT_FINGERPRINT",
    "SNAPSHOT_ID",
    "SPA_BOOT",
    "SPA_SUPPORT_P",
    "STAGES",
    "STAT_IDS",
    "STRATEGY_SOURCE_FILES",
    "TIMING_N",
    "TOP_K",
    "TRACKER_PATH",
    "WANT_TOP2_EPS",
    "copy_plan",
    "experiment_dir",
    "overlayed_path",
    "parity_dir",
    "pre_registered_spec",
]
