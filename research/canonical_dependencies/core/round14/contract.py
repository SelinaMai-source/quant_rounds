"""Round 14 冻结契约。只改 Top1/Top2 目标资金，不改选股身份。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import RESULT_ROOT, ROUND12_ROOT, ROUND13_ROOT, ROUND14_ROOT
from ..round11.contract import (
    BASELINE_V4,
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
DE_RISKING_ONLY = "DE_RISKING_ONLY"
RISK_ALLOCATION_INCREMENT = "RISK_ALLOCATION_INCREMENT"
FAIL_ALPHA_MUTATION = "FAIL_ALPHA_MUTATION"

MANIFEST_NAME = "round14_manifest_v1.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round14_risk_budgeting_386d7f47.plan.md")
PLAN_COPY = ROUND14_ROOT / "ROUND14_PLAN.md"
TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FINAL_REPORT = ROUND14_ROOT / "ROUND14_RISK_BUDGET_REPORT.md"

R14_00 = "R14_00_CORRECTED_V4"
A01 = "A01"
B01 = "B01"
B02 = "B02"
B03 = "B03"
C01 = "C01"

CANDIDATE_IDS = (A01, B01, B02, B03, C01)
CONTROL_IDS = tuple(f"CTRL_EXP_{cid}" for cid in CANDIDATE_IDS)
PRE_REGISTERED_IDS = (R14_00, *CANDIDATE_IDS, *CONTROL_IDS)
STAT_IDS = ("v4", *CANDIDATE_IDS)
COMPLEXITY_ORDER = (B01, B02, A01, B03, C01)

R12_01_DIR = ROUND12_ROOT / "parity" / R12_01
R13_00_DIR = ROUND13_ROOT / "parity" / "R13_00_BASELINE_V4"

SIGMA_WINDOW = 20
SIGMA_MIN_OBS = 15
RHO_WINDOW = 60
RHO_MIN_OBS = 40
REF_SIGNAL_DAYS = 250
REF_MIN_OBS = 100
INCLUDE_D = True

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
SPA_BOOT = 2000
EXEC_BPS = (0.0, 10.0, 20.0, 30.0, 50.0)
FALLBACK_P50_BPS = 20.0
FALLBACK_P75_BPS = 30.0

S8_UP_RATIO = 0.45
B01A_T1 = 0.45

PROMOTION_MAX_CAGR_DROP_PP = 1.0
PROMOTION_SHARPE_SLACK = 0.03
PROMOTION_MDD_SLACK_PP = 0.50
PROMOTION_LOYO_SLACK_PP = 0.50
IMPROVE_N_REQUIRED = 3
SPA_SUPPORT_P = 0.10

STAGES = (
    "parity",
    "risk",
    "freeze",
    "candidates",
    "controls",
    "scenarios",
    "execution",
    "capacity",
    "tail",
    "bootstrap",
    "matched_bootstrap",
    "cpcv",
    "spa",
    "forward",
    "promotion",
    "report",
)

PROTOCOL_SOURCE_FILES = (
    "core/round14/__init__.py",
    "core/round14/contract.py",
    "core/round14/common.py",
    "core/round14/runner.py",
    "core/round14/parity.py",
    "core/round14/risk.py",
    "core/round14/freeze.py",
    "core/round14/weights.py",
    "core/round14/tests.py",
    "core/round14/candidates.py",
    "core/round14/scenarios.py",
    "core/round14/stats.py",
    "core/round14/docs.py",
    "core/round14/pipeline.py",
    "tools/run_round14.py",
)

CANDIDATE_FORMULAS = {
    A01: (
        "Top1=v4. Top2 piecewise-linear on exact agreement × top2_minus_median_p250. "
        "A: p<=0.20→0; 0.20-0.50 linear to 5%; >0.50 linear to 10%. "
        "B: p<=0.50→0, then linear to 5%. C: 0. No sigmoid/power."
    ),
    B01: "w2=w2_v4 * min(1, median_sigma2_250/sigma2_20); only decrease; missing→v4",
    B02: "w2=w2_v4 * min(1, median_MRC2_250/MRC2_v4) if MRC2>0 and ref available; MRC2<=0→factor=1",
    B03: "same portfolio_risk_factor=min(1, median_sigma_portfolio_250/sigma_portfolio) multiplies w1 and w2; no vol grid",
    C01: "w1=v4; w2=A01 w2_smooth * B02-style MRC factor using Sigma from (w1_v4, w2_smooth); no extra stack",
}

PRE_REGISTERED_QUESTIONS: dict[str, str] = {
    "Q1": "Manifest 是否在风险诊断之后、任何候选绩效之前冻结？sha256？",
    "Q2": "R14_00 是否与 R12_01 / R13_00 精确 parity？失败是否 STOP？",
    "Q3": "风险面板是否 PIT（含 D 收盘）？sigma1_20/sigma2_20 最少 15 日？缺失是否回退 v4 原仓且不填 0？",
    "Q4": "rho12_60 最少 40 日？不足是否显式标记、不默默当 0？",
    "Q5": "Sigma / sigma_portfolio / MRC2 / RC_SHARE2 公式是否按冻结定义？MRC2≤0 时风险帽因子是否为 1？",
    "Q6": "参考分布是否过去 250 个 SIGNAL DAY（不含当日）、最少 100？不足是否回退 v4？",
    "Q7": "01_risk_feature_diagnostics.md 是否先于 freeze？诊断有没有改公式？",
    "Q8": "正式候选是否仅 A01/B01/B02/B03/C01？有无笛卡尔积或新搜阈值？",
    "Q9": "身份锁：逐日 Top1/Top2 code 是否等于 v4？失败是否 FAIL_ALPHA_MUTATION？",
    "Q10": "零 Top2 是否用 MIN_KEEP？test_zero_top2_does_not_pull_top3 是否通过？",
    "Q11": "A01 是否单调分段线性（agreement × 连续 p）？Top1=v4？有无 sigmoid/幂？",
    "Q12": "B01 是否 w2=w2_v4*min(1, median_sigma2_250/sigma2_20) 且只减不增？",
    "Q13": "B02 是否 w2=w2_v4*min(1, median_MRC2_250/MRC2_v4)？B02 vs CTRL_EXP_B02 结论？",
    "Q14": "B03 是否同一 portfolio_risk_factor 同时乘 w1 与 w2？有无 10/12/15% 网格？",
    "Q15": "C01 是否 A01 w2_smooth × B02 风格 MRC 因子（v4 Top1 + smooth Top2 估 Σ）？有无再叠层？",
    "Q16": "CTRL_EXP 是否 scale=E_cand/E_v4、w_ctrl=w_v4*scale、无风险信息、同日总暴露？",
    "Q17": "各候选相对对照是 RISK_ALLOCATION_INCREMENT 还是 DE_RISKING_ONLY？单位暴露 PnL？",
    "Q18": "主评价是否 HISTORICAL_FEE_CORRECTED？锚点是否 R12_01/R13_00 = 52.54%/1.812/-25.46%？",
    "Q19": "情景 S0–S9 是否同一套切点？S8=B01A up_ratio<0.45？S9=最差滚动 12M？",
    "Q20": "执行 0/10/20/30/50bp 与容量 300k/1M/3M（10M diagnostic）结果？",
    "Q21": "20 日块 bootstrap N=5000 的 rank mean/std、p10/p90、top1/top3？",
    "Q22": "matched-exposure paired bootstrap：P(候选>对照)？",
    "Q23": "CPCV（embargo=12、块≥60）median/mean/p10/worst？是否被写成 forward OOS？",
    "Q24": "Family SPA：H0=A01/B01/B02/B03/C01 相对 v4 无更优；raw p 与 adjusted p？",
    "Q25": "v4 最大 5 段回撤中，风险层是否在 ex-ante 风险高时减亏？右尾保留 vs 左尾削减？",
    "Q26": "Promotion 门（RETURN / STABILITY/RISK_BUDGET + 10 项中 3 项且含 7/8/9 + 模型选择≥2）谁过？失败哪扇门？",
    "Q27": "DE_RISKING_ONLY 有没有被升 v5？ALPHA_BASELINE 与 DEPLOYMENT_P50/P75 是否分开？",
    "Q28": "正式裁决 KEEP_BASELINE_V4 还是 RECOMMENDED_BASELINE_V5 + AWAITING_USER_LOCK？有没有把 v5 写成 CURRENT？追踪表有没有写「写入时间」？",
}


def experiment_dir(code: str) -> Path:
    return ROUND14_ROOT / "candidates" / str(code)


def control_dir(code: str) -> Path:
    return ROUND14_ROOT / "controls" / str(code)


def parity_dir(code: str) -> Path:
    return ROUND14_ROOT / "parity" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND14_ROOT / "weights" / f"overlayed_{code}.parquet"


def ctrl_code(cid: str) -> str:
    return f"CTRL_EXP_{cid}"


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round14_tree

    ensure_round14_tree()
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
        R14_00: {
            "role": "corrected_v4_parity_anchor",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "parent": R12_01,
            "is_v5": False,
        },
    }
    for cid in CANDIDATE_IDS:
        table[cid] = {
            "role": "risk_budget_candidate",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "formula": CANDIDATE_FORMULAS[cid],
            "top_k": TOP_K,
            "is_v5": False,
            "search": False,
        }
        table[ctrl_code(cid)] = {
            "role": "exposure_matched_control",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "parent": cid,
            "is_v5": False,
            "risk_information": False,
        }
    if code not in table:
        raise KeyError(code)
    return {"code": code, **table[code]}


__all__ = [
    "A01",
    "AWAITING_USER_LOCK",
    "AUTO_PROMOTE_V5",
    "B01",
    "B01A_T1",
    "B02",
    "B03",
    "BASELINE_V4",
    "BLEND_PRED",
    "BLOCK_DAYS",
    "BOOT_N",
    "BOOT_N_MIN",
    "C01",
    "CANDIDATE_FORMULAS",
    "CANDIDATE_IDS",
    "CAPITALS",
    "COMPLEXITY_ORDER",
    "CONTRACT_CASH",
    "CONTROL_IDS",
    "CURSOR_PLAN",
    "DE_RISKING_ONLY",
    "DIAGNOSTIC_CAPITAL",
    "EXEC_BPS",
    "FAIL_ALPHA_MUTATION",
    "FALLBACK_P50_BPS",
    "FALLBACK_P75_BPS",
    "FINAL_REPORT",
    "IMPROVE_N_REQUIRED",
    "INCLUDE_D",
    "KEEP_BASELINE_V4",
    "MANIFEST_NAME",
    "NEAR_TIE_CAGR_PP",
    "NEAR_TIE_SHARPE",
    "PARITY_ABS_NAV",
    "PARITY_METRIC_ABS",
    "PARENT_V4",
    "PARTICIPATION",
    "PLAN_COPY",
    "PRE_REGISTERED_IDS",
    "PRE_REGISTERED_QUESTIONS",
    "PROMOTION_LOYO_SLACK_PP",
    "PROMOTION_MAX_CAGR_DROP_PP",
    "PROMOTION_MDD_SLACK_PP",
    "PROMOTION_SHARPE_SLACK",
    "PROTOCOL_SOURCE_FILES",
    "R12_01",
    "R12_01_DIR",
    "R12_01_TARGETS",
    "R13_00_DIR",
    "R14_00",
    "RANK_NAV",
    "REF_MIN_OBS",
    "REF_SIGNAL_DAYS",
    "RECOMMENDED_BASELINE_V5",
    "RHO_MIN_OBS",
    "RHO_WINDOW",
    "RISK_ALLOCATION_INCREMENT",
    "RISK_PRED",
    "S8_UP_RATIO",
    "SIGMA_MIN_OBS",
    "SIGMA_WINDOW",
    "SNAPSHOT_FINGERPRINT",
    "SNAPSHOT_ID",
    "SPA_BOOT",
    "SPA_SUPPORT_P",
    "STAGES",
    "STAT_IDS",
    "STRATEGY_SOURCE_FILES",
    "TOP_K",
    "TRACKER_PATH",
    "WANT_TOP2_EPS",
    "copy_plan",
    "ctrl_code",
    "control_dir",
    "experiment_dir",
    "overlayed_path",
    "parity_dir",
    "pre_registered_spec",
]
