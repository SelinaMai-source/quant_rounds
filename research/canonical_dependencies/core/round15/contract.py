"""Round 15 冻结契约。期权尾部 × Top1 × 可成交入场。默认 KEEP_BASELINE_V4。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import RESULT_ROOT, ROUND12_ROOT, ROUND13_ROOT, ROUND14_ROOT, ROUND15_ROOT
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
from ..round12.contract import MINUTE_ROOT, R12_01, R12_01_TARGETS, RANK_NAV, TOP_K

KEEP_BASELINE_V4 = "KEEP_BASELINE_V4"
RECOMMENDED_BASELINE_V5 = "RECOMMENDED_BASELINE_V5"
AWAITING_USER_LOCK = "AWAITING_USER_LOCK"
AUTO_PROMOTE_V5 = False
PARENT_V4 = "R9_B03"
DE_RISKING_ONLY = "DE_RISKING_ONLY"
FAIL_ALPHA_MUTATION = "FAIL_ALPHA_MUTATION"
HISTORICAL_DEVELOPMENT_ONLY = "HISTORICAL_DEVELOPMENT_ONLY"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
EXECUTION_FULL_PERIOD_NOT_IDENTIFIABLE = "EXECUTION_FULL_PERIOD_NOT_IDENTIFIABLE"
FULL_HISTORY_NOT_AVAILABLE = "FULL_HISTORY_NOT_AVAILABLE"
UNQUANTIFIED = "UNQUANTIFIED"

MANIFEST_NAME = "round15_manifest_v1.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round15_option_top1_exec_97ffb322.plan.md")
PLAN_COPY = ROUND15_ROOT / "ROUND15_PLAN.md"
TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FINAL_REPORT = ROUND15_ROOT / "ROUND15_RESEARCH_REPORT.md"

R15_00 = "R15_00_CORRECTED_V4"
O01 = "O01"
O02 = "O02"
O03 = "O03"
O04 = "O04"
T01 = "T01"
T02 = "T02"
X01 = "X01"
X02 = "X02"

OPTION_IDS = (O01, O02, O03, O04)
TOP1_IDS = (T01, T02)
ALPHA_IDS = (*OPTION_IDS, *TOP1_IDS)
EXEC_IDS = (X01, X02)
CANDIDATE_IDS = ALPHA_IDS
PRE_REGISTERED_IDS = (R15_00, *ALPHA_IDS, *EXEC_IDS)

R12_01_DIR = ROUND12_ROOT / "parity" / R12_01
R13_00_DIR = ROUND13_ROOT / "parity" / "R13_00_BASELINE_V4"
R14_00_DIR = ROUND14_ROOT / "parity" / "R14_00_CORRECTED_V4"

ETF_UNDERLYINGS = ("510050.SH", "510300.SH", "510500.SH", "159915.SZ", "588000.SH")
OPTION_SECURITY_TYPE = "EXTRA_ETF_OP"

PCT_WINDOW = 252
PCT_MIN_OBS = 126
SKEW_Q = 0.90
IV_DELTA_Q = 0.90
LOOKBACK_VETO = None  # 90 分位用过去有效期权日，窗长 PCT_WINDOW
O02_HALVE = 0.5
O04_BETA_WINDOW = 60
PERM_N = 500
BLOCK_DAYS = 20
BOOT_N = 5000
BOOT_N_MIN = 2000
SPA_BOOT = 2000

T02_MIN_OOS_R2 = 0.02
T02_MIN_N = 200
T01_MIN_N = 80
IV_MIN_CHAIN = 4
MINUTE_YEARS = (2018, 2019, 2020, 2021)
X02_HM = "09:35"
AUCTION_HM = "09:25"

PROMOTION_MAX_CAGR_DROP_PP = 1.0
PROMOTION_SHARPE_SLACK = 0.03
PROMOTION_MDD_SLACK_PP = 0.50
PROMOTION_MDD_RETURN_SLACK_PP = 1.0
IMPROVE_N_REQUIRED = 3
MATCHED_P = 0.60

FORBIDDEN_SEARCH = (
    "top2_0_5_10_grid",
    "vol_mrc_cap_search",
    "v3_v4_averaging",
    "O05_plus",
    "T03_plus",
    "option_top1_combo_after_nav",
    "best_minute_search",
    "compare_subsample_cagr_to_full_period",
    "auto_promote_CURRENT_v5",
    "write_写入时间_to_tracker",
    "invent_option_data",
)

SKIP_CODES = (
    "SKIP_IV_SURFACE",
    "SKIP_PUT_SKEW25",
    "SKIP_F3",
    "SKIP_ADVANCED_OPTION_FACTORS",
    "SKIP_AUCTION_SIGNAL",
    "SKIP_O01",
    "SKIP_O02",
    "SKIP_O03",
    "SKIP_O04",
    "SKIP_T01",
    "SKIP_T02",
    "SKIP_X01",
    "SKIP_X02",
)

STAGES = (
    "parity",
    "inventory",
    "sdk",
    "freeze",
    "factors",
    "residual",
    "options",
    "top1",
    "execution",
    "survivorship",
    "stats",
    "forward",
    "promotion",
    "report",
)

PROTOCOL_SOURCE_FILES = (
    "core/round15/__init__.py",
    "core/round15/contract.py",
    "core/round15/common.py",
    "core/round15/runner.py",
    "core/round15/parity.py",
    "core/round15/inventory.py",
    "core/round15/sdk.py",
    "core/round15/freeze.py",
    "core/round15/factors.py",
    "core/round15/residual.py",
    "core/round15/options.py",
    "core/round15/top1.py",
    "core/round15/execution.py",
    "core/round15/survivorship.py",
    "core/round15/stats.py",
    "core/round15/docs.py",
    "core/round15/pipeline.py",
    "tools/run_round15.py",
)

CANDIDATE_FORMULAS = {
    O01: "parent=v4. PUT_SKEW25 > past 90p of valid option days → Top2 MIN_KEEP, Top1 unchanged.",
    O02: "parent=v4. past-5d ΔATM_IV30 > past 90p → portfolio target budget ×0.5. Controls=constant avg risk + year/regime random timing.",
    O03: "parent=v4. skew AND convexity both > own 90p → Top2 MIN_KEEP. Convexity missing → SKIP_O03.",
    O04: "stock vs matching-ETF past-return beta × that ETF option tail; walk-forward ridge on OOF residual net; five/ten frozen.",
    T01: "keep original Top1 rank; skip Top1 order if OOF predicted net margin < locked execution-cost hurdle.",
    T02: "limited OOF-residual rerank within same-day eligible Top1 candidates. Unreliable features → SKIP_T02.",
    X01: "first causally executable minute bar fill; common minute years only; not an Alpha family.",
    X02: "fixed 09:35 bar fill when the bar exists; common minute years only; not an Alpha family.",
}

PRE_REGISTERED_QUESTIONS: dict[str, str] = {
    "Q1": "Manifest 是否在盘点/SDK 之后、任何新策略 NAV 之前冻结？sha256？",
    "Q2": "R15_00 是否与 R12_01 / R13_00 / R14_00 精确 parity？失败是否 STOP？",
    "Q3": "本地是否有期权库？分钟覆盖（2017–2021 / 2022+）与首 bar、09:25 事实？",
    "Q4": "AmazingData SDK 是否按已安装包签名 inspect（而非手册 PDF 示例）？login 字段是否打码？",
    "Q5": "若可登录，是否先小样本再 resume 下载？无期权且无法下载是否诚实 SKIP？",
    "Q6": "SKIP 规则（SKIP_IV_SURFACE / SKIP_PUT_SKEW25 / SKIP_F3 / SKIP_ADVANCED_OPTION_FACTORS / SKIP_AUCTION_SIGNAL / SKIP_O0x）是否在看 NAV 前写入 freeze？",
    "Q7": "期权报价 PIT：timestamp ≤ D close？available_at 是否记录？失败是否 STOP_OPTION_BACKTEST？",
    "Q8": "O_F1–F4 定义是否在绩效前锁死？百分位是否只用过去 252 个有效期权日（最少 126）？",
    "Q9": "期权状态对未来广度/连板/Top1·Top2 净收益/左尾的增量，以及对已有股票风险因子的 OOS 残差（R2/AUC/Brier）？",
    "Q10": "O01–O04 哪些实际跑了、哪些 SKIP？理由？有无 O05+ 或看完收益再组合 Option+Top1？",
    "Q11": "O01 是否 PUT_SKEW25>90p → Top2=0、Top1 不变？对照是否同样数量的 Top2 veto 日？",
    "Q12": "O02 对照是否恒定平均风险 + 年/体制随机择时（≥500），而不是逐日 exposure-match？",
    "Q13": "O03 是否 skew 与 convexity 同时过 90 分位？凸性不足是否 SKIP_O03？",
    "Q14": "O04 是否股票×对应 ETF beta × 该 ETF option tail？覆盖不可靠是否 SKIP_O04？",
    "Q15": "Top1 归因（净/毛 PnL、成本、尾部、score/gap/流动性/封板质量、失败模式）？D+1 竞价是否进入 D-close 模型？",
    "Q16": "T01 是否保持原 Top1 排名、训练期锁定净边际阈值，预测净边 < 成本则不下单？",
    "Q17": "T02 是否仅在当日 eligible Top1 候选内有限 rerank？无可靠特征是否 SKIP_T02？",
    "Q18": "X01 是否第一根因果可成交 bar？X02 是否固定 09:35？是否仅共同分钟年份（2017–2021）？有无 09:25？",
    "Q19": "有没有拿 2017–2021 子样本 CAGR 去对 2018–2026 全样本？是否标记 EXECUTION_FULL_PERIOD_NOT_IDENTIFIABLE？",
    "Q20": "零 Top2 是否 MIN_KEEP？有无 Top3 补位？正式回测是否走 run_r12_backtest + 历史印花税？",
    "Q21": "Survivorship：AmazingData hist status 是否可用？否则是否 UNQUANTIFIED（不标 PASS）？",
    "Q22": "主评价是否 HISTORICAL_FEE_CORRECTED？锚点是否 R12_01 = 52.54%/1.812/-25.46%？",
    "Q23": "统计：期权一组 family、Top1 另一组；paired block bootstrap / HAC / CPCV / SPA？",
    "Q24": "Forward：快照是否止于 2026-07-24？是否 FORWARD_OOS_NOT_AVAILABLE？有无 FORWARD_VALIDATION_SUPPORTED / LIVE_VALIDATED？",
    "Q25": "RETURN 门：PIT PASS、CAGR>v4、Sharpe≥v4、MaxDD 恶化≤1.0pp、非 DE_RISKING_ONLY、CPCV median Sharpe≥v4？",
    "Q26": "STABILITY 门：CAGR≥v4−1.0pp、Sharpe≥v4−0.03、MaxDD 恶化≤0.50pp，且至少 3 项结构性改善？",
    "Q27": "EXECUTABILITY：是否仅因 X01/X02 好看而升？Alpha 候选在共同窗口上是否仍满足 RETURN 或 STABILITY？",
    "Q28": "正式裁决 KEEP_BASELINE_V4 还是 RECOMMENDED_BASELINE_V5 + AWAITING_USER_LOCK？有没有把 v5 写成 CURRENT？追踪表有没有写「写入时间」？",
}

SEAL_D_FEATURES = (
    "fb_limit_close_minute_ratio",
    "h5_close_limit_retention",
    "fb_touch_count",
    "fb_close_net_to_turnover",
    "h6_close_amount_ratio",
    "h6_last30_amount_ratio",
)
BANNED_DPLUS1 = ("d1_open_gap", "top2_d1_open_gap", "auction_0925", "d1_auction")


def experiment_dir(code: str) -> Path:
    return ROUND15_ROOT / "candidates" / str(code)


def control_dir(code: str) -> Path:
    return ROUND15_ROOT / "controls" / str(code)


def parity_dir(code: str) -> Path:
    return ROUND15_ROOT / "parity" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND15_ROOT / "weights" / f"overlayed_{code}.parquet"


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round15_tree

    ensure_round15_tree()
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
        R15_00: {
            "role": "corrected_v4_parity_anchor",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "parent": R12_01,
            "is_v5": False,
        },
    }
    for cid in OPTION_IDS:
        table[cid] = {
            "role": "option_alpha_candidate",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "formula": CANDIDATE_FORMULAS[cid],
            "parent": "v4",
            "family": "option",
            "is_v5": False,
            "search": False,
        }
    for cid in TOP1_IDS:
        table[cid] = {
            "role": "top1_alpha_candidate",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "formula": CANDIDATE_FORMULAS[cid],
            "parent": "v4",
            "family": "top1",
            "is_v5": False,
            "search": False,
        }
    for cid in EXEC_IDS:
        table[cid] = {
            "role": "execution_benchmark_not_alpha",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "formula": CANDIDATE_FORMULAS[cid],
            "parent": "v4",
            "family": "execution",
            "is_v5": False,
            "counts_as_alpha": False,
        }
    if code not in table:
        raise KeyError(code)
    return {"code": code, **table[code]}


__all__ = [
    "ALPHA_IDS",
    "AWAITING_USER_LOCK",
    "AUTO_PROMOTE_V5",
    "BANNED_DPLUS1",
    "BASELINE_V4",
    "BLEND_PRED",
    "BLOCK_DAYS",
    "BOOT_N",
    "BOOT_N_MIN",
    "CANDIDATE_FORMULAS",
    "CANDIDATE_IDS",
    "CURSOR_PLAN",
    "DE_RISKING_ONLY",
    "ETF_UNDERLYINGS",
    "EXEC_IDS",
    "EXECUTION_FULL_PERIOD_NOT_IDENTIFIABLE",
    "FAIL_ALPHA_MUTATION",
    "FINAL_REPORT",
    "FORBIDDEN_SEARCH",
    "FORWARD_OOS_NOT_AVAILABLE",
    "FULL_HISTORY_NOT_AVAILABLE",
    "HISTORICAL_DEVELOPMENT_ONLY",
    "IMPROVE_N_REQUIRED",
    "IV_DELTA_Q",
    "IV_MIN_CHAIN",
    "KEEP_BASELINE_V4",
    "MANIFEST_NAME",
    "MATCHED_P",
    "MINUTE_ROOT",
    "MINUTE_YEARS",
    "O01",
    "O02",
    "O02_HALVE",
    "O03",
    "O04",
    "O04_BETA_WINDOW",
    "OPTION_IDS",
    "OPTION_SECURITY_TYPE",
    "PARITY_ABS_NAV",
    "PARITY_METRIC_ABS",
    "PARENT_V4",
    "PCT_MIN_OBS",
    "PCT_WINDOW",
    "PERM_N",
    "PLAN_COPY",
    "PRE_REGISTERED_IDS",
    "PRE_REGISTERED_QUESTIONS",
    "PROMOTION_MAX_CAGR_DROP_PP",
    "PROMOTION_MDD_RETURN_SLACK_PP",
    "PROMOTION_MDD_SLACK_PP",
    "PROMOTION_SHARPE_SLACK",
    "PROTOCOL_SOURCE_FILES",
    "R12_01",
    "R12_01_DIR",
    "R12_01_TARGETS",
    "R13_00_DIR",
    "R14_00_DIR",
    "R15_00",
    "RANK_NAV",
    "RECOMMENDED_BASELINE_V5",
    "RISK_PRED",
    "SEAL_D_FEATURES",
    "SKIP_CODES",
    "SKEW_Q",
    "SNAPSHOT_FINGERPRINT",
    "SNAPSHOT_ID",
    "SPA_BOOT",
    "STAGES",
    "STRATEGY_SOURCE_FILES",
    "T01",
    "T01_MIN_N",
    "T02",
    "T02_MIN_N",
    "T02_MIN_OOS_R2",
    "TOP1_IDS",
    "TOP_K",
    "TRACKER_PATH",
    "UNQUANTIFIED",
    "X01",
    "X02",
    "X02_HM",
    "AUCTION_HM",
    "copy_plan",
    "control_dir",
    "experiment_dir",
    "overlayed_path",
    "parity_dir",
    "pre_registered_spec",
]
