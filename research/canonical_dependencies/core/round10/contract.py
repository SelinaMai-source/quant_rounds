"""baseline_v4 契约。真值只认 result/round9/R9_B03 artifact 与 round9 B03_CELL。"""

from __future__ import annotations

import json
from pathlib import Path

from ..paths import RESULT_ROOT, ROUND9_ROOT, ROUND10_ROOT
from ..round9.contract import (
    B01A,
    B03_CELL,
    BASELINE_V1,
    BASELINE_V2,
    BASELINE_V3,
    BLEND_PRED,
    CANDIDATE_CODE,
    DISAGREE_ALIAS,
    DISAGREE_LABEL,
    DISAGREE_MULT,
    FIVE_FEATURES,
    FIVE_HEAD_PRED,
    GAP_FLOOR,
    RANK_NAV,
    RISK_PRED,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
    SEEDS,
    SEP_LOW,
    SEP_MID,
    TEN_FEATURES,
    TEN_HEAD_PRED,
    TEN_PANEL,
    TEN_WEIGHT,
    TOP_K,
    V2_OVERLAY,
    XS_BOTTOM,
    XS_LOW_MULT,
)

PARENT_CODE = "R9_B03"
BASELINE_V4 = PARENT_CODE
R10_00 = "R10_00_baseline_v4"
STRUCTURAL_PARENT = "R8_E02"

ANCHOR_DIR = ROUND9_ROOT / PARENT_CODE

# 精确值来自 result/round9/R9_B03/summary.json，禁止四舍五入后再当目标。
TARGET_CAGR = 0.5373748002477328
TARGET_SHARPE = 1.8433564243143739
TARGET_MAXDD = -0.24751010811952434
TARGET_N_TRADES = 1149
TARGET_TOTAL_RETURN = 38.63211154586613
TARGET_LOYO_MED = 0.47336040019298853
TARGET_LOYO_WORST = 0.3354717292490337
TARGET_AVG_EXP = 0.1371103940963492
PARITY_ABS_NAV = 1e-4

B03_OVERLAY = {"kind": "preregistered", "allocator": "B03_conservative"}
TOP1_LOCK = True
E02_MULTIPLIED = False

# 九格：A→0/5/10；B→0/0/5；C→0/0/0。LOW/MID/HIGH = p≤20 / 20–50 / >50。replacement，不再乘 E02。
B03_CELL_ROWS = (
    {"agree": "A", "sep": "LOW", "nav": 0.00},
    {"agree": "A", "sep": "MID", "nav": 0.05},
    {"agree": "A", "sep": "HIGH", "nav": 0.10},
    {"agree": "B", "sep": "LOW", "nav": 0.00},
    {"agree": "B", "sep": "MID", "nav": 0.00},
    {"agree": "B", "sep": "HIGH", "nav": 0.05},
    {"agree": "C", "sep": "LOW", "nav": 0.00},
    {"agree": "C", "sep": "MID", "nav": 0.00},
    {"agree": "C", "sep": "HIGH", "nav": 0.00},
)

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
MANIFEST_NAME = "round10_manifest_v1.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round10_live_robustness_c168cce8.plan.md")
PLAN_COPY = ROUND10_ROOT / "ROUND10_PLAN.md"
PARITY_MD = ROUND10_ROOT / "00_baseline_v4_parity.md"

BOOTSTRAP_P_DELTA_CAGR = 0.6045

V5_GATES = {
    "cagr_gt": TARGET_CAGR,
    "sharpe_gt": TARGET_SHARPE,
    "maxdd_ge": -0.2575,
    "loyo_med_ge": TARGET_LOYO_MED,
    "loyo_worst_ge": TARGET_LOYO_WORST,
    "noise_pct": 0.0025,
    "prefer_p_delta_cagr": 0.70,
    "auto_promote_v5": False,
}

WF_PREDICT_YEARS = tuple(range(2018, 2027))
SHRINK_K = 50
PERTURB_SEEDS = (17, 42, 73, 101, 202)
PERTURB_SCALES = (0.005, 0.01)
ADVERSE_BPS = (0, 10, 20, 30, 50, 75, 100)
BASE_CASH = 300_000.0
CAPITAL_LEVELS = (300_000.0, 1_000_000.0, 3_000_000.0, 10_000_000.0)
PARTICIPATION = (0.01, 0.03, 0.05)
COMPLEXITY_CAGR = 0.0050
COMPLEXITY_SHARPE = 0.03
STABILITY_CAGR_BAND = 0.005
SEP_NEIGHBOR = {
    "center": (0.20, 0.50),
    "low": (0.15, 0.45),
    "high": (0.25, 0.55),
}
# A01：只用 agreement。A 取慷慨映射 10%，B 取 5%，C 为 0。
A01_AGREE_NAV = {"A": 0.10, "B": 0.05, "C": 0.00}
# A02：只用 separation。LOW/MID/HIGH = 0/5/10。
A02_SEP_NAV = {"LOW": 0.00, "MID": 0.05, "HIGH": 0.10}
# A03：0% 角改 5%；A04：10% 高置信改 5%。
A03_CELL = {k: (0.05 if float(v) == 0.0 else float(v)) for k, v in B03_CELL.items()}
A04_CELL = {k: (0.05 if float(v) == 0.10 else float(v)) for k, v in B03_CELL.items()}
INDUSTRY_SNAPSHOT = Path(
    "/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features/industry_membership"
)
HERMES_DAILY_RAW = Path(
    "/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features/daily_raw"
)
HERMES_DAILY_ADJ = Path(
    "/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features/daily_post_adjusted"
)
HERMES_VALUATION = Path(
    "/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features/daily_valuation"
)


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round10_tree

    ensure_round10_tree()
    assert_project_result_path(PLAN_COPY)
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    if not text.startswith("<!-- 本文件是开工计划"):
        PLAN_COPY.write_text(header + text, encoding="utf-8")
    else:
        PLAN_COPY.write_text(text, encoding="utf-8")
    return PLAN_COPY


def inspect_anchor() -> dict:
    """从 R9_B03 artifact 读真值，不靠 prompt 回忆。"""
    import pandas as pd

    report = json.loads((ANCHOR_DIR / "report.json").read_text(encoding="utf-8"))
    config = json.loads((ANCHOR_DIR / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((ANCHOR_DIR / "summary.json").read_text(encoding="utf-8"))
    trades = pd.read_parquet(ANCHOR_DIR / "trades.parquet")
    trades["signal_date"] = pd.to_datetime(trades["signal_date"]).dt.normalize()
    ranks = trades.groupby("signal_date")["score"].rank(method="first", ascending=False)
    day_nunique = trades.groupby("signal_date")["allocation_multiplier"].nunique()
    unique_mult = sorted(float(x) for x in trades["allocation_multiplier"].dropna().unique())
    metrics = report.get("metrics") or {}
    overlay = report.get("overlay") or {}
    overlay_spec = config.get("overlay_spec") or {}
    drift = {
        "cagr": abs(float(metrics.get("annualized_return", 0)) - TARGET_CAGR),
        "sharpe": abs(float(metrics.get("sharpe_zero_rate", 0)) - TARGET_SHARPE),
        "maxdd": abs(float(metrics.get("max_drawdown", 0)) - TARGET_MAXDD),
        "trades": abs(int(metrics.get("completed_trades", -1)) - TARGET_N_TRADES),
        "total_return": abs(float(metrics.get("total_return", 0)) - TARGET_TOTAL_RETURN),
        "loyo_median": abs(float(metrics.get("loyo_median", 0)) - TARGET_LOYO_MED),
        "loyo_worst": abs(float(metrics.get("loyo_worst", 0)) - TARGET_LOYO_WORST),
    }
    return {
        "report": report,
        "config": config,
        "summary": summary,
        "n_trades": int(len(trades)),
        "n_rank1": int((ranks == 1).sum()),
        "n_rank2": int((ranks == 2).sum()),
        "unique_multipliers": unique_mult,
        "days_with_split_mult": int((day_nunique > 1).sum()),
        "n_signal_days_in_trades": int(day_nunique.shape[0]),
        "overlay_spec": overlay_spec,
        "allocator": overlay_spec.get("allocator") or overlay.get("allocator"),
        "e02_multiplied": overlay.get("e02_multiplied"),
        "top1_lock": overlay.get("top1_lock"),
        "kind": overlay_spec.get("kind"),
        "report_overlay": overlay,
        "top_k": report.get("top_k"),
        "fracs": report.get("rank_name_budget_nav_fractions"),
        "exit_mode": (report.get("config") or {}).get("exit_mode"),
        "snapshot_id": report.get("snapshot_id"),
        "snapshot_fingerprint": report.get("snapshot_fingerprint"),
        "loyo_median": metrics.get("loyo_median"),
        "loyo_worst": metrics.get("loyo_worst"),
        "anchor_drift": drift,
        "anchor_matches_hardcoded": all(v < 1e-12 for v in drift.values()),
        "b03_cell": dict(B03_CELL),
    }


__all__ = [
    "ANCHOR_DIR",
    "B01A",
    "B03_CELL",
    "B03_CELL_ROWS",
    "B03_OVERLAY",
    "BASELINE_V1",
    "BASELINE_V2",
    "BASELINE_V3",
    "BASELINE_V4",
    "BLEND_PRED",
    "BOOTSTRAP_P_DELTA_CAGR",
    "CANDIDATE_CODE",
    "CURSOR_PLAN",
    "DISAGREE_ALIAS",
    "DISAGREE_LABEL",
    "DISAGREE_MULT",
    "E02_MULTIPLIED",
    "FIVE_FEATURES",
    "FIVE_HEAD_PRED",
    "GAP_FLOOR",
    "MANIFEST_NAME",
    "PARENT_CODE",
    "PARITY_ABS_NAV",
    "PARITY_MD",
    "PLAN_COPY",
    "R10_00",
    "RANK_NAV",
    "RISK_PRED",
    "ROLLING_MIN_PERIODS",
    "ROLLING_WINDOW",
    "SEEDS",
    "SEP_LOW",
    "SEP_MID",
    "STRUCTURAL_PARENT",
    "TARGET_AVG_EXP",
    "TARGET_CAGR",
    "TARGET_LOYO_MED",
    "TARGET_LOYO_WORST",
    "TARGET_MAXDD",
    "TARGET_N_TRADES",
    "TARGET_SHARPE",
    "TARGET_TOTAL_RETURN",
    "TEN_FEATURES",
    "TEN_HEAD_PRED",
    "TEN_PANEL",
    "TEN_WEIGHT",
    "TOP1_LOCK",
    "TOP_K",
    "TRACKER_PATH",
    "V2_OVERLAY",
    "V5_GATES",
    "WF_PREDICT_YEARS",
    "XS_BOTTOM",
    "XS_LOW_MULT",
    "copy_plan",
    "inspect_anchor",
]
