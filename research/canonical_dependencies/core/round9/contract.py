"""baseline_v3 契约。真值只认 result/round8/R8_E02 artifact 与 round8 overlays/geometry。"""

from __future__ import annotations

import json
from pathlib import Path

from ..paths import RESULT_ROOT, ROUND8_ROOT, ROUND9_ROOT
from ..round8.contract import (
    B01A,
    BASELINE_V1,
    BASELINE_V2,
    BLEND_PRED,
    CANDIDATE_CODE,
    FIVE_FEATURES,
    FIVE_HEAD_PRED,
    GAP_FLOOR,
    RANK_NAV,
    RISK_PRED,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
    SEEDS,
    TEN_FEATURES,
    TEN_HEAD_PRED,
    TEN_PANEL,
    TEN_WEIGHT,
    TOP_K,
    V2_OVERLAY,
    XS_BOTTOM,
    XS_LOW_MULT,
)

PARENT_CODE = "R8_E02"
BASELINE_V3 = PARENT_CODE
R9_00 = "R9_00_baseline_v3"

ANCHOR_DIR = ROUND8_ROOT / PARENT_CODE

# 精确值来自 result/round8/R8_E02/report.json，禁止四舍五入后再当目标。
TARGET_CAGR = 0.5299285369905697
TARGET_SHARPE = 1.7964712763996256
TARGET_MAXDD = -0.24504544808039253
TARGET_N_TRADES = 1783
TARGET_TOTAL_RETURN = 37.01949791961103
TARGET_LOYO_MED = 0.4617879099240748
TARGET_LOYO_WORST = 0.3304383573245333
PARITY_ABS_NAV = 1e-4

E02_OVERLAY = {"kind": "custom", "e02": True}
DISAGREE_MULT = {"A": 1.00, "B": 0.75, "C": 0.50}
DISAGREE_ALIAS = {"HIGH": "A", "MEDIUM": "B", "LOW": "C"}
DISAGREE_LABEL = {"A": "HIGH", "B": "MEDIUM", "C": "LOW"}
TOP1_LOCK = True  # E02 只乘 Top2；Top1 必须等于 E02

# 项目 canonical Meta XGB。禁止 subsample/colsample < 1，禁止调参。
META_XGB = {
    "max_depth": 2,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "n_estimators": 50,
    "learning_rate": 0.05,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "random_state": 17,
    "n_jobs": 2,
}

TOP2_NAV = 0.10
TOP2_LADDER = (0.0, 0.05, 0.10)
SEP_LOW = 0.20
SEP_MID = 0.50
# 预注册 9 格：agreement A/B/C × separation LOW/MID/HIGH → Top2 NAV 0/5/10。不是再乘 E02。
B01_CELL = {
    ("A", "LOW"): 0.05, ("A", "MID"): 0.10, ("A", "HIGH"): 0.10,
    ("B", "LOW"): 0.00, ("B", "MID"): 0.05, ("B", "HIGH"): 0.10,
    ("C", "LOW"): 0.00, ("C", "MID"): 0.00, ("C", "HIGH"): 0.05,
}
B03_CELL = {
    ("A", "LOW"): 0.00, ("A", "MID"): 0.05, ("A", "HIGH"): 0.10,
    ("B", "LOW"): 0.00, ("B", "MID"): 0.00, ("B", "HIGH"): 0.05,
    ("C", "LOW"): 0.00, ("C", "MID"): 0.00, ("C", "HIGH"): 0.00,
}
AGREE_SCORE = {"A": 1.0, "B": 0.5, "C": 0.0}
SEP_SCORE = {"LOW": 0.0, "MID": 0.5, "HIGH": 1.0}

# 恰好 7 个：3 个 PIT + agreement + corr + count + top2 gap。
META_FEATURES = (
    "top1_minus_median_p250",
    "top2_minus_median_p250",
    "top1_minus_top2_p250",
    "agreement_score",
    "rank_corr",
    "candidate_count",
    "top2_d1_open_gap",
)
WF_PREDICT_YEARS = tuple(range(2018, 2027))

SEAL_FAMILY = ("fb_limit_close_minute_ratio", "h5_close_limit_retention", "fb_touch_count")
FLOW_CANDIDATES = (
    "fb_close_net_to_turnover",
    "h6_large_net_to_turnover",
    "h6_main_net_to_turnover",
    "h6_elg_net_to_turnover",
    "h6_retail_net_to_turnover",
)
INTRA_CANDIDATES = (
    "h6_close_amount_ratio",
    "h6_last30_amount_ratio",
    "h6_last15_amount_ratio",
    "fb_open_amount_ratio",
)
LOW_ACTIVITY_ZERO_SHARE = 0.80
COMPLEXITY_CAGR = 0.0050
COMPLEXITY_SHARPE = 0.03

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
MANIFEST_NAME = "round9_manifest_v1.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round9_top2_allocator_144121e6.plan.md")
PLAN_COPY = ROUND9_ROOT / "ROUND9_PLAN.md"
PARITY_MD = ROUND9_ROOT / "00_baseline_v3_parity.md"

V4_GATES = {
    "cagr_gt": TARGET_CAGR,
    "sharpe_gt": TARGET_SHARPE,
    "maxdd_ge": -0.2550,
    "loyo_med_ge": TARGET_LOYO_MED,
    "loyo_worst_ge": TARGET_LOYO_WORST,
    "noise_pct": 0.0025,
    "auto_promote_v4": False,
}


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round9_tree

    ensure_round9_tree()
    assert_project_result_path(PLAN_COPY)
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    if not text.startswith("<!-- 本文件是开工计划"):
        PLAN_COPY.write_text(header + text, encoding="utf-8")
    else:
        PLAN_COPY.write_text(text, encoding="utf-8")
    return PLAN_COPY


def inspect_anchor() -> dict:
    """从 R8_E02 artifact 读真值，不靠 prompt 回忆。"""
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
        "e02_flag": bool(overlay_spec.get("e02")),
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
    }


__all__ = [
    "ANCHOR_DIR",
    "B01A",
    "BASELINE_V1",
    "BASELINE_V2",
    "BASELINE_V3",
    "BLEND_PRED",
    "CANDIDATE_CODE",
    "CURSOR_PLAN",
    "DISAGREE_ALIAS",
    "DISAGREE_LABEL",
    "DISAGREE_MULT",
    "E02_OVERLAY",
    "FIVE_FEATURES",
    "FIVE_HEAD_PRED",
    "GAP_FLOOR",
    "AGREE_SCORE",
    "B01_CELL",
    "B03_CELL",
    "COMPLEXITY_CAGR",
    "COMPLEXITY_SHARPE",
    "FLOW_CANDIDATES",
    "INTRA_CANDIDATES",
    "LOW_ACTIVITY_ZERO_SHARE",
    "MANIFEST_NAME",
    "META_FEATURES",
    "META_XGB",
    "PARENT_CODE",
    "PARITY_ABS_NAV",
    "PARITY_MD",
    "PLAN_COPY",
    "R9_00",
    "RANK_NAV",
    "RISK_PRED",
    "ROLLING_MIN_PERIODS",
    "ROLLING_WINDOW",
    "SEAL_FAMILY",
    "SEEDS",
    "SEP_LOW",
    "SEP_MID",
    "SEP_SCORE",
    "TARGET_CAGR",
    "TARGET_LOYO_MED",
    "TARGET_LOYO_WORST",
    "TARGET_MAXDD",
    "TARGET_N_TRADES",
    "TARGET_SHARPE",
    "TARGET_TOTAL_RETURN",
    "TEN_FEATURES",
    "TOP2_LADDER",
    "TOP2_NAV",
    "WF_PREDICT_YEARS",
    "TEN_HEAD_PRED",
    "TEN_PANEL",
    "TEN_WEIGHT",
    "TOP1_LOCK",
    "TOP_K",
    "TRACKER_PATH",
    "V2_OVERLAY",
    "V4_GATES",
    "XS_BOTTOM",
    "XS_LOW_MULT",
    "copy_plan",
    "inspect_anchor",
]
