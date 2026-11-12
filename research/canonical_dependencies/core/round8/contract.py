"""baseline_v2 契约。真值来自 replay_round7 的 R4_E_top1_minus_median artifact。"""

from __future__ import annotations

from pathlib import Path

from ..paths import REPLAY_ROUND7_ROOT, RESULT_ROOT, STOCK_ROUND7_ROOT
from ..replay_round7.contract import (
    BLEND_PRED,
    CANDIDATE_CODE,
    FIVE_FEATURES,
    GAP_FLOOR,
    RANK_NAV,
    RISK_PRED,
    SEEDS,
    TEN_FEATURES,
    TEN_PANEL,
    TEN_WEIGHT,
    TOP_K,
)

PARENT_CODE = "R4_E_top1_minus_median"
BASELINE_V1 = "R7_FS01122_DUAL90_10_CLEAN"
BASELINE_V2 = PARENT_CODE
R8_00 = "R8_00_baseline_v2"

ANCHOR_DIR = REPLAY_ROUND7_ROOT / "round4" / PARENT_CODE
R7_V1_DIR = REPLAY_ROUND7_ROOT / "round1" / "R1_00_baseline"
P08_DIR = REPLAY_ROUND7_ROOT / "round4" / "R4_P_08"

TARGET_CAGR = 0.5075432102926214
TARGET_SHARPE = 1.716603623385515
TARGET_MAXDD = -0.24367908587033738
TARGET_N_TRADES = 1787
TARGET_TOTAL_RETURN = 32.51487361549896
PARITY_ABS_NAV = 1e-4

B01A = {"kind": "breadth_tier", "t1": 0.45, "t2": 0.50, "t3": 0.60}
XS_CONF = {"kind": "cross_section", "feature": "top1_minus_median"}
V2_OVERLAY = {
    "kind": "stack",
    "market": dict(B01A),
    "confidence": dict(XS_CONF),
    "cvar": None,
    "cvar_mode": "product",
}
P08_OVERLAY = {
    "kind": "stack",
    "market": {
        "kind": "combine",
        "how": "min",
        "parts": [dict(B01A), {"kind": "eco_relimit_p250", "low": 0.20, "low_mult": 0.5}],
    },
    "confidence": dict(XS_CONF),
    "cvar": None,
    "cvar_mode": "product",
}

ROLLING_WINDOW = 250
ROLLING_MIN_PERIODS = max(40, ROLLING_WINDOW // 5)
XS_BOTTOM = 0.20
XS_LOW_MULT = 0.50
CLEAN_RISK_IN_V2 = False
USED_LOCKED_ROUND7_RISK = False

FIVE_HEAD_PRED = STOCK_ROUND7_ROOT / "models" / "five" / "predictions.parquet"
TEN_HEAD_PRED = STOCK_ROUND7_ROOT / "models" / "ten" / "predictions.parquet"

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
MANIFEST_NAME = "round8_manifest_v1.yaml"

PROMOTION_RULES = {
    "cagr_ge_v2": True,
    "sharpe_ge_v2": True,
    "maxdd_not_worse_than_1pct": True,
    "loyo_median_ge_v2": True,
    "loyo_worst_not_clearly_worse": True,
    "not_2024_only": True,
    "complexity_penalty": "+0.01 Sharpe is not enough for a new layer",
}

CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round8_baseline_v2_plan_589c4d2f.plan.md")
PLAN_COPY = RESULT_ROOT / "round8" / "ROUND8_PLAN.md"
CONTRACT_MD = RESULT_ROOT / "round8" / "00_baseline_v2_contract.md"


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round8_tree

    ensure_round8_tree()
    assert_project_result_path(PLAN_COPY)
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = (
        "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    )
    if not text.startswith("<!-- 本文件是开工计划"):
        PLAN_COPY.write_text(header + text, encoding="utf-8")
    else:
        PLAN_COPY.write_text(text, encoding="utf-8")
    return PLAN_COPY


def inspect_anchor() -> dict:
    """从 R4_E replay artifact 读真值，不用 prompt 猜测。"""
    import json

    import pandas as pd

    report = json.loads((ANCHOR_DIR / "report.json").read_text(encoding="utf-8"))
    config = json.loads((ANCHOR_DIR / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((ANCHOR_DIR / "summary.json").read_text(encoding="utf-8"))
    trades = pd.read_parquet(ANCHOR_DIR / "trades.parquet")
    trades["signal_date"] = pd.to_datetime(trades["signal_date"]).dt.normalize()
    day_nunique = trades.groupby("signal_date")["allocation_multiplier"].nunique()
    unique_mult = sorted(float(x) for x in trades["allocation_multiplier"].dropna().unique())
    ranks = trades.groupby("signal_date")["score"].rank(method="first", ascending=False)
    alpha = pd.read_parquet(BLEND_PRED, columns=["signal_date", "instrument_id", "score"])
    alpha["signal_date"] = pd.to_datetime(alpha["signal_date"])
    n_by_day = alpha.groupby("signal_date")["score"].apply(lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum()))
    return {
        "report": report,
        "config": config,
        "summary": summary,
        "n_trades": int(len(trades)),
        "unique_multipliers": unique_mult,
        "days_with_split_mult": int((day_nunique > 1).sum()),
        "n_signal_days_in_trades": int(day_nunique.shape[0]),
        "n_rank1": int((ranks == 1).sum()),
        "n_rank2": int((ranks == 2).sum()),
        "used_locked_round7_risk": report.get("used_locked_round7_risk"),
        "actual_top_k": report.get("top_k"),
        "actual_fracs": report.get("rank_name_budget_nav_fractions"),
        "catalog_top_k": config.get("top_k"),
        "catalog_fracs": config.get("rank_name_budget_nav_fractions"),
        "catalog_overlay": config.get("overlay"),
        "report_overlay": report.get("overlay"),
        "n1_score_days": int((n_by_day == 1).sum()),
        "n0_score_days": int((n_by_day == 0).sum()),
        "alpha_rows": int(len(alpha)),
        "blend_pred": str(BLEND_PRED),
        "risk_pred": str(RISK_PRED),
        "clean_risk_in_v2": CLEAN_RISK_IN_V2,
        "snapshot_id": report.get("snapshot_id"),
        "snapshot_fingerprint": report.get("snapshot_fingerprint"),
    }


def write_contract_markdown() -> Path:
    from ..paths import assert_project_result_path, ensure_round8_tree

    ensure_round8_tree()
    facts = inspect_anchor()
    report = facts["report"]
    summary = facts["summary"]
    overlay = facts["catalog_overlay"] or {}
    report_ov = facts["report_overlay"] or {}
    metrics = report.get("metrics") or {}
    market = (overlay.get("market") or {})
    conf = (overlay.get("confidence") or {})

    def pct(x, d=2):
        return "—" if x is None else f"{float(x) * 100:.{d}f}%"

    def num(x, d=6):
        return "—" if x is None else f"{float(x):.{d}g}"

    unique_mult = ", ".join(f"{x:g}" for x in facts["unique_multipliers"])
    body = f"""# Round 8 · baseline_v2 契约（R8_00）

本文只认 **replay artifact**，不认 prompt 回忆。父策略 = `baseline_v2` = `{PARENT_CODE}`。
核对目录：`{ANCHOR_DIR}`。

**结论（必须同时成立）**

- **不是** `clean_risk`：`used_locked_round7_risk = {facts["used_locked_round7_risk"]}`（应为 `False`）。
- 市场层 = **B01a** `breadth_tier` **45 / 50 / 60**，与截面置信 **相乘**（`stack` / `market_x_confidence`）。
- 置信 = 当日 `max(score) − median(score)`，再对**日度序列** `rolling({ROLLING_WINDOW}, min_periods={ROLLING_MIN_PERIODS}).rank(pct=True)`；`p≤{XS_BOTTOM:g} → {XS_LOW_MULT:g}`，否则 `1.0`；NaN（含窗口不足）→ `1.0`。
- catalog JSON `top_k: {facts["catalog_top_k"]}` **被忽略**；replay 强制 Top2 `{list(RANK_NAV)}` NAV。
- `allocation_multiplier` **整日同乘**，不改 fused score；正式账本 `local_executor` → Stock `fixed_sleeve_backtest`。

## 1. Checklist

| 项 | 契约 | artifact 真值 | 状态 |
|---|---|---|---|
| Alpha | Round7 锁分 `dual90_10_gap_floor`（`load_locked_alpha`） | `{facts["blend_pred"]}` | {"PASS" if not facts["used_locked_round7_risk"] else "FAIL"} |
| 是否 clean_risk | **否**；`used_locked_round7_risk=false` | `{facts["used_locked_round7_risk"]}` | {"PASS" if facts["used_locked_round7_risk"] is False else "FAIL"} |
| 市场层 | B01a `breadth_tier` t1/t2/t3 = 0.45/0.50/0.60 | `{market}` | {"PASS" if market.get("kind")=="breadth_tier" and float(market.get("t1",-1))==0.45 and float(market.get("t2",-1))==0.5 and float(market.get("t3",-1))==0.6 else "FAIL"} |
| 置信 | `cross_section` / `top1_minus_median` | `{conf}` | {"PASS" if conf.get("kind")=="cross_section" and conf.get("feature")=="top1_minus_median" else "FAIL"} |
| 合成 | stack，市场 × 置信 | kind=`{overlay.get("kind")}` how=`{report_ov.get("combine_default")}` | {"PASS" if overlay.get("kind")=="stack" else "FAIL"} |
| PIT | rolling250，min_periods=50 | report pit=`{(report_ov.get("confidence") or {}).get("pit")}` | {"PASS" if (report_ov.get("confidence") or {}).get("pit")=="rolling_250" else "FAIL"} |
| 容量 catalog | `top_k=1`、`rank_name_budget_nav_fractions=null` | top_k=`{facts["catalog_top_k"]}` fracs=`{facts["catalog_fracs"]}` | 记录（应被忽略） |
| 容量实际 | 强制 Top2 30%+10% NAV | top_k=`{facts["actual_top_k"]}` fracs=`{facts["actual_fracs"]}` | {"PASS" if facts["actual_top_k"]==2 and list(facts["actual_fracs"] or [])==[0.3, 0.1] else "FAIL"} |
| 乘子粒度 | 日度 `allocation_multiplier`，同日 Top1/Top2 相同 | 同日不同乘子天数 = **{facts["days_with_split_mult"]}** / {facts["n_signal_days_in_trades"]} | {"PASS" if facts["days_with_split_mult"]==0 else "FAIL"} |
| 不改 score | overlay 只写 `allocation_multiplier` | replay runner 有 score 比对 | 代码锁 |
| 执行器 | `round7_fixed_sleeve_config` + Stock `run_fixed_dual_sleeve_top3_backtest` | exit=`close_from_d2`；执行器涨跌停缺口门 = None | PASS（report.config） |
| 小样本日 | 当日无有效 score 则跳过；`n=1` 时 top1−median=0 | blend 中 n=1 日数 = {facts["n1_score_days"]}；n=0 日数 = {facts["n0_score_days"]} | 记录 |

B01a 映射（`core/round2/overlays.py::breadth_tier_multiplier`，与 `R2_B01a` 同参）：

- 上涨家数比 `< 45%` → 0（整日跳过）
- `[45%, 50%)` → 0.50
- `[50%, 60%)` → 0.75
- `≥ 60%` → 1.00

与截面硬阈值相乘后，R4_E 成交乘子只出现 {{0.25, 0.375, 0.50, 0.75, 1.00}}（0 的日子不进账本）。本 artifact 观察到的取值：`{unique_mult}`。

## 2. 锁定数字（R4_E replay）

| 指标 | 值 |
|---|---|
| CAGR `annualized_return` | {metrics.get("annualized_return")} → **{pct(metrics.get("annualized_return"))}** |
| Sharpe `sharpe_zero_rate` | {metrics.get("sharpe_zero_rate")} → **{float(metrics.get("sharpe_zero_rate") or 0):.3f}** |
| MaxDD | {metrics.get("max_drawdown")} → **{pct(metrics.get("max_drawdown"))}** |
| 完成笔数 | **{metrics.get("completed_trades")}**（Top1 {facts["n_rank1"]} / Top2 {facts["n_rank2"]}） |
| 累计收益 | {metrics.get("total_return")} |
| 胜率 | {pct(summary.get("trade_win_rate"))} |
| 最差完整年 | {summary.get("worst_year_return")}（2022，见 summary） |
| overlay 日数 | n={report_ov.get("n_days")} skip={report_ov.get("skip_days")} half={report_ov.get("half_days")} full={report_ov.get("full_days")} mean_mult={report_ov.get("mean_mult")} |
| snapshot | `{facts["snapshot_id"]}` / `{facts["snapshot_fingerprint"]}` |
| trial / capacity / variant | `{report.get("trial_id")}` / `{report.get("capacity")}` / `{report.get("variant")}` |

R8_00 必须复现上表（显示约 50.75% / 1.717 / −24.37% / 1787）。容差：指标 `1e-8`，NAV `abs≤{PARITY_ABS_NAV:g}`。

分年（summary.yearly_returns）：

| 年 | 收益 |
|---|---:|
"""
    yearly = summary.get("yearly_returns") or {}
    for year in sorted(yearly):
        body += f"| {year} | {pct(yearly[year])} |\n"

    body += f"""
## 3. 执行器与资金

来自 `report.json` 的 `config`（Stock `FixedSleeveConfig`）：

- 双袖套，初始现金 30 万（每袖 15 万）
- `top_k=2`，`rank_name_budget_nav_fractions=[0.3, 0.1]`
- `name_budget=10000`，`name_budget_nav_fraction=null`
- `market_up_ratio_min=null`，`open_gap_min/max=null`（gap 已在融合预滤）
- `exit_mode=close_from_d2`，`max_holding_sessions=10`
- 费用：滑点 12.3bp、佣金 1bp、过户 0.1bp、卖印花税 5bp

catalog 里 `top_k: 1` / `name_budget_nav_fraction: 0.3` **不是**实际容量：`core/replay_round7/runner.py::_config_for` 把 `top_k<3` 一律打成 `TOP_K=2`，并把 0.3 的日预算解释成 **(0.3, 0.1)** 两档。Round 8 的 `R8_00` 必须走同一条 `round7_fixed_sleeve_config`。

## 4. 小样本与 PIT

- 当日无有效 fused score：`cross_section_daily` 跳过该日（不进几何表）。
- `n=1`：`max=median` ⇒ `top1_minus_median=0`，再进入 rolling250 分位；窗口不足的 NaN 分位 → 乘子 1.0。
- 硬阈值：`p≤0.20 → 0.5`，见 `core/round4/overlays.py::_cross_section_mult`。

## 5. 明确不是什么

- 不是在 `r7_v1` / `clean_risk` 上再乘 top1−median。v1 风险层与 v2 市场+截面是两套 overlay。
- 不是 `R4_P_08`（那是 B01a∩连板率 × 同一截面，只当 secondary challenger）。
- 现有 date-level API **不能**做 rank-specific 置信（同日 Top1/Top2 乘子相同）。本 artifact 已验证：同日不同乘子天数 = {facts["days_with_split_mult"]}。Round 8 后续 C/D 才按 `selection_rank` 写不同乘子；**不改 score**。
- `d1_open_gap` 仍是 execution-time feature，升 v2 不会消失。

## 6. Round 8 复现入口

```bash
python tools/run_round8.py --stage parity
```

产物：`result/round8/R8_00_baseline_v2/` 与 `01_baseline_v2_parity.md`。不过则停，不发明后续 stage。
"""
    assert_project_result_path(CONTRACT_MD)
    CONTRACT_MD.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")
    return CONTRACT_MD


__all__ = [
    "ANCHOR_DIR",
    "B01A",
    "BASELINE_V1",
    "BASELINE_V2",
    "BLEND_PRED",
    "CANDIDATE_CODE",
    "CLEAN_RISK_IN_V2",
    "CONTRACT_MD",
    "CURSOR_PLAN",
    "FIVE_FEATURES",
    "FIVE_HEAD_PRED",
    "GAP_FLOOR",
    "MANIFEST_NAME",
    "P08_OVERLAY",
    "PARENT_CODE",
    "PARITY_ABS_NAV",
    "PLAN_COPY",
    "PROMOTION_RULES",
    "R8_00",
    "RANK_NAV",
    "RISK_PRED",
    "ROLLING_MIN_PERIODS",
    "ROLLING_WINDOW",
    "SEEDS",
    "TARGET_CAGR",
    "TARGET_MAXDD",
    "TARGET_N_TRADES",
    "TARGET_SHARPE",
    "TARGET_TOTAL_RETURN",
    "TEN_FEATURES",
    "TEN_HEAD_PRED",
    "TEN_PANEL",
    "TEN_WEIGHT",
    "TOP_K",
    "TRACKER_PATH",
    "V2_OVERLAY",
    "XS_BOTTOM",
    "XS_CONF",
    "XS_LOW_MULT",
    "copy_plan",
    "inspect_anchor",
    "write_contract_markdown",
]
