"""Round11 风险 intake。先写表、再冻结 manifest。禁止先看新策略绩效。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND11_ROOT, ROUND12_ROOT
from .common import dump_json, write_md
from .contract import copy_plan
from .freeze import freeze_manifest, require_frozen

INTAKE_MD = ROUND12_ROOT / "00_round11_risk_intake.md"


def run_intake(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    copy_plan()
    rows = [
        {
            "id": "R1",
            "item": "selection / PBO",
            "status": "PARTIALLY_FIXABLE",
            "note": (
                "Round11 TRUE PBO n=80 是 `_load_daily_matrix(max_cols=80)` 截断；"
                "1311 条日 NAV 已在 daily_matrix_index.json。本轮抬升矩阵 + 可比宇宙。"
                "暗试仍不可识别。"
            ),
        },
        {
            "id": "R2",
            "item": "forward OOS",
            "status": "NOT_FIXABLE_WITH_HISTORICAL_DATA",
            "note": "快照止于 2026-07-24。CPCV/bootstrap 不得冒充 OOS。默认 FORWARD_OOS_NOT_AVAILABLE。",
        },
        {
            "id": "R3",
            "item": "execution",
            "status": "PARTIALLY_FIXABLE",
            "note": (
                "2018–2021 分钟可校准；无 09:25 竞价；2022–2026 只能情景外推。"
                "d1_open_gap + 同一官方 open 成交 = SAME_OPEN_EXECUTION_OPTIMISM。"
            ),
        },
        {
            "id": "R4",
            "item": "survivorship",
            "status": "PARTIALLY_FIXABLE",
            "note": "可建 PIT（daily_raw + instruments + special_treatment known_at）。缺独立退市公告 PIT 则不得标 PASS。",
        },
        {
            "id": "R5",
            "item": "cost",
            "status": "DOCUMENTED",
            "note": "R12_01 固化 R11_V4_CORRECTED。佣金/过户仍 FEE_HISTORY_UNVERIFIED。",
        },
        {
            "id": "R6",
            "item": "DSR 口径",
            "status": "FIXABLE",
            "note": "锁日收益→日 Sharpe→日样本数。禁止年化 Sharpe 配日 N。",
        },
        {
            "id": "R7",
            "item": "AlphaForge",
            "status": "PARTIALLY_FIXABLE",
            "note": "只对 https://github.com/xiejinglover/alpha-forge 隔离重试一次。失败不阻断。",
        },
    ]
    r11 = {
        "parity": "PASS",
        "corrected_v4": {
            "cagr": 0.5253672918108749,
            "sharpe": 1.8120914675763191,
            "max_drawdown": -0.25455988069271596,
        },
        "legacy_v4": {
            "cagr": 0.5373748002477328,
            "sharpe": 1.8433564243143739,
            "max_drawdown": -0.24751010811952434,
        },
        "true_pbo_truncated_n": 80,
        "daily_matrix_n": 1311,
        "n_trials": {"documented": 1315, "discovered": 1339, "upper": 2678},
        "dsr_daily_discovered": 0.9804213517613413,
        "placebo_method": "PLACEBO_DATE_WEIGHTED_ISOLATED_FILL",
        "placebo_percentile": 100.0,
        "survivorship": "UNQUANTIFIED",
        "alphaforge": "ALPHAFORGE_REPO_ACCESS_FAIL",
        "verdict": "HISTORICAL_DEVELOPMENT_CONTINUE",
        "sources": [
            str(ROUND11_ROOT / "ROUND11_ROBUSTNESS_OVERFIT_AUDIT.md"),
            str(ROUND11_ROOT / "13_risk_register.md"),
        ],
    }
    lines = [
        "# Round 12 · Round11 风险 intake",
        "",
        "本表在 **冻结 manifest 之前** 写完。不读、不跑新策略绩效。",
        "Round 12 = Risk Remediation & Deployability，**不是** Alpha 搜索。",
        "默认 `KEEP_BASELINE_V4`，不得自动升 `baseline_v5`。",
        "",
        "## 风险对照",
        "",
        "| ID | 项 | 状态 | 说明 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['id']} | {row['item']} | `{row['status']}` | {row['note']} |")
    lines += [
        "",
        "## 已锁定、本轮不再重搜的事实",
        "",
        f"- legacy v4 `R9_B03` CAGR=`{r11['legacy_v4']['cagr']}` Sharpe=`{r11['legacy_v4']['sharpe']}` MaxDD=`{r11['legacy_v4']['max_drawdown']}`",
        f"- `R11_V4_CORRECTED` CAGR=`{r11['corrected_v4']['cagr']}` Sharpe=`{r11['corrected_v4']['sharpe']}` MaxDD=`{r11['corrected_v4']['max_drawdown']}`",
        f"- Round11 TRUE PBO 被 `max_cols=80` 截断；可对齐日收益 n=`{r11['daily_matrix_n']}`",
        f"- trial documented/discovered/upper = {r11['n_trials']['documented']}/{r11['n_trials']['discovered']}/{r11['n_trials']['upper']}",
        f"- DSR(full discovered, 日频同频) = `{r11['dsr_daily_discovered']}`",
        f"- Round11 placebo 方法 = `{r11['placebo_method']}`（本轮升级全引擎）",
        "- 全市场 1 分钟仅 2017–2021 全量；首 bar=09:30；无 09:25；2022+ 几乎不可用",
        "- PIT 材料：Hermes v7 daily_raw + instruments + special_treatment；复用 `first_board_hermes.universe` 的 known_at 规则",
        "- **禁止** 用当前证券列表或 `universe_membership.effective_to` 当历史宇宙",
        "",
        "## 本轮明确不做",
        "",
        "- 不发明额外策略变体、不搜 cap、不因 R12_03 好看再发明 25bp",
        "- 不把 CPCV / bootstrap 写成 forward OOS",
        "- 不把可恢复子集 PBO 外推到 1339/2678",
        "- 追踪表不写「写入时间」",
        "",
    ]
    write_md(INTAKE_MD, "\n".join(lines))
    frozen = freeze_manifest(overwrite=False)
    payload = {
        "status": "DONE",
        "rows": rows,
        "round11": r11,
        "manifest": frozen,
        "intake_md": str(INTAKE_MD),
        "note": "intake+freeze 完成后再允许 parity。新策略成绩必须在 freeze 之后。",
    }
    dump_json(ROUND12_ROOT / "intake.json", payload)
    require_frozen()
    return payload


__all__ = ["run_intake"]
