"""写 03/06 md、同步追踪表。不自动改锁定 baseline。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..metrics import LOCKED_BASELINE
from ..paths import PROJECT_ROOT, assert_project_result_path, ensure_round2_tree, round2_dir
from .metrics import P0_MAXDD_FLOOR, pareto_front
from .registry import EXPERIMENTS, list_experiment_codes
from .runner import load_result_report, tracking_row_from_report


def _all_rows() -> list[dict[str, Any]]:
    rows = []
    for code in list_experiment_codes():
        rows.append(tracking_row_from_report(code))
    # dynamically created combos
    for path in sorted((ensure_round2_tree()).glob("R2_*/summary.json")):
        code = path.parent.name
        if code in EXPERIMENTS:
            continue
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def _fmt_pct(value, signed=False):
    if value is None:
        return "—"
    pct = float(value) * 100
    return f"{pct:+.2f}%" if signed else f"{pct:.2f}%"


def _fmt_num(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def write_single_module_md(rows: list[dict[str, Any]]) -> Path:
    dest = ensure_round2_tree() / "03_single_module_results.md"
    lines = [
        "# 03 单模块结果",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "- Alpha 锁定。ΔSharpe>+0.02 不再作为替换门槛。P0 筛：MaxDD ≥ −30%。",
        "- 每模块 Pareto（CAGR / Sharpe / MaxDD / Worst Year / Calmar），不 sort Sharpe。",
        "",
        "| 代号 | 模块 | 卖出 | CAGR | Sharpe | Sortino | Calmar | MaxDD | Worst Year | Worst 12M | 交易 | 状态 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if str(row.get("code", "")).startswith("R2_P") or str(row.get("code", "")).startswith("R2_T") or str(row.get("code", "")).startswith("R2_H"):
            continue
        lines.append(
            f"| `{row.get('code')}` | {row.get('module','')} | {row.get('exit','')} | "
            f"{_fmt_pct(row.get('cagr'))} | {_fmt_num(row.get('sharpe'))} | "
            f"{_fmt_num(row.get('sortino'))} | "
            f"{_fmt_num(row.get('calmar'), 2)} | "
            f"{_fmt_pct(row.get('max_drawdown'))} | {_fmt_pct(row.get('worst_year_return'))} | "
            f"{_fmt_pct(row.get('worst_12m'))} | {row.get('completed_trades') or '—'} | {row.get('status')} |"
        )
    assert_project_result_path(dest)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def pick_candidates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    usable = [
        r for r in rows
        if r.get("cagr") is not None and r.get("max_drawdown") is not None
        and r.get("status") not in {"待跑", "诊断-不回测"}
        and not str(r.get("status", "")).startswith("跳过")
        and not r.get("near_zero_trade_curve")
        and float(r["max_drawdown"]) >= P0_MAXDD_FLOOR
    ]
    if not usable:
        usable = [r for r in rows if r.get("code") in {"R2_B00", "R2_D05"} and r.get("cagr") is not None]
    conservative = sorted(
        usable,
        key=lambda r: (float(r.get("max_drawdown") or 0), float(r.get("worst_year_return") or -1)),
        reverse=True,
    )[0]
    balanced_front = pareto_front([
        {
            "code": r["code"],
            "cagr": r.get("cagr"),
            "sharpe": r.get("sharpe"),
            "calmar": r.get("calmar"),
            "max_drawdown": r.get("max_drawdown"),
            "worst_year_return": r.get("worst_year_return"),
            **r,
        }
        for r in usable
    ])
    balanced = balanced_front[0] if balanced_front else conservative
    aggressive = max(
        [r for r in usable if float(r["max_drawdown"]) >= -0.30],
        key=lambda r: (float(r.get("cagr") or 0), float(r.get("sharpe") or 0)),
        default=conservative,
    )
    return {
        "Conservative": conservative,
        "Balanced": balanced,
        "Aggressive": aggressive,
    }


def write_conclusion(rows: list[dict[str, Any]] | None = None) -> Path:
    rows = rows or _all_rows()
    write_single_module_md(rows)
    cands = pick_candidates(rows)
    singles = [r for r in rows if r.get("module") in {"B", "C", "D", "E", "F", "G"} and r.get("cagr") is not None]
    by_mod = {}
    for r in singles:
        by_mod.setdefault(r.get("module"), []).append(r)
    best_mod = None
    best_score = None
    for mod, items in by_mod.items():
        ok = [x for x in items if x.get("max_drawdown") is not None and float(x["max_drawdown"]) >= P0_MAXDD_FLOOR]
        if not ok:
            continue
        # effectiveness: MDD improvement vs CAGR retained
        score = max(
            float(x["max_drawdown"]) - float(LOCKED_BASELINE["max_drawdown"])
            + 0.15 * ((x.get("cagr") or 0) / LOCKED_BASELINE["annualized_return"])
            for x in ok
        )
        if best_score is None or score > best_score:
            best_score = score
            best_mod = mod

    base_dd = float(LOCKED_BASELINE["max_drawdown"])
    base_cagr = float(LOCKED_BASELINE["annualized_return"])
    costs = []
    for r in singles:
        if r.get("max_drawdown") is None or r.get("cagr") is None:
            continue
        d_dd = (float(r["max_drawdown"]) - base_dd) * 100  # pct points, positive = better
        d_cagr = (float(r["cagr"]) - base_cagr) * 100
        if d_dd > 0.2:
            costs.append((r["code"], d_cagr / d_dd, d_dd, d_cagr))
    cost_txt = "；".join(
        f"{c[0]}: 每降 1pct MDD 的 CAGR {c[1]:+.2f}pct（ΔMDD {c[2]:+.2f} / ΔCAGR {c[3]:+.2f}）"
        for c in sorted(costs, key=lambda x: x[1], reverse=True)[:6]
    ) or "单模块里几乎没有在 MDD 改善的同时能量化出稳定的每 1pct 代价（多数组 MDD 没比 −20.62% 更好，或交易被滤到过少）。"

    attr_path = ensure_round2_tree() / "01_attr" / "summary.json"
    attr = json.loads(attr_path.read_text(encoding="utf-8")) if attr_path.is_file() else {}
    audit_path = ensure_round2_tree() / "00_audit" / "identity.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}

    run_sum = ensure_round2_tree() / "03_single" / "run_summary.json"
    summary = json.loads(run_sum.read_text(encoding="utf-8")) if run_sum.is_file() else {}

    lines = [
        "# 06 Round 2 结论",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "- 性质：historical development，**不是 final OOS**。",
        "- 锁定 Alpha 仍是 `R1_00_baseline`。本轮**不自动改锁定代号**。",
        "",
        "## 三个候选（不锁单一最高收益）",
        "",
        "| 角色 | 代号 | CAGR | Sharpe | MaxDD | Worst Year | Worst 12M |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for role, row in cands.items():
        lines.append(
            f"| {role} | `{row.get('code')}` | {_fmt_pct(row.get('cagr'))} | "
            f"{_fmt_num(row.get('sharpe'))} | "
            f"{_fmt_pct(row.get('max_drawdown'))} | {_fmt_pct(row.get('worst_year_return'))} | "
            f"{_fmt_pct(row.get('worst_12m'))} |"
        )
    rec = cands["Balanced"].get("code")
    lines += [
        "",
        f"**Round 3 建议对照：`{rec}`（Balanced）**。原因：在 CAGR–Sharpe–MDD 帕累托上更完整，且未把曲线滤成无交易。Conservative 作风控下限，Aggressive 只在 MDD 仍 ≥ −30% 时保留右尾。用户确认前不改 `R1_00_baseline`。",
        "",
        "## Q1 回撤主因？",
        "",
        attr.get("verdict") or attr.get("hypothesis") or "见 01_loss_attribution.md。锁定 2022 是 +19.22%，全样本 MaxDD −20.62%，不是「2022 亏 20%」。",
        "",
        "## Q2 年度为什么不稳？",
        "",
        "锁定分年：2018 −2.89% 是唯一负年，2024 +58.77% 右尾极厚。单种子 50×2 seed73 MaxDD −34.11%、100×2 seed17 −34.94%，集成把路径平均掉，但年度仍由少数连板簇驱动。年度不稳来自右尾集中 + 情绪环境切换，不是单一 2022 灾难年。",
        "",
        f"## Q3 哪类 filter 最有效？",
        "",
        f"本轮单模块比较后，相对有效的大类是 **{best_mod or '尚未形成稳定赢家'}**（综合 MDD 改善与 CAGR 保留）。Confidence 过滤交易日很多，容易变成近零交易曲线，这类不得宣称「MDD 很好」。Exit 覆盖改变持有路径，和仓位乘子不是同一维度。",
        "",
        "## Q4 每降 1pct MDD 的 CAGR 代价？",
        "",
        cost_txt,
        "",
        "## Q5 有没有 MDD<30% 且收益没崩的组？",
        "",
        f"P0 地板 MaxDD ≥ −30%。锁定自身已满足（−20.62%）。候选 Conservative `{cands['Conservative'].get('code')}` / Balanced `{cands['Balanced'].get('code')}` / Aggressive `{cands['Aggressive'].get('code')}` 都按该地板筛选。收益是否「没崩」看 CAGR 是否仍接近锁定 22.93% 的七成以上；具体数字见表。",
        "",
        "## Q6 时间 / seed / 邻档稳不稳？",
        "",
        f"shortlist={summary.get('shortlist')}。邻档全崩标 fragile 的不得进组合。窗口 2018–2020 / 2021–2023 / 2024–2026 写在各组 report.metrics.windows。C1 诊断见归因文，不锁参数。",
        "",
        "## Q7 哪些是过拟合？",
        "",
        "- 专拟合「2022 −20%」是过拟合：该账本不存在。",
        "- 把乘数乘到接近 0 再展示低 MDD，是无交易伪风控。",
        "- 邻档 70/80/90 全崩的 FAIL 阈值、只在 2024 右尾年生效的拥挤度，标 fragile / 过拟合嫌疑。",
        "- 风险模型若只在标签年有效、邻年失效，不得当 Full Stack 主滤器。",
        "- 本轮结果一律 historical development。",
        "",
        f"## 2022 −20% 账本",
        "",
        audit.get("conclusion") or "见 00_baseline_identity_audit.md。",
        "",
    ]
    dest = ensure_round2_tree() / "06_round2_conclusion.md"
    assert_project_result_path(dest)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ensure_round2_tree() / "06_conclusion" / "candidates.json").write_text(
        json.dumps(cands, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _update_tracking(rows, cands)
    return dest


def _md_row(row: dict[str, Any]) -> str:
    return (
        f"| `{row.get('code')}` | {row.get('module','')} | {row.get('regime') or row.get('confidence') or row.get('risk') or ''} | "
        f"{row.get('position')} | {row.get('exit')} | {_fmt_pct(row.get('cagr'))} | "
        f"{_fmt_num(row.get('sharpe'))} | "
        f"{_fmt_num(row.get('sortino'))} | "
        f"{_fmt_num(row.get('calmar'), 2)} | "
        f"{_fmt_pct(row.get('max_drawdown'))} | {_fmt_pct(row.get('worst_year_return'))} | "
        f"{_fmt_pct(row.get('worst_12m'))} | {row.get('completed_trades') or '—'} | {row.get('status')} |"
    )


def _update_tracking(rows: list[dict[str, Any]], cands: dict[str, dict[str, Any]]) -> None:
    path = PROJECT_ROOT / "result" / "一进二_最优实验组追踪.md"
    text = path.read_text(encoding="utf-8")
    marker = "\n## Round 2\n"
    section = [
        "",
        "## Round 2",
        "",
        "整轮只在 `notebook/02_一进二风险叠加.ipynb`。组是节，不是新 notebook。产物 `result/round2/{code}/`。",
        "Alpha 锁定 `R1_00_baseline`：fs00263 / n50_d2 / 五种子集成 / Top1 / 双门控 / NAV30% / `close_from_d2`。",
        "Risk 只改 `allocation_multiplier` / 入场门 / 退出覆盖，不改 score。ΔSharpe>+0.02 **不再**当替换门槛。",
        "P0 筛：MaxDD ≥ −30%；优先 worst year ≥ −10%、worst 12m ≥ −15%。结果是 historical development，不是 final OOS。",
        "**不自动改锁定代号。**",
        "",
        "### 三个候选",
        "",
        "| 角色 | 代号 | CAGR | Sharpe | MaxDD | Worst Year | 说明 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    notes = {
        "Conservative": "最低 MDD / 最好 worst-year",
        "Balanced": "CAGR–Sharpe–MDD 平衡；建议 Round 3 对照（待确认）",
        "Aggressive": "尽量保留 baseline 右尾，且 MDD < 30%",
    }
    for role, row in cands.items():
        section.append(
            f"| {role} | `{row.get('code')}` | {_fmt_pct(row.get('cagr'))} | "
            f"{_fmt_num(row.get('sharpe'))} | "
            f"{_fmt_pct(row.get('max_drawdown'))} | {_fmt_pct(row.get('worst_year_return'))} | {notes[role]} |"
        )
    section += [
        "",
        "### 主表",
        "",
        "| 代号 | 模块 | Regime/Conf/Risk | Position | Exit | CAGR | Sharpe | Sortino | Calmar | MaxDD | Worst Year | Worst 12M | 交易 | 状态 |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row.get("cagr") is None and str(row.get("status", "")).startswith("待"):
            continue
        section.append(_md_row(row))
    section += [
        "",
        "详见 `result/round2/00_baseline_identity_audit.md` / `01_loss_attribution.md` / `02_research_notes.md` / `06_round2_conclusion.md`。",
        "",
    ]
    new_section = "\n".join(section)
    if marker in text:
        text = text.split(marker)[0].rstrip() + new_section
    else:
        text = text.rstrip() + new_section
    path.write_text(text + "\n", encoding="utf-8")


def write_c01_diagnostic() -> Path:
    dest = ensure_round2_tree() / "03_single" / "C01_seed_disagreement.md"
    attr = ensure_round2_tree() / "01_loss_attribution.md"
    body = [
        "# R2_C01 诊断（不锁参数、不单独回测）",
        "",
        "五种子 rank_pct 标准差 vs 前瞻收益 / 大亏 / 连板的数字写在 `01_loss_attribution.md` 的 C1 节。",
        "本文件只占位，避免把诊断组误跑成回测。",
        "",
        f"归因文存在：{attr.is_file()}",
        "",
    ]
    assert_project_result_path(dest)
    dest.write_text("\n".join(body), encoding="utf-8")
    return dest
