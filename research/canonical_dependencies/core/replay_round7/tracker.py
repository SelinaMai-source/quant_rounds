"""写 result/一进二_Round7基线_实验组追踪.md。版式对齐旧「最优实验组追踪」。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..paths import RESULT_ROOT, assert_project_result_path, replay_round7_dir
from .catalog import overlay_jobs
from .contract import (
    CANDIDATE_CODE,
    TARGET_CAGR,
    TARGET_MAXDD,
    TARGET_N_TRADES,
    TARGET_SHARPE,
    TARGET_TOTAL_RETURN,
)


def load_summary(round_name: str, code: str) -> dict[str, Any] | None:
    dest = replay_round7_dir(round_name, code)
    summary_path = dest / "summary.json"
    report_path = dest / "report.json"
    raw: dict[str, Any] | None = None
    if summary_path.is_file():
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report.get("metrics") or {}
        merged = {
            "code": code,
            "status": report.get("status") or (raw or {}).get("status"),
            "title_zh": (raw or {}).get("title_zh") or report.get("title_zh") or code,
            "cagr": (raw or {}).get("cagr") if raw and raw.get("cagr") is not None else metrics.get("annualized_return"),
            "sharpe": (raw or {}).get("sharpe") if raw and raw.get("sharpe") is not None else metrics.get("sharpe_zero_rate"),
            "max_drawdown": (raw or {}).get("max_drawdown") if raw and raw.get("max_drawdown") is not None else metrics.get("max_drawdown"),
            "total_return": (raw or {}).get("total_return") if raw and raw.get("total_return") is not None else metrics.get("total_return"),
            "completed_trades": (raw or {}).get("completed_trades") if raw and raw.get("completed_trades") is not None else metrics.get("completed_trades"),
            "trade_win_rate": (raw or {}).get("trade_win_rate") if raw and raw.get("trade_win_rate") is not None else metrics.get("trade_win_rate"),
            "average_holding_sessions": (
                (raw or {}).get("average_holding_sessions")
                if raw and raw.get("average_holding_sessions") is not None
                else metrics.get("average_holding_sessions")
            ),
            "worst_year_return": (
                (raw or {}).get("worst_year_return")
                if raw and raw.get("worst_year_return") is not None
                else metrics.get("worst_complete_year_return") or metrics.get("worst_year_return")
            ),
            "cagr_ex2024": (raw or {}).get("cagr_ex2024") if raw and raw.get("cagr_ex2024") is not None else metrics.get("cagr_ex2024"),
            "yearly_returns": (raw or {}).get("yearly_returns") or metrics.get("yearly_returns") or {},
            "error": report.get("error"),
        }
        return merged
    return raw


def _pct(x, digits=2, signed=True) -> str:
    if x is None:
        return "—"
    v = float(x)
    txt = f"{v:.{digits}%}"
    if signed and v > 0:
        txt = "+" + txt
    return txt.replace("-", "−")


def _sh(x) -> str:
    if x is None:
        return "—"
    return f"{float(x):.3f}"


def _n(x) -> str:
    if x is None:
        return "—"
    return str(int(x))


def _hold(x) -> str:
    if x is None:
        return "—"
    return f"{float(x):.2f}"


def _win(x) -> str:
    if x is None:
        return "—"
    return f"{float(x):.2%}"


def _worst_year_cell(s: dict[str, Any] | None) -> str:
    if not s:
        return "—"
    yr = s.get("yearly_returns") or {}
    worst = None
    worst_y = None
    for y in range(2018, 2026):
        v = yr.get(str(y), yr.get(y))
        if v is None:
            continue
        v = float(v)
        if worst is None or v < worst:
            worst, worst_y = v, y
    if worst is None:
        w = s.get("worst_year_return")
        return "—" if w is None else _pct(w)
    return f"{_pct(worst)}（{worst_y}）"


def _status_label(s: dict[str, Any] | None) -> str:
    if s is None:
        return "待跑"
    st = str(s.get("status") or "")
    if st in {"ABORT_SEMANTIC_SHIFT", "ABORT"}:
        return st
    if st in {"skipped", "ABANDONED", "FAILED"}:
        return st
    if s.get("cagr") is None and not s.get("total_return"):
        return st or "待跑"
    return "已跑"


def _delta_sh(s, base) -> str:
    if not s or not base or s.get("sharpe") is None or base.get("sharpe") is None:
        return "—"
    d = float(s["sharpe"]) - float(base["sharpe"])
    return f"{d:+.3f}".replace("-", "−")


def _delta_ret(s, base) -> str:
    if not s or not base or s.get("total_return") is None or base.get("total_return") is None:
        return "—"
    d = (float(s["total_return"]) - float(base["total_return"])) * 100
    return f"{d:+.2f}pct".replace("-", "−")


def _row_full(s: dict[str, Any] | None, ops: str, code: str, base: dict[str, Any] | None = None) -> str:
    st = _status_label(s)
    if s is None or st in {"ABORT_SEMANTIC_SHIFT", "ABORT", "skipped", "ABANDONED", "FAILED", "待跑"}:
        extra = "" if not s or not s.get("error") else f"；{s.get('error')}"
        return f"| `{code}` | {ops} | — | — | — | — | — | — | — | — | {st}{extra} |"
    tot = s.get("total_return")
    tot_s = _pct(tot) if tot is not None else "—"
    return (
        f"| `{code}` | {ops} | {_sh(s.get('sharpe'))} | {_pct(s.get('cagr'), signed=False)} | {tot_s} | "
        f"{_pct(s.get('max_drawdown'))} | {_win(s.get('trade_win_rate'))} | {_hold(s.get('average_holding_sessions'))} | "
        f"{_delta_sh(s, base) if base is not None else '—'} | {_delta_ret(s, base) if base is not None else '—'} | {st} |"
    )


def _row_short(s: dict[str, Any] | None, ops: str, code: str) -> str:
    st = _status_label(s)
    if s is None or st in {"ABORT_SEMANTIC_SHIFT", "ABORT", "skipped", "ABANDONED", "FAILED", "待跑"}:
        return f"| {ops} | `{code}` | — | — | — | — | {st} |"
    return (
        f"| {ops} | `{code}` | {_pct(s.get('cagr'), signed=False)} | {_sh(s.get('sharpe'))} | "
        f"{_pct(s.get('max_drawdown'))} | {_worst_year_cell(s)} | {st} |"
    )


def _row_mid(s: dict[str, Any] | None, ops: str, code: str) -> str:
    st = _status_label(s)
    if s is None or st in {"ABORT_SEMANTIC_SHIFT", "ABORT", "skipped", "ABANDONED", "FAILED", "待跑"}:
        return f"| {ops} | `{code}` | — | — | — | — | — | {st} |"
    tot = s.get("total_return")
    return (
        f"| {ops} | `{code}` | {_sh(s.get('sharpe'))} | {_pct(s.get('cagr'), signed=False)} | "
        f"{_pct(tot) if tot is not None else '—'} | {_pct(s.get('max_drawdown'))} | {_n(s.get('completed_trades'))} | {st} |"
    )


def _yrow(s: dict[str, Any] | None, name: str) -> str:
    yr = (s or {}).get("yearly_returns") or {}
    cells = []
    for y in range(2018, 2027):
        v = yr.get(str(y), yr.get(y))
        cells.append("—" if v is None else _pct(v))
    return f"| `{name}` | " + " | ".join(cells) + " |"


def _beats_alpha(candidate: dict[str, Any], base: dict[str, Any]) -> bool:
    if not candidate or not base:
        return False
    need = ("cagr", "sharpe", "completed_trades")
    if any(candidate.get(k) is None or base.get(k) is None for k in need):
        return False
    if float(candidate["completed_trades"]) < 0.80 * float(base["completed_trades"]):
        return False
    return float(candidate["cagr"]) > float(base["cagr"]) + 0.005 and float(candidate["sharpe"]) > float(base["sharpe"]) + 0.02


def _jobs_by_round(round_name: str) -> list:
    seen = set()
    out = []
    for job in overlay_jobs():
        if job.round_name != round_name or job.code in seen:
            continue
        seen.add(job.code)
        out.append(job)
    return out


def write_tracker() -> Path:
    dest = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
    assert_project_result_path(dest)

    r100 = load_summary("round1", "R1_00_baseline") or {}
    v2 = load_summary("round4", "R4_00_baseline_v2") or load_summary("round3", "R3_03") or {}
    v3 = load_summary("round4", "R4_B05") or load_summary("round5", "R5_00_baseline_v3") or {}
    base = r100

    train_hits = []
    overlay_hits = []
    for rnd in ("round5", "round6"):
        root = replay_round7_dir(rnd)
        if not root.is_dir():
            continue
        for child in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(("R5_", "R6_"))):
            s = load_summary(rnd, child.name)
            cfg = child / "config.json"
            alpha_src = ""
            if cfg.is_file():
                alpha_src = str(json.loads(cfg.read_text(encoding="utf-8")).get("alpha_source") or "")
            if s and _beats_alpha(s, base):
                if alpha_src == "local_retrain":
                    train_hits.append((child.name, s))
                else:
                    overlay_hits.append((child.name, s))
    for job in _jobs_by_round("round2") + _jobs_by_round("round3") + _jobs_by_round("round4"):
        s = load_summary(job.round_name, job.code)
        if s and _beats_alpha(s, base):
            overlay_hits.append((job.code, s))

    lines: list[str] = []
    lines += [
        "# 一进二策略：Round7 基线实验组追踪",
        "",
        "旧表仍有效、**未删未改锁**：`result/一进二_最优实验组追踪.md`（fs00263 Top1，`baseline_v3=R4_B05`）；",
        "`result/一进二_fs01122_Top2_实验组追踪.md`（fs01122 Top2 25/15）。本表只服务 Round7 开发基线重放。",
        "",
        "historical development，**不是 final OOS / production**。`metric_version` = `r3_locked_window_v1`。",
        "Stock Round7 结论仍写 `baseline_v3_unchanged=true`、`baseline_v4_created=false`；",
        f"本线把候选 **`{CANDIDATE_CODE}`** 当作 **新开发基线** 重放 R1–R6，不自动升版。",
        "",
        "> **execution-time feature**：`d1_open_gap` 进了 ranker（D+1 开盘缺口）。与 fs00263 D-close 六因子 **不可混比**。",
        "> `d1_open_gap` 在 09:25 已知，研究成交用同一日开盘价；数字不表述为可直接实盘执行。",
        "",
        "## 共享回测契约",
        "",
        "入场 Alpha **锁死** Round7 双头 90/10（不改 score）。overlay / 卖出 / 课题内重训叠在这套契约上。",
        "正式数字走 `local_executor` → Stock `fixed_sleeve_backtest`。C03 用 **当前 Round7 双头分数** rolling250。",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 当前锁定（本线开发基线） | **`{CANDIDATE_CODE}`** = `R1_00_baseline`（clean_risk × Top2 30/10） |",
        "| 组合 / Alpha | 双头 no-sampling XGB（`subsample=colsample=1.0`）：five=`label_e2_net` / ten=`label_exit_rule_net`；seed 17/42/73/101/202 |",
        "| 融合 | 同日 percentile 后 **0.90 ten + 0.10 five**；预滤 `d1_open_gap ≥ -0.01` |",
        "| 容量 / 变体 | Top2；`rank_name_budget_nav_fractions=(0.3, 0.1)` |",
        "| 信号区间 | 2018-01-01～2026-06-01 |",
        "| 行情延伸 | 至 2026-07-24（为未平仓连板留退出窗） |",
        "| 选股 | Top2；分数含 D+1 `d1_open_gap`；D 收盘后打分，D+1 开盘买 |",
        "| 资金 | 双袖套，初始 30 万；两档预算 = 开仓时净值的 **30% + 10%**；`name_budget=10000` |",
        "| 执行器门控 | 无上涨家数门、无 open_gap 上下沿（gap 已在融合层） |",
        "| 风险层（默认） | clean_risk：广度 ≥45% 且风格不低于开发窗 p20 → 满仓/0.75；弱广度可用拍卖救援 ×0.25；涨停家数弱于 20 日均则再 ×0.5。阈值窗 2018–2021 |",
        "| 卖出 | `close_from_d2`（D+2 起首个可卖收盘） |",
        "| 费用 | 滑点 12.3bp、佣金 1bp、过户 0.1bp、卖印花税 5bp |",
        "| Round 1 卖点代理 | **日频**：11:00 = 当日 **open**；14:50 = 当日 **close**；止损 `ma_price < MA5×1.05` |",
        "| C03 | **当前模型分数** Top1−Top2 rolling250，禁止复用 fs00263 |",
        "| F 组 | 本线已是 Top2：旧 F「再买 Top2」→ `ABORT_SEMANTIC_SHIFT` |",
        "| 账本 | `result/replay_round7/`；Stock `result/round7/` 只读；旧 `result/roundN/` 与 `replay_fs01122/` 不覆盖 |",
        "",
        "年化 \\((最终资产/300000)^{1/years}-1\\)，`years=(最后交易日-首个交易日).days/365.25`；",
        "Sharpe = 日收益均值 / 日收益标准差 × \\(\\sqrt{252}\\)，无风险利率 0。",
        "评价窗 2018-01-02～2026-07-24（`r3_locked_window_v1`）。消融组**不**回退 `close_from_d2`。",
        "",
        (
            f"Parity：本地 `R1_00_baseline` 对齐 Stock `{CANDIDATE_CODE}`："
            f"累计 **{TARGET_TOTAL_RETURN:.6f}** / **{TARGET_N_TRADES}** 笔 / "
            f"MaxDD **{TARGET_MAXDD:.6f}** / Sharpe **{TARGET_SHARPE:.4f}** / 年化 **{TARGET_CAGR:.4%}**。"
        ),
        "",
        "## 当前最优 baseline",
        "",
        "本线入场都是 Round7 双头 90/10。**v1 = Round7 候选本身**（clean_risk 已含在锁里）。",
        "v2 / v3 只表示「旧追踪表那套 overlay 结构套在 Round7 Alpha 上」，数字会变，**不改**旧表锁，也 **不自动** 替换本线 v1。",
        "",
        "| 名称 | 操作/构成 | Sharpe | 年化 | 累计 | MaxDD | 最差年 | 笔数 | 备注 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    def base_row(s, name, ops, note):
        st = _status_label(s)
        if st != "已跑":
            lines.append(f"| {name} | {ops} | — | — | — | — | — | — | {st} |")
            return
        tot = s.get("total_return")
        lines.append(
            f"| {name} | {ops} | {_sh(s.get('sharpe'))} | {_pct(s.get('cagr'), signed=False)} | "
            f"{_pct(tot) if tot is not None else '—'} | {_pct(s.get('max_drawdown'))} | "
            f"{_worst_year_cell(s)} | {_n(s.get('completed_trades'))} | {note} |"
        )

    base_row(
        r100,
        f"`r7_v1`（`R1_00_baseline` / `{CANDIDATE_CODE}`）",
        "双头 90/10 + clean_risk + Top2 30/10，`close_from_d2`",
        "本线开发基线；parity 通过；最差完整年见左",
    )
    base_row(
        v2,
        "`r7_v2 结构候选`（`R3_03` / `R4_00`）",
        "旧 v2：上涨家数分档 × 分差半仓，套在 Round7 Alpha 上",
        "overlay 去风险，不是新 Alpha",
    )
    base_row(
        v3,
        "`r7_v3 结构候选`（`R4_B05`）",
        "旧 v3：min(B01a, 连板率半仓) × 分差半仓，套在 Round7 Alpha 上",
        "overlay 去风险，不是新 Alpha",
    )

    lines += [
        "",
        "### 分年",
        "",
        "2026 为 YTD，不得当完整年。",
        "",
        "| | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _yrow(r100, "r7_v1"),
        _yrow(v2, "r7_v2 结构"),
        _yrow(v3, "r7_v3 结构"),
        "",
        "## Round 1 主表",
        "",
        "同一套 Round7 风险层入场，只改日频代理卖出。达门槛也不自动改锁定代号。",
        "",
        "| 实验组 | 卖出 | Sharpe | 年化 | 累计 | MaxDD | 胜率 | 平均持有日 | ΔSharpe | Δret | 状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for code, ops in (
        ("R1_00_baseline", "D+2 起首个可卖收盘"),
        ("R1_01_tp1100", "仅 11:00 止盈（开盘价代理）"),
        ("R1_02_tp1450", "仅 14:50 止盈（收盘价代理）"),
        ("R1_03_ma5stop", "仅 14:50 止损 `price < MA5×1.05`"),
        ("R1_04_full", "11:00 止盈 → 14:50 止盈 → 14:50 止损"),
    ):
        lines.append(_row_full(load_summary("round1", code), ops, code, base))

    r2 = {j.code: j for j in _jobs_by_round("round2")}

    def r2_block(heading: str, codes: list[str], *, short: bool = True) -> None:
        lines.append("")
        lines.append(heading)
        lines.append("")
        if short:
            lines.append("| 操作 | 代号 | CAGR | Sharpe | MaxDD | 最差年 | 状态 |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
            for code in codes:
                job = r2.get(code)
                ops = job.title_zh if job else code
                lines.append(_row_short(load_summary("round2", code), ops, code))
        else:
            lines.append("| 操作 | 代号 | Sharpe | 年化 | 累计 | MaxDD | 笔数 | 状态 |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
            for code in codes:
                job = r2.get(code)
                ops = job.title_zh if job else code
                lines.append(_row_mid(load_summary("round2", code), ops, code))

    lines += [
        "",
        "## Round 2",
        "",
        "Alpha 锁死 Round7 双头分；只改仓位乘子 / 入场门 / 退出，不改 score。C03 按 **当前 Round7 分** rolling250。",
        "诊断组（如 `R2_C01`）不回测。达门槛也不自动改锁定代号。",
    ]
    r2_block("### B 市场状态", [c for c in r2 if c.startswith("R2_B") and "_R2_" not in c])
    r2_block("### C 置信度", [c for c in r2 if c.startswith("R2_C") and "_R2_" not in c])
    r2_block("### D 仓位", [c for c in r2 if c.startswith("R2_D") and "_R2_" not in c])
    r2_block("### E 退出", [c for c in r2 if c.startswith("R2_E") and "_R2_" not in c])
    r2_block("### F 风险模型", [c for c in r2 if c.startswith("R2_F") and "_R2_" not in c])
    r2_block("### G 新风险因子", [c for c in r2 if c.startswith("R2_G") and "_R2_" not in c])
    r2_block("### 两个模块叠加", [c for c in r2 if c.startswith("R2_P")], short=False)
    r2_block("### 三个模块叠加", [c for c in r2 if c.startswith("R2_T")], short=False)
    r2_block("### 多模块一起加", [c for c in r2 if c.startswith("R2_H")], short=False)

    def dump_round(round_name: str, heading: str, intro: str) -> None:
        lines.append("")
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(intro)
        lines.append("")
        lines.append("| 操作 | 代号 | Sharpe | 年化 | 累计 | MaxDD | 笔数 | 状态 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for job in _jobs_by_round(round_name):
            lines.append(_row_mid(load_summary(round_name, job.code), job.title_zh.replace("|", "/"), job.code))

    dump_round(
        "round3",
        "Round 3",
        "锁分 overlay；C03 换源为 Round7 当前分。含短名单、消融、双置信度、仓位 20/25/30、邻域、去 2024 / LOYO / 成本。诊断不回测的已跳过。",
    )
    dump_round(
        "round4",
        "Round 4",
        "换 B / 换 C / 两两 / 三元 / 随机对照。旧 F「Top1 上再买 Top2」在本线已是 Top2，标 `ABORT_SEMANTIC_SHIFT`；另留 `R4_F_TOP3_diag`。",
    )
    dump_round(
        "round5",
        "Round 5 锁分组",
        "锁 Round7 Alpha，把旧 v3 overlay 结构（及 H 组）套上来。训练组（只改学习问题、风险层锁 Round7 clean_risk 规则并按新分重算）见表后。",
    )

    lines += [
        "",
        "### Round 5 训练组（父特征 = Round7 14 因子；风险层 = clean_risk 规则）",
        "",
        "| 操作 | 代号 | Sharpe | 年化 | 累计 | MaxDD | 笔数 | 状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    r5_train_codes = (
        [f"R5_D{i:02d}" for i in range(6)]
        + [f"R5_E{i:02d}" for i in range(1, 7)]
        + ["R5_F00", "R5_F01", "R5_F02", "R5_F03", "R5_F04", "R5_F05_huber", "R5_G01", "R5_G02", "R5_G03"]
        + ["R5_H01", "R5_H02", "R5_H03"]
    )
    seen_r5 = {j.code for j in _jobs_by_round("round5")}
    for code in r5_train_codes:
        s = load_summary("round5", code)
        if not s:
            continue
        if code in seen_r5 and s.get("cagr") is None:
            continue
        title = s.get("title_zh") or code
        lines.append(_row_mid(s, title, code))

    dump_round(
        "round6",
        "Round 6 锁分组",
        "锁 Round7 Alpha 的 overlay 对照。训练组（窗长 / 表征 / 目标；风险层锁 clean_risk 规则）见表后。A04 0.8 仅 FORENSIC_ONLY，不跑。",
    )
    lines += [
        "",
        "### Round 6 训练组",
        "",
        "| 操作 | 代号 | Sharpe | 年化 | 累计 | MaxDD | 笔数 | 状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    r6_train_codes = (
        ["R6_CANONICAL_RETRAIN"]
        + [f"R6_B{i:02d}" for i in range(1, 8)]
        + [f"R6_C{i:02d}" for i in range(1, 5)]
        + [f"R6_D{i:02d}" for i in range(1, 5)]
        + [f"R6_E{i:02d}" for i in range(1, 5)]
        + ["R6_G01", "R6_H01"]
        + [f"R6_K{i:02d}" for i in range(1, 7)]
    )
    seen_r6 = {j.code for j in _jobs_by_round("round6")}
    for code in r6_train_codes:
        s = load_summary("round6", code)
        if not s:
            continue
        title = s.get("title_zh") or code
        lines.append(_row_mid(s, title, code))

    lines += [
        "",
        "## 评选说明",
        "",
        "- 本线开发基线 = **`r7_v1` = Round7 候选**。只推荐替换，若 **Alpha**（新分数 / 重训）同时抬年化与 Sharpe，且笔数未腰斩。",
        "- 仅 overlay 降仓把 MaxDD 收窄、年化不升，视为去风险，**不当**新开发基线。",
        "- 默认 **不自动改锁**，不写 `baseline_v4`。",
    ]
    if train_hits:
        names = ", ".join(f"`{c}`" for c, _ in train_hits[:10])
        lines.append(f"- 训练组里年化+Sharpe 同时超过 Round7 底的：{names}。仍不自动升版，只作候选。")
    else:
        lines.append("- 训练组 **没有** 组在年化+Sharpe 上同时超过 Round7 底，不推荐替换开发基线。")
    if overlay_hits:
        names = ", ".join(f"`{c}`" for c, _ in overlay_hits[:10])
        lines.append(f"- overlay 组年化+Sharpe 同时更好的有 {names}。它们仍用 Round7 锁分，只改仓位/置信度，**不当**新 Alpha 底。")
    else:
        lines.append("- overlay 组没有同时抬年化与 Sharpe 的，不构成换底理由。")
    lines += [
        "",
        "产物根：`result/replay_round7/`。旧两张追踪表未删。Stock `result/round7/` 只读、未改写。",
        "",
    ]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest
