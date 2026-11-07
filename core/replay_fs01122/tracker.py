"""写 result/一进二_fs01122_Top2_实验组追踪.md。不改旧表、不改 baseline_v3。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..paths import RESULT_ROOT, assert_project_result_path, replay_fs01122_dir
from .catalog import overlay_jobs
from .contract import TARGET_MAXDD, TARGET_N_TRADES, TARGET_TOTAL_RETURN


def _pct(x, digits=2) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if abs(v) > 5:  # already percent-like? treat as ratio
        pass
    return f"{v:+.{digits}%}" if v < 0 or True else f"{v:.{digits}%}"


def _cagr(x) -> str:
    if x is None:
        return "—"
    return f"{float(x):.2%}"


def _sh(x) -> str:
    if x is None:
        return "—"
    return f"{float(x):.3f}"


def _mdd(x) -> str:
    if x is None:
        return "—"
    return f"{float(x):.2%}"


def _n(x) -> str:
    if x is None:
        return "—"
    return str(int(x))


def load_summary(round_name: str, code: str) -> dict[str, Any] | None:
    for name in ("summary.json", "report.json"):
        path = replay_fs01122_dir(round_name, code) / name
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if name == "report.json":
                m = raw.get("metrics") or {}
                return {
                    "code": code,
                    "status": raw.get("status"),
                    "cagr": m.get("annualized_return"),
                    "sharpe": m.get("sharpe_zero_rate"),
                    "max_drawdown": m.get("max_drawdown"),
                    "total_return": m.get("total_return"),
                    "completed_trades": m.get("completed_trades"),
                    "trade_win_rate": m.get("trade_win_rate"),
                    "average_holding_sessions": m.get("average_holding_sessions"),
                    "worst_year_return": m.get("worst_complete_year_return") or m.get("worst_year_return"),
                    "cagr_ex2024": m.get("cagr_ex2024"),
                    "yearly_returns": m.get("yearly_returns"),
                    "title_zh": raw.get("title_zh") or code,
                }
            return raw
    return None


def _row(s: dict[str, Any] | None, title: str, code: str) -> str:
    if s is None:
        return f"| `{code}` | {title} | — | — | — | — | — | 待跑 |"
    st = s.get("status") or ""
    if st in {"ABORT_SEMANTIC_SHIFT", "ABORT"}:
        return f"| `{code}` | {title} | — | — | — | — | — | {st} |"
    tot = s.get("total_return")
    tot_s = f"{float(tot):+.2%}" if tot is not None else "—"
    return (
        f"| `{code}` | {title} | {_sh(s.get('sharpe'))} | {_cagr(s.get('cagr'))} | {tot_s} | "
        f"{_mdd(s.get('max_drawdown'))} | {_n(s.get('completed_trades'))} | 已跑 |"
    )


def _pick_best(codes: list[tuple[str, str]]) -> dict[str, Any] | None:
    best = None
    best_key = None
    for round_name, code in codes:
        s = load_summary(round_name, code)
        if not s or s.get("cagr") is None:
            continue
        key = (float(s.get("sharpe") or 0), float(s.get("cagr") or 0), -abs(float(s.get("max_drawdown") or 1)))
        if best_key is None or key > best_key:
            best_key = key
            best = {**s, "code": code, "round": round_name}
    return best


def write_tracker() -> Path:
    dest = RESULT_ROOT / "一进二_fs01122_Top2_实验组追踪.md"
    assert_project_result_path(dest)
    r100 = load_summary("round1", "R1_00_baseline") or {}
    b01a = load_summary("round2", "R2_B01a") or load_summary("round3", "R3_B01a") or {}
    c03 = load_summary("round2", "R2_C03_half") or load_summary("round3", "R3_02") or {}
    v2 = load_summary("round4", "R4_00_baseline_v2") or load_summary("round3", "R3_03") or {}
    v3 = load_summary("round4", "R4_B05") or load_summary("round5", "R5_00_baseline_v3") or {}

    yearly = (r100.get("yearly_returns") or {}) if r100 else {}

    def yrow(s, name):
        yr = (s or {}).get("yearly_returns") or {}
        cells = []
        for y in range(2018, 2027):
            v = yr.get(str(y))
            cells.append("—" if v is None else f"{float(v):+.2%}")
        return f"| `{name}` | " + " | ".join(cells) + " |"

    lines = [
        "# 一进二策略：fs01122 Top2 实验组追踪",
        "",
        "旧追踪表仍有效：`result/一进二_最优实验组追踪.md`（fs00263 Top1 线）。**不改**旧文档的 `baseline_v3 = R4_B05`。",
        "",
        "> **execution-time feature**：`d1_open_gap` 进了 ranker 特征（D+1 开盘缺口）。本线与 fs00263 D-close 六因子 **不可混比**。本轮不改 Stock 公式。",
        "",
        "## 共享回测契约",
        "",
        "入场 Alpha **锁** Stock fs01122 预滤集成（不重训 Stock）。只改卖出 / 仓位乘子 / 课题内重训。正式数字走 `local_executor` → Stock `fixed_sleeve_backtest`。",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        "| 新表 v1 | **`fs01122_v1`** = `R1_00_baseline`（parity 对齐 Stock `rank25_15`） |",
        "| 组合 / Alpha | `fs01122`（registry 13 因子 + `d1_open_gap` = 14） |",
        "| 容量 | **n100_d2 五种子集成** + 预滤 gap floor |",
        "| 变体 | Top2；`rank_name_budget_nav_fractions=(0.25, 0.15)` |",
        "| 信号区间 | 2018-01-01～2026-06-01 |",
        "| 行情延伸 | 至 2026-07-24 |",
        "| 选股 | Top2；分数含 D+1 `d1_open_gap` |",
        "| 资金 | 双袖套，初始 30 万；两档预算 = 开仓时净值的 **25% + 15%** |",
        "| 门控 | 上涨家数比 ≥45%；D+1 开盘缺口 `open_gap_min=-0.01`（无 +7% 上沿） |",
        "| 卖出 | 默认 `close_from_d2` |",
        "| C03 | **fs01122 当日分数** 重算 rolling250，禁止复用 fs00263 C03 |",
        "| 市场层 | B01a / 连板率规则与旧 R4 相同，乘到 **两档预算** |",
        "| F 组 | 本线已是 Top2：旧 F「再买 Top2」→ `ABORT_SEMANTIC_SHIFT`；另跑 `R4_F_TOP3_diag` |",
        "| 账本 | `result/replay_fs01122/`；旧 `result/roundN/` 不覆盖 |",
        "",
        f"Parity 目标：累计 **+943.69%**（{TARGET_TOTAL_RETURN:.6f}）/ **{TARGET_N_TRADES}** 笔 / MaxDD **−27.50%**（{TARGET_MAXDD:.6f}）。",
        "",
        "## 当前最优（本线，只推荐不改旧锁）",
        "",
        "v1 = 截图/parity 通过的 fs01122 Top2。v2/v3 在本线 **重新评选** 候选（旧 B01a×C03、旧 R4_B05 结构套在新 Alpha 上，数字会变）。",
        "",
        "| 名称 | 操作/构成 | Sharpe | 年化 | 累计 | MaxDD | 笔数 | 备注 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        _row(r100, "无风险层，close_from_d2，parity 底", "fs01122_v1 / R1_00_baseline"),
        _row(v2, "上涨家数分档 × 分差半仓（旧 v2 结构）", "fs01122_v2 候选 / R3_03 或 R4_00"),
        _row(v3, "min(B01a, 连板率半仓) × 分差半仓（旧 v3 结构）", "fs01122_v3 候选 / R4_B05"),
        "",
        "### 分年",
        "",
        "2026 为 YTD，不得当完整年。",
        "",
        "| | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        yrow(r100, "fs01122_v1"),
        yrow(v2, "fs01122_v2 候选"),
        yrow(v3, "fs01122_v3 候选"),
        "",
        "## Round 1 主表",
        "",
        "同一套 fs01122 Top2 25/15 入场，只改日频代理卖出。",
        "",
        "| 实验组 | 卖出 | Sharpe | 年化 | 累计 | MaxDD | 笔数 | 状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for code, title in (
        ("R1_00_baseline", "D+2 起首个可卖收盘"),
        ("R1_01_tp1100", "仅 11:00 止盈（开盘价代理）"),
        ("R1_02_tp1450", "仅 14:50 止盈（收盘价代理）"),
        ("R1_03_ma5stop", "仅 14:50 止损 price<MA5×1.05"),
        ("R1_04_full", "11:00 止盈 → 14:50 止盈 → 14:50 止损"),
    ):
        s = load_summary("round1", code)
        lines.append(_row(s, title, code))

    def section(round_name: str, heading: str, codes: list[str] | None = None) -> None:
        lines.append("")
        lines.append(f"## {heading}")
        lines.append("")
        lines.append("| 代号 | 操作 | Sharpe | 年化 | 累计 | MaxDD | 笔数 | 状态 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        jobs = [j for j in overlay_jobs() if j.round_name == round_name]
        if codes:
            want = set(codes)
            jobs = [j for j in jobs if j.code in want] or jobs
        # de-dup preserve order
        seen = set()
        for job in jobs:
            if job.code in seen:
                continue
            seen.add(job.code)
            s = load_summary(round_name, job.code)
            title = job.title_zh.replace("|", "/")
            lines.append(_row(s, title, job.code))

    section("round2", "Round 2（锁分 overlay）")
    section("round3", "Round 3（锁分 overlay；C03 换源）")
    section("round4", "Round 4（锁分 overlay；F 语义见 ABORT）")
    section("round5", "Round 5 锁分组 + 训练组")
    # also dump trained R5 D-G even if not in overlay catalog
    for code in [f"R5_D{i:02d}" for i in range(6)] + [f"R5_E{i:02d}" for i in range(1, 7)] + [
        "R5_F00", "R5_F01", "R5_F02", "R5_F03", "R5_F04", "R5_F05_huber", "R5_G01", "R5_G02", "R5_G03",
    ]:
        s = load_summary("round5", code)
        if s:
            lines.append(_row(s, s.get("title_zh") or code, code))
    section("round6", "Round 6 锁分组 + 训练组")
    for code in (
        ["R6_CANONICAL_RETRAIN"]
        + [f"R6_B{i:02d}" for i in range(1, 8)]
        + [f"R6_C{i:02d}" for i in range(1, 5)]
        + [f"R6_D{i:02d}" for i in range(1, 5)]
        + [f"R6_E{i:02d}" for i in range(1, 5)]
        + ["R6_G01", "R6_H01"]
        + [f"R6_K{i:02d}" for i in range(1, 7)]
    ):
        s = load_summary("round6", code)
        if s:
            lines.append(_row(s, s.get("title_zh") or code, code))

    lines += [
        "",
        "## 评选说明",
        "",
        "- `fs01122_v1`：parity 通过的 R1_00。",
        "- `fs01122_v2` 候选：旧 B01a×C03 结构套在新 Alpha（`R3_03` / `R4_00_baseline_v2` 重放）。",
        "- `fs01122_v3` 候选：旧 R4_B05 结构套在新 Alpha。**不改**旧追踪表锁。",
        "",
        f"产物根：`result/replay_fs01122/`。旧表 `result/一进二_最优实验组追踪.md` 未删、未改锁。",
        "",
    ]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _ = b01a, c03, yearly
    return dest
