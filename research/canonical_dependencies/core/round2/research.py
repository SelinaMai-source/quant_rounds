"""P0：02_research_notes。只列本机可复现的研报因子。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..paths import assert_project_result_path, ensure_round2_tree
from .data import LIMIT_DOWN_RELATIVE, MARKET_CONTEXT_RELATIVE, load_limit_down_context, load_market_context


def write_research_notes() -> Path:
    ensure_round2_tree()
    ctx = load_market_context()
    down = load_limit_down_context()
    ctx_missing = {c: float(ctx[c].isna().mean()) for c in ctx.columns if c != "trade_date"}
    down_missing = {c: float(down[c].isna().mean()) for c in down.columns if c != "trade_date"}

    lines = [
        "# 02 Research notes（只保留本机可复现项）",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "- 公开检索：涨停情绪 / 赚钱效应 / 晋级率 / 拥挤度。龙虎榜 / 完整 L2 / 新闻 → future work。",
        "- 不是 final OOS。条目必须写 source / formula / 可见时点 / 缺失率 / PIT。",
        "",
        "## 公开研究（只作语义对照，不把券商回测数字当本机结果）",
        "",
        "1. 国泰海通《大类资产与中观配置研究（五）：从涨停板、「打板策略」到赚钱效应引发的情绪择时指标》（2025）：涨停/跌停占比、净涨停占比、打板收益、跌停次日收益。来源：公开新闻稿/慧博摘要。",
        "2. 短线复盘框架（公开专栏）：涨停家数、最高连板、晋级率、炸板率、回封/修复率、涨停溢价率。阈值（如涨停>50、晋级>40%）是经验数，**本轮只用分位粗档，不抄绝对值**。",
        "3. 拥挤度：行业首板数量、成交额拥挤、主力净流入。本机有行业首板与 money_flow，无完整龙虎榜。",
        "",
        "## 本机可复现因子",
        "",
        "### R2-G 将做成 overlay 的 3 个",
        "",
        "| id | source | formula | 可见时点 | 缺失率 | PIT |",
        "|---|---|---|---|---|---|",
        f"| UP_NUM / 涨停家数 | `{MARKET_CONTEXT_RELATIVE}` ← `daily_limit_events` `limit_type=up` 且收盘仍封 | `market_limit_count`；250d 分位 | D 收盘 | {ctx_missing.get('market_limit_count', 0):.2%} | 是；D+1 开盘前可用 |",
        f"| FAIL_RATIO / 炸板率 | 同上 `market_failed_limit_rate` | 1 − 收盘封住/当日触板 | D 收盘 | {ctx_missing.get('market_failed_limit_rate', 0):.2%} | 是 |",
        f"| SUC / CON | `market_seal_success_rate` / `prev_limit_relimit_rate` | 封成率；昨日涨停今日再涨停 | SUC=D 收盘；CON 已用 prev_ 滞后 | SUC {ctx_missing.get('market_seal_success_rate', 0):.2%} / CON {ctx_missing.get('prev_limit_relimit_rate', 0):.2%} | 是 |",
        f"| 跌停 count / 压力 | `{LIMIT_DOWN_RELATIVE}` | `market_close_limit_down_count` / `market_down_pressure` | D 收盘 | down_count {down_missing.get('market_close_limit_down_count', 0):.2%} | 是 |",
        "| 主力资金弱 | Hermes `daily_money_flow` 市场合计 | `sum(main_net_inflow)/sum(turnover_value)`，250d 分位 | D 收盘 `available_at` 当日 | 以实算为准 | 是 |",
        "| 行业首板拥挤 | 候选面板 `ind_first_board_count_log` | Top1 当日截面分位 | D 收盘 | 候选缺失另计 | 是 |",
        "",
        "Round 2 G 组只用这 3 条 overlay：`R2_G01` 主力资金、`R2_G02` 行业首板拥挤、`R2_G03` 跌停压力。B 组已覆盖 UP_NUM / FAIL / SUC-CON / 跌停 count。",
        "",
        "## 明确 missing（禁止编造）",
        "",
        "| 字段 | 原因 | 状态 |",
        "|---|---|---|",
        "| 龙虎榜席位 / 游资名录 | 本机 Hermes 快照无完整 LHB 表 | future work |",
        "| 全市场 1 分钟 / 完整 L2 | AmazingData 分钟只覆盖收盘涨停候选日 | future work |",
        "| 新闻 / 公告情绪 | 无新闻库 | future work |",
        "| 最高连板高度精确梯队 | cache 无 height 字段；`prev_limit_relimit_rate` 只给晋级率 | missing |",
        "| 涨停溢价率（昨日涨停今日收益） | 可事后算，但作当日门控需用 **昨日** 溢价，本轮未单列 G 组 | 可后续 |",
        "| UP_NUM / FAIL_RATIO 字段名 | Hermes 表里就没有这两个名字，只能从 `daily_limit_events` 或已有 cache 构造 | 已构造 |",
        "",
        "## 覆盖与缺失",
        "",
        f"- market_context 行数 {len(ctx)}，日期 {ctx['trade_date'].min().date()} → {ctx['trade_date'].max().date()}",
        f"- limit_down_context 行数 {len(down)}，日期 {down['trade_date'].min().date()} → {down['trade_date'].max().date()}",
        f"- market_context 列缺失率：{ {k: round(v, 4) for k, v in ctx_missing.items()} }",
        "- 末端日期 `market_touched_limit_count=0`、封成率 NaN 已见 2026-07-24 样本；分位计算 `min_periods=40`，NaN 当日乘数按 1.0 不编造。",
        "",
        "## 使用纪律",
        "",
        "- 只乘在 `allocation_multiplier` / 入场门 / 退出覆盖上，不改 fs00263 score。",
        "- Very Good 环境不加杠杆到 >30% NAV。",
        "- 结果叫 historical development，不是 final OOS。",
        "",
    ]
    dest = ensure_round2_tree() / "02_research_notes.md"
    assert_project_result_path(dest)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest
