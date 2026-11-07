"""容量：复用 Round10 artifact；统一补跑 v1–v4 的 30万/1M/3M（无 ADV 帽）。"""

from __future__ import annotations

import json
from typing import Any

from ..paths import ROUND10_ROOT, ROUND11_ROOT
from ..replay_round7.inject import inject_round7_into_overlay_caches
from ..stock_readonly import load_prepared_market
from .common import dump_json, pct, stage_dir, write_md
from .contract import BASELINES
from .runner import run_spec_to

CAPACITY_MD = ROUND11_ROOT / "11_capacity.md"
CAPITALS = (300_000.0, 1_000_000.0, 3_000_000.0)


def _round10_capacity() -> dict[str, Any]:
    md = ROUND10_ROOT / "07_capacity_liquidity.md"
    rows = []
    for code in ("R10_G01_1m_p3", "R10_G02_3m_p3", "R10_G03_10m_p3", "R10_G04_1m_p1", "R10_G05_1m_p5"):
        path = ROUND10_ROOT / code / "summary.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        "source": str(md),
        "exists": md.is_file(),
        "verdict": "ROBUST_UP_TO_1M" if md.is_file() else "MISSING",
        "summaries": rows,
        "note": "Round10 G 组带日成交额 participation 帽，是 diagnostic，不是本轮无帽 1M/3M。",
    }


def run_capacity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = stage_dir("capacity")
    reused = _round10_capacity()
    inject_round7_into_overlay_caches()
    loaded = load_prepared_market(warmup_ma5=True)
    unified: dict[str, Any] = {}
    for key, spec in BASELINES.items():
        for cash in CAPITALS:
            tag = f"R11_{key.upper()}_CAP_{int(cash)}"
            extra = {
                "fee_schedule": "LEGACY_FLAT_5BP",
                "initial_cash": cash,
                "capacity_note": "无 ADV 帽；只改 initial_cash / sleeve_cash。30万=契约默认。",
            }
            if abs(cash - 300_000.0) < 1e-6:
                extra["reuses_legacy_if_identical"] = True
            ran = run_spec_to(
                spec,
                dest / tag,
                prepared_market=loaded["prepared"],
                market_meta=loaded,
                resume=resume,
                initial_cash=cash,
                extra_report=extra,
            )
            m = (ran.get("report") or {}).get("metrics") or {}
            unified[tag] = {
                "version": key,
                "initial_cash": cash,
                "cagr": m.get("annualized_return"),
                "sharpe": m.get("sharpe_zero_rate"),
                "max_drawdown": m.get("max_drawdown"),
                "completed_trades": m.get("completed_trades"),
            }
    payload = {
        "status": "DONE",
        "round10_reused": reused,
        "unified": unified,
        "post_hoc_diagnostic": True,
    }
    dump_json(dest / "capacity.json", payload)
    lines = [
        "# Round 11 · 容量",
        "",
        f"Round10 artifact 复用：`{reused.get('verdict')}`。{reused.get('note')}",
        "",
        "## 统一无帽 30万 / 1M / 3M（legacy 5bp）",
        "",
        "| 实验 | 资本 | CAGR | Sharpe | MaxDD | 笔数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag, row in unified.items():
        lines.append(
            f"| `{tag}` | {int(row['initial_cash'])} | {pct(row.get('cagr'))} | "
            f"{row.get('sharpe')} | {pct(row.get('max_drawdown'))} | {row.get('completed_trades')} |"
        )
    write_md(CAPACITY_MD, "\n".join(lines))
    return payload


__all__ = ["run_capacity"]
