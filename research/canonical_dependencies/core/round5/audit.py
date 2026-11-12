"""B：Hermes 表审计 → 02_factor_data_audit.md。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from ..paths import (
    CANDIDATE_PANEL_RELATIVE,
    MARKET_END,
    PREDICTION_START,
    SNAPSHOT_RELATIVE,
    STOCK_ROOT,
    assert_project_result_path,
    ensure_round5_tree,
    stock_path,
)
from ..stock_readonly import ensure_stock_src_on_path
from .factors import EXCLUDE, FAMILY_COLUMNS, NO_NEW_DEPENDENCY, build_factor_panel, coverage_table
from .registry import LOCKED_FEATURES

TABLES = (
    "daily_raw",
    "daily_post_adjusted",
    "daily_money_flow",
    "daily_limit_events",
    "daily_intraday_factors",
    "daily_valuation",
    "industry_membership",
    "trade_calendar",
    "universe_membership",
)


def _sample_table(name: str) -> dict[str, Any]:
    ensure_stock_src_on_path()
    from zyquant.data import DataSnapshot

    snap = DataSnapshot(stock_path(SNAPSHOT_RELATIVE), verify_hashes=False)
    start = PREDICTION_START.replace(year=2018)
    end = min(MARKET_END, start.replace(year=2018, month=3, day=31))
    try:
        frame = snap.table(name, start=start, end=end, cutoff=snap.metadata.as_of_date)
    except TypeError:
        frame = snap.table(name, start=start, end=end, cutoff=snap.metadata.as_of_date, fields=None)
    cols = list(frame.columns)
    n = int(len(frame))
    return {
        "table": name,
        "sample_rows": n,
        "n_cols": len(cols),
        "columns": cols[:40],
        "has_trade_date": "trade_date" in cols,
        "has_instrument": "instrument_id" in cols,
        "pit_note": "snapshot cutoff = as_of_date；正式特征只用 D 收盘可见字段。",
    }


def write_audit(*, resume: bool = True) -> dict[str, Any]:
    dest = ensure_round5_tree() / "02_audit"
    dest.mkdir(parents=True, exist_ok=True)
    js = dest / "audit.json"
    if resume and js.is_file() and (ensure_round5_tree() / "02_factor_data_audit.md").is_file():
        return json.loads(js.read_text(encoding="utf-8"))

    panel = build_factor_panel(resume=resume)
    cov = coverage_table(panel)
    tables = []
    for name in TABLES:
        try:
            tables.append(_sample_table(name))
        except Exception as exc:
            tables.append({"table": name, "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "written_at": datetime.now().astimezone().isoformat(),
        "snapshot": str(stock_path(SNAPSHOT_RELATIVE)),
        "stock_root": str(STOCK_ROOT),
        "candidate_panel": str(stock_path(CANDIDATE_PANEL_RELATIVE)),
        "panel_rows": int(len(panel)),
        "panel_cols": int(panel.shape[1]),
        "locked_features": list(LOCKED_FEATURES),
        "exclude": sorted(EXCLUDE),
        "no_new_dependency": sorted(NO_NEW_DEPENDENCY),
        "families": {k: list(v) for k, v in FAMILY_COLUMNS.items()},
        "coverage": cov.to_dict("records"),
        "tables": tables,
        "pit_rules": [
            "标签成熟：actual_exit_date（或 label_end_date）必须 < prediction_date。",
            "fb_touch_count EXCLUDE，除非 AmazingData 分钟重建。",
            "g_prim_* NO_NEW_DEPENDENCY。",
            "Walk-forward：满 3 个完整年 → 下一年；新因子覆盖晚则推迟最早预测年。",
        ],
    }
    assert_project_result_path(js)
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        "# 02 因子数据审计",
        "",
        f"- 生成时间：{payload['written_at']}",
        f"- 快照：`{payload['snapshot']}`（只读）",
        f"- 候选面板：`{payload['candidate_panel']}`",
        f"- 面板行数：{payload['panel_rows']}，列数：{payload['panel_cols']}",
        "",
        "## PIT / 排除",
        "",
    ]
    for rule in payload["pit_rules"]:
        lines.append(f"- {rule}")
    lines += [
        "",
        "## 已用锁 6 因子",
        "",
        ", ".join(f"`{c}`" for c in LOCKED_FEATURES),
        "",
        "## 覆盖率",
        "",
        "| 因子 | 族 | 覆盖 | 首年 | 末年 | 旗标 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["coverage"]:
        flags = []
        if row.get("exclude"):
            flags.append("EXCLUDE")
        if row.get("no_new_dependency"):
            flags.append("NO_NEW_DEPENDENCY")
        if not row.get("present"):
            flags.append("MISSING")
        lines.append(
            f"| `{row['factor']}` | {row.get('family')} | {float(row.get('coverage') or 0):.1%} "
            f"| {row.get('first_year') or '—'} | {row.get('last_year') or '—'} | {','.join(flags) or '—'} |"
        )
    lines += ["", "## Hermes 表（2018Q1 抽样）", "", "| 表 | 行 | 列数 | 有日期 | 有标的 |", "|---|---:|---:|:---:|:---:|"]
    for t in tables:
        if t.get("error"):
            lines.append(f"| `{t['table']}` | — | — | — | {t['error']} |")
        else:
            lines.append(
                f"| `{t['table']}` | {t.get('sample_rows')} | {t.get('n_cols')} "
                f"| {'是' if t.get('has_trade_date') else '否'} | {'是' if t.get('has_instrument') else '否'} |"
            )
    lines += [
        "",
        "## 候选族",
        "",
    ]
    for fam, cols in FAMILY_COLUMNS.items():
        lines.append(f"- **{fam}**：{', '.join(f'`{c}`' for c in cols)}")
    lines.append("")
    md = ensure_round5_tree() / "02_factor_data_audit.md"
    assert_project_result_path(md)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
