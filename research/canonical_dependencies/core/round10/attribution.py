"""A01–A04 完整回测 + 结构归因。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND10_ROOT, assert_project_result_path
from .contract import PARENT_CODE, R10_00, TARGET_CAGR, TARGET_MAXDD, TARGET_SHARPE
from .daily import build_daily_frame, persist_daily
from .registry import specs_for_stage
from .runner import load_summary, run_codes


def _pct(x, digits=2, signed=True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    txt = f"{float(x):.{digits}%}"
    if signed and float(x) > 0:
        txt = "+" + txt
    return txt.replace("-", "−")


def _sh(x) -> str:
    return "—" if x is None else f"{float(x):.3f}"


def run_attribution(*, resume: bool = True, jobs: int = 1, experiment: str = "") -> dict[str, Any]:
    codes = [c.strip() for c in experiment.split(",") if c.strip()] if experiment else [s.code for s in specs_for_stage("attribution")]
    daily = build_daily_frame()
    persist_daily(daily)
    bt = run_codes(codes, resume=resume, jobs=jobs)
    v4 = load_summary(R10_00) or {}
    rows = []
    for code in [R10_00, *codes]:
        s = load_summary(code) or {}
        rows.append({
            "code": code,
            "cagr": s.get("cagr"),
            "sharpe": s.get("sharpe"),
            "max_drawdown": s.get("max_drawdown"),
            "loyo_median": s.get("loyo_median"),
            "loyo_worst": s.get("loyo_worst"),
            "completed_trades": s.get("completed_trades"),
            "top2_pnl": s.get("top2_pnl"),
            "d_cagr": (None if s.get("cagr") is None else float(s["cagr"]) - float(v4.get("cagr") or TARGET_CAGR)),
            "d_sharpe": (None if s.get("sharpe") is None else float(s["sharpe"]) - float(v4.get("sharpe") or TARGET_SHARPE)),
        })
    a01 = next((r for r in rows if r["code"] == "R10_A01"), {})
    a02 = next((r for r in rows if r["code"] == "R10_A02"), {})
    a03 = next((r for r in rows if r["code"] == "R10_A03"), {})
    a04 = next((r for r in rows if r["code"] == "R10_A04"), {})

    # A03 把 0→5：若明显变差 → avoid-bad 是主因
    # A04 把 10→5：若明显变差 → preserve-good 是主因
    # A01/A02 若都明显弱于 v4 → 交互
    def _drop(row):
        return None if row.get("d_cagr") is None else float(row["d_cagr"])

    story = []
    if _drop(a03) is not None and _drop(a03) < -0.005:
        story.append("avoid-bad")
    if _drop(a04) is not None and _drop(a04) < -0.005:
        story.append("preserve-good")
    if _drop(a01) is not None and _drop(a02) is not None and _drop(a01) < -0.003 and _drop(a02) < -0.003:
        story.append("interaction")
    if not story:
        if _drop(a03) is not None and abs(_drop(a03)) <= 0.005 and (_drop(a04) or 0) < -0.005:
            story.append("preserve-good")
        elif _drop(a04) is not None and abs(_drop(a04)) <= 0.005 and (_drop(a03) or 0) < -0.005:
            story.append("avoid-bad")
        else:
            story.append("mixed / interaction")
    verdict = " + ".join(story)

    cell = []
    for agree in ("A", "B", "C"):
        for sep in ("LOW", "MID", "HIGH"):
            block = daily.loc[daily["agreement"].eq(agree) & daily["sep_bucket"].eq(sep)]
            ret = pd.to_numeric(block.get("top2_ret"), errors="coerce")
            cell.append({
                "agreement": agree, "sep": sep, "n": int(len(block)),
                "mean_top2_ret": float(ret.mean()) if ret.notna().any() else None,
                "win": float((ret > 0).mean()) if ret.notna().any() else None,
                "large_loss": float((ret <= -0.08).mean()) if ret.notna().any() else None,
            })
    payload = {
        "parent": PARENT_CODE,
        "verdict": verdict,
        "rows": rows,
        "cells": cell,
        "note": "A03 测 avoid-bad；A04 测 preserve-good；A01/A02 测单因子是否够。Δ 只对 v4。",
        "jobs": {k: v.get("status") for k, v in (bt.get("jobs") or {}).items()},
        "failed": bt.get("failed") or [],
    }
    path = ROUND10_ROOT / "metrics" / "attribution.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    lines = [
        "| ID | CAGR | ΔCAGR | Sharpe | ΔSharpe | MaxDD | LOYO med | LOYO worst | n | Top2 PnL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        top2 = "—" if r.get("top2_pnl") is None else f"{float(r['top2_pnl']):,.0f}"
        lines.append(
            f"| `{r['code']}` | {_pct(r.get('cagr'))} | {_pct(r.get('d_cagr'))} | {_sh(r.get('sharpe'))} | "
            f"{_sh(r.get('d_sharpe'))} | {_pct(r.get('max_drawdown'))} | {_pct(r.get('loyo_median'))} | "
            f"{_pct(r.get('loyo_worst'))} | {r.get('completed_trades') or '—'} | {top2} |"
        )
    cell_lines = [
        "| agree | sep | n | mean Top2 ret | win | large-loss |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in cell:
        cell_lines.append(
            f"| {c['agreement']} | {c['sep']} | {c['n']} | {_pct(c.get('mean_top2_ret'), 3)} | "
            f"{_pct(c.get('win'), signed=False)} | {_pct(c.get('large_loss'), signed=False)} |"
        )
    md = ROUND10_ROOT / "01_v4_structural_attribution.md"
    assert_project_result_path(md)
    md.write_text(
        f"""# 01 — baseline_v4 结构归因

Parent = **baseline_v4 = {PARENT_CODE}**。对照 CAGR {_pct(v4.get('cagr') or TARGET_CAGR)} / Sharpe {_sh(v4.get('sharpe') or TARGET_SHARPE)} / MaxDD {_pct(v4.get('max_drawdown') or TARGET_MAXDD)}。
正式账本 `local_executor` → Stock `fixed_sleeve`。Δ 只对 v4。

## 结论

**{verdict}**

- `R10_A01` 去掉 separation，只留 agreement。
- `R10_A02` 去掉 agreement，只留 separation。
- `R10_A03` 把 B03 的 0% 角改成 5%：若变差，说明增量主要来自 **avoid-bad**。
- `R10_A04` 把 10% 高置信改成 5%：若变差，说明增量主要来自 **preserve-good**。
- 若 A01 与 A02 都明显弱于 v4，则 **agreement × separation 交互** 不可拆。

## 正式结果

{chr(10).join(lines)}

## Agreement × Separation 单元格（v4 Top2 票）

{chr(10).join(cell_lines)}
""",
        encoding="utf-8",
    )
    return payload


__all__ = ["run_attribution"]
