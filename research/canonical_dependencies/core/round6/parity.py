"""门禁 0：R6_00_baseline_v3 必须与 result/round4/R4_B05 逐笔一致。不过就停。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..metrics import CAGR_ABS_TOL, METRIC_VERSION, SHARPE_ABS_TOL
from ..paths import ROUND4_ROOT, ROUND6_ROOT, assert_project_result_path, ensure_round6_tree
from ..round5.parity import compare_ledgers
from .registry import (
    BASELINE_V3,
    BASELINE_V3_ALIAS,
    V3_CAGR,
    V3_MDD,
    V3_SHARPE,
    V3_TRADES,
    get_spec,
)
from .runner import run_spec


def _write(path: Path, text: str) -> Path:
    assert_project_result_path(path)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _load_r4() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    root = ROUND4_ROOT / BASELINE_V3
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    trades = pd.read_parquet(root / "trades.parquet")
    nav = pd.read_parquet(root / "nav.parquet")
    return report, trades, nav


def write_parity_markdown(
    *,
    predictions=None,
    prepared=None,
    meta=None,
    resume: bool = True,
    run_now: bool = True,
) -> dict[str, Any]:
    ensure_round6_tree()
    spec = get_spec(BASELINE_V3_ALIAS)
    ran = run_spec(
        spec, predictions=predictions, prepared_market=prepared, market_meta=meta,
        resume=resume, run_now=run_now,
    )
    r4_report, r4_trades, r4_nav = _load_r4()
    r6_dir = ROUND6_ROOT / BASELINE_V3_ALIAS
    r6_report = json.loads((r6_dir / "report.json").read_text(encoding="utf-8")) if (r6_dir / "report.json").is_file() else {}
    r6_trades = pd.read_parquet(r6_dir / "trades.parquet") if (r6_dir / "trades.parquet").is_file() else pd.DataFrame()
    r6_nav = pd.read_parquet(r6_dir / "nav.parquet") if (r6_dir / "nav.parquet").is_file() else pd.DataFrame()

    cmp = compare_ledgers(r6_trades, r4_trades, r6_nav, r4_nav)
    r4m = r4_report.get("metrics") or {}
    r6m = r6_report.get("metrics") or {}
    cagr_ok = r6m.get("annualized_return") is not None and abs(
        float(r6m["annualized_return"]) - float(r4m.get("annualized_return") or V3_CAGR)
    ) <= CAGR_ABS_TOL
    sharpe_ok = r6m.get("sharpe_zero_rate") is not None and abs(
        float(r6m["sharpe_zero_rate"]) - float(r4m.get("sharpe_zero_rate") or V3_SHARPE)
    ) <= SHARPE_ABS_TOL
    mdd_ok = r6m.get("max_drawdown") is not None and abs(
        float(r6m["max_drawdown"]) - float(r4m.get("max_drawdown") or V3_MDD)
    ) <= 1e-5
    trades_ok = int(r6m.get("completed_trades") or 0) == int(r4m.get("completed_trades") or V3_TRADES)
    display_ok = bool(cagr_ok and sharpe_ok and mdd_ok and trades_ok)
    passed = bool(cmp["passed"] and display_ok)

    dest = ROUND6_ROOT / "00_parity"
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_at": datetime.now().astimezone().isoformat(),
        "passed": passed,
        "metric_version": METRIC_VERSION,
        "r6_status": ran.get("status"),
        "compare": cmp,
        "display": {
            "cagr_ok": cagr_ok, "sharpe_ok": sharpe_ok, "mdd_ok": mdd_ok, "trades_ok": trades_ok,
            "r6": {
                "cagr": r6m.get("annualized_return"),
                "sharpe": r6m.get("sharpe_zero_rate"),
                "max_drawdown": r6m.get("max_drawdown"),
                "completed_trades": r6m.get("completed_trades"),
                "cagr_ex2024": r6m.get("cagr_ex2024"),
            },
            "r4": {
                "cagr": r4m.get("annualized_return"),
                "sharpe": r4m.get("sharpe_zero_rate"),
                "max_drawdown": r4m.get("max_drawdown"),
                "completed_trades": r4m.get("completed_trades"),
                "cagr_ex2024": r4m.get("cagr_ex2024"),
            },
        },
        "target": "23.86 / 1.454 / −14.28 / 877",
        "note": "复用锁分 fs00263 + R4_B05 overlay，不重训。",
    }
    assert_project_result_path(dest / "parity.json")
    (dest / "parity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )

    mismatch_txt = "无" if not cmp["mismatches"] else json.dumps(cmp["mismatches"][:8], ensure_ascii=False, default=str)
    md = f"""# 00 baseline_v3 逐笔对齐

- 生成时间：{payload['written_at']}
- 对照：`result/round4/R4_B05/` ↔ `result/round6/R6_00_baseline_v3/`
- `metric_version`：`{METRIC_VERSION}`
- 分数：锁分 fs00263 五种子集成（不重训）+ R4_B05 overlay
- 结论：**{'通过' if passed else '未通过，停'}**

## 指标

| | R4_B05 | R6_00 | 对齐 |
|---|---:|---:|:---:|
| CAGR | {_pct(r4m.get('annualized_return'))} | {_pct(r6m.get('annualized_return'))} | {'是' if cagr_ok else '否'} |
| Sharpe | {float(r4m.get('sharpe_zero_rate') or 0):.3f} | {float(r6m.get('sharpe_zero_rate') or 0):.3f} | {'是' if sharpe_ok else '否'} |
| MaxDD | {_pct(r4m.get('max_drawdown'))} | {_pct(r6m.get('max_drawdown'))} | {'是' if mdd_ok else '否'} |
| 去2024 | {_pct(r4m.get('cagr_ex2024'))} | {_pct(r6m.get('cagr_ex2024'))} | — |
| 笔数 | {r4m.get('completed_trades')} | {r6m.get('completed_trades')} | {'是' if trades_ok else '否'} |

目标复现：23.86 / 1.454 / −14.28 / 877。

## 逐笔

| 项 | 结果 |
|---|---|
| 笔数 | R6={cmp['n_r5']} R4={cmp['n_r4']} {'对齐' if cmp['n_match'] else '不对'} |
| 买卖日 / 标的 / 股数 | {'对齐' if cmp['trade_identity_ok'] else '不对'} |
| 费用 | {'对齐' if cmp['fee_ok'] else '不对'} |
| 日 NAV | {'对齐' if cmp['nav_ok'] else '不对'}（max abs={cmp.get('nav_max_abs')}） |

不一致样本：{mismatch_txt}

不过就停，不准进入 forensics / 训练。
"""
    _write(ROUND6_ROOT / "00_baseline_v3_parity.md", md)
    return payload


def load_parity_status() -> dict[str, Any]:
    path = ROUND6_ROOT / "00_parity" / "parity.json"
    if not path.is_file():
        return {"passed": False, "reason": "parity 未写"}
    return json.loads(path.read_text(encoding="utf-8"))
