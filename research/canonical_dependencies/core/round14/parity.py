"""R14_00 精确重放 v4。parity 失败必须 STOP。本 stage 在 freeze 之前。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import ROUND14_ROOT
from ..round11.parity import compare_ledgers
from ..round12.runner import hist_hooks
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import (
    PARITY_ABS_NAV,
    PARITY_METRIC_ABS,
    R12_01_DIR,
    R12_01_TARGETS,
    R13_00_DIR,
    R14_00,
    parity_dir,
)
from .runner import build_version_overlayed, load_market, run_r14_backtest

PARITY_MD = ROUND14_ROOT / "00_anchor_parity.md"
PARITY_JSON = ROUND14_ROOT / "parity" / "parity.json"


def _metrics_ok(observed: dict[str, Any], targets: dict[str, Any]) -> bool:
    try:
        return (
            abs(float(observed.get("cagr") or 0) - float(targets["cagr"])) < PARITY_METRIC_ABS
            and abs(float(observed.get("sharpe") or 0) - float(targets["sharpe"])) < PARITY_METRIC_ABS
            and abs(float(observed.get("max_drawdown") or 0) - float(targets["max_drawdown"])) < PARITY_METRIC_ABS
            and int(observed.get("completed_trades") or -1) == int(targets["completed_trades"])
        )
    except (TypeError, ValueError, KeyError):
        return False


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    ov4, _, _ = build_version_overlayed("v4")
    dest = parity_dir(R14_00)
    loaded = load_market()
    ran = run_r14_backtest(
        dest,
        overlayed=ov4,
        validation_hooks=hist_hooks(),
        extra_report={
            "parity_anchor": str(R12_01_DIR),
            "fee_schedule": "HISTORICAL_FEE_CORRECTED",
            "code": R14_00,
        },
        resume=resume,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    local_trades = pd.read_parquet(dest / "trades.parquet")
    local_nav = pd.read_parquet(dest / "nav.parquet")
    anchor_trades = pd.read_parquet(R12_01_DIR / "trades.parquet")
    anchor_nav = pd.read_parquet(R12_01_DIR / "nav.parquet")
    ledger = compare_ledgers(
        local_trades,
        anchor_trades,
        local_nav,
        anchor_nav,
        target_n_trades=int(R12_01_TARGETS["completed_trades"]),
    )
    report = json.loads((dest / "report.json").read_text(encoding="utf-8"))
    observed = observed_from_report(report)
    metric_ok = _metrics_ok(observed, R12_01_TARGETS)
    r13_ok = None
    r13_ledger = None
    if R13_00_DIR.is_dir() and (R13_00_DIR / "nav.parquet").is_file():
        r13_trades = pd.read_parquet(R13_00_DIR / "trades.parquet")
        r13_nav = pd.read_parquet(R13_00_DIR / "nav.parquet")
        r13_ledger = compare_ledgers(
            local_trades,
            r13_trades,
            local_nav,
            r13_nav,
            target_n_trades=int(R12_01_TARGETS["completed_trades"]),
        )
        r13_ok = bool(r13_ledger.get("passed"))
    passed = bool(ledger["passed"] and metric_ok)
    item = {
        "code": R14_00,
        "anchor_dir": str(R12_01_DIR),
        "r13_anchor_dir": str(R13_00_DIR),
        "out_dir": str(dest),
        "status": "PASS" if passed else "REPRODUCIBILITY_FAIL",
        "passed": passed,
        "targets": dict(R12_01_TARGETS),
        "observed": observed,
        "metric_ok": metric_ok,
        "ledger": {k: ledger.get(k) for k in ("passed", "nav_max_abs", "n_local", "n_anchor", "n_mismatch")},
        "r13_ledger_passed": r13_ok,
        "r13_ledger": {k: (r13_ledger or {}).get(k) for k in ("passed", "nav_max_abs", "n_mismatch")} if r13_ledger else None,
        "run": {k: ran.get(k) for k in ("status", "code", "out_dir")},
        "fee_schedule": "HISTORICAL_FEE_CORRECTED",
        "nav_path": str(dest / "nav.parquet"),
    }
    payload = {
        "status": "PASS" if passed else "REPRODUCIBILITY_FAIL",
        "passed": passed,
        "fail_code": None if passed else "REPRODUCIBILITY_FAIL",
        "nav_abs_tol": PARITY_ABS_NAV,
        "metric_abs_tol": PARITY_METRIC_ABS,
        "anchors": {R14_00: item},
        "note": "parity 失败必须 STOP，不得为过关改策略或改容差。",
    }
    dump_json(PARITY_JSON, payload)
    lines = [
        "# Round 14 · 锚点 parity",
        "",
        f"结果：**{payload['status']}**。指标容差 `{PARITY_METRIC_ABS:g}`，NAV abs≤`{PARITY_ABS_NAV:g}`。",
        "",
        f"| ID | 锚点 | 目标 CAGR | 观察 CAGR | 目标 Sharpe | 观察 Sharpe | 目标笔数 | 观察笔数 | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
        f"| `{R14_00}` | `{R12_01_DIR}` | {pct(R12_01_TARGETS.get('cagr'))} | {pct(observed.get('cagr'))} | "
        f"{float(R12_01_TARGETS.get('sharpe') or 0):.6f} | {float(observed.get('sharpe') or 0):.6f} | "
        f"{R12_01_TARGETS.get('completed_trades')} | {observed.get('completed_trades')} | {item['status']} |",
        "",
        f"R13_00 ledger passed = `{r13_ok}`。失败 = `REPRODUCIBILITY_FAIL`，停止后续 stage。",
        "",
    ]
    write_md(PARITY_MD, "\n".join(lines))
    if not passed:
        raise RuntimeError("REPRODUCIBILITY_FAIL STOP")
    return payload


def load_parity_status() -> dict[str, Any]:
    if not PARITY_JSON.is_file():
        return {"passed": False, "status": "missing"}
    return load_json(PARITY_JSON)


__all__ = ["load_parity_status", "run_parity"]
