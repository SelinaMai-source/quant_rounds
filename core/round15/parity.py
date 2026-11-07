"""R15_00 精确重放 v4。parity 失败必须 STOP。本 stage 在 freeze 之前。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import ROUND15_ROOT
from ..round11.parity import compare_ledgers
from ..round12.runner import hist_hooks
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import (
    PARITY_ABS_NAV,
    PARITY_METRIC_ABS,
    R12_01_DIR,
    R12_01_TARGETS,
    R13_00_DIR,
    R14_00_DIR,
    R15_00,
    parity_dir,
)
from .runner import build_version_overlayed, load_market, run_r15_backtest

PARITY_MD = ROUND15_ROOT / "00_anchor_parity.md"
PARITY_JSON = ROUND15_ROOT / "parity" / "parity.json"


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


def _ledger_against(local_trades, local_nav, anchor_dir, n_trades: int):
    if not anchor_dir.is_dir() or not (anchor_dir / "nav.parquet").is_file():
        return None
    a_trades = pd.read_parquet(anchor_dir / "trades.parquet")
    a_nav = pd.read_parquet(anchor_dir / "nav.parquet")
    return compare_ledgers(
        local_trades,
        a_trades,
        local_nav,
        a_nav,
        target_n_trades=n_trades,
    )


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    ov4, _, _ = build_version_overlayed("v4")
    dest = parity_dir(R15_00)
    loaded = load_market()
    ran = run_r15_backtest(
        dest,
        overlayed=ov4,
        validation_hooks=hist_hooks(),
        extra_report={
            "parity_anchor": str(R12_01_DIR),
            "fee_schedule": "HISTORICAL_FEE_CORRECTED",
            "code": R15_00,
        },
        resume=resume,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    local_trades = pd.read_parquet(dest / "trades.parquet")
    local_nav = pd.read_parquet(dest / "nav.parquet")
    n_trades = int(R12_01_TARGETS["completed_trades"])
    ledger = _ledger_against(local_trades, local_nav, R12_01_DIR, n_trades) or {"passed": False}
    r13 = _ledger_against(local_trades, local_nav, R13_00_DIR, n_trades)
    r14 = _ledger_against(local_trades, local_nav, R14_00_DIR, n_trades)
    report = json.loads((dest / "report.json").read_text(encoding="utf-8"))
    observed = observed_from_report(report)
    metric_ok = _metrics_ok(observed, R12_01_TARGETS)
    passed = bool(ledger.get("passed") and metric_ok)
    item = {
        "code": R15_00,
        "anchor_dir": str(R12_01_DIR),
        "r13_anchor_dir": str(R13_00_DIR),
        "r14_anchor_dir": str(R14_00_DIR),
        "out_dir": str(dest),
        "status": "PASS" if passed else "REPRODUCIBILITY_FAIL",
        "passed": passed,
        "targets": dict(R12_01_TARGETS),
        "observed": observed,
        "metric_ok": metric_ok,
        "ledger": {k: ledger.get(k) for k in ("passed", "nav_max_abs", "n_local", "n_anchor", "n_mismatch")},
        "r13_ledger_passed": None if r13 is None else bool(r13.get("passed")),
        "r14_ledger_passed": None if r14 is None else bool(r14.get("passed")),
        "run": {k: ran.get(k) for k in ("status", "code", "out_dir")},
        "fee_schedule": "HISTORICAL_FEE_CORRECTED",
        "nav_path": str(dest / "nav.parquet"),
        "trades_path": str(dest / "trades.parquet"),
    }
    payload = {
        "status": "PASS" if passed else "REPRODUCIBILITY_FAIL",
        "passed": passed,
        "fail_code": None if passed else "REPRODUCIBILITY_FAIL",
        "nav_abs_tol": PARITY_ABS_NAV,
        "metric_abs_tol": PARITY_METRIC_ABS,
        "anchors": {R15_00: item},
        "note": "parity 失败必须 STOP，不得为过关改策略或改容差。",
    }
    dump_json(PARITY_JSON, payload)
    lines = [
        "# Round 15 · 锚点 parity",
        "",
        f"结果：**{payload['status']}**。指标容差 `{PARITY_METRIC_ABS:g}`，NAV abs≤`{PARITY_ABS_NAV:g}`。",
        "",
        "| ID | 锚点 | 目标 CAGR | 观察 CAGR | 目标 Sharpe | 观察 Sharpe | 目标笔数 | 观察笔数 | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
        f"| `{R15_00}` | `{R12_01_DIR}` | {pct(R12_01_TARGETS.get('cagr'))} | {pct(observed.get('cagr'))} | "
        f"{float(R12_01_TARGETS.get('sharpe') or 0):.6f} | {float(observed.get('sharpe') or 0):.6f} | "
        f"{R12_01_TARGETS.get('completed_trades')} | {observed.get('completed_trades')} | {item['status']} |",
        "",
        f"R13_00 ledger passed = `{item['r13_ledger_passed']}`。R14_00 ledger passed = `{item['r14_ledger_passed']}`。",
        "失败 = `REPRODUCIBILITY_FAIL`，停止后续 stage。",
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
