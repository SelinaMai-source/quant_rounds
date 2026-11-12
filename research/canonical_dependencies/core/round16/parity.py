"""R16_00 精确重放 baseline_v4。失败 STOP。锚点 = R15_00 精确值。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import ROUND16_ROOT
from ..round11.parity import compare_ledgers
from ..round12.runner import hist_hooks
from ..round15.runner import build_version_overlayed, load_market, run_r15_backtest
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import (
    PARITY_ABS_NAV,
    PARITY_METRIC_ABS,
    R12_01_DIR,
    R13_00_DIR,
    R14_00_DIR,
    R15_00_DIR,
    R16_00,
    V4_ANCHOR,
    parity_dir,
)

PARITY_MD = ROUND16_ROOT / "00_anchor_parity.md"
PARITY_JSON = ROUND16_ROOT / "parity" / "parity.json"


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
    if resume and PARITY_JSON.is_file():
        existing = load_json(PARITY_JSON)
        if existing.get("passed"):
            return existing
    ov4, _, _ = build_version_overlayed("v4")
    dest = parity_dir(R16_00)
    loaded = load_market()
    ran = run_r15_backtest(
        dest,
        overlayed=ov4,
        validation_hooks=hist_hooks(),
        extra_report={
            "parity_anchor": str(R15_00_DIR),
            "fee_schedule": "HISTORICAL_FEE_CORRECTED",
            "code": R16_00,
            "round": 16,
        },
        resume=resume,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    local_trades = pd.read_parquet(dest / "trades.parquet")
    local_nav = pd.read_parquet(dest / "nav.parquet")
    n_trades = int(V4_ANCHOR["completed_trades"])
    ledger15 = _ledger_against(local_trades, local_nav, R15_00_DIR, n_trades) or {"passed": False}
    ledger12 = _ledger_against(local_trades, local_nav, R12_01_DIR, n_trades)
    r13 = _ledger_against(local_trades, local_nav, R13_00_DIR, n_trades)
    r14 = _ledger_against(local_trades, local_nav, R14_00_DIR, n_trades)
    report = json.loads((dest / "report.json").read_text(encoding="utf-8"))
    observed = observed_from_report(report)
    metric_ok = _metrics_ok(observed, V4_ANCHOR)
    passed = bool(ledger15.get("passed") and metric_ok)
    item = {
        "code": R16_00,
        "anchor_dir": str(R15_00_DIR),
        "out_dir": str(dest),
        "status": "PASS" if passed else "REPRODUCIBILITY_FAIL",
        "passed": passed,
        "targets": dict(V4_ANCHOR),
        "observed": observed,
        "metric_ok": metric_ok,
        "ledger_r15": {k: ledger15.get(k) for k in ("passed", "nav_max_abs", "n_local", "n_anchor", "n_mismatch")},
        "r12_ledger_passed": None if ledger12 is None else bool(ledger12.get("passed")),
        "r13_ledger_passed": None if r13 is None else bool(r13.get("passed")),
        "r14_ledger_passed": None if r14 is None else bool(r14.get("passed")),
        "run": {k: ran.get(k) for k in ("status", "code", "out_dir")},
        "fee_schedule": "HISTORICAL_FEE_CORRECTED",
    }
    payload = {
        "status": "PASS" if passed else "REPRODUCIBILITY_FAIL",
        "passed": passed,
        "fail_code": None if passed else "REPRODUCIBILITY_FAIL",
        "nav_abs_tol": PARITY_ABS_NAV,
        "metric_abs_tol": PARITY_METRIC_ABS,
        "anchors": {R16_00: item},
        "note": "parity 失败必须 STOP。锚点为 Round15 R15_00 精确值。",
    }
    dump_json(PARITY_JSON, payload)
    write_md(
        PARITY_MD,
        "\n".join([
            "# Round 16 · 锚点 parity",
            "",
            f"结果：**{payload['status']}**。",
            f"目标 CAGR=`{V4_ANCHOR['cagr']}` Sharpe=`{V4_ANCHOR['sharpe']}` MaxDD=`{V4_ANCHOR['max_drawdown']}`。",
            f"观察 CAGR=`{observed.get('cagr')}` Sharpe=`{observed.get('sharpe')}` MaxDD=`{observed.get('max_drawdown')}` 笔数=`{observed.get('completed_trades')}`。",
            f"R15 ledger passed=`{ledger15.get('passed')}` R12=`{item['r12_ledger_passed']}` R13=`{item['r13_ledger_passed']}` R14=`{item['r14_ledger_passed']}`。",
            "",
        ]),
    )
    if not passed:
        raise RuntimeError("REPRODUCIBILITY_FAIL STOP")
    return payload


def load_parity_status() -> dict[str, Any]:
    if not PARITY_JSON.is_file():
        return {"passed": False, "status": "missing"}
    return load_json(PARITY_JSON)
