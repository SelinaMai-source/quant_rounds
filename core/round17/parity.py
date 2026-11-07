"""R17_A00 精确重放 v4（HISTORICAL_FEE_CORRECTED）。失败 STOP。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import ROUND16_5_ROOT, ROUND17_ROOT
from ..round11.parity import compare_ledgers
from .backtest import cached_overlay_v4, run_canonical
from .common import dump_json, load_json, observed_from_report, write_md
from .contract import A00, PARITY_METRIC_ABS, V4_ANCHOR, parity_dir

PARITY_MD = ROUND17_ROOT / "01_v4_parity.md"
PARITY_JSON = ROUND17_ROOT / "parity" / "parity.json"


def load_parity_status() -> dict[str, Any]:
    return load_json(PARITY_JSON)


def metrics_close(obs: dict[str, Any], target: dict[str, Any], *, abs_tol: float) -> bool:
    try:
        ok = (
            abs(float(obs.get("cagr") or 0) - float(target["cagr"])) < abs_tol
            and abs(float(obs.get("sharpe") or 0) - float(target["sharpe"])) < abs_tol
            and abs(float(obs.get("max_drawdown") or 0) - float(target["max_drawdown"])) < abs_tol
        )
        if "completed_trades" in target and target["completed_trades"] is not None:
            ok = ok and int(obs.get("completed_trades") or -1) == int(target["completed_trades"])
        return bool(ok)
    except (TypeError, ValueError, KeyError):
        return False


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and PARITY_JSON.is_file():
        existing = load_json(PARITY_JSON)
        if existing.get("passed"):
            return existing
    ov4 = cached_overlay_v4()
    dest = parity_dir(A00)
    ran = run_canonical(
        dest,
        ov4,
        extra={"code": A00, "role": "exact_v4_replay", "fee_schedule": "HISTORICAL_FEE_CORRECTED"},
        resume=resume,
        write=True,
    )
    obs = ran.get("observed") or observed_from_report(ran.get("report") or {})
    metric_ok = metrics_close(obs, V4_ANCHOR, abs_tol=PARITY_METRIC_ABS)
    r16 = ROUND16_5_ROOT / "parity" / "R16_5_V00_BASELINE_V4"
    ledger = {"passed": False, "status": "MISSING_ANCHOR"}
    if (r16 / "nav.parquet").is_file() and (dest / "nav.parquet").is_file():
        ledger = compare_ledgers(
            pd.read_parquet(dest / "trades.parquet"),
            pd.read_parquet(r16 / "trades.parquet"),
            pd.read_parquet(dest / "nav.parquet"),
            pd.read_parquet(r16 / "nav.parquet"),
            target_n_trades=int(V4_ANCHOR["completed_trades"]),
        )
    passed = bool(metric_ok and ledger.get("passed"))
    payload = {
        "status": "PASS" if passed else "REPRODUCIBILITY_FAIL_OR_MISSING",
        "passed": passed,
        "code": A00,
        "observed": obs,
        "anchor": V4_ANCHOR,
        "metric_ok": metric_ok,
        "abs_tol": PARITY_METRIC_ABS,
        "ledger_vs_r16_5_v00": ledger,
        "note": "A00 original 1.0/1.0 must match R9_B03 fee-corrected numbers.",
    }
    dump_json(PARITY_JSON, payload)
    lines = [
        "# 01 v4 parity（R17_A00）",
        "",
        f"status=`{payload['status']}` metric_ok=`{metric_ok}` ledger=`{ledger.get('passed')}`",
        "",
        f"- observed CAGR=`{obs.get('cagr')}` Sharpe=`{obs.get('sharpe')}` MaxDD=`{obs.get('max_drawdown')}` trades=`{obs.get('completed_trades')}`",
        f"- anchor CAGR=`{V4_ANCHOR['cagr']}` Sharpe=`{V4_ANCHOR['sharpe']}` MaxDD=`{V4_ANCHOR['max_drawdown']}` trades=`{V4_ANCHOR['completed_trades']}`",
        "",
        "失败则 STOP，不得进入搜索。",
    ]
    write_md(PARITY_MD, "\n".join(lines))
    if not passed:
        raise RuntimeError(f"REPRODUCIBILITY_FAIL_OR_MISSING STOP A00={payload['status']} obs={obs}")
    return payload
