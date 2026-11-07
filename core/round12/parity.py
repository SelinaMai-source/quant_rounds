"""R12_00 legacy v4 与 R12_01 corrected-cost 精确 parity。过不去停。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import ROUND12_ROOT
from ..round11.parity import compare_ledgers
from ..stock_readonly import load_prepared_market
from .common import dump_json, observed_from_report, pct, write_md
from .contract import (
    PARITY_ABS_NAV,
    PARITY_METRIC_ABS,
    R11_V4_CORRECTED_DIR,
    R12_00,
    R12_01,
    R12_01_TARGETS,
    V4_DIR,
    V4_TARGETS,
    parity_dir,
)
from .freeze import require_frozen
from .runner import hist_hooks, run_r12_backtest, v4_spec

PARITY_MD = ROUND12_ROOT / "01_anchor_parity.md"
PARITY_JSON = ROUND12_ROOT / "parity" / "parity.json"


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


def _one(
    code: str,
    anchor_dir,
    targets: dict[str, Any],
    *,
    prepared_market,
    market_meta,
    resume: bool,
    hooks: dict[str, Any] | None,
    fee: str,
) -> dict[str, Any]:
    dest = parity_dir(code)
    ran = run_r12_backtest(
        dest,
        prepared_market=prepared_market,
        market_meta=market_meta,
        validation_hooks=hooks,
        extra_report={"fee_schedule": fee, "parity_anchor": str(anchor_dir)},
        resume=resume,
    )
    local_trades = pd.read_parquet(dest / "trades.parquet")
    local_nav = pd.read_parquet(dest / "nav.parquet")
    anchor_trades = pd.read_parquet(anchor_dir / "trades.parquet")
    anchor_nav = pd.read_parquet(anchor_dir / "nav.parquet")
    ledger = compare_ledgers(
        local_trades,
        anchor_trades,
        local_nav,
        anchor_nav,
        target_n_trades=int(targets["completed_trades"]),
    )
    report = json.loads((dest / "report.json").read_text(encoding="utf-8"))
    observed = observed_from_report(report)
    metric_ok = _metrics_ok(observed, targets)
    passed = bool(ledger["passed"] and metric_ok)
    return {
        "code": code,
        "anchor_dir": str(anchor_dir),
        "out_dir": str(dest),
        "status": "PASS" if passed else "REPRODUCIBILITY_FAIL",
        "passed": passed,
        "targets": dict(targets),
        "observed": observed,
        "metric_ok": metric_ok,
        "ledger": ledger,
        "run": {k: ran.get(k) for k in ("status", "code", "out_dir")},
        "fee_schedule": fee,
    }


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    loaded = load_prepared_market(warmup_ma5=True)
    items = {
        "R12_00": _one(
            R12_00,
            V4_DIR,
            V4_TARGETS,
            prepared_market=loaded["prepared"],
            market_meta=loaded,
            resume=resume,
            hooks=None,
            fee="LEGACY_FLAT_5BP",
        ),
        "R12_01": _one(
            R12_01,
            R11_V4_CORRECTED_DIR,
            R12_01_TARGETS,
            prepared_market=loaded["prepared"],
            market_meta=loaded,
            resume=resume,
            hooks=hist_hooks(),
            fee="HISTORICAL_FEE_CORRECTED",
        ),
    }
    all_pass = all(v.get("passed") for v in items.values())
    payload = {
        "status": "PASS" if all_pass else "REPRODUCIBILITY_FAIL",
        "passed": all_pass,
        "fail_code": None if all_pass else "REPRODUCIBILITY_FAIL",
        "nav_abs_tol": PARITY_ABS_NAV,
        "metric_abs_tol": PARITY_METRIC_ABS,
        "spec": v4_spec().code,
        "anchors": items,
        "note": "R12_01 是 REALISM_REFERENCE，不是 v5。",
    }
    dump_json(PARITY_JSON, payload)
    lines = [
        "# Round 12 · 锚点 parity",
        "",
        f"结果：**{payload['status']}**。容差指标 `{PARITY_METRIC_ABS:g}`，NAV abs≤`{PARITY_ABS_NAV:g}`。",
        "复现只写 `result/round12/parity/`，不覆盖 legacy / Round11。",
        "",
        "| ID | 锚点 | 目标 CAGR | 观察 CAGR | 目标 Sharpe | 观察 Sharpe | 目标笔数 | 观察笔数 | NAV max abs | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, item in items.items():
        tgt = item["targets"]
        obs = item["observed"]
        ledger = item["ledger"]
        lines.append(
            f"| `{item['code']}` | `{item['anchor_dir']}` | {pct(tgt.get('cagr'))} | {pct(obs.get('cagr'))} | "
            f"{float(tgt.get('sharpe') or 0):.6f} | {float(obs.get('sharpe') or 0):.6f} | "
            f"{tgt.get('completed_trades')} | {obs.get('completed_trades')} | "
            f"{ledger.get('nav_max_abs')} | {item['status']} |"
        )
    lines += [
        "",
        f"- R12_00 必须对齐 legacy `{v4_spec().code}`。",
        "- R12_01 必须对齐 `R11_V4_CORRECTED`，是 REALISM_REFERENCE，**不是 v5**。",
        "- 失败 = `REPRODUCIBILITY_FAIL`，停止后续 stage，不为过关改策略。",
        "",
    ]
    write_md(PARITY_MD, "\n".join(lines))
    if not all_pass:
        raise RuntimeError("REPRODUCIBILITY_FAIL STOP")
    return payload


def load_parity_status() -> dict[str, Any]:
    from .common import load_json

    if not PARITY_JSON.is_file():
        return {"passed": False, "status": "missing"}
    return load_json(PARITY_JSON)


__all__ = ["load_parity_status", "run_parity"]
