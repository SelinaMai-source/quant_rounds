"""V00/V01 精确重放。失败 STOP。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import ROUND16_5_ROOT
from ..round11.parity import compare_ledgers
from ..round15.runner import apply_top2_veto
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import (
    O01_ANCHOR,
    PARITY_ABS_NAV,
    PARITY_METRIC_ABS,
    R15_00_DIR,
    R16_00_DIR,
    R16_O01_DIR,
    V00,
    V01,
    V4_ANCHOR,
    overlayed_path,
    parity_dir,
)
from .engine import (
    cached_overlay_v4,
    metrics_close,
    o01_signal_dates,
    run_engine,
    save_overlay,
    with_rank,
)

PARITY_MD = ROUND16_5_ROOT / "00_parity.md"
PARITY_JSON = ROUND16_5_ROOT / "parity" / "parity.json"


def _ledger(local_trades, local_nav, anchor_dir, n_trades: int):
    if not anchor_dir.is_dir() or not (anchor_dir / "nav.parquet").is_file():
        return {"passed": False, "status": "MISSING_ARTIFACT", "path": str(anchor_dir)}
    a_trades = pd.read_parquet(anchor_dir / "trades.parquet")
    a_nav = pd.read_parquet(anchor_dir / "nav.parquet")
    rec = compare_ledgers(
        local_trades,
        a_trades,
        local_nav,
        a_nav,
        target_n_trades=n_trades,
    )
    rec["path"] = str(anchor_dir)
    return rec


def _identity_check(local: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, Any]:
    lt = with_rank(local)
    at = with_rank(anchor)
    keys = ["signal_date", "instrument_id"]
    l2 = lt.loc[lt["_rank"].le(2), keys + ["_rank", "allocation_multiplier", "target_name_budget"]].copy()
    a2 = at.loc[at["_rank"].le(2), keys + ["_rank", "allocation_multiplier", "target_name_budget"]].copy()
    merged = l2.merge(a2, on=keys + ["_rank"], how="outer", suffixes=("_loc", "_anc"), indicator=True)
    both = merged["_merge"].eq("both")
    n_both = int(both.sum())
    n_only = int((~both).sum())
    alloc_ok = True
    budget_ok = True
    if n_both:
        alloc_ok = bool(
            (pd.to_numeric(merged.loc[both, "allocation_multiplier_loc"], errors="coerce")
             - pd.to_numeric(merged.loc[both, "allocation_multiplier_anc"], errors="coerce"))
            .abs()
            .fillna(1)
            .le(1e-12)
            .all()
        )
        budget_ok = bool(
            (pd.to_numeric(merged.loc[both, "target_name_budget_loc"], errors="coerce")
             - pd.to_numeric(merged.loc[both, "target_name_budget_anc"], errors="coerce"))
            .abs()
            .fillna(1)
            .le(1e-4)
            .all()
        )
    return {
        "n_local_top2": int(len(l2)),
        "n_anchor_top2": int(len(a2)),
        "n_matched": n_both,
        "n_identity_mismatch": n_only,
        "alloc_ok": alloc_ok,
        "budget_ok": budget_ok,
        "passed": bool(n_only == 0 and alloc_ok and budget_ok),
    }


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and PARITY_JSON.is_file():
        existing = load_json(PARITY_JSON)
        if existing.get("passed"):
            return existing
    ov4 = cached_overlay_v4()
    save_overlay(V00, ov4)
    dest0 = parity_dir(V00)
    ran0 = run_engine(
        dest0,
        ov4,
        extra={"code": V00, "round": "16.5", "fee_schedule": "HISTORICAL_FEE_CORRECTED", "role": "V00"},
        resume=resume,
        write=True,
    )
    t0 = pd.read_parquet(dest0 / "trades.parquet")
    n0 = pd.read_parquet(dest0 / "nav.parquet")
    obs0 = observed_from_report(json.loads((dest0 / "report.json").read_text(encoding="utf-8")))
    metric0 = metrics_close(obs0, V4_ANCHOR, abs_tol=PARITY_METRIC_ABS)
    led16 = _ledger(t0, n0, R16_00_DIR, int(V4_ANCHOR["completed_trades"]))
    led15 = _ledger(t0, n0, R15_00_DIR, int(V4_ANCHOR["completed_trades"]))
    ident0 = None
    if (R16_00_DIR / "trades.parquet").is_file():
        ident0 = _identity_check(t0, pd.read_parquet(R16_00_DIR / "trades.parquet"))
    v00_pass = bool(metric0 and led16.get("passed"))

    dates = o01_signal_dates()
    ov1 = apply_top2_veto(ov4, dates)
    save_overlay(V01, ov1)
    dest1 = parity_dir(V01)
    ran1 = run_engine(
        dest1,
        ov1,
        extra={
            "code": V01,
            "round": "16.5",
            "fee_schedule": "HISTORICAL_FEE_CORRECTED",
            "role": "V01",
            "n_signal_dates": int(len(dates)),
            "signal": "PUT_SKEW25 past-only 90/252/126 shift(1)",
        },
        resume=resume,
        write=True,
    )
    t1 = pd.read_parquet(dest1 / "trades.parquet")
    n1 = pd.read_parquet(dest1 / "nav.parquet")
    obs1 = observed_from_report(json.loads((dest1 / "report.json").read_text(encoding="utf-8")))
    metric1 = metrics_close(obs1, O01_ANCHOR, abs_tol=PARITY_METRIC_ABS)
    led_o01 = _ledger(t1, n1, R16_O01_DIR, int(O01_ANCHOR["completed_trades"]))
    ident1 = None
    if (R16_O01_DIR / "trades.parquet").is_file():
        ident1 = _identity_check(t1, pd.read_parquet(R16_O01_DIR / "trades.parquet"))
    v01_pass = bool(metric1 and led_o01.get("passed"))
    passed = bool(v00_pass and v01_pass)
    item0 = {
        "code": V00,
        "status": "PASS" if v00_pass else "REPRODUCIBILITY_FAIL",
        "passed": v00_pass,
        "targets": dict(V4_ANCHOR),
        "observed": obs0,
        "metric_ok": metric0,
        "ledger_r16": {k: led16.get(k) for k in ("passed", "nav_max_abs", "n_local", "n_anchor", "n_mismatch", "status")},
        "ledger_r15": {k: led15.get(k) for k in ("passed", "nav_max_abs", "n_local", "n_anchor", "n_mismatch", "status")},
        "identity": ident0,
        "out_dir": str(dest0),
        "run_status": ran0.get("status"),
    }
    item1 = {
        "code": V01,
        "status": "PASS" if v01_pass else "REPRODUCIBILITY_FAIL",
        "passed": v01_pass,
        "targets": dict(O01_ANCHOR),
        "observed": obs1,
        "metric_ok": metric1,
        "n_signal_dates": int(len(dates)),
        "ledger_r16_o01": {k: led_o01.get(k) for k in ("passed", "nav_max_abs", "n_local", "n_anchor", "n_mismatch", "status")},
        "identity": ident1,
        "out_dir": str(dest1),
        "run_status": ran1.get("status"),
    }
    payload = {
        "status": "PASS" if passed else "REPRODUCIBILITY_FAIL",
        "passed": passed,
        "fail_code": None if passed else "REPRODUCIBILITY_FAIL",
        "nav_abs_tol": PARITY_ABS_NAV,
        "metric_abs_tol": PARITY_METRIC_ABS,
        "anchors": {V00: item0, V01: item1},
        "note": "parity 失败必须 STOP，不做统计。",
    }
    dump_json(PARITY_JSON, payload)
    write_md(
        PARITY_MD,
        "\n".join([
            "# Round 16.5 · 00 parity",
            "",
            f"结果：**{payload['status']}**。",
            "",
            f"- V00 目标 CAGR=`{V4_ANCHOR['cagr']}` Sharpe=`{V4_ANCHOR['sharpe']}` MaxDD=`{V4_ANCHOR['max_drawdown']}` 笔数=`{V4_ANCHOR['completed_trades']}`",
            f"- V00 观察 CAGR=`{obs0.get('cagr')}` Sharpe=`{obs0.get('sharpe')}` MaxDD=`{obs0.get('max_drawdown')}` 笔数=`{obs0.get('completed_trades')}` metric_ok=`{metric0}` R16 ledger=`{led16.get('passed')}`",
            f"- V01 目标 CAGR=`{O01_ANCHOR['cagr']}` Sharpe=`{O01_ANCHOR['sharpe']}` MaxDD=`{O01_ANCHOR['max_drawdown']}` 笔数=`{O01_ANCHOR['completed_trades']}`",
            f"- V01 观察 CAGR=`{obs1.get('cagr')}` Sharpe=`{obs1.get('sharpe')}` MaxDD=`{obs1.get('max_drawdown')}` 笔数=`{obs1.get('completed_trades')}` metric_ok=`{metric1}` R16 O01 ledger=`{led_o01.get('passed')}`",
            f"- O01 信号日（含 2015+）=`{len(dates)}`",
            "",
            "失败则 STOP。",
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
