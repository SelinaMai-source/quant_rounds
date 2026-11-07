"""a-replay：核对 C0；精确重放 O01 为 R19_A_C1。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...paths import ROUND16_5_ROOT
from ...round11.parity import compare_ledgers
from ...round16_5.engine import metrics_close
from ..common import dump_json, write_md
from ..contract import O01_ANCHOR, PARITY_METRIC_ABS, V4_ANCHOR
from ..parity import _identity_check, _overlay_identity
from .engine import o01_signal_dates, run_veto
from .support import (
    A_C0,
    A_C1,
    R16_5_V01_DIR,
    REPLAY_JSON,
    load_nav_trades,
    overlayed_path,
    pack_obs_from_dir,
    parity_dir,
    require_c0,
    resume_ok,
    skip_result,
    slim_obs,
    ta_md,
    ymd_list,
)

ROUND16_5_V01_OVERLAY = ROUND16_5_ROOT / "weights" / "overlayed_R16_5_V01_O01_EXACT.parquet"


def run_replay(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    existing = resume_ok(REPLAY_JSON) if resume else None
    if existing is not None:
        return existing
    try:
        c0 = require_c0()
    except FileNotFoundError:
        payload = skip_result("a-replay", "C0_MISSING", status="BLOCKED")
        dump_json(REPLAY_JSON, payload)
        return payload
    c0_obs = pack_obs_from_dir(c0)
    c0_ok = metrics_close(c0_obs, V4_ANCHOR, abs_tol=PARITY_METRIC_ABS)
    try:
        dates = o01_signal_dates()
    except Exception as exc:
        payload = skip_result("a-replay", f"O01_SIGNAL_UNAVAILABLE:{type(exc).__name__}:{exc}")
        payload["c0"] = {"passed": c0_ok, "observed": c0_obs}
        dump_json(REPLAY_JSON, payload)
        return payload
    if dates is None or len(dates) == 0:
        payload = skip_result("a-replay", "O01_SIGNAL_EMPTY")
        payload["c0"] = {"passed": c0_ok, "observed": c0_obs}
        dump_json(REPLAY_JSON, payload)
        return payload

    dest = parity_dir(A_C1)
    ran = run_veto(
        dest,
        dates,
        extra={
            "code": A_C1,
            "round": 19,
            "role": "exact_o01",
            "fee_schedule": "HISTORICAL_FEE_CORRECTED",
            "formula": "PUT_SKEW25.shift(1) > rolling(252,min_periods=126).quantile(0.90) → apply_top2_veto",
        },
        resume=resume,
        write=True,
        save_code=A_C1,
    )
    obs = ran.get("observed") or pack_obs_from_dir(dest)
    trades, nav = load_nav_trades(dest)
    if nav is None:
        nav = ran.get("nav")
    if trades is None:
        trades = ran.get("trades")
    metric_ok = metrics_close(obs, O01_ANCHOR, abs_tol=PARITY_METRIC_ABS)
    ledger = {"passed": False, "status": "MISSING_ANCHOR"}
    ident = None
    if (R16_5_V01_DIR / "nav.parquet").is_file() and nav is not None and trades is not None:
        ledger = compare_ledgers(
            trades,
            pd.read_parquet(R16_5_V01_DIR / "trades.parquet"),
            nav,
            pd.read_parquet(R16_5_V01_DIR / "nav.parquet"),
            target_n_trades=int(O01_ANCHOR["completed_trades"]),
        )
        ledger["status"] = "PASS" if ledger.get("passed") else "FAIL"
        ident = _identity_check(trades, pd.read_parquet(R16_5_V01_DIR / "trades.parquet"))
    ov_ident = _overlay_identity(pd.read_parquet(overlayed_path(A_C1)), ROUND16_5_V01_OVERLAY)
    c1_passed = bool(metric_ok and ledger.get("passed") and (ident or {}).get("passed", True))
    payload = {
        "status": "DONE",
        "c0": {
            "code": A_C0,
            "passed": bool(c0_ok),
            "observed": slim_obs(c0_obs),
            "anchor": dict(V4_ANCHOR),
            "out_dir": str(c0),
        },
        "c1": {
            "code": A_C1,
            "passed": c1_passed,
            "metric_ok": metric_ok,
            "observed": slim_obs(obs),
            "anchor": dict(O01_ANCHOR),
            "n_signal_dates": int(len(dates)),
            "signal_dates_head": ymd_list(dates)[:8],
            "ledger_vs_r16_5_v01": {
                k: ledger.get(k)
                for k in ("passed", "status", "nav_max_abs", "trade_rows_local", "trade_rows_anchor")
            },
            "identity_top1_top2": ident,
            "identity_overlay": ov_ident,
            "out_dir": str(dest),
            "overlay_path": str(overlayed_path(A_C1)),
        },
        "exact_parity": bool(c0_ok and c1_passed),
        "n_trading_observations_c0": c0_obs.get("n_trading_observations"),
    }
    dump_json(REPLAY_JSON, payload)
    write_md(
        ta_md("replay.md"),
        "\n".join(
            [
                "# Round19 Track A · a-replay",
                "",
                f"C0 passed=`{c0_ok}` CAGR=`{c0_obs.get('cagr')}` Sharpe=`{c0_obs.get('sharpe')}` trades=`{c0_obs.get('completed_trades')}`。",
                f"C1 passed=`{c1_passed}` CAGR=`{obs.get('cagr')}` Sharpe=`{obs.get('sharpe')}` MaxDD=`{obs.get('max_drawdown')}` trades=`{obs.get('completed_trades')}`。",
                f"n_signal=`{len(dates)}` ledger_vs_R16.5_V01=`{ledger.get('passed')}` exact_parity=`{payload['exact_parity']}`。",
                "",
            ]
        ),
    )
    return payload
