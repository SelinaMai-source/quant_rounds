"""Stage 0：复用 R21 parity 对照，本轮再跑 R21_1_E0 / A00 锁 v5 Round7 5bp。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_1_ROOT, ROUND21_ROOT
from ..round21.contract import FORMAL_PROMOTION_BLOCKED, N_TRADING_OBSERVATIONS_ANCHOR
from .common import close_enough, dump_json, load_json, write_md
from .contract import (
    A00,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    E0_CODE,
    FEE_SCHEDULE,
    NAV_END_ABS_TOL,
    O01_FORMULA,
    PARITY_MD,
    PARITY_METRIC_ABS,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    exp_dir,
    overlayed_path,
)

PARITY_JSON = ROUND21_1_ROOT / "parity" / "parity.json"


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and PARITY_JSON.is_file() and PARITY_MD.is_file():
        existing = load_json(PARITY_JSON)
        if existing.get("status") in {"PASS", "FAIL"} and existing.get("observed"):
            return existing

    r21 = load_json(ROUND21_ROOT / "parity" / "parity.json")
    from ..round15.runner import apply_top2_veto
    from ..round16_5.engine import o01_signal_dates
    from ..round17.backtest import cached_overlay_v4
    from ..round20.parity import _pack, _run_r7
    from ..paths import assert_project_result_path

    ov4 = cached_overlay_v4()
    dates = o01_signal_dates()
    ov5 = apply_top2_veto(ov4, dates)
    ov_path = overlayed_path("R21_1_V5_R16_O01")
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(ov_path)
    ov5.to_parquet(ov_path, index=False)
    dest = ROUND21_1_ROOT / "parity" / "R21_1_V5_R16_O01"
    ran = _run_r7(
        dest,
        ov5,
        extra={
            "code": "R21_1_V5_R16_O01",
            "round": "21.1",
            "role": "exact_v5_replay_round7_caliber",
            "parent": "R9_B03",
            "formula": O01_FORMULA,
            "overlay": "cached_overlay_v4 → o01_signal_dates → apply_top2_veto",
            "current": CURRENT_BASELINE,
            "alias": CURRENT_ALIAS,
            "r21_parity_status": r21.get("status"),
        },
    )
    packed = _pack(ran, dest)
    packed["code"] = "baseline_v5"
    packed["alias"] = CURRENT_ALIAS
    packed["n_signal_dates"] = int(len(dates))
    packed["formula"] = O01_FORMULA
    dump_json(dest / "replay_summary.json", packed)

    a00 = exp_dir(A00)
    a00.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    for name in ("report.json", "nav.parquet", "trades.parquet", "replay_summary.json"):
        sp = dest / name
        if not sp.is_file():
            continue
        tp = a00 / name
        if name.endswith(".parquet"):
            pd.read_parquet(sp).to_parquet(tp, index=False)
        else:
            tp.write_bytes(sp.read_bytes())

    checks = {
        "cagr": close_enough(packed.get("cagr"), V5_ANCHOR["cagr"], abs_tol=PARITY_METRIC_ABS),
        "sharpe": close_enough(packed.get("sharpe"), V5_ANCHOR["sharpe"], abs_tol=PARITY_METRIC_ABS),
        "max_drawdown": close_enough(packed.get("max_drawdown"), V5_ANCHOR["max_drawdown"], abs_tol=PARITY_METRIC_ABS),
        "completed_trades": int(packed.get("completed_trades") or -1) == int(V5_ANCHOR["completed_trades"]),
        "n_obs": int(packed.get("n_trading_observations") or -1) == int(N_TRADING_OBSERVATIONS_ANCHOR),
    }
    nav_ok = close_enough(packed.get("nav_end"), V5_NAV_END_EXACT, abs_tol=NAV_END_ABS_TOL)
    passed = bool(all(checks.values()) and nav_ok)
    status = "PASS" if passed else "FAIL"
    payload = {
        "status": status,
        "passed": passed,
        "fail_code": None if passed else FORMAL_PROMOTION_BLOCKED,
        "formal_promotion_blocked": not passed,
        "code": "R21_1_V5_R16_O01",
        "e0_code": E0_CODE,
        "current": CURRENT_BASELINE,
        "alias": CURRENT_ALIAS,
        "validation_hooks": None,
        "fee_schedule": FEE_SCHEDULE,
        "historical_stamp_tax": False,
        "observed": packed,
        "anchor": dict(V5_ANCHOR),
        "checks": checks,
        "nav_end_ok": nav_ok,
        "abs_tol": PARITY_METRIC_ABS,
        "nav_end_abs_tol": NAV_END_ABS_TOL,
        "n_signal_dates": int(len(dates)),
        "formula": O01_FORMULA,
        "out_dir": str(dest),
        "overlay_path": str(ov_path),
        "r21_parity_readonly": {
            "status": r21.get("status"),
            "passed": r21.get("passed"),
            "path": str(ROUND21_ROOT / "parity" / "parity.json"),
            "overwritten": False,
        },
        "note": "FAIL 只停止正式晋级。不覆盖 result/round21。E0 训练在 train stage。",
    }
    dump_json(PARITY_JSON, payload)
    dump_json(a00 / "summary.json", {**packed, "code": A00, "role": "exact_v5_replay", "status": "copied"})
    lines = [
        "# Round21.1 · 00 baseline_v5 parity（Round7 5bp）",
        "",
        f"结果：**{status}**。CURRENT=`{CURRENT_BASELINE}` = `{CURRENT_ALIAS}`。AUTO_PROMOTE=False。",
        "",
        f"- 目标 CAGR=`{V5_ANCHOR['cagr']}` Sharpe=`{V5_ANCHOR['sharpe']}` MaxDD=`{V5_ANCHOR['max_drawdown']}`",
        f"- trades=`{V5_ANCHOR['completed_trades']}` NAV_end=`{V5_NAV_END_EXACT}`",
        f"- 观测 CAGR=`{packed.get('cagr')}` Sharpe=`{packed.get('sharpe')}` MaxDD=`{packed.get('max_drawdown')}`",
        f"- trades=`{packed.get('completed_trades')}` NAV_end=`{packed.get('nav_end')}`",
        f"- checks={checks} nav_end_ok=`{nav_ok}`",
        f"- R21 parity 只读对照 status=`{r21.get('status')}` 未覆盖。",
        "",
    ]
    write_md(PARITY_MD, "\n".join(lines))
    return payload
