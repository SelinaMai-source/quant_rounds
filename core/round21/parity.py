"""重放 baseline_v5 = R16_O01，Round7 5bp 口径。复用 round20 / replay_r7_caliber 链。FAIL 只停正式晋级。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_ROOT, assert_project_result_path
from ..round20.parity import _pack, _run_r7
from .common import close_enough, dump_json, load_json, write_md
from .contract import (
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    FEE_SCHEDULE,
    FORMAL_PROMOTION_BLOCKED,
    NAV_END_ABS_TOL,
    N_TRADING_OBSERVATIONS_ANCHOR,
    O01_FORMULA,
    PARITY_MD,
    PARITY_METRIC_ABS,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    overlayed_path,
    parity_dir,
)

PARITY_JSON = ROUND21_ROOT / "parity" / "parity.json"


def load_parity_status() -> dict[str, Any]:
    return load_json(PARITY_JSON)


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and PARITY_JSON.is_file() and PARITY_MD.is_file():
        existing = load_json(PARITY_JSON)
        if existing.get("status") in {"PASS", "FAIL"} and existing.get("observed"):
            return existing

    from ..round15.runner import apply_top2_veto
    from ..round16_5.engine import o01_signal_dates
    from ..round17.backtest import cached_overlay_v4

    ov4 = cached_overlay_v4()
    dates = o01_signal_dates()
    ov5 = apply_top2_veto(ov4, dates)
    ov_path = overlayed_path("R21_V5_R16_O01")
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(ov_path)
    ov5.to_parquet(ov_path, index=False)
    dest = parity_dir("R21_V5_R16_O01")
    ran = _run_r7(
        dest,
        ov5,
        extra={
            "code": "R21_V5_R16_O01",
            "round": 21,
            "role": "exact_v5_replay_round7_caliber",
            "parent": "R9_B03",
            "formula": O01_FORMULA,
            "overlay": "cached_overlay_v4 → o01_signal_dates → apply_top2_veto",
            "current": CURRENT_BASELINE,
            "alias": CURRENT_ALIAS,
            "chain": "core.round20.parity._run_r7 / tools.replay_r7_caliber",
        },
    )
    packed = _pack(ran, dest)
    packed["code"] = "baseline_v5"
    packed["alias"] = CURRENT_ALIAS
    packed["n_signal_dates"] = int(len(dates))
    packed["formula"] = O01_FORMULA
    dump_json(dest / "replay_summary.json", packed)

    checks = {
        "cagr": close_enough(packed.get("cagr"), V5_ANCHOR["cagr"], abs_tol=PARITY_METRIC_ABS),
        "sharpe": close_enough(packed.get("sharpe"), V5_ANCHOR["sharpe"], abs_tol=PARITY_METRIC_ABS),
        "max_drawdown": close_enough(packed.get("max_drawdown"), V5_ANCHOR["max_drawdown"], abs_tol=PARITY_METRIC_ABS),
        "completed_trades": int(packed.get("completed_trades") or -1) == int(V5_ANCHOR["completed_trades"]),
        "n_obs": int(packed.get("n_trading_observations") or -1) == int(N_TRADING_OBSERVATIONS_ANCHOR),
    }
    nav_ok = close_enough(packed.get("nav_end"), V5_NAV_END_EXACT, abs_tol=NAV_END_ABS_TOL)
    passed = bool(all(checks.values()))
    status = "PASS" if passed else "FAIL"
    payload = {
        "status": status,
        "passed": passed,
        "fail_code": None if passed else FORMAL_PROMOTION_BLOCKED,
        "formal_promotion_blocked": not passed,
        "code": "R21_V5_R16_O01",
        "current": CURRENT_BASELINE,
        "alias": CURRENT_ALIAS,
        "engine": "cached_overlay_v4 → o01_signal_dates → apply_top2_veto → round7_fixed_sleeve_config + run_local_fixed_sleeve_backtest",
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
        "note": "FAIL 只停止正式晋级评价；仍写本 artifact。禁止改 O01/B03/90-10。",
    }
    dump_json(PARITY_JSON, payload)
    lines = [
        "# Round21 · 00 baseline_v5 parity（Round7 5bp 口径）",
        "",
        f"结果：**{status}**。CURRENT=`{CURRENT_BASELINE}` = `{CURRENT_ALIAS}` USER_LOCK。",
        "",
        "engine=`cached_overlay_v4` → `o01_signal_dates` → `apply_top2_veto` → `round7_fixed_sleeve_config` + `run_local_fixed_sleeve_backtest`。复用 `core.round20.parity` / `replay_r7_caliber`。**不传** `validation_hooks`。卖税 5bp。",
        "",
        "## 目标",
        "",
        f"- CAGR=`{V5_ANCHOR['cagr']}` Sharpe=`{V5_ANCHOR['sharpe']}` MaxDD=`{V5_ANCHOR['max_drawdown']}`",
        f"- trades=`{V5_ANCHOR['completed_trades']}` n_obs=`{V5_ANCHOR['n_obs']}` NAV_end=`{V5_ANCHOR['nav_end']}`",
        "",
        "## 观测",
        "",
        f"- CAGR=`{packed.get('cagr')}` Sharpe=`{packed.get('sharpe')}` MaxDD=`{packed.get('max_drawdown')}`",
        f"- trades=`{packed.get('completed_trades')}` n_obs=`{packed.get('n_trading_observations')}` NAV_end=`{packed.get('nav_end')}`",
        "",
        "## 检查",
        "",
        f"- cagr=`{checks['cagr']}` sharpe=`{checks['sharpe']}` max_drawdown=`{checks['max_drawdown']}`",
        f"- completed_trades=`{checks['completed_trades']}` n_obs=`{checks['n_obs']}` nav_end_ok=`{nav_ok}`",
        f"- abs_tol=`{PARITY_METRIC_ABS}`",
        "",
        f"FAIL → `{FORMAL_PROMOTION_BLOCKED}`（停止正式晋级评价，盘点/冻结仍继续）。",
        "",
        f"out=`{dest}` overlay=`{ov_path}`",
        "",
    ]
    write_md(PARITY_MD, "\n".join(lines))
    return payload
