"""Stage 0：v5 Round7 5bp parity。失败不得晋级。不写 round22 / round22_1 / round21* 账本。"""

from __future__ import annotations

from typing import Any

from ..paths import (
    ROUND21_1_ROOT,
    ROUND21_2_ROOT,
    ROUND21_3_ROOT,
    ROUND21_4_ROOT,
    ROUND21_ROOT,
    ROUND22_1_ROOT,
    ROUND22_ROOT,
    assert_project_result_path,
)
from ..round21.contract import N_TRADING_OBSERVATIONS_ANCHOR
from .common import close_enough, dump_json, load_json, write_md
from .contract import (
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    E0_CODE,
    FEE_SCHEDULE,
    MD,
    NAV_END_ABS_TOL,
    O01_FORMULA,
    PARITY_METRIC_ABS,
    ROUND22_2_ROOT,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    exp_dir,
    overlayed_path,
)

PARITY_JSON = ROUND22_2_ROOT / "parity" / "parity.json"


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and PARITY_JSON.is_file() and MD["parity"].is_file():
        existing = load_json(PARITY_JSON)
        if existing.get("status") in {"PASS", "FAIL"} and existing.get("observed"):
            return existing
    summary_path = ROUND22_2_ROOT / "parity" / "R222_V5_R16_O01" / "replay_summary.json"
    if resume and summary_path.is_file():
        packed = load_json(summary_path)
        return _finish_parity(packed)

    r21 = load_json(ROUND21_ROOT / "parity" / "parity.json")
    r211 = load_json(ROUND21_1_ROOT / "parity" / "parity.json")
    r212 = load_json(ROUND21_2_ROOT / "parity" / "parity.json")
    r213 = load_json(ROUND21_3_ROOT / "parity" / "parity.json")
    r214 = load_json(ROUND21_4_ROOT / "parity" / "parity.json")
    r22 = load_json(ROUND22_ROOT / "parity" / "parity.json")
    r221 = load_json(ROUND22_1_ROOT / "parity" / "parity.json")
    from ..round15.runner import apply_top2_veto
    from ..round16_5.engine import o01_signal_dates
    from ..round17.backtest import cached_overlay_v4
    from ..round20.parity import _pack, _run_r7

    ov4 = cached_overlay_v4()
    dates = o01_signal_dates()
    ov5 = apply_top2_veto(ov4, dates)
    ov_path = overlayed_path("R222_V5_R16_O01")
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(ov_path)
    ov5.to_parquet(ov_path, index=False)
    dest = ROUND22_2_ROOT / "parity" / "R222_V5_R16_O01"
    ran = _run_r7(
        dest,
        ov5,
        extra={
            "code": "R222_V5_R16_O01",
            "round": "22.2",
            "role": "exact_v5_replay_round7_caliber",
            "parent": "R9_B03",
            "formula": O01_FORMULA,
            "overlay": "cached_overlay_v4 → o01_signal_dates → apply_top2_veto",
            "current": CURRENT_BASELINE,
            "alias": CURRENT_ALIAS,
            "fee_schedule": FEE_SCHEDULE,
            "validation_hooks": None,
            "r21_parity_status": r21.get("status"),
            "r21_1_parity_status": r211.get("status"),
            "r21_2_parity_status": r212.get("status"),
            "r21_3_parity_status": r213.get("status"),
            "r21_4_parity_status": r214.get("status"),
            "r22_parity_status": r22.get("status"),
            "r22_1_parity_status": r221.get("status"),
        },
    )
    packed = _pack(ran, dest)
    packed["code"] = "baseline_v5"
    packed["alias"] = CURRENT_ALIAS
    packed["n_signal_dates"] = int(len(dates))
    packed["formula"] = O01_FORMULA
    dump_json(dest / "replay_summary.json", packed)

    e0 = exp_dir(E0_CODE)
    e0.mkdir(parents=True, exist_ok=True)
    import shutil

    for name in ("report.json", "nav.parquet", "trades.parquet", "replay_summary.json"):
        sp = dest / name
        if not sp.is_file() or sp.stat().st_size < 8:
            continue
        shutil.copy2(sp, e0 / name)

    return _finish_parity(packed)


def _finish_parity(packed: dict[str, Any]) -> dict[str, Any]:
    obs = {
        "cagr": packed.get("cagr"),
        "sharpe": packed.get("sharpe"),
        "max_drawdown": packed.get("max_drawdown"),
        "completed_trades": packed.get("completed_trades"),
        "n_obs": packed.get("n_trading_observations") or packed.get("n_obs"),
        "nav_end": packed.get("nav_end"),
    }
    passed = (
        close_enough(obs.get("cagr"), V5_ANCHOR["cagr"], abs_tol=PARITY_METRIC_ABS)
        and close_enough(obs.get("sharpe"), V5_ANCHOR["sharpe"], abs_tol=PARITY_METRIC_ABS)
        and close_enough(obs.get("max_drawdown"), V5_ANCHOR["max_drawdown"], abs_tol=PARITY_METRIC_ABS)
        and int(obs.get("completed_trades") or -1) == int(V5_ANCHOR["completed_trades"])
        and close_enough(obs.get("nav_end"), V5_NAV_END_EXACT, abs_tol=NAV_END_ABS_TOL)
    )
    payload = {
        "status": "PASS" if passed else "FAIL",
        "passed": bool(passed),
        "observed": obs,
        "anchor": dict(V5_ANCHOR),
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
        "n_obs_anchor": N_TRADING_OBSERVATIONS_ANCHOR,
        "formal_promotion_allowed": bool(passed),
        "current_baseline_changed": False,
        "auto_promote": False,
        "round": "22.2",
        "code": "R222_V5_R16_O01",
    }
    dump_json(PARITY_JSON, payload)
    lines = [
        "# Round22.2 · 00 v5 parity",
        "",
        f"status=`{payload['status']}` fee=`{FEE_SCHEDULE}`",
        "",
        f"- observed CAGR={obs.get('cagr')} Sharpe={obs.get('sharpe')} MaxDD={obs.get('max_drawdown')} trades={obs.get('completed_trades')} NAV={obs.get('nav_end')}",
        f"- anchor CAGR={V5_ANCHOR['cagr']} Sharpe={V5_ANCHOR['sharpe']} MaxDD={V5_ANCHOR['max_drawdown']} trades={V5_ANCHOR['completed_trades']} NAV={V5_NAV_END_EXACT}",
        "",
        "CURRENT_BASELINE_CHANGED=FALSE。不自动升 v6。不覆盖 result/round22 或 result/round22_1。",
        "decision=`D_close` order=`D+1_open` same_open_execution_optimism=True。",
        "",
    ]
    write_md(MD["parity"], "\n".join(lines))
    return payload
