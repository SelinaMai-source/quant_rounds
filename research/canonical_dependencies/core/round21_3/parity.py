"""Stage 0：v5 Round7 5bp parity。失败不得晋级。不写 R21/R21.1/R21.2 账本。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_1_ROOT, ROUND21_2_ROOT, ROUND21_3_ROOT, ROUND21_ROOT
from ..round21.contract import FORMAL_PROMOTION_BLOCKED, N_TRADING_OBSERVATIONS_ANCHOR
from .common import close_enough, dump_json, load_json, write_md
from .contract import (
    A00,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    E0_CODE,
    FEE_SCHEDULE,
    MD,
    NAV_END_ABS_TOL,
    O01_FORMULA,
    PARITY_METRIC_ABS,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    exp_dir,
    overlayed_path,
)

PARITY_JSON = ROUND21_3_ROOT / "parity" / "parity.json"


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and PARITY_JSON.is_file() and MD["parity"].is_file():
        existing = load_json(PARITY_JSON)
        if existing.get("status") in {"PASS", "FAIL"} and existing.get("observed"):
            return existing

    r21 = load_json(ROUND21_ROOT / "parity" / "parity.json")
    r211 = load_json(ROUND21_1_ROOT / "parity" / "parity.json")
    r212 = load_json(ROUND21_2_ROOT / "parity" / "parity.json")
    from ..round15.runner import apply_top2_veto
    from ..round16_5.engine import o01_signal_dates
    from ..round17.backtest import cached_overlay_v4
    from ..round20.parity import _pack, _run_r7
    from ..paths import assert_project_result_path

    ov4 = cached_overlay_v4()
    dates = o01_signal_dates()
    ov5 = apply_top2_veto(ov4, dates)
    ov_path = overlayed_path("R21_3_V5_R16_O01")
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(ov_path)
    ov5.to_parquet(ov_path, index=False)
    dest = ROUND21_3_ROOT / "parity" / "R21_3_V5_R16_O01"
    ran = _run_r7(
        dest,
        ov5,
        extra={
            "code": "R21_3_V5_R16_O01",
            "round": "21.3",
            "role": "exact_v5_replay_round7_caliber",
            "parent": "R9_B03",
            "formula": O01_FORMULA,
            "overlay": "cached_overlay_v4 → o01_signal_dates → apply_top2_veto",
            "current": CURRENT_BASELINE,
            "alias": CURRENT_ALIAS,
            "r21_parity_status": r21.get("status"),
            "r21_1_parity_status": r211.get("status"),
            "r21_2_parity_status": r212.get("status"),
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
    for name in ("report.json", "nav.parquet", "trades.parquet", "replay_summary.json"):
        sp = dest / name
        if not sp.is_file():
            continue
        tp = a00 / name
        if sp.suffix == ".parquet":
            import pandas as pd

            pd.read_parquet(sp).to_parquet(tp, index=False)
        else:
            tp.write_text(sp.read_text(encoding="utf-8"), encoding="utf-8")

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
        "formal_promotion_blocked": None if passed else FORMAL_PROMOTION_BLOCKED,
        "e0_code": E0_CODE,
        "r21_parity": r21.get("status"),
        "r21_1_parity": r211.get("status"),
        "r21_2_parity": r212.get("status"),
        "current_baseline_changed": False,
    }
    dump_json(PARITY_JSON, payload)
    write_md(
        MD["parity"],
        "\n".join(
            [
                "# Round21.3 · 00 v5 parity",
                "",
                f"status=`{payload['status']}` 口径=`{FEE_SCHEDULE}` engine=`cached_overlay_v4 → o01 → apply_top2_veto → local_executor`",
                f"anchor CAGR=`{V5_ANCHOR['cagr']}` Sharpe=`{V5_ANCHOR['sharpe']}` MaxDD=`{V5_ANCHOR['max_drawdown']}` trades=`{V5_ANCHOR['completed_trades']}` NAV=`{V5_NAV_END_EXACT}`",
                f"observed CAGR=`{obs.get('cagr')}` Sharpe=`{obs.get('sharpe')}` MaxDD=`{obs.get('max_drawdown')}` trades=`{obs.get('completed_trades')}` NAV=`{obs.get('nav_end')}`",
                f"R21=`{r21.get('status')}` R21.1=`{r211.get('status')}` R21.2=`{r212.get('status')}`",
                "FAIL 只停正式晋级。不改 O01/B03/90-10。CURRENT 不变。",
                "",
            ]
        ),
    )
    return payload
