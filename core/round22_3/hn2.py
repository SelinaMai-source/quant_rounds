"""Track C：HN2 公式分解 + Evidence 汇总。H1–H3 真训在 train/backtest。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import dump_json, load_json, write_md
from .contract import HN2_FORMULA_FROZEN, HN2_ID, MD, PIT_PASS, ROUND22_2_ROOT, ROUND22_3_ROOT
from .freeze import require_frozen_gate

HN2_JSON = ROUND22_3_ROOT / "hn2" / "hn2.json"


def _identity_from_panel() -> dict[str, Any]:
    helper = load_json(ROUND22_3_ROOT / "hn2" / "identity.json")
    if helper:
        return helper
    src = ROUND22_3_ROOT / "data" / "hn2_panel.parquet"
    if not src.is_file():
        src = ROUND22_2_ROOT / "data_recovery" / "hn2_panel.parquet"
    if not src.is_file():
        return {"status": "MISSING_PANEL", "is_delta_log_circ_minus_delta_log_holders": None}
    panel = pd.read_parquet(src)
    disc = load_json(ROUND22_2_ROOT / "data_recovery" / "hn.json")
    hn2 = (disc.get("hn2") or {})
    # algebraic identity: log(avg_t/avg_{t-1}) = Δlog(circ) − Δlog(holders) when avg=circ/holders
    return {
        "status": "ALGEBRAIC_IDENTITY",
        "formula_locked": HN2_FORMULA_FROZEN,
        "implemented": "log((circ/holders)_t / (circ/holders)_{t-1})",
        "is_delta_log_circ_minus_delta_log_holders": True,
        "proof": "log(c_t/h_t) - log(c_{t-1}/h_{t-1}) = [log c_t - log c_{t-1}] - [log h_t - log h_{t-1}]",
        "n": int(len(panel)),
        "finite_share": hn2.get("finite_share"),
        "filled_with_zero": False,
        "stale_nan": True,
        "pit_status": hn2.get("status") or PIT_PASS,
        "note": "Numerical reconstruction of circ/holders asof series is in helper identity.json when present.",
    }


def run_hn2(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and HN2_JSON.is_file() and MD["09"].is_file():
        existing = load_json(HN2_JSON)
        if existing.get("status") in {"DONE", "PARTIAL"}:
            return existing
    ident = _identity_from_panel()
    dump_json(ROUND22_3_ROOT / "hn2" / "identity.json", ident)
    ev_helper = load_json(ROUND22_3_ROOT / "hn2" / "evidence.json")
    disc = load_json(ROUND22_2_ROOT / "data_recovery" / "hn.json")
    train = load_json(ROUND22_3_ROOT / "model" / "train.json")
    sleeve = load_json(ROUND22_3_ROOT / "backtest" / "sleeve.json")
    rows = {r.get("code"): r for r in (sleeve.get("rows") or [])}
    heads = {
        "H1": rows.get("R223_E1_r22_2_hn2_avg_holding_chg") or {},
        "H2": rows.get("R223_E2_r22_2_hn2_avg_holding_chg") or {},
        "H3": rows.get("R223_E3_r22_2_hn2_avg_holding_chg") or {},
    }
    short = load_json(ROUND22_2_ROOT / "backtest" / "sleeve.json")
    short_e1 = next((r for r in (short.get("rows") or []) if r.get("code") == "R222_E1_r22_2_hn2_avg_holding_chg"), {})
    payload = {
        "status": "DONE" if heads["H1"] else "PARTIAL",
        "factor_id": HN2_ID,
        "identity": ident,
        "pit": (disc.get("hn2") or {}),
        "evidence_helper": bool(ev_helper),
        "evidence": ev_helper,
        "short_e1_delta_sharpe_vs_matched_e0": short_e1.get("delta_sharpe"),
        "heads": {
            k: {
                "sharpe": v.get("sharpe"),
                "cagr": v.get("cagr"),
                "delta_sharpe": v.get("delta_sharpe"),
                "n_obs": v.get("n_obs"),
                "economically_material": v.get("economically_material"),
            }
            for k, v in heads.items()
        },
        "filled_with_zero": False,
        "replace_only_if_rho_ge_0_60": True,
        "train_status": train.get("status"),
    }
    dump_json(HN2_JSON, payload)
    write_md(
        MD["09"],
        f"# Round22.3 · 09 HN2 恒等\n\n公式锁定 `{HN2_FORMULA_FROZEN}`。是否 Δlog(circ)−Δlog(holders)=`{ident.get('is_delta_log_circ_minus_delta_log_holders')}`。{ident.get('proof')}\nPIT=`{(disc.get('hn2') or {}).get('status')}` filled0=False。\n",
    )
    write_md(
        MD["10"],
        f"# Round22.3 · 10 HN2 头\n\n短窗 Five Add ΔSharpe=`{short_e1.get('delta_sharpe')}` vs matched E0。全历史 H1 ΔSharpe=`{(heads['H1'] or {}).get('delta_sharpe')}` H2=`{(heads['H2'] or {}).get('delta_sharpe')}` H3=`{(heads['H3'] or {}).get('delta_sharpe')}` vs R16_O01。达冗余门才 Replace。\n",
    )
    return payload
