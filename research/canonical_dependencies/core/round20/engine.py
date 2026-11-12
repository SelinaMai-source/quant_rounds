"""新 Alpha 接入 v5 引擎：90/10 fuse + B03 + apply_top2_veto O01 + Round7 5bp。不传 hist_hooks。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND20_ROOT, assert_project_result_path
from .common import dump_json, load_json, slim_obs
from .contract import FEE_SCHEDULE, O01_FORMULA, overlayed_path
from .parity import _pack, _run_r7


def run_v5_fused_backtest(
    dest: Path,
    five: pd.DataFrame,
    ten: pd.DataFrame,
    *,
    extra: dict[str, Any] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """fuse_heads → apply_b03 → apply_top2_veto(O01) → Round7 5bp。MAIN 表不传 validation_hooks。"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")
    extra = dict(extra or {})
    extra.setdefault("fee_schedule", FEE_SCHEDULE)
    extra.setdefault("validation_hooks", None)
    extra.setdefault("historical_stamp_tax", False)
    extra.setdefault("formula", O01_FORMULA)
    extra.setdefault(
        "engine",
        "fuse_heads 90/10 → apply_b03 → o01_signal_dates → apply_top2_veto → "
        "round7_fixed_sleeve_config + run_local_fixed_sleeve_backtest (no hist_hooks)",
    )
    if resume and (dest / "report.json").is_file() and (dest / "nav.parquet").is_file():
        report = load_json(dest / "report.json")
        nav = pd.read_parquet(dest / "nav.parquet")
        packed = _pack({"status": "resumed", "report": report, "nav": nav, "out_dir": str(dest)}, dest)
        packed["status"] = packed.get("status") or "resumed"
        packed["resumed"] = True
        return slim_obs(packed) | {k: packed.get(k) for k in ("yearly_returns", "out_dir", "nav_end", "status", "resumed")}

    from ..round15.runner import apply_top2_veto
    from ..round16_5.engine import o01_signal_dates
    from ..round17.backtest import apply_b03, fuse_heads

    fused = fuse_heads(five, ten)
    fused.to_parquet(dest / "predictions.parquet", index=False)
    overlayed, meta = apply_b03(fused)
    dump_json(dest / "overlay_meta.json", meta)
    dates = o01_signal_dates()
    ov5 = apply_top2_veto(overlayed, dates)
    ov_path = overlayed_path(dest.name)
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(ov_path)
    ov5.to_parquet(ov_path, index=False)
    ov5.to_parquet(dest / "overlayed.parquet", index=False)
    extra["n_o01_signal_dates"] = int(len(dates))
    ran = _run_r7(dest, ov5, extra)
    packed = _pack(ran, dest)
    packed["status"] = "ran"
    packed["resumed"] = False
    packed["n_o01_signal_dates"] = int(len(dates))
    packed["overlay_path"] = str(ov_path)
    obs = slim_obs(packed)
    obs.update({
        "yearly_returns": packed.get("yearly_returns") or {},
        "out_dir": str(dest),
        "nav_end": packed.get("nav_end"),
        "status": "ran",
        "resumed": False,
        "n_o01_signal_dates": int(len(dates)),
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
    })
    dump_json(dest / "summary.json", obs)
    return obs


def copy_a00_from_parity(dest: Path) -> dict[str, Any]:
    """R20_A00 = exact v5 replay。已有 Phase A parity PASS，复制确认，不重训。"""
    from .parity import PARITY_JSON, load_parity_status
    from .contract import overlayed_path as ovp, parity_dir

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "summary.json")
    parity = load_parity_status()
    if not parity:
        parity = load_json(PARITY_JSON)
    obs = dict(parity.get("observed") or {})
    src = parity_dir("R20_V5_R16_O01")
    for name in ("report.json", "nav.parquet", "trades.parquet", "replay_summary.json"):
        sp = src / name
        if sp.is_file():
            tp = dest / name
            if name.endswith(".parquet"):
                pd.read_parquet(sp).to_parquet(tp, index=False)
            else:
                tp.write_bytes(sp.read_bytes())
    src_ov = ovp("R20_V5_R16_O01")
    if src_ov.is_file():
        pd.read_parquet(src_ov).to_parquet(dest / "overlayed.parquet", index=False)
    payload = slim_obs(obs)
    payload.update({
        "code": "R20_A00",
        "role": "exact_v5_replay",
        "status": "PASS" if parity.get("passed") else (parity.get("status") or "MISSING"),
        "formal": True,
        "retrain": False,
        "force_retrain": False,
        "model": "locked_v5",
        "parity_passed": bool(parity.get("passed")),
        "copied_from": str(src),
        "out_dir": str(dest),
        "yearly_returns": obs.get("yearly_returns") or {},
        "nav_end": obs.get("nav_end"),
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
        "note": "copied/confirmed Phase A v5 Round7-caliber parity; not a retrain",
    })
    dump_json(dest / "summary.json", payload)
    return payload


__all__ = ["copy_a00_from_parity", "run_v5_fused_backtest"]
