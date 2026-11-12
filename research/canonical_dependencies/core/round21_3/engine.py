"""v5 引擎包装：只写 result/round21_3。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import assert_project_result_path
from ..round20.parity import _pack, _run_r7
from .common import dump_json, load_json, slim_obs
from .contract import FEE_SCHEDULE, O01_FORMULA, overlayed_path


def run_v5_fused_backtest(dest: Path, five: pd.DataFrame, ten: pd.DataFrame, *, extra: dict[str, Any] | None = None, resume: bool = True) -> dict[str, Any]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")
    extra = dict(extra or {})
    extra.setdefault("fee_schedule", FEE_SCHEDULE)
    extra.setdefault("validation_hooks", None)
    extra.setdefault("historical_stamp_tax", False)
    extra.setdefault("formula", O01_FORMULA)
    extra.setdefault("engine", "fuse_heads 90/10 → apply_b03 → o01 → apply_top2_veto → local_executor")
    extra.setdefault("round", "21.3")
    if resume and (dest / "report.json").is_file() and (dest / "nav.parquet").is_file():
        report = load_json(dest / "report.json")
        nav = pd.read_parquet(dest / "nav.parquet")
        packed = _pack({"status": "resumed", "report": report, "nav": nav, "out_dir": str(dest)}, dest)
        return slim_obs(packed) | {k: packed.get(k) for k in ("yearly_returns", "out_dir", "nav_end", "status")}
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
    ov5.to_parquet(ov_path, index=False)
    ov5.to_parquet(dest / "overlayed.parquet", index=False)
    ran = _run_r7(dest, ov5, extra)
    packed = _pack(ran, dest)
    obs = slim_obs(packed)
    obs.update({"yearly_returns": packed.get("yearly_returns") or {}, "out_dir": str(dest), "nav_end": packed.get("nav_end"), "status": "ran", "fee_schedule": extra.get("fee_schedule", FEE_SCHEDULE)})
    dump_json(dest / "summary.json", obs)
    return obs
