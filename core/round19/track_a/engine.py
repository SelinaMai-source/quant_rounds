"""Track A 全引擎：C0 同款 run_canonical。禁止写 round16*/round18。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ...round15.runner import apply_top2_veto
from ...round16_5.engine import o01_signal_dates, option_observable_dates, put_skew_series
from ...round17.backtest import cached_overlay_v4, run_canonical
from ..common import dump_json, observed_from_report
from .support import overlayed_path


def run_overlayed(
    dest,
    overlayed: pd.DataFrame,
    *,
    extra: dict[str, Any] | None = None,
    resume: bool = True,
    write: bool = True,
    extra_buy_bps: float | None = None,
) -> dict[str, Any]:
    dest = Path(dest)
    ran = run_canonical(
        dest,
        overlayed,
        extra=extra or {},
        resume=resume,
        write=write,
        extra_buy_bps=extra_buy_bps,
    )
    report = ran.get("report") or {}
    obs = ran.get("observed") or observed_from_report(report)
    ran["observed"] = obs
    if write and obs.get("cagr") is not None and not (dest / "summary.json").is_file():
        dump_json(dest / "summary.json", {**obs, "code": dest.name})
    return ran


def run_veto(
    dest,
    veto_dates,
    *,
    extra: dict[str, Any] | None = None,
    resume: bool = True,
    write: bool = True,
    extra_buy_bps: float | None = None,
    overlayed: pd.DataFrame | None = None,
    save_code: str | None = None,
) -> dict[str, Any]:
    ov = overlayed if overlayed is not None else cached_overlay_v4()
    ov = apply_top2_veto(ov, veto_dates)
    if save_code:
        path = overlayed_path(save_code)
        path.parent.mkdir(parents=True, exist_ok=True)
        ov.to_parquet(path, index=False)
    return run_overlayed(
        dest,
        ov,
        extra=extra,
        resume=resume,
        write=write,
        extra_buy_bps=extra_buy_bps,
    )


__all__ = [
    "apply_top2_veto",
    "cached_overlay_v4",
    "o01_signal_dates",
    "option_observable_dates",
    "put_skew_series",
    "run_overlayed",
    "run_veto",
]
