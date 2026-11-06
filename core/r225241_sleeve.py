"""正式袖套。复用 local_executor 的 Round7 契约，摘要写到新目录。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .local_executor import round7_fixed_sleeve_config, run_local_fixed_sleeve_backtest, write_backtest_artifacts
from .paths import MARKET_END, PREDICTION_START
from .r225241_io import dump_json, finite_or_none, load_json
from .research_sleeve import metrics_from_nav_file
from .round22_3.contract import FEE_SCHEDULE, V5_ANCHOR, V5_NAV_END_EXACT, material_vs_v5

_MARKET: dict[str, Any] | None = None


def market() -> dict[str, Any]:
    global _MARKET
    if _MARKET is None:
        from .round17.backtest import cached_market

        _MARKET = cached_market()
    return _MARKET


def apply_b03_o01(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    from .round15.runner import apply_top2_veto
    from .round16_5.engine import o01_signal_dates
    from .round17.backtest import apply_b03

    overlayed, meta = apply_b03(scores)
    final = apply_top2_veto(overlayed, o01_signal_dates())
    return final, meta


def run_round7(
    dest: Path,
    scores: pd.DataFrame,
    extra: dict[str, Any],
    *,
    resume: bool = True,
    validation_hooks: dict[str, Any] | None = None,
    base_nav: Path | None = None,
    apply_v5_gate: bool = True,
    scenario: bool = False,
    counts_as_alpha: bool = True,
    already_overlayed: bool = False,
) -> dict[str, Any]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    code = str(extra.get("code") or dest.name)
    summary_path = dest / "summary.json"
    if resume and summary_path.is_file() and (dest / "nav.parquet").is_file():
        cached = load_json(summary_path)
        if cached.get("code") == code and cached.get("status") == "ran":
            return cached
    frame = scores.copy()
    if "signal_date" not in frame.columns and "trade_date" in frame.columns:
        frame = frame.rename(columns={"trade_date": "signal_date"})
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    keep = ["signal_date", "instrument_id", "score"]
    if "allocation_multiplier" in frame.columns:
        frame["allocation_multiplier"] = pd.to_numeric(frame["allocation_multiplier"], errors="coerce")
        keep.append("allocation_multiplier")
    frame = frame.dropna(subset=["signal_date", "instrument_id", "score"])
    frame = frame.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort")
    if not already_overlayed:
        frame, meta = apply_b03_o01(frame[["signal_date", "instrument_id", "score"]])
        dump_json(dest / "overlay_meta.json", meta)
        keep = [col for col in ("signal_date", "instrument_id", "score", "allocation_multiplier") if col in frame.columns]
    frame[keep].to_parquet(dest / "overlayed.parquet", index=False)
    config = round7_fixed_sleeve_config(exit_mode="close_from_d2")
    result = run_local_fixed_sleeve_backtest(
        frame[keep],
        market()["prepared"],
        config=config,
        start_date=PREDICTION_START,
        end_date=MARKET_END,
        validation_hooks=validation_hooks,
    )
    from .round4.runner import attach_round4_fields

    metrics = result["report"].setdefault("metrics", {})
    attach_round4_fields(result["nav"], result["trades"], metrics)
    payload = {
        "fee_schedule": FEE_SCHEDULE,
        "historical_stamp_tax": False,
        "validation_hooks": validation_hooks,
        "engine": "round7_fixed_sleeve_config + run_local_fixed_sleeve_backtest",
        "restarted_capital": False,
        "compare_to": extra.get("compare_to") or "R16_O01",
    }
    payload.update(extra)
    payload["code"] = code
    payload["scenario"] = bool(scenario or extra.get("scenario"))
    written = write_backtest_artifacts(result, dest, extra_report=payload)
    report = written["report"]
    attach_round4_fields(result["nav"], result["trades"], report.setdefault("metrics", {}))
    dump_json(dest / "report.json", report)
    obs = metrics_from_nav_file(dest / "nav.parquet")
    trades = (report.get("metrics") or {}).get("completed_trades")
    if trades is None:
        trades = int(len(result["trades"])) if result.get("trades") is not None else None
    obs.update(
        {
            "code": code,
            "completed_trades": trades,
            "status": "ran",
            "fee_schedule": FEE_SCHEDULE,
            "compare_to": payload["compare_to"],
            "restarted_capital": False,
            "scenario": bool(payload["scenario"]),
            "counts_as_alpha": bool(counts_as_alpha and not payload["scenario"]),
            "round": extra.get("round"),
            "scope": extra.get("scope"),
            "initial_cash": finite_or_none((report.get("config") or {}).get("initial_cash")) or 300000.0,
        }
    )
    if apply_v5_gate:
        gate = material_vs_v5(
            obs,
            top_changed=bool(extra.get("top_changed", True)),
            fee_same_direction=bool(extra.get("fee_same_direction", True)),
            not_only_year=bool(extra.get("not_only_year", True)),
        )
        obs.update(gate)
        if payload["scenario"] or not counts_as_alpha:
            obs["economically_material"] = False
            obs["counts_as_alpha"] = False
    else:
        obs["economically_material"] = False
        obs["gate_id"] = "NOT_THE_1TO2_GATE"
        obs["delta_cagr_vs_v5_not_used"] = None if obs.get("cagr") is None else float(obs["cagr"]) - float(V5_ANCHOR["cagr"])
        obs["v5_nav_reference_not_a_benchmark"] = V5_NAV_END_EXACT
    dump_json(summary_path, obs)
    return obs
