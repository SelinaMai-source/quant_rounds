"""Inner Replay 适配器：把 backtest/signal 窗口真正传入执行引擎。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.local_executor import round7_fixed_sleeve_config, run_local_fixed_sleeve_backtest, write_backtest_artifacts
from core.r225241_io import dump_json, finite_or_none
from core.r225241_sleeve import apply_b03_o01, market
from core.research_sleeve import metrics_from_nav_file
from core.round22_3.contract import FEE_SCHEDULE
from core.round23_3.contract import MAX_INNER_NAV_DAYS


def run_windowed_sleeve(
    dest: Path,
    scores: pd.DataFrame,
    *,
    signal_start: str | pd.Timestamp,
    signal_end: str | pd.Timestamp,
    backtest_start: str | pd.Timestamp | None = None,
    backtest_end: str | pd.Timestamp | None = None,
    initial_cash: float = 300_000.0,
    already_overlayed: bool = False,
    prepared_market: Any = None,
    extra: dict[str, Any] | None = None,
    resume: bool = True,
    fail_on_full_history: bool = True,
) -> dict[str, Any]:
    """正式引擎回放，但强制日期窗；不再硬编码 2018–2026。"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    summary_path = dest / "summary.json"
    code = str((extra or {}).get("code") or dest.name)
    if resume and summary_path.is_file() and (dest / "nav.parquet").is_file():
        from core.research_integrity_v2.io_v2 import load_json

        cached = load_json(summary_path)
        if cached.get("code") == code and cached.get("status") in {"ran", "FAIL_FULL_HISTORY_WINDOW"}:
            return cached

    sig_start = pd.Timestamp(signal_start)
    sig_end = pd.Timestamp(signal_end)
    bt_start = pd.Timestamp(backtest_start or signal_start)
    bt_end = pd.Timestamp(backtest_end or signal_end)

    frame = scores.copy()
    if "signal_date" not in frame.columns and "trade_date" in frame.columns:
        frame = frame.rename(columns={"trade_date": "signal_date"})
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.loc[frame["signal_date"].between(sig_start, sig_end)].copy()
    keep = ["signal_date", "instrument_id", "score"]
    if "allocation_multiplier" in frame.columns:
        frame["allocation_multiplier"] = pd.to_numeric(frame["allocation_multiplier"], errors="coerce")
        keep.append("allocation_multiplier")
    frame = frame.dropna(subset=["signal_date", "instrument_id", "score"])
    frame = frame.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort")
    if frame.empty:
        empty = {
            "code": code,
            "status": "EMPTY_SIGNAL_BLOCK",
            "signal_start": str(sig_start.date()),
            "signal_end": str(sig_end.date()),
            "actual_backtest_start": str(bt_start.date()),
            "actual_backtest_end": str(bt_end.date()),
            "n_signal_days": 0,
            "n_nav_days": 0,
            "n_nav_days_in_signal_block": 0,
            "n_trades": 0,
            "completed_trades": 0,
            "window_fail": False,
            "initial_cash": float(initial_cash),
        }
        dump_json(summary_path, empty)
        dump_json(dest / "inner_block_manifest.json", empty)
        # 占位空净值，便于上层跳过
        pd.DataFrame({"trade_date": [], "total_asset": []}).to_parquet(dest / "nav.parquet", index=False)
        return empty
    if not already_overlayed:
        frame, meta = apply_b03_o01(frame[["signal_date", "instrument_id", "score"]])
        dump_json(dest / "overlay_meta.json", meta)
        keep = [c for c in ("signal_date", "instrument_id", "score", "allocation_multiplier") if c in frame.columns]
    frame[keep].to_parquet(dest / "overlayed.parquet", index=False)

    config = round7_fixed_sleeve_config(exit_mode="close_from_d2")
    # 初始资金写入 config（若 dataclass 支持）
    if hasattr(config, "initial_cash"):
        try:
            object.__setattr__(config, "initial_cash", float(initial_cash))
        except Exception:
            try:
                config.initial_cash = float(initial_cash)  # type: ignore[attr-defined]
            except Exception:
                pass

    result = run_local_fixed_sleeve_backtest(
        frame[keep],
        prepared_market if prepared_market is not None else market()["prepared"],
        config=config,
        start_date=bt_start,
        end_date=bt_end,
    )
    from core.round4.runner import attach_round4_fields

    metrics = result["report"].setdefault("metrics", {})
    attach_round4_fields(result["nav"], result["trades"], metrics)
    payload = {
        "fee_schedule": FEE_SCHEDULE,
        "engine": "round23_3.windowed_sleeve",
        "code": code,
        "signal_start": str(sig_start.date()),
        "signal_end": str(sig_end.date()),
        "actual_backtest_start": str(bt_start.date()),
        "actual_backtest_end": str(bt_end.date()),
        "initial_cash": float(initial_cash),
        "round": (extra or {}).get("round", "23.3"),
        "scope": (extra or {}).get("scope", "INNER_REPLAY"),
    }
    if extra:
        payload.update({k: v for k, v in extra.items() if k not in payload})
    written = write_backtest_artifacts(result, dest, extra_report=payload)
    report = written["report"]
    obs = metrics_from_nav_file(dest / "nav.parquet")
    nav = pd.read_parquet(dest / "nav.parquet")
    n_nav = int(len(nav))
    trades = (report.get("metrics") or {}).get("completed_trades")
    if trades is None:
        trades = int(len(result["trades"])) if result.get("trades") is not None else 0
    # 选择指标：仅用信号窗内入场对应的净值日；期末按可得价估值（日历已截断到 bt_end）
    nav_dates = pd.to_datetime(nav["trade_date"] if "trade_date" in nav.columns else nav.index)
    in_block = (nav_dates >= sig_start) & (nav_dates <= sig_end)
    block_nav = nav.loc[in_block].copy() if in_block.any() else nav.iloc[0:0].copy()
    n_signal_days = int(frame["signal_date"].nunique())
    status = "ran"
    if fail_on_full_history and n_nav > MAX_INNER_NAV_DAYS:
        status = "FAIL_FULL_HISTORY_WINDOW"
    obs.update(
        {
            "code": code,
            "status": status,
            "completed_trades": int(trades or 0),
            "signal_start": str(sig_start.date()),
            "signal_end": str(sig_end.date()),
            "actual_backtest_start": str(pd.Timestamp(nav_dates.min()).date()) if len(nav_dates) else str(bt_start.date()),
            "actual_backtest_end": str(pd.Timestamp(nav_dates.max()).date()) if len(nav_dates) else str(bt_end.date()),
            "n_signal_days": n_signal_days,
            "n_nav_days": n_nav,
            "n_nav_days_in_signal_block": int(len(block_nav)),
            "n_trades": int(trades or 0),
            "initial_cash": finite_or_none((report.get("config") or {}).get("initial_cash")) or float(initial_cash),
            "fee_schedule": FEE_SCHEDULE,
            "window_fail": status == "FAIL_FULL_HISTORY_WINDOW",
        }
    )
    dump_json(summary_path, obs)
    dump_json(
        dest / "inner_block_manifest.json",
        {
            "signal_start": obs["signal_start"],
            "signal_end": obs["signal_end"],
            "actual_backtest_start": obs["actual_backtest_start"],
            "actual_backtest_end": obs["actual_backtest_end"],
            "n_signal_days": n_signal_days,
            "n_nav_days": n_nav,
            "n_trades": int(trades or 0),
            "status": status,
        },
    )
    return obs
