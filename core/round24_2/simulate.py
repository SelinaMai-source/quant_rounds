"""单股机会模拟器：对齐组合引擎约束，不做近似退出公式。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.local_executor import round7_fixed_sleeve_config
from core.research_integrity_v2.time_contract import shanghai_ts


def simulate_single_opportunity(
    *,
    signal_date,
    instrument_id: str,
    entry_policy: str = "next_open_d1",
    exit_policy: str = "close_from_d2",
    cost_contract: str = "ROUND7_FLAT_5BP",
    market_data_version: str = "cached_market",
    asof_end: str = "2026-07-24",
    market: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if market is None:
        from core.r225241_sleeve import market as load_market

        market = load_market()
    prepared = market["prepared"]
    calendar = pd.DatetimeIndex(prepared.calendar)
    lookup = prepared.market_lookup
    sig = pd.Timestamp(signal_date)
    asof = pd.Timestamp(asof_end)
    cfg = round7_fixed_sleeve_config(exit_mode=exit_policy)
    out = {
        "signal_date": str(sig.date()),
        "instrument_id": str(instrument_id),
        "entry_policy": entry_policy,
        "exit_policy": exit_policy,
        "cost_contract": cost_contract,
        "market_data_version": market_data_version,
        "entry_order_time": shanghai_ts(sig).isoformat(),
        "planned_entry_date": None,
        "actual_entry_time": None,
        "entry_status": "NOT_ATTEMPTED",
        "actual_entry_price": None,
        "actual_exit_time": None,
        "exit_status": None,
        "label_available_at": None,
        "gross_return": None,
        "net_return": None,
        "fees": None,
        "dividends": 0.0,
        "holding_days": None,
        "censor_status": None,
        "profitability_label": None,
        "action_payoff": None,
        "action_flag": None,
    }
    if sig not in calendar:
        out["entry_status"] = "INVALID_SIGNAL_DATE"
        out["censor_status"] = "INVALID"
        return out
    i = int(calendar.get_loc(sig))
    if i + 1 >= len(calendar):
        out["entry_status"] = "NO_NEXT_SESSION"
        out["censor_status"] = "RIGHT_CENSORED"
        return out
    entry_day = calendar[i + 1]
    out["planned_entry_date"] = str(entry_day.date())
    # 不偷偷递延到更远可交易日
    try:
        bar = lookup.loc[(entry_day, instrument_id)]
        sig_bar = lookup.loc[(sig, instrument_id)]
    except KeyError:
        out["entry_status"] = "UNFILLED_ACTION"
        out["action_payoff"] = 0.0
        out["action_flag"] = "UNFILLED_ACTION"
        out["profitability_label"] = None
        out["censor_status"] = "UNFILLED"
        return out
    limit_up = float(bar["limit_up"]) if np.isfinite(bar.get("limit_up", np.nan)) and 0 < float(bar["limit_up"]) < 99_999 else np.nan
    open_px = float(bar["open"]) if np.isfinite(bar["open"]) else np.nan
    sealed = bool(np.isfinite(limit_up) and np.isfinite(open_px) and open_px + 1e-6 >= limit_up - float(cfg.limit_up_tolerance))
    paused = bool(bar.get("paused", False))
    if paused or not np.isfinite(open_px) or sealed:
        out["entry_status"] = "UNFILLED_ACTION"
        out["action_payoff"] = 0.0
        out["action_flag"] = "UNFILLED_ACTION"
        out["profitability_label"] = None
        out["censor_status"] = "UNFILLED"
        return out
    # 成本分项（训练标签用；组合 NAV 另由引擎扣费，不得重复）
    costs = cfg.costs
    entry_fee_bps = float(costs.slippage_bps + costs.commission_bps + costs.transfer_fee_bps)
    exit_fee_bps = float(costs.slippage_bps + costs.commission_bps + costs.transfer_fee_bps + costs.sell_tax_bps)
    entry_px = open_px * (1.0 + entry_fee_bps / 10000.0)
    out["actual_entry_time"] = shanghai_ts(entry_day).isoformat()
    out["actual_entry_price"] = float(entry_px)
    out["entry_status"] = "FILLED"
    # T+1：最早 exit_day = entry 后下一交易日（close_from_d2 相对信号为 D+2）
    # 与引擎 exit_mode=close_from_d2：信号日 D，买入 D+1，自 D+2 起可按收盘卖，涨停顺延
    hold_start_idx = i + 2
    exit_day = None
    exit_px = None
    for j in range(hold_start_idx, len(calendar)):
        day = calendar[j]
        if day > asof:
            out["exit_status"] = "RIGHT_CENSORED"
            out["censor_status"] = "RIGHT_CENSORED"
            out["label_available_at"] = None
            out["profitability_label"] = None
            out["action_payoff"] = None
            out["action_flag"] = "RIGHT_CENSORED"
            return out
        try:
            xb = lookup.loc[(day, instrument_id)]
        except KeyError:
            continue
        if bool(xb.get("paused", False)):
            continue
        close = float(xb["close"]) if np.isfinite(xb["close"]) else np.nan
        xlimit = float(xb["limit_up"]) if np.isfinite(xb.get("limit_up", np.nan)) and 0 < float(xb["limit_up"]) < 99_999 else np.nan
        still_up = bool(np.isfinite(xlimit) and np.isfinite(close) and close + 1e-6 >= xlimit - float(cfg.limit_up_tolerance))
        if still_up:
            continue
        # 跌停不可卖则继续持有
        xlimit_down = float(xb["limit_down"]) if "limit_down" in xb and np.isfinite(xb.get("limit_down", np.nan)) else np.nan
        if np.isfinite(xlimit_down) and np.isfinite(close) and close - 1e-6 <= xlimit_down + float(cfg.limit_up_tolerance):
            continue
        if not np.isfinite(close):
            continue
        exit_day = day
        exit_px = close * (1.0 - exit_fee_bps / 10000.0)
        break
    if exit_day is None:
        out["exit_status"] = "RIGHT_CENSORED"
        out["censor_status"] = "RIGHT_CENSORED"
        out["label_available_at"] = None
        out["profitability_label"] = None
        out["action_flag"] = "RIGHT_CENSORED"
        return out
    gross = float(exit_px / entry_px - 1.0)  # already fee-adjusted in prices => net-like
    # 上面 entry/exit 已嵌入费用，gross_return 报未费前，net 报费后
    raw_gross = float(close / open_px - 1.0)
    out["actual_exit_time"] = shanghai_ts(exit_day).isoformat()
    out["exit_status"] = "EXITED"
    out["label_available_at"] = shanghai_ts(exit_day).isoformat()
    out["gross_return"] = raw_gross
    out["net_return"] = gross
    out["fees"] = float(entry_fee_bps + exit_fee_bps)
    out["holding_days"] = int((exit_day - entry_day).days)
    out["censor_status"] = "COMPLETE"
    out["profitability_label"] = float(gross)
    out["action_payoff"] = float(gross)
    out["action_flag"] = "FILLED"
    out["signal_close"] = float(sig_bar["close"]) if np.isfinite(sig_bar["close"]) else None
    return out


def tail_seal_fraction(sealed_minutes: np.ndarray, expected_minutes: int) -> float:
    """从收盘向前连续封板分钟 / 预期可观察分钟；缺 bar 打断。"""
    if expected_minutes <= 0 or len(sealed_minutes) == 0:
        return float("nan")
    run = 0
    for v in sealed_minutes[::-1]:
        if not bool(v):
            break
        run += 1
    return float(run / expected_minutes)


def longest_run(sealed_minutes: np.ndarray) -> int:
    best = cur = 0
    for v in sealed_minutes:
        if bool(v):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)
