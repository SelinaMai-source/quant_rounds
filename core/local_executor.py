"""本地 fork 双袖套执行循环：买入/费用/门控/涨跌停与 Stock 原文一致，只加日频代理卖点。

Stock 执行器只有 close_from_d2 / next_open_after_non_limit_close / ma5_next_open。
新 exit_mode 不能写回 Stock，只能在课题内 fork。

日频代理（组名语义带 daily；不是真实 11:00/14:50）：
- 11:00 价 / 是否涨停 → 当日 open
- 14:50 价 / 是否涨停 → 当日 close
- 止损：复权近似 ma_price < ma5 × 1.05（与 ma5_next_open 同一套 ma5）

消融组不回退到 close_from_d2。
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .metrics import attach_nav_metrics
from .paths import MARKET_END, PREDICTION_START, assert_project_result_path
from .stock_readonly import ensure_stock_src_on_path, sha256_file

STAMP_TAX_CUTOVER = pd.Timestamp("2023-08-28")
HISTORICAL_PRE_CUTOVER_SELL_TAX_BPS = 10.0
HISTORICAL_POST_CUTOVER_SELL_TAX_BPS = 5.0
_STAMP_PATCH_LOCK = threading.Lock()


def historical_sell_tax_bps(day: Any) -> float:
    """Validation-only 公开印花税：卖出日 <2023-08-28 → 10bp，否则 5bp。"""
    return (
        HISTORICAL_PRE_CUTOVER_SELL_TAX_BPS
        if pd.Timestamp(day) < STAMP_TAX_CUTOVER
        else HISTORICAL_POST_CUTOVER_SELL_TAX_BPS
    )


def apply_buy_adverse_bps(config, extra_bps: float):
    """只加买侧滑点；卖侧仍用 config.costs.slippage_bps。"""
    return replace(
        config,
        high_gap_threshold=-10.0,
        high_gap_extra_entry_slippage_bps=float(extra_bps),
    )


@contextmanager
def historical_stamp_tax_patch():
    """内存替换 Stock `_close_position` 的卖印花税。不写 Stock 文件。"""
    ensure_stock_src_on_path()
    import first_board_hermes.fixed_sleeve_backtest as fsb

    orig = fsb._close_position

    def wrapped(sleeve, position, day, bar, config, *, reference_field: str = "close"):
        tax = historical_sell_tax_bps(day)
        new_cfg = replace(config, costs=replace(config.costs, sell_tax_bps=float(tax)))
        return orig(sleeve, position, day, bar, new_cfg, reference_field=reference_field)

    with _STAMP_PATCH_LOCK:
        fsb._close_position = wrapped
        try:
            yield
        finally:
            fsb._close_position = orig


STOCK_EXIT_MODES = frozenset({
    "close_from_d2",
    "next_open_after_non_limit_close",
    "ma5_next_open",
})
DAILY_PROXY_EXIT_MODES = frozenset({
    "daily_tp1100",
    "daily_tp1450",
    "daily_ma5stop",
    "daily_full",
})
MA5_STOP_BUFFER = 1.05
PROFIT_THRESHOLD = 0.0


def daily_proxy_flags(exit_mode: str) -> dict[str, bool]:
    mode = str(exit_mode)
    if mode not in DAILY_PROXY_EXIT_MODES:
        raise ValueError(f"not a daily-proxy exit_mode: {exit_mode}")
    return {
        "tp_1100": mode in {"daily_tp1100", "daily_full"},
        "tp_1450": mode in {"daily_tp1450", "daily_full"},
        "ma5_stop": mode in {"daily_ma5stop", "daily_full"},
        "fallback_close_from_d2": False,
    }


def _import_stock_executor():
    ensure_stock_src_on_path()
    from first_board_hermes.fixed_sleeve_backtest import (
        FixedSleeveConfig,
        PreparedFixedSleeveMarket,
        _apply_actions,
        _build_report,
        _close_position,
        _finite,
        _market_row,
        _open_cohort,
        _portfolio_nav_at_open,
        _prepare_entry_price_overrides,
        _prepare_predictions,
        _sell_block_reason,
        _sell_open_block_reason,
        _Sleeve,
        _validate_config,
        prepare_fixed_sleeve_market,
        run_fixed_dual_sleeve_top3_backtest,
    )
    return {
        "FixedSleeveConfig": FixedSleeveConfig,
        "PreparedFixedSleeveMarket": PreparedFixedSleeveMarket,
        "_apply_actions": _apply_actions,
        "_build_report": _build_report,
        "_close_position": _close_position,
        "_finite": _finite,
        "_market_row": _market_row,
        "_open_cohort": _open_cohort,
        "_portfolio_nav_at_open": _portfolio_nav_at_open,
        "_prepare_entry_price_overrides": _prepare_entry_price_overrides,
        "_prepare_predictions": _prepare_predictions,
        "_sell_block_reason": _sell_block_reason,
        "_sell_open_block_reason": _sell_open_block_reason,
        "_Sleeve": _Sleeve,
        "_validate_config": _validate_config,
        "prepare_fixed_sleeve_market": prepare_fixed_sleeve_market,
        "run_fixed_dual_sleeve_top3_backtest": run_fixed_dual_sleeve_top3_backtest,
    }


def baseline_fixed_sleeve_config(*, exit_mode: str = "close_from_d2"):
    """fs00263 / n50_d2 / top1_both_gates_nav30pct_v1 入场契约，只换 exit_mode。"""
    stock = _import_stock_executor()
    return stock["FixedSleeveConfig"](
        top_k=1,
        name_budget=90_000.0,
        name_budget_nav_fraction=0.3,
        market_up_ratio_min=0.45,
        open_gap_min=-0.02,
        open_gap_max=0.07,
        exit_mode=exit_mode,
        ma_window=5,
        max_holding_sessions=10,
    )


def fs01122_top2_fixed_sleeve_config(*, exit_mode: str = "close_from_d2"):
    """fs01122 / n100_d2 / Top2 / 25%+15% NAV / open_gap_min=-0.01。对齐 Stock rank25_15。"""
    from .paths import FS01122_OPEN_GAP_MIN, FS01122_RANK_NAV_FRACTIONS, FS01122_TOP_K

    stock = _import_stock_executor()
    return stock["FixedSleeveConfig"](
        top_k=int(FS01122_TOP_K),
        name_budget=45_000.0,
        name_budget_nav_fraction=None,
        rank_name_budget_nav_fractions=tuple(FS01122_RANK_NAV_FRACTIONS),
        market_up_ratio_min=0.45,
        open_gap_min=float(FS01122_OPEN_GAP_MIN),
        open_gap_max=None,
        exit_mode=exit_mode,
        ma_window=5,
        max_holding_sessions=10,
    )


def round7_fixed_sleeve_config(*, exit_mode: str = "close_from_d2"):
    """Round7 锁定契约：Top2 / 30%+10% NAV / 无执行器涨跌停缺口门 / close_from_d2。"""
    stock = _import_stock_executor()
    return stock["FixedSleeveConfig"](
        top_k=2,
        name_budget=10_000.0,
        name_budget_nav_fraction=None,
        rank_name_budget_nav_fractions=(0.3, 0.1),
        market_up_ratio_min=None,
        market_up_ratio_full_allocation_min=None,
        reduced_name_budget_nav_fraction=None,
        open_gap_min=None,
        open_gap_max=None,
        exit_mode=exit_mode,
        ma_window=5,
        max_holding_sessions=10,
    )


def _sell_1100_block_reason(bar: pd.Series | None, tolerance: float) -> str | None:
    """11:00 日频代理：open 代表该时点价格与涨停。"""
    if bar is None:
        return "missing_market_row"
    paused = bar["paused"] if "paused" in bar.index else None
    if bool(paused) if pd.notna(paused) else False:
        return "paused"
    if not _finite_local(bar.get("volume")) or float(bar["volume"]) <= 0:
        return "zero_volume"
    if not _finite_local(bar.get("open")):
        return "invalid_open"
    if _finite_local(bar.get("limit_up")) and float(bar["open"]) >= (
        float(bar["limit_up"]) - tolerance
    ):
        return "limit_up_hold"
    if (
        _finite_local(bar.get("limit_down"))
        and _finite_local(bar.get("high"))
        and float(bar["high"]) <= float(bar["limit_down"]) + 1e-12
    ):
        return "one_price_limit_down"
    if _finite_local(bar.get("limit_down")) and float(bar["open"]) <= (
        float(bar["limit_down"]) + tolerance
    ):
        return "open_limit_down"
    return None


def _finite_local(value: Any) -> bool:
    return pd.notna(value) and np.isfinite(float(value))


def _in_profit(proxy_price: Any, entry_price: Any) -> bool:
    return (
        _finite_local(proxy_price)
        and _finite_local(entry_price)
        and float(proxy_price) > float(entry_price) + PROFIT_THRESHOLD
    )


def _ma5_stop_hit(bar: pd.Series | None, *, buffer: float = MA5_STOP_BUFFER) -> bool | None:
    """True=触发止损；False=未触发；None=MA 不可用（不卖，记 unavailable）。"""
    if bar is None:
        return None
    if not _finite_local(bar.get("ma_price")) or not _finite_local(bar.get("ma5")):
        return None
    return float(bar["ma_price"]) < float(bar["ma5"]) * float(buffer)


def _count_delay(reason: str | None, delayed: dict[str, int]) -> None:
    if reason == "limit_up_hold":
        delayed["limit_up"] += 1
    elif reason in {"one_price_limit_down", "open_limit_down"}:
        delayed["limit_down"] += 1
    else:
        delayed["unavailable"] += 1


def run_local_fixed_sleeve_backtest(
    predictions: pd.DataFrame,
    market,
    *,
    corporate_actions: pd.DataFrame | None = None,
    entry_price_overrides: pd.DataFrame | None = None,
    config=None,
    start_date: Any | None = None,
    end_date: Any | None = None,
    overlay_hooks: dict[str, Any] | None = None,
    validation_hooks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Stock native modes via the original executor; daily-proxy / risk hooks via the fork.

    ``validation_hooks`` 只服务 Round11 核验：``historical_stamp_tax`` /
    ``buy_extra_slippage_bps``。默认 None，不影响 legacy / parity。
    """
    stock = _import_stock_executor()
    if config is None:
        config = stock["FixedSleeveConfig"]()
    vh = dict(validation_hooks or {})
    if vh.get("buy_extra_slippage_bps"):
        config = apply_buy_adverse_bps(config, vh["buy_extra_slippage_bps"])
    use_hist = bool(vh.get("historical_stamp_tax"))
    if overlay_hooks:
        validate_cfg = config
        if config.exit_mode not in STOCK_EXIT_MODES:
            validate_cfg = replace(config, exit_mode="close_from_d2")
        stock["_validate_config"](validate_cfg)
        return _run_overlay_loop(
            predictions,
            market,
            corporate_actions=corporate_actions,
            entry_price_overrides=entry_price_overrides,
            config=config,
            start_date=start_date,
            end_date=end_date,
            overlay_hooks=overlay_hooks,
            stock=stock,
        )
    if config.exit_mode in STOCK_EXIT_MODES:
        stock["_validate_config"](config)
        kwargs = dict(
            predictions=predictions,
            market=market,
            corporate_actions=corporate_actions,
            entry_price_overrides=entry_price_overrides,
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        if use_hist:
            with historical_stamp_tax_patch():
                result = stock["run_fixed_dual_sleeve_top3_backtest"](**kwargs)
            report = result.setdefault("report", {})
            report["fee_schedule"] = "HISTORICAL_FEE_CORRECTED"
            report.setdefault("validation_hooks", {}).update({
                "historical_stamp_tax": True,
                "stamp_tax_cutover": str(STAMP_TAX_CUTOVER.date()),
                "pre_cutover_sell_tax_bps": HISTORICAL_PRE_CUTOVER_SELL_TAX_BPS,
                "post_cutover_sell_tax_bps": HISTORICAL_POST_CUTOVER_SELL_TAX_BPS,
            })
            return result
        return stock["run_fixed_dual_sleeve_top3_backtest"](**kwargs)
    if config.exit_mode not in DAILY_PROXY_EXIT_MODES:
        raise ValueError(f"unsupported exit_mode: {config.exit_mode}")
    flags = daily_proxy_flags(config.exit_mode)
    if flags["fallback_close_from_d2"]:
        raise RuntimeError("daily-proxy ablations must not fall back to close_from_d2")
    stock["_validate_config"](replace(config, exit_mode="close_from_d2"))
    return _run_daily_proxy_loop(
        predictions,
        market,
        corporate_actions=corporate_actions,
        entry_price_overrides=entry_price_overrides,
        config=config,
        start_date=start_date,
        end_date=end_date,
        flags=flags,
        stock=stock,
    )


def _run_daily_proxy_loop(
    predictions: pd.DataFrame,
    market,
    *,
    corporate_actions,
    entry_price_overrides,
    config,
    start_date,
    end_date,
    flags: dict[str, bool],
    stock: dict[str, Any],
) -> dict[str, Any]:
    PreparedFixedSleeveMarket = stock["PreparedFixedSleeveMarket"]
    _prepare_predictions = stock["_prepare_predictions"]
    _prepare_entry_price_overrides = stock["_prepare_entry_price_overrides"]
    prepare_fixed_sleeve_market = stock["prepare_fixed_sleeve_market"]
    _Sleeve = stock["_Sleeve"]
    _apply_actions = stock["_apply_actions"]
    _portfolio_nav_at_open = stock["_portfolio_nav_at_open"]
    _open_cohort = stock["_open_cohort"]
    _market_row = stock["_market_row"]
    _sell_block_reason = stock["_sell_block_reason"]
    _close_position = stock["_close_position"]
    _build_report = stock["_build_report"]
    _finite = stock["_finite"]

    scores = _prepare_predictions(predictions)
    entry_price_lookup = _prepare_entry_price_overrides(entry_price_overrides)
    if isinstance(market, PreparedFixedSleeveMarket):
        if corporate_actions is not None:
            raise ValueError("corporate_actions must be included when preparing market state")
        if market.ma_window != config.ma_window:
            raise ValueError("prepared market ma_window does not match execution config")
        bars = market.bars
        calendar = market.calendar
        market_lookup = market.market_lookup
        market_up_ratio = market.market_up_ratio
        action_lookup = market.action_lookup
    else:
        prepared = prepare_fixed_sleeve_market(
            market, corporate_actions=corporate_actions, ma_window=config.ma_window
        )
        bars = prepared.bars
        calendar = prepared.calendar
        market_lookup = prepared.market_lookup
        market_up_ratio = prepared.market_up_ratio
        action_lookup = prepared.action_lookup
    if start_date is not None:
        calendar = calendar[calendar >= pd.Timestamp(start_date)]
    if end_date is not None:
        calendar = calendar[calendar <= pd.Timestamp(end_date)]
    if len(calendar) < 3:
        raise ValueError("backtest calendar must contain at least three sessions")
    calendar_position = {day: offset for offset, day in enumerate(calendar)}

    entry_schedule: dict[pd.Timestamp, dict[str, Any]] = {}
    insufficient_horizon: list[dict[str, Any]] = []
    for signal_date, block in scores.groupby("signal_date", sort=True):
        signal_date = pd.Timestamp(signal_date)
        offset = calendar_position.get(signal_date)
        if offset is None or offset + 2 >= len(calendar):
            insufficient_horizon.append({
                "event": "signal_rejected", "reason": "insufficient_horizon",
                "signal_date": signal_date, "execution_date": pd.NaT,
                "sleeve_id": pd.NA, "cohort_id": pd.NA,
                "instrument_id": pd.NA,
            })
            continue
        chosen = block.sort_values(
            ["score", "instrument_id"], ascending=[False, True]
        )
        if config.top_k is not None:
            chosen = chosen.head(config.top_k)
        score_gate_rejections = 0
        if config.minimum_score is not None:
            score_gate_rejections = int(chosen["score"].le(config.minimum_score).sum())
            chosen = chosen.loc[chosen["score"].gt(config.minimum_score)].copy()
        chosen = chosen.copy()
        chosen["selection_rank"] = np.arange(1, len(chosen) + 1, dtype=int)
        up_ratio = market_up_ratio.get(signal_date, np.nan)
        market_gate_pass = (
            config.market_up_ratio_min is None
            or (_finite(up_ratio) and float(up_ratio) >= config.market_up_ratio_min)
        )
        allocation_tier = "full"
        effective_name_budget_nav_fraction = config.name_budget_nav_fraction
        if (
            market_gate_pass
            and config.market_up_ratio_full_allocation_min is not None
            and float(up_ratio) < config.market_up_ratio_full_allocation_min
        ):
            allocation_tier = "reduced"
            effective_name_budget_nav_fraction = (
                config.reduced_name_budget_nav_fraction
            )
        entry_schedule[calendar[offset + 1]] = {
            "signal_date": signal_date,
            "planned_exit_date": calendar[offset + 2],
            "sleeve_id": int((offset + 1) % config.sleeve_count),
            "market_up_ratio": up_ratio,
            "market_gate_pass": market_gate_pass,
            "allocation_tier": allocation_tier,
            "name_budget_nav_fraction": effective_name_budget_nav_fraction,
            "score_gate_rejections": score_gate_rejections,
            "picks": chosen[[
                "instrument_id", "score", "selection_rank", "allocation_multiplier",
            ]].to_dict("records"),
        }

    sleeves = [_Sleeve(sleeve_id=index, cash=config.sleeve_cash)
               for index in range(config.sleeve_count)]
    trades: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = list(insufficient_horizon)
    positions: list[dict[str, Any]] = []
    sleeve_states: list[dict[str, Any]] = []
    nav_rows: list[dict[str, Any]] = []
    cashflows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    skipped_insufficient_sleeve_cash = 0
    market_gate_rejected_signals = 0
    score_gate_rejected_names = 0
    delayed = {"limit_up": 0, "limit_down": 0, "unavailable": 0}

    for day in calendar:
        day = pd.Timestamp(day)
        _apply_actions(day, sleeves, action_lookup, cashflows)

        scheduled = entry_schedule.get(day)
        if scheduled is not None:
            score_gate_rejected_names += int(scheduled["score_gate_rejections"])
            if not scheduled["market_gate_pass"]:
                market_gate_rejected_signals += 1
                rejections.append({
                    "event": "signal_rejected", "reason": "market_up_ratio_below_min",
                    "signal_date": scheduled["signal_date"], "execution_date": day,
                    "sleeve_id": scheduled["sleeve_id"], "cohort_id": pd.NA,
                    "instrument_id": pd.NA,
                    "market_up_ratio": scheduled["market_up_ratio"],
                })
            else:
                sleeve = sleeves[int(scheduled["sleeve_id"])]
                entry_portfolio_nav = _portfolio_nav_at_open(
                    sleeves, market_lookup, day
                )
                opened = _open_cohort(
                    sleeve, scheduled, day, market_lookup, config,
                    rejections, cost_rows, entry_portfolio_nav,
                    entry_price_lookup,
                )
                if not opened and sleeve.cash < config.lot_size:
                    skipped_insufficient_sleeve_cash += 1

        for sleeve in sleeves:
            if not sleeve.positions:
                continue
            sleeve.occupied_sessions += 1
            for position_key, position in list(sleeve.positions.items()):
                instrument_id = position.instrument_id
                bar = _market_row(market_lookup, day, instrument_id)
                if bar is not None and _finite(bar.get("close")):
                    position.last_close = float(bar["close"])
                if day < position.planned_exit_date:
                    continue
                sold = _try_daily_proxy_exit(
                    sleeve, position, position_key, day, bar, config, flags,
                    trades, cost_rows, rejections, delayed,
                    _sell_block_reason, _close_position,
                )
                if sold:
                    del sleeve.positions[position_key]

        total_market_value = 0.0
        for sleeve in sleeves:
            sleeve_market_value = 0.0
            for position in sleeve.positions.values():
                market_value = float(position.quantity) * float(position.last_close)
                sleeve_market_value += market_value
                positions.append({
                    "trade_date": day, "sleeve_id": sleeve.sleeve_id,
                    "cohort_id": position.cohort_id,
                    "instrument_id": position.instrument_id,
                    "quantity": position.quantity, "last_close": position.last_close,
                    "market_value": market_value,
                    "signal_date": position.signal_date,
                    "planned_exit_date": position.planned_exit_date,
                })
            total_market_value += sleeve_market_value
            sleeve_states.append({
                "trade_date": day, "sleeve_id": sleeve.sleeve_id,
                "cash": sleeve.cash, "market_value": sleeve_market_value,
                "equity": sleeve.cash + sleeve_market_value,
                "occupied": bool(sleeve.positions),
                "live_cohort_count": len({p.cohort_id for p in sleeve.positions.values()}),
                "position_count": len(sleeve.positions),
            })
        total_cash = float(sum(sleeve.cash for sleeve in sleeves))
        nav_rows.append({
            "trade_date": day, "cash": total_cash,
            "market_value": total_market_value,
            "total_asset": total_cash + total_market_value,
            "live_sleeves": sum(bool(sleeve.positions) for sleeve in sleeves),
        })

    frames = {
        "trades": pd.DataFrame(trades),
        "rejections": pd.DataFrame(rejections),
        "positions": pd.DataFrame(positions),
        "sleeve_states": pd.DataFrame(sleeve_states),
        "nav": pd.DataFrame(nav_rows),
        "cashflows": pd.DataFrame(cashflows),
        "costs": pd.DataFrame(cost_rows),
    }
    report = _build_report(
        frames, sleeves, config, skipped_insufficient_sleeve_cash, delayed,
        market_gate_rejected_signals, score_gate_rejected_names,
    )
    trade_frame = frames["trades"]
    extra = {
        "tp_1100_exit_count": int(trade_frame["exit_trigger"].eq("tp_1100").sum())
        if not trade_frame.empty and "exit_trigger" in trade_frame else 0,
        "tp_1450_exit_count": int(trade_frame["exit_trigger"].eq("tp_1450").sum())
        if not trade_frame.empty and "exit_trigger" in trade_frame else 0,
        "ma5_stop_105_exit_count": int(
            trade_frame["exit_trigger"].eq("ma5_stop_105").sum()
        ) if not trade_frame.empty and "exit_trigger" in trade_frame else 0,
        "daily_proxy": {
            "open_as_1100": True,
            "close_as_1450": True,
            "ma5_stop_buffer": MA5_STOP_BUFFER,
            "profit_vs": "entry_price",
            "fallback_close_from_d2": False,
            "flags": flags,
        },
    }
    report["metrics"].update(extra)
    report["contract"] = (
        report["contract"]
        + "; daily-proxy 11:00=open 14:50=close; no close_from_d2 fallback"
    )
    return {**frames, "report": report}


def _try_daily_proxy_exit(
    sleeve,
    position,
    position_key: str,
    day: pd.Timestamp,
    bar,
    config,
    flags: dict[str, bool],
    trades: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    delayed: dict[str, int],
    _sell_block_reason,
    _close_position,
) -> bool:
    """Return True if the position was closed. Never falls back to close_from_d2."""
    if flags.get("fallback_close_from_d2"):
        raise RuntimeError("daily-proxy ablations must not fall back to close_from_d2")

    if flags["tp_1100"]:
        reason_1100 = _sell_1100_block_reason(bar, config.limit_up_tolerance)
        if reason_1100 is not None:
            _count_delay(reason_1100, delayed)
            rejections.append({
                "event": "sell_1100_rejected", "reason": reason_1100,
                "signal_date": position.signal_date, "execution_date": day,
                "sleeve_id": sleeve.sleeve_id, "cohort_id": position.cohort_id,
                "instrument_id": position.instrument_id,
            })
        elif _in_profit(bar.get("open") if bar is not None else None, position.entry_price):
            position.exit_trigger = "tp_1100"
            trade, sell_costs = _close_position(
                sleeve, position, day, bar, config, reference_field="open"
            )
            trades.append(trade)
            cost_rows.extend(sell_costs)
            return True

    if not flags["tp_1450"] and not flags["ma5_stop"]:
        return False

    reason_1450 = _sell_block_reason(bar, config.limit_up_tolerance)
    if reason_1450 is not None:
        _count_delay(reason_1450, delayed)
        rejections.append({
            "event": "sell_1450_rejected", "reason": reason_1450,
            "signal_date": position.signal_date, "execution_date": day,
            "sleeve_id": sleeve.sleeve_id, "cohort_id": position.cohort_id,
            "instrument_id": position.instrument_id,
        })
        return False

    if flags["tp_1450"] and _in_profit(
        bar.get("close") if bar is not None else None, position.entry_price
    ):
        position.exit_trigger = "tp_1450"
        trade, sell_costs = _close_position(
            sleeve, position, day, bar, config, reference_field="close"
        )
        trades.append(trade)
        cost_rows.extend(sell_costs)
        return True

    if flags["ma5_stop"]:
        hit = _ma5_stop_hit(bar)
        if hit is None:
            delayed["unavailable"] += 1
            rejections.append({
                "event": "sell_1450_rejected", "reason": "ma5_unavailable",
                "signal_date": position.signal_date, "execution_date": day,
                "sleeve_id": sleeve.sleeve_id, "cohort_id": position.cohort_id,
                "instrument_id": position.instrument_id,
            })
            return False
        if hit:
            position.exit_trigger = "ma5_stop_105"
            trade, sell_costs = _close_position(
                sleeve, position, day, bar, config, reference_field="close"
            )
            trades.append(trade)
            cost_rows.extend(sell_costs)
            return True
    return False


def _nav_dd_multiplier(nav: float, peak: float, rule: dict[str, Any]) -> float:
    """Causal size from current NAV vs peak. Recovery is automatic as NAV rises."""
    if peak <= 0 or not np.isfinite(nav) or not np.isfinite(peak):
        return 1.0
    drawdown = nav / peak - 1.0
    hard = float(rule.get("hard", 0.15))
    soft = float(rule.get("soft", 0.10))
    soft_mult = float(rule.get("soft_mult", 0.5))
    if drawdown <= -hard:
        return 0.0
    if drawdown <= -soft:
        return soft_mult
    return 1.0


def _run_overlay_loop(
    predictions: pd.DataFrame,
    market,
    *,
    corporate_actions,
    entry_price_overrides,
    config,
    start_date,
    end_date,
    overlay_hooks: dict[str, Any],
    stock: dict[str, Any],
) -> dict[str, Any]:
    """Stock loop + optional NAV sizing / exit overrides. Not a third backtester."""
    PreparedFixedSleeveMarket = stock["PreparedFixedSleeveMarket"]
    _prepare_predictions = stock["_prepare_predictions"]
    _prepare_entry_price_overrides = stock["_prepare_entry_price_overrides"]
    prepare_fixed_sleeve_market = stock["prepare_fixed_sleeve_market"]
    _Sleeve = stock["_Sleeve"]
    _apply_actions = stock["_apply_actions"]
    _portfolio_nav_at_open = stock["_portfolio_nav_at_open"]
    _open_cohort = stock["_open_cohort"]
    _market_row = stock["_market_row"]
    _sell_block_reason = stock["_sell_block_reason"]
    _sell_open_block_reason = stock["_sell_open_block_reason"]
    _close_position = stock["_close_position"]
    _build_report = stock["_build_report"]
    _finite = stock["_finite"]

    scores = _prepare_predictions(predictions)
    entry_price_lookup = _prepare_entry_price_overrides(entry_price_overrides)
    if isinstance(market, PreparedFixedSleeveMarket):
        if corporate_actions is not None:
            raise ValueError("corporate_actions must be included when preparing market state")
        if market.ma_window != config.ma_window:
            raise ValueError("prepared market ma_window does not match execution config")
        calendar = market.calendar
        market_lookup = market.market_lookup
        market_up_ratio = market.market_up_ratio
        action_lookup = market.action_lookup
    else:
        prepared = prepare_fixed_sleeve_market(
            market, corporate_actions=corporate_actions, ma_window=config.ma_window
        )
        calendar = prepared.calendar
        market_lookup = prepared.market_lookup
        market_up_ratio = prepared.market_up_ratio
        action_lookup = prepared.action_lookup
    if start_date is not None:
        calendar = calendar[calendar >= pd.Timestamp(start_date)]
    if end_date is not None:
        calendar = calendar[calendar <= pd.Timestamp(end_date)]
    if len(calendar) < 3:
        raise ValueError("backtest calendar must contain at least three sessions")
    calendar_position = {day: offset for offset, day in enumerate(calendar)}

    entry_schedule: dict[pd.Timestamp, dict[str, Any]] = {}
    insufficient_horizon: list[dict[str, Any]] = []
    for signal_date, block in scores.groupby("signal_date", sort=True):
        signal_date = pd.Timestamp(signal_date)
        offset = calendar_position.get(signal_date)
        if offset is None or offset + 2 >= len(calendar):
            insufficient_horizon.append({
                "event": "signal_rejected", "reason": "insufficient_horizon",
                "signal_date": signal_date, "execution_date": pd.NaT,
                "sleeve_id": pd.NA, "cohort_id": pd.NA,
                "instrument_id": pd.NA,
            })
            continue
        chosen = block.sort_values(
            ["score", "instrument_id"], ascending=[False, True]
        )
        if config.top_k is not None:
            chosen = chosen.head(config.top_k)
        score_gate_rejections = 0
        if config.minimum_score is not None:
            score_gate_rejections = int(chosen["score"].le(config.minimum_score).sum())
            chosen = chosen.loc[chosen["score"].gt(config.minimum_score)].copy()
        chosen = chosen.copy()
        chosen["selection_rank"] = np.arange(1, len(chosen) + 1, dtype=int)
        up_ratio = market_up_ratio.get(signal_date, np.nan)
        market_gate_pass = (
            config.market_up_ratio_min is None
            or (_finite(up_ratio) and float(up_ratio) >= config.market_up_ratio_min)
        )
        allocation_tier = "full"
        effective_name_budget_nav_fraction = config.name_budget_nav_fraction
        if (
            market_gate_pass
            and config.market_up_ratio_full_allocation_min is not None
            and float(up_ratio) < config.market_up_ratio_full_allocation_min
        ):
            allocation_tier = "reduced"
            effective_name_budget_nav_fraction = (
                config.reduced_name_budget_nav_fraction
            )
        entry_schedule[calendar[offset + 1]] = {
            "signal_date": signal_date,
            "planned_exit_date": calendar[offset + 2],
            "sleeve_id": int((offset + 1) % config.sleeve_count),
            "market_up_ratio": up_ratio,
            "market_gate_pass": market_gate_pass,
            "allocation_tier": allocation_tier,
            "name_budget_nav_fraction": effective_name_budget_nav_fraction,
            "score_gate_rejections": score_gate_rejections,
            "picks": chosen[[
                "instrument_id", "score", "selection_rank", "allocation_multiplier",
            ]].to_dict("records"),
        }

    sleeves = [_Sleeve(sleeve_id=index, cash=config.sleeve_cash)
               for index in range(config.sleeve_count)]
    trades: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = list(insufficient_horizon)
    positions: list[dict[str, Any]] = []
    sleeve_states: list[dict[str, Any]] = []
    nav_rows: list[dict[str, Any]] = []
    cashflows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    skipped_insufficient_sleeve_cash = 0
    market_gate_rejected_signals = 0
    score_gate_rejected_names = 0
    delayed = {"limit_up": 0, "limit_down": 0, "unavailable": 0}
    peak_nav = float(config.initial_cash)
    nav_dd_rule = overlay_hooks.get("nav_dd")
    early_exit = overlay_hooks.get("early_exit")
    ma5_buffer = overlay_hooks.get("ma5_buffer")
    regime_bad_dates = {
        pd.Timestamp(x) for x in (overlay_hooks.get("regime_bad_dates") or [])
    }
    low_conf_dates = {
        pd.Timestamp(x) for x in (overlay_hooks.get("low_conf_dates") or [])
    }
    exit_mode = str(config.exit_mode)

    for day in calendar:
        day = pd.Timestamp(day)
        _apply_actions(day, sleeves, action_lookup, cashflows)

        if exit_mode == "next_open_after_non_limit_close" or early_exit == "low_conf_no_con":
            for sleeve in sleeves:
                for position_key, position in list(sleeve.positions.items()):
                    if position.exit_armed_on is None or day <= position.exit_armed_on:
                        continue
                    bar = _market_row(market_lookup, day, position.instrument_id)
                    reason = _sell_open_block_reason(bar, config.limit_up_tolerance)
                    if reason is not None:
                        _count_delay(reason, delayed)
                        rejections.append({
                            "event": "sell_open_rejected", "reason": reason,
                            "signal_date": position.signal_date, "execution_date": day,
                            "sleeve_id": sleeve.sleeve_id, "cohort_id": position.cohort_id,
                            "instrument_id": position.instrument_id,
                            "exit_armed_on": position.exit_armed_on,
                        })
                        continue
                    trade, sell_costs = _close_position(
                        sleeve, position, day, bar, config, reference_field="open"
                    )
                    trades.append(trade)
                    cost_rows.extend(sell_costs)
                    del sleeve.positions[position_key]

        scheduled = entry_schedule.get(day)
        if scheduled is not None:
            score_gate_rejected_names += int(scheduled["score_gate_rejections"])
            if not scheduled["market_gate_pass"]:
                market_gate_rejected_signals += 1
                rejections.append({
                    "event": "signal_rejected", "reason": "market_up_ratio_below_min",
                    "signal_date": scheduled["signal_date"], "execution_date": day,
                    "sleeve_id": scheduled["sleeve_id"], "cohort_id": pd.NA,
                    "instrument_id": pd.NA,
                    "market_up_ratio": scheduled["market_up_ratio"],
                })
            else:
                entry_portfolio_nav = _portfolio_nav_at_open(
                    sleeves, market_lookup, day
                )
                peak_nav = max(peak_nav, float(entry_portfolio_nav))
                extra_mult = 1.0
                if nav_dd_rule:
                    extra_mult = _nav_dd_multiplier(
                        float(entry_portfolio_nav), peak_nav, nav_dd_rule
                    )
                if extra_mult <= 0:
                    market_gate_rejected_signals += 1
                    rejections.append({
                        "event": "signal_rejected", "reason": "nav_dd_hard_skip",
                        "signal_date": scheduled["signal_date"], "execution_date": day,
                        "sleeve_id": scheduled["sleeve_id"], "cohort_id": pd.NA,
                        "instrument_id": pd.NA,
                        "entry_portfolio_nav": entry_portfolio_nav,
                        "peak_nav": peak_nav,
                    })
                else:
                    if extra_mult < 1.0:
                        for pick in scheduled["picks"]:
                            pick["allocation_multiplier"] = min(
                                1.0,
                                max(1e-12, float(pick["allocation_multiplier"]) * extra_mult),
                            )
                    sleeve = sleeves[int(scheduled["sleeve_id"])]
                    opened = _open_cohort(
                        sleeve, scheduled, day, market_lookup, config,
                        rejections, cost_rows, entry_portfolio_nav,
                        entry_price_lookup,
                    )
                    if not opened and sleeve.cash < config.lot_size:
                        skipped_insufficient_sleeve_cash += 1

        for sleeve in sleeves:
            if not sleeve.positions:
                continue
            sleeve.occupied_sessions += 1
            for position_key, position in list(sleeve.positions.items()):
                instrument_id = position.instrument_id
                bar = _market_row(market_lookup, day, position.instrument_id)
                if bar is not None and _finite(bar.get("close")):
                    position.last_close = float(bar["close"])

                if exit_mode == "next_open_after_non_limit_close":
                    if position.exit_armed_on is not None:
                        continue
                    if bar is None or (
                        bool(bar["paused"]) if pd.notna(bar.get("paused")) else False
                    ) or not _finite(bar.get("volume")) or float(bar["volume"]) <= 0 \
                            or not _finite(bar.get("close")):
                        delayed["unavailable"] += 1
                        continue
                    if _finite(bar.get("limit_up")) and float(bar["close"]) >= (
                        float(bar["limit_up"]) - config.limit_up_tolerance
                    ):
                        delayed["limit_up"] += 1
                    else:
                        position.exit_armed_on = day
                        position.exit_trigger = "non_limit_close"
                    continue

                if early_exit == "low_conf_no_con":
                    if position.exit_armed_on is not None:
                        continue
                    signal_day = pd.Timestamp(position.signal_date)
                    if signal_day in low_conf_dates:
                        offset = calendar_position.get(signal_day)
                        d1 = calendar[offset + 1] if offset is not None and offset + 1 < len(calendar) else None
                        if d1 is not None and day == pd.Timestamp(d1):
                            if bar is None or not _finite(bar.get("close")):
                                delayed["unavailable"] += 1
                                continue
                            sealed = (
                                _finite(bar.get("limit_up"))
                                and float(bar["close"]) >= (
                                    float(bar["limit_up"]) - config.limit_up_tolerance
                                )
                            )
                            if not sealed:
                                position.exit_armed_on = day
                                position.exit_trigger = "low_conf_no_con"
                                continue

                if early_exit == "regime_deteriorate" and day >= position.planned_exit_date:
                    if day in regime_bad_dates:
                        reason = _sell_block_reason(bar, config.limit_up_tolerance)
                        if reason is None:
                            position.exit_trigger = "regime_deteriorate"
                            trade, sell_costs = _close_position(
                                sleeve, position, day, bar, config, reference_field="close"
                            )
                            trades.append(trade)
                            cost_rows.extend(sell_costs)
                            del sleeve.positions[position_key]
                            continue
                        _count_delay(reason, delayed)

                if ma5_buffer is not None and day >= position.planned_exit_date:
                    hit = _ma5_stop_hit(bar, buffer=float(ma5_buffer))
                    if hit is True:
                        reason = _sell_block_reason(bar, config.limit_up_tolerance)
                        if reason is None:
                            position.exit_trigger = f"ma5_stop_{ma5_buffer}"
                            trade, sell_costs = _close_position(
                                sleeve, position, day, bar, config, reference_field="close"
                            )
                            trades.append(trade)
                            cost_rows.extend(sell_costs)
                            del sleeve.positions[position_key]
                            continue
                        _count_delay(reason, delayed)

                if day < position.planned_exit_date:
                    continue
                reason = _sell_block_reason(bar, config.limit_up_tolerance)
                if reason is not None:
                    _count_delay(reason, delayed)
                    rejections.append({
                        "event": "sell_rejected", "reason": reason,
                        "signal_date": position.signal_date, "execution_date": day,
                        "sleeve_id": sleeve.sleeve_id, "cohort_id": position.cohort_id,
                        "instrument_id": instrument_id,
                    })
                    continue
                trade, sell_costs = _close_position(sleeve, position, day, bar, config)
                trades.append(trade)
                cost_rows.extend(sell_costs)
                del sleeve.positions[position_key]

        total_market_value = 0.0
        for sleeve in sleeves:
            sleeve_market_value = 0.0
            for position in sleeve.positions.values():
                market_value = float(position.quantity) * float(position.last_close)
                sleeve_market_value += market_value
                positions.append({
                    "trade_date": day, "sleeve_id": sleeve.sleeve_id,
                    "cohort_id": position.cohort_id,
                    "instrument_id": position.instrument_id,
                    "quantity": position.quantity, "last_close": position.last_close,
                    "market_value": market_value,
                    "signal_date": position.signal_date,
                    "planned_exit_date": position.planned_exit_date,
                })
            total_market_value += sleeve_market_value
            sleeve_states.append({
                "trade_date": day, "sleeve_id": sleeve.sleeve_id,
                "cash": sleeve.cash, "market_value": sleeve_market_value,
                "equity": sleeve.cash + sleeve_market_value,
                "occupied": bool(sleeve.positions),
                "live_cohort_count": len({p.cohort_id for p in sleeve.positions.values()}),
                "position_count": len(sleeve.positions),
            })
        total_cash = float(sum(sleeve.cash for sleeve in sleeves))
        nav_rows.append({
            "trade_date": day, "cash": total_cash,
            "market_value": total_market_value,
            "total_asset": total_cash + total_market_value,
            "live_sleeves": sum(bool(sleeve.positions) for sleeve in sleeves),
        })

    frames = {
        "trades": pd.DataFrame(trades),
        "rejections": pd.DataFrame(rejections),
        "positions": pd.DataFrame(positions),
        "sleeve_states": pd.DataFrame(sleeve_states),
        "nav": pd.DataFrame(nav_rows),
        "cashflows": pd.DataFrame(cashflows),
        "costs": pd.DataFrame(cost_rows),
    }
    report = _build_report(
        frames, sleeves, config, skipped_insufficient_sleeve_cash, delayed,
        market_gate_rejected_signals, score_gate_rejected_names,
    )
    report["metrics"]["overlay_hooks"] = {
        key: overlay_hooks[key]
        for key in ("nav_dd", "early_exit", "ma5_buffer")
        if key in overlay_hooks and overlay_hooks[key] is not None
    }
    report["contract"] = (
        report["contract"] + "; risk overlay via allocation_multiplier / exit override"
    )
    return {**frames, "report": report}


def write_backtest_artifacts(
    result: dict[str, Any],
    out_dir,
    *,
    extra_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write nav/trades/report.json under 课题 result/round1|round2|round3|round4/{code}/. Never Stock/runs."""
    from pathlib import Path

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")
    outputs: dict[str, Any] = {}
    for name in ("trades", "rejections", "positions", "sleeve_states", "nav", "cashflows", "costs"):
        frame = result[name]
        path = dest / f"{name}.parquet"
        assert_project_result_path(path)
        frame.to_parquet(path, index=False)
        outputs[name] = {"rows": int(len(frame)), "sha256": sha256_file(path)}
    report = dict(result["report"])
    if extra_report:
        report.update(extra_report)
    if "nav" in result and not result["nav"].empty:
        attach_nav_metrics(
            result["nav"],
            report.setdefault("metrics", {}),
            initial_cash=float(report.get("config", {}).get("initial_cash", 300_000.0)),
        )
    report_path = dest / "report.json"
    assert_project_result_path(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {"out_dir": str(dest), "outputs": outputs, "report": report}


def default_backtest_window() -> dict[str, Any]:
    return {
        "start_date": PREDICTION_START,
        "end_date": MARKET_END,
    }
