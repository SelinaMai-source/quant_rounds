"""引擎 invariant / impossible fill。涨停排队只做 NO_FILL/PARTIAL 诊断，不调策略。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND11_ROOT
from ..stock_readonly import load_prepared_market
from .common import dump_json, load_anchor_nav, load_anchor_trades, stage_dir, write_md
from .contract import BASELINES

ENGINE_MD = ROUND11_ROOT / "03_engine_invariants.md"
NAV_TOL = 1e-4
PRICE_TOL = 1e-6


def _finite(value: Any) -> bool:
    try:
        return pd.notna(value) and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _nav_identity(nav: pd.DataFrame) -> dict[str, Any]:
    n = nav.copy()
    cash = pd.to_numeric(n["cash"], errors="coerce")
    mv = pd.to_numeric(n["market_value"], errors="coerce")
    tot = pd.to_numeric(n["total_asset"], errors="coerce")
    delta = (cash + mv - tot).abs()
    return {
        "rows": int(len(n)),
        "max_abs_cash_plus_mv_minus_nav": float(delta.max()) if len(delta) else None,
        "n_break": int((delta > NAV_TOL).sum()),
        "passed": bool(len(delta) and float(delta.max()) <= NAV_TOL),
    }


def _lookup_bar(market_lookup: pd.DataFrame, day, instrument_id: str):
    key = (pd.Timestamp(day), str(instrument_id))
    try:
        row = market_lookup.loc[key]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return None
        return row.iloc[0]
    return row


def _trade_diagnostics(trades: pd.DataFrame, market_lookup: pd.DataFrame) -> dict[str, Any]:
    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.normalize()
    t["actual_exit_date"] = pd.to_datetime(t["actual_exit_date"]).dt.normalize()
    t["instrument_id"] = t["instrument_id"].astype(str)
    impossible = []
    limit_up_buys = 0
    paused_buys = 0
    price_out = 0
    for rec in t.itertuples(index=False):
        buy = _lookup_bar(market_lookup, rec.entry_date, rec.instrument_id)
        sell = _lookup_bar(market_lookup, rec.actual_exit_date, rec.instrument_id)
        flags = []
        if buy is None:
            flags.append("missing_buy_bar")
        else:
            if bool(buy["paused"]) if pd.notna(buy.get("paused")) else False:
                paused_buys += 1
                flags.append("buy_while_paused")
            if _finite(buy.get("limit_up")) and _finite(buy.get("open")):
                if float(buy["open"]) >= float(buy["limit_up"]) - 0.005:
                    limit_up_buys += 1
                    flags.append("buy_open_limit_up")
            px = float(rec.entry_price)
            lo = float(buy["low"]) if _finite(buy.get("low")) else None
            hi = float(buy["high"]) if _finite(buy.get("high")) else None
            if lo is not None and hi is not None and (px < lo - PRICE_TOL or px > hi + PRICE_TOL):
                price_out += 1
                flags.append("buy_price_outside_high_low")
        if sell is not None:
            px = float(rec.exit_price)
            lo = float(sell["low"]) if _finite(sell.get("low")) else None
            hi = float(sell["high"]) if _finite(sell.get("high")) else None
            if lo is not None and hi is not None and (px < lo - PRICE_TOL or px > hi + PRICE_TOL):
                price_out += 1
                flags.append("sell_price_outside_high_low")
            if bool(sell["paused"]) if pd.notna(sell.get("paused")) else False:
                flags.append("sell_while_paused")
        if flags:
            impossible.append({
                "signal_date": str(pd.Timestamp(rec.signal_date).date()),
                "instrument_id": rec.instrument_id,
                "flags": flags,
            })
    return {
        "n_trades": int(len(t)),
        "n_flagged": int(len(impossible)),
        "buy_open_limit_up": int(limit_up_buys),
        "buy_while_paused": int(paused_buys),
        "price_outside_high_low": int(price_out),
        "samples": impossible[:20],
        "passed": bool(len(impossible) == 0),
    }


def _rejection_queue(rejections: pd.DataFrame | None) -> dict[str, Any]:
    if rejections is None or rejections.empty:
        return {
            "status": "NO_REJECTION_FILE_OR_EMPTY",
            "limit_up_buy": "NO_FILL_OR_UNKNOWN",
            "note": "涨停排队只做诊断，不调策略。无逐笔队列数据 = 不能证 PARTIAL。",
        }
    r = rejections.copy()
    reasons = r["reason"].astype(str).value_counts().to_dict() if "reason" in r.columns else {}
    events = r["event"].astype(str).value_counts().to_dict() if "event" in r.columns else {}
    return {
        "n": int(len(r)),
        "reasons": {str(k): int(v) for k, v in reasons.items()},
        "events": {str(k): int(v) for k, v in events.items()},
        "limit_up_diagnostic": "NO_FILL",
        "partial_fill_engine": False,
        "note": "引擎对开盘涨停是 NO_FILL（open_limit_up），无 PARTIAL 队列模型。不调策略。",
    }


def run_engine(**_: Any) -> dict[str, Any]:
    dest = stage_dir("engine")
    loaded = load_prepared_market(warmup_ma5=True)
    market_lookup = loaded["prepared"].market_lookup
    baselines: dict[str, Any] = {}
    all_nav_ok = True
    any_impossible = False
    for key, spec in BASELINES.items():
        nav = load_anchor_nav(spec)
        trades = load_anchor_trades(spec)
        nav_id = _nav_identity(nav)
        fills = _trade_diagnostics(trades, market_lookup)
        rej_path = spec.anchor_dir / "rejections.parquet"
        rejections = pd.read_parquet(rej_path) if rej_path.is_file() else None
        queue = _rejection_queue(rejections)
        if not nav_id["passed"]:
            all_nav_ok = False
        if not fills["passed"]:
            any_impossible = True
        baselines[key] = {
            "code": spec.code,
            "nav_identity": nav_id,
            "fills": fills,
            "queue": queue,
            "status": "PASS" if nav_id["passed"] and fills["passed"] else "BUG_DIAGNOSTIC",
        }
        dump_json(dest / f"{key}_engine.json", baselines[key])
    payload = {
        "status": "PASS" if all_nav_ok and not any_impossible else "BUG_DIAGNOSTIC",
        "nav_identity_all_pass": all_nav_ok,
        "impossible_fill_found": any_impossible,
        "baselines": baselines,
        "note": (
            "发现引擎问题只记 BUG_DIAGNOSTIC，不静默覆盖 legacy。"
            "本轮不为过关改策略。涨停排队无盘口 = NO_FILL diagnostic。"
        ),
        "post_hoc_diagnostic": True,
    }
    dump_json(dest / "engine.json", payload)
    lines = [
        "# Round 11 · 引擎 invariant / impossible fill",
        "",
        f"总判定：**{payload['status']}**。NAV = 现金 + 市值，容差 `{NAV_TOL:g}`。",
        "涨停排队只记 `NO_FILL` / 无 PARTIAL 模型。不调策略。",
        "",
        "| 版本 | NAV max|Δ| | NAV 破行 | 成交笔数 | 异常旗标 | 开盘涨停买 | 价出高低 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, item in baselines.items():
        nav = item["nav_identity"]
        fills = item["fills"]
        lines.append(
            "| {k} | {d} | {nb} | {nt} | {nf} | {lu} | {po} | {st} |".format(
                k=key,
                d=nav.get("max_abs_cash_plus_mv_minus_nav"),
                nb=nav.get("n_break"),
                nt=fills.get("n_trades"),
                nf=fills.get("n_flagged"),
                lu=fills.get("buy_open_limit_up"),
                po=fills.get("price_outside_high_low"),
                st=item.get("status"),
            )
        )
    lines += [
        "",
        "## 规则",
        "",
        "- 价出高低 / 停牌成交 / 一字涨停买入 = impossible fill，记 `BUG_DIAGNOSTIC`",
        "- 不覆盖 legacy 账本，不把诊断重放写成正式数字",
        "- 无逐笔队列：不能声称 PARTIAL fill 已被建模",
        "",
    ]
    write_md(ENGINE_MD, "\n".join(lines))
    return payload


__all__ = ["run_engine"]
