"""Track C：用已有 pred 重建 Top1/Top2 slot。新增字段不得改 NAV。"""

from __future__ import annotations

import shutil
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND22_1_ROOT, ROUND22_ROOT, assert_project_result_path
from ..round11.common import daily_from_nav
from .common import close_enough, dump_json, file_sha256, finite_or_none, load_json, write_csv, write_md
from .contract import (
    AUDIT_FAILURE,
    COMPLETE,
    E0_CODE,
    FAILED,
    FEE_SCHEDULE,
    FORMAL_SLEEVE_CODES,
    MD,
    NAV_END_ABS_TOL,
    PARITY_METRIC_ABS,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    exp_dir,
)
from .freeze import require_frozen_gate

LEDGER_JSON = ROUND22_1_ROOT / "ledger" / "ledger.json"


def _net_cash(frame: pd.DataFrame) -> pd.Series:
    sell = pd.to_numeric(frame.get("sell_proceeds"), errors="coerce").fillna(0.0)
    div = pd.to_numeric(frame.get("cash_dividend"), errors="coerce").fillna(0.0)
    buy = pd.to_numeric(frame.get("buy_cost"), errors="coerce").fillna(0.0)
    return sell + div - buy


def _rank_from_overlay(overlay: pd.DataFrame) -> pd.DataFrame:
    work = overlay.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work["raw_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    if "allocation_multiplier" in work.columns:
        alloc = pd.to_numeric(work["allocation_multiplier"], errors="coerce")
        eligible = work.loc[alloc.fillna(0) > 0].copy()
    else:
        eligible = work.copy()
    eligible["eligible_rank"] = eligible.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    keep = eligible[["signal_date", "instrument_id", "raw_rank", "eligible_rank", "score"]].rename(columns={"score": "overlay_score"})
    raw = work[["signal_date", "instrument_id", "raw_rank"]].rename(columns={"raw_rank": "raw_rank_all"})
    out = work[["signal_date", "instrument_id"]].merge(keep, on=["signal_date", "instrument_id"], how="left")
    out = out.merge(raw, on=["signal_date", "instrument_id"], how="left")
    out["raw_rank"] = out["raw_rank"].fillna(out["raw_rank_all"])
    slot = np.where(out["eligible_rank"].eq(1), "TOP1", np.where(out["eligible_rank"].eq(2), "TOP2", "OTHER"))
    out["selected_slot"] = slot
    return out[["signal_date", "instrument_id", "raw_rank", "eligible_rank", "selected_slot", "overlay_score"]]


def _build_overlay_from_pred(src) -> pd.DataFrame | None:
    five_p = src / "five_pred.parquet"
    ten_p = src / "ten_pred.parquet"
    ov_p = src / "overlayed.parquet"
    pred_p = src / "predictions.parquet"
    if ov_p.is_file() and ov_p.stat().st_size > 8:
        return pd.read_parquet(ov_p)
    if five_p.is_file() and ten_p.is_file():
        from ..round15.runner import apply_top2_veto
        from ..round16_5.engine import o01_signal_dates
        from ..round17.backtest import apply_b03, fuse_heads

        five = pd.read_parquet(five_p)
        ten = pd.read_parquet(ten_p)
        fused = fuse_heads(five, ten)
        overlayed, _ = apply_b03(fused)
        return apply_top2_veto(overlayed, o01_signal_dates())
    if pred_p.is_file():
        from ..round15.runner import apply_top2_veto
        from ..round16_5.engine import o01_signal_dates
        from ..round17.backtest import apply_b03

        fused = pd.read_parquet(pred_p)
        overlayed, _ = apply_b03(fused)
        return apply_top2_veto(overlayed, o01_signal_dates())
    return None


def _augment(code: str) -> dict[str, Any]:
    src = ROUND22_ROOT / "candidates" / code
    dest = exp_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "trades.parquet")
    trades_src = src / "trades.parquet"
    nav_src = src / "nav.parquet"
    if not trades_src.is_file():
        # E0 may live under parity
        alt = ROUND22_ROOT / "parity" / "R22_V5_R16_O01" / "trades.parquet"
        if code == E0_CODE and alt.is_file():
            trades_src = alt
            nav_src = ROUND22_ROOT / "parity" / "R22_V5_R16_O01" / "nav.parquet"
        else:
            return {"code": code, "status": "MISSING_TRADES"}
    trades = pd.read_parquet(trades_src)
    orig_cols = list(trades.columns)
    overlay = _build_overlay_from_pred(src)
    if overlay is None and code == E0_CODE:
        ov_path = ROUND22_ROOT / "weights" / "overlayed_R22_V5_R16_O01.parquet"
        if ov_path.is_file():
            overlay = pd.read_parquet(ov_path)
        else:
            from ..round15.runner import apply_top2_veto
            from ..round16_5.engine import o01_signal_dates
            from ..round17.backtest import cached_overlay_v4

            overlay = apply_top2_veto(cached_overlay_v4(), o01_signal_dates())
    if overlay is None:
        return {"code": code, "status": "MISSING_PRED"}
    ranks = _rank_from_overlay(overlay)
    work = trades.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    merged = work.merge(ranks, on=["signal_date", "instrument_id"], how="left")
    merged["net_ret"] = _net_cash(merged)
    hashes = load_json(src / "prediction_hash.json")
    feats = load_json(src / "model_hash.json")
    merged["model_hash"] = (hashes.get("five_pred") or "") + "/" + (hashes.get("ten_pred") or "")
    merged["feature_hash"] = (feats.get("five_feature_set_hash") or "") + "/" + (feats.get("ten_feature_set_hash") or "")
    # 禁止改成交价/仓位/退出/费率
    for col in orig_cols:
        if col in {"score"}:
            continue
        if col in merged.columns and col in work.columns:
            if pd.api.types.is_numeric_dtype(work[col]):
                if not np.allclose(
                    pd.to_numeric(merged[col], errors="coerce").fillna(0).to_numpy(dtype=float),
                    pd.to_numeric(work[col], errors="coerce").fillna(0).to_numpy(dtype=float),
                    atol=1e-9,
                    equal_nan=True,
                ):
                    raise RuntimeError(f"{code} ledger mutated column {col}")
    merged.to_parquet(dest / "trades.parquet", index=False)
    if nav_src.is_file():
        shutil.copy2(nav_src, dest / "nav.parquet")
    for name in ("report.json", "summary.json", "five_pred.parquet", "ten_pred.parquet", "prediction_hash.json", "model_hash.json", "feature_list.json"):
        sp = src / name
        if sp.is_file():
            shutil.copy2(sp, dest / name)
    n_top1 = int((merged["selected_slot"] == "TOP1").sum())
    n_top2 = int((merged["selected_slot"] == "TOP2").sum())
    n_other = int((merged["selected_slot"] == "OTHER").sum())
    n_slot_na = int(merged["selected_slot"].isna().sum()) if "selected_slot" in merged.columns else 0
    nav_end = None
    sharpe = None
    if (dest / "nav.parquet").is_file():
        nav = pd.read_parquet(dest / "nav.parquet")
        nav_end = float(nav["total_asset"].iloc[-1]) if "total_asset" in nav.columns and len(nav) else None
        daily = daily_from_nav(nav) if "total_asset" in nav.columns else None
        if daily is not None and len(daily) > 2:
            vol = float(daily.std(ddof=1))
            sharpe = float(daily.mean() / vol * np.sqrt(252.0)) if vol > 0 else None
    src_nav_end = None
    if nav_src.is_file():
        sn = pd.read_parquet(nav_src)
        src_nav_end = float(sn["total_asset"].iloc[-1]) if "total_asset" in sn.columns and len(sn) else None
    nav_unchanged = close_enough(nav_end, src_nav_end, abs_tol=NAV_END_ABS_TOL) if src_nav_end is not None else False
    return {
        "code": code,
        "status": COMPLETE,
        "n_trades": int(len(merged)),
        "n_top1": n_top1,
        "n_top2": n_top2,
        "n_other": n_other,
        "n_slot_na": n_slot_na,
        "nav_end": nav_end,
        "src_nav_end": src_nav_end,
        "nav_unchanged": bool(nav_unchanged),
        "sharpe": sharpe,
        "five_pred": file_sha256(src / "five_pred.parquet") if (src / "five_pred.parquet").is_file() else hashes.get("five_pred"),
        "ten_pred": file_sha256(src / "ten_pred.parquet") if (src / "ten_pred.parquet").is_file() else hashes.get("ten_pred"),
        "replay_only": True,
        "retrained": False,
    }


def _e0_parity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    e0 = next((r for r in rows if r.get("code") == E0_CODE), None)
    if e0 is None:
        return {"status": FAILED, "reason": "E0 missing"}
    dest = exp_dir(E0_CODE)
    src = ROUND22_ROOT / "candidates" / E0_CODE
    if not (src / "trades.parquet").is_file():
        src = ROUND22_ROOT / "parity" / "R22_V5_R16_O01"
    same_orders = False
    if (dest / "trades.parquet").is_file() and (src / "trades.parquet").is_file():
        a = pd.read_parquet(dest / "trades.parquet")
        b = pd.read_parquet(src / "trades.parquet")
        ka = set(zip(pd.to_datetime(a["signal_date"]).astype(str), a["instrument_id"].astype(str)))
        kb = set(zip(pd.to_datetime(b["signal_date"]).astype(str), b["instrument_id"].astype(str)))
        same_orders = ka == kb
    same_nav = close_enough(e0.get("nav_end"), V5_NAV_END_EXACT, abs_tol=NAV_END_ABS_TOL)
    same_sharpe = close_enough(e0.get("sharpe"), V5_ANCHOR["sharpe"], abs_tol=PARITY_METRIC_ABS)
    return {
        "status": "PASS" if same_orders and same_nav and same_sharpe else "FAIL",
        "same_orders": same_orders,
        "same_nav": same_nav,
        "same_sharpe": same_sharpe,
        "nav_end": e0.get("nav_end"),
        "sharpe": e0.get("sharpe"),
    }


def _nav_bridge(code: str) -> dict[str, Any]:
    dest = exp_dir(code)
    nav_p = dest / "nav.parquet"
    tr_p = dest / "trades.parquet"
    if not nav_p.is_file() or not tr_p.is_file():
        return {"code": code, "status": AUDIT_FAILURE, "reason": "missing nav/trades"}
    nav = pd.read_parquet(nav_p)
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    nav = nav.sort_values("trade_date")
    trades = pd.read_parquet(tr_p)
    start = float(nav["total_asset"].iloc[0])
    end = float(nav["total_asset"].iloc[-1])
    realized = float(_net_cash(trades).sum())
    # 日频：用 nav 的 total_asset 差分核对
    nav = nav.copy()
    nav["d_nav"] = nav["total_asset"].diff()
    if "cash" in nav.columns and "market_value" in nav.columns:
        last_unreal = float(nav["market_value"].iloc[-1])
        first_unreal = float(nav["market_value"].iloc[0])
    else:
        last_unreal = 0.0
        first_unreal = 0.0
    reconstructed = start + realized + (last_unreal - first_unreal)
    gap = end - reconstructed
    ok = abs(gap) <= max(1.0, 1e-4 * abs(end))
    # 日频 merge 诊断
    daily = daily_from_nav(nav)
    return {
        "code": code,
        "status": COMPLETE if ok else AUDIT_FAILURE,
        "start": start,
        "end": end,
        "realized_trade_pnl": realized,
        "unrealized_end": last_unreal,
        "unrealized_start": first_unreal,
        "reconstructed": reconstructed,
        "gap": gap,
        "n_nav_days": int(len(nav)),
        "n_daily_ret": int(len(daily)),
        "formula": "start + (sell_proceeds+cash_dividend-buy_cost) + Δunrealized = end",
    }


def _top_net(code: str, base_code: str = E0_CODE) -> dict[str, Any]:
    dest = exp_dir(code)
    base = exp_dir(base_code)
    if not (dest / "trades.parquet").is_file() or not (base / "trades.parquet").is_file():
        return {"code": code, "top_net_ok": None, "reason": "missing augmented trades", "was_missing_columns": True}
    a = pd.read_parquet(dest / "trades.parquet")
    b = pd.read_parquet(base / "trades.parquet")
    if "net_ret" not in a.columns or "selected_slot" not in a.columns:
        return {"code": code, "top_net_ok": False, "reason": "missing columns after augment", "was_missing_columns": True}
    a_top = a.loc[a["selected_slot"].isin(["TOP1", "TOP2"])]
    b_top = b.loc[b["selected_slot"].isin(["TOP1", "TOP2"])] if "selected_slot" in b.columns else b
    na = float(pd.to_numeric(a_top["net_ret"], errors="coerce").sum()) if len(a_top) else None
    nb = float(pd.to_numeric(b_top["net_ret"], errors="coerce").sum()) if len(b_top) else None
    ok = None if na is None or nb is None else bool(na + 1e-9 >= nb or code == base_code)
    return {
        "code": code,
        "top1_n": int((a["selected_slot"] == "TOP1").sum()),
        "top2_n": int((a["selected_slot"] == "TOP2").sum()),
        "top_net_sum": na,
        "base_top_net_sum": nb,
        "top_net_ok": True if code == base_code else ok,
        "was_missing_columns": False,
        "round22_false_reason": "trades.parquet 无 rank/slot/net_ret，stats._net 找不到列",
    }


def run_ledger(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and LEDGER_JSON.is_file() and MD["03"].is_file():
        existing = load_json(LEDGER_JSON)
        if existing.get("status") in {COMPLETE, "DONE"}:
            return existing
    rows = []
    for code in FORMAL_SLEEVE_CODES:
        rows.append(_augment(code))
    e0 = _e0_parity(rows)
    bridges = [_nav_bridge(r["code"]) for r in rows if r.get("status") == COMPLETE]
    nets = [_top_net(r["code"]) for r in rows if r.get("status") == COMPLETE]
    write_csv(ROUND22_1_ROOT / "ledger" / "sleeve_slots.csv", rows)
    write_csv(ROUND22_1_ROOT / "ledger" / "nav_bridge.csv", bridges)
    write_csv(ROUND22_1_ROOT / "ledger" / "top_net.csv", nets)
    # 费用两口径：只复制 round22 已有分表，不重跑
    fee_src = ROUND22_ROOT / "backtest" / "fee_both_calibers.csv"
    if fee_src.is_file():
        shutil.copy2(fee_src, ROUND22_1_ROOT / "ledger" / "fee_both_calibers.csv")
    n_ok = int(sum(r.get("nav_unchanged") for r in rows if r.get("nav_unchanged") is not None))
    payload = {
        "status": COMPLETE if e0.get("status") == "PASS" and n_ok == len([r for r in rows if r.get("status") == COMPLETE]) else "PARTIAL",
        "e0_parity": e0,
        "n": len(rows),
        "n_nav_unchanged": n_ok,
        "retrained": False,
        "replay_only": True,
        "nav_performance_changed": False,
        "note": "账本字段补齐、绩效未改",
        "rows": rows,
        "bridges": bridges,
        "top_net": nets,
        "fee_schedule_main": FEE_SCHEDULE,
        "fee_schedule_alt": "HISTORICAL_FEE_CORRECTED",
    }
    dump_json(LEDGER_JSON, payload)
    write_md(
        MD["03"],
        "\n".join(
            [
                "# Round22.1 · 03 Top1/Top2 账本",
                "",
                "用已有 `five_pred`/`ten_pred` → `fuse_heads` → B03 → O01 veto 写 `raw_rank` / `eligible_rank` / `selected_slot∈{TOP1,TOP2,OTHER}`。",
                "净收益 = `sell_proceeds + cash_dividend - buy_cost`（已含佣金/印花税/过户）。禁止 `exit/entry-1`。",
                f"E0 Same Orders=`{e0.get('same_orders')}` Same NAV=`{e0.get('same_nav')}` Same Sharpe=`{e0.get('same_sharpe')}`。",
                f"正式袖套 {len(rows)} 条，NAV 未改 {n_ok} 条。未重训。",
                "Round22 `top_net_ok=False` 是缺 `rank/slot/net_ret` 列，不是已算但不合格。",
                "",
            ]
        ),
    )
    n_fail = int(sum(b.get("status") == AUDIT_FAILURE for b in bridges))
    write_md(
        MD["08"],
        f"# Round22.1 · 08 NAV 桥\n\n日频 merge 复用 21.4 思路，输出在 round22_1。对不上先 `{AUDIT_FAILURE}`。失败条数={n_fail}。公式：期初 + 已实现交易损益 + 分红 − 费用 ± 未实现 = 期末。\n",
    )
    return payload
