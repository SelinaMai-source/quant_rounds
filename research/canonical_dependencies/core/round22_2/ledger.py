"""Top1/Top2 真账本。复用 22.1 slot 字段。净收益 = sell_proceeds+cash_dividend−buy_cost。禁止因缺列写 top_net_ok=False。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import assert_project_result_path
from ..round11.common import daily_from_nav
from .common import close_enough, dump_json, file_sha256, finite_or_none, load_json, write_csv, write_md
from .contract import AUDIT_FAILURE, COMPLETE, E0_CODE, FAILED, MD, NAV_END_ABS_TOL, ROUND22_2_ROOT, exp_dir
from .freeze import require_frozen_gate

LEDGER_JSON = ROUND22_2_ROOT / "ledger" / "ledger.json"


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


def _build_overlay(src) -> pd.DataFrame | None:
    five_p = src / "five_pred.parquet"
    ten_p = src / "ten_pred.parquet"
    ov_p = src / "overlayed.parquet"
    if ov_p.is_file() and ov_p.stat().st_size > 8:
        return pd.read_parquet(ov_p)
    if five_p.is_file() and ten_p.is_file():
        from ..round15.runner import apply_top2_veto
        from ..round16_5.engine import o01_signal_dates
        from ..round17.backtest import apply_b03, fuse_heads

        fused = fuse_heads(pd.read_parquet(five_p), pd.read_parquet(ten_p))
        overlayed, _ = apply_b03(fused)
        return apply_top2_veto(overlayed, o01_signal_dates())
    return None


def _augment(code: str) -> dict[str, Any]:
    dest = exp_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "trades.parquet")
    trades_src = dest / "trades.parquet"
    if not trades_src.is_file():
        return {"code": code, "status": "MISSING_TRADES", "top_net_ok": None, "was_missing_columns": False, "reason": "no trades yet"}
    trades = pd.read_parquet(trades_src)
    orig_cols = list(trades.columns)
    overlay = _build_overlay(dest)
    if overlay is None:
        return {"code": code, "status": "MISSING_PRED", "top_net_ok": None, "was_missing_columns": False, "reason": "no overlay/pred"}
    ranks = _rank_from_overlay(overlay)
    work = trades.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    merged = work.merge(ranks, on=["signal_date", "instrument_id"], how="left")
    merged["net_ret"] = _net_cash(merged)
    hashes = load_json(dest / "prediction_hash.json")
    feats = load_json(dest / "model_hash.json")
    merged["model_hash"] = (hashes.get("five_pred") or "") + "/" + (hashes.get("ten_pred") or "")
    merged["feature_hash"] = (feats.get("five_feature_set_hash") or "") + "/" + (feats.get("ten_feature_set_hash") or "")
    for col in orig_cols:
        if col in {"score"}:
            continue
        if col in merged.columns and col in work.columns and pd.api.types.is_numeric_dtype(work[col]):
            if not np.allclose(
                pd.to_numeric(merged[col], errors="coerce").fillna(0).to_numpy(dtype=float),
                pd.to_numeric(work[col], errors="coerce").fillna(0).to_numpy(dtype=float),
                atol=1e-9,
                equal_nan=True,
            ):
                raise RuntimeError(f"{code} ledger mutated column {col}")
    merged.to_parquet(dest / "trades.parquet", index=False)
    return {
        "code": code,
        "status": COMPLETE,
        "n_trades": int(len(merged)),
        "n_top1": int((merged["selected_slot"] == "TOP1").sum()),
        "n_top2": int((merged["selected_slot"] == "TOP2").sum()),
        "has_slot": True,
        "has_net_ret": True,
        "was_missing_columns": False,
        "five_pred": file_sha256(dest / "five_pred.parquet"),
        "ten_pred": file_sha256(dest / "ten_pred.parquet"),
    }


def _top_net(code: str, base_code: str = E0_CODE) -> dict[str, Any]:
    dest = exp_dir(code)
    base = exp_dir(base_code)
    if not (dest / "trades.parquet").is_file() or not (base / "trades.parquet").is_file():
        return {"code": code, "top_net_ok": None, "reason": "missing trades", "was_missing_columns": False}
    a = pd.read_parquet(dest / "trades.parquet")
    b = pd.read_parquet(base / "trades.parquet")
    need = {"net_ret", "selected_slot"}
    if not need.issubset(a.columns) or not need.issubset(b.columns):
        return {"code": code, "top_net_ok": None, "reason": "slot/net_ret not yet built; not a failed economic test", "was_missing_columns": True}
    a_top = a.loc[a["selected_slot"].isin(["TOP1", "TOP2"])]
    b_top = b.loc[b["selected_slot"].isin(["TOP1", "TOP2"])]
    na = float(pd.to_numeric(a_top["net_ret"], errors="coerce").sum()) if len(a_top) else None
    nb = float(pd.to_numeric(b_top["net_ret"], errors="coerce").sum()) if len(b_top) else None
    ok = na is not None and nb is not None and na + 1e-9 >= nb
    return {
        "code": code,
        "top_net_ok": bool(ok),
        "cand_top_net": na,
        "base_top_net": nb,
        "delta_top_net": None if na is None or nb is None else na - nb,
        "formula": "sell_proceeds + cash_dividend - buy_cost",
        "was_missing_columns": False,
    }


def _nav_bridge(code: str) -> dict[str, Any]:
    dest = exp_dir(code)
    if not (dest / "nav.parquet").is_file() or not (dest / "trades.parquet").is_file():
        return {"code": code, "status": AUDIT_FAILURE, "reason": "missing nav/trades"}
    nav = pd.read_parquet(dest / "nav.parquet")
    trades = pd.read_parquet(dest / "trades.parquet")
    start = float(nav["total_asset"].iloc[0])
    end = float(nav["total_asset"].iloc[-1])
    realized = float(_net_cash(trades).sum())
    if "market_value" in nav.columns:
        last_unreal = float(nav["market_value"].iloc[-1])
        first_unreal = float(nav["market_value"].iloc[0])
    else:
        last_unreal = 0.0
        first_unreal = 0.0
    reconstructed = start + realized + (last_unreal - first_unreal)
    gap = end - reconstructed
    ok = abs(gap) <= max(1.0, 1e-4 * abs(end))
    return {
        "code": code,
        "status": COMPLETE if ok else AUDIT_FAILURE,
        "start": start,
        "end": end,
        "realized_trade_pnl": realized,
        "reconstructed": reconstructed,
        "gap": gap,
        "formula": "start + (sell_proceeds+cash_dividend-buy_cost) + Δunrealized = end",
    }


def run_ledger(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and LEDGER_JSON.is_file() and MD["16"].is_file():
        existing = load_json(LEDGER_JSON)
        if existing.get("status") in {"DONE", "PARTIAL", "NOT_RUN"}:
            return existing
    sleeve = load_json(ROUND22_2_ROOT / "backtest" / "sleeve.json")
    rows_in = sleeve.get("rows") or []
    codes = [r.get("code") for r in rows_in if r.get("code")]
    if not codes:
        payload = {"status": "NOT_RUN", "reason": "no formal sleeves", "n": 0}
        dump_json(LEDGER_JSON, payload)
        write_md(MD["16"], "# Round22.2 · 16 Top1/Top2\n\n无正式袖套，未写 top_net_ok=False。缺列 ≠ 已算不合格。\n")
        return payload
    aug = [_augment(c) for c in codes]
    tops = [_top_net(c) for c in codes if c != E0_CODE]
    bridges = [_nav_bridge(c) for c in codes]
    write_csv(ROUND22_2_ROOT / "ledger" / "top_net.csv", tops)
    write_csv(ROUND22_2_ROOT / "ledger" / "nav_bridge.csv", bridges)
    payload = {
        "status": "DONE",
        "n": len(codes),
        "augment": aug,
        "tops": tops,
        "bridges": bridges,
        "missing_column_false_forbidden": True,
        "n_missing_columns": int(sum(bool(t.get("was_missing_columns")) for t in tops)),
        "n_top_net_ok": int(sum(t.get("top_net_ok") is True for t in tops)),
        "n_bridge_fail": int(sum(b.get("status") == AUDIT_FAILURE for b in bridges)),
    }
    dump_json(LEDGER_JSON, payload)
    write_md(MD["16"], f"# Round22.2 · 16 Top1/Top2\n\n复用 22.1 slot 字段。净收益 = sell_proceeds+cash_dividend−buy_cost。top_net_ok True={payload['n_top_net_ok']}。缺列条数={payload['n_missing_columns']}（缺列不得解释为已算不合格）。\n")
    return payload
