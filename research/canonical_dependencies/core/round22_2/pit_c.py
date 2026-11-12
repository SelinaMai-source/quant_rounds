"""Track A：核对 C1/C4 冻结公式，并按 UPDATE_TIME vs D_close 做严格 PIT 时间线。"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ..paths import ROUND22_1_ROOT
from ..round21.contract import HERMES_REBUILD_RAW
from ..round21.panel import load_candidate_keys, market_to_instrument
from .common import dump_json, finite_or_none, load_json, write_csv, write_md
from .contract import (
    BLOCKED_DATA,
    BLOCKED_PIT,
    C1_FORMULA_FROZEN,
    C1_ID,
    C4_FORMULA_FROZEN,
    C4_ID,
    C_PANEL_22_1,
    FULL_HISTORY_STRICT_PIT,
    KEYS,
    LAGGED_SAFE_RESEARCH,
    MD,
    PIT_MIN_FINITE,
    RESEARCH_ONLY_BACKFILL,
    ROUND22_2_ROOT,
    SUBPERIOD_STRICT_PIT,
)
from .freeze import require_frozen_gate

PIT_JSON = ROUND22_2_ROOT / "pit" / "c_pit.json"
TIMELINE_CSV = ROUND22_2_ROOT / "pit" / "C_FACTOR_PIT_TIMELINE.csv"
MATCHED_JSON = ROUND22_2_ROOT / "matched" / "matched_sample.json"


def _formula_identity() -> dict[str, Any]:
    from ..round22_1 import recover as rec_mod

    src = inspect.getsource(rec_mod._maybe_rebuild_c)
    has_c1 = "NET_FLOW_XL" in src and "TURNOVER_VALUE" in src
    has_c4 = "shift(1).rolling(5" in src or "rolling(5" in src
    has_buy = "BUY_VALUE_XL" in src and "SELL_VALUE_XL" in src
    # 22.1 代码明确：表无 BUY_VALUE_XL，用 NET_FLOW_XL/TURNOVER_VALUE，不换高收益公式。
    return {
        "c1_frozen": C1_FORMULA_FROZEN,
        "c4_frozen": C4_FORMULA_FROZEN,
        "source": "core/round22_1/recover.py::_maybe_rebuild_c",
        "code_uses_net_flow_xl_over_turnover": has_c1,
        "code_uses_lagged_sign_mean": has_c4,
        "code_mentions_missing_buy_value_xl": has_buy,
        "formula_changed": False,
        "status": "MATCH" if has_c1 and has_c4 else "MISMATCH",
    }


def _load_mf_times() -> pd.DataFrame:
    root = HERMES_REBUILD_RAW / "mkt_equ_mf_new"
    sample = next(root.rglob("*.parquet"), None)
    if sample is None:
        raise RuntimeError("mkt_equ_mf_new missing parquet")
    cols = list(pq.read_schema(sample).names)
    need = [c for c in ("TRADE_DATE", "TICKER_SYMBOL", "EXCHANGE_CD", "UPDATE_TIME", "NET_FLOW_XL", "TURNOVER_VALUE", "NET_FLOW") if c in cols]
    dataset = ds.dataset(str(root), format="parquet")
    table = dataset.to_table(columns=need)
    frame = table.to_pandas()
    frame["trade_date"] = pd.to_datetime(frame["TRADE_DATE"], errors="coerce")
    frame["update_time"] = pd.to_datetime(frame.get("UPDATE_TIME"), errors="coerce")
    exch = frame["EXCHANGE_CD"].astype(str) if "EXCHANGE_CD" in frame.columns else pd.Series("", index=frame.index)
    tick = frame["TICKER_SYMBOL"].astype(str)
    raw_code = tick.where(~exch.isin(["XSHG", "XSHE", "XBEI", "SH", "SZ", "BJ"]), tick + "." + exch)
    frame["instrument_id"] = raw_code.map(market_to_instrument)
    return frame.dropna(subset=["trade_date", "instrument_id"])


def _year_timeline(mf: pd.DataFrame) -> list[dict[str, Any]]:
    d_close = mf["trade_date"] + pd.Timedelta(hours=15)
    lag_h = (mf["update_time"] - d_close).dt.total_seconds() / 3600.0
    same_cal = mf["update_time"].dt.normalize() == mf["trade_date"].dt.normalize()
    next_cal = mf["update_time"].dt.normalize() == (mf["trade_date"] + pd.Timedelta(days=1)).dt.normalize()
    backfill = mf["update_time"].dt.year > mf["trade_date"].dt.year + 1
    le_close = mf["update_time"] <= d_close
    le_next_close = mf["update_time"] <= (mf["trade_date"] + pd.Timedelta(days=1, hours=15))
    work = mf.assign(
        year=mf["trade_date"].dt.year,
        lag_h=lag_h,
        same_cal=same_cal,
        next_cal=next_cal,
        backfill=backfill,
        le_close=le_close,
        le_next_close=le_next_close,
    )
    rows = []
    for y, block in work.groupby("year"):
        if pd.isna(y):
            continue
        n = int(len(block))
        upd = block["update_time"]
        rows.append(
            {
                "year": int(y),
                "n_rows": n,
                "n_update_finite": int(upd.notna().sum()),
                "share_update_finite": float(upd.notna().mean()) if n else None,
                "share_update_le_d_close": float(block["le_close"].mean()) if n else None,
                "share_update_le_d1_1500": float(block["le_next_close"].mean()) if n else None,
                "share_same_calendar_day": float(block["same_cal"].mean()) if n else None,
                "share_next_calendar_day": float(block["next_cal"].mean()) if n else None,
                "share_backfill_year_plus2": float(block["backfill"].mean()) if n else None,
                "median_lag_hours_after_d_close": float(block["lag_h"].median()) if block["lag_h"].notna().any() else None,
                "p90_lag_hours_after_d_close": float(block["lag_h"].quantile(0.90)) if block["lag_h"].notna().any() else None,
                "decision_cutoff": f"{int(y)}-12-31 15:00 D_close",
                "available_at_field": "UPDATE_TIME",
            }
        )
    return rows


def _status_from_timeline(rows: list[dict[str, Any]], *, already_lagged: bool) -> str:
    formal = [r for r in rows if 2018 <= int(r["year"]) <= 2026]
    if not formal:
        return BLOCKED_PIT
    le = [finite_or_none(r.get("share_update_le_d_close")) for r in formal]
    le = [v for v in le if v is not None]
    nxt = [finite_or_none(r.get("share_update_le_d1_1500")) for r in formal]
    nxt = [v for v in nxt if v is not None]
    bf = [finite_or_none(r.get("share_backfill_year_plus2")) for r in formal]
    bf = [v for v in bf if v is not None]
    if bf and float(np.median(bf)) >= 0.20:
        return RESEARCH_ONLY_BACKFILL
    if already_lagged:
        # C4 用 D-5..D-1。若 UPDATE_TIME 多数在 T+1 15:00 前，则 D 日决策可用 T-1 及更早。
        if nxt and all(v >= 0.80 for v in nxt) and (not le or float(np.median(le)) < 0.80):
            return SUBPERIOD_STRICT_PIT if any(v < 0.80 for v in nxt[:3]) else FULL_HISTORY_STRICT_PIT
        if nxt and all(v >= 0.80 for v in nxt):
            return FULL_HISTORY_STRICT_PIT
        if nxt and any(v >= 0.80 for v in nxt):
            return SUBPERIOD_STRICT_PIT
        return LAGGED_SAFE_RESEARCH
    if le and all(v >= 0.80 for v in le):
        return FULL_HISTORY_STRICT_PIT
    if nxt and all(v >= 0.80 for v in nxt) and (not le or float(np.median(le)) < 0.80):
        return LAGGED_SAFE_RESEARCH
    if le and any(v >= 0.80 for v in le):
        return SUBPERIOD_STRICT_PIT
    if nxt and any(v >= 0.80 for v in nxt):
        return LAGGED_SAFE_RESEARCH
    return BLOCKED_PIT


def _matched_dates(mf: pd.DataFrame, *, lag_days: int, status: str) -> dict[str, Any]:
    keys = load_candidate_keys()[KEYS].drop_duplicates()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    keys["instrument_id"] = keys["instrument_id"].astype(str)
    sub = mf[["trade_date", "instrument_id", "update_time"]].dropna(subset=["trade_date", "instrument_id"])
    if lag_days > 0:
        sub = sub.sort_values(["instrument_id", "trade_date"])
        sub["src_date"] = sub.groupby("instrument_id")["trade_date"].shift(lag_days)
        sub["src_update"] = sub.groupby("instrument_id")["update_time"].shift(lag_days)
        # 决策日 D 使用 D-lag 的流水，要求该条 UPDATE_TIME ≤ D_close
        use = sub.dropna(subset=["src_date"]).copy()
        use["d_close"] = use["trade_date"] + pd.Timedelta(hours=15)
        ok = use["src_update"] <= use["d_close"]
        use = use.loc[ok]
        merged = keys.merge(use[["trade_date", "instrument_id"]], on=KEYS, how="inner")
    else:
        sub = sub.copy()
        sub["d_close"] = sub["trade_date"] + pd.Timedelta(hours=15)
        ok = sub["update_time"] <= sub["d_close"]
        merged = keys.merge(sub.loc[ok, KEYS], on=KEYS, how="inner")
    days = pd.to_datetime(merged["trade_date"]).drop_duplicates().sort_values()
    years = sorted({int(y) for y in days.dt.year.dropna()})
    n_years = (int(days.max().year) - int(days.min().year) + 1) if len(days) else 0
    train_years = max(0, n_years - 1)
    dest = ROUND22_2_ROOT / "matched" / "matched_dates.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trade_date": days}).to_parquet(dest, index=False)
    return {
        "status": status,
        "n_key_rows": int(len(merged)),
        "n_dates": int(len(days)),
        "min_date": None if days.empty else str(days.min().date()),
        "max_date": None if days.empty else str(days.max().date()),
        "years": years,
        "n_calendar_years": n_years,
        "approx_train_years": train_years,
        "lag_days": lag_days,
        "dates_path": str(dest),
        "compare_to": "retrained_matched_v5",
        "not_truncated_full_history_nav": True,
        "insufficient_history": train_years < 3,
    }


def run_pit_c(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and PIT_JSON.is_file() and TIMELINE_CSV.is_file() and MD["02"].is_file():
        existing = load_json(PIT_JSON)
        if existing.get("status") in {"DONE", "PARTIAL", BLOCKED_PIT, FULL_HISTORY_STRICT_PIT, SUBPERIOD_STRICT_PIT, LAGGED_SAFE_RESEARCH, RESEARCH_ONLY_BACKFILL}:
            return existing
    ident = _formula_identity()
    dump_json(ROUND22_2_ROOT / "diagnostics" / "c_formula_identity.json", ident)
    write_md(
        MD["01"],
        "\n".join(
            [
                "# Round22.2 · 01 C 公式身份",
                "",
                f"C1 冻结 `{C1_FORMULA_FROZEN}`。表无 BUY_VALUE_XL，不得改成别的高收益式。",
                f"C4 冻结 `{C4_FORMULA_FROZEN}`。",
                f"对照 `core/round22_1/recover.py` status=`{ident['status']}` formula_changed=`False`。",
                "本轮不发明新 C 公式。必须滞后则新版本号。",
                "",
            ]
        ),
    )
    if ident["status"] != "MATCH":
        payload = {"status": "BLOCKED", "reason": "frozen formula mismatch; refuse train", "identity": ident}
        dump_json(PIT_JSON, payload)
        return payload
    root = HERMES_REBUILD_RAW / "mkt_equ_mf_new"
    if not root.exists():
        payload = {"status": BLOCKED_DATA, "reason": "mkt_equ_mf_new missing"}
        dump_json(PIT_JSON, payload)
        return payload
    mf = _load_mf_times()
    years = _year_timeline(mf)
    write_csv(TIMELINE_CSV, years)
    dump_json(ROUND22_2_ROOT / "pit" / "c_year_update.json", {"rows": years})
    c1_status = _status_from_timeline(years, already_lagged=False)
    c4_status = _status_from_timeline(years, already_lagged=True)
    lag_c1 = 1 if c1_status == LAGGED_SAFE_RESEARCH else 0
    c1_version = "r22_2_c1_xl_net_imbalance_lag1" if lag_c1 else C1_ID
    c4_version = C4_ID
    if lag_c1:
        # 新版本号，禁止同名悄悄改期
        note_lag = "C1 同日 UPDATE_TIME 多数晚于 D_close。新版本 r22_2_c1_xl_net_imbalance_lag1 = D-1 的 NET_FLOW_XL/TURNOVER_VALUE。未改原 C1 公式名。"
    else:
        note_lag = "未改期。C1 仍用冻结公式同日值。"
    matched_c1 = _matched_dates(mf, lag_days=lag_c1, status=c1_status)
    # C4 公式已是 D-5..D-1。同日 UPDATE_TIME 晚于 D_close 时，用 D-1 条的 UPDATE_TIME ≤ 决策日 D_close。
    matched_c4 = _matched_dates(mf, lag_days=max(1, 0 if c4_status == FULL_HISTORY_STRICT_PIT else 1), status=c4_status)
    # 联合 matched：C1 与 C4 都可用的日期
    d1 = pd.read_parquet(matched_c1["dates_path"])
    d4 = pd.read_parquet(matched_c4["dates_path"])
    both = set(pd.to_datetime(d1["trade_date"])) & set(pd.to_datetime(d4["trade_date"]))
    both_days = pd.DatetimeIndex(sorted(both))
    both_path = ROUND22_2_ROOT / "matched" / "matched_dates_c1_c4.parquet"
    pd.DataFrame({"trade_date": both_days}).to_parquet(both_path, index=False)
    union_status = c1_status if c1_status == c4_status else SUBPERIOD_STRICT_PIT
    if RESEARCH_ONLY_BACKFILL in {c1_status, c4_status}:
        union_status = RESEARCH_ONLY_BACKFILL
    if BLOCKED_PIT in {c1_status, c4_status} and union_status != RESEARCH_ONLY_BACKFILL:
        union_status = BLOCKED_PIT
    panel_ok = C_PANEL_22_1.is_file()
    payload = {
        "status": union_status,
        "passed_strict_full_history": union_status == FULL_HISTORY_STRICT_PIT,
        "identity": ident,
        "c1": {
            "factor_id": C1_ID,
            "model_id": c1_version,
            "pit_status": c1_status,
            "lag_days": lag_c1,
            "formula": C1_FORMULA_FROZEN if lag_c1 == 0 else f"lag1({C1_FORMULA_FROZEN})",
            "silent_rename": False,
            "note": note_lag,
        },
        "c4": {
            "factor_id": C4_ID,
            "model_id": c4_version,
            "pit_status": c4_status,
            "lag_days": 0,
            "formula": C4_FORMULA_FROZEN,
            "silent_rename": False,
        },
        "timeline_csv": str(TIMELINE_CSV),
        "n_mf_rows": int(len(mf)),
        "matched_c1": matched_c1,
        "matched_c4": matched_c4,
        "matched_both": {
            "n_dates": int(len(both_days)),
            "min_date": None if both_days.empty else str(both_days.min().date()),
            "max_date": None if both_days.empty else str(both_days.max().date()),
            "path": str(both_path),
            "insufficient_history": (int(both_days.max().year) - int(both_days.min().year) + 1) < 4 if len(both_days) else True,
        },
        "c_panel_22_1": str(C_PANEL_22_1),
        "c_panel_exists": panel_ok,
        "do_not_compare_short_window_to_full_v5": True,
        "coverage_only_is_not_strict_pit": True,
        "finite_share_min": PIT_MIN_FINITE,
    }
    dump_json(PIT_JSON, payload)
    dump_json(MATCHED_JSON, {"c1": matched_c1, "c4": matched_c4, "both": payload["matched_both"]})
    write_md(
        MD["02"],
        "\n".join(
            [
                "# Round22.2 · 02 C PIT 时间线",
                "",
                f"禁止只因 `trade_date ≤ decision_date` 判 PASS。本轮用 `UPDATE_TIME` vs `D_close=15:00`。",
                f"C1 状态 `{c1_status}` 模型 id `{c1_version}` lag=`{lag_c1}`。",
                f"C4 状态 `{c4_status}`。C4 公式本身已是 D-5..D-1。",
                f"时间线 `{TIMELINE_CSV}`。",
                note_lag,
                "",
            ]
        ),
    )
    write_md(
        MD["03"],
        "\n".join(
            [
                "# Round22.2 · 03 Matched PIT Sample",
                "",
                f"C1 matched n_dates=`{matched_c1.get('n_dates')}` {matched_c1.get('min_date')}–{matched_c1.get('max_date')} train_years≈`{matched_c1.get('approx_train_years')}` insufficient=`{matched_c1.get('insufficient_history')}`。",
                f"C4 matched n_dates=`{matched_c4.get('n_dates')}`。联合 `{len(both_days)}` 日。",
                "对照是**重新训练的 matched v5**，不是截断全历史 NAV。不足 3 年训练标 INSUFFICIENT_HISTORY / NON_CANONICAL_EXPLORATORY。",
                "禁止把短窗 Sharpe 和 2018–2026 v5 混表。",
                "",
            ]
        ),
    )
    return payload
