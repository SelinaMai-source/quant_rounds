"""二板特征。缺分钟、行业时降级，不把缺失填成 0。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from core.limit_rules import ST_CUTOVER, board_of, is_st_name, limit_pct
from core.r225241_io import dump_json, write_parquet

from .contract import FEATURE_COVERAGE_MIN, FULL_FEATURES, REDUCED_FEATURES, ROOT, ROUND24

HERMES = Path("/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features")
SLIM = ROUND24 / "data" / "daily_slim.parquet"
CAND = ROUND24 / "data" / "second_board.parquet"
MINUTE_ROOTS = (
    Path("/data1/stock-mxy/data/source_snapshots/amazingdata-cn-a-close-limit-min1-20130104-20211231-v1/minute_bars"),
    Path("/data1/stock-mxy/data/source_snapshots/amazingdata-cn-a-close-limit-min1-20220104-20260724-v1/minute_bars"),
)


def audit_board_state() -> dict[str, Any]:
    if not SLIM.is_file() or not CAND.is_file():
        return {"status": "MISSING_ARTIFACT", "board_ok": False}
    cand = pd.read_parquet(CAND, columns=["trade_date", "instrument_id", "is_st"])
    cand["trade_date"] = pd.to_datetime(cand["trade_date"])
    window = cand.loc[cand["trade_date"].between("2026-07-01", "2026-07-10")]
    panel_keys = set(zip(window["trade_date"], window["instrument_id"].astype(str)))
    inst = pd.read_parquet(HERMES / "instruments", columns=["instrument_id", "symbol", "exchange", "list_date"])
    inst["instrument_id"] = inst["instrument_id"].astype(str)
    st = pd.read_parquet(HERMES / "special_treatment", columns=["instrument_id", "name", "effective_from", "effective_to", "known_at"])
    st["instrument_id"] = st["instrument_id"].astype(str)
    for col in ("effective_from", "effective_to", "known_at"):
        st[col] = pd.to_datetime(st[col])
    st = st.loc[st["name"].map(is_st_name)]
    st_map: dict[str, list[tuple]] = {}
    for rec in st.itertuples(index=False):
        st_map.setdefault(str(rec.instrument_id), []).append(
            (
                pd.Timestamp(rec.effective_from) if pd.notna(rec.effective_from) else pd.Timestamp("1990-01-01"),
                pd.Timestamp(rec.effective_to) if pd.notna(rec.effective_to) else pd.Timestamp("2262-01-01"),
                pd.Timestamp(rec.known_at) if pd.notna(rec.known_at) else pd.Timestamp("1990-01-01"),
            )
        )
    meta = inst.drop_duplicates("instrument_id").set_index("instrument_id")
    panel = cand.loc[cand["trade_date"].between("2026-06-15", "2026-07-24"), ["trade_date", "instrument_id", "is_st"]].copy()
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    slim = pd.read_parquet(SLIM, columns=["trade_date", "instrument_id", "close", "pre_close", "paused", "volume"])
    slim["trade_date"] = pd.to_datetime(slim["trade_date"])
    slim = slim.loc[slim["trade_date"].between("2026-06-15", "2026-07-24")]
    slim["instrument_id"] = slim["instrument_id"].astype(str)
    joined = panel.merge(slim, on=["trade_date", "instrument_id"], how="left")
    checked = 0
    st_checked = 0
    mismatches = 0
    flat_only = 0
    examples = []
    for rec in joined.itertuples(index=False):
        info = meta.loc[rec.instrument_id] if rec.instrument_id in meta.index else None
        paused = bool(rec.paused) if pd.notna(rec.paused) else False
        if info is None or not np.isfinite(rec.pre_close) or rec.pre_close <= 0 or paused:
            continue
        day = pd.Timestamp(rec.trade_date)
        list_date = pd.Timestamp(info["list_date"]) if pd.notna(info["list_date"]) else pd.Timestamp("1990-01-01")
        if day < list_date + pd.Timedelta(days=40):
            continue
        board = board_of(str(info["symbol"]), str(info["exchange"]))
        is_st = False
        for start, end, known in st_map.get(rec.instrument_id, []):
            if known <= day and start <= day < end:
                is_st = True
                break
        amp = limit_pct(board, day, 30, is_st)
        if amp is None:
            continue
        legal = float(np.round(rec.pre_close * (1.0 + amp) + 1e-8, 2))
        flat = float(np.round(rec.pre_close * 1.099 + 1e-8, 2))
        sealed_legal = np.isfinite(rec.close) and rec.close + 1e-6 >= legal - 1e-4
        sealed_flat = np.isfinite(rec.close) and rec.close + 1e-6 >= flat - 1e-4
        checked += 1
        if is_st:
            st_checked += 1
        if not sealed_legal:
            mismatches += 1
            if len(examples) < 8:
                examples.append({"date": str(day.date()), "instrument_id": rec.instrument_id, "legal": legal, "flat": flat, "close": float(rec.close), "is_st": bool(is_st), "in_panel": True})
        if sealed_flat and not sealed_legal and abs(legal - flat) > 0.02:
            flat_only += 1
    board_ok = checked > 0 and mismatches == 0 and flat_only == 0
    return {
        "status": "OK",
        "checked_panel_rows": int(checked),
        "checked_main_st_rows": int(st_checked),
        "mismatches": int(mismatches),
        "flat_1099_only_inclusions": int(flat_only),
        "panel_keys_in_short_window": int(len(panel_keys)),
        "board_ok": board_ok,
        "st_cutover": str(ST_CUTOVER.date()),
        "used_flat_1099_for_all_names": False,
        "audit_rule": "只核对二板面板成员是否按法定涨跌幅封板。一板 ST 封板不要求进入二板面板。",
        "examples": examples,
        "rebuild_required": not board_ok,
    }


def build_features(*, resume: bool = True) -> dict[str, Any]:
    out = ROOT / "data" / "second_board_features.parquet"
    meta_path = ROOT / "data" / "feature_status.json"
    if resume and out.is_file() and meta_path.is_file():
        return __import__("core.r225241_io", fromlist=["load_json"]).load_json(meta_path)
    cand = pd.read_parquet(CAND)
    cand["trade_date"] = pd.to_datetime(cand["trade_date"])
    cand["instrument_id"] = cand["instrument_id"].astype(str)
    cand = cand.loc[cand["trade_date"].between("2018-01-02", "2026-07-24")].copy()
    print("[24.1] daily features", flush=True)
    daily = _daily_features(cand)
    print("[24.1] minute features", flush=True)
    minute = _minute_features(daily)
    print("[24.1] industry stub", flush=True)
    industry = _industry_feature(daily)
    frame = daily.merge(minute, on=["trade_date", "instrument_id"], how="left").merge(industry, on=["trade_date", "instrument_id"], how="left")
    coverage = {
        "r241_second_board_first_touch_time": float(frame["r241_second_board_first_touch_time"].notna().mean()),
        "r241_second_board_seal_duration": float(frame["r241_second_board_seal_duration"].notna().mean()),
        "r241_second_board_peer_strength": float(frame["r241_second_board_peer_strength"].notna().mean()),
    }
    blocked = []
    if coverage["r241_second_board_first_touch_time"] < FEATURE_COVERAGE_MIN or coverage["r241_second_board_seal_duration"] < FEATURE_COVERAGE_MIN:
        blocked.append("BLOCKED_MINUTE_DATA")
    if coverage["r241_second_board_peer_strength"] < FEATURE_COVERAGE_MIN:
        blocked.append("BLOCKED_PIT")
    feature_set = "REDUCED" if blocked else "FULL"
    columns = list(REDUCED_FEATURES if feature_set == "REDUCED" else FULL_FEATURES)
    keep = ["trade_date", "instrument_id", "label_2to3_net", "label_2to3_tradable", "label_2to3_seal", "next2_date", "is_st", "amount"] + columns
    # realized label attached later; keep base labels
    extra = [c for c in ("label_2to3_realized_exit_net_v1", "open_limit_up_entry") if c in frame.columns]
    stored = frame[keep + extra].copy()
    for col in columns:
        stored[col] = pd.to_numeric(stored[col], errors="coerce")
    if stored[columns].fillna(0).eq(0).all().all() and stored[columns].notna().any().any():
        pass
    stored_zero_fill = False
    write_parquet(out, stored)
    payload = {
        "status": "DONE",
        "feature_set": feature_set,
        "features": columns,
        "blocked": blocked,
        "coverage": coverage,
        "filled_with_zero": stored_zero_fill,
        "n_rows": int(len(stored)),
        "mixed_full_and_reduced": False,
    }
    dump_json(meta_path, payload)
    return payload


def _daily_features(cand: pd.DataFrame) -> pd.DataFrame:
    slim = pd.read_parquet(SLIM, columns=["trade_date", "instrument_id", "open", "high", "close", "pre_close", "volume", "amount", "paused", "limit_up"])
    slim["trade_date"] = pd.to_datetime(slim["trade_date"])
    slim["instrument_id"] = slim["instrument_id"].astype(str)
    wanted = set(cand["instrument_id"])
    hist = slim.loc[slim["instrument_id"].isin(wanted)].sort_values(["instrument_id", "trade_date"])
    valid = (~hist["paused"].fillna(False)) & hist["amount"].gt(0) & hist["close"].gt(0)
    hist = hist.loc[valid].copy()
    hist["amount_lag_median20"] = hist.groupby("instrument_id")["amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).median())
    hist["close_lag5"] = hist.groupby("instrument_id")["close"].shift(5)
    hist["amount_lag1"] = hist.groupby("instrument_id")["amount"].shift(1)
    hist["limit_px"] = pd.to_numeric(hist["limit_up"], errors="coerce").where(lambda s: s.between(0, 99_999))
    prev_limit = hist.groupby("instrument_id")["limit_px"].shift(1)
    prev_close = hist.groupby("instrument_id")["close"].shift(1)
    hist["prev_sealed"] = prev_close.add(1e-6).ge(prev_limit.sub(1e-4)) & prev_limit.notna()
    merged = cand.merge(
        hist[["trade_date", "instrument_id", "close", "amount_lag_median20", "close_lag5", "amount_lag1", "prev_sealed", "limit_px"]],
        on=["trade_date", "instrument_id"],
        how="left",
    )
    amount = pd.to_numeric(merged["amount"], errors="coerce")
    merged["r241_second_board_amount_log"] = np.where(amount > 0, np.log(amount), np.nan)
    merged["r241_second_board_turnover_surprise"] = amount / pd.to_numeric(merged["amount_lag_median20"], errors="coerce")
    merged["r241_second_board_return_5"] = merged["close"] / merged["close_lag5"] - 1.0
    lag_amount = pd.to_numeric(merged["amount_lag1"], errors="coerce").where(merged["prev_sealed"].fillna(False))
    merged["r241_first_to_second_amount_ratio"] = amount / lag_amount
    breadth = _breadth(slim)
    merged = merged.merge(breadth, on="trade_date", how="left")
    merged = _realized_label(merged, slim)
    return merged


def _seal_streak(codes: np.ndarray, eligible: np.ndarray, sealed: np.ndarray) -> np.ndarray:
    try:
        from numba import njit
    except ImportError:
        return _seal_streak_python(codes, eligible, sealed)

    @njit
    def _run(codes, eligible, sealed):
        n = len(codes)
        streak = np.zeros(n, dtype=np.int16)
        run = 0
        prev = np.int64(-1)
        for i in range(n):
            if codes[i] != prev:
                run = 0
                prev = codes[i]
            if not eligible[i]:
                streak[i] = run
                continue
            run = run + 1 if sealed[i] else 0
            streak[i] = run
        return streak

    return _run(codes, eligible, sealed)


def _seal_streak_python(codes: np.ndarray, eligible: np.ndarray, sealed: np.ndarray) -> np.ndarray:
    streak = np.zeros(len(codes), dtype=np.int16)
    run = 0
    prev = -1
    for i in range(len(codes)):
        if codes[i] != prev:
            run = 0
            prev = int(codes[i])
        if not eligible[i]:
            streak[i] = run
            continue
        run = run + 1 if sealed[i] else 0
        streak[i] = run
    return streak


def _breadth(slim: pd.DataFrame) -> pd.DataFrame:
    work = slim.sort_values(["instrument_id", "trade_date"])
    limit_up = pd.to_numeric(work["limit_up"], errors="coerce")
    real_limit = limit_up.where(limit_up.between(0, 99_999))
    eligible = (~work["paused"].fillna(False)) & work["volume"].gt(0) & work["pre_close"].gt(0) & real_limit.notna()
    sealed = eligible & work["close"].add(1e-6).ge(real_limit.sub(1e-4))
    codes, _ = pd.factorize(work["instrument_id"], sort=False)
    eligible_arr = eligible.to_numpy()
    sealed_arr = sealed.to_numpy()
    streak = _seal_streak(codes.astype(np.int64), eligible_arr.astype(np.bool_), sealed_arr.astype(np.bool_))
    ge2 = eligible_arr & (streak >= 2)
    dates = work["trade_date"].to_numpy()
    ge2 = eligible_arr & (streak >= 2)
    day = pd.DataFrame({"trade_date": dates, "eligible": eligible_arr, "ge2": ge2})
    agg = day.groupby("trade_date", sort=False).agg(eligible=("eligible", "sum"), ge2=("ge2", "sum"))
    agg["r241_second_board_market_breadth"] = np.where(agg["eligible"] > 0, agg["ge2"] / agg["eligible"], np.nan)
    return agg[["r241_second_board_market_breadth"]].reset_index()


def _minute_features(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[["trade_date", "instrument_id"]].drop_duplicates()
    by_day: dict[pd.Timestamp, set[str]] = {}
    for rec in keys.itertuples(index=False):
        by_day.setdefault(pd.Timestamp(rec.trade_date), set()).add(str(rec.instrument_id))
    rows = []
    limit_map = frame.drop_duplicates(["trade_date", "instrument_id"]).set_index(["trade_date", "instrument_id"])
    for day, names in by_day.items():
        path = None
        stamp = day.strftime("%Y-%m-%d")
        for root in MINUTE_ROOTS:
            candidate = root / f"trade_date={stamp}" / "part.parquet"
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            continue
        bars = pd.read_parquet(path, columns=["instrument_id", "bar_time", "high", "close", "volume"])
        bars["instrument_id"] = bars["instrument_id"].astype(str)
        bars = bars.loc[bars["instrument_id"].isin(names)]
        if bars.empty:
            continue
        bars["bar_time"] = pd.to_datetime(bars["bar_time"], utc=True).dt.tz_convert("Asia/Shanghai")
        for iid, block in bars.groupby("instrument_id", sort=False):
            try:
                value = limit_map.loc[(day, iid), "limit_px"]
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                limit_px = float(value)
            except (KeyError, TypeError, ValueError):
                limit_px = np.nan
            # 第二板当天的涨停价用当日 high 的法定触板，优先用面板没有的日线 limit。这里用 high 的最大值不能代替涨停价。
            block = block.sort_values("bar_time")
            if not np.isfinite(limit_px):
                rows.append({"trade_date": day, "instrument_id": iid, "r241_second_board_first_touch_time": np.nan, "r241_second_board_seal_duration": np.nan})
                continue
            touch = block.loc[block["high"] + 1e-6 >= limit_px - 1e-4]
            if touch.empty:
                first = np.nan
            else:
                clock = touch["bar_time"].iloc[0]
                first = float(clock.hour * 60 + clock.minute)
            sealed_min = (block["close"] + 1e-6 >= limit_px - 1e-4).to_numpy()
            observable = int((block["volume"].fillna(0) >= 0).sum())
            longest = _longest_run(sealed_min)
            duration = float(longest / observable) if observable else np.nan
            rows.append({"trade_date": day, "instrument_id": iid, "r241_second_board_first_touch_time": first, "r241_second_board_seal_duration": duration})
    if not rows:
        return pd.DataFrame(columns=["trade_date", "instrument_id", "r241_second_board_first_touch_time", "r241_second_board_seal_duration"])
    return pd.DataFrame(rows)


def _longest_run(flags: np.ndarray) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def _industry_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """行业涨停占比必须是 PIT 行业上的全市场封板占比。候选截面占比不能代替。缺这个口径就整列 NA。"""
    out = frame[["trade_date", "instrument_id"]].drop_duplicates().copy()
    out["r241_second_board_peer_strength"] = np.nan
    return out


def _realized_label(frame: pd.DataFrame, slim: pd.DataFrame) -> pd.DataFrame:
    """模型无关的卖出模拟。未成交不是亏损 0。净值不读取这个标签。"""
    hist = slim.loc[slim["instrument_id"].isin(set(frame["instrument_id"]))].sort_values(["instrument_id", "trade_date"])
    hist["limit_px"] = pd.to_numeric(hist["limit_up"], errors="coerce").where(lambda s: s.between(0, 99_999))
    grouped = {iid: block.reset_index(drop=True) for iid, block in hist.groupby("instrument_id", sort=False)}
    realized = []
    open_sealed = []
    for rec in frame.itertuples(index=False):
        block = grouped.get(rec.instrument_id)
        if block is None:
            realized.append(np.nan)
            open_sealed.append(False)
            continue
        idx = block.index[block["trade_date"].eq(rec.trade_date)]
        if len(idx) == 0:
            realized.append(np.nan)
            open_sealed.append(False)
            continue
        pos = int(idx[0])
        future = block.iloc[pos + 1 : pos + 12]
        if future.empty:
            realized.append(np.nan)
            open_sealed.append(False)
            continue
        buy = future.iloc[0]
        sealed = np.isfinite(buy["limit_px"]) and np.isfinite(buy["open"]) and float(buy["open"]) + 1e-6 >= float(buy["limit_px"]) - 0.005
        open_sealed.append(bool(sealed))
        if sealed or not np.isfinite(buy["open"]) or float(buy["open"]) <= 0:
            realized.append(np.nan)
            continue
        sell = None
        for _, day in future.iloc[1:].iterrows():
            paused = bool(day["paused"]) if pd.notna(day["paused"]) else False
            if paused or not np.isfinite(day["volume"]) or float(day["volume"]) <= 0 or not np.isfinite(day["close"]):
                continue
            if np.isfinite(day["limit_px"]) and float(day["close"]) + 1e-6 >= float(day["limit_px"]) - 0.005:
                continue
            sell = day
            break
        if sell is None:
            realized.append(np.nan)
            continue
        buy_px = float(buy["open"]) * (1.0 + 12.3 / 10_000.0)
        sell_px = float(sell["close"]) * (1.0 - 12.3 / 10_000.0)
        buy_fee = (1.0 + 0.1) / 10_000.0
        sell_fee = (1.0 + 0.1 + 5.0) / 10_000.0
        net = (sell_px * (1.0 - sell_fee)) / (buy_px * (1.0 + buy_fee)) - 1.0
        realized.append(float(net))
    frame = frame.copy()
    frame["label_2to3_realized_exit_net_v1"] = realized
    frame["open_limit_up_entry"] = open_sealed
    return frame
