"""二进三特征：保留 2015+ 预热；缺行业不删分钟列；tail_seal 新 ID。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.research_integrity_v2.io_v2 import dump_json, write_df, write_parquet
from core.round24_1.features import (
    MINUTE_ROOTS,
    SLIM,
    CAND,
    _breadth,
    _daily_features,
    _longest_run,
    audit_board_state,
)
from core.round24_2.contract import CAND_START, CORE3, PERF_END, ROOT, TARGET7
from core.round24_2.simulate import longest_run, tail_seal_fraction


def build_universe_v2(*, resume: bool = True) -> dict[str, Any]:
    out = ROOT / "data" / "universe_v2.parquet"
    if resume and out.is_file():
        old = pd.read_parquet(CAND, columns=["trade_date", "instrument_id"])
        new = pd.read_parquet(out, columns=["trade_date", "instrument_id"])
        return {
            "status": "REUSED",
            "n_old": int(len(old)),
            "n_new": int(len(new)),
            "path": str(out),
        }
    audit = audit_board_state()
    dump_json(ROOT / "board_audit.json", audit)
    cand = pd.read_parquet(CAND)
    cand["trade_date"] = pd.to_datetime(cand["trade_date"])
    cand["instrument_id"] = cand["instrument_id"].astype(str)
    # 连续两日收盘合法涨停；停牌/缺失不得静默跨过 —— 沿用 round24 已建面板，并审计差异
    cand = cand.loc[cand["trade_date"].between(CAND_START, PERF_END)].copy()
    # ST 排除：与旧 T0 一致 —— is_st False
    if "is_st" in cand.columns:
        cand["eligible_st_ok"] = ~cand["is_st"].fillna(False).astype(bool)
    else:
        cand["eligible_st_ok"] = True
    write_parquet(out, cand)
    old = pd.read_parquet(CAND, columns=["trade_date", "instrument_id"])
    old["trade_date"] = pd.to_datetime(old["trade_date"])
    old["instrument_id"] = old["instrument_id"].astype(str)
    old = old.loc[old["trade_date"].between(CAND_START, PERF_END)]
    merged = old.merge(cand[["trade_date", "instrument_id"]], on=["trade_date", "instrument_id"], how="outer", indicator=True)
    diff = {
        "only_old": int((merged["_merge"] == "left_only").sum()),
        "only_new": int((merged["_merge"] == "right_only").sum()),
        "both": int((merged["_merge"] == "both").sum()),
    }
    write_df(ROOT / "universe_old_vs_new.csv", pd.DataFrame([diff]))
    return {"status": "DONE", "audit": audit, "diff": diff, "n": int(len(cand)), "path": str(out)}


def _minute_features_v2(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[["trade_date", "instrument_id"]].drop_duplicates()
    by_day: dict[pd.Timestamp, set[str]] = {}
    for rec in keys.itertuples(index=False):
        by_day.setdefault(pd.Timestamp(rec.trade_date), set()).add(str(rec.instrument_id))
    limit_map = frame.drop_duplicates(["trade_date", "instrument_id"]).set_index(["trade_date", "instrument_id"])
    rows = []
    expected = 240  # 交易分钟近似；午休不计入（分钟 bar 本身不含午休）
    for day, names in by_day.items():
        path = None
        stamp = day.strftime("%Y-%m-%d")
        for root in MINUTE_ROOTS:
            candidate = root / f"trade_date={stamp}" / "part.parquet"
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            for iid in names:
                rows.append(
                    {
                        "trade_date": day,
                        "instrument_id": iid,
                        "second_board_first_touch_time": np.nan,
                        "second_board_tail_seal_fraction": np.nan,
                        "second_board_longest_run_legacy": np.nan,
                        "minute_missing_reason": "NO_MINUTE_PARTITION",
                    }
                )
            continue
        bars = pd.read_parquet(path, columns=["instrument_id", "bar_time", "high", "close", "volume"])
        bars["instrument_id"] = bars["instrument_id"].astype(str)
        bars = bars.loc[bars["instrument_id"].isin(names)]
        bars["bar_time"] = pd.to_datetime(bars["bar_time"], utc=True).dt.tz_convert("Asia/Shanghai")
        for iid, block in bars.groupby("instrument_id", sort=False):
            try:
                value = limit_map.loc[(day, iid), "limit_px"]
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                limit_px = float(value)
            except Exception:
                limit_px = np.nan
            block = block.sort_values("bar_time")
            if not np.isfinite(limit_px):
                rows.append(
                    {
                        "trade_date": day,
                        "instrument_id": iid,
                        "second_board_first_touch_time": np.nan,
                        "second_board_tail_seal_fraction": np.nan,
                        "second_board_longest_run_legacy": np.nan,
                        "minute_missing_reason": "NO_LIMIT_PX",
                    }
                )
                continue
            # 交易分钟序号：按当日 bar 顺序（已排除午休空档）
            touch = block.loc[block["high"] + 1e-6 >= limit_px - 1e-4]
            if touch.empty:
                first_touch = np.nan
            else:
                ordered = block.reset_index(drop=True)
                first_touch = float(ordered.index[ordered["bar_time"].eq(touch["bar_time"].iloc[0])][0])
            sealed = (block["close"] + 1e-6 >= limit_px - 1e-4).to_numpy()
            # 缺 volume bar 打断：volume nan 视为打断
            sealed = sealed & block["volume"].notna().to_numpy()
            rows.append(
                {
                    "trade_date": day,
                    "instrument_id": iid,
                    "second_board_first_touch_time": first_touch,
                    "second_board_tail_seal_fraction": tail_seal_fraction(sealed, expected),
                    "second_board_longest_run_legacy": longest_run(sealed),
                    "minute_missing_reason": None,
                }
            )
    return pd.DataFrame(rows)


def build_features(*, resume: bool = True) -> dict[str, Any]:
    out = ROOT / "data" / "second_board_features.parquet"
    meta_path = ROOT / "data" / "feature_registry.csv"
    status_path = ROOT / "data" / "feature_status.json"
    if resume and out.is_file() and status_path.is_file():
        from core.research_integrity_v2.io_v2 import load_json

        return load_json(status_path)
    univ = pd.read_parquet(ROOT / "data" / "universe_v2.parquet")
    univ["trade_date"] = pd.to_datetime(univ["trade_date"])
    # 训练可用 2015+；绩效另切
    print("[24.2] daily features from 2015+", flush=True)
    daily = _daily_features(univ)
    # rename r241_* to TARGET7 names where applicable
    rename = {
        "r241_second_board_amount_log": "second_board_amount_log",
        "r241_second_board_turnover_surprise": "second_board_turnover_surprise",
        "r241_second_board_return_5": "second_board_return_5",
        "r241_first_to_second_amount_ratio": "first_to_second_amount_ratio",
        "r241_second_board_market_breadth": "second_board_market_breadth",
    }
    daily = daily.rename(columns=rename)
    amount = pd.to_numeric(daily.get("amount"), errors="coerce")
    daily["amount_log"] = np.where(amount > 0, np.log(amount), np.nan)
    daily["return_5"] = pd.to_numeric(daily.get("second_board_return_5"), errors="coerce")
    daily["amount_vs_5"] = pd.to_numeric(daily.get("second_board_turnover_surprise"), errors="coerce")
    print("[24.2] minute features F5/F6", flush=True)
    minute = _minute_features_v2(daily)
    frame = daily.merge(minute, on=["trade_date", "instrument_id"], how="left")
    # 行业不进入本轮，但不得因此删除分钟列
    cov = {
        "second_board_first_touch_time": float(frame["second_board_first_touch_time"].notna().mean()),
        "second_board_tail_seal_fraction": float(frame["second_board_tail_seal_fraction"].notna().mean()),
        "industry_not_in_round": True,
    }
    write_df(ROOT / "minute_feature_coverage.csv", pd.DataFrame([cov]))
    cols = list(CORE3) + list(TARGET7)
    keep = ["trade_date", "instrument_id", "is_st", "eligible_st_ok", "amount", "label_2to3_net", "label_2to3_tradable", "label_2to3_seal", "label_2to3_realized_exit_net_v1", "open_limit_up_entry", "next2_date"]
    keep = [c for c in keep if c in frame.columns] + [c for c in cols if c in frame.columns]
    # also keep longest_run legacy id
    if "second_board_longest_run_legacy" in frame.columns:
        keep.append("second_board_longest_run_legacy")
    stored = frame[keep].copy()
    write_parquet(out, stored)
    reg = [{"feature": c, "set": "CORE3" if c in CORE3 else "TARGET7", "present": c in stored.columns} for c in list(CORE3) + list(TARGET7)]
    write_df(meta_path, pd.DataFrame(reg))
    payload = {
        "status": "DONE",
        "n_rows": int(len(stored)),
        "min_date": str(pd.to_datetime(stored["trade_date"]).min().date()),
        "max_date": str(pd.to_datetime(stored["trade_date"]).max().date()),
        "coverage": cov,
        "industry_blocked_but_minutes_kept": True,
        "no_full_reduced_knife": True,
        "filled_with_zero": False,
    }
    dump_json(status_path, payload)
    return payload
