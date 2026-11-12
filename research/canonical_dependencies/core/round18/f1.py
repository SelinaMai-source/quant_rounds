"""F1 = log1p(V_seal / (A_below + 0.01*A_day)). 缺失不填 0。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ..paths import ROUND18_ROOT, assert_project_result_path
from .common import dump_json
from .contract import F1_EPS_A_DAY, F1_NAME, HERMES_V7, MIN1_2013, MIN1_2022, TOUCH_TIME_APPROXIMATED

KEYS = ["trade_date", "instrument_id"]
OPEN_SEC = 9 * 3600 + 30 * 60
F1_PARQUET = ROUND18_ROOT / "factors" / "f1.parquet"
F1_META = ROUND18_ROOT / "factors" / "f1_meta.json"


def _min_root(day: pd.Timestamp) -> Path | None:
    d = day.date()
    if d.year <= 2021:
        root = MIN1_2013 / "minute_bars" / f"trade_date={d.isoformat()}"
    else:
        root = MIN1_2022 / "minute_bars" / f"trade_date={d.isoformat()}"
    return root if root.is_dir() else None


def _bar_seconds(bar_time: pd.Series) -> pd.Series:
    ts = pd.to_datetime(bar_time, utc=True).dt.tz_convert("Asia/Shanghai")
    return (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).astype("float64")


def _load_events(keys: pd.DataFrame) -> pd.DataFrame:
    table = ds.dataset(str(HERMES_V7 / "daily_limit_events"), format="parquet")
    ev = table.to_table(
        columns=[
            "trade_date", "instrument_id", "limit_type", "limit_price",
            "close_limit_value", "first_limit_time_seconds",
        ]
    ).to_pandas()
    ev["trade_date"] = pd.to_datetime(ev["trade_date"])
    ev["instrument_id"] = ev["instrument_id"].astype(str)
    ev = ev.loc[ev["limit_type"].astype(str).eq("up")].copy()
    work = keys.merge(ev, on=KEYS, how="left")
    return work


def _load_daily(keys: pd.DataFrame) -> pd.DataFrame:
    table = ds.dataset(str(HERMES_V7 / "daily_raw"), format="parquet")
    raw = table.to_table(columns=["trade_date", "instrument_id", "amount", "close", "limit_up"]).to_pandas()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["instrument_id"] = raw["instrument_id"].astype(str)
    return keys.merge(raw, on=KEYS, how="left")


def _a_below_one_day(minutes: pd.DataFrame, day_keys: pd.DataFrame) -> pd.DataFrame:
    out = day_keys[KEYS].copy()
    out["A_below"] = np.nan
    out["touch_seconds"] = pd.to_numeric(day_keys.get("first_limit_time_seconds"), errors="coerce")
    out["touch_flag"] = np.where(out["touch_seconds"].notna() & np.isfinite(out["touch_seconds"]), "first_limit_time_seconds", None)
    if minutes.empty or day_keys.empty:
        return out
    m = minutes.copy()
    m["instrument_id"] = m["instrument_id"].astype(str)
    m["bar_sec"] = _bar_seconds(m["bar_time"])
    m["close"] = pd.to_numeric(m["close"], errors="coerce")
    m["high"] = pd.to_numeric(m["high"], errors="coerce")
    m["amount"] = pd.to_numeric(m["amount"], errors="coerce")
    lim = day_keys[KEYS + ["limit_px", "first_limit_time_seconds"]].copy()
    lim["instrument_id"] = lim["instrument_id"].astype(str)
    lim["limit_px"] = pd.to_numeric(lim["limit_px"], errors="coerce")
    lim["first_limit_time_seconds"] = pd.to_numeric(lim["first_limit_time_seconds"], errors="coerce")
    need_proxy = lim["first_limit_time_seconds"].isna() | ~np.isfinite(lim["first_limit_time_seconds"])
    if need_proxy.any():
        hit = m.merge(lim.loc[need_proxy, KEYS + ["limit_px"]], on="instrument_id", how="inner")
        hit = hit.loc[hit["high"].ge(hit["limit_px"])]
        proxy = hit.groupby("instrument_id", sort=False)["bar_sec"].min()
        mapped = lim["instrument_id"].map(proxy)
        use = need_proxy & mapped.notna()
        lim.loc[use, "first_limit_time_seconds"] = mapped.loc[use].to_numpy()
        out.loc[use.to_numpy(), "touch_flag"] = TOUCH_TIME_APPROXIMATED
        out.loc[use.to_numpy(), "touch_seconds"] = mapped.loc[use].to_numpy()
    joined = m.merge(lim, on="instrument_id", how="inner")
    mask = (
        joined["first_limit_time_seconds"].notna()
        & joined["limit_px"].notna()
        & joined["bar_sec"].ge(joined["first_limit_time_seconds"])
        & joined["close"].lt(joined["limit_px"])
    )
    below = joined.loc[mask].groupby("instrument_id", sort=False)["amount"].sum()
    out["A_below"] = out["instrument_id"].map(below)
    has_touch = out["touch_seconds"].notna() & np.isfinite(out["touch_seconds"])
    out.loc[has_touch & out["A_below"].isna(), "A_below"] = 0.0
    return out


def _process_day(root: str, part: pd.DataFrame) -> pd.DataFrame:
    files = list(Path(root).glob("*.parquet"))
    minutes = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    return _a_below_one_day(minutes, part)


def build_f1(keys: pd.DataFrame, *, resume: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    if resume and F1_PARQUET.is_file():
        frame = pd.read_parquet(F1_PARQUET)
        meta = __import__("json").loads(F1_META.read_text(encoding="utf-8")) if F1_META.is_file() else {"resumed": True}
        return frame, meta
    work = keys[KEYS].drop_duplicates().copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    ev = _load_events(work)
    daily = _load_daily(work)
    base = work.merge(
        ev[KEYS + ["close_limit_value", "first_limit_time_seconds", "limit_price"]],
        on=KEYS, how="left",
    ).merge(daily[KEYS + ["amount", "limit_up"]], on=KEYS, how="left")
    base["V_seal"] = pd.to_numeric(base["close_limit_value"], errors="coerce")
    base["A_day"] = pd.to_numeric(base["amount"], errors="coerce")
    base["limit_px"] = pd.to_numeric(base["limit_price"], errors="coerce")
    miss_lim = base["limit_px"].isna()
    base.loc[miss_lim, "limit_px"] = pd.to_numeric(base.loc[miss_lim, "limit_up"], errors="coerce")
    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = []
    n_min_missing = 0
    n_approx = 0
    parts = []
    missing_parts = []
    for day, block in base.groupby("trade_date", sort=True):
        root = _min_root(pd.Timestamp(day))
        part = block[KEYS + ["limit_px", "first_limit_time_seconds"]].copy()
        if root is None:
            n_min_missing += int(len(part))
            miss = part[KEYS].copy()
            miss["A_below"] = np.nan
            miss["touch_seconds"] = np.nan
            miss["touch_flag"] = "SKIP_DATA"
            missing_parts.append(miss)
            continue
        tasks.append((str(root), part))
    parts.extend(missing_parts)
    if tasks:
        workers = min(16, max(1, len(tasks)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_process_day, root, part) for root, part in tasks]
            for fut in as_completed(futs):
                got = fut.result()
                n_approx += int((got["touch_flag"] == TOUCH_TIME_APPROXIMATED).sum())
                parts.append(got)
    below = pd.concat(parts, ignore_index=True) if parts else work.assign(A_below=np.nan, touch_seconds=np.nan, touch_flag=None)
    out = base.merge(below, on=KEYS, how="left")
    v = pd.to_numeric(out["V_seal"], errors="coerce")
    a_below = pd.to_numeric(out["A_below"], errors="coerce")
    a_day = pd.to_numeric(out["A_day"], errors="coerce")
    denom = a_below + F1_EPS_A_DAY * a_day
    valid = v.notna() & denom.notna() & denom.gt(0) & np.isfinite(v) & np.isfinite(denom)
    out[F1_NAME] = np.where(valid, np.log1p(v / denom), np.nan)
    keep = KEYS + [F1_NAME, "V_seal", "A_below", "A_day", "touch_seconds", "touch_flag"]
    frame = out[keep].copy()
    meta = {
        "name": F1_NAME,
        "formula": "log1p(V_seal / (A_below + 0.01*A_day))",
        "n": int(len(frame)),
        "n_valid": int(frame[F1_NAME].notna().sum()),
        "n_touch_approx": n_approx,
        "n_minute_missing": n_min_missing,
        "filled_with_zero": False,
        "v_seal_source": "daily_limit_events.close_limit_value",
        "a_day_source": "daily_raw.amount",
    }
    assert_project_result_path(F1_PARQUET)
    F1_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(F1_PARQUET, index=False)
    dump_json(F1_META, meta)
    return frame, meta
