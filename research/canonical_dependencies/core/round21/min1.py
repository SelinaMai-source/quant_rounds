"""A/C 分钟公式。复用 round18/f1.py 的分钟 join，公式必须新（A1 ≠ R18 F1）。缺失不填 0。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_ROOT
from ..round18.contract import TOUCH_TIME_APPROXIMATED
from ..round18.f1 import KEYS, _bar_seconds, _min_root
from .contract import TAIL_START_SEC
from .panel import load_hermes_table

AC_CACHE = ROUND21_ROOT / "data" / "ac_factors.parquet"
AC_META = ROUND21_ROOT / "data" / "ac_factors.json"

AC_FACTORS = (
    "r21_a1_vw_limit_gap",
    "r21_a2_minutes_to_reseal",
    "r21_a3_retouch_count",
    "r21_a4_tail_seal_share",
    "r21_c1_min1_amihud",
    "r21_c2_post_touch_amt_vs_impact",
    "r21_c3_post_touch_path_eff",
    "r21_c4_downside_quad_var",
)


def _empty_day(part: pd.DataFrame, *, touch_flag: str | None = "SKIP_DATA") -> pd.DataFrame:
    out = part[KEYS].copy()
    for col in AC_FACTORS:
        out[col] = np.nan
    out["touch_seconds"] = np.nan
    out["touch_flag"] = touch_flag
    return out


def _one_name(g: pd.DataFrame, limit_px: float, touch: float) -> dict[str, float]:
    rec = {k: np.nan for k in AC_FACTORS}
    close = pd.to_numeric(g["close"], errors="coerce")
    high = pd.to_numeric(g["high"], errors="coerce")
    low = pd.to_numeric(g["low"], errors="coerce")
    open_ = pd.to_numeric(g["open"], errors="coerce")
    amount = pd.to_numeric(g["amount"], errors="coerce")
    bar_sec = pd.to_numeric(g["bar_sec"], errors="coerce")

    tail = np.isfinite(bar_sec) & bar_sec.ge(TAIL_START_SEC) & np.isfinite(close) & np.isfinite(limit_px)
    if int(tail.sum()) > 0:
        rec["r21_a4_tail_seal_share"] = float(np.mean(close.loc[tail].to_numpy(dtype=float) >= limit_px))

    if not (np.isfinite(touch) and np.isfinite(limit_px)):
        return rec
    after = np.isfinite(bar_sec) & bar_sec.ge(touch)
    if int(after.sum()) == 0:
        return rec

    a_close = close.loc[after]
    a_high = high.loc[after]
    a_low = low.loc[after]
    a_open = open_.loc[after]
    a_amt = amount.loc[after]
    a_sec = bar_sec.loc[after]

    below = after & close.lt(limit_px) & amount.gt(0) & np.isfinite(close) & np.isfinite(amount)
    if int(below.sum()) > 0:
        gap = limit_px - close.loc[below].to_numpy(dtype=float)
        w = amount.loc[below].to_numpy(dtype=float)
        wsum = float(w.sum())
        if wsum > 0:
            rec["r21_a1_vw_limit_gap"] = float(np.dot(gap, w) / wsum)

    ordered = pd.DataFrame(
        {
            "sec": a_sec.to_numpy(dtype=float),
            "close": a_close.to_numpy(dtype=float),
            "high": a_high.to_numpy(dtype=float),
            "low": a_low.to_numpy(dtype=float),
            "open": a_open.to_numpy(dtype=float),
            "amount": a_amt.to_numpy(dtype=float),
        }
    ).sort_values("sec", kind="mergesort")
    oc = ordered["close"].to_numpy(dtype=float)
    osec = ordered["sec"].to_numpy(dtype=float)
    opened = np.isfinite(oc) & (oc < limit_px)
    if opened.any():
        first_leave = float(osec[np.flatnonzero(opened)[0]])
        reseal = np.isfinite(oc) & (oc >= limit_px) & (osec > first_leave)
        if reseal.any():
            rec["r21_a2_minutes_to_reseal"] = (float(osec[np.flatnonzero(reseal)[0]]) - float(touch)) / 60.0

    # 再触板：需要全日顺序以取 t-1 close。
    full = pd.DataFrame(
        {
            "sec": bar_sec.to_numpy(dtype=float),
            "close": close.to_numpy(dtype=float),
            "high": high.to_numpy(dtype=float),
        }
    ).sort_values("sec", kind="mergesort")
    prev_close = full["close"].shift(1)
    up = (
        full["high"].ge(limit_px)
        & prev_close.lt(limit_px)
        & full["sec"].gt(touch)
        & np.isfinite(full["high"])
        & np.isfinite(prev_close)
    )
    rec["r21_a3_retouch_count"] = float(int(up.sum()))

    amihud_ok = (
        np.isfinite(ordered["open"])
        & np.isfinite(ordered["close"])
        & np.isfinite(ordered["amount"])
        & ordered["open"].gt(0)
        & ordered["amount"].gt(0)
    )
    if int(amihud_ok.sum()) > 0:
        ret = np.abs(ordered.loc[amihud_ok, "close"] / ordered.loc[amihud_ok, "open"] - 1.0)
        rec["r21_c1_min1_amihud"] = float((ret / ordered.loc[amihud_ok, "amount"]).mean())

    amt_sum = float(ordered["amount"].sum(min_count=1)) if ordered["amount"].notna().any() else np.nan
    hi = ordered["high"].max(skipna=True)
    lo = ordered["low"].min(skipna=True)
    if np.isfinite(amt_sum) and np.isfinite(hi) and np.isfinite(lo) and float(hi - lo) > 0:
        rec["r21_c2_post_touch_amt_vs_impact"] = float(amt_sum / float(hi - lo))

    cvalid = ordered["close"].to_numpy(dtype=float)
    cvalid = cvalid[np.isfinite(cvalid)]
    if len(cvalid) >= 2:
        dlt = np.diff(cvalid)
        path = float(np.abs(dlt).sum())
        if path > 0:
            rec["r21_c3_post_touch_path_eff"] = float(abs(cvalid[-1] - cvalid[0]) / path)
        qv = float(np.square(dlt).sum())
        if qv > 0:
            rec["r21_c4_downside_quad_var"] = float(np.square(np.minimum(dlt, 0.0)).sum() / qv)
    return rec


def _ac_one_day(minutes: pd.DataFrame, day_keys: pd.DataFrame) -> pd.DataFrame:
    out = _empty_day(day_keys, touch_flag=None)
    out["touch_seconds"] = pd.to_numeric(day_keys.get("first_limit_time_seconds"), errors="coerce")
    if minutes.empty or day_keys.empty:
        return out
    m = minutes.copy()
    m["instrument_id"] = m["instrument_id"].astype(str)
    m["bar_sec"] = _bar_seconds(m["bar_time"])
    for col in ("open", "high", "low", "close", "amount"):
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")
    lim = day_keys[KEYS + ["limit_px", "first_limit_time_seconds"]].copy()
    lim["instrument_id"] = lim["instrument_id"].astype(str)
    lim["limit_px"] = pd.to_numeric(lim["limit_px"], errors="coerce")
    lim["first_limit_time_seconds"] = pd.to_numeric(lim["first_limit_time_seconds"], errors="coerce")
    need_proxy = lim["first_limit_time_seconds"].isna() | ~np.isfinite(lim["first_limit_time_seconds"])
    if need_proxy.any() and "high" in m.columns:
        hit = m.merge(lim.loc[need_proxy, KEYS[1:] + ["limit_px"]], on="instrument_id", how="inner")
        hit = hit.loc[hit["high"].ge(hit["limit_px"])]
        proxy = hit.groupby("instrument_id", sort=False)["bar_sec"].min()
        mapped = lim["instrument_id"].map(proxy)
        use = need_proxy & mapped.notna()
        lim.loc[use, "first_limit_time_seconds"] = mapped.loc[use].to_numpy()
        out.loc[use.to_numpy(), "touch_flag"] = TOUCH_TIME_APPROXIMATED
        out.loc[use.to_numpy(), "touch_seconds"] = mapped.loc[use].to_numpy()
    have = out["touch_seconds"].notna() & out["touch_flag"].isna()
    out.loc[have, "touch_flag"] = "first_limit_time_seconds"

    joined = m.merge(lim[["instrument_id", "limit_px", "first_limit_time_seconds"]], on="instrument_id", how="inner")
    if joined.empty:
        return out
    recs = []
    for iid, g in joined.groupby("instrument_id", sort=False):
        limit_px = float(pd.to_numeric(g["limit_px"].iloc[0], errors="coerce"))
        touch = float(pd.to_numeric(g["first_limit_time_seconds"].iloc[0], errors="coerce"))
        rec = _one_name(g, limit_px, touch)
        rec["instrument_id"] = str(iid)
        recs.append(rec)
    if not recs:
        return out
    got = pd.DataFrame(recs)
    out = out.drop(columns=list(AC_FACTORS)).merge(got, on="instrument_id", how="left")
    return out


def _process_day(root: str, part: pd.DataFrame) -> pd.DataFrame:
    files = list(Path(root).glob("*.parquet"))
    minutes = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    return _ac_one_day(minutes, part)


def _load_limit_px(keys: pd.DataFrame) -> pd.DataFrame:
    ev = load_hermes_table(
        "daily_limit_events",
        ["trade_date", "instrument_id", "limit_type", "limit_price", "first_limit_time_seconds", "close_limit_value"],
        instruments=keys["instrument_id"].unique(),
    )
    ev = ev.loc[ev["limit_type"].astype(str).eq("up")].copy()
    daily = load_hermes_table(
        "daily_raw",
        ["trade_date", "instrument_id", "limit_up", "amount", "close"],
        instruments=keys["instrument_id"].unique(),
    )
    work = keys[KEYS].merge(
        ev[KEYS + ["limit_price", "first_limit_time_seconds", "close_limit_value"]],
        on=KEYS,
        how="left",
    )
    work = work.merge(daily[KEYS + ["limit_up", "amount", "close"]], on=KEYS, how="left")
    work["limit_px"] = pd.to_numeric(work["limit_price"], errors="coerce")
    miss = work["limit_px"].isna()
    work.loc[miss, "limit_px"] = pd.to_numeric(work.loc[miss, "limit_up"], errors="coerce")
    return work


def build_ac_factors(keys: pd.DataFrame, *, jobs: int = 8, resume: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if resume and AC_CACHE.is_file():
        cached = pd.read_parquet(AC_CACHE)
        cached["trade_date"] = pd.to_datetime(cached["trade_date"])
        cached["instrument_id"] = cached["instrument_id"].astype(str)
        if set(AC_FACTORS).issubset(cached.columns) and len(cached) == len(keys):
            from .common import load_json

            meta = load_json(AC_META) if AC_META.is_file() else {"resumed": True}
            meta["resumed"] = True
            return cached, meta

    base = _load_limit_px(keys)
    tasks = []
    n_min_missing = 0
    parts: list[pd.DataFrame] = []
    for day, block in base.groupby("trade_date", sort=True):
        root = _min_root(pd.Timestamp(day))
        part = block[KEYS + ["limit_px", "first_limit_time_seconds"]].copy()
        if root is None:
            n_min_missing += int(len(part))
            parts.append(_empty_day(part, touch_flag="SKIP_DATA"))
            continue
        tasks.append((str(root), part))
    n_approx = 0
    if tasks:
        workers = min(max(1, int(jobs)), max(1, len(tasks)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_process_day, root, part) for root, part in tasks]
            done = 0
            for fut in as_completed(futs):
                got = fut.result()
                n_approx += int((got["touch_flag"] == TOUCH_TIME_APPROXIMATED).sum())
                parts.append(got)
                done += 1
                if done % 400 == 0:
                    print(f"[round21/min1] days {done}/{len(tasks)}", flush=True)
    ac = pd.concat(parts, ignore_index=True) if parts else _empty_day(base)
    out = keys[KEYS].merge(ac, on=KEYS, how="left")
    out = out.merge(
        base[KEYS + ["close_limit_value", "amount", "close", "limit_px"]],
        on=KEYS,
        how="left",
    )
    meta = {
        "n": int(len(out)),
        "n_minute_missing_rows": int(n_min_missing),
        "n_touch_approx": int(n_approx),
        "filled_with_zero": False,
        "a1_is_not_r18_f1": True,
        "reuse": "core/round18/f1.py:_min_root,_bar_seconds,minute join",
        "valid": {c: int(pd.to_numeric(out[c], errors="coerce").notna().sum()) for c in AC_FACTORS},
    }
    AC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(AC_CACHE, index=False)
    from .common import dump_json

    dump_json(AC_META, meta)
    return out, meta
