"""构造 A/B/D/E。窗口只用 D 及以前已可得的信息。缺失保持 NA，不填 0。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ..paths import ROUND22_ROOT
from ..round18.f1 import _bar_seconds, _min_root
from ..round21.panel import load_candidate_keys
from .common import dump_json, load_json, write_md
from .contract import BLOCKED_DATA, HERMES_V7, KEYS, MD, NEVER_SUSPENDED

DATA = ROUND22_ROOT / "data"
NAME_PATH = DATA / "name_daily.parquet"
MKT_PATH = DATA / "market_daily.parquet"
TOUCH_PATH = DATA / "minute_high_touch.parquet"
PEER_PATH = DATA / "peer_factors.parquet"
PANEL_PATH = DATA / "factor_panel_abde.parquet"
META_PATH = ROUND22_ROOT / "construct" / "abde.json"
OPEN_SEC = 9 * 3600 + 30 * 60
SCAN_COLS = (
    "trade_date",
    "instrument_id",
    "high",
    "close",
    "pre_close",
    "volume",
    "amount",
    "limit_up",
    "paused",
)


def _touch_day(payload: tuple) -> list[tuple]:
    day, root, pairs = payload
    if not root or not pairs:
        return []
    files = list(Path(root).glob("*.parquet"))
    if not files:
        return []
    want = {iid for iid, _ in pairs}
    limits = {iid: lim for iid, lim in pairs if np.isfinite(lim)}
    frames = []
    for path in files:
        names = set(pq.read_schema(path).names)
        if "instrument_id" not in names or "high" not in names or "bar_time" not in names:
            continue
        part = pd.read_parquet(path, columns=["instrument_id", "high", "bar_time"])
        part["instrument_id"] = part["instrument_id"].astype(str)
        part = part.loc[part["instrument_id"].isin(want)]
        if not part.empty:
            frames.append(part)
    if not frames:
        return []
    minutes = pd.concat(frames, ignore_index=True)
    minutes["bar_sec"] = _bar_seconds(minutes["bar_time"])
    minutes["high"] = pd.to_numeric(minutes["high"], errors="coerce")
    out = []
    for iid, block in minutes.groupby("instrument_id", sort=False):
        lim = limits.get(str(iid))
        if lim is None or not np.isfinite(lim) or lim <= 0:
            continue
        hit = block.loc[block["high"] >= lim * (1 - 1e-4), "bar_sec"]
        hit = hit[np.isfinite(hit)]
        if hit.empty:
            continue
        out.append((day, str(iid), float(hit.min())))
    return out


def _scan(ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scan_meta_path = DATA / "scan_meta.json"
    if NAME_PATH.is_file() and MKT_PATH.is_file():
        saved = load_json(scan_meta_path)
        return pd.read_parquet(NAME_PATH), pd.read_parquet(MKT_PATH), {**saved, "resumed": True, "paused_column": bool(saved.get("paused_column", True))}
    root = HERMES_V7 / "daily_raw"
    if not root.exists():
        raise RuntimeError(f"{BLOCKED_DATA} missing daily_raw")
    dataset = ds.dataset(str(root), format="parquet")
    available = set(dataset.schema.names)
    cols = [c for c in SCAN_COLS if c in available]
    if "paused" not in cols:
        paused_ok = False
    else:
        paused_ok = True
    mkt_parts = []
    name_parts = []
    n_batches = 0
    for batch in dataset.to_batches(columns=cols, batch_size=300_000):
        n_batches += 1
        frame = batch.to_pandas()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        pre = pd.to_numeric(frame["pre_close"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        lim = pd.to_numeric(frame["limit_up"], errors="coerce") if "limit_up" in frame.columns else pd.Series(np.nan, index=frame.index)
        vol = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else pd.Series(np.nan, index=frame.index)
        amt = pd.to_numeric(frame["amount"], errors="coerce") if "amount" in frame.columns else pd.Series(np.nan, index=frame.index)
        if paused_ok:
            paused = frame["paused"].fillna(False).astype(bool)
        else:
            paused = pd.Series(False, index=frame.index)
        tradable = pre.gt(0) & close.gt(0) & ~paused
        if "volume" in frame.columns:
            tradable = tradable & vol.fillna(0).gt(0)
        ret = close / pre - 1.0
        amp = lim / pre - 1.0
        close_lim = tradable & lim.gt(0) & close.ge(lim * (1.0 - 1e-4))
        touch = tradable & lim.gt(0) & high.ge(lim * (1.0 - 1e-4))
        ret_ok = ret.where(tradable)
        tmp = pd.DataFrame(
            {
                "trade_date": frame["trade_date"],
                "n": tradable.astype(np.float64),
                "n_close": close_lim.astype(np.float64),
                "n_touch": touch.astype(np.float64),
                "ret_sum": ret_ok,
                "ret_n": ret_ok.notna().astype(np.float64),
            }
        )
        mkt_parts.append(tmp.groupby("trade_date", sort=False).sum(numeric_only=True))
        keep = frame["instrument_id"].isin(ids)
        if bool(keep.any()):
            name_parts.append(
                pd.DataFrame(
                    {
                        "trade_date": frame.loc[keep, "trade_date"].to_numpy(),
                        "instrument_id": frame.loc[keep, "instrument_id"].to_numpy(),
                        "ret": ret.loc[keep].to_numpy(dtype=np.float32),
                        "amount": amt.loc[keep].to_numpy(dtype=np.float32),
                        "close_lim": close_lim.loc[keep].to_numpy(dtype=np.float32),
                        "touch": touch.loc[keep].to_numpy(dtype=np.float32),
                        "paused": paused.loc[keep].to_numpy(dtype=bool),
                        "amp": amp.loc[keep].to_numpy(dtype=np.float32),
                        "limit_up": lim.loc[keep].to_numpy(dtype=np.float32),
                    }
                )
            )
        if n_batches % 20 == 0:
            print(f"[r22 scan] batches={n_batches}", flush=True)
    if not mkt_parts or not name_parts:
        raise RuntimeError(f"{BLOCKED_DATA} daily scan empty")
    mkt = pd.concat(mkt_parts).groupby(level=0).sum()
    mkt.index = pd.to_datetime(mkt.index)
    mkt = mkt.sort_index()
    mkt["close_br"] = mkt["n_close"] / mkt["n"].replace(0, np.nan)
    mkt["touch_br"] = mkt["n_touch"] / mkt["n"].replace(0, np.nan)
    mkt["mkt_ret"] = mkt["ret_sum"] / mkt["ret_n"].replace(0, np.nan)
    name = pd.concat(name_parts, ignore_index=True)
    name["trade_date"] = pd.to_datetime(name["trade_date"])
    name = name.drop_duplicates(KEYS).sort_values(KEYS)
    DATA.mkdir(parents=True, exist_ok=True)
    name.to_parquet(NAME_PATH, index=False)
    mkt.reset_index(names="trade_date").to_parquet(MKT_PATH, index=False)
    meta = {"resumed": False, "paused_column": paused_ok, "n_batches": n_batches, "n_name_rows": int(len(name))}
    dump_json(DATA / "scan_meta.json", meta)
    return name, mkt.reset_index(names="trade_date"), meta


def _halt_long(name: pd.DataFrame) -> pd.DataFrame:
    """在个股已观测交易日上计算，再 shift(1)。缺失日不当成未停牌。"""
    frames = []
    for iid, block in name.groupby("instrument_id", sort=False):
        g = block.sort_values("trade_date")
        col = g["paused"].fillna(False).to_numpy(dtype=bool)
        n = len(col)
        since_j = np.full(n, np.nan)
        dur_j = np.full(n, np.nan)
        cnt_j = np.full(n, np.nan)
        rec_j = np.full(n, np.nan)
        csum = np.cumsum(col)
        ever = False
        last = -1
        spell = 0
        for i, flag in enumerate(col):
            if flag:
                spell = spell + 1 if ever and i == last + 1 else 1
                last = i
                ever = True
                since_j[i] = 0.0
                dur_j[i] = float(spell)
            elif ever:
                since_j[i] = float(i - last)
                dur_j[i] = float(spell)
            if ever:
                j0 = 0 if i < 59 else i - 59
                prev = 0.0 if j0 == 0 else float(csum[j0 - 1])
                cnt_j[i] = float(csum[i]) - prev
                rec_j[i] = 1.0 if since_j[i] <= 5 else 0.0
        since = np.full(n, np.nan)
        dur = np.full(n, np.nan)
        cnt = np.full(n, np.nan)
        rec = np.full(n, np.nan)
        if n > 1:
            since[1:] = since_j[:-1]
            dur[1:] = dur_j[:-1]
            cnt[1:] = cnt_j[:-1]
            rec[1:] = rec_j[:-1]
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": g["trade_date"].to_numpy(),
                    "instrument_id": iid,
                    "r22_e1_days_since_halt": since,
                    "r22_e2_halt_count_60": cnt,
                    "r22_e3_resumed_within_5d": rec,
                    "r22_e4_last_halt_duration": dur,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=KEYS + ["r22_e1_days_since_halt", "r22_e2_halt_count_60", "r22_e3_resumed_within_5d", "r22_e4_last_halt_duration"])
    return pd.concat(frames, ignore_index=True)


def _lookup(wide: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    stacked = wide.stack()
    stacked.index = stacked.index.set_names(["trade_date", "instrument_id"])
    idx = pd.MultiIndex.from_frame(keys[KEYS])
    return stacked.reindex(idx).to_numpy(dtype=np.float64)


def _family_a(name: pd.DataFrame, mkt: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    mkt = mkt.copy()
    mkt["trade_date"] = pd.to_datetime(mkt["trade_date"])
    mkt = mkt.drop_duplicates("trade_date").set_index("trade_date").sort_index()
    wide_ret = name.pivot_table(index="trade_date", columns="instrument_id", values="ret", aggfunc="last").sort_index()
    wide_c = name.pivot_table(index="trade_date", columns="instrument_id", values="close_lim", aggfunc="last").reindex(wide_ret.index)
    wide_t = name.pivot_table(index="trade_date", columns="instrument_id", values="touch", aggfunc="last").reindex(wide_ret.index)
    br_c = mkt["close_br"].reindex(wide_ret.index)
    br20 = br_c.shift(1).rolling(20, min_periods=10).mean()
    own_c = wide_c.shift(1).rolling(20, min_periods=10).mean()
    own_t = wide_t.shift(1).rolling(20, min_periods=10).mean()
    own_t5 = wide_t.shift(1).rolling(5, min_periods=3).mean()
    r1 = wide_ret.shift(1)
    m1 = mkt["mkt_ret"].reindex(wide_ret.index).shift(1)
    try:
        beta = r1.rolling(60, min_periods=40).corr(m1)
    except Exception:
        beta = pd.DataFrame({c: r1[c].rolling(60, min_periods=40).corr(m1) for c in r1.columns}, index=r1.index)
    std_r = r1.rolling(60, min_periods=40).std()
    std_m = m1.rolling(60, min_periods=40).std()
    beta = beta.mul(std_r, axis=0).div(std_m.replace(0, np.nan), axis=0)
    ret20 = r1.rolling(20, min_periods=10).mean()
    m20 = m1.rolling(20, min_periods=10).mean()
    a3 = ret20.sub(beta.mul(m20, axis=0), axis=0)
    out = keys[KEYS].copy()
    out["r22_a1_own_close_limit_excess_20"] = _lookup(own_c.sub(br20, axis=0), keys)
    out["r22_a2_own_touch_limit_excess_20"] = _lookup(own_t.sub(mkt["touch_br"].reindex(wide_ret.index).shift(1).rolling(20, min_periods=10).mean(), axis=0), keys)
    out["r22_a3_ret20_mkt_beta_resid"] = _lookup(a3, keys)
    out["r22_a4_touch5_over_mkt_breadth"] = _lookup(own_t5.div(br20 + 1e-6, axis=0), keys)
    out["r22_aux_mkt_beta"] = _lookup(beta, keys)
    return out


def _peers(name: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    if PEER_PATH.is_file():
        return pd.read_parquet(PEER_PATH)
    wide_ret = name.pivot_table(index="trade_date", columns="instrument_id", values="ret", aggfunc="last").sort_index()
    wide_amt = name.pivot_table(index="trade_date", columns="instrument_id", values="amount", aggfunc="last").reindex(wide_ret.index)
    wide_amp = name.pivot_table(index="trade_date", columns="instrument_id", values="amp", aggfunc="last").reindex(wide_ret.index)
    dates = list(wide_ret.index)
    pos = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    cols = list(wide_ret.columns)
    col_i = {c: i for i, c in enumerate(cols)}
    arr = wide_ret.to_numpy(dtype=np.float64)
    amt = wide_amt.to_numpy(dtype=np.float64)
    amp = wide_amp.to_numpy(dtype=np.float64)
    innov = pd.DataFrame(amt, index=wide_ret.index, columns=cols)
    innov = innov.shift(1) / innov.shift(2).rolling(20, min_periods=10).mean() - 1.0
    innov_arr = innov.to_numpy(dtype=np.float64)
    ret_d1 = pd.DataFrame(arr, index=wide_ret.index, columns=cols).shift(1).to_numpy(dtype=np.float64)
    amp_d1 = pd.DataFrame(amp, index=wide_ret.index, columns=cols).shift(1).to_numpy(dtype=np.float64)
    grouped: dict[pd.Timestamp, list[int]] = {}
    for day, iid in keys[KEYS].itertuples(index=False):
        j = col_i.get(str(iid))
        if j is None:
            continue
        grouped.setdefault(pd.Timestamp(day), []).append(j)
    rows = []
    n_done = 0
    for day, members in grouped.items():
        i = pos.get(pd.Timestamp(day))
        if i is None or i < 60:
            continue
        window = arr[i - 60 : i]
        finite = np.isfinite(window)
        members = list(dict.fromkeys(members))
        for j in members:
            mj = finite[:, j]
            if int(mj.sum()) < 40:
                rows.append((day, cols[j], np.nan, np.nan, np.nan, np.nan))
                continue
            x = window[mj, j]
            sub = window[mj]
            mask = np.isfinite(sub)
            n_ov = mask.sum(axis=0)
            sub0 = np.where(mask, sub, 0.0)
            sx = (mask * x[:, None]).sum(axis=0)
            sy = sub0.sum(axis=0)
            sxx = ((x[:, None] ** 2) * mask).sum(axis=0)
            syy = (sub0 ** 2).sum(axis=0)
            sxy = (x[:, None] * sub0).sum(axis=0)
            num = n_ov * sxy - sx * sy
            den = np.sqrt(np.maximum(n_ov * sxx - sx**2, 0) * np.maximum(n_ov * syy - sy**2, 0))
            corr = np.divide(num, den, out=np.full(len(cols), np.nan), where=(n_ov >= 40) & (den > 0))
            corr[j] = np.nan
            pos_idx = np.flatnonzero(np.isfinite(corr) & (corr > 0))
            if len(pos_idx) == 0:
                rows.append((day, cols[j], np.nan, np.nan, np.nan, np.nan))
                continue
            if len(pos_idx) > 5:
                pick = pos_idx[np.argpartition(-corr[pos_idx], 5)[:5]]
            else:
                pick = pos_idx
            r = ret_d1[i, pick]
            r = r[np.isfinite(r)]
            b1 = float(np.mean(r)) if len(r) else np.nan
            b2 = float(np.std(r, ddof=1)) if len(r) >= 2 else np.nan
            inn = innov_arr[i, pick]
            inn = inn[np.isfinite(inn)]
            b3 = float(np.mean(inn)) if len(inn) else np.nan
            rr = ret_d1[i, pick] / amp_d1[i, pick]
            good = np.isfinite(rr) & np.isfinite(amp_d1[i, pick]) & (amp_d1[i, pick] > 0.01)
            b4 = float(np.mean(rr[good])) if int(good.sum()) else np.nan
            rows.append((day, cols[j], b1, b2, b3, b4))
        n_done += 1
        if n_done % 100 == 0:
            print(f"[r22 peer] dates={n_done}", flush=True)
    out = pd.DataFrame(rows, columns=["trade_date", "instrument_id", "r22_b1_dyn_peer_ret_d1", "r22_b2_dyn_peer_ret_disp", "r22_b3_dyn_peer_to_innov", "r22_b4_dyn_peer_ret_over_amp"])
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out.to_parquet(PEER_PATH, index=False)
    return out


def _touches(keys: pd.DataFrame, name: pd.DataFrame, jobs: int) -> pd.DataFrame:
    if TOUCH_PATH.is_file():
        return pd.read_parquet(TOUCH_PATH)
    lim = name[KEYS + ["limit_up"]].copy()
    work = keys[KEYS].merge(lim, on=KEYS, how="left")
    tasks = []
    for day, block in work.groupby("trade_date", sort=True):
        root = _min_root(pd.Timestamp(day))
        pairs = [(str(iid), float(v) if pd.notna(v) else np.nan) for iid, v in zip(block["instrument_id"], block["limit_up"])]
        tasks.append((str(pd.Timestamp(day).date()), str(root) if root else "", pairs))
    found: list[tuple] = []
    missing = 0
    runnable = [t for t in tasks if t[1]]
    missing = len(tasks) - len(runnable)
    workers = max(1, min(int(jobs), 8, len(runnable) or 1))
    if runnable and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_touch_day, task) for task in runnable]
            for fut in as_completed(futs):
                found.extend(fut.result())
    else:
        for task in runnable:
            found.extend(_touch_day(task))
    touch = pd.DataFrame(found, columns=["day", "instrument_id", "touch_sec"])
    if touch.empty:
        touch = pd.DataFrame(columns=["trade_date", "instrument_id", "touch_sec"])
    else:
        touch["trade_date"] = pd.to_datetime(touch["day"])
        touch = touch.drop(columns=["day"]).drop_duplicates(KEYS)
    touch.to_parquet(TOUCH_PATH, index=False)
    dump_json(DATA / "minute_high_touch.json", {"n_tasks": len(tasks), "n_missing_root": missing, "n_hits": int(len(touch)), "definition": "min bar_sec where minute high >= limit_up"})
    return touch


def _family_d(keys: pd.DataFrame, name: pd.DataFrame, touch: pd.DataFrame) -> pd.DataFrame:
    work = keys[KEYS].merge(touch, on=KEYS, how="left")
    work = work.merge(name[KEYS + ["amount", "amp"]], on=KEYS, how="left")
    seal_path_candidates = []
    from ..round7.contract import FIVE_HEAD

    panel = FIVE_HEAD.panel_dir / "candidate_panel.parquet"
    if panel.is_file() and "fb_close_seal_streak_norm" in set(pq.read_schema(panel).names):
        seal = pd.read_parquet(panel, columns=["trade_date", "instrument_id", "fb_close_seal_streak_norm"])
        seal["trade_date"] = pd.to_datetime(seal["trade_date"])
        seal["instrument_id"] = seal["instrument_id"].astype(str)
        work = work.merge(seal, on=KEYS, how="left")
    else:
        work["fb_close_seal_streak_norm"] = np.nan
    sec = pd.to_numeric(work["touch_sec"], errors="coerce")
    lead = (sec - OPEN_SEC) / 60.0
    lead = lead.where(sec.notna())
    work["r22_d2_minute_high_lead_min"] = lead
    cnt = work.groupby("trade_date")["r22_d2_minute_high_lead_min"].transform(lambda s: s.notna().sum())
    work["r22_d1_minute_high_touch_rank"] = np.where(
        cnt.ge(2),
        work.groupby("trade_date")["r22_d2_minute_high_lead_min"].rank(method="average", pct=True),
        np.nan,
    )
    # D3：过去 120 个候选日的截面回归，系数不用当天。
    work = work.sort_values(KEYS)
    d3 = np.full(len(work), np.nan)
    days = list(work["trade_date"].drop_duplicates().sort_values())
    day_pos = {d: i for i, d in enumerate(days)}
    y = work["r22_d2_minute_high_lead_min"].to_numpy(dtype=float)
    x1 = np.log1p(pd.to_numeric(work["amount"], errors="coerce").to_numpy(dtype=float))
    x2 = pd.to_numeric(work["amp"], errors="coerce").to_numpy(dtype=float)
    groups = {d: np.flatnonzero(work["trade_date"].eq(d).to_numpy()) for d in days}
    for d, idx in groups.items():
        p = day_pos[d]
        if p < 20:
            continue
        hist_days = days[max(0, p - 120) : p]
        hist_idx = np.concatenate([groups[h] for h in hist_days]) if hist_days else np.array([], dtype=int)
        yy = y[hist_idx]
        ok = np.isfinite(yy) & np.isfinite(x1[hist_idx]) & np.isfinite(x2[hist_idx])
        if int(ok.sum()) < 30:
            continue
        design = np.column_stack([np.ones(int(ok.sum())), x1[hist_idx][ok], x2[hist_idx][ok]])
        try:
            coef, *_ = np.linalg.lstsq(design, yy[ok], rcond=None)
        except np.linalg.LinAlgError:
            continue
        cur_ok = np.isfinite(y[idx]) & np.isfinite(x1[idx]) & np.isfinite(x2[idx])
        if not cur_ok.any():
            continue
        pred = coef[0] + coef[1] * x1[idx][cur_ok] + coef[2] * x2[idx][cur_ok]
        d3[idx[cur_ok]] = y[idx][cur_ok] - pred
    work["r22_d3_lead_resid_past_only"] = d3
    seal_v = pd.to_numeric(work["fb_close_seal_streak_norm"], errors="coerce")
    work["r22_d4_lead_x_seal_streak"] = work["r22_d2_minute_high_lead_min"] * seal_v
    return work[KEYS + ["r22_d1_minute_high_touch_rank", "r22_d2_minute_high_lead_min", "r22_d3_lead_resid_past_only", "r22_d4_lead_x_seal_streak"]]


def run_construct_abde(*, resume: bool = True, jobs: int = 4, **_: Any) -> dict[str, Any]:
    if resume and PANEL_PATH.is_file() and META_PATH.is_file():
        existing = load_json(META_PATH)
        if existing.get("status") in {"DONE", "PARTIAL"}:
            return existing
    keys = load_candidate_keys()
    keys = keys[KEYS].drop_duplicates().copy()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    keys["instrument_id"] = keys["instrument_id"].astype(str)
    ids = set(keys["instrument_id"])
    name, mkt, scan_meta = _scan(ids)
    if "trade_date" not in mkt.columns:
        mkt = mkt.reset_index()
    print("[r22] family A", flush=True)
    fam_a = _family_a(name, mkt, keys)
    print("[r22] family E", flush=True)
    if scan_meta.get("paused_column") is False:
        e_block = BLOCKED_DATA
        fam_e = keys[KEYS].copy()
        for col in ("r22_e1_days_since_halt", "r22_e2_halt_count_60", "r22_e3_resumed_within_5d", "r22_e4_last_halt_duration"):
            fam_e[col] = np.nan
        fam_e["r22_e_status"] = BLOCKED_DATA
    else:
        e_block = None
        halted = _halt_long(name)
        fam_e = keys[KEYS].merge(halted, on=KEYS, how="left")
        never = fam_e["r22_e1_days_since_halt"].isna()
        fam_e["r22_e_status"] = np.where(never, NEVER_SUSPENDED, "OK")
    print("[r22] family B peers", flush=True)
    peers = _peers(name, keys)
    print("[r22] family D minute high", flush=True)
    touch = _touches(keys, name, jobs)
    fam_d = _family_d(keys, name, touch)
    panel = keys.merge(fam_a, on=KEYS, how="left").merge(peers, on=KEYS, how="left").merge(fam_d, on=KEYS, how="left").merge(fam_e, on=KEYS, how="left")
    DATA.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    value_cols = [c for c in panel.columns if c.startswith("r22_") and c != "r22_e_status" and not c.startswith("r22_aux")]
    finite = {c: float(pd.to_numeric(panel[c], errors="coerce").notna().mean()) for c in value_cols}
    a1 = pd.to_numeric(panel["r22_a1_own_close_limit_excess_20"], errors="coerce")
    a4 = pd.to_numeric(panel["r22_a4_touch5_over_mkt_breadth"], errors="coerce")
    beta = pd.to_numeric(panel["r22_aux_mkt_beta"], errors="coerce")
    def _rho(u, v):
        m = u.notna() & v.notna()
        if int(m.sum()) < 50:
            return None
        return float(u.loc[m].corr(v.loc[m], method="spearman"))
    payload = {
        "status": "PARTIAL" if e_block else "DONE",
        "scan": scan_meta,
        "e_status": e_block or "FROM_PAUSED_SHIFT1",
        "peer_universe": "instruments_in_first_board_candidate_history",
        "peer_rule": "60d return corr ending D-1, overlap>=40, top_k=5 positive, exclude self",
        "touch_definition": "minute high >= limit_up, not vendor tick time",
        "d3": "EXPANDING_PAST_ONLY",
        "d4_label": "COMPOSITE_OF_KNOWN_FACTORS",
        "finite_share": finite,
        "rho_a1_a4": _rho(a1, a4),
        "rho_a1_beta": _rho(a1, beta),
        "n_rows": int(len(panel)),
        "filled_with_zero": False,
    }
    dump_json(META_PATH, payload)
    write_md(
        MD["09"],
        "\n".join(
            [
                "# Round22 · 09 构造 A/B/D/E",
                "",
                f"status=`{payload['status']}` rows=`{payload['n_rows']}`。",
                f"peer 宇宙=`{payload['peer_universe']}`。规则=`{payload['peer_rule']}`。",
                f"触板定义=`{payload['touch_definition']}`。D3=`{payload['d3']}`。D4=`{payload['d4_label']}`。",
                f"A1 与 A4 Spearman=`{payload['rho_a1_a4']}`。A1 与市场 beta Spearman=`{payload['rho_a1_beta']}`。",
                f"E=`{payload['e_status']}`。从未停牌记 `{NEVER_SUSPENDED}`，数值保持 NA，不填 0。",
                f"有限值占比=`{finite}`。",
                "",
                "注意力窗口只用 D-60…D-1。收盘涨停与 high 触板两套口径没有混称。",
                "",
            ]
        ),
    )
    return payload
