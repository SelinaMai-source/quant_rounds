"""实现 T/U/V。缺失不填 0。等价 G/H/K/L 记 DUPLICATE。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_2_ROOT, ROUND21_3_ROOT
from ..round18.f1 import _bar_seconds, _min_root
from ..round21.panel import KEYS, asof_by_instrument, load_candidate_keys, load_hermes_table, load_trade_calendar
from ..round21_2.construct import _finite, _known_close, _load_raw_table, _trading_days_between
from .common import dump_json, write_md
from .contract import DUPLICATE, EPS, MD, NEW_FACTOR_IDS
from .hypotheses import require_hypotheses

PANEL_PATH = ROUND21_3_ROOT / "data" / "factor_panel.parquet"
CONSTRUCT_JSON = ROUND21_3_ROOT / "construct" / "construct.json"
DUP_CSV = ROUND21_3_ROOT / "construct" / "formula_equivalence.csv"
U_PARTS = ROUND21_3_ROOT / "data" / "u_parts"
U_COLS = (
    "r21_u1_pre_post_hit_impact",
    "r21_u2_limit_vs_pullback_vol",
    "r21_u3_state_persist_switch",
    "r21_u4_pre_post_vol_conc",
    "r21_u5_open_recover_eff",
    "r21_u5_opened_flag",
)


def _cache(name: str) -> Path:
    return ROUND21_3_ROOT / "data" / f"{name}.parquet"


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        if not mask.any():
            return float("nan")
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def build_t(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache("t_factors")
    if path.is_file():
        return pd.read_parquet(path), {"resumed": True}
    ids = set(keys["instrument_id"].astype(str))
    equd = _load_raw_table("mkt_equd", ["TICKER_SYMBOL", "EXCHANGE_CD", "TRADE_DATE", "DEAL_AMOUNT", "UPDATE_TIME"], ids)
    mf = _load_raw_table(
        "mkt_equ_mf_new",
        [
            "TICKER_SYMBOL", "EXCHANGE_CD", "TRADE_DATE",
            "BUY_ORD_S", "BUY_ORD_M", "BUY_ORD_L", "BUY_ORD_XL",
            "BUY_VOL_XL", "SELL_VOL_XL", "BUY_ORD_XL", "SELL_ORD_XL",
            "UPDATE_TIME",
        ],
        ids,
    )
    daily = load_hermes_table("daily_raw", KEYS + ["amount"], instruments=ids)
    work = keys[KEYS].merge(daily, on=KEYS, how="left")
    e = equd[KEYS + ["DEAL_AMOUNT", "update_time"]].drop_duplicates(KEYS)
    work = work.merge(e, on=KEYS, how="left")
    work = work.loc[:, ~work.columns.duplicated()].copy()
    if "DEAL_AMOUNT" not in work.columns:
        work["DEAL_AMOUNT"] = np.nan
    work = work.sort_values(KEYS).reset_index(drop=True)
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work["DEAL_AMOUNT"] = pd.to_numeric(work["DEAL_AMOUNT"], errors="coerce")
    amt = work["amount"]
    deal = work["DEAL_AMOUNT"]
    ticket = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = amt.notna() & deal.gt(0)
    ticket.loc[ok] = amt.loc[ok] / deal.loc[ok]
    work["_ticket"] = ticket
    g = work.groupby("instrument_id", sort=False)
    med20 = g["_ticket"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).median())
    t1 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok1 = ticket.notna() & med20.gt(0)
    t1.loc[ok1] = ticket.loc[ok1] / med20.loc[ok1] - 1.0
    work["r21_t1_ticket_vs_med20"] = t1
    med_deal = g["DEAL_AMOUNT"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).median())
    t5 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok5 = deal.gt(0) & med_deal.gt(0)
    t5.loc[ok5] = deal.loc[ok5] / med_deal.loc[ok5] - 1.0
    work["r21_t5_deal_count_vs_med20"] = t5
    log_deal = np.log(deal.where(deal > 0))
    log_amt = np.log(amt.where(amt > 0))
    work["_log_deal"] = log_deal
    work["_log_amt"] = log_amt
    g = work.groupby("instrument_id", sort=False)
    work["r21_t4_dlog_count_minus_dlog_amt"] = g["_log_deal"].diff() - g["_log_amt"].diff()

    m = mf.drop_duplicates(KEYS)
    extra = [c for c in m.columns if c not in {"TICKER_SYMBOL", "EXCHANGE_CD", "TRADE_DATE"}]
    work = work.merge(m[extra], on=KEYS, how="left", suffixes=("", "_mf"))
    work = work.loc[:, ~work.columns.duplicated()].copy()

    def _col(name: str) -> pd.Series:
        if name not in work.columns:
            return pd.Series(np.nan, index=work.index, dtype="float64")
        s = work[name]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return _finite(s)

    bvx = _col("BUY_VOL_XL")
    box = _col("BUY_ORD_XL")
    svx = _col("SELL_VOL_XL")
    sox = _col("SELL_ORD_XL")
    g2 = pd.Series(np.nan, index=work.index, dtype="float64")
    ok2 = bvx.gt(0) & box.gt(0) & svx.gt(0) & sox.gt(0)
    g2.loc[ok2] = np.log((bvx.loc[ok2] / box.loc[ok2]) / (svx.loc[ok2] / sox.loc[ok2]))
    work["_g2"] = g2
    t3 = work.groupby("instrument_id", sort=False)["_g2"].diff()
    work["r21_t3_large_ticket_asym_chg"] = t3

    buckets = ["BUY_ORD_S", "BUY_ORD_M", "BUY_ORD_L", "BUY_ORD_XL"]
    for c in buckets:
        work[c] = _col(c)
    tot = work[buckets].sum(axis=1)
    shares = work[buckets].div(tot.replace(0, np.nan), axis=0)
    t2 = pd.Series(np.nan, index=work.index, dtype="float64")
    for iid, idx in work.groupby("instrument_id", sort=False).groups.items():
        ix = list(idx)
        mat = shares.loc[ix].to_numpy(dtype=float)
        hist = np.full(4, np.nan)
        vals = np.full(len(ix), np.nan)
        for i in range(len(ix)):
            past = mat[max(0, i - 20) : i]
            past = past[np.isfinite(past).all(axis=1)]
            if len(past) < 5 or not np.isfinite(mat[i]).all():
                continue
            hist = past.mean(axis=0)
            vals[i] = _js(mat[i], hist)
        t2.loc[ix] = vals
    work["r21_t2_bucket_js_div"] = t2
    keep = KEYS + [c for c in work.columns if c.startswith("r21_t")]
    out = work[keep].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out, {"status": "DONE", "source": "raw mkt_equd + mkt_equ_mf_new"}


def _u_part_path(day: pd.Timestamp) -> Path:
    return U_PARTS / f"{pd.Timestamp(day).strftime('%Y%m%d')}.parquet"


def _empty_u(part: pd.DataFrame, day: pd.Timestamp | None = None) -> pd.DataFrame:
    out = part[KEYS].copy()
    for c in U_COLS:
        out[c] = np.nan
    if day is not None:
        out["trade_date"] = pd.Timestamp(day)
    elif "trade_date" not in out.columns and not part.empty:
        out["trade_date"] = part["trade_date"].iloc[0]
    return out


def _u_one_day(minutes: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    day = pd.Timestamp(part["trade_date"].iloc[0]) if not part.empty and "trade_date" in part.columns else None
    out = _empty_u(part, day)
    if minutes.empty or part.empty:
        return out
    m = minutes.copy()
    m["instrument_id"] = m["instrument_id"].astype(str)
    if "bar_time" in m.columns:
        m["bar_sec"] = _bar_seconds(m["bar_time"])
    elif "bar_sec" not in m.columns:
        return out
    vol = pd.to_numeric(m.get("volume", m.get("vol")), errors="coerce")
    m["volume"] = vol
    m["close"] = pd.to_numeric(m.get("close"), errors="coerce")
    recs = []
    joined = m.merge(part[["instrument_id", "touch_seconds", "limit_price"]], on="instrument_id", how="inner")
    for iid, g in joined.groupby("instrument_id", sort=False):
        rec = {"instrument_id": str(iid)}
        v = pd.to_numeric(g["volume"], errors="coerce").to_numpy(dtype=float)
        sec = pd.to_numeric(g["bar_sec"], errors="coerce").to_numpy(dtype=float)
        cl = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=float)
        touch = float(pd.to_numeric(g["touch_seconds"].iloc[0], errors="coerce"))
        lim = float(pd.to_numeric(g["limit_price"].iloc[0], errors="coerce"))
        ok = np.isfinite(v) & np.isfinite(sec) & np.isfinite(cl)
        if int(ok.sum()) < 5 or not np.isfinite(touch):
            recs.append(rec)
            continue
        vv, ss, cc = v[ok], sec[ok], cl[ok]
        pre = ss < touch
        post = ss >= touch
        dcl = np.abs(np.diff(cc, prepend=cc[0]))
        if int(pre.sum()) > 0 and int(post.sum()) > 0:
            rec["r21_u1_pre_post_hit_impact"] = float(dcl[post].mean() - dcl[pre].mean())
            pre_v = vv[pre]
            post_v = vv[post]
            def _hhi(x):
                x = x[np.isfinite(x) & (x > 0)]
                if len(x) < 2:
                    return float("nan")
                p = x / x.sum()
                return float(np.sum(p * p))

            rec["r21_u4_pre_post_vol_conc"] = _hhi(post_v) - _hhi(pre_v)
        if np.isfinite(lim) and int(post.sum()) > 0:
            onboard = post & (cc >= lim * 0.999)
            pull = post & (cc < lim * 0.999)
            vo = float(vv[onboard].sum()) if onboard.any() else 0.0
            vp = float(vv[pull].sum()) if pull.any() else float("nan")
            if np.isfinite(vp) and vp > 0:
                rec["r21_u2_limit_vs_pullback_vol"] = vo / vp - 1.0
            states = onboard.astype(int)
            if len(states) > 1:
                switches = int(np.sum(np.diff(states) != 0))
                rec["r21_u3_state_persist_switch"] = switches / max(int(post.sum()), 1)
            opened = bool(pull.any())
            rec["r21_u5_opened_flag"] = float(opened)
            if opened:
                # 第一次离开板后回到板
                post_idx = np.where(post)[0]
                left = None
                back = None
                for i in post_idx:
                    if cc[i] < lim * 0.999 and left is None:
                        left = i
                    if left is not None and cc[i] >= lim * 0.999:
                        back = i
                        break
                if left is not None and back is not None:
                    rec["r21_u5_open_recover_eff"] = 1.0 / max(float(ss[back] - ss[left]), 1.0)
        recs.append(rec)
    if recs:
        got = pd.DataFrame(recs)
        out = out.drop(columns=[c for c in out.columns if c.startswith("r21_u")]).merge(got, on="instrument_id", how="left")
        out["trade_date"] = part["trade_date"].iloc[0]
    return out


def build_u(keys: pd.DataFrame, *, jobs: int = 8) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache("u_factors")
    if path.is_file():
        return pd.read_parquet(path), {"resumed": True}
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from ..round21.min1 import _load_limit_px

    ac_path = Path(__file__).resolve().parents[2] / "result" / "round21" / "data" / "ac_factors.parquet"
    base = _load_limit_px(keys)
    if ac_path.is_file():
        ac = pd.read_parquet(ac_path, columns=KEYS + ["touch_seconds"])
        ac["trade_date"] = pd.to_datetime(ac["trade_date"])
        ac["instrument_id"] = ac["instrument_id"].astype(str)
        base = base.merge(ac, on=KEYS, how="left", suffixes=("", "_ac"))
        if "touch_seconds" not in base.columns and "touch_seconds_ac" in base.columns:
            base["touch_seconds"] = base["touch_seconds_ac"]
        elif "touch_seconds_ac" in base.columns:
            base["touch_seconds"] = _finite(base["touch_seconds"]).fillna(_finite(base["touch_seconds_ac"]))
    else:
        base["touch_seconds"] = _finite(base.get("first_limit_time_seconds"))
    if "limit_price" not in base.columns:
        ev = load_hermes_table("daily_limit_events", KEYS + ["limit_price"], instruments=set(keys["instrument_id"].astype(str)))
        base = base.merge(ev, on=KEYS, how="left")
    parts = []
    tasks = []
    U_PARTS.mkdir(parents=True, exist_ok=True)
    n_resume = 0
    for day, block in base.groupby("trade_date", sort=True):
        cached = _u_part_path(pd.Timestamp(day))
        if cached.is_file():
            parts.append(pd.read_parquet(cached))
            n_resume += 1
            continue
        root = _min_root(pd.Timestamp(day))
        part = block[KEYS + [c for c in ("touch_seconds", "limit_price") if c in block.columns]].copy()
        if root is None:
            empty = _empty_u(part, pd.Timestamp(day))
            empty.to_parquet(cached, index=False)
            parts.append(empty)
            continue
        tasks.append((str(root), part, cached))
    print(f"[round21.3/u] resume_days={n_resume} todo_days={len(tasks)}", flush=True)
    if tasks:
        done = 0
        if int(jobs) <= 1:
            for root, part, cached in tasks:
                files = list(Path(root).glob("*.parquet"))
                minutes = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
                got = _u_one_day(minutes, part)
                got.to_parquet(cached, index=False)
                parts.append(got)
                done += 1
                if done % 50 == 0:
                    print(f"[round21.3/u] days {done}/{len(tasks)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
                futs = [pool.submit(_u_job, root, part, str(cached)) for root, part, cached in tasks]
                for fut in as_completed(futs):
                    parts.append(fut.result())
                    done += 1
                    if done % 50 == 0:
                        print(f"[round21.3/u] days {done}/{len(tasks)}", flush=True)
    out = pd.concat(parts, ignore_index=True) if parts else keys[KEYS].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out, {"status": "DONE", "n": int(len(out)), "u_parts": str(U_PARTS), "resumed_days": n_resume, "computed_days": len(tasks)}


def _u_job(root: str, part: pd.DataFrame, cached: str | None = None) -> pd.DataFrame:
    files = list(Path(root).glob("*.parquet"))
    minutes = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    got = _u_one_day(minutes, part)
    if cached:
        Path(cached).parent.mkdir(parents=True, exist_ok=True)
        got.to_parquet(cached, index=False)
    return got


def build_v(keys: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache("v_factors")
    if path.is_file():
        return pd.read_parquet(path), {"resumed": True}
    sc = load_hermes_table("share_capital", ["instrument_id", "available_at", "announced_at", "tradable_shares", "effective_from"])
    sc["available_at"] = pd.to_datetime(sc["available_at"], errors="coerce")
    sc["effective_from"] = pd.to_datetime(sc.get("effective_from"), errors="coerce")
    sc["tradable_shares"] = _finite(sc["tradable_shares"])
    sc = sc.dropna(subset=["instrument_id", "available_at"]).sort_values(["instrument_id", "available_at"])
    sc["prev_float"] = sc.groupby("instrument_id", sort=False)["tradable_shares"].shift(1)
    sc["float_chg"] = sc["tradable_shares"] / sc["prev_float"] - 1.0
    left = keys[KEYS].copy()
    left["known_at"] = _known_close(left["trade_date"])
    joined = asof_by_instrument(
        left,
        sc,
        left_on="known_at",
        right_on="available_at",
        columns=["tradable_shares", "prev_float", "float_chg", "effective_from", "announced_at"],
    )
    ids = set(keys["instrument_id"].astype(str))
    daily = load_hermes_table("daily_raw", KEYS + ["volume"], instruments=ids)
    daily = daily.sort_values(KEYS)
    daily["volume"] = _finite(daily["volume"])
    daily["vol20"] = daily.groupby("instrument_id", sort=False)["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    work = joined.merge(daily[KEYS + ["vol20"]], on=KEYS, how="left")
    v1 = _trading_days_between(pd.to_datetime(work.get("available_at"), errors="coerce"), pd.to_datetime(work.get("effective_from"), errors="coerce"), calendar)
    # 生效已过：若 effective < trade_date 则 NA（前瞻窗关闭）
    eff = pd.to_datetime(work.get("effective_from"), errors="coerce")
    v1 = v1.where(eff.ge(pd.to_datetime(work["trade_date"])))
    chg = _finite(work.get("float_chg"))
    shock = _finite(work.get("tradable_shares")) - _finite(work.get("prev_float"))
    v2 = np.sign(chg) * np.log1p(np.abs(shock))
    v2 = pd.Series(v2, index=work.index, dtype="float64")
    v2 = v2.where(_finite(work.get("prev_float")).notna())
    v3 = pd.Series(np.nan, index=work.index, dtype="float64")
    v20 = _finite(work["vol20"])
    ok = shock.notna() & v20.gt(0)
    v3.loc[ok] = shock.loc[ok] / v20.loc[ok]
    # overlap: events already known, effective in (D, D+20 trading days]
    v4 = pd.Series(np.nan, index=work.index, dtype="float64")
    sc2 = sc.dropna(subset=["effective_from", "available_at"]).copy()
    sc2["instrument_id"] = sc2["instrument_id"].astype(str)
    if not sc2.empty:
        cal = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
        cal_map = {pd.Timestamp(d): i for i, d in enumerate(cal)}
        ev = sc2[["instrument_id", "available_at", "effective_from"]].copy()
        ev["_e_idx"] = pd.to_datetime(ev["effective_from"]).dt.normalize().map(cal_map)
        left = work[KEYS + ["known_at"]].copy()
        left["instrument_id"] = left["instrument_id"].astype(str)
        left["_i"] = np.arange(len(left))
        left["_d_idx"] = pd.to_datetime(left["trade_date"]).dt.normalize().map(cal_map)
        merged = left.merge(ev, on="instrument_id", how="left")
        delta = merged["_e_idx"] - merged["_d_idx"]
        ok = (
            merged["available_at"].notna()
            & (merged["available_at"] <= merged["known_at"])
            & delta.notna()
            & delta.gt(0)
            & delta.le(20)
        )
        cnt = merged.loc[ok].groupby("_i").size()
        v4 = pd.Series(0.0, index=work.index, dtype="float64")
        v4.loc[cnt.index] = cnt.astype("float64")
        v4 = v4.where(work["instrument_id"].astype(str).isin(set(ev["instrument_id"])))
    out = work[KEYS].copy()
    out["r21_v1_ann_to_effective_days"] = v1
    out["r21_v2_classified_float_shock"] = v2
    out["r21_v3_float_shock_vs_liq"] = v3
    out["r21_v4_future_event_overlap"] = v4
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out, {"status": "DONE"}


def _dup_check(panel: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    g1p = ROUND21_2_ROOT / "data" / "g_factors.parquet"
    l2p = ROUND21_2_ROOT / "data" / "l_factors.parquet"
    k2p = ROUND21_2_ROOT / "data" / "k_factors.parquet"
    checks = []
    if g1p.is_file() and "r21_t1_ticket_vs_med20" in panel.columns:
        g = pd.read_parquet(g1p)
        m = panel[KEYS + ["r21_t1_ticket_vs_med20"]].merge(g[KEYS + ["r21_g1_avg_ticket_value"]], on=KEYS, how="inner")
        checks.append(("r21_t1_ticket_vs_med20", "r21_g1_avg_ticket_value", m))
    if g1p.is_file() and "r21_t3_large_ticket_asym_chg" in panel.columns:
        g = pd.read_parquet(g1p)
        m = panel[KEYS + ["r21_t3_large_ticket_asym_chg"]].merge(g[KEYS + ["r21_g2_large_ticket_asym"]], on=KEYS, how="inner")
        checks.append(("r21_t3_large_ticket_asym_chg", "r21_g2_large_ticket_asym", m))
    if l2p.is_file() and "r21_v3_float_shock_vs_liq" in panel.columns:
        l = pd.read_parquet(l2p)
        m = panel[KEYS + ["r21_v3_float_shock_vs_liq"]].merge(l[KEYS + ["r21_l2_float_shock_vs_liq"]], on=KEYS, how="inner")
        checks.append(("r21_v3_float_shock_vs_liq", "r21_l2_float_shock_vs_liq", m))
    if l2p.is_file() and "r21_v1_ann_to_effective_days" in panel.columns:
        l = pd.read_parquet(l2p)
        if "r21_l3_days_since_float_up_ann" in l.columns:
            m = panel[KEYS + ["r21_v1_ann_to_effective_days"]].merge(l[KEYS + ["r21_l3_days_since_float_up_ann"]], on=KEYS, how="inner")
            checks.append(("r21_v1_ann_to_effective_days", "r21_l3_days_since_float_up_ann", m))
    if k2p.is_file() and "r21_u4_pre_post_vol_conc" in panel.columns:
        k = pd.read_parquet(k2p)
        for old in ("r21_k2_vol_time_dispersion", "r21_k3_minute_vol_entropy"):
            if old in k.columns:
                m = panel[KEYS + ["r21_u4_pre_post_vol_conc"]].merge(k[KEYS + [old]], on=KEYS, how="inner")
                checks.append(("r21_u4_pre_post_vol_conc", old, m))
    for new, old, m in checks:
        a = pd.to_numeric(m[new], errors="coerce")
        b = pd.to_numeric(m[old], errors="coerce")
        mask = a.notna() & b.notna()
        rho = float(a.loc[mask].corr(b.loc[mask], method="spearman")) if int(mask.sum()) > 30 else None
        equal = bool(mask.any() and np.allclose(a.loc[mask].to_numpy(), b.loc[mask].to_numpy(), equal_nan=False, rtol=1e-6, atol=1e-8))
        dup = bool(equal or (rho is not None and abs(rho) >= 0.999))
        rows.append({"new": new, "old": old, "spearman": rho, "elementwise_equal": equal, "verdict": DUPLICATE if dup else "DISTINCT"})
    return rows


def run_construct(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    if resume and PANEL_PATH.is_file() and CONSTRUCT_JSON.is_file() and MD["04"].is_file():
        from .common import load_json

        existing = load_json(CONSTRUCT_JSON)
        t_blocked = (existing.get("families") or {}).get("T", {}).get("status") == "BLOCKED_DATA"
        if existing.get("status") == "DONE" and not t_blocked:
            return existing
        if t_blocked:
            cache_t = _cache("t_factors")
            if cache_t.is_file():
                cache_t.unlink()
    require_hypotheses()
    keys = load_candidate_keys()
    cal = load_trade_calendar()
    blocked: list[str] = []
    frames = []
    meta: dict[str, Any] = {}
    try:
        t, mt = build_t(keys)
        frames.append(t)
        meta["T"] = mt
    except Exception as exc:
        import traceback

        blocked.extend([c for c in NEW_FACTOR_IDS if c.startswith("r21_t")])
        meta["T"] = {"status": "BLOCKED_DATA", "reason": str(exc), "traceback": traceback.format_exc()}
    try:
        u, mu = build_u(keys, jobs=jobs)
        frames.append(u)
        meta["U"] = mu
    except Exception as exc:
        blocked.extend([c for c in NEW_FACTOR_IDS if c.startswith("r21_u")])
        meta["U"] = {"status": "BLOCKED_DATA", "reason": str(exc)}
    try:
        v, mv = build_v(keys, cal)
        frames.append(v)
        meta["V"] = mv
    except Exception as exc:
        blocked.extend([c for c in NEW_FACTOR_IDS if c.startswith("r21_v")])
        meta["V"] = {"status": "BLOCKED_DATA", "reason": str(exc)}
    panel = keys[KEYS].copy()
    for fr in frames:
        extra = [c for c in fr.columns if c not in KEYS]
        panel = panel.merge(fr[KEYS + extra], on=KEYS, how="left")
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    dups = _dup_check(panel)
    pd.DataFrame(dups).to_csv(DUP_CSV, index=False)
    computed = [c for c in NEW_FACTOR_IDS if c in panel.columns and pd.to_numeric(panel[c], errors="coerce").notna().any()]
    payload = {
        "status": "DONE",
        "n_computed": len(computed),
        "computed": computed,
        "blocked": blocked,
        "duplicates": dups,
        "families": meta,
        "panel": str(PANEL_PATH),
        "filled_with_zero": False,
    }
    dump_json(CONSTRUCT_JSON, payload)
    write_md(MD["04"], f"# Round21.3 · 04 构造\n\ncomputed=`{computed}` blocked=`{blocked}` dups=`{dups}` 缺失不填 0。\n")
    return payload
