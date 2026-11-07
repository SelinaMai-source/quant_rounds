"""实现 W/X/Y/Z。缺失不填 0。等价记 DUPLICATE。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_4_ROOT
from ..round21.panel import KEYS, load_candidate_keys, load_hermes_table
from .common import dump_json, write_md
from .contract import BLOCKED_DATA, DUPLICATE, EPS, MD, WXYZ_FACTOR_IDS
from .hypotheses import require_hypotheses

PANEL_PATH = ROUND21_4_ROOT / "data" / "factor_panel.parquet"
CONSTRUCT_JSON = ROUND21_4_ROOT / "construct" / "construct.json"
DUP_CSV = ROUND21_4_ROOT / "construct" / "formula_equivalence.csv"


def _finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _peer_key(iid: pd.Series) -> pd.Series:
    s = iid.astype(str)
    exch = s.str.split(".").str[-1]
    prefix = s.str.split(".").str[0].str[:3]
    return exch + ":" + prefix


def _load_daily(ids: set[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys([*KEYS, "open", "high", "low", "close", "pre_close", "volume", "amount", "turnover", "limit_up", "pct_change"]))
    daily = load_hermes_table("daily_raw", cols, instruments=ids)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily["instrument_id"] = daily["instrument_id"].astype(str)
    for c in ("open", "high", "low", "close", "pre_close", "volume", "amount", "turnover", "limit_up", "pct_change"):
        if c in daily.columns:
            daily[c] = _finite(daily[c])
    if "turnover" not in daily.columns or daily["turnover"].isna().all():
        if "amount" in daily.columns and "close" in daily.columns:
            daily["turnover"] = daily["amount"]
    daily = daily.sort_values(KEYS).drop_duplicates(KEYS)
    return daily


def _board_flag(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    pre = work["pre_close"] if "pre_close" in work.columns else work["close"].groupby(work["instrument_id"]).shift(1)
    lim = work["limit_up"] if "limit_up" in work.columns else pd.Series(np.nan, index=work.index)
    ratio = lim / pre.replace(0, np.nan)
    work["_board"] = np.where(ratio >= 1.15, "20", np.where(ratio >= 1.08, "10", np.where(ratio >= 1.04, "ST5", "other")))
    near = (work["high"] >= 0.98 * lim) & lim.gt(0)
    unsealed = near & (work["close"] < 0.995 * lim)
    work["_near"] = near.astype("float")
    work["_unsealed"] = unsealed.astype("float")
    work["_gap"] = np.where(near & lim.gt(0), (lim - work["close"]) / lim, np.nan)
    return work


def build_wxyz(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if PANEL_PATH.is_file():
        return pd.read_parquet(PANEL_PATH), {"resumed": True}
    ids = set(keys["instrument_id"].astype(str))
    daily = _load_daily(ids)
    if daily.empty:
        raise RuntimeError(f"{BLOCKED_DATA} empty daily_raw")
    daily = _board_flag(daily)
    g = daily.groupby("instrument_id", sort=False)
    close = daily["close"]
    ret = g["close"].pct_change()
    vol = daily["volume"] if "volume" in daily.columns else pd.Series(np.nan, index=daily.index)
    amt = daily["amount"] if "amount" in daily.columns else pd.Series(np.nan, index=daily.index)
    to = daily["turnover"] if "turnover" in daily.columns else amt
    daily["_ret"] = ret
    daily["_up"] = ret.gt(0).astype(float)
    daily["_dn"] = ret.lt(0).astype(float)
    daily["_absret"] = ret.abs()
    daily["_impact"] = daily["_absret"] / to.replace(0, np.nan)

    def roll_mean(s, n, minp):
        return g[s].transform(lambda x: x.shift(1).rolling(n, min_periods=minp).mean()) if s in daily.columns else pd.Series(np.nan, index=daily.index)

    # W1 涨日/跌日量比
    vol_up = g["volume"].transform(lambda s: s.where(ret.reindex(s.index) > 0).shift(1).rolling(20, min_periods=5).mean())
    vol_dn = g["volume"].transform(lambda s: s.where(ret.reindex(s.index) < 0).shift(1).rolling(20, min_periods=5).mean())
    w1 = vol_up / vol_dn.replace(0, np.nan) - 1.0

    w2 = g["_impact"].transform(lambda s: s.shift(1).rolling(20, min_periods=8).std())

    max20 = g["amount"].transform(lambda s: s.shift(2).rolling(20, min_periods=5).max())
    w3 = amt.shift(1) / max20.replace(0, np.nan) - 1.0
    # W4 响应不对称：涨日量份额 − 跌日量份额（向量化，避免逐行 corr）
    w4 = (vol_up - vol_dn) / (vol_up + vol_dn).replace(0, np.nan)

    x1 = g["_unsealed"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    x2 = g["_gap"].transform(lambda s: s.shift(1).rolling(20, min_periods=3).mean())
    near = daily["_near"].fillna(0.0)
    x3 = g["_near"].transform(lambda s: _streak(s.shift(1)))
    x4_num = g.apply(lambda d: (d["volume"].where(d["_near"] > 0)).shift(1).rolling(20, min_periods=5).sum(), include_groups=False)
    x4_den = g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).sum())
    x4 = (near.shift(1) * vol).groupby(daily["instrument_id"]).transform(lambda s: s.rolling(20, min_periods=5).sum()) / x4_den.replace(0, np.nan)

    daily["_to_innov"] = to.shift(1) / g["turnover"].transform(lambda s: s.shift(2).rolling(20, min_periods=5).max()).replace(0, np.nan) - 1.0
    daily["_dto"] = g["turnover"].transform(lambda s: s.shift(1) - s.shift(6))
    daily["_peer"] = _peer_key(daily["instrument_id"])
    daily["_to_lag"] = to.shift(1)
    y1 = _cs_demean(daily, "_to_innov", "_peer")
    y2 = _cs_demean(daily, "_dto", "_peer")
    y3 = _cs_rank(daily, "_to_lag", "_peer")
    y4 = _cs_demean(daily, "_impact", "_peer")

    std5 = g["_ret"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).std())
    std20 = g["_ret"].transform(lambda s: s.shift(1).rolling(20, min_periods=8).std())
    z1 = std5 / std20.replace(0, np.nan)
    down = g["_ret"].transform(lambda s: s.shift(1).clip(upper=0).pow(2).rolling(20, min_periods=8).mean())
    var20 = g["_ret"].transform(lambda s: s.shift(1).pow(2).rolling(20, min_periods=8).mean())
    z2 = down / var20.replace(0, np.nan)
    to20 = g["turnover"].transform(lambda s: s.shift(1).rolling(20, min_periods=8).mean())
    z3 = std20 / to20.replace(0, np.nan)
    mn = g["low"].transform(lambda s: s.shift(1).rolling(20, min_periods=8).min()) if "low" in daily.columns else g["close"].transform(lambda s: s.shift(1).rolling(20, min_periods=8).min())
    mx = g["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=8).max()) if "high" in daily.columns else g["close"].transform(lambda s: s.shift(1).rolling(20, min_periods=8).max())
    z4 = (close.shift(1) - mn) / (mx - mn).replace(0, np.nan)

    out = daily[KEYS].copy()
    out["r21_w1_updown_vol_ratio"] = w1.reindex(daily.index).to_numpy()
    out["r21_w2_impact_instability"] = w2.to_numpy()
    out["r21_w3_volume_innov_residual"] = w3.to_numpy()
    out["r21_w4_response_asymmetry"] = pd.Series(w4).reindex(daily.index).to_numpy()
    out["r21_x1_near_limit_unsealed_freq20"] = x1.to_numpy()
    out["r21_x2_close_to_limit_gap"] = x2.to_numpy()
    out["r21_x3_near_limit_persist"] = x3.to_numpy()
    out["r21_x4_touch_day_vol_share"] = x4.to_numpy()
    out["r21_y1_turnover_innov_minus_peer"] = y1.to_numpy()
    out["r21_y2_turnover_accel_diff"] = y2.to_numpy()
    out["r21_y3_peer_quantile_exposure"] = y3.to_numpy()
    out["r21_y4_impact_diff_vs_peer"] = y4.to_numpy()
    out["r21_z1_vol5_vol20_ratio"] = z1.to_numpy()
    out["r21_z2_downside_semivar_ratio"] = z2.to_numpy()
    out["r21_z3_vol_per_turnover"] = z3.to_numpy()
    out["r21_z4_range_position"] = z4.to_numpy()

    panel = keys[KEYS].merge(out, on=KEYS, how="left")
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    return panel, {"resumed": False, "n": int(len(panel))}


def _streak(s: pd.Series) -> pd.Series:
    x = (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)
    out = np.zeros(len(x), dtype=float)
    run = 0
    for i, v in enumerate(x.to_numpy()):
        run = run + 1 if v else 0
        out[i] = run if run else np.nan
    return pd.Series(out, index=s.index)


def _cs_demean(daily: pd.DataFrame, col: str, peer: str) -> pd.Series:
    s = pd.to_numeric(daily[col], errors="coerce")
    tmp = daily.assign(_v=s, _p=daily[peer].astype(str), _d=daily["trade_date"])
    med = tmp.groupby(["_d", "_p"])["_v"].transform("median")
    cnt = tmp.groupby(["_d", "_p"])["_v"].transform("count")
    out = s - med
    out = out.where(cnt >= 5)
    return out


def _cs_rank(daily: pd.DataFrame, col: str, peer: str) -> pd.Series:
    s = pd.to_numeric(daily[col], errors="coerce")
    tmp = daily.assign(_v=s, _p=daily[peer].astype(str), _d=daily["trade_date"])
    rk = tmp.groupby(["_d", "_p"])["_v"].rank(pct=True)
    cnt = tmp.groupby(["_d", "_p"])["_v"].transform("count")
    return rk.where(cnt >= 5)


def _dup_check(panel: pd.DataFrame) -> list[dict[str, Any]]:
    from ..round7.contract import FIVE_FEATURES

    rows = []
    for fid in WXYZ_FACTOR_IDS:
        if fid not in panel.columns:
            continue
        a = pd.to_numeric(panel[fid], errors="coerce")
        best, best_abs = None, 0.0
        for j in FIVE_FEATURES:
            if j not in panel.columns:
                continue
            b = pd.to_numeric(panel[j], errors="coerce")
            mask = a.notna() & b.notna()
            if int(mask.sum()) < 50:
                continue
            rho = float(a.loc[mask].corr(b.loc[mask], method="spearman"))
            if abs(rho) > best_abs:
                best_abs, best = abs(rho), j
        verdict = DUPLICATE if best_abs >= 0.995 else "DISTINCT"
        rows.append({"new": fid, "old": best, "abs_rho": best_abs, "verdict": verdict})
    return rows


def run_construct(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and CONSTRUCT_JSON.is_file() and PANEL_PATH.is_file() and MD["21"].is_file():
        existing = dump_json  # noqa: keep import used
        from .common import load_json

        existing = load_json(CONSTRUCT_JSON)
        if existing.get("status") in {"DONE", "PARTIAL"}:
            return existing
    require_hypotheses()
    from .hypotheses import run_hypotheses

    run_hypotheses(resume=True)
    keys = load_candidate_keys()
    try:
        panel, meta = build_wxyz(keys)
    except Exception as exc:
        payload = {"status": "BLOCKED", "reason": str(exc), "computed": [], "blocked": list(WXYZ_FACTOR_IDS)}
        dump_json(CONSTRUCT_JSON, payload)
        write_md(MD["21"], f"# Round21.4 · 21 构造\n\nBLOCKED `{exc}`\n")
        return payload
    dups = _dup_check(panel)
    pd.DataFrame(dups).to_csv(DUP_CSV, index=False)
    computed = []
    blocked = []
    for fid in WXYZ_FACTOR_IDS:
        if fid not in panel.columns:
            blocked.append(fid)
            continue
        n = int(np.isfinite(pd.to_numeric(panel[fid], errors="coerce").to_numpy(dtype=float)).sum())
        if n <= 0:
            blocked.append(fid)
        else:
            computed.append(fid)
    payload = {
        "status": "DONE" if computed else "PARTIAL",
        "n_computed": len(computed),
        "computed": computed,
        "blocked": blocked,
        "duplicates": dups,
        "panel": str(PANEL_PATH),
        "n_rows": int(len(panel)),
        "filled_with_zero": False,
        "meta": meta,
    }
    dump_json(CONSTRUCT_JSON, payload)
    write_md(MD["21"], f"# Round21.4 · 21 构造\n\ncomputed=`{computed}` blocked=`{blocked}` dups=`{[d for d in dups if d.get('verdict')==DUPLICATE]}`\n")
    return payload
