"""匹配对照。O01/O03 ≥1000 perm；O02 常数平均暴露 + 年分层 permute（禁止数学等价逐日暴露对照）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_ROOT
from ..round11.common import daily_from_nav, sharpe_of
from .backtest import load_backtest, load_treat_dates
from .common import dump_json, load_json, write_md
from .contract import O01, O02, O03, O04, PERM_N, R16_00, experiment_dir, parity_dir
from .freeze import require_frozen

CTRL_JSON = ROUND16_ROOT / "controls" / "controls.json"
CTRL_MD = ROUND16_ROOT / "09_matched_controls.md"


def _daily(code: str) -> pd.Series | None:
    if code in {R16_00, "v4"}:
        p = parity_dir(R16_00) / "nav.parquet"
    else:
        p = experiment_dir(code) / "nav.parquet"
    if not p.is_file():
        return None
    return daily_from_nav(pd.read_parquet(p))


def _top2_pnl() -> pd.Series:
    p = parity_dir(R16_00) / "trades.parquet"
    if not p.is_file():
        return pd.Series(dtype=float)
    t = pd.read_parquet(p)
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["_rank"] = t.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    t["net"] = pd.to_numeric(t.get("sell_proceeds"), errors="coerce") - pd.to_numeric(t.get("buy_cost"), errors="coerce")
    s = t.loc[t["_rank"].eq(2)].groupby("signal_date")["net"].sum()
    return s.sort_index()


def _perm_veto(v4: pd.Series, cand: pd.Series, treat: pd.DatetimeIndex, *, n: int, seed: int) -> dict[str, Any]:
    """同年随机同等数量 Top2 veto 日。用候选相对 v4 的日收益作检验，不是逐日暴露对照。"""
    aligned = pd.concat({"v4": v4, "c": cand}, axis=1).dropna()
    if aligned.empty or len(treat) == 0:
        return {"n": 0, "p_cand_gt_ctrl": None}
    treat = pd.DatetimeIndex(pd.to_datetime(treat).normalize())
    n_treat = int(aligned.index.isin(treat).sum())
    rng = np.random.default_rng(seed)
    actual = float(sharpe_of(aligned["c"]))
    beat = 0
    n_ok = 0
    years = aligned.index.year
    for _ in range(n):
        picked = []
        for _, g in aligned.groupby(years):
            n_y = int(g.index.isin(treat).sum())
            if n_y <= 0:
                continue
            chosen = rng.choice(g.index.to_numpy(), size=min(n_y, len(g)), replace=False)
            picked.extend(list(chosen))
        if not picked:
            continue
        dummy = aligned["v4"].copy()
        dummy.loc[dummy.index.isin(pd.DatetimeIndex(picked))] = aligned.loc[
            aligned.index.isin(pd.DatetimeIndex(picked)), "c"
        ]
        sh = sharpe_of(dummy)
        if sh is not None and np.isfinite(sh):
            n_ok += 1
            if actual > sh:
                beat += 1
    return {
        "n": n,
        "n_ok": n_ok,
        "n_treat_days": n_treat,
        "p_cand_gt_ctrl": (beat / n_ok) if n_ok else None,
        "kind": "year_stratified_same_count_top2_veto",
        "not_daily_exposure_match": True,
        "actual_sharpe": actual,
    }


def _o02_controls(v4: pd.Series, cand: pd.Series, treat: pd.DatetimeIndex, *, n: int) -> dict[str, Any]:
    aligned = pd.concat({"v4": v4, "c": cand}, axis=1).dropna()
    if aligned.empty:
        return {"constant_avg": None, "year_permute": None}
    treat = pd.DatetimeIndex(pd.to_datetime(treat).normalize())
    mask = aligned.index.isin(treat)
    # 常数平均暴露：所有日乘以候选的平均相对收益比（不是逐日复制 O02 乘数）
    scale = float((aligned["c"] / aligned["v4"].replace(0, np.nan)).mean())
    if not np.isfinite(scale) or scale <= 0:
        scale = 0.75
    constant = aligned["v4"] * scale
    const_sh = sharpe_of(constant)
    cand_sh = sharpe_of(aligned["c"])
    year_perm = _perm_veto(v4, cand, treat, n=n, seed=20261014)
    year_perm["kind"] = "year_regime_permuted_timing_not_daily_exposure_match"
    return {
        "constant_avg_exposure": {
            "scale": scale,
            "sharpe": const_sh,
            "cand_sharpe": cand_sh,
            "cand_gt_constant": (cand_sh > const_sh) if (cand_sh is not None and const_sh is not None) else None,
            "not_mathematically_equivalent_daily_control": True,
        },
        "year_permute": year_perm,
    }


def run_controls(**_: Any) -> dict[str, Any]:
    require_frozen()
    bt = load_backtest()
    treat = load_treat_dates()
    results = bt.get("results") or {}
    v4 = _daily("v4")
    out: dict[str, Any] = {}
    n = max(int(PERM_N), 1000)
    for code in (O01, O03):
        row = results.get(code) or {}
        if row.get("skip") or v4 is None:
            out[code] = {"status": row.get("status") or "SKIP", "n": 0}
            continue
        cand = _daily(code)
        dates = pd.DatetimeIndex(pd.to_datetime(treat.get(code) or []))
        out[code] = _perm_veto(v4, cand, dates, n=n, seed=20261016 + hash(code) % 97)
        out[code]["status"] = "DONE"
    row = results.get(O02) or {}
    if row.get("skip") or v4 is None:
        out[O02] = {"status": row.get("status") or "SKIP"}
    else:
        cand = _daily(O02)
        dates = pd.DatetimeIndex(pd.to_datetime(treat.get(O02) or []))
        out[O02] = _o02_controls(v4, cand, dates, n=n)
        out[O02]["status"] = "DONE"
    out[O04] = {"status": (results.get(O04) or {}).get("status") or "SKIP_O04", "note": "equity-only vs equity×option same universe; skipped with candidate"}
    payload = {
        "status": "DONE",
        "perm_n": n,
        "results": out,
        "not_daily_exposure_match": True,
        "note": "O01/O03 同年同等数量随机 Top2 veto ≥1000 perm。O02 常数平均暴露 + 年分层 permute。",
    }
    dump_json(CTRL_JSON, payload)
    write_md(CTRL_MD, f"# Round16 · 09 匹配对照\n\nperm_n=`{n}`\n\n{out}\n")
    return payload


def load_controls() -> dict[str, Any]:
    return load_json(CTRL_JSON)
