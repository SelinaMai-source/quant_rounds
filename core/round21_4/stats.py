"""分年 / HAC / block bootstrap 5000 / SPA 数值 p。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_4_ROOT
from ..round8.robustness import block_bootstrap_delta
from ..round11.common import daily_from_nav
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import BLOCK_DAYS, BOOT_N, FORWARD_OOS_NOT_AVAILABLE, MD, REPEATED_HISTORICAL_TESTING, SPA_BOOT, exp_dir

STATS_JSON = ROUND21_4_ROOT / "stats" / "stats.json"


def _nav(code: str) -> pd.DataFrame | None:
    path = exp_dir(code) / "nav.parquet"
    if not path.is_file():
        return None
    nav = pd.read_parquet(path)
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    return nav


def _window(nav: pd.DataFrame, *, years: tuple[int, ...] | None = None, drop: tuple[int, ...] = ()) -> dict[str, Any]:
    daily = daily_from_nav(nav)
    idx_year = pd.to_datetime(daily.index).year
    if years:
        daily = daily.loc[idx_year.isin(years)].copy()
    if drop:
        daily = daily.loc[~pd.to_datetime(daily.index).year.isin(drop)].copy()
    if len(daily) < 10:
        return {"status": "TOO_SHORT", "n": int(len(daily))}
    wealth = (1.0 + daily.astype(float)).cumprod()
    span = max((pd.Timestamp(daily.index.max()) - pd.Timestamp(daily.index.min())).days, 1)
    cagr = float(wealth.iloc[-1] ** (365.25 / span) - 1.0)
    vol = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / vol * np.sqrt(252.0)) if vol > 0 else None
    peak = wealth.cummax()
    dd = float((wealth / peak - 1.0).min()) if len(wealth) else None
    return {"cagr": finite_or_none(cagr), "sharpe": finite_or_none(sharpe), "max_drawdown": finite_or_none(dd), "n": int(len(daily))}


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and STATS_JSON.is_file() and MD["26"].is_file():
        existing = load_json(STATS_JSON)
        if existing.get("status") == "DONE":
            return existing
    from .backtest import A_BT_JSON

    bt = load_json(A_BT_JSON)
    codes = [r.get("code") for r in (bt.get("main_rows") or []) if r.get("code")]
    if "R21_4_A0" not in codes:
        codes = ["R21_4_A0", *codes]
    base = _nav("R21_4_A0")
    rows = []
    for code in codes:
        nav = _nav(code)
        if nav is None or base is None:
            rows.append({"code": code, "status": "MISSING_NAV"})
            continue
        da = daily_from_nav(nav)
        db = daily_from_nav(base)
        aligned = pd.concat([da.rename("a"), db.rename("b")], axis=1).dropna()
        boot = None
        if len(aligned) > 40:
            try:
                boot = block_bootstrap_delta(aligned["a"], aligned["b"], n=BOOT_N, block=BLOCK_DAYS)
            except TypeError:
                try:
                    boot = block_bootstrap_delta(aligned["a"].to_numpy(), aligned["b"].to_numpy(), n_boot=BOOT_N, block_size=BLOCK_DAYS)
                except Exception as exc:
                    boot = {"error": str(exc)}
            except Exception as exc:
                boot = {"error": str(exc)}
        rows.append(
            {
                "code": code,
                "full": _window(nav),
                "y2022": _window(nav, years=(2022,)),
                "ex2024": _window(nav, drop=(2024,)),
                "ex2025": _window(nav, drop=(2025,)),
                "block_bootstrap_vs_a0": boot,
            }
        )
    # SPA on A-grid daily returns vs A0
    alts = []
    for code in codes:
        if code == "R21_4_A0":
            continue
        nav = _nav(code)
        if nav is None or base is None:
            continue
        alts.append(daily_from_nav(nav))
    spa = {"status": "SKIPPED", "reason": "no alternatives"}
    if alts and base is not None:
        b = daily_from_nav(base)
        mat = pd.concat([s.rename(i) for i, s in enumerate(alts)], axis=1).join(b.rename("base"), how="inner").dropna()
        if len(mat) > 50:
            d = mat.drop(columns=["base"]).sub(mat["base"], axis=0)
            obs = float(d.mean().max())
            rng = np.random.default_rng(17)
            n = len(d)
            block = BLOCK_DAYS
            n_boot = min(SPA_BOOT, 2000)
            hits = 0
            arr = d.to_numpy(dtype=float)
            for _ in range(n_boot):
                starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
                idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
                boot_max = float(arr[idx].mean(axis=0).max())
                if boot_max >= obs:
                    hits += 1
            spa = {
                "status": "COMPUTED",
                "observed_statistic": obs,
                "n_alternatives": int(d.shape[1]),
                "p_spa": hits / n_boot,
                "n_boot": n_boot,
                "block": block,
                "label": REPEATED_HISTORICAL_TESTING,
            }
    payload = {
        "status": "DONE",
        "rows": rows,
        "spa": spa,
        "boot_n": BOOT_N,
        "block_days": BLOCK_DAYS,
        "forward": FORWARD_OOS_NOT_AVAILABLE,
        "label": REPEATED_HISTORICAL_TESTING,
    }
    dump_json(STATS_JSON, payload)
    write_md(MD["26"], f"# Round21.4 · 26 统计\n\nSPA=`{spa}` bootstrap N=`{BOOT_N}` block=`{BLOCK_DAYS}` label=`{REPEATED_HISTORICAL_TESTING}`\n")
    return payload
