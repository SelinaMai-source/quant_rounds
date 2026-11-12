"""L1 NAV 逐日对账。禁止只引用 summary.json。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_3_ROOT
from ..round11.common import daily_from_nav
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import AUDIT_FAILURE, L1_E0, L1_E1, L1_E2, L1_E3, MD, V5_ANCHOR, V5_NAV_END_EXACT, exp_dir

RECON_CSV = ROUND21_3_ROOT / "L1_NAV_RECONCILIATION.csv"
RECON_JSON = ROUND21_3_ROOT / "nav" / "l1_recon.json"


def _nav(code: str) -> pd.DataFrame | None:
    path = exp_dir(code) / "nav.parquet"
    if not path.is_file():
        return None
    nav = pd.read_parquet(path)
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    return nav.sort_values("trade_date")


def _trades(code: str) -> pd.DataFrame:
    path = exp_dir(code) / "trades.parquet"
    if not path.is_file():
        return pd.DataFrame()
    t = pd.read_parquet(path)
    for c in ("trade_date", "signal_date", "entry_date", "exit_date"):
        if c in t.columns:
            t[c] = pd.to_datetime(t[c], errors="coerce")
    return t


def _metrics(nav: pd.DataFrame) -> dict[str, Any]:
    daily = daily_from_nav(nav)
    if len(daily) < 10:
        return {"status": "TOO_SHORT"}
    wealth = (1.0 + daily.astype(float)).cumprod()
    span = max((pd.Timestamp(daily.index.max()) - pd.Timestamp(daily.index.min())).days, 1)
    cagr = float(wealth.iloc[-1] ** (365.25 / span) - 1.0)
    vol = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / vol * np.sqrt(252.0)) if vol > 0 else None
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    trough_i = int(dd.argmin())
    peak_i = int(wealth.iloc[: trough_i + 1].argmax()) if trough_i >= 0 else 0
    rec = dd.iloc[trough_i:]
    rec_ok = rec[rec >= -1e-12]
    recover = None if rec_ok.empty else str(pd.Timestamp(rec_ok.index[0]).date())
    roll = daily.rolling(252, min_periods=60).apply(lambda x: float((1.0 + x).prod() - 1.0), raw=False)
    worst12 = float(roll.min()) if roll.notna().any() else None
    return {
        "cagr": finite_or_none(cagr),
        "sharpe": finite_or_none(sharpe),
        "vol": finite_or_none(vol),
        "max_drawdown": finite_or_none(float(dd.min())),
        "peak_date": str(pd.Timestamp(wealth.index[peak_i]).date()) if len(wealth) else None,
        "trough_date": str(pd.Timestamp(wealth.index[trough_i]).date()) if len(wealth) else None,
        "recover_date": recover,
        "worst12m": finite_or_none(worst12),
        "n": int(len(daily)),
        "nav_end": finite_or_none(float(nav["total_asset"].iloc[-1])),
    }


def run_nav_recon(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and RECON_CSV.is_file() and RECON_JSON.is_file() and MD["22"].is_file():
        existing = load_json(RECON_JSON)
        if existing.get("status") == "DONE":
            return existing
    codes = [L1_E0, L1_E1, L1_E2, L1_E3]
    navs = {c: _nav(c) for c in codes}
    missing = [c for c, n in navs.items() if n is None]
    if missing:
        payload = {"status": AUDIT_FAILURE, "missing": missing, "reason": "NAV missing; rerun l1_backtest"}
        dump_json(RECON_JSON, payload)
        write_md(MD["22"], f"# Round21.3 · 22 NAV 对账\n\n**{AUDIT_FAILURE}** missing=`{missing}`\n")
        return payload
    base = navs[L1_E0][["trade_date", "total_asset"]].rename(columns={"total_asset": "baseline_nav"})
    cand = navs[L1_E3][["trade_date", "total_asset"]].rename(columns={"total_asset": "candidate_nav"})
    merged = base.merge(cand, on="trade_date", how="outer").sort_values("trade_date")
    merged["nav_delta"] = merged["candidate_nav"] - merged["baseline_nav"]
    first_fork = None
    if merged["nav_delta"].notna().any():
        nz = merged.loc[merged["nav_delta"].abs() > 1e-6]
        if not nz.empty:
            first_fork = pd.Timestamp(nz["trade_date"].iloc[0])
    trades0 = _trades(L1_E0)
    trades3 = _trades(L1_E3)
    rows = []
    after = merged if first_fork is None else merged.loc[merged["trade_date"] >= first_fork]
    for _, rec in after.iterrows():
        d = pd.Timestamp(rec["trade_date"])
        changed = []
        if not trades0.empty and not trades3.empty:
            def _day(t, day):
                col = "trade_date" if "trade_date" in t.columns else ("signal_date" if "signal_date" in t.columns else None)
                if col is None:
                    return set()
                sub = t.loc[pd.to_datetime(t[col]).dt.normalize() == d.normalize()]
                key = "instrument_id" if "instrument_id" in sub.columns else None
                if key is None:
                    return set()
                return set(sub[key].astype(str))

            a, b = _day(trades0, d), _day(trades3, d)
            changed = sorted(a.symmetric_difference(b))
        pos = None
        if "positions" in navs[L1_E3].columns:
            hit = navs[L1_E3].loc[navs[L1_E3]["trade_date"] == d]
            if not hit.empty:
                pos = str(hit["positions"].iloc[0])
        rows.append(
            {
                "date": str(d.date()),
                "baseline_nav": finite_or_none(rec.get("baseline_nav")),
                "candidate_nav": finite_or_none(rec.get("candidate_nav")),
                "nav_delta": finite_or_none(rec.get("nav_delta")),
                "positions": pos,
                "changed_trades": "|".join(changed) if changed else "",
                "pnl_explanation": (
                    "first_fork" if first_fork is not None and d == first_fork else
                    ("trade_diff" if changed else "nav_drift_or_same_selection")
                ),
            }
        )
    RECON_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RECON_CSV, index=False)
    metrics = {c: _metrics(navs[c]) for c in codes}
    e0m, e3m = metrics[L1_E0], metrics[L1_E3]
    dd_note = (
        f"NAV_end Δ={finite_or_none((e3m.get('nav_end') or 0) - (e0m.get('nav_end') or 0))} "
        f"MaxDD {e0m.get('max_drawdown')} → {e3m.get('max_drawdown')}. "
        "若 NAV 几乎不变而 MaxDD 改善，通常是同一期末财富下路径峰值更高或回撤区间交易替换。"
    )
    # 独立复算 vs v5
    recon_ok = True
    reasons = []
    if e0m.get("nav_end") is not None and abs(float(e0m["nav_end"]) - float(V5_NAV_END_EXACT)) > 1.0:
        recon_ok = False
        reasons.append("E0 nav_end != v5")
    payload = {
        "status": "DONE" if recon_ok else AUDIT_FAILURE,
        "first_fork": None if first_fork is None else str(first_fork.date()),
        "n_rows": len(rows),
        "metrics": metrics,
        "dd_note": dd_note,
        "csv": str(RECON_CSV),
        "reasons": reasons,
        "v5_anchor": dict(V5_ANCHOR),
    }
    dump_json(RECON_JSON, payload)
    write_md(
        MD["22"],
        "\n".join(
            [
                "# Round21.3 · 22 L1 NAV 逐日对账",
                "",
                f"status=`{payload['status']}` first_fork=`{payload['first_fork']}` rows=`{len(rows)}`",
                f"独立复算 E0 CAGR=`{e0m.get('cagr')}` Sharpe=`{e0m.get('sharpe')}` Vol=`{e0m.get('vol')}` MaxDD=`{e0m.get('max_drawdown')}` 峰=`{e0m.get('peak_date')}` 谷=`{e0m.get('trough_date')}` 恢复=`{e0m.get('recover_date')}` Worst12M=`{e0m.get('worst12m')}`",
                f"独立复算 E3 CAGR=`{e3m.get('cagr')}` Sharpe=`{e3m.get('sharpe')}` Vol=`{e3m.get('vol')}` MaxDD=`{e3m.get('max_drawdown')}` 峰=`{e3m.get('peak_date')}` 谷=`{e3m.get('trough_date')}` 恢复=`{e3m.get('recover_date')}` Worst12M=`{e3m.get('worst12m')}`",
                dd_note,
                f"CSV=`{RECON_CSV}` 禁止只引用 summary.json。",
                "",
            ]
        ),
    )
    return payload
