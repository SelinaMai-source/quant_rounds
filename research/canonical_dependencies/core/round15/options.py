"""O01–O04：仅跑 freeze 时数据足够的候选。对照 matched。禁止 O05+。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND15_ROOT
from ..round8.overlays import MIN_KEEP
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import (
    O01,
    O02,
    O02_HALVE,
    O03,
    O04,
    OPTION_IDS,
    PERM_N,
    R15_00,
    control_dir,
    experiment_dir,
    overlayed_path,
    parity_dir,
)
from .factors import load_factor_daily, load_factors
from .freeze import load_skips, require_frozen
from .runner import (
    apply_budget_scale,
    apply_top2_veto,
    build_version_overlayed,
    hist_hooks,
    load_market,
    run_r15_backtest,
)

OPT_JSON = ROUND15_ROOT / "options" / "options.json"
OPT_MD = ROUND15_ROOT / "06_options.md"
CTRL_JSON = ROUND15_ROOT / "controls" / "option_controls.json"


def _summarize(ran: dict[str, Any]) -> dict[str, Any]:
    report = ran.get("report") or {}
    obs = observed_from_report(report)
    obs.update({"code": ran.get("code"), "status": ran.get("status"), "out_dir": ran.get("out_dir")})
    return obs


def _save_ov(code: str, ov: pd.DataFrame) -> None:
    path = overlayed_path(code)
    path.parent.mkdir(parents=True, exist_ok=True)
    ov.to_parquet(path, index=False)


def _signal_from_factor(col: str, q: float) -> pd.DatetimeIndex:
    daily = load_factor_daily()
    if daily.empty or col not in daily.columns:
        return pd.DatetimeIndex([])
    work = daily.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    s = pd.to_numeric(work.set_index("signal_date")[col], errors="coerce").sort_index()
    past_q = s.shift(1).rolling(252, min_periods=126).quantile(q)
    hit = s.shift(1) > past_q
    return pd.DatetimeIndex(hit.index[hit.fillna(False)])


def _veto_dates() -> pd.DatetimeIndex:
    return _signal_from_factor("PUT_SKEW25", 0.90)


def _iv_delta_dates() -> pd.DatetimeIndex:
    daily = load_factor_daily()
    if daily.empty or "ATM_IV30" not in daily.columns:
        return pd.DatetimeIndex([])
    work = daily.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    iv = pd.to_numeric(work.set_index("signal_date")["ATM_IV30"], errors="coerce").sort_index()
    d5 = iv.diff(5)
    past_q = d5.shift(1).rolling(252, min_periods=126).quantile(0.90)
    hit = d5.shift(1) > past_q
    return pd.DatetimeIndex(hit.index[hit.fillna(False)])


def _convexity_dates() -> pd.DatetimeIndex:
    skew = set(_veto_dates())
    conv = set(_signal_from_factor("PUT_WING_CONVEXITY", 0.90))
    both = sorted(skew & conv)
    return pd.DatetimeIndex(both)


def _run(code: str, ov: pd.DataFrame, dest, *, resume: bool, loaded) -> dict[str, Any]:
    _save_ov(code, ov)
    ran = run_r15_backtest(
        dest,
        overlayed=ov,
        validation_hooks=hist_hooks(),
        extra_report={"fee_schedule": "HISTORICAL_FEE_CORRECTED", "candidate": code, "family": "option"},
        resume=resume,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    out = _summarize(ran)
    out["nav_path"] = str(dest / "nav.parquet")
    return out


def _timing_perm_p(v4_daily: pd.Series, treat_dates: pd.DatetimeIndex, cand_daily: pd.Series) -> dict[str, Any]:
    """年分层随机择时：打乱处理日，用候选-对照日收益差的 Sharpe。不是逐日 exposure-match。"""
    if v4_daily is None or cand_daily is None or treat_dates is None or len(treat_dates) == 0:
        return {"n": 0, "p_cand_gt_ctrl": None}
    aligned = pd.concat({"v4": v4_daily, "c": cand_daily}, axis=1).dropna()
    if aligned.empty:
        return {"n": 0, "p_cand_gt_ctrl": None}
    treat = pd.DatetimeIndex(pd.to_datetime(treat_dates).normalize())
    years = aligned.index.year
    rng = np.random.default_rng(20261013)
    actual = float(aligned["c"].mean() / (aligned["c"].std(ddof=1) + 1e-12) * np.sqrt(252))
    beat = 0
    n_ok = 0
    n_treat = int((aligned.index.isin(treat)).sum())
    for _ in range(PERM_N):
        picked = []
        for y, g in aligned.groupby(years):
            n_y = int(g.index.isin(treat).sum())
            if n_y <= 0:
                continue
            chosen = rng.choice(g.index.to_numpy(), size=min(n_y, len(g)), replace=False)
            picked.extend(list(chosen))
        if not picked:
            continue
        # 对照：在随机日施加与候选相同的日收益差幅度（候选相对 v4）
        dummy = aligned["v4"].copy()
        dummy.loc[dummy.index.isin(pd.DatetimeIndex(picked))] = aligned.loc[
            aligned.index.isin(pd.DatetimeIndex(picked)), "c"
        ]
        vol = float(dummy.std(ddof=1))
        sh = float(dummy.mean() / vol * np.sqrt(252)) if vol > 0 else np.nan
        if np.isfinite(sh):
            n_ok += 1
            if actual > sh:
                beat += 1
    return {
        "n": PERM_N,
        "n_ok": n_ok,
        "n_treat_days": n_treat,
        "p_cand_gt_ctrl": (beat / n_ok) if n_ok else None,
        "kind": "year_stratified_random_timing",
        "not_daily_exposure_match": True,
    }


def run_options(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and OPT_JSON.is_file():
        existing = load_json(OPT_JSON)
        if existing.get("status") in {"DONE", "SKIP_ALL_OPTIONS"}:
            return existing
    skips = load_skips()
    fac = load_factors()
    ov4, _, _ = build_version_overlayed("v4")
    loaded = load_market()
    results: dict[str, Any] = {}
    ran_any = False

    def skip_row(code: str, reason: str) -> dict[str, Any]:
        return {"code": code, "status": reason, "skip": True, "eligible": False}

    if skips.get("SKIP_O01", True) or fac.get("skip"):
        results[O01] = skip_row(O01, "SKIP_O01" if skips.get("SKIP_O01", True) else "SKIP_IV_SURFACE")
    else:
        dates = _veto_dates()
        if len(dates) == 0:
            results[O01] = skip_row(O01, "SKIP_O01")
        else:
            ov = apply_top2_veto(ov4, dates)
            results[O01] = _run(O01, ov, experiment_dir(O01), resume=resume, loaded=loaded)
            results[O01]["eligible"] = True
            results[O01]["n_veto_days"] = int(len(dates))
            ran_any = True

    if skips.get("SKIP_O02", True) or fac.get("skip"):
        results[O02] = skip_row(O02, "SKIP_O02" if skips.get("SKIP_O02", True) else "SKIP_IV_SURFACE")
    else:
        dates = _iv_delta_dates()
        if len(dates) == 0:
            results[O02] = skip_row(O02, "SKIP_O02")
        else:
            ov = apply_budget_scale(ov4, dates, O02_HALVE)
            results[O02] = _run(O02, ov, experiment_dir(O02), resume=resume, loaded=loaded)
            results[O02]["eligible"] = True
            results[O02]["n_scale_days"] = int(len(dates))
            ran_any = True

    if skips.get("SKIP_O03", True) or fac.get("skip"):
        results[O03] = skip_row(O03, "SKIP_O03")
    else:
        dates = _convexity_dates()
        if len(dates) == 0:
            results[O03] = skip_row(O03, "SKIP_O03")
        else:
            ov = apply_top2_veto(ov4, dates)
            results[O03] = _run(O03, ov, experiment_dir(O03), resume=resume, loaded=loaded)
            results[O03]["eligible"] = True
            results[O03]["n_veto_days"] = int(len(dates))
            ran_any = True

    if skips.get("SKIP_O04", True) or fac.get("skip"):
        results[O04] = skip_row(O04, "SKIP_O04")
    else:
        results[O04] = skip_row(O04, "SKIP_O04")

    v4_obs = {}
    p = parity_dir(R15_00) / "report.json"
    if p.is_file():
        import json

        v4_obs = observed_from_report(json.loads(p.read_text(encoding="utf-8")))
        v4_obs["code"] = R15_00
        v4_obs["nav_path"] = str(parity_dir(R15_00) / "nav.parquet")

    payload = {
        "status": "DONE" if ran_any else "SKIP_ALL_OPTIONS",
        "results": results,
        "v4": v4_obs,
        "extra_candidates": False,
        "no_O05": True,
        "no_option_top1_combo": True,
        "ran_any": ran_any,
    }
    dump_json(OPT_JSON, payload)
    lines = [
        "# Round 15 · 期权候选 O01–O04",
        "",
        "仅跑 freeze 时数据足够的组。无 O05+。不看完收益再组合 Option+Top1。",
        "",
        "| ID | status | CAGR | Sharpe | MaxDD | 笔数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for code in OPTION_IDS:
        row = results.get(code) or {}
        lines.append(
            f"| `{code}` | {row.get('status')} | {pct(row.get('cagr'))} | {row.get('sharpe')} | "
            f"{pct(row.get('max_drawdown'))} | {row.get('completed_trades')} |"
        )
    write_md(OPT_MD, "\n".join(lines) + "\n")
    return payload


def load_options() -> dict[str, Any]:
    return load_json(OPT_JSON)


__all__ = ["MIN_KEEP", "load_options", "run_options"]
