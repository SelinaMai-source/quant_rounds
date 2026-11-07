"""重放 baseline_v4 = R9_B03。FAIL 只停正式晋级，仍写 00_shared_baseline_parity.md。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND19_ROOT, assert_project_result_path
from ..round11.parity import compare_ledgers
from ..round16.common import cagr_of
from ..round16_5.engine import metrics_close
from ..round17.backtest import cached_overlay_v4, run_canonical
from .common import daily_from_nav, dump_json, load_json, observed_from_report, write_md
from .contract import (
    A_C0,
    CAGR_OF_252_PCT_DISPLAY,
    CALENDAR_CAGR_PCT_DISPLAY,
    FORMAL_PROMOTION_BLOCKED,
    N_TRADING_OBSERVATIONS_ANCHOR,
    PARITY_ABS_NAV,
    PARITY_MD,
    PARITY_METRIC_ABS,
    R16_5_V00_DIR,
    R16_5_V00_OVERLAY,
    R18_A00_DIR,
    V4_ANCHOR,
    overlayed_path,
    parity_dir,
)

PARITY_JSON = ROUND19_ROOT / "parity" / "parity.json"


def load_parity_status() -> dict[str, Any]:
    return load_json(PARITY_JSON)


def _with_rank(frame: pd.DataFrame) -> pd.DataFrame:
    t = frame.copy()
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["instrument_id"] = t["instrument_id"].astype(str)
    t["_rank"] = t.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    return t


def _identity_check(local: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, Any]:
    lt = _with_rank(local)
    at = _with_rank(anchor)
    keys = ["signal_date", "instrument_id"]
    need = keys + ["_rank", "allocation_multiplier"]
    extra = ["target_name_budget"] if "target_name_budget" in lt.columns and "target_name_budget" in at.columns else []
    l2 = lt.loc[lt["_rank"].le(2), need + extra].copy()
    a2 = at.loc[at["_rank"].le(2), need + extra].copy()
    merged = l2.merge(a2, on=keys + ["_rank"], how="outer", suffixes=("_loc", "_anc"), indicator=True)
    both = merged["_merge"].eq("both")
    n_both = int(both.sum())
    n_only = int((~both).sum())
    alloc_ok = True
    budget_ok = True
    if n_both:
        alloc_ok = bool(
            (
                pd.to_numeric(merged.loc[both, "allocation_multiplier_loc"], errors="coerce")
                - pd.to_numeric(merged.loc[both, "allocation_multiplier_anc"], errors="coerce")
            )
            .abs()
            .fillna(1)
            .le(1e-12)
            .all()
        )
        if extra:
            budget_ok = bool(
                (
                    pd.to_numeric(merged.loc[both, "target_name_budget_loc"], errors="coerce")
                    - pd.to_numeric(merged.loc[both, "target_name_budget_anc"], errors="coerce")
                )
                .abs()
                .fillna(1)
                .le(1e-4)
                .all()
            )
    n_top1 = int((l2["_rank"] == 1).sum())
    n_top2 = int((l2["_rank"] == 2).sum())
    return {
        "n_local_top2": int(len(l2)),
        "n_anchor_top2": int(len(a2)),
        "n_matched": n_both,
        "n_identity_mismatch": n_only,
        "n_top1": n_top1,
        "n_top2": n_top2,
        "alloc_ok": alloc_ok,
        "budget_ok": budget_ok,
        "passed": bool(n_only == 0 and alloc_ok and budget_ok),
    }


def _overlay_identity(local: pd.DataFrame, anchor_path) -> dict[str, Any]:
    from pathlib import Path

    path = Path(anchor_path)
    if not path.is_file():
        return {"passed": False, "status": "MISSING_ARTIFACT", "path": str(path)}
    return _identity_check(local, pd.read_parquet(path))


def _ledger(local_trades, local_nav, anchor_dir, n_trades: int) -> dict[str, Any]:
    from pathlib import Path

    anchor_dir = Path(anchor_dir)
    if not (anchor_dir / "nav.parquet").is_file() or not (anchor_dir / "trades.parquet").is_file():
        return {"passed": False, "status": "MISSING_ARTIFACT", "path": str(anchor_dir)}
    rec = compare_ledgers(
        local_trades,
        pd.read_parquet(anchor_dir / "trades.parquet"),
        local_nav,
        pd.read_parquet(anchor_dir / "nav.parquet"),
        target_n_trades=n_trades,
    )
    rec["path"] = str(anchor_dir)
    rec["status"] = "PASS" if rec.get("passed") else "FAIL"
    return rec


def _cagr_views(nav: pd.DataFrame, report: dict[str, Any]) -> dict[str, Any]:
    daily = daily_from_nav(nav)
    metrics = report.get("metrics") or {}
    calendar = metrics.get("annualized_return")
    trading = cagr_of(daily)
    n_nav = int(len(nav))
    n_obs = int(len(daily))
    n_metric = metrics.get("metric_n_daily_returns")
    return {
        "calendar_365_25": {
            "cagr": calendar,
            "formula": "(final_asset / initial_cash) ** (1 / years) - 1; years=(last-first).days/365.25",
            "display_pct": CALENDAR_CAGR_PCT_DISPLAY,
            "metric_years": metrics.get("metric_years"),
            "metric_nav_start": metrics.get("metric_nav_start"),
            "metric_nav_end": metrics.get("metric_nav_end"),
        },
        "trading_252_cagr_of": {
            "cagr": trading,
            "formula": "(prod(1+r)) ** (252 / n_days) - 1",
            "display_pct": CAGR_OF_252_PCT_DISPLAY,
            "n_days": n_obs,
        },
        "n_nav_rows": n_nav,
        "n_trading_observations": n_obs,
        "metric_n_daily_returns": n_metric,
        "note": "2075 是交易日收益观测（日收益条数），不是成交笔数。1149 才是 completed_trades。",
    }


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and PARITY_JSON.is_file():
        existing = load_json(PARITY_JSON)
        if existing.get("passed") and PARITY_MD.is_file():
            return existing
    ov4 = cached_overlay_v4()
    ov_path = overlayed_path(A_C0)
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(ov_path)
    ov4.to_parquet(ov_path, index=False)
    dest = parity_dir(A_C0)
    ran = run_canonical(
        dest,
        ov4,
        extra={
            "code": A_C0,
            "round": 19,
            "role": "exact_v4_replay",
            "fee_schedule": "HISTORICAL_FEE_CORRECTED",
            "parent": "R9_B03",
        },
        resume=resume,
        write=True,
    )
    report = ran.get("report") or {}
    if not report and (dest / "report.json").is_file():
        report = load_json(dest / "report.json")
    trades = ran.get("trades")
    if trades is None and (dest / "trades.parquet").is_file():
        trades = pd.read_parquet(dest / "trades.parquet")
    nav = ran.get("nav")
    if nav is None and (dest / "nav.parquet").is_file():
        nav = pd.read_parquet(dest / "nav.parquet")
    obs = ran.get("observed") or observed_from_report(report)
    cagr_views = _cagr_views(nav, report) if nav is not None else {}
    n_obs = int(cagr_views.get("n_trading_observations") or 0)
    n_trades = int(obs.get("completed_trades") or -1)
    trade_vs_days_ok = bool(n_trades == int(V4_ANCHOR["completed_trades"]) and n_obs == N_TRADING_OBSERVATIONS_ANCHOR)
    metric_ok = metrics_close(obs, V4_ANCHOR, abs_tol=PARITY_METRIC_ABS)
    ledger_r16 = _ledger(trades, nav, R16_5_V00_DIR, int(V4_ANCHOR["completed_trades"]))
    ledger_r18 = _ledger(trades, nav, R18_A00_DIR, int(V4_ANCHOR["completed_trades"]))
    ident_trades = None
    if (R16_5_V00_DIR / "trades.parquet").is_file() and trades is not None:
        ident_trades = _identity_check(trades, pd.read_parquet(R16_5_V00_DIR / "trades.parquet"))
    ident_overlay = _overlay_identity(ov4, R16_5_V00_OVERLAY)
    identity_ok = bool((ident_trades or {}).get("passed") and ident_overlay.get("passed"))
    passed = bool(metric_ok and ledger_r16.get("passed") and trade_vs_days_ok and identity_ok)
    status = "PASS" if passed else "FAIL"
    payload = {
        "status": status,
        "passed": passed,
        "fail_code": None if passed else FORMAL_PROMOTION_BLOCKED,
        "formal_promotion_blocked": not passed,
        "code": A_C0,
        "engine": "cached_overlay_v4 + run_canonical (HISTORICAL_FEE_CORRECTED + B03)",
        "observed": obs,
        "anchor": dict(V4_ANCHOR),
        "metric_ok": metric_ok,
        "trade_vs_days_ok": trade_vs_days_ok,
        "n_completed_trades": n_trades,
        "n_trading_observations": n_obs,
        "cagr_views": cagr_views,
        "abs_tol": PARITY_METRIC_ABS,
        "nav_abs_tol": PARITY_ABS_NAV,
        "ledger_vs_r16_5_v00": {k: ledger_r16.get(k) for k in ("passed", "status", "nav_max_abs", "trade_rows_local", "trade_rows_anchor", "identity_mismatches", "field_mismatch_rows", "path")},
        "ledger_vs_r18_a00": {k: ledger_r18.get(k) for k in ("passed", "status", "nav_max_abs", "trade_rows_local", "trade_rows_anchor", "identity_mismatches", "field_mismatch_rows", "path")},
        "identity_top1_top2_trades": ident_trades,
        "identity_overlay_allocation_multiplier": ident_overlay,
        "out_dir": str(dest),
        "overlay_path": str(ov_path),
        "note": "FAIL 只停止正式晋级评价；取证可继续。仍写本 artifact。禁止把 2075 写成 trade_count。",
    }
    dump_json(PARITY_JSON, payload)
    if nav is not None:
        summary = {
            **obs,
            "n_trading_observations": n_obs,
            "n_nav_rows": int(cagr_views.get("n_nav_rows") or 0),
            "cagr_calendar_365_25": (cagr_views.get("calendar_365_25") or {}).get("cagr"),
            "cagr_of_252": (cagr_views.get("trading_252_cagr_of") or {}).get("cagr"),
            "code": A_C0,
        }
        dump_json(dest / "summary.json", summary)
    cal = (cagr_views.get("calendar_365_25") or {}).get("cagr")
    t252 = (cagr_views.get("trading_252_cagr_of") or {}).get("cagr")
    lines = [
        "# Round19 · 00 shared baseline_v4 parity",
        "",
        f"结果：**{status}**。engine=`cached_overlay_v4` + `run_canonical`。",
        "",
        "## 精确锚点（主评价 = 日历年 annualized_return）",
        "",
        f"- anchor CAGR=`{V4_ANCHOR['cagr']}` Sharpe=`{V4_ANCHOR['sharpe']}` MaxDD=`{V4_ANCHOR['max_drawdown']}` completed_trades=`{V4_ANCHOR['completed_trades']}`",
        f"- observed CAGR=`{obs.get('cagr')}` Sharpe=`{obs.get('sharpe')}` MaxDD=`{obs.get('max_drawdown')}` completed_trades=`{obs.get('completed_trades')}`",
        f"- metric_ok=`{metric_ok}` abs_tol=`{PARITY_METRIC_ABS}`",
        "",
        "## 2075 交易日观测 vs 1149 成交",
        "",
        f"- n_nav_rows=`{cagr_views.get('n_nav_rows')}`（含首日，无日收益）",
        f"- n_trading_observations / metric_n_daily_returns=`{n_obs}` / `{cagr_views.get('metric_n_daily_returns')}`",
        f"- completed_trades=`{n_trades}`",
        f"- trade_vs_days_ok=`{trade_vs_days_ok}`（要求 trades=1149 且观测=2075）",
        "",
        "Round16.5 主报告第 5 节把 `metric_n_daily_returns=2075` 误写成「笔数」。`00_parity.md` 当时是 1149，以账本为准。",
        "",
        "## CAGR 口径（禁止混报）",
        "",
        f"- **主口径** 日历年 365.25：`annualized_return={cal}` → {CALENDAR_CAGR_PCT_DISPLAY}",
        f"- **对照** 252 交易日 `cagr_of`：`{t252}` → {CAGR_OF_252_PCT_DISPLAY}",
        "- Round16.5 `11_same_window.md` 的 55.07% 来自 `cagr_of`，不得替换 52.54%。",
        "",
        "## 账本 / Top1·Top2 / allocation_multiplier",
        "",
        f"- vs R16.5 V00 ledger=`{ledger_r16.get('passed')}` nav_max_abs=`{ledger_r16.get('nav_max_abs')}`",
        f"- vs R18 A00 ledger=`{ledger_r18.get('passed')}` nav_max_abs=`{ledger_r18.get('nav_max_abs')}`",
        f"- trades Top1/Top2 identity=`{(ident_trades or {}).get('passed')}` alloc_ok=`{(ident_trades or {}).get('alloc_ok')}`",
        f"- overlay vs R16.5 V00 identity=`{ident_overlay.get('passed')}` alloc_ok=`{ident_overlay.get('alloc_ok')}`",
        "",
        f"FAIL → `{FORMAL_PROMOTION_BLOCKED}`（停止正式晋级评价，取证可继续）。本文件在 FAIL 时仍落盘。",
        "",
        f"out=`{dest}`",
        "",
    ]
    write_md(PARITY_MD, "\n".join(lines))
    return payload
