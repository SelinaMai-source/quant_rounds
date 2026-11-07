#!/usr/bin/env python3
"""重跑 baseline_v4=R9_B03 与 baseline_v5=R16_O01。canonical 双袖套 + HISTORICAL_FEE_CORRECTED。

禁止写 Stock。禁止覆盖 result/round16* 与 result/round18。禁止改 CURRENT 锁。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEST_ROOT = ROOT / "result" / "replay_v4_v5"
V4_DIR = DEST_ROOT / "v4_R9_B03"
V5_DIR = DEST_ROOT / "v5_R16_O01"
ABS_TOL = 1e-8
NAV_DELTA_ANCHOR = 691458.0688966699
NAV_END_V4_ANCHOR = 11118167.503299337
NAV_END_V5_ANCHOR = 11809625.572196007


def _log(msg: str) -> None:
    print(msg, flush=True)


def _dump(path: Path, payload: Any) -> Path:
    from core.paths import assert_project_result_path

    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _finite(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def _close(a: Any, b: Any, *, abs_tol: float = ABS_TOL) -> bool:
    fa, fb = _finite(a), _finite(b)
    if fa is None or fb is None:
        return False
    return abs(fa - fb) < abs_tol


def _pack(ran: dict[str, Any], dest: Path) -> dict[str, Any]:
    from core.round11.common import daily_from_nav, observed_from_report
    from core.round16.common import cagr_of
    from core.round16_5.common import nav_end

    report = ran.get("report") or {}
    if not report and (dest / "report.json").is_file():
        report = json.loads((dest / "report.json").read_text(encoding="utf-8"))
    nav = ran.get("nav")
    if nav is None and (dest / "nav.parquet").is_file():
        import pandas as pd

        nav = pd.read_parquet(dest / "nav.parquet")
    trades = ran.get("trades")
    if trades is None and (dest / "trades.parquet").is_file():
        import pandas as pd

        trades = pd.read_parquet(dest / "trades.parquet")
    obs = ran.get("observed") or observed_from_report(report)
    metrics = report.get("metrics") or {}
    daily = daily_from_nav(nav) if nav is not None else None
    n_obs = int(len(daily)) if daily is not None else None
    yearly = obs.get("yearly_returns") or metrics.get("yearly_returns") or {}
    worst_year = None
    worst_ret = None
    for year, ret in yearly.items():
        fv = _finite(ret)
        if fv is None:
            continue
        if worst_ret is None or fv < worst_ret:
            worst_ret = fv
            worst_year = str(year)
    end_nav = nav_end(nav) if nav is not None else None
    calendar = _finite(metrics.get("annualized_return"))
    if calendar is None:
        calendar = _finite(obs.get("cagr"))
    trading_cagr = _finite(cagr_of(daily)) if daily is not None and len(daily) else None
    return {
        "cagr": calendar,
        "cagr_calendar_365_25": calendar,
        "cagr_of_252": trading_cagr,
        "sharpe": _finite(obs.get("sharpe") or metrics.get("sharpe_zero_rate")),
        "max_drawdown": _finite(obs.get("max_drawdown") or metrics.get("max_drawdown")),
        "completed_trades": obs.get("completed_trades") if obs.get("completed_trades") is not None else metrics.get("completed_trades"),
        "n_trading_observations": n_obs,
        "n_nav_rows": int(len(nav)) if nav is not None else None,
        "metric_n_daily_returns": metrics.get("metric_n_daily_returns"),
        "total_return": _finite(obs.get("total_return") or metrics.get("total_return")),
        "loyo_median": _finite(obs.get("loyo_median") or metrics.get("loyo_median")),
        "loyo_worst": _finite(obs.get("loyo_worst") or metrics.get("loyo_worst")),
        "yearly_returns": {str(k): _finite(v) for k, v in yearly.items()},
        "worst_year": worst_year,
        "worst_year_return": worst_ret,
        "nav_end": end_nav,
        "metric_nav_start": metrics.get("metric_nav_start"),
        "metric_nav_end": metrics.get("metric_nav_end"),
        "metric_years": metrics.get("metric_years"),
        "fee_schedule": report.get("fee_schedule") or "HISTORICAL_FEE_CORRECTED",
        "out_dir": str(dest),
        "nav_path": str(dest / "nav.parquet") if (dest / "nav.parquet").is_file() else None,
        "trades_path": str(dest / "trades.parquet") if (dest / "trades.parquet").is_file() else None,
        "report_path": str(dest / "report.json") if (dest / "report.json").is_file() else None,
    }


def _match_status(obs: dict[str, Any], anchor: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    checks = {
        "cagr": _close(obs.get("cagr"), anchor.get("cagr")),
        "sharpe": _close(obs.get("sharpe"), anchor.get("sharpe")),
        "max_drawdown": _close(obs.get("max_drawdown"), anchor.get("max_drawdown")),
        "completed_trades": int(obs.get("completed_trades") or -1) == int(anchor.get("completed_trades") or -2),
    }
    if extra:
        for key, (got, want) in extra.items():
            if isinstance(got, int) and isinstance(want, int):
                checks[key] = got == want
            else:
                checks[key] = _close(got, want)
    ok = all(checks.values())
    return {
        "status": "MATCH" if ok else "PARITY_DRIFT",
        "checks": checks,
        "anchor": dict(anchor),
        "observed": {
            "cagr": obs.get("cagr"),
            "sharpe": obs.get("sharpe"),
            "max_drawdown": obs.get("max_drawdown"),
            "completed_trades": obs.get("completed_trades"),
            "n_trading_observations": obs.get("n_trading_observations"),
            "nav_end": obs.get("nav_end"),
            "total_return": obs.get("total_return"),
        },
    }


def main() -> int:
    from core.paths import assert_project_result_path
    from core.round15.runner import apply_top2_veto
    from core.round16_5.contract import O01_ANCHOR, V4_ANCHOR
    from core.round16_5.engine import o01_signal_dates
    from core.round17.backtest import cached_overlay_v4, run_canonical
    from core.round19.contract import N_TRADING_OBSERVATIONS_ANCHOR, O01_FORMULA

    assert ROOT.name == "一进二策略", ROOT
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(DEST_ROOT / "summary.json")
    _log(f"ROOT={ROOT}")
    _log(f"DEST={DEST_ROOT}")
    _log("engine=cached_overlay_v4 + run_canonical (HISTORICAL_FEE_CORRECTED); v5=apply_top2_veto frozen O01")

    _log("STEP overlay v4")
    ov4 = cached_overlay_v4()
    ov4_path = DEST_ROOT / "overlayed_v4_R9_B03.parquet"
    ov4.to_parquet(ov4_path, index=False)
    _log(f"overlay_v4 rows={len(ov4)} → {ov4_path}")

    _log("STEP backtest v4 R9_B03 resume=False")
    ran_v4 = run_canonical(
        V4_DIR,
        ov4,
        extra={
            "code": "REPLAY_V4_R9_B03",
            "role": "replay_baseline_v4",
            "fee_schedule": "HISTORICAL_FEE_CORRECTED",
            "parent": "R9_B03",
            "engine": "cached_overlay_v4 + run_canonical",
        },
        resume=False,
        write=True,
    )
    v4 = _pack(ran_v4, V4_DIR)
    v4["code"] = "baseline_v4"
    v4["alias"] = "R9_B03"
    _dump(V4_DIR / "replay_summary.json", v4)
    _log(
        "v4 "
        f"CAGR={v4['cagr']} Sharpe={v4['sharpe']} MaxDD={v4['max_drawdown']} "
        f"trades={v4['completed_trades']} n_obs={v4['n_trading_observations']} NAV_end={v4['nav_end']}"
    )

    _log("STEP O01 signal dates (frozen PUT_SKEW25, no threshold search)")
    dates = o01_signal_dates()
    _log(f"o01_signal_dates n={len(dates)} formula={O01_FORMULA}")
    ov5 = apply_top2_veto(ov4, dates)
    ov5_path = DEST_ROOT / "overlayed_v5_R16_O01.parquet"
    ov5.to_parquet(ov5_path, index=False)

    _log("STEP backtest v5 R16_O01 resume=False")
    ran_v5 = run_canonical(
        V5_DIR,
        ov5,
        extra={
            "code": "REPLAY_V5_R16_O01",
            "role": "replay_baseline_v5",
            "fee_schedule": "HISTORICAL_FEE_CORRECTED",
            "parent": "R9_B03",
            "formula": O01_FORMULA,
            "engine": "cached_overlay_v4 + apply_top2_veto + run_canonical",
        },
        resume=False,
        write=True,
    )
    v5 = _pack(ran_v5, V5_DIR)
    v5["code"] = "baseline_v5"
    v5["alias"] = "R16_O01"
    v5["n_signal_dates"] = int(len(dates))
    v5["formula"] = O01_FORMULA
    _dump(V5_DIR / "replay_summary.json", v5)
    _log(
        "v5 "
        f"CAGR={v5['cagr']} Sharpe={v5['sharpe']} MaxDD={v5['max_drawdown']} "
        f"trades={v5['completed_trades']} n_obs={v5['n_trading_observations']} NAV_end={v5['nav_end']}"
    )

    nav_delta = None
    if v4.get("nav_end") is not None and v5.get("nav_end") is not None:
        nav_delta = float(v5["nav_end"]) - float(v4["nav_end"])

    v4_match = _match_status(
        v4,
        V4_ANCHOR,
        extra={
            "n_trading_observations": (int(v4.get("n_trading_observations") or -1), int(N_TRADING_OBSERVATIONS_ANCHOR)),
            "nav_end": (v4.get("nav_end"), NAV_END_V4_ANCHOR),
        },
    )
    v5_match = _match_status(
        v5,
        O01_ANCHOR,
        extra={"nav_end": (v5.get("nav_end"), NAV_END_V5_ANCHOR)},
    )
    delta_match = _close(nav_delta, NAV_DELTA_ANCHOR)
    overall = (
        "MATCH"
        if v4_match["status"] == "MATCH" and v5_match["status"] == "MATCH" and delta_match
        else "PARITY_DRIFT"
    )
    payload = {
        "status": overall,
        "fee_schedule": "HISTORICAL_FEE_CORRECTED",
        "engine_v4": "cached_overlay_v4 + run_canonical",
        "engine_v5": "cached_overlay_v4 + apply_top2_veto + run_canonical",
        "formula_v5": O01_FORMULA,
        "current_lock_untouched": "CURRENT_ALPHA_BASELINE = baseline_v5 = R16_O01 USER_LOCK",
        "cagr_primary": "calendar_year_annualized_return_365_25",
        "cagr_not_used_as_primary": "252_day_cagr_of",
        "completed_trades_vs_n_obs": "completed_trades is fills; n_trading_observations is daily NAV return rows",
        "v4": v4,
        "v5": v5,
        "nav_end_v4": v4.get("nav_end"),
        "nav_end_v5": v5.get("nav_end"),
        "nav_delta": nav_delta,
        "nav_delta_anchor": NAV_DELTA_ANCHOR,
        "v4_vs_lock": v4_match,
        "v5_vs_lock": v5_match,
        "nav_delta_vs_lock": {"status": "MATCH" if delta_match else "PARITY_DRIFT", "observed": nav_delta, "anchor": NAV_DELTA_ANCHOR},
        "overlay_v4": str(ov4_path),
        "overlay_v5": str(ov5_path),
        "note": "Fresh replay. Does not overwrite result/round16* or result/round18. Does not unlock CURRENT.",
    }
    _dump(DEST_ROOT / "summary.json", payload)
    lines = [
        "# baseline_v4 / baseline_v5 重跑",
        "",
        f"总评：`{overall}`。费用=`HISTORICAL_FEE_CORRECTED`。CURRENT 锁未改。",
        "",
        "## v4 = R9_B03",
        "",
        f"- engine=`cached_overlay_v4 + run_canonical`",
        f"- CAGR(日历年)=`{v4.get('cagr')}`（对照 252 日 cagr_of=`{v4.get('cagr_of_252')}`，非正式口径）",
        f"- Sharpe=`{v4.get('sharpe')}` MaxDD=`{v4.get('max_drawdown')}`",
        f"- completed_trades=`{v4.get('completed_trades')}` n_trading_observations=`{v4.get('n_trading_observations')}` n_nav_rows=`{v4.get('n_nav_rows')}`",
        f"- NAV_end=`{v4.get('nav_end')}` total_return=`{v4.get('total_return')}`",
        f"- vs lock `{v4_match['status']}`",
        "",
        "## v5 = R16_O01",
        "",
        f"- engine=`cached_overlay_v4 + apply_top2_veto + run_canonical`",
        f"- formula=`{O01_FORMULA}` n_signal=`{len(dates)}`",
        f"- CAGR(日历年)=`{v5.get('cagr')}`（对照 252 日 cagr_of=`{v5.get('cagr_of_252')}`，非正式口径）",
        f"- Sharpe=`{v5.get('sharpe')}` MaxDD=`{v5.get('max_drawdown')}`",
        f"- completed_trades=`{v5.get('completed_trades')}` n_trading_observations=`{v5.get('n_trading_observations')}` n_nav_rows=`{v5.get('n_nav_rows')}`",
        f"- NAV_end=`{v5.get('nav_end')}` total_return=`{v5.get('total_return')}`",
        f"- vs lock `{v5_match['status']}`",
        "",
        "## NAV",
        "",
        f"- NAV_end v4=`{v4.get('nav_end')}` v5=`{v5.get('nav_end')}` Δ=`{nav_delta}`",
        f"- Δ vs lock `{payload['nav_delta_vs_lock']['status']}` anchor=`{NAV_DELTA_ANCHOR}`",
        "",
        f"产物目录：`{DEST_ROOT}`",
        "",
    ]
    md_path = DEST_ROOT / "REPLAY_V4_V5.md"
    assert_project_result_path(md_path)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    _log(f"DONE status={overall} summary={DEST_ROOT / 'summary.json'}")
    print(json.dumps({k: payload[k] for k in ("status", "nav_end_v4", "nav_end_v5", "nav_delta")}, ensure_ascii=False))
    return 0 if overall == "MATCH" else 0


if __name__ == "__main__":
    raise SystemExit(main())
