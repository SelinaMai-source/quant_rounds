#!/usr/bin/env python3
"""按 Round7 口径重跑 baseline_v4=R9_B03 与 baseline_v5=R16_O01。

引擎 = local_executor → Stock fixed_sleeve；费用 = ROUND7_FLAT_5BP（卖税 5bp 一口价）。
不打开 historical_stamp_tax。不覆盖 result/round7 / round16* / round18。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEST_ROOT = ROOT / "result" / "replay_r7_caliber"
V4_DIR = DEST_ROOT / "v4_R9_B03"
V5_DIR = DEST_ROOT / "v5_R16_O01"
ABS_TOL = 1e-8
FEE_SCHEDULE = "ROUND7_FLAT_5BP"


def caliber_dict() -> dict:
    return {
        "fee_schedule": FEE_SCHEDULE,
        "engine": "local_executor → Stock run_fixed_dual_sleeve_top3_backtest",
        "config_fn": "round7_fixed_sleeve_config",
        "exit_mode": "close_from_d2",
        "top_k": 2,
        "rank_name_budget_nav_fractions": [0.3, 0.1],
        "name_budget": 10_000.0,
        "initial_cash": 300_000.0,
        "sleeve_count": 2,
        "sleeve_cash": 150_000.0,
        "costs": {
            "slippage_bps": 12.3,
            "commission_bps": 1.0,
            "transfer_fee_bps": 0.1,
            "sell_tax_bps": 5.0,
        },
        "historical_stamp_tax": False,
        "cagr_daycount": 365.25,
        "cagr_formula": "(final_asset / initial_cash) ** (1 / years) - 1; years=(last-first).days/365.25",
        "sharpe_sqrt_days": 252,
        "risk_free_rate": 0.0,
        "metric_version": "r3_locked_window_v1",
        "b03_in_round7_entry": False,
        "validation_hooks": None,
        "note": (
            "与旧 R9 5bp flat 相同。HISTORICAL_FEE_CORRECTED（2023-08-28 前 10bp）"
            "不再是默认评价，除非用户明确要求。"
        ),
    }
R9_FLAT5_ANCHOR = {
    "cagr": 0.5373748002477328,
    "sharpe": 1.8433564243143739,
    "max_drawdown": -0.24751010811952434,
    "completed_trades": 1149,
    "total_return": 38.63211154586613,
    "nav_end": 11889633.46375984,
}
FEE_CORR = {
    "v4_cagr": 0.5253672918108749,
    "v4_sharpe": 1.8120914675763191,
    "v4_max_drawdown": -0.25455988069271596,
    "v4_trades": 1149,
    "v4_total_return": 36.06055834433112,
    "v4_nav_end": 11118167.503299337,
    "v5_cagr": 0.5361620306263046,
    "v5_sharpe": 1.8404039786206279,
    "v5_max_drawdown": -0.24792541309267202,
    "v5_trades": 1107,
    "v5_total_return": 38.36541857398669,
    "v5_nav_end": 11809625.572196007,
    "nav_delta": 691458.0688966699,
}


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


def _run_r7(dest: Path, overlayed, extra: dict[str, Any]) -> dict[str, Any]:
    from core.local_executor import (
        round7_fixed_sleeve_config,
        run_local_fixed_sleeve_backtest,
        write_backtest_artifacts,
    )
    from core.paths import MARKET_END, PREDICTION_START, assert_project_result_path
    from core.round4.runner import attach_round4_fields
    from core.round17.backtest import cached_market

    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")
    market_meta = cached_market()
    config = round7_fixed_sleeve_config(exit_mode="close_from_d2")
    result = run_local_fixed_sleeve_backtest(
        overlayed,
        market_meta["prepared"],
        config=config,
        start_date=PREDICTION_START,
        end_date=MARKET_END,
    )
    metrics = result["report"].setdefault("metrics", {})
    attach_round4_fields(result["nav"], result["trades"], metrics)
    payload = {
        "fee_schedule": FEE_SCHEDULE,
        "historical_stamp_tax": False,
        "validation_hooks": None,
        "caliber": caliber_dict(),
        "engine": "round7_fixed_sleeve_config + run_local_fixed_sleeve_backtest (no hist patch)",
    }
    payload.update(extra)
    written = write_backtest_artifacts(result, dest, extra_report=payload)
    report = written["report"]
    attach_round4_fields(result["nav"], result["trades"], report.setdefault("metrics", {}))
    (dest / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "report": report,
        "nav": result["nav"],
        "trades": result["trades"],
        "out_dir": str(dest),
    }


def _pack(ran: dict[str, Any], dest: Path) -> dict[str, Any]:
    from core.round11.common import daily_from_nav, observed_from_report
    from core.round16.common import cagr_of
    from core.round16_5.common import nav_end

    report = ran.get("report") or {}
    nav = ran.get("nav")
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
    end_nav = nav_end(nav) if nav is not None else _finite(metrics.get("final_asset"))
    calendar = _finite(metrics.get("annualized_return"))
    if calendar is None:
        calendar = _finite(obs.get("cagr"))
    trading_cagr = _finite(cagr_of(daily)) if daily is not None and len(daily) else None
    cfg = report.get("config") or {}
    costs = cfg.get("costs") or {}
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
        "fee_schedule": report.get("fee_schedule"),
        "sell_tax_bps": costs.get("sell_tax_bps"),
        "slippage_bps": costs.get("slippage_bps"),
        "commission_bps": costs.get("commission_bps"),
        "transfer_fee_bps": costs.get("transfer_fee_bps"),
        "initial_cash": cfg.get("initial_cash"),
        "rank_name_budget_nav_fractions": cfg.get("rank_name_budget_nav_fractions"),
        "out_dir": str(dest),
        "nav_path": str(dest / "nav.parquet") if (dest / "nav.parquet").is_file() else None,
        "trades_path": str(dest / "trades.parquet") if (dest / "trades.parquet").is_file() else None,
        "report_path": str(dest / "report.json") if (dest / "report.json").is_file() else None,
    }


def _match_status(obs: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "cagr": _close(obs.get("cagr"), anchor.get("cagr")),
        "sharpe": _close(obs.get("sharpe"), anchor.get("sharpe")),
        "max_drawdown": _close(obs.get("max_drawdown"), anchor.get("max_drawdown")),
        "completed_trades": int(obs.get("completed_trades") or -1) == int(anchor.get("completed_trades") or -2),
    }
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
    from core.round16_5.engine import o01_signal_dates
    from core.round17.backtest import cached_overlay_v4
    from core.round19.contract import O01_FORMULA

    assert ROOT.name == "一进二策略", ROOT
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(DEST_ROOT / "summary.json")
    caliber = caliber_dict()
    _dump(DEST_ROOT / "caliber.json", caliber)
    _log(f"ROOT={ROOT}")
    _log(f"DEST={DEST_ROOT}")
    _log(f"caliber fee={FEE_SCHEDULE} engine={caliber['engine']}")
    _log("historical_stamp_tax=False; not HISTORICAL_FEE_CORRECTED")

    _log("STEP overlay v4 (B03 on locked alpha)")
    ov4 = cached_overlay_v4()
    ov4_path = DEST_ROOT / "overlayed_v4_R9_B03.parquet"
    ov4.to_parquet(ov4_path, index=False)
    _log(f"overlay_v4 rows={len(ov4)} → {ov4_path}")

    _log("STEP backtest v4 R9_B03 Round7 口径")
    ran_v4 = _run_r7(
        V4_DIR,
        ov4,
        extra={
            "code": "REPLAY_R7_CALIBER_V4_R9_B03",
            "role": "replay_baseline_v4_round7_caliber",
            "parent": "R9_B03",
            "overlay": "cached_overlay_v4 B03",
        },
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

    _log("STEP backtest v5 R16_O01 Round7 口径")
    ran_v5 = _run_r7(
        V5_DIR,
        ov5,
        extra={
            "code": "REPLAY_R7_CALIBER_V5_R16_O01",
            "role": "replay_baseline_v5_round7_caliber",
            "parent": "R9_B03",
            "formula": O01_FORMULA,
            "overlay": "cached_overlay_v4 + apply_top2_veto O01",
        },
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

    v4_vs_r9 = _match_status(v4, R9_FLAT5_ANCHOR)
    payload = {
        "status": "complete",
        "fee_schedule": FEE_SCHEDULE,
        "historical_stamp_tax": False,
        "default_evaluation": "ROUND7_FLAT_5BP",
        "not_default": "HISTORICAL_FEE_CORRECTED",
        "engine_v4": "cached_overlay_v4 + round7_fixed_sleeve_config + run_local_fixed_sleeve_backtest",
        "engine_v5": "cached_overlay_v4 + apply_top2_veto + round7_fixed_sleeve_config + run_local_fixed_sleeve_backtest",
        "formula_v5": O01_FORMULA,
        "current_lock_untouched": "CURRENT_ALPHA_BASELINE = baseline_v5 = R16_O01 USER_LOCK",
        "cagr_primary": "calendar_year_annualized_return_365_25",
        "cagr_not_used_as_primary": "252_day_cagr_of",
        "completed_trades_vs_n_obs": "completed_trades is fills; n_trading_observations is daily NAV return rows",
        "b03": "v4/v5 overlay only; not part of Round7 entry itself",
        "caliber": caliber,
        "v4": v4,
        "v5": v5,
        "nav_end_v4": v4.get("nav_end"),
        "nav_end_v5": v5.get("nav_end"),
        "nav_delta": nav_delta,
        "v4_vs_r9_flat5": v4_vs_r9,
        "vs_historical_fee_corrected": {
            "v4_cagr_delta": None if v4.get("cagr") is None else float(v4["cagr"]) - FEE_CORR["v4_cagr"],
            "v4_sharpe_delta": None if v4.get("sharpe") is None else float(v4["sharpe"]) - FEE_CORR["v4_sharpe"],
            "v4_nav_delta": None if v4.get("nav_end") is None else float(v4["nav_end"]) - FEE_CORR["v4_nav_end"],
            "v5_cagr_delta": None if v5.get("cagr") is None else float(v5["cagr"]) - FEE_CORR["v5_cagr"],
            "v5_sharpe_delta": None if v5.get("sharpe") is None else float(v5["sharpe"]) - FEE_CORR["v5_sharpe"],
            "v5_nav_delta": None if v5.get("nav_end") is None else float(v5["nav_end"]) - FEE_CORR["v5_nav_end"],
            "nav_delta_r7": nav_delta,
            "nav_delta_fee_corrected": FEE_CORR["nav_delta"],
            "fee_corrected_lock": FEE_CORR,
        },
        "overlay_v4": str(ov4_path),
        "overlay_v5": str(ov5_path),
        "note": "Fresh Round7-caliber replay. Does not overwrite result/round7, result/round16*, result/round18. Does not unlock CURRENT.",
    }
    _dump(DEST_ROOT / "summary.json", payload)
    yr4 = v4.get("yearly_returns") or {}
    yr5 = v5.get("yearly_returns") or {}
    lines = [
        "# Round7 口径重跑 baseline_v4 / baseline_v5",
        "",
        f"费用=`{FEE_SCHEDULE}`（卖印花税 5bp 一口价）。**不是** `HISTORICAL_FEE_CORRECTED`。CURRENT 锁未改：`baseline_v5=R16_O01` USER_LOCK。",
        "",
        "## 口径",
        "",
        f"- engine=`{caliber['engine']}`",
        f"- config=`{caliber['config_fn']}` Top2 `RANK_NAV={caliber['rank_name_budget_nav_fractions']}` 初始现金=`{caliber['initial_cash']}`",
        f"- costs slippage={caliber['costs']['slippage_bps']}bp commission={caliber['costs']['commission_bps']}bp transfer={caliber['costs']['transfer_fee_bps']}bp sell_tax={caliber['costs']['sell_tax_bps']}bp flat",
        f"- CAGR `{caliber['cagr_formula']}`",
        "- B03 只在 v4/v5 overlay，不在 Round7 入口本身",
        "",
        "## v4 = R9_B03",
        "",
        f"- CAGR(日历年)=`{v4.get('cagr')}`（对照 252 日 cagr_of=`{v4.get('cagr_of_252')}`，非正式口径）",
        f"- Sharpe=`{v4.get('sharpe')}` MaxDD=`{v4.get('max_drawdown')}`",
        f"- completed_trades=`{v4.get('completed_trades')}` n_trading_observations=`{v4.get('n_trading_observations')}` n_nav_rows=`{v4.get('n_nav_rows')}`",
        f"- NAV_end=`{v4.get('nav_end')}` total_return=`{v4.get('total_return')}`",
        f"- vs 旧 R9 5bp flat `{v4_vs_r9['status']}`（1.843 / 53.74% 家族）",
        "",
        "## v5 = R16_O01",
        "",
        f"- formula=`{O01_FORMULA}` n_signal=`{len(dates)}`",
        f"- CAGR(日历年)=`{v5.get('cagr')}`（对照 252 日 cagr_of=`{v5.get('cagr_of_252')}`，非正式口径）",
        f"- Sharpe=`{v5.get('sharpe')}` MaxDD=`{v5.get('max_drawdown')}`",
        f"- completed_trades=`{v5.get('completed_trades')}` n_trading_observations=`{v5.get('n_trading_observations')}` n_nav_rows=`{v5.get('n_nav_rows')}`",
        f"- NAV_end=`{v5.get('nav_end')}` total_return=`{v5.get('total_return')}`",
        "",
        "## NAV",
        "",
        f"- NAV_end v4=`{v4.get('nav_end')}` v5=`{v5.get('nav_end')}` Δ(v5−v4)=`{nav_delta}`",
        "",
        "## 分年",
        "",
        f"- v4 yearly=`{yr4}`",
        f"- v5 yearly=`{yr5}`",
        "",
        f"产物目录：`{DEST_ROOT}`",
        "",
    ]
    md_path = DEST_ROOT / "REPLAY_R7_CALIBER.md"
    assert_project_result_path(md_path)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    _log(f"DONE summary={DEST_ROOT / 'summary.json'}")
    print(json.dumps(
        {
            "fee_schedule": FEE_SCHEDULE,
            "v4_cagr": v4.get("cagr"),
            "v4_sharpe": v4.get("sharpe"),
            "v5_cagr": v5.get("cagr"),
            "v5_sharpe": v5.get("sharpe"),
            "nav_delta": nav_delta,
            "v4_vs_r9_flat5": v4_vs_r9["status"],
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
