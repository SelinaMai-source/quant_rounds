"""重放 baseline_v5 = R16_O01，Round7 5bp 口径。FAIL 只停正式晋级，仍写 artifact。"""

from __future__ import annotations

from typing import Any

from ..paths import MARKET_END, PREDICTION_START, ROUND20_ROOT, assert_project_result_path
from .common import (
    close_enough,
    daily_from_nav,
    dump_json,
    finite_or_none,
    load_json,
    observed_from_report,
    write_md,
)
from .contract import (
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    FEE_SCHEDULE,
    FORMAL_PROMOTION_BLOCKED,
    NAV_END_ABS_TOL,
    N_TRADING_OBSERVATIONS_ANCHOR,
    O01_FORMULA,
    PARITY_MD,
    PARITY_METRIC_ABS,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    overlayed_path,
    parity_dir,
)

PARITY_JSON = ROUND20_ROOT / "parity" / "parity.json"


def load_parity_status() -> dict[str, Any]:
    return load_json(PARITY_JSON)


def _run_r7(dest, overlayed, extra: dict[str, Any]) -> dict[str, Any]:
    from ..local_executor import (
        round7_fixed_sleeve_config,
        run_local_fixed_sleeve_backtest,
        write_backtest_artifacts,
    )
    from ..round4.runner import attach_round4_fields
    from ..round17.backtest import cached_market

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
        "engine": "round7_fixed_sleeve_config + run_local_fixed_sleeve_backtest (no hist patch)",
        "note": "copied chain from tools/replay_r7_caliber.py; NO validation_hooks",
    }
    payload.update(extra)
    written = write_backtest_artifacts(result, dest, extra_report=payload)
    report = written["report"]
    attach_round4_fields(result["nav"], result["trades"], report.setdefault("metrics", {}))
    dump_json(dest / "report.json", report)
    return {
        "report": report,
        "nav": result["nav"],
        "trades": result["trades"],
        "out_dir": str(dest),
    }


def _pack(ran: dict[str, Any], dest) -> dict[str, Any]:
    from ..round16.common import cagr_of
    from ..round16_5.common import nav_end

    report = ran.get("report") or {}
    nav = ran.get("nav")
    obs = ran.get("observed") or observed_from_report(report)
    metrics = report.get("metrics") or {}
    daily = daily_from_nav(nav) if nav is not None else None
    n_obs = int(len(daily)) if daily is not None else None
    yearly = obs.get("yearly_returns") or metrics.get("yearly_returns") or {}
    end_nav = nav_end(nav) if nav is not None else finite_or_none(metrics.get("final_asset"))
    calendar = finite_or_none(metrics.get("annualized_return"))
    if calendar is None:
        calendar = finite_or_none(obs.get("cagr"))
    trading_cagr = finite_or_none(cagr_of(daily)) if daily is not None and len(daily) else None
    cfg = report.get("config") or {}
    costs = cfg.get("costs") or {}
    return {
        "cagr": calendar,
        "cagr_calendar_365_25": calendar,
        "cagr_of_252": trading_cagr,
        "sharpe": finite_or_none(obs.get("sharpe") or metrics.get("sharpe_zero_rate")),
        "max_drawdown": finite_or_none(obs.get("max_drawdown") or metrics.get("max_drawdown")),
        "completed_trades": obs.get("completed_trades") if obs.get("completed_trades") is not None else metrics.get("completed_trades"),
        "n_trading_observations": n_obs,
        "n_obs": n_obs,
        "n_nav_rows": int(len(nav)) if nav is not None else None,
        "total_return": finite_or_none(obs.get("total_return") or metrics.get("total_return")),
        "yearly_returns": {str(k): finite_or_none(v) for k, v in yearly.items()},
        "nav_end": end_nav,
        "fee_schedule": report.get("fee_schedule") or FEE_SCHEDULE,
        "sell_tax_bps": costs.get("sell_tax_bps"),
        "initial_cash": cfg.get("initial_cash"),
        "rank_name_budget_nav_fractions": cfg.get("rank_name_budget_nav_fractions"),
        "out_dir": str(dest),
        "nav_path": str(dest / "nav.parquet") if (dest / "nav.parquet").is_file() else None,
        "trades_path": str(dest / "trades.parquet") if (dest / "trades.parquet").is_file() else None,
        "report_path": str(dest / "report.json") if (dest / "report.json").is_file() else None,
        "validation_hooks": None,
    }


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and PARITY_JSON.is_file() and PARITY_MD.is_file():
        existing = load_json(PARITY_JSON)
        if existing.get("status") in {"PASS", "FAIL"} and existing.get("observed"):
            return existing

    from ..round15.runner import apply_top2_veto
    from ..round16_5.engine import o01_signal_dates
    from ..round17.backtest import cached_overlay_v4

    ov4 = cached_overlay_v4()
    dates = o01_signal_dates()
    ov5 = apply_top2_veto(ov4, dates)
    ov_path = overlayed_path("R20_V5_R16_O01")
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(ov_path)
    ov5.to_parquet(ov_path, index=False)
    dest = parity_dir("R20_V5_R16_O01")
    ran = _run_r7(
        dest,
        ov5,
        extra={
            "code": "R20_V5_R16_O01",
            "round": 20,
            "role": "exact_v5_replay_round7_caliber",
            "parent": "R9_B03",
            "formula": O01_FORMULA,
            "overlay": "cached_overlay_v4 → o01_signal_dates → apply_top2_veto",
            "current": CURRENT_BASELINE,
            "alias": CURRENT_ALIAS,
        },
    )
    packed = _pack(ran, dest)
    packed["code"] = "baseline_v5"
    packed["alias"] = CURRENT_ALIAS
    packed["n_signal_dates"] = int(len(dates))
    packed["formula"] = O01_FORMULA
    dump_json(dest / "replay_summary.json", packed)

    checks = {
        "cagr": close_enough(packed.get("cagr"), V5_ANCHOR["cagr"], abs_tol=PARITY_METRIC_ABS),
        "sharpe": close_enough(packed.get("sharpe"), V5_ANCHOR["sharpe"], abs_tol=PARITY_METRIC_ABS),
        "max_drawdown": close_enough(packed.get("max_drawdown"), V5_ANCHOR["max_drawdown"], abs_tol=PARITY_METRIC_ABS),
        "completed_trades": int(packed.get("completed_trades") or -1) == int(V5_ANCHOR["completed_trades"]),
        "n_obs": int(packed.get("n_trading_observations") or -1) == int(N_TRADING_OBSERVATIONS_ANCHOR),
    }
    nav_ok = close_enough(packed.get("nav_end"), V5_NAV_END_EXACT, abs_tol=NAV_END_ABS_TOL) or close_enough(
        packed.get("nav_end"), V5_ANCHOR["nav_end"], abs_tol=NAV_END_ABS_TOL
    )
    passed = bool(all(checks.values()))
    status = "PASS" if passed else "FAIL"
    payload = {
        "status": status,
        "passed": passed,
        "fail_code": None if passed else FORMAL_PROMOTION_BLOCKED,
        "formal_promotion_blocked": not passed,
        "code": "R20_V5_R16_O01",
        "current": CURRENT_BASELINE,
        "alias": CURRENT_ALIAS,
        "engine": "cached_overlay_v4 → o01_signal_dates → apply_top2_veto → round7_fixed_sleeve_config + run_local_fixed_sleeve_backtest",
        "validation_hooks": None,
        "fee_schedule": FEE_SCHEDULE,
        "historical_stamp_tax": False,
        "observed": packed,
        "anchor": dict(V5_ANCHOR),
        "checks": checks,
        "nav_end_ok": nav_ok,
        "abs_tol": PARITY_METRIC_ABS,
        "nav_end_abs_tol": NAV_END_ABS_TOL,
        "n_signal_dates": int(len(dates)),
        "formula": O01_FORMULA,
        "out_dir": str(dest),
        "overlay_path": str(ov_path),
        "note": "FAIL 只停止正式晋级评价；仍写本 artifact。禁止改 O01/B03/90-10。",
    }
    dump_json(PARITY_JSON, payload)
    lines = [
        "# Round20 · 00 baseline_v5 parity（Round7 5bp 口径）",
        "",
        f"结果：**{status}**。CURRENT=`{CURRENT_BASELINE}` = `{CURRENT_ALIAS}` USER_LOCK。",
        "",
        "engine=`cached_overlay_v4` → `o01_signal_dates` → `apply_top2_veto` → `round7_fixed_sleeve_config` + `run_local_fixed_sleeve_backtest`。**不传** `validation_hooks`。卖税 5bp 一口价。CAGR 日历 365.25。",
        "",
        "## 目标",
        "",
        f"- CAGR=`{V5_ANCHOR['cagr']}` Sharpe=`{V5_ANCHOR['sharpe']}` MaxDD=`{V5_ANCHOR['max_drawdown']}`",
        f"- trades=`{V5_ANCHOR['completed_trades']}` n_obs=`{V5_ANCHOR['n_obs']}` NAV_end(计划展示)=`{V5_ANCHOR['nav_end']}`",
        "",
        "## 观测",
        "",
        f"- CAGR=`{packed.get('cagr')}` Sharpe=`{packed.get('sharpe')}` MaxDD=`{packed.get('max_drawdown')}`",
        f"- trades=`{packed.get('completed_trades')}` n_obs=`{packed.get('n_trading_observations')}` NAV_end=`{packed.get('nav_end')}`",
        "",
        "## 检查",
        "",
        f"- cagr=`{checks['cagr']}` sharpe=`{checks['sharpe']}` max_drawdown=`{checks['max_drawdown']}`",
        f"- completed_trades=`{checks['completed_trades']}` n_obs=`{checks['n_obs']}` nav_end_ok=`{nav_ok}`",
        f"- abs_tol=`{PARITY_METRIC_ABS}`",
        "",
        f"FAIL → `{FORMAL_PROMOTION_BLOCKED}`（停止正式晋级评价，盘点/冻结仍继续）。本文件在 FAIL 时仍落盘。",
        "",
        f"out=`{dest}` overlay=`{ov_path}`",
        "",
    ]
    write_md(PARITY_MD, "\n".join(lines))
    return payload
