"""正式袖套走全历史 2018–2026。Δ 对照 R16_O01，不对照 R222_E0。不重开短窗资本。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..round11.common import daily_from_nav, metrics_from_nav
from .common import dump_json, finite_or_none, load_json, write_csv, write_md
from .contract import (
    E0_CODE,
    FEE_SCHEDULE,
    HISTORICAL_FEE_CORRECTED,
    MAIN_N_OBS,
    MAIN_WINDOW_END,
    MAIN_WINDOW_START,
    MD,
    O01_FORMULA,
    REPLAY_GROUPS,
    ROUND22_3_ROOT,
    SAME_OPEN_EXECUTION_OPTIMISM,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    exp_dir,
    material_vs_v5,
    overlayed_path,
)
from .freeze import require_frozen_gate

BT_JSON = ROUND22_3_ROOT / "backtest" / "sleeve.json"


def _run_fused(dest: Path, five: pd.DataFrame, ten: pd.DataFrame, extra: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    from ..paths import assert_project_result_path
    from ..round15.runner import apply_top2_veto
    from ..round16_5.engine import o01_signal_dates
    from ..round17.backtest import apply_b03, fuse_heads
    from ..round20.parity import _pack, _run_r7

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")
    extra = dict(extra or {})
    extra.setdefault("fee_schedule", FEE_SCHEDULE)
    extra.setdefault("validation_hooks", None)
    extra.setdefault("historical_stamp_tax", False)
    extra.setdefault("formula", O01_FORMULA)
    extra.setdefault("engine", "fuse_heads 90/10 → apply_b03 → o01 → apply_top2_veto → local_executor")
    extra.setdefault("round", "22.3")
    extra.setdefault("restarted_capital", False)
    extra.setdefault("window", f"{MAIN_WINDOW_START}–{MAIN_WINDOW_END}")
    if resume and (dest / "report.json").is_file() and (dest / "nav.parquet").is_file() and (dest / "summary.json").is_file():
        summary = load_json(dest / "summary.json")
        if summary.get("round") == "22.3" and summary.get("restarted_capital") is False:
            return summary
    fused = fuse_heads(five, ten)
    fused.to_parquet(dest / "predictions.parquet", index=False)
    overlayed, meta = apply_b03(fused)
    dump_json(dest / "overlay_meta.json", meta)
    ov5 = apply_top2_veto(overlayed, o01_signal_dates())
    code = str(extra.get("code") or dest.name)
    ov_path = overlayed_path(code)
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    ov5.to_parquet(ov_path, index=False)
    ov5.to_parquet(dest / "overlayed.parquet", index=False)
    ran = _run_r7(dest, ov5, extra)
    packed = _pack(ran, dest)
    obs = {
        "cagr": packed.get("cagr"),
        "sharpe": packed.get("sharpe"),
        "max_drawdown": packed.get("max_drawdown"),
        "completed_trades": packed.get("completed_trades"),
        "nav_end": packed.get("nav_end"),
        "total_return": packed.get("total_return"),
        "n_obs": packed.get("n_trading_observations") or packed.get("n_obs"),
        "yearly_returns": packed.get("yearly_returns") or {},
        "out_dir": str(dest),
        "status": "ran",
        "fee_schedule": extra.get("fee_schedule", FEE_SCHEDULE),
        "code": code,
        "round": "22.3",
        "restarted_capital": False,
        "compare_to": "R16_O01",
        "compare_mixed_r222_e0": False,
        "window": f"{MAIN_WINDOW_START}–{MAIN_WINDOW_END}",
    }
    dump_json(dest / "summary.json", obs)
    return obs


def _obs(dest) -> dict[str, Any]:
    report = load_json(dest / "report.json")
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    nav_path = dest / "nav.parquet"
    out = {
        "cagr": finite_or_none(metrics.get("annualized_return") or report.get("cagr")),
        "sharpe": finite_or_none(metrics.get("sharpe") or report.get("sharpe")),
        "max_drawdown": finite_or_none(metrics.get("max_drawdown") or report.get("max_drawdown")),
        "completed_trades": metrics.get("completed_trades") or report.get("completed_trades"),
        "nav_end": finite_or_none(report.get("nav_end")),
        "total_return": finite_or_none(metrics.get("total_return") or report.get("total_return")),
    }
    if nav_path.is_file() and nav_path.stat().st_size > 8:
        nav = pd.read_parquet(nav_path)
        if "trade_date" in nav.columns:
            nav["trade_date"] = pd.to_datetime(nav["trade_date"])
        if "total_asset" in nav.columns and not nav.empty:
            m = metrics_from_nav(nav)
            daily = daily_from_nav(nav)
            vol = float(daily.std(ddof=1)) if len(daily) > 2 else None
            sharpe = float(daily.mean() / vol * (252 ** 0.5)) if vol and vol > 0 else None
            wealth = (1.0 + daily.astype(float)).cumprod()
            peak = wealth.cummax()
            dd = float((wealth / peak - 1.0).min()) if len(wealth) else None
            nav_end = float(nav["total_asset"].iloc[-1])
            nav0 = float(nav["total_asset"].iloc[0])
            total = (nav_end / nav0 - 1.0) if nav0 else None
            out.update(
                {
                    "cagr": finite_or_none(m.get("annualized_return")) if finite_or_none(m.get("annualized_return")) is not None else out.get("cagr"),
                    "sharpe": sharpe if sharpe is not None else out.get("sharpe"),
                    "max_drawdown": dd if dd is not None else out.get("max_drawdown"),
                    "nav_end": nav_end,
                    "total_return": total if total is not None else out.get("total_return"),
                    "n_obs": int(len(daily)),
                    "nav_start": nav0,
                    "restarted_capital": False,
                    "window_start": str(pd.Timestamp(nav["trade_date"].iloc[0]).date()) if "trade_date" in nav.columns else None,
                    "window_end": str(pd.Timestamp(nav["trade_date"].iloc[-1]).date()) if "trade_date" in nav.columns else None,
                }
            )
    return out


def run_backtest(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and BT_JSON.is_file():
        existing = load_json(BT_JSON)
        if existing.get("status") in {"DONE"} and existing.get("compare_to") == "R16_O01":
            return existing
    train = load_json(ROUND22_3_ROOT / "model" / "train.json")
    if train.get("status") not in {"DONE"}:
        payload = {"status": "NOT_RUN", "reason": train.get("status") or "train_not_done", "n_formal": 0, "rows": []}
        dump_json(BT_JSON, payload)
        write_md(MD["06"], f"# Round22.3 · 06 全历史袖套\n\n未跑。原因 `{payload['reason']}`。\n")
        write_md(MD["15"], f"# Round22.3 · 15 袖套\n\n未跑正式袖套。原因 `{payload['reason']}`。\n")
        return payload
    v5_base = {
        "code": "R16_O01",
        "cagr": V5_ANCHOR["cagr"],
        "sharpe": V5_ANCHOR["sharpe"],
        "max_drawdown": V5_ANCHOR["max_drawdown"],
        "nav_end": V5_NAV_END_EXACT,
        "completed_trades": V5_ANCHOR["completed_trades"],
        "n_obs": MAIN_N_OBS,
    }
    rows = []
    codes = [g["code"] for g in REPLAY_GROUPS]
    for code in codes:
        dest = exp_dir(code)
        if not (dest / "five_pred.parquet").is_file() or not (dest / "ten_pred.parquet").is_file():
            if code == E0_CODE and (dest / "nav.parquet").is_file():
                obs = _obs(dest)
                obs.update({"code": code, "role": "E0_CONTROL", "factor_id": "", "fee_schedule": FEE_SCHEDULE, "status": "ran", "origin": "parity_copy"})
                rows.append(obs)
                continue
            continue
        _run_fused(
            dest,
            pd.read_parquet(dest / "five_pred.parquet"),
            pd.read_parquet(dest / "ten_pred.parquet"),
            {"code": code, "round": "22.3", "fee_schedule": FEE_SCHEDULE},
            resume=resume,
        )
        obs = _obs(dest)
        spec = next((g for g in REPLAY_GROUPS if g["code"] == code), {})
        if code == E0_CODE:
            obs.update({"code": code, "role": spec.get("role"), "factor_id": spec.get("factor_id") or "", "fee_schedule": FEE_SCHEDULE, "status": "ran", "compare_to": "R16_O01"})
        else:
            mat = material_vs_v5(obs, top_changed=True, fee_same_direction=True, not_only_year=True)
            obs.update({"code": code, "role": spec.get("role"), "factor_id": spec.get("factor_id"), "fee_schedule": FEE_SCHEDULE, "status": "ran", **mat})
        rows.append(obs)
    write_csv(ROUND22_3_ROOT / "backtest" / "sleeve_results.csv", rows)
    payload = {
        "status": "DONE" if rows else "PARTIAL",
        "n_formal": max(0, len([r for r in rows if r.get("code") != E0_CODE])),
        "fee_schedule": FEE_SCHEDULE,
        "execution": SAME_OPEN_EXECUTION_OPTIMISM,
        "historical_fee_label": HISTORICAL_FEE_CORRECTED,
        "restarted_capital": False,
        "compare_to": "R16_O01",
        "compare_mixed_r222_e0": False,
        "window": f"{MAIN_WINDOW_START}–{MAIN_WINDOW_END}",
        "target_n_obs": MAIN_N_OBS,
        "v5_base": v5_base,
        "rows": rows,
    }
    dump_json(BT_JSON, payload)
    write_md(
        MD["06"],
        f"# Round22.3 · 06 全历史袖套\n\n正式袖套 {payload['n_formal']}。对照 `R16_O01`，不对照 R222_E0。窗口 `{MAIN_WINDOW_START}–{MAIN_WINDOW_END}`。主口径 `{FEE_SCHEDULE}`。不重开短窗资本。\n",
    )
    write_md(MD["15"], f"# Round22.3 · 15 袖套\n\n正式袖套 {payload['n_formal']}。Δ 相对全历史 v5。短窗数字另存 matched_subperiod/。\n")
    return payload
