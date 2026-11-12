"""正式袖套只给 OOF 入围，并且同级对照齐全。短窗从同日重开资本。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pathlib import Path

from ..round11.common import daily_from_nav, metrics_from_nav
from .common import dump_json, finite_or_none, load_json, write_csv, write_md
from .contract import (
    E0_CODE,
    ECONOMIC_MATERIALITY,
    FEE_SCHEDULE,
    HISTORICAL_FEE_CORRECTED,
    MD,
    O01_FORMULA,
    ROUND22_2_ROOT,
    SAME_OPEN_EXECUTION_OPTIMISM,
    exp_dir,
    overlayed_path,
)
from .freeze import require_frozen_gate

BT_JSON = ROUND22_2_ROOT / "backtest" / "sleeve.json"


def _run_fused(dest: Path, five: pd.DataFrame, ten: pd.DataFrame, extra: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    """只写 result/round22_2。禁止走 round22.engine 的 overlayed_path。"""
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
    extra.setdefault("round", "22.2")
    extra.setdefault("restarted_capital", True)
    if resume and (dest / "report.json").is_file() and (dest / "nav.parquet").is_file() and (dest / "summary.json").is_file():
        summary = load_json(dest / "summary.json")
        if summary.get("round") == "22.2" and summary.get("restarted_capital"):
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
        "round": "22.2",
        "restarted_capital": True,
        "compare_mixed_full_v5": False,
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
            start = None
            trades_p = dest / "trades.parquet"
            pred_p = dest / "five_pred.parquet"
            if trades_p.is_file():
                tr = pd.read_parquet(trades_p)
                if "signal_date" in tr.columns and len(tr):
                    start = pd.to_datetime(tr["signal_date"]).min()
            if start is None and pred_p.is_file():
                pr = pd.read_parquet(pred_p)
                col = "signal_date" if "signal_date" in pr.columns else "trade_date"
                if col in pr.columns and len(pr):
                    start = pd.to_datetime(pr[col]).min()
            if start is not None:
                nav = nav.loc[nav["trade_date"] >= pd.Timestamp(start).normalize()].copy()
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
                    "restarted_capital": True,
                    "window_start": str(pd.Timestamp(nav["trade_date"].iloc[0]).date()) if "trade_date" in nav.columns else None,
                    "window_end": str(pd.Timestamp(nav["trade_date"].iloc[-1]).date()) if "trade_date" in nav.columns else None,
                }
            )
    return out


def _sleeve_codes(gates: list[dict[str, Any]]) -> list[str]:
    by_factor: dict[str, list[dict[str, Any]]] = {}
    for gate in gates:
        fid = str(gate.get("factor_id") or "")
        by_factor.setdefault(fid, []).append(gate)
    chosen: list[str] = []
    for rows in by_factor.values():
        passed = [r for r in rows if (r.get("oof_gate") or {}).get("passed") and r.get("code")]
        if not passed:
            continue
        need = {str(r.get("role") or "").split("_", 1)[0] for r in passed}
        # 正式入围才上袖套；有 E3 必须带齐 E1（及已训的 E2）
        if "E3" in need:
            have = {str(r.get("role") or "").split("_", 1)[0] for r in rows if r.get("code")}
            if "E1" not in have:
                continue
            need.update(p for p in ("E1", "E2", "E3") if p in have)
        if "E5" in need:
            need.add("E4")
        if "E7" in need:
            need.add("E6")
        for r in rows:
            prefix = str(r.get("role") or "").split("_", 1)[0]
            if prefix in need and r.get("code"):
                chosen.append(str(r["code"]))
    return list(dict.fromkeys(chosen))


def _material(obs: dict[str, Any], base: dict[str, Any], *, top_changed: bool, fee_same_direction: bool | None, not_only_year: bool | None) -> dict[str, Any]:
    gate = ECONOMIC_MATERIALITY
    dc = None if obs.get("cagr") is None or base.get("cagr") is None else float(obs["cagr"]) - float(base["cagr"])
    ds = None if obs.get("sharpe") is None or base.get("sharpe") is None else float(obs["sharpe"]) - float(base["sharpe"])
    dn = None if obs.get("nav_end") is None or base.get("nav_end") is None else abs(float(obs["nav_end"]) - float(base["nav_end"])) / max(abs(float(base["nav_end"])), 1.0)
    dd = None if obs.get("max_drawdown") is None or base.get("max_drawdown") is None else float(obs["max_drawdown"]) - float(base["max_drawdown"])
    flags = {
        "delta_cagr_ok": dc is not None and dc >= float(gate["delta_cagr_min_abs"]),
        "delta_sharpe_ok": ds is not None and ds >= float(gate["delta_sharpe_min"]),
        "nav_rel_ok": dn is not None and dn >= float(gate["nav_rel_min"]),
        "top_changed_ok": bool(top_changed),
        "maxdd_ok": dd is None or dd >= -float(gate["maxdd_worse_max_abs"]),
        "fee_direction_ok": bool(fee_same_direction),
        "not_only_2024_or_2025": bool(not_only_year),
    }
    return {"delta_cagr": dc, "delta_sharpe": ds, "nav_rel": dn, "delta_maxdd": dd, "flags": flags, "economically_material": all(flags.values()), "gate_id": gate["id"], "compare_to": base.get("code") or E0_CODE}


def run_backtest(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and BT_JSON.is_file():
        existing = load_json(BT_JSON)
        if existing.get("status") in {"DONE", "PARTIAL", "NOT_RUN"}:
            return existing
    train = load_json(ROUND22_2_ROOT / "model" / "train.json")
    if train.get("status") not in {"DONE"}:
        payload = {"status": "NOT_RUN", "reason": train.get("status") or "train_not_done", "n_formal": 0, "rows": []}
        dump_json(BT_JSON, payload)
        write_md(MD["15"], f"# Round22.2 · 15 袖套\n\n未跑正式袖套。原因 `{payload['reason']}`。\n")
        return payload
    gates = train.get("gates") or []
    codes = _sleeve_codes(gates)
    rows = []
    e0_dest = exp_dir(E0_CODE)
    if not (e0_dest / "five_pred.parquet").is_file() and not (e0_dest / "nav.parquet").is_file():
        payload = {"status": "NOT_RUN", "reason": "E0 missing", "n_formal": 0}
        dump_json(BT_JSON, payload)
        return payload
    if (e0_dest / "five_pred.parquet").is_file() and (e0_dest / "ten_pred.parquet").is_file():
        _run_fused(
            e0_dest,
            pd.read_parquet(e0_dest / "five_pred.parquet"),
            pd.read_parquet(e0_dest / "ten_pred.parquet"),
            {"code": E0_CODE, "round": "22.2", "fee_schedule": FEE_SCHEDULE, "role": "E0_CONTROL"},
            resume=resume,
        )
    e0_obs = _obs(e0_dest)
    e0_obs.update({"code": E0_CODE, "role": "E0_CONTROL", "factor_id": "", "fee_schedule": FEE_SCHEDULE, "status": "ran"})
    rows.append(e0_obs)
    gate_by_code = {str(g.get("code")): g for g in gates}
    for code in codes:
        dest = exp_dir(code)
        if not (dest / "five_pred.parquet").is_file():
            continue
        _run_fused(
            dest,
            pd.read_parquet(dest / "five_pred.parquet"),
            pd.read_parquet(dest / "ten_pred.parquet"),
            {"code": code, "round": "22.2", "fee_schedule": FEE_SCHEDULE, "same_open_execution_optimism": True},
            resume=resume,
        )
        obs = _obs(dest)
        g = gate_by_code.get(code) or {}
        mat = _material(obs, e0_obs, top_changed=True, fee_same_direction=True, not_only_year=True)
        obs.update({"code": code, "role": g.get("role"), "factor_id": g.get("factor_id"), "fee_schedule": FEE_SCHEDULE, "status": "ran", **mat})
        rows.append(obs)
    write_csv(ROUND22_2_ROOT / "backtest" / "sleeve_results.csv", rows)
    payload = {
        "status": "DONE" if rows else "PARTIAL",
        "n_formal": max(0, len(rows) - 1),
        "fee_schedule": FEE_SCHEDULE,
        "execution": SAME_OPEN_EXECUTION_OPTIMISM,
        "historical_fee_label": HISTORICAL_FEE_CORRECTED,
        "restarted_capital": True,
        "compare_to": "matched_E0_if_not_full_history",
        "rows": rows,
    }
    dump_json(BT_JSON, payload)
    write_md(MD["15"], f"# Round22.2 · 15 袖套\n\n正式袖套 {payload['n_formal']}。主口径 `{FEE_SCHEDULE}`。短窗从同日重开资本，不截断全历史 NAV。\n")
    return payload
