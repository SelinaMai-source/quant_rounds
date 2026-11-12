"""正式袖套只给 OOF 入围配置，并且 E1/E3、E4/E5 同级都有账本。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND22_ROOT
from ..round11.common import daily_from_nav, metrics_from_nav
from .common import dump_json, finite_or_none, load_json, write_csv, write_md
from .contract import (
    ECONOMIC_MATERIALITY,
    FEE_SCHEDULE,
    HISTORICAL_FEE_CORRECTED,
    MD,
    SAME_OPEN_EXECUTION_OPTIMISM,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    exp_dir,
)
from .engine import run_v5_fused_backtest
from .freeze import require_frozen_gate

BT_JSON = ROUND22_ROOT / "backtest" / "sleeve.json"
SLEEVE_CSV = ROUND22_ROOT / "backtest" / "sleeve_results.csv"


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
                }
            )
    return out


def _sleeve_codes(gates: list[dict[str, Any]]) -> list[str]:
    by_factor: dict[str, list[dict[str, Any]]] = {}
    for gate in gates:
        fid = gate.get("factor_id") or ""
        by_factor.setdefault(str(fid), []).append(gate)
    chosen: list[str] = []
    tiers = (("E1_", "E2_", "E3_"), ("E4_", "E5_"), ("E6_", "E7_"), ("E8_", "E9_"))
    for rows in by_factor.values():
        for prefixes in tiers:
            group = [r for r in rows if any(str(r.get("role") or "").startswith(p.rstrip("_")) or p.rstrip("_") in str(r.get("role") or "") for p in prefixes)]
            # role is E1_FIVE_ADD etc.
            group = [r for r in rows if str(r.get("role") or "").split("_", 1)[0] in {p.rstrip("_") for p in prefixes}]
            if not group:
                continue
            if any((r.get("oof_gate") or {}).get("passed") for r in group):
                chosen.extend(str(r["code"]) for r in group if r.get("code"))
    return list(dict.fromkeys(chosen))


def _material(obs: dict[str, Any], *, top_changed: bool, fee_same_direction: bool | None, not_only_year: bool | None) -> dict[str, Any]:
    gate = ECONOMIC_MATERIALITY
    dc = None if obs.get("cagr") is None else float(obs["cagr"]) - float(V5_ANCHOR["cagr"])
    ds = None if obs.get("sharpe") is None else float(obs["sharpe"]) - float(V5_ANCHOR["sharpe"])
    dn = None if obs.get("nav_end") is None else abs(float(obs["nav_end"]) - float(V5_NAV_END_EXACT)) / float(V5_NAV_END_EXACT)
    dd = None if obs.get("max_drawdown") is None else float(obs["max_drawdown"]) - float(V5_ANCHOR["max_drawdown"])
    flags = {
        "delta_cagr_ok": dc is not None and dc >= float(gate["delta_cagr_min_abs"]),
        "delta_sharpe_ok": ds is not None and ds >= float(gate["delta_sharpe_min"]),
        "nav_rel_ok": dn is not None and dn >= float(gate["nav_rel_min"]),
        "top_changed_ok": bool(top_changed),
        "maxdd_ok": dd is None or dd >= -float(gate["maxdd_worse_max_abs"]),
        "fee_direction_ok": bool(fee_same_direction),
        "not_only_2024_or_2025": bool(not_only_year),
    }
    return {"delta_cagr": dc, "delta_sharpe": ds, "nav_rel": dn, "delta_maxdd": dd, "flags": flags, "economically_material": all(flags.values()), "gate_id": gate["id"]}


def _fee(code: str, *, resume: bool = True) -> dict[str, Any]:
    ov = exp_dir(code) / "overlayed.parquet"
    if not ov.is_file():
        alt = ROUND22_ROOT / "weights" / f"overlayed_{code}.parquet"
        ov = alt if alt.is_file() else ov
    if code == "R22_E0" and not ov.is_file():
        ov = ROUND22_ROOT / "weights" / "overlayed_R22_V5_R16_O01.parquet"
    dest = ROUND22_ROOT / "backtest" / "fee_sensitivity" / code
    if resume and (dest / "nav.parquet").is_file():
        obs = _obs(dest)
        if obs.get("sharpe") is not None:
            return {"code": code, "status": "ran", "fee": HISTORICAL_FEE_CORRECTED, **obs}
    if not ov.is_file():
        return {"code": code, "status": "BLOCKED", "reason": "no_overlay", "sharpe": None, "fee": HISTORICAL_FEE_CORRECTED}
    from ..round17.backtest import run_canonical

    ran = run_canonical(
        dest,
        pd.read_parquet(ov),
        extra={"code": code, "fee_schedule": HISTORICAL_FEE_CORRECTED, "historical_stamp_tax": True, "round": "22"},
        resume=resume,
    )
    obs = _obs(dest)
    if obs.get("sharpe") is None:
        obs = ran.get("observed") or obs
    payload = {"code": code, "status": "ran" if finite_or_none(obs.get("sharpe")) is not None else "BLOCKED", "fee": HISTORICAL_FEE_CORRECTED, **obs}
    dump_json(dest / "fee_summary.json", payload)
    return payload


def _execution_note() -> dict[str, Any]:
    trades = ROUND22_ROOT / "parity" / "R22_V5_R16_O01" / "trades.parquet"
    payload: dict[str, Any] = {
        "same_open_execution_optimism": True,
        "decision": "D_close",
        "order": "D+1_open",
        "engine_changed": False,
        "status": "AUDIT_ONLY",
    }
    if trades.is_file() and trades.stat().st_size > 8:
        frame = pd.read_parquet(trades)
        payload["n_trades"] = int(len(frame))
        payload["columns"] = list(frame.columns)[:30]
        limit_cols = [c for c in frame.columns if "limit" in c.lower()]
        payload["limit_columns"] = limit_cols
        if not limit_cols:
            payload["unfillable_open_share"] = None
            payload["reason"] = "成交表没有涨停开盘标记。敏感性只保留已审计的 same_open_execution_optimism，不另造不可成交比例。"
    else:
        payload["reason"] = "没有可读成交表"
    dump_json(ROUND22_ROOT / "diagnostics" / "execution_sensitivity.json", payload)
    return payload


def run_backtest(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and BT_JSON.is_file():
        existing = load_json(BT_JSON)
        if existing.get("status") in {"DONE", "PARTIAL"}:
            return existing
    train = load_json(ROUND22_ROOT / "model" / "train.json")
    gates = train.get("gates") or []
    codes = _sleeve_codes(gates)
    rows = []
    e0_dest = exp_dir("R22_E0")
    if not (e0_dest / "nav.parquet").is_file():
        src = ROUND22_ROOT / "parity" / "R22_V5_R16_O01"
        e0_dest.mkdir(parents=True, exist_ok=True)
        for name in ("nav.parquet", "report.json", "trades.parquet", "replay_summary.json"):
            sp = src / name
            if sp.is_file() and sp.stat().st_size > 8:
                (e0_dest / name).write_bytes(sp.read_bytes())
    e0_obs = _obs(e0_dest)
    gate_by_code = {str(g.get("code")): g for g in gates}
    e0_obs.update({"code": "R22_E0", "role": "E0_CONTROL", "factor_id": "", "fee_schedule": FEE_SCHEDULE, "status": "ran"})
    rows.append(e0_obs)
    for code in codes:
        dest = exp_dir(code)
        meta = gate_by_code.get(code) or {}
        five_p, ten_p = dest / "five_pred.parquet", dest / "ten_pred.parquet"
        if not five_p.is_file() or not ten_p.is_file():
            rows.append({"code": code, "role": meta.get("role"), "factor_id": meta.get("factor_id"), "status": "SKIP_NO_OOF"})
            continue
        if not ((dest / "nav.parquet").is_file() and (dest / "report.json").is_file()):
            run_v5_fused_backtest(
                dest,
                pd.read_parquet(five_p),
                pd.read_parquet(ten_p),
                extra={"code": code, "fee_schedule": FEE_SCHEDULE, "validation_hooks": None, "round": "22"},
                resume=resume,
            )
        obs = _obs(dest)
        obs.update({"code": code, "role": meta.get("role"), "factor_id": meta.get("factor_id"), "status": "ran" if obs.get("sharpe") is not None else "BLOCKED", "fee_schedule": FEE_SCHEDULE})
        rows.append(obs)
    fee_rows = [_fee("R22_E0", resume=resume)]
    for code in codes:
        fee_rows.append(_fee(code, resume=resume))
    fee_map = {r["code"]: r for r in fee_rows}
    base_fee = fee_map.get("R22_E0") or {}
    packed = []
    for row in rows:
        fee = fee_map.get(row.get("code")) or {}
        ds = None if row.get("sharpe") is None else float(row["sharpe"]) - float(V5_ANCHOR["sharpe"])
        fee_ds = None
        if fee.get("sharpe") is not None and base_fee.get("sharpe") is not None and row.get("code") != "R22_E0":
            fee_ds = float(fee["sharpe"]) - float(base_fee["sharpe"])
        same = None if ds is None or fee_ds is None else (ds == 0 and fee_ds == 0) or (ds > 0 and fee_ds > 0) or (ds < 0 and fee_ds < 0)
        mat = _material(row, top_changed=False, fee_same_direction=same, not_only_year=False)
        packed.append({**row, "delta_sharpe": ds, "fee_sharpe": fee.get("sharpe"), "fee_delta_sharpe": fee_ds, "material": mat})
    execution = _execution_note()
    payload = {
        "status": "DONE",
        "n_full_backtests": int(sum(r.get("status") == "ran" for r in rows)),
        "n_candidate_sleeves": len(codes),
        "codes": codes,
        "rows": packed,
        "execution": execution,
        "same_open_execution_optimism": SAME_OPEN_EXECUTION_OPTIMISM,
    }
    dump_json(BT_JSON, payload)
    write_csv(SLEEVE_CSV, [{k: v for k, v in r.items() if k != "material"} | {"economically_material": (r.get("material") or {}).get("economically_material")} for r in packed])
    write_csv(ROUND22_ROOT / "backtest" / "fee_both_calibers.csv", fee_rows)
    write_md(MD["17"], f"# Round22 · 17 全袖套\n\n主口径 `{FEE_SCHEDULE}`。候选袖套数={len(codes)}。正式回测行={payload['n_full_backtests']}（含 E0）。没有 OOF 入围就不跑候选袖套。\n")
    return payload
