"""全袖套 Round7 5bp + HISTORICAL_FEE_CORRECTED。必须有真实 Sharpe。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND21_2_ROOT, ROUND21_3_ROOT
from ..round11.common import daily_from_nav, metrics_from_nav, observed_from_report
from .common import dump_json, finite_or_none, load_json, slim_obs, write_md
from .contract import (
    E0_CODE,
    FEE_SCHEDULE,
    HISTORICAL_FEE_CORRECTED,
    L1_E0,
    L1_E1,
    L1_E2,
    L1_E3,
    L1_SPECS,
    MD,
    SAME_OPEN_EXECUTION_OPTIMISM,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    exp_dir,
    material_vs_v5,
)
from .train import MATRIX_JSON, MODEL_JSON, train_one

BT_JSON = ROUND21_3_ROOT / "backtest" / "decision.json"
L1_BT_JSON = ROUND21_3_ROOT / "backtest" / "l1_decision.json"
BT_CSV = ROUND21_3_ROOT / "portfolio_backtest_results.csv"
YEAR_CSV = ROUND21_3_ROOT / "yearly_results.csv"
CMP_CSV = ROUND21_3_ROOT / "final_comparison.csv"
FEE_CSV = ROUND21_3_ROOT / "backtest" / "fee_both_calibers.csv"
SENS_ROOT = ROUND21_3_ROOT / "backtest" / "fee_sensitivity"


def _metrics_from_dest(dest) -> dict[str, Any]:
    nav_p = dest / "nav.parquet"
    report = load_json(dest / "report.json")
    obs = observed_from_report(report) if report else {}
    if nav_p.is_file():
        nav = pd.read_parquet(nav_p)
        if not nav.empty and "total_asset" in nav.columns:
            m = metrics_from_nav(nav)
            daily = daily_from_nav(nav)
            vol = float(daily.std(ddof=1)) if len(daily) > 2 else None
            sharpe = float(daily.mean() / vol * (252 ** 0.5)) if vol and vol > 0 else None
            obs = {
                **obs,
                "cagr": finite_or_none(m.get("annualized_return") or obs.get("cagr")),
                "sharpe": finite_or_none(sharpe if sharpe is not None else m.get("sharpe_zero_rate") or obs.get("sharpe")),
                "max_drawdown": finite_or_none(m.get("max_drawdown") or obs.get("max_drawdown")),
                "nav_end": finite_or_none(float(nav["total_asset"].iloc[-1])),
                "total_return": finite_or_none(m.get("total_return") or obs.get("total_return")),
                "n_obs": int(len(daily)),
            }
    return obs


def _fee_breakdown(dest) -> dict[str, Any]:
    trades_p = dest / "trades.parquet"
    out = {"commission": None, "stamp_tax": None, "transfer": None, "turnover": None, "n_trades": None}
    if not trades_p.is_file():
        return out
    t = pd.read_parquet(trades_p)
    out["n_trades"] = int(len(t))
    for col, key in (
        ("commission", "commission"),
        ("commission_fee", "commission"),
        ("stamp_tax", "stamp_tax"),
        ("stamp", "stamp_tax"),
        ("transfer_fee", "transfer"),
        ("transfer", "transfer"),
        ("turnover", "turnover"),
        ("amount", "turnover"),
    ):
        if col in t.columns and out.get(key) is None:
            out[key] = finite_or_none(pd.to_numeric(t[col], errors="coerce").sum())
    report = load_json(dest / "report.json") or {}
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    aliases = {
        "commission": ("commission", "commission_fee"),
        "stamp_tax": ("stamp_tax", "stamp"),
        "transfer": ("transfer_fee", "transfer"),
        "turnover": ("turnover", "turnover_value", "amount"),
    }
    for key, names in aliases.items():
        if out.get(key) is not None:
            continue
        for name in names:
            if finite_or_none(metrics.get(name)) is not None:
                out[key] = finite_or_none(metrics.get(name))
                break
            if finite_or_none(report.get(name)) is not None:
                out[key] = finite_or_none(report.get(name))
                break
            fees = report.get("fees") if isinstance(report.get("fees"), dict) else {}
            if finite_or_none(fees.get(name)) is not None:
                out[key] = finite_or_none(fees.get(name))
                break
    costs_p = dest / "costs.parquet"
    if costs_p.is_file():
        c = pd.read_parquet(costs_p)
        for col, key in (("commission", "commission"), ("transfer_fee", "transfer"), ("stamp_tax", "stamp_tax"), ("turnover", "turnover")):
            if out.get(key) is None and col in c.columns:
                out[key] = finite_or_none(pd.to_numeric(c[col], errors="coerce").sum())
    return out


def _import_21_2_sleeve(code: str, dest) -> bool:
    """预测哈希一致才拷 21.2 全袖套。不是 alias ADD。E1 21.2 无 NAV 则返回 False。"""
    old = str(code).replace("R21_3_", "R21_2_")
    src = ROUND21_2_ROOT / "candidates" / old
    if not (src / "nav.parquet").is_file() or not (src / "report.json").is_file():
        return False
    new_hash = load_json(dest / "prediction_hash.json")
    old_hash = load_json(src / "prediction_hash.json")
    if not new_hash or new_hash != old_hash:
        return False
    names = (
        "nav.parquet",
        "report.json",
        "trades.parquet",
        "cashflows.parquet",
        "costs.parquet",
        "positions.parquet",
        "rejections.parquet",
        "sleeve_states.parquet",
        "overlayed.parquet",
        "overlay_meta.json",
        "predictions.parquet",
    )
    import shutil

    for name in names:
        sp = src / name
        if sp.is_file():
            shutil.copy2(sp, dest / name)
    ov = dest / "overlayed.parquet"
    if ov.is_file():
        from .contract import overlayed_path

        op = overlayed_path(code)
        op.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ov, op)
    dump_json(
        dest / "imported_sleeve_from_21_2.json",
        {"source": str(src), "prediction_hash": new_hash, "reason": "identical pred hash; replay sleeve not alias ADD"},
    )
    return True


def run_full_backtest(code: str, spec: dict[str, Any] | None = None, *, resume: bool = True) -> dict[str, Any]:
    dest = exp_dir(code)
    if resume and (dest / "nav.parquet").is_file() and (dest / "summary.json").is_file():
        existing = load_json(dest / "summary.json")
        if finite_or_none(existing.get("sharpe")) is not None:
            existing["resumed"] = True
            return existing
    if resume and not (dest / "nav.parquet").is_file():
        _import_21_2_sleeve(code, dest)
    five_p, ten_p = dest / "five_pred.parquet", dest / "ten_pred.parquet"
    if not five_p.is_file() or not ten_p.is_file():
        if spec is None:
            return {"code": code, "status": "SKIP_NO_OOF"}
        train_one(spec, resume=True)
    if not five_p.is_file() or not ten_p.is_file():
        return {"code": code, "status": "SKIP_NO_OOF"}
    from .engine import run_v5_fused_backtest

    five = pd.read_parquet(five_p)
    ten = pd.read_parquet(ten_p)
    extra = {"code": code, "role": (spec or {}).get("role"), "fee_schedule": FEE_SCHEDULE, "validation_hooks": None, "round": "21.3"}
    obs = run_v5_fused_backtest(dest, five, ten, extra=extra, resume=resume)
    recomputed = _metrics_from_dest(dest)
    fees = _fee_breakdown(dest)
    oof = load_json(dest / "oof.json")
    payload = {
        **obs,
        **{k: recomputed.get(k) if recomputed.get(k) is not None else obs.get(k) for k in ("cagr", "sharpe", "max_drawdown", "nav_end", "total_return")},
        "code": code,
        "role": (spec or {}).get("role"),
        "factor_id": (spec or {}).get("factor_id") or oof.get("factor_id"),
        "five_oof": oof.get("five_oof"),
        "ten_oof": oof.get("ten_oof"),
        "five_drops": oof.get("five_drops"),
        "ten_drops": oof.get("ten_drops"),
        "fee_schedule": FEE_SCHEDULE,
        "fees": fees,
        "looked_at_combo_nav": True,
    }
    if payload.get("sharpe") is None:
        payload["status"] = "BLOCKED"
        payload["reason"] = "sharpe_none_after_recompute"
    dump_json(dest / "summary.json", payload)
    return payload


def _fee_sens(code: str, *, resume: bool = True) -> dict[str, Any]:
    ov = exp_dir(code) / "overlayed.parquet"
    dest = SENS_ROOT / code
    if not ov.is_file():
        return {"code": code, "status": "BLOCKED", "reason": "no_overlayed", "fee": HISTORICAL_FEE_CORRECTED, "sharpe": None}
    leftover = load_json(dest / "report.json") if (dest / "report.json").is_file() else {}
    hooks = leftover.get("validation_hooks")
    broken_hooks = isinstance(hooks, str)
    if (dest / "nav.parquet").is_file():
        recomputed = _metrics_from_dest(dest)
        if finite_or_none(recomputed.get("sharpe")) is not None:
            fees = _fee_breakdown(dest)
            report = leftover or load_json(dest / "report.json")
            obs = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
            payload = {
                "code": code,
                "status": "ran",
                "reason": "recomputed_from_nav_despite_string_hooks" if broken_hooks else "resumed_nav",
                "fee": HISTORICAL_FEE_CORRECTED,
                "cagr": recomputed.get("cagr") or obs.get("annualized_return"),
                "sharpe": recomputed.get("sharpe"),
                "max_drawdown": recomputed.get("max_drawdown") or obs.get("max_drawdown"),
                "completed_trades": obs.get("completed_trades") or leftover.get("completed_trades"),
                "nav_end": recomputed.get("nav_end"),
                "total_return": recomputed.get("total_return") or obs.get("total_return"),
                "fees": fees,
                "same_open_execution": SAME_OPEN_EXECUTION_OPTIMISM,
            }
            dump_json(dest / "fee_summary.json", payload)
            return payload
    if broken_hooks or ((dest / "report.json").is_file() and not (dest / "nav.parquet").is_file()):
        resume = False
    from ..round17.backtest import run_canonical

    # run_canonical 自己注入 hist_hooks() dict；extra 禁止覆盖成字符串（21.2 Q26 根因）。
    extra = {
        "code": code,
        "fee_schedule": HISTORICAL_FEE_CORRECTED,
        "historical_stamp_tax": True,
        "round": "21.3",
        "same_open_execution": SAME_OPEN_EXECUTION_OPTIMISM,
    }
    ran = run_canonical(dest, pd.read_parquet(ov), extra=extra, resume=resume)
    obs = ran.get("observed") or {}
    if not obs:
        obs = _metrics_from_dest(dest)
    recomputed = _metrics_from_dest(dest)
    fees = _fee_breakdown(dest)
    sharpe = recomputed.get("sharpe") if recomputed.get("sharpe") is not None else finite_or_none(obs.get("sharpe"))
    payload = {
        "code": code,
        "status": "ran" if sharpe is not None else "BLOCKED",
        "reason": None if sharpe is not None else "sharpe_none",
        "fee": HISTORICAL_FEE_CORRECTED,
        "cagr": recomputed.get("cagr") or obs.get("cagr"),
        "sharpe": sharpe,
        "max_drawdown": recomputed.get("max_drawdown") or obs.get("max_drawdown"),
        "completed_trades": obs.get("completed_trades") or recomputed.get("completed_trades") or (load_json(dest / "report.json").get("metrics") or {}).get("completed_trades"),
        "nav_end": recomputed.get("nav_end") or obs.get("nav_end"),
        "total_return": recomputed.get("total_return") or obs.get("total_return"),
        "fees": fees,
        "same_open_execution": SAME_OPEN_EXECUTION_OPTIMISM,
    }
    dump_json(dest / "fee_summary.json", payload)
    dump_json(dest / "summary.json", payload)
    return payload


def _pack_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    v5_sh = float(V5_ANCHOR["sharpe"])
    v5_cg = float(V5_ANCHOR["cagr"])
    v5_dd = float(V5_ANCHOR["max_drawdown"])
    main = []
    for r in rows:
        main.append(
            {
                "code": r.get("code"),
                "role": r.get("role"),
                "factor_id": r.get("factor_id"),
                "cagr": r.get("cagr"),
                "sharpe": r.get("sharpe"),
                "max_drawdown": r.get("max_drawdown"),
                "completed_trades": r.get("completed_trades"),
                "nav_end": r.get("nav_end"),
                "delta_sharpe": None if finite_or_none(r.get("sharpe")) is None else float(r["sharpe"]) - v5_sh,
                "delta_cagr": None if finite_or_none(r.get("cagr")) is None else float(r["cagr"]) - v5_cg,
                "delta_maxdd": None if finite_or_none(r.get("max_drawdown")) is None else float(r["max_drawdown"]) - v5_dd,
                "fee_schedule": FEE_SCHEDULE,
                "material": material_vs_v5(r),
            }
        )
    return main


def run_l1_backtest(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and L1_BT_JSON.is_file() and MD["15"].is_file():
        existing = load_json(L1_BT_JSON)
        if existing.get("status") == "DONE":
            return existing
    specs = {s["code"]: s for s in L1_SPECS}
    codes = [L1_E0, L1_E1, L1_E2, L1_E3]
    rows = []
    for code in codes:
        print(f"[round21.3] L1 BACKTEST {code}", flush=True)
        rows.append(run_full_backtest(code, specs.get(code), resume=resume))
    fee_rows = [_fee_sens(code, resume=resume) for code in codes]
    main_rows = _pack_rows(rows)
    year_rows = []
    for r in rows:
        for y, val in (r.get("yearly_returns") or {}).items():
            year_rows.append({"code": r.get("code"), "year": y, "return": val, "track": "L1"})
    pd.DataFrame(main_rows).to_csv(ROUND21_3_ROOT / "backtest" / "l1_four_way.csv", index=False)
    pd.DataFrame(fee_rows).to_csv(ROUND21_3_ROOT / "backtest" / "l1_fee_corrected.csv", index=False)
    contrasts = {}
    by = {r["code"]: r for r in rows}
    for a, b, name in (
        (L1_E1, L1_E0, "E1-E0"),
        (L1_E2, L1_E0, "E2-E0"),
        (L1_E3, L1_E0, "E3-E0"),
        (L1_E3, L1_E1, "E3-E1"),
        (L1_E3, L1_E2, "E3-E2"),
    ):
        ra, rb = by.get(a) or {}, by.get(b) or {}
        contrasts[name] = {
            "delta_sharpe": None if ra.get("sharpe") is None or rb.get("sharpe") is None else float(ra["sharpe"]) - float(rb["sharpe"]),
            "delta_cagr": None if ra.get("cagr") is None or rb.get("cagr") is None else float(ra["cagr"]) - float(rb["cagr"]),
            "delta_nav": None if ra.get("nav_end") is None or rb.get("nav_end") is None else float(ra["nav_end"]) - float(rb["nav_end"]),
            "delta_maxdd": None if ra.get("max_drawdown") is None or rb.get("max_drawdown") is None else float(ra["max_drawdown"]) - float(rb["max_drawdown"]),
        }
    # 核心问题：43 元来自 L1 还是删成交额对数？
    source = "UNIDENTIFIABLE"
    if contrasts.get("E1-E0", {}).get("delta_nav") is not None and contrasts.get("E3-E1", {}).get("delta_nav") is not None:
        d_drop = abs(contrasts["E1-E0"]["delta_nav"] or 0)
        d_l1 = abs(contrasts["E3-E1"]["delta_nav"] or 0)
        if d_drop > d_l1 * 2:
            source = "MOSTLY_DROP_G0_AMOUNT_LOG"
        elif d_l1 > d_drop * 2:
            source = "MOSTLY_ADD_L1"
        else:
            source = "MIXED_OR_NOISE"
    fee_dir = True
    e0f = next((x for x in fee_rows if x.get("code") == L1_E0), {})
    e3f = next((x for x in fee_rows if x.get("code") == L1_E3), {})
    if e0f.get("sharpe") is not None and e3f.get("sharpe") is not None:
        main_sign = 0
        if (by.get(L1_E3) or {}).get("sharpe") is not None:
            main_sign = 1 if float(by[L1_E3]["sharpe"]) > float((by.get(L1_E0) or {}).get("sharpe") or 0) else -1
        fee_sign = 1 if float(e3f["sharpe"]) > float(e0f["sharpe"]) else -1
        fee_dir = main_sign == 0 or fee_sign == main_sign
    payload = {
        "status": "DONE",
        "n_full_backtests": len(rows),
        "main_rows": main_rows,
        "fee_rows": fee_rows,
        "contrasts": contrasts,
        "forty_three_source": source,
        "fee_direction_holds": fee_dir,
        "mixed_into_main": False,
        "same_open_execution": SAME_OPEN_EXECUTION_OPTIMISM,
    }
    dump_json(L1_BT_JSON, payload)
    write_md(
        MD["15"],
        "\n".join(
            [
                "# Round21.3 · 15 L1 四路全引擎",
                "",
                f"E0–E3 全袖套。口径=`{FEE_SCHEDULE}`。43元来源=`{source}`。",
                f"对照：{contrasts}",
                f"费用校正分表 N=`{len(fee_rows)}` direction_holds=`{fee_dir}`。",
                "",
            ]
        ),
    )
    return payload


def run_backtest(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and BT_JSON.is_file() and CMP_CSV.is_file():
        existing = load_json(BT_JSON)
        if existing.get("status") == "DONE":
            return existing
    l1 = load_json(L1_BT_JSON)
    matrix = load_json(MATRIX_JSON)
    model = load_json(MODEL_JSON)
    specs = {s["code"]: s for s in (matrix.get("specs") or [])}
    from .interact import INT_JSON

    for rec in (load_json(INT_JSON).get("rows") or []):
        spec = rec.get("spec")
        if spec and spec.get("code"):
            specs.setdefault(spec["code"], spec)
    codes = [E0_CODE]
    for rec in (load_json(INT_JSON).get("rows") or []):
        if rec.get("status") in {"OOF", "DONE"} and rec.get("code"):
            codes.append(rec["code"])
    for r in model.get("rows") or []:
        if r.get("code") == E0_CODE:
            continue
        if r.get("formal") and r.get("role") in {"E2_ADD", "E3_REPLACE", "E5_CLUSTER_REPLACE", "E6_INTERACT"}:
            codes.append(r["code"])
    for s in matrix.get("specs") or []:
        if s.get("role") in {"E1_REMOVE", "E2_ADD", "E3_REPLACE"} and s["code"] not in codes:
            if s.get("factor_id") and s["code"].startswith("R21_3_"):
                codes.append(s["code"])
    codes = list(dict.fromkeys(codes))
    rows = []
    for code in codes:
        print(f"[round21.3] BACKTEST {code}", flush=True)
        rows.append(run_full_backtest(code, specs.get(code), resume=resume))
    main_rows = _pack_rows(rows)
    year_rows = []
    for r in rows:
        for y, val in (r.get("yearly_returns") or {}).items():
            year_rows.append({"code": r.get("code"), "year": y, "return": val})
    BT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(main_rows).to_csv(BT_CSV, index=False)
    pd.DataFrame(year_rows).to_csv(YEAR_CSV, index=False)
    pd.DataFrame(main_rows).to_csv(CMP_CSV, index=False)
    ranked = [r for r in rows if finite_or_none(r.get("sharpe")) is not None]
    best = max(ranked, key=lambda r: float(r["sharpe"])) if ranked else {}
    fee_codes = list(dict.fromkeys([E0_CODE, best.get("code"), L1_E1, L1_E3]))
    fee_rows = [_fee_sens(str(c), resume=resume) for c in fee_codes if c]
    pd.DataFrame(fee_rows).to_csv(FEE_CSV, index=False)
    payload = {
        "status": "DONE",
        "n_full_backtests": len(rows),
        "best_vs_v5": {
            "code": best.get("code"),
            "sharpe": best.get("sharpe"),
            "cagr": best.get("cagr"),
            "max_drawdown": best.get("max_drawdown"),
            "nav_end": best.get("nav_end"),
            "completed_trades": best.get("completed_trades"),
            "material": material_vs_v5(best),
        },
        "main_rows": main_rows,
        "fee_sensitivity": fee_rows,
        "l1": l1,
        "mixed_into_main": False,
        "same_open_execution": SAME_OPEN_EXECUTION_OPTIMISM,
    }
    dump_json(BT_JSON, payload)
    _write_factor_portfolios(main_rows)
    write_md(
        MD["18"],
        f"# Round21.3 · 18 费用敏感\n\n主表 `{FEE_SCHEDULE}`。校正 `{HISTORICAL_FEE_CORRECTED}` 必须有真实 Sharpe。mixed=`False`。\n`{SAME_OPEN_EXECUTION_OPTIMISM}`\n",
    )
    return payload


def _write_factor_portfolios(main_rows: list[dict[str, Any]]) -> None:
    """把正式回测行写回 factors/<id>/portfolio.json，避免 evaluate_factor 因空行 SKIPPED。"""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in main_rows:
        fid = r.get("factor_id")
        if not fid:
            continue
        by.setdefault(str(fid), []).append(r)
    for fid, rs in by.items():
        dest = ROUND21_3_ROOT / "factors" / fid
        dest.mkdir(parents=True, exist_ok=True)
        dump_json(dest / "portfolio.json", {"status": "DONE", "rows": rs, "n": len(rs)})
        dump_json(
            dest / "incrementality.json",
            {
                "status": "DONE",
                "n_portfolio_rows": len(rs),
                "rows": [{"code": x.get("code"), "sharpe": x.get("sharpe"), "delta_sharpe": x.get("delta_sharpe")} for x in rs],
            },
        )
