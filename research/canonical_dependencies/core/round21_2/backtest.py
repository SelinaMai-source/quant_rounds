"""全袖套 Round7 5bp。费用敏感分表。不改 O01/B03/90-10。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND21_2_ROOT
from .common import dump_json, finite_or_none, load_json, slim_obs, write_md
from .contract import E0_CODE, FEE_SCHEDULE, HISTORICAL_FEE_CORRECTED, MD, V5_ANCHOR, exp_dir
from .train import MATRIX_JSON, MODEL_JSON, train_one

BT_JSON = ROUND21_2_ROOT / "backtest" / "decision.json"
BT_CSV = ROUND21_2_ROOT / "portfolio_backtest_results.csv"
YEAR_CSV = ROUND21_2_ROOT / "yearly_results.csv"
CMP_CSV = ROUND21_2_ROOT / "final_comparison.csv"
SENS_ROOT = ROUND21_2_ROOT / "backtest" / "fee_sensitivity"


def run_full_backtest(code: str, spec: dict[str, Any] | None = None, *, resume: bool = True) -> dict[str, Any]:
    dest = exp_dir(code)
    if resume and (dest / "nav.parquet").is_file() and (dest / "summary.json").is_file():
        existing = load_json(dest / "summary.json")
        if existing.get("sharpe") is not None:
            existing["resumed"] = True
            return existing
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
    extra = {"code": code, "role": (spec or {}).get("role"), "fee_schedule": FEE_SCHEDULE, "validation_hooks": None, "round": "21.2"}
    obs = run_v5_fused_backtest(dest, five, ten, extra=extra, resume=resume)
    oof = load_json(dest / "oof.json")
    payload = {
        **obs,
        "code": code,
        "role": (spec or {}).get("role"),
        "five_oof": oof.get("five_oof"),
        "ten_oof": oof.get("ten_oof"),
        "five_drops": oof.get("five_drops"),
        "ten_drops": oof.get("ten_drops"),
        "fee_schedule": FEE_SCHEDULE,
        "looked_at_combo_nav": True,
    }
    dump_json(dest / "summary.json", payload)
    return payload


def _fee_sens(code: str) -> dict[str, Any]:
    ov = exp_dir(code) / "overlayed.parquet"
    if not ov.is_file():
        return {"code": code, "status": "SKIP_FEE"}
    from ..round17.backtest import run_canonical

    dest = SENS_ROOT / code
    extra = {"code": code, "fee_schedule": HISTORICAL_FEE_CORRECTED, "historical_stamp_tax": True, "validation_hooks": None, "round": "21.2"}
    ran = run_canonical(dest, pd.read_parquet(ov), extra=extra)
    return {"code": code, "status": ran.get("status"), "sharpe": (ran.get("report") or {}).get("sharpe"), "fee": HISTORICAL_FEE_CORRECTED}


def run_backtest(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and BT_JSON.is_file() and MD["15"].is_file():
        existing = load_json(BT_JSON)
        if existing.get("status") == "DONE":
            return existing
    matrix = load_json(MATRIX_JSON)
    model = load_json(MODEL_JSON)
    by_inc = {r.get("code"): r for r in (model.get("rows") or [])}
    specs = {s["code"]: s for s in (matrix.get("specs") or [])}
    codes = [E0_CODE]
    for r in model.get("rows") or []:
        if r.get("code") == E0_CODE:
            continue
        if r.get("formal") and r.get("role") in {"E2_ADD", "E3_REPLACE", "E5_CLUSTER_REPLACE"}:
            if (r.get("increment") or {}).get("passed") or r.get("role") == "E3_REPLACE":
                codes.append(r["code"])
    # 至少回测 E0 + 每个因子的 E2/E3
    for s in matrix.get("specs") or []:
        if s.get("role") in {"E2_ADD", "E3_REPLACE"} and s["code"] not in codes:
            codes.append(s["code"])
    codes = list(dict.fromkeys(codes))
    rows = []
    for code in codes:
        print(f"[round21.2] BACKTEST {code}", flush=True)
        rows.append(run_full_backtest(code, specs.get(code), resume=resume))
    v5_sh = float(V5_ANCHOR["sharpe"])
    v5_cg = float(V5_ANCHOR["cagr"])
    v5_dd = float(V5_ANCHOR["max_drawdown"])
    ranked = [r for r in rows if finite_or_none(r.get("sharpe")) is not None]
    best = max(ranked, key=lambda r: float(r["sharpe"])) if ranked else {}
    better = bool(
        best
        and finite_or_none(best.get("sharpe")) is not None
        and float(best["sharpe"]) > v5_sh + 1e-12
        and finite_or_none(best.get("cagr")) is not None
        and float(best["cagr"]) > v5_cg + 1e-12
    )
    main_rows = []
    year_rows = []
    for r in rows:
        main_rows.append(
            {
                "code": r.get("code"),
                "role": r.get("role"),
                "cagr": r.get("cagr"),
                "sharpe": r.get("sharpe"),
                "max_drawdown": r.get("max_drawdown"),
                "completed_trades": r.get("completed_trades"),
                "nav_end": r.get("nav_end"),
                "delta_sharpe": None if finite_or_none(r.get("sharpe")) is None else float(r["sharpe"]) - v5_sh,
                "delta_cagr": None if finite_or_none(r.get("cagr")) is None else float(r["cagr"]) - v5_cg,
                "delta_maxdd": None if finite_or_none(r.get("max_drawdown")) is None else float(r["max_drawdown"]) - v5_dd,
                "fee_schedule": FEE_SCHEDULE,
            }
        )
        for y, val in (r.get("yearly_returns") or {}).items():
            year_rows.append({"code": r.get("code"), "year": y, "return": val})
    BT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(main_rows).to_csv(BT_CSV, index=False)
    pd.DataFrame(year_rows).to_csv(YEAR_CSV, index=False)
    pd.DataFrame(main_rows).to_csv(CMP_CSV, index=False)
    fee_rows = []
    finalists = [best.get("code"), E0_CODE]
    for code in dict.fromkeys([c for c in finalists if c]):
        fee_rows.append(_fee_sens(str(code)))
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
            "delta_sharpe": None if finite_or_none(best.get("sharpe")) is None else float(best["sharpe"]) - v5_sh,
            "delta_cagr": None if finite_or_none(best.get("cagr")) is None else float(best["cagr"]) - v5_cg,
            "delta_maxdd": None if finite_or_none(best.get("max_drawdown")) is None else float(best["max_drawdown"]) - v5_dd,
            "better_than_v5": better,
        },
        "main_rows": main_rows,
        "fee_sensitivity": fee_rows,
        "mixed_into_main": False,
    }
    dump_json(BT_JSON, payload)
    write_md(
        MD["15"],
        "\n".join(
            [
                "# Round21.2 · 15 全引擎回测",
                "",
                f"N=`{len(rows)}` 口径=`{FEE_SCHEDULE}` best=`{best.get('code')}` Sharpe=`{best.get('sharpe')}` ΔSharpe=`{payload['best_vs_v5'].get('delta_sharpe')}` better=`{better}`",
                "并列 v5 不算升级。费用校正分表。",
                "",
            ]
        ),
    )
    write_md(MD["18"], f"# Round21.2 · 18 费用敏感\n\n主表 `{FEE_SCHEDULE}`。校正 `{HISTORICAL_FEE_CORRECTED}` 分表 mixed=`False`。\n{fee_rows}\n")
    return payload
