"""全袖套 Round7 5bp 回测。费用敏感分表。不改 O01/B03/90-10。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .common import dump_json, finite_or_none, load_json, slim_obs, write_md
from .engine import run_v5_fused_backtest
from .contract import (
    AWAITING_USER_REVIEW,
    BACKTEST_MD,
    E0_CODE,
    FEE_SCHEDULE,
    HISTORICAL_FEE_CORRECTED,
    KEEP_BASELINE_V5,
    RECOMMEND_NEW_ALPHA_CANDIDATE,
    V5_ANCHOR,
    exp_dir,
)
from .train import MATRIX_JSON, MODEL_JSON, train_one

from ..paths import ROUND21_1_ROOT

BT_JSON = ROUND21_1_ROOT / "backtest" / "decision.json"
SENS_ROOT = ROUND21_1_ROOT / "backtest" / "fee_sensitivity"


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
    five = pd.read_parquet(five_p)
    ten = pd.read_parquet(ten_p)
    extra = {
        "code": code,
        "role": (spec or {}).get("role"),
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
        "round": "21.1",
    }
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
        "five_extras": oof.get("five_extras"),
        "ten_extras": oof.get("ten_extras"),
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
        "looked_at_combo_nav": True,
    }
    dump_json(dest / "summary.json", payload)
    return payload


def _fee_sens(code: str) -> dict[str, Any]:
    ov = exp_dir(code) / "overlayed.parquet"
    if not ov.is_file():
        return {"code": code, "status": "SKIP_FEE", "reason": "no overlayed"}
    from ..round17.backtest import run_canonical

    dest = SENS_ROOT / code
    extra = {
        "code": code,
        "role": HISTORICAL_FEE_CORRECTED,
        "fee_schedule": HISTORICAL_FEE_CORRECTED,
        "not_main_table": True,
        "mixed_into_main": False,
    }
    ran = run_canonical(dest, pd.read_parquet(ov), extra=extra, resume=True, write=True)
    raw = ran.get("observed") if isinstance(ran, dict) else None
    obs = slim_obs(raw if isinstance(raw, dict) else {})
    obs["code"] = code
    obs["table"] = HISTORICAL_FEE_CORRECTED
    dump_json(dest / "sensitivity_summary.json", obs)
    return obs


def run_backtest(*, resume: bool = True, jobs: int = 1, **_: Any) -> dict[str, Any]:
    if resume and BT_JSON.is_file() and BACKTEST_MD.is_file():
        existing = load_json(BT_JSON)
        if existing.get("status") == "DONE":
            return existing
    from .train import backfill_artifacts

    backfill_artifacts()
    matrix = load_json(MATRIX_JSON)
    model = load_json(MODEL_JSON)
    increment_pass = {str(c) for c in (model.get("increment_pass_codes") or [])}
    if not increment_pass:
        increment_pass = {
            str(r.get("code"))
            for r in (model.get("rows") or [])
            if (r.get("increment") or {}).get("passed")
        }
    spec_map = {s["code"]: s for s in (matrix.get("specs") or [])}
    # 正式候选：E0 + 全部 formal specs（四路 + 条件 E4/E5）
    codes = [s["code"] for s in (matrix.get("specs") or []) if s.get("formal")]
    if E0_CODE not in codes:
        codes = [E0_CODE, *codes]
    ran = []

    def _one(code: str) -> dict[str, Any]:
        print(f"[round21.1] FULL_BT {code}", flush=True)
        return run_full_backtest(code, spec_map.get(code), resume=resume)

    if int(jobs) <= 1:
        for code in codes:
            ran.append(_one(code))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=min(int(jobs), 4)) as pool:
            futs = {pool.submit(_one, code): code for code in codes}
            got = {}
            for fut in as_completed(futs):
                got[futs[fut]] = fut.result()
            ran = [got[c] for c in codes if c in got]
    rows = []
    fee_rows = []
    for code in codes:
        obs = load_json(exp_dir(code) / "summary.json")
        rec = {
            "code": code,
            "table": "MAIN_ROUND7_FLAT_5BP",
            "role": obs.get("role"),
            "factor_id": (spec_map.get(code) or {}).get("factor_id"),
            "cagr": obs.get("cagr"),
            "sharpe": obs.get("sharpe"),
            "max_drawdown": obs.get("max_drawdown"),
            "completed_trades": obs.get("completed_trades"),
            "nav_end": obs.get("nav_end"),
            "yearly_returns": obs.get("yearly_returns") or {},
            "status": obs.get("status"),
        }
        rows.append(rec)
        fid = rec.get("factor_id")
        if fid:
            dump_json(ROUND21_1_ROOT / "factors" / fid / "portfolio.json", rec)
        # 终候选另跑 HISTORICAL_FEE_CORRECTED，不混主表。E0 作对照。
        if code == E0_CODE or code in increment_pass:
            try:
                fee_rows.append(_fee_sens(code))
            except Exception as exc:
                fee_rows.append({"code": code, "status": "SKIP_FEE", "reason": str(exc)})
        else:
            fee_rows.append({"code": code, "status": "NOT_TRIGGERED", "reason": "not a finalist; fee table is for 终候选 only"})

    others = [r for r in rows if r.get("code") != E0_CODE and r.get("sharpe") is not None]
    best = max(others, key=lambda r: float(r["sharpe"])) if others else None
    v5_sh = float(V5_ANCHOR["sharpe"])
    v5_cagr = float(V5_ANCHOR["cagr"])
    better = False
    if best:
        sh, cg = finite_or_none(best.get("sharpe")), finite_or_none(best.get("cagr"))
        better = (sh is not None and sh > v5_sh + 1e-12) or (cg is not None and cg > v5_cagr + 1e-12)
    # 晋级还要求 E3>E0 且新因子有贡献；最终在 report 收口
    decision = RECOMMEND_NEW_ALPHA_CANDIDATE if better else KEEP_BASELINE_V5
    payload = {
        "status": "DONE",
        "decision": decision,
        "lock": AWAITING_USER_REVIEW if better else None,
        "current_unchanged": "baseline_v5=R16_O01",
        "auto_promote_current": False,
        "do_not_lock_v6": True,
        "n_full_backtests": len(rows),
        "main_table": FEE_SCHEDULE,
        "main_rows": rows,
        "fee_sensitivity_table": fee_rows,
        "mixed_into_main": False,
        "best_vs_v5": None
        if best is None
        else {
            "code": best.get("code"),
            "sharpe": best.get("sharpe"),
            "cagr": best.get("cagr"),
            "max_drawdown": best.get("max_drawdown"),
            "nav_end": best.get("nav_end"),
            "completed_trades": best.get("completed_trades"),
            "v5_sharpe": v5_sh,
            "v5_cagr": v5_cagr,
            "delta_sharpe": None if best.get("sharpe") is None else float(best["sharpe"]) - v5_sh,
            "delta_cagr": None if best.get("cagr") is None else float(best["cagr"]) - v5_cagr,
            "better_than_v5": better,
        },
    }
    dump_json(BT_JSON, payload)
    lines = [
        "# Round21.1 · 11 正式回测",
        "",
        f"决策=**{decision}** CURRENT 仍是 `baseline_v5=R16_O01`。AUTO_PROMOTE=False。不锁 v6。",
        f"全引擎 N=`{len(rows)}`。主表=`{FEE_SCHEDULE}`。费用敏感另表。",
        "",
        "| code | CAGR | Sharpe | MaxDD | trades | NAV_end |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.get('code')}` | {r.get('cagr')} | {r.get('sharpe')} | "
            f"{r.get('max_drawdown')} | {r.get('completed_trades')} | {r.get('nav_end')} |"
        )
    write_md(BACKTEST_MD, "\n".join(lines))
    return payload
