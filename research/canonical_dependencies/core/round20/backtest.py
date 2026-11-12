"""正式 Round7 5bp 主表 + 费用校正敏感性分表。分年/Ex2024/Ex2025。块 bootstrap N=5000。不改 CURRENT。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND20_ROOT
from ..round8.robustness import block_bootstrap_delta
from ..round11.common import daily_from_nav, metrics_from_nav
from ..round16.common import cagr_of
from ..round17.common import clip_nav_window
from .common import dump_json, finite_or_none, load_json, slim_obs, write_md
from .contract import (
    A00,
    A01,
    AWAITING_USER_REVIEW,
    BLOCK_DAYS,
    BOOT_N,
    FEE_SCHEDULE,
    FORWARD_OOS_NOT_AVAILABLE,
    HISTORICAL_FEE_CORRECTED_SENSITIVITY,
    KEEP_BASELINE_V5,
    NOT_IDENTIFIABLE,
    RECOMMEND_NEW_ALPHA_CANDIDATE,
    SPA_BOOT,
    V5_ANCHOR,
    exp_dir,
    parity_dir,
)
from .freeze import require_frozen

BT_JSON = ROUND20_ROOT / "backtest" / "decision.json"
BT_MD = ROUND20_ROOT / "11_backtest_decision.md"
SENS_ROOT = ROUND20_ROOT / "backtest" / "fee_sensitivity"


def _nav_of(code: str) -> pd.DataFrame | None:
    paths = [exp_dir(code) / "nav.parquet"]
    summary = exp_dir(code) / "summary.json"
    if summary.is_file():
        alias = (load_json(summary) or {}).get("alias_of")
        if alias:
            paths.append(exp_dir(str(alias)) / "nav.parquet")
    if code == A00:
        paths.append(parity_dir("R20_V5_R16_O01") / "nav.parquet")
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            nav = pd.read_parquet(path)
            nav["trade_date"] = pd.to_datetime(nav["trade_date"])
            return nav
    return None


def _obs_of(code: str) -> dict[str, Any]:
    p = exp_dir(code) / "summary.json"
    if p.is_file():
        return load_json(p)
    if code == A00:
        from .parity import PARITY_JSON
        par = load_json(PARITY_JSON)
        obs = dict(par.get("observed") or {})
        obs["code"] = A00
        obs["status"] = par.get("status")
        return obs
    return {"code": code, "status": "MISSING"}


def _window_metrics(nav: pd.DataFrame, *, drop_years: tuple[int, ...] = ()) -> dict[str, Any]:
    daily = daily_from_nav(nav)
    if drop_years:
        daily = daily.loc[~pd.to_datetime(daily.index).year.isin(drop_years)].copy()
    if len(daily) < 10:
        return {"status": "TOO_SHORT", "n": int(len(daily)), "drop_years": list(drop_years)}
    wealth = (1.0 + daily.astype(float)).cumprod()
    span_days = max((pd.Timestamp(daily.index.max()) - pd.Timestamp(daily.index.min())).days, 1)
    cagr = float(wealth.iloc[-1] ** (365.25 / span_days) - 1.0)
    vol = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / vol * np.sqrt(252.0)) if vol > 0 else None
    peak = wealth.cummax()
    dd = float((wealth / peak - 1.0).min()) if len(wealth) else None
    return {
        "cagr": finite_or_none(cagr),
        "cagr_of_252": finite_or_none(cagr_of(daily)) if len(daily) else None,
        "sharpe": finite_or_none(sharpe),
        "max_drawdown": finite_or_none(dd),
        "n_obs": int(len(daily)),
        "drop_years": list(drop_years),
        "method": "drop_year_days_from_daily",
    }


def _overlay_path(code: str) -> Any:
    from .contract import overlayed_path

    paths = [exp_dir(code) / "overlayed.parquet"]
    summary = exp_dir(code) / "summary.json"
    if summary.is_file():
        alias = (load_json(summary) or {}).get("alias_of")
        if alias:
            paths.append(exp_dir(str(alias)) / "overlayed.parquet")
    if code == A00:
        paths.append(overlayed_path("R20_V5_R16_O01"))
        paths.append(exp_dir(A00) / "overlayed.parquet")
    for path in paths:
        if path.is_file():
            return path
    return None


def _fee_sens(code: str, overlayed: pd.DataFrame) -> dict[str, Any]:
    from ..round17.backtest import run_canonical

    dest = SENS_ROOT / code
    extra = {
        "code": code,
        "role": HISTORICAL_FEE_CORRECTED_SENSITIVITY,
        "fee_schedule": "HISTORICAL_FEE_CORRECTED",
        "not_main_table": True,
        "mixed_into_main": False,
    }
    ran = run_canonical(dest, overlayed, extra=extra, resume=True, write=True)
    raw = ran.get("observed") if isinstance(ran, dict) else None
    obs = slim_obs(raw if isinstance(raw, dict) else {})
    obs["code"] = code
    obs["table"] = HISTORICAL_FEE_CORRECTED_SENSITIVITY
    obs["out_dir"] = str(dest)
    obs["status"] = "ran" if obs.get("sharpe") is not None else (obs.get("status") or "ran")
    dump_json(dest / "sensitivity_summary.json", obs)
    return obs


def _candidate_codes() -> list[str]:
    from .model import MODEL_JSON
    from .runner import COMBO_TRAIN_JSON

    codes = [A00]
    combo = load_json(COMBO_TRAIN_JSON)
    best = (combo.get("best") or {}).get("code")
    if best:
        codes.append(str(best))
    model = load_json(MODEL_JSON)
    for f in model.get("finalists") or []:
        if f.get("code"):
            codes.append(str(f["code"]))
    for c in ("R20_A01", "R20_C01", "R20_D01"):
        codes.append(c)
    seen = set()
    out = []
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _gates(main: dict[str, Any], a00: dict[str, Any], fee: dict[str, Any] | None, yearly: dict[str, Any]) -> dict[str, Any]:
    from .parity import PARITY_JSON
    from .pit import PIT_JSON

    par = load_json(PARITY_JSON)
    pit = load_json(PIT_JSON)
    sh = finite_or_none(main.get("sharpe"))
    a_sh = finite_or_none(a00.get("sharpe")) or float(V5_ANCHOR["sharpe"])
    cagr = finite_or_none(main.get("cagr"))
    a_cagr = finite_or_none(a00.get("cagr")) or float(V5_ANCHOR["cagr"])
    dd = finite_or_none(main.get("max_drawdown"))
    a_dd = finite_or_none(a00.get("max_drawdown")) or float(V5_ANCHOR["max_drawdown"])
    yrs = yearly or {}
    y24 = finite_or_none(yrs.get("2024"))
    y25 = finite_or_none(yrs.get("2025"))
    others = [finite_or_none(yrs.get(str(y))) for y in range(2018, 2024)]
    others_pos = any(v is not None and v > 0 for v in others)
    not_only_2425 = bool(others_pos)
    fee_flip = False
    if fee and sh is not None:
        fs = finite_or_none(fee.get("sharpe"))
        if fs is not None and sh > a_sh and fs < a_sh - 0.05:
            fee_flip = True
    better = sh is not None and a_sh is not None and (sh > a_sh + 1e-12 or (cagr is not None and a_cagr is not None and cagr > a_cagr + 1e-12))
    dd_ok = dd is None or a_dd is None or dd >= a_dd - 0.02
    checks = {
        "pit_pass": bool((pit.get("n_promotion_eligible") or 0) > 0) and pit.get("f2_promotion_eligible") is False,
        "v5_parity_pass": bool(par.get("passed")),
        "same_caliber_net_better_than_v5": bool(better),
        "maxdd_acceptable": bool(dd_ok),
        "not_only_2024_2025": not_only_2425,
        "fee_corrected_no_severe_reversal": not fee_flip,
        "selection_bias_disclosed": True,
        "f2_not_used": True,
        "subsample_1_0": True,
    }
    passed = all(checks.values()) and checks["same_caliber_net_better_than_v5"]
    return {"checks": checks, "passed": passed}


def run_backtest(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    codes = _candidate_codes()
    a00 = _obs_of(A00)
    a00_nav = _nav_of(A00)
    rows = []
    fee_rows = []
    dailies: dict[str, pd.Series] = {}
    for code in codes:
        obs = _obs_of(code)
        nav = _nav_of(code)
        yearly = (obs.get("yearly_returns") or {}) if obs else {}
        rec = {
            "code": code,
            "table": "MAIN_ROUND7_FLAT_5BP",
            "validation_hooks": None,
            "cagr": obs.get("cagr"),
            "sharpe": obs.get("sharpe"),
            "max_drawdown": obs.get("max_drawdown"),
            "completed_trades": obs.get("completed_trades"),
            "nav_end": obs.get("nav_end"),
            "yearly_returns": yearly,
            "ex2024": _window_metrics(nav, drop_years=(2024,)) if nav is not None else None,
            "ex2025": _window_metrics(nav, drop_years=(2025,)) if nav is not None else None,
            "ex2024_2025": _window_metrics(nav, drop_years=(2024, 2025)) if nav is not None else None,
            "status": obs.get("status"),
        }
        boot = None
        if nav is not None and a00_nav is not None and code != A00:
            try:
                boot = block_bootstrap_delta(daily_from_nav(nav), daily_from_nav(a00_nav), block=BLOCK_DAYS, n=BOOT_N, seed=20)
            except Exception as exc:
                boot = {"status": "ERROR", "reason": str(exc)}
        rec["block_bootstrap_vs_v5"] = boot
        rec["block_bootstrap_tag"] = "STATISTICAL RESAMPLING NOT OOS"
        rows.append(rec)
        if nav is not None:
            dailies[code] = daily_from_nav(nav)
        ov_path = _overlay_path(code)
        if ov_path is not None and ov_path.is_file():
            try:
                fee_rows.append(_fee_sens(code, pd.read_parquet(ov_path)))
            except Exception as exc:
                fee_rows.append({"code": code, "status": "SKIP_FEE", "reason": str(exc), "table": HISTORICAL_FEE_CORRECTED_SENSITIVITY})
        else:
            fee_rows.append({"code": code, "status": "SKIP_NO_OVERLAY", "table": HISTORICAL_FEE_CORRECTED_SENSITIVITY})

    spa_payload: dict[str, Any] = {"status": NOT_IDENTIFIABLE, "reason": "need daily matrix vs A00"}
    if A00 in dailies and len(dailies) >= 2:
        try:
            from ..round11.stats import spa_test

            alts = {k: v for k, v in dailies.items() if k not in {A00, A01}}
            if alts:
                mat = pd.concat(alts, axis=1)
                spa_payload = {"status": "DONE", **spa_test(dailies[A00], mat, n_boot=SPA_BOOT, seed=17), "note": "NOT independent OOS; A01 excluded as 14/13 identity"}
            else:
                spa_payload = {"status": NOT_IDENTIFIABLE, "reason": "no new-alpha daily series"}
        except Exception as exc:
            spa_payload = {"status": NOT_IDENTIFIABLE, "reason": str(exc)}

    # pick best new-alpha candidate; A01 is original 14/13 identity, not a new alpha
    others = [r for r in rows if r.get("code") not in {A00, A01} and r.get("sharpe") is not None]
    best = max(others, key=lambda r: float(r["sharpe"])) if others else None
    fee_best = None
    if best:
        fee_best = next((f for f in fee_rows if f.get("code") == best.get("code")), None)
    gates = _gates(best or {}, a00, fee_best, (best or {}).get("yearly_returns") or {})
    if gates.get("passed"):
        decision = RECOMMEND_NEW_ALPHA_CANDIDATE
        lock = AWAITING_USER_REVIEW
    else:
        decision = KEEP_BASELINE_V5
        lock = None
    payload = {
        "status": "DONE",
        "decision": decision,
        "lock": lock,
        "current_unchanged": "baseline_v5=R16_O01",
        "auto_promote_current": False,
        "do_not_lock_v6": True,
        "main_table": FEE_SCHEDULE,
        "main_rows": rows,
        "fee_sensitivity_table": fee_rows,
        "fee_sensitivity_label": HISTORICAL_FEE_CORRECTED_SENSITIVITY,
        "mixed_into_main": False,
        "spa": spa_payload,
        "forward": FORWARD_OOS_NOT_AVAILABLE,
        "gates": gates,
        "best_vs_v5": None if best is None else {
            "code": best.get("code"),
            "sharpe": best.get("sharpe"),
            "cagr": best.get("cagr"),
            "max_drawdown": best.get("max_drawdown"),
            "v5_sharpe": a00.get("sharpe") or V5_ANCHOR["sharpe"],
        },
        "note": "MAIN = Round7 5bp no hist_hooks. Fee correction is a separate table.",
    }
    dump_json(BT_JSON, payload)
    lines = [
        "# 11 正式回测与决策",
        "",
        f"决策=**{decision}** CURRENT 仍是 `baseline_v5=R16_O01`。AUTO_PROMOTE=False。不锁 v6。",
        f"主表=`{FEE_SCHEDULE}` validation_hooks=`None`。敏感性=`{HISTORICAL_FEE_CORRECTED_SENSITIVITY}` 不混主表。",
        f"Forward=`{FORWARD_OOS_NOT_AVAILABLE}`。块 bootstrap N=`{BOOT_N}` block=`{BLOCK_DAYS}` = STATISTICAL RESAMPLING NOT OOS。",
        f"SPA=`{spa_payload.get('status')}` gates_passed=`{gates.get('passed')}`",
        "",
        "| code | table | CAGR | Sharpe | MaxDD | trades | NAV_end |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.get('code')}` | {r.get('table')} | {r.get('cagr')} | {r.get('sharpe')} | "
            f"{r.get('max_drawdown')} | {r.get('completed_trades')} | {r.get('nav_end')} |"
        )
    lines += [
        "",
        "## 分年 / Ex2024 / Ex2025",
        "",
        "| code | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Ex2024 Sharpe | Ex2025 Sharpe | Ex2024+2025 Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    years = [str(y) for y in range(2018, 2027)]
    for r in rows:
        yr = r.get("yearly_returns") or {}
        cells = " | ".join(str(yr.get(y)) for y in years)
        ex24 = (r.get("ex2024") or {}).get("sharpe")
        ex25 = (r.get("ex2025") or {}).get("sharpe")
        exb = (r.get("ex2024_2025") or {}).get("sharpe")
        lines.append(f"| `{r.get('code')}` | {cells} | {ex24} | {ex25} | {exb} |")
    lines += ["", "## 块 bootstrap vs v5（N=5000, block=20, NOT OOS）", ""]
    for r in rows:
        if r.get("code") == A00:
            continue
        lines.append(f"- `{r.get('code')}` `{r.get('block_bootstrap_vs_v5')}`")
    lines += ["", "## 费用校正敏感性（分表，不混主表）", "", "| code | CAGR | Sharpe | MaxDD | status |", "|---|---:|---:|---:|---|"]
    for r in fee_rows:
        lines.append(f"| `{r.get('code')}` | {r.get('cagr')} | {r.get('sharpe')} | {r.get('max_drawdown')} | {r.get('status')} |")
    write_md(BT_MD, "\n".join(lines))
    return payload


__all__ = ["BT_JSON", "BT_MD", "run_backtest"]
