"""有 OOF 增量者才全引擎回测。主表 Round7 5bp。分年/Ex2024/Ex2025。费用敏感性分表。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_ROOT
from ..round8.robustness import block_bootstrap_delta
from ..round11.common import daily_from_nav
from ..round16.common import cagr_of
from .common import dump_json, finite_or_none, load_json, slim_obs, write_md
from .contract import (
    A00,
    A01,
    AWAITING_USER_REVIEW,
    BACKTEST_MD,
    BLOCK_DAYS,
    BOOT_N,
    FEE_SCHEDULE,
    FIXED_CODE,
    FORWARD_OOS_NOT_AVAILABLE,
    HISTORICAL_FEE_CORRECTED_SENSITIVITY,
    KEEP_BASELINE_V5,
    NESTED_CODE,
    NOT_IDENTIFIABLE,
    RECOMMEND_NEW_ALPHA_CANDIDATE,
    SPA_BOOT,
    V5_ANCHOR,
    exp_dir,
    overlayed_path,
    parity_dir,
)
from .freeze import require_frozen
from .model import MODEL_JSON
from .runner import run_full_backtest

BT_JSON = ROUND21_ROOT / "backtest" / "decision.json"
SENS_ROOT = ROUND21_ROOT / "backtest" / "fee_sensitivity"
FACTORS_DIR = ROUND21_ROOT / "factors"


def _nav_of(code: str) -> pd.DataFrame | None:
    paths = [exp_dir(code) / "nav.parquet"]
    if code == A00:
        paths.append(parity_dir("R21_V5_R16_O01") / "nav.parquet")
    for path in paths:
        if path.is_file():
            nav = pd.read_parquet(path)
            nav["trade_date"] = pd.to_datetime(nav["trade_date"])
            return nav
    return None


def _obs_of(code: str) -> dict[str, Any]:
    p = exp_dir(code) / "summary.json"
    if p.is_file():
        return load_json(p)
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
    }


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
    dump_json(dest / "sensitivity_summary.json", obs)
    return obs


def _candidate_codes() -> list[str]:
    model = load_json(MODEL_JSON)
    codes = [A00, A01, FIXED_CODE, NESTED_CODE]
    for c in model.get("increment_pass_codes") or []:
        codes.append(str(c))
    seen, out = set(), []
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def run_backtest(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and BT_JSON.is_file() and BACKTEST_MD.is_file():
        existing = load_json(BT_JSON)
        if existing.get("status") == "DONE":
            return existing
    from .model import MATRIX_JSON

    matrix = load_json(MATRIX_JSON)
    spec_map = {s["code"]: s for s in (matrix.get("specs") or [])}
    codes = _candidate_codes()
    ran = []
    for code in codes:
        print(f"[round21] FULL_BT {code}", flush=True)
        ran.append(run_full_backtest(code, spec_map.get(code), resume=resume))
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
                boot = block_bootstrap_delta(daily_from_nav(nav), daily_from_nav(a00_nav), block=BLOCK_DAYS, n=BOOT_N, seed=21)
            except Exception as exc:
                boot = {"status": NOT_IDENTIFIABLE, "reason": str(exc)}
        else:
            boot = {"status": NOT_IDENTIFIABLE, "reason": "need paired daily vs A00"} if code != A00 else None
        rec["block_bootstrap_vs_v5"] = boot
        rec["block_bootstrap_tag"] = "STATISTICAL RESAMPLING NOT OOS"
        rows.append(rec)
        if nav is not None:
            dailies[code] = daily_from_nav(nav)
        ov = exp_dir(code) / "overlayed.parquet"
        if not ov.is_file() and code == A00:
            ov = overlayed_path("R21_V5_R16_O01")
        if ov.is_file():
            try:
                fee_rows.append(_fee_sens(code, pd.read_parquet(ov)))
            except Exception as exc:
                fee_rows.append({"code": code, "status": "SKIP_FEE", "reason": str(exc)})
        fid = (spec_map.get(code) or {}).get("factor_id")
        if fid:
            dump_json(FACTORS_DIR / fid / "portfolio.json", rec)

    spa_payload: dict[str, Any] = {"status": NOT_IDENTIFIABLE, "reason": "need daily matrix vs A00"}
    if A00 in dailies and len(dailies) >= 2:
        try:
            from ..round11.stats import spa_test

            alts = {k: v for k, v in dailies.items() if k not in {A00, A01}}
            if alts:
                mat = pd.concat(alts, axis=1)
                spa_payload = {"status": "DONE", **spa_test(dailies[A00], mat, n_boot=SPA_BOOT, seed=17), "note": "NOT independent OOS"}
            else:
                spa_payload = {"status": NOT_IDENTIFIABLE, "reason": "no new-alpha daily series"}
        except Exception as exc:
            spa_payload = {"status": NOT_IDENTIFIABLE, "reason": str(exc)}

    others = [r for r in rows if r.get("code") not in {A00, A01} and r.get("sharpe") is not None]
    best = max(others, key=lambda r: float(r["sharpe"])) if others else None
    v5_sh = finite_or_none(a00.get("sharpe")) or float(V5_ANCHOR["sharpe"])
    v5_cagr = finite_or_none(a00.get("cagr")) or float(V5_ANCHOR["cagr"])
    better = False
    if best:
        sh = finite_or_none(best.get("sharpe"))
        cg = finite_or_none(best.get("cagr"))
        better = (sh is not None and sh > v5_sh + 1e-12) or (cg is not None and cg > v5_cagr + 1e-12)
    decision = RECOMMEND_NEW_ALPHA_CANDIDATE if better else KEEP_BASELINE_V5
    lock = AWAITING_USER_REVIEW if better else None
    payload = {
        "status": "DONE",
        "decision": decision,
        "lock": lock,
        "current_unchanged": "baseline_v5=R16_O01",
        "auto_promote_current": False,
        "do_not_lock_v6": True,
        "n_full_backtests": len(rows),
        "main_table": FEE_SCHEDULE,
        "main_rows": rows,
        "fee_sensitivity_table": fee_rows,
        "mixed_into_main": False,
        "spa": spa_payload,
        "forward": FORWARD_OOS_NOT_AVAILABLE,
        "best_vs_v5": None
        if best is None
        else {
            "code": best.get("code"),
            "sharpe": best.get("sharpe"),
            "cagr": best.get("cagr"),
            "max_drawdown": best.get("max_drawdown"),
            "nav_end": best.get("nav_end"),
            "v5_sharpe": v5_sh,
            "v5_cagr": v5_cagr,
            "delta_sharpe": None if best.get("sharpe") is None else float(best["sharpe"]) - v5_sh,
            "delta_cagr": None if best.get("cagr") is None else float(best["cagr"]) - v5_cagr,
            "better_than_v5": better,
        },
        "note": "MAIN = Round7 5bp no hist_hooks. PRIOR R20: no combo beat v5. 不把失败换名。",
    }
    dump_json(BT_JSON, payload)
    lines = [
        "# 11 正式回测（仅 OOF 增量 + 锚点）",
        "",
        f"决策=**{decision}** CURRENT 仍是 `baseline_v5=R16_O01`。AUTO_PROMOTE=False。",
        f"全引擎回测 N=`{len(rows)}`。主表=`{FEE_SCHEDULE}`。",
        "",
        "| code | CAGR | Sharpe | MaxDD | trades | NAV_end |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.get('code')}` | {r.get('cagr')} | {r.get('sharpe')} | "
            f"{r.get('max_drawdown')} | {r.get('completed_trades')} | {r.get('nav_end')} |"
        )
    lines += ["", "## 分年 / Ex2024 / Ex2025", ""]
    years = [str(y) for y in range(2018, 2027)]
    lines.append("| code | " + " | ".join(years) + " | Ex2024 | Ex2025 |")
    lines.append("|---|" + "---:|" * 11)
    for r in rows:
        yr = r.get("yearly_returns") or {}
        cells = " | ".join(str(yr.get(y)) for y in years)
        ex24 = (r.get("ex2024") or {}).get("sharpe")
        ex25 = (r.get("ex2025") or {}).get("sharpe")
        lines.append(f"| `{r.get('code')}` | {cells} | {ex24} | {ex25} |")
    write_md(BACKTEST_MD, "\n".join(lines))
    return payload


__all__ = ["BT_JSON", "run_backtest"]
