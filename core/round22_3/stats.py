"""分年 / 费用 / Bootstrap / 全历史 SPA。对照 R16_O01。短窗 SPA 只作附录。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..round8.robustness import block_bootstrap_delta
from ..round11.common import daily_from_nav
from .common import dump_json, finite_or_none, load_json, write_csv, write_md
from .contract import (
    BLOCK_DAYS,
    BOOT_N,
    E0_CODE,
    FORWARD_OOS_NOT_AVAILABLE,
    HISTORICAL_FEE_CORRECTED,
    MD,
    NOT_IDENTIFIABLE,
    NOT_TRIGGERED,
    REPEATED_HISTORICAL_TESTING,
    ROUND22_3_ROOT,
    SAME_OPEN_EXECUTION_OPTIMISM,
    SPA_BOOT,
    exp_dir,
)
from .freeze import require_frozen_gate

STATS_JSON = ROUND22_3_ROOT / "stats" / "stats.json"


def _nav(code: str) -> pd.DataFrame | None:
    path = exp_dir(code) / "nav.parquet"
    if not path.is_file() or path.stat().st_size < 8:
        return None
    nav = pd.read_parquet(path)
    if "trade_date" not in nav.columns or "total_asset" not in nav.columns:
        return None
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    return nav


def _window(nav: pd.DataFrame, *, years: tuple[int, ...] | None = None, drop: tuple[int, ...] = ()) -> dict[str, Any]:
    daily = daily_from_nav(nav)
    idx_year = pd.to_datetime(daily.index).year
    if years:
        daily = daily.loc[idx_year.isin(years)].copy()
    if drop:
        daily = daily.loc[~pd.to_datetime(daily.index).year.isin(drop)].copy()
    if len(daily) < 10:
        return {"status": "TOO_SHORT", "n": int(len(daily)), "sharpe": None, "cagr": None}
    wealth = (1.0 + daily.astype(float)).cumprod()
    span = max((pd.Timestamp(daily.index.max()) - pd.Timestamp(daily.index.min())).days, 1)
    cagr = float(wealth.iloc[-1] ** (365.25 / span) - 1.0)
    vol = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / vol * np.sqrt(252.0)) if vol > 0 else None
    peak = wealth.cummax()
    dd = float((wealth / peak - 1.0).min()) if len(wealth) else None
    return {"cagr": finite_or_none(cagr), "sharpe": finite_or_none(sharpe), "max_drawdown": finite_or_none(dd), "n": int(len(daily)), "status": "OK"}


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and STATS_JSON.is_file() and MD["17"].is_file():
        existing = load_json(STATS_JSON)
        if existing.get("status") in {"DONE", "PARTIAL", "NOT_RUN"}:
            return existing
    sleeve = load_json(ROUND22_3_ROOT / "backtest" / "sleeve.json")
    rows = sleeve.get("rows") or []
    codes = [r.get("code") for r in rows if r.get("code")]
    if len(codes) < 2:
        payload = {
            "status": "NOT_RUN",
            "reason": "no candidate sleeves",
            "spa": {"status": NOT_IDENTIFIABLE, "n_alternatives": 0},
            "seed": {"status": NOT_TRIGGERED},
            "forward": FORWARD_OOS_NOT_AVAILABLE,
            "testing_label": REPEATED_HISTORICAL_TESTING,
        }
        dump_json(STATS_JSON, payload)
        dump_json(ROUND22_3_ROOT / "seed" / "seed_robustness.json", {"status": NOT_TRIGGERED, "reason": "经济门未过或无候选"})
        write_md(MD["17"], f"# Round22.3 · 17 分年/费用\n\n未跑。`{payload['reason']}`。\n")
        write_md(MD["18"], f"# Round22.3 · 18 Bootstrap/SPA\n\nSPA `{NOT_IDENTIFIABLE}`。\n")
        return payload
    yearly = []
    for code in codes:
        nav = _nav(code)
        if nav is None:
            continue
        rec = {"code": code}
        rec["all"] = _window(nav)
        rec["y2022"] = _window(nav, years=(2022,))
        rec["ex2024"] = _window(nav, drop=(2024,))
        rec["ex2025"] = _window(nav, drop=(2025,))
        rec["y2018_2020"] = _window(nav, years=(2018, 2019, 2020))
        rec["y2021_2023"] = _window(nav, years=(2021, 2022, 2023))
        rec["y2024_2025"] = _window(nav, years=(2024, 2025))
        yearly.append(rec)
    write_csv(ROUND22_3_ROOT / "stats" / "yearly.csv", yearly)
    e0 = _nav(E0_CODE)
    boots = []
    alts = []
    alt_codes = []
    if e0 is not None:
        base_daily = daily_from_nav(e0)
        for code in codes:
            if code == E0_CODE:
                continue
            nav = _nav(code)
            if nav is None:
                continue
            daily = daily_from_nav(nav)
            aligned = pd.concat([base_daily.rename("e0"), daily.rename("c")], axis=1, join="inner").dropna()
            if len(aligned) < 40:
                boots.append({"code": code, "status": NOT_IDENTIFIABLE, "reason": "aligned_days<40"})
                continue
            delta = aligned["c"].to_numpy(dtype=float) - aligned["e0"].to_numpy(dtype=float)
            boot = block_bootstrap_delta(aligned["c"], aligned["e0"], block=BLOCK_DAYS, n=BOOT_N)
            rec = {
                "code": code,
                "status": boot.get("status") if isinstance(boot, dict) else "OK",
                "n": BOOT_N,
                "block": BLOCK_DAYS,
                "observed_mean_delta": float(np.mean(delta)),
                "n_aligned": int(len(aligned)),
            }
            if isinstance(boot, dict):
                rec.update({k: boot.get(k) for k in ("median_ann_delta", "p_delta_sharpe_gt0", "p_delta_ret_gt0", "p05_ann_delta", "p95_ann_delta")})
            boots.append(rec)
            alts.append(aligned["c"].to_numpy(dtype=float))
            alt_codes.append(code)
    spa = {"status": NOT_IDENTIFIABLE, "n_alternatives": len(alts), "n_boot": SPA_BOOT, "e0_excluded": True, "alt_codes": alt_codes}
    if e0 is not None and len(alts) >= 2:
        try:
            from ..round11.stats import spa_test

            base_daily = daily_from_nav(e0)
            alt_df = pd.DataFrame({code: pd.Series(a, index=base_daily.index[: len(a)]) for code, a in zip(alt_codes, alts)})
            if alt_df.shape[1] >= 2 and len(alt_df.dropna()) >= 80:
                spa_res = spa_test(base_daily, alt_df, n_boot=SPA_BOOT)
                spa = {"n_alternatives": len(alts), "n_boot": SPA_BOOT, "e0_excluded": True, "alt_codes": alt_codes, **({} if not isinstance(spa_res, dict) else spa_res)}
            else:
                spa["reason"] = "incomplete_aligned_matrix"
        except Exception as exc:
            spa["reason"] = type(exc).__name__
    dump_json(ROUND22_3_ROOT / "bootstrap" / "bootstrap.json", {"rows": boots, "n": BOOT_N, "block": BLOCK_DAYS})
    dump_json(ROUND22_3_ROOT / "spa" / "spa.json", spa)
    dump_json(ROUND22_3_ROOT / "seed" / "seed_robustness.json", {"status": NOT_TRIGGERED, "reason": "经济门未过，不另跑种子扫描"})
    dump_json(ROUND22_3_ROOT / "coverage" / "forward.json", {"status": FORWARD_OOS_NOT_AVAILABLE, "independent_forward": False})
    c4e1 = next((b for b in boots if b.get("code") == "R223_E1_r22_c4_flow_persist_5"), None)
    dump_json(ROUND22_3_ROOT / "c4" / "c4_e1_paired_bootstrap.json", c4e1 or {"status": NOT_IDENTIFIABLE})
    ledger = load_json(ROUND22_3_ROOT / "ledger" / "ledger.json")
    top_map = {t.get("code"): t for t in (ledger.get("tops") or [])}
    material = []
    for rec in rows:
        if rec.get("code") == E0_CODE:
            continue
        flags = dict(rec.get("flags") or {})
        top = top_map.get(rec.get("code")) or {}
        if top.get("top_net_ok") is True:
            flags["top_changed_ok"] = True
        elif top.get("was_missing_columns"):
            flags["top_changed_ok"] = False
        yrec = next((y for y in yearly if y.get("code") == rec.get("code")), None)
        y0 = next((y for y in yearly if y.get("code") == E0_CODE), None)
        not_only = True
        if yrec and y0:
            ex24 = (yrec.get("ex2024") or {}).get("sharpe")
            ex25 = (yrec.get("ex2025") or {}).get("sharpe")
            b24 = (y0.get("ex2024") or {}).get("sharpe")
            b25 = (y0.get("ex2025") or {}).get("sharpe")
            only_recent = True
            if ex24 is not None and b24 is not None and float(ex24) > float(b24):
                only_recent = False
            if ex25 is not None and b25 is not None and float(ex25) > float(b25):
                only_recent = False
            not_only = not only_recent
        flags["not_only_2024_or_2025"] = bool(not_only)
        flags["increment_from_new_factor"] = "E4" not in str(rec.get("role") or "") and "E6" not in str(rec.get("role") or "")
        econ = all(flags.get(k) for k in ("delta_cagr_ok", "delta_sharpe_ok", "nav_rel_ok", "top_changed_ok", "maxdd_ok", "fee_direction_ok", "not_only_2024_or_2025")) and flags.get("increment_from_new_factor")
        material.append({**rec, "flags": flags, "economically_material": bool(econ), "top_net": top, "compare_to": "R16_O01"})
    payload = {
        "status": "DONE",
        "yearly": yearly,
        "bootstrap": boots,
        "spa": spa,
        "seed": {"status": NOT_TRIGGERED},
        "forward": FORWARD_OOS_NOT_AVAILABLE,
        "execution": SAME_OPEN_EXECUTION_OPTIMISM,
        "fee_label": HISTORICAL_FEE_CORRECTED,
        "testing_label": REPEATED_HISTORICAL_TESTING,
        "material": material,
        "compare_to": "R16_O01",
        "compare_mixed_r222_e0": False,
    }
    dump_json(STATS_JSON, payload)
    write_md(MD["17"], f"# Round22.3 · 17 分年/费用\n\n窗口 2022 / Ex2024 / Ex2025 已写 `stats/yearly.csv`。费用 `{HISTORICAL_FEE_CORRECTED}`。执行 `{SAME_OPEN_EXECUTION_OPTIMISM}`。\n")
    write_md(MD["18"], f"# Round22.3 · 18 Bootstrap/SPA\n\n全历史 SPA status=`{spa.get('status')}` n_alternatives=`{spa.get('n_alternatives')}` e0_excluded=`True`。Bootstrap N={BOOT_N} block={BLOCK_DAYS}。\n")
    return payload
