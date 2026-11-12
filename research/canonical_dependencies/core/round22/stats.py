"""分年、费用方向、block bootstrap N=5000、SPA 数值字段。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND22_ROOT
from ..round8.robustness import block_bootstrap_delta
from ..round11.common import daily_from_nav
from .common import dump_json, finite_or_none, load_json, write_csv, write_md
from .contract import (
    BLOCK_DAYS,
    BOOT_N,
    E0_CODE,
    ECONOMIC_MATERIALITY,
    FORWARD_OOS_NOT_AVAILABLE,
    MD,
    NOT_IDENTIFIABLE,
    NOT_TRIGGERED,
    REPEATED_HISTORICAL_TESTING,
    SPA_BOOT,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    exp_dir,
)

STATS_JSON = ROUND22_ROOT / "stats" / "stats.json"
WINDOWS = {
    "y2018_2020": (2018, 2019, 2020),
    "y2021_2023": (2021, 2022, 2023),
    "y2024_2025": (2024, 2025),
    "y2026_ytd": (2026,),
}


def _nav(code: str) -> pd.DataFrame | None:
    path = exp_dir(code) / "nav.parquet"
    if not path.is_file() or path.stat().st_size < 8:
        alt = ROUND22_ROOT / "parity" / "R22_V5_R16_O01" / "nav.parquet"
        if code == E0_CODE and alt.is_file() and alt.stat().st_size > 8:
            path = alt
        else:
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


def _trade_keys(code: str) -> set[tuple] | None:
    path = exp_dir(code) / "trades.parquet"
    if not path.is_file() or path.stat().st_size < 8:
        if code == E0_CODE:
            path = ROUND22_ROOT / "parity" / "R22_V5_R16_O01" / "trades.parquet"
    if not path.is_file() or path.stat().st_size < 8:
        return None
    frame = pd.read_parquet(path)
    date_col = next((c for c in ("trade_date", "signal_date", "entry_date", "open_date") if c in frame.columns), None)
    id_col = next((c for c in ("instrument_id", "order_book_id", "symbol") if c in frame.columns), None)
    if date_col is None or id_col is None:
        return None
    work = frame
    rank_col = next((c for c in ("rank", "slot", "pick_rank") if c in frame.columns), None)
    if rank_col is not None:
        work = frame.loc[pd.to_numeric(frame[rank_col], errors="coerce").isin([1, 2])]
    keys = set()
    for day, iid in zip(pd.to_datetime(work[date_col]), work[id_col].astype(str)):
        keys.add((str(pd.Timestamp(day).date()), iid))
    return keys


def _net(code: str) -> float | None:
    path = exp_dir(code) / "trades.parquet"
    if not path.is_file() or path.stat().st_size < 8:
        return None
    frame = pd.read_parquet(path)
    for col in ("label_e2_net", "label_exit_rule_net", "net_ret", "pnl_pct"):
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            if len(series):
                return float(series.mean())
    return None


def _material(full: dict[str, Any], *, top_changed: bool | None, fee_same: bool | None, ex2024: dict[str, Any], ex2025: dict[str, Any], base_ex2024: dict[str, Any], base_ex2025: dict[str, Any], top_net_ok: bool | None) -> dict[str, Any]:
    gate = ECONOMIC_MATERIALITY
    dc = None if full.get("cagr") is None else float(full["cagr"]) - float(V5_ANCHOR["cagr"])
    ds = None if full.get("sharpe") is None else float(full["sharpe"]) - float(V5_ANCHOR["sharpe"])
    nav = full.get("nav_end")
    dn = None if nav is None else abs(float(nav) - float(V5_NAV_END_EXACT)) / float(V5_NAV_END_EXACT)
    dd = None if full.get("max_drawdown") is None else float(full["max_drawdown"]) - float(V5_ANCHOR["max_drawdown"])
    d_ex4 = None
    d_ex5 = None
    if ex2024.get("sharpe") is not None and base_ex2024.get("sharpe") is not None:
        d_ex4 = float(ex2024["sharpe"]) - float(base_ex2024["sharpe"])
    if ex2025.get("sharpe") is not None and base_ex2025.get("sharpe") is not None:
        d_ex5 = float(ex2025["sharpe"]) - float(base_ex2025["sharpe"])
    not_only = d_ex4 is not None and d_ex5 is not None and d_ex4 > 0 and d_ex5 > 0
    flags = {
        "delta_cagr_ok": dc is not None and dc >= float(gate["delta_cagr_min_abs"]),
        "delta_sharpe_ok": ds is not None and ds >= float(gate["delta_sharpe_min"]),
        "nav_rel_ok": dn is not None and dn >= float(gate["nav_rel_min"]),
        "top_changed_ok": bool(top_changed),
        "maxdd_ok": dd is not None and dd >= -float(gate["maxdd_worse_max_abs"]),
        "fee_direction_ok": bool(fee_same),
        "not_only_2024_or_2025": bool(not_only),
        "top_net_ok": bool(top_net_ok),
    }
    return {
        "delta_cagr": dc,
        "delta_sharpe": ds,
        "nav_rel": dn,
        "delta_maxdd": dd,
        "delta_sharpe_ex2024": d_ex4,
        "delta_sharpe_ex2025": d_ex5,
        "flags": flags,
        "economically_material": all(flags.values()),
        "gate_id": gate["id"],
    }


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and STATS_JSON.is_file() and MD["19"].is_file():
        existing = load_json(STATS_JSON)
        if existing.get("status") == "DONE":
            return existing
    bt = load_json(ROUND22_ROOT / "backtest" / "sleeve.json")
    rows_bt = [r for r in (bt.get("rows") or []) if r.get("status") == "ran" and r.get("code")]
    codes = [str(r["code"]) for r in rows_bt]
    if E0_CODE not in codes:
        codes = [E0_CODE, *codes]
    base_nav = _nav(E0_CODE)
    base_full = _window(base_nav) if base_nav is not None else {}
    base_keys = _trade_keys(E0_CODE)
    base_net = _net(E0_CODE)
    fee_rows = []
    fee_path = ROUND22_ROOT / "backtest" / "fee_both_calibers.csv"
    if fee_path.is_file():
        fee_rows = pd.read_csv(fee_path).to_dict(orient="records")
    fee_map = {str(r.get("code")): r for r in fee_rows}
    base_fee_s = finite_or_none((fee_map.get(E0_CODE) or {}).get("sharpe"))
    base_ex4 = _window(base_nav, drop=(2024,)) if base_nav is not None else {}
    base_ex5 = _window(base_nav, drop=(2025,)) if base_nav is not None else {}
    yearly_rows = []
    boot_rows = []
    material_rows = []
    alts = []
    for code in codes:
        nav = _nav(code)
        if nav is None:
            yearly_rows.append({"code": code, "status": "MISSING_NAV"})
            continue
        rec: dict[str, Any] = {"code": code, "full": _window(nav), "ex2024": _window(nav, drop=(2024,)), "ex2025": _window(nav, drop=(2025,)), "ex2024_2025": _window(nav, drop=(2024, 2025))}
        for name, years in WINDOWS.items():
            rec[name] = _window(nav, years=years)
        yearly_rows.append(rec)
        if code == E0_CODE or base_nav is None:
            continue
        da = daily_from_nav(nav)
        db = daily_from_nav(base_nav)
        aligned = pd.concat([da.rename("a"), db.rename("b")], axis=1).dropna()
        boot = {"status": "too_short", "n": int(len(aligned))}
        if len(aligned) > 40:
            boot = block_bootstrap_delta(aligned["a"], aligned["b"], n=BOOT_N, block=BLOCK_DAYS, seed=17)
        boot_rows.append({"code": code, "block_bootstrap_vs_e0": boot})
        alts.append(da.rename(code))
        keys = _trade_keys(code)
        top_changed = None if keys is None or base_keys is None else bool(keys.symmetric_difference(base_keys))
        net = _net(code)
        top_net_ok = None if net is None or base_net is None else bool(net + 1e-12 >= base_net)
        fee_s = finite_or_none((fee_map.get(code) or {}).get("sharpe"))
        ds = None if rec["full"].get("sharpe") is None else float(rec["full"]["sharpe"]) - float(V5_ANCHOR["sharpe"])
        fee_ds = None if fee_s is None or base_fee_s is None else float(fee_s) - float(base_fee_s)
        fee_same = None if ds is None or fee_ds is None else ((ds > 0 and fee_ds > 0) or (ds < 0 and fee_ds < 0) or (abs(ds) <= 1e-12 and abs(fee_ds) <= 1e-12))
        full_for_gate = dict(rec["full"])
        match = next((r for r in rows_bt if r.get("code") == code), {})
        full_for_gate["nav_end"] = match.get("nav_end")
        mat = _material(full_for_gate, top_changed=top_changed, fee_same=fee_same, ex2024=rec["ex2024"], ex2025=rec["ex2025"], base_ex2024=base_ex4, base_ex2025=base_ex5, top_net_ok=top_net_ok)
        material_rows.append({"code": code, "factor_id": match.get("factor_id"), "role": match.get("role"), **mat})
    spa: dict[str, Any]
    if not alts or base_nav is None:
        spa = {"status": NOT_IDENTIFIABLE, "observed_statistic": None, "n_alternatives": 0, "p_spa": None, "reason": "no alternatives", "label": REPEATED_HISTORICAL_TESTING}
    else:
        b = daily_from_nav(base_nav)
        mat = pd.concat(alts, axis=1).join(b.rename("base"), how="inner").dropna()
        if len(mat) <= 50 or mat.shape[1] < 2:
            spa = {"status": NOT_IDENTIFIABLE, "observed_statistic": None, "n_alternatives": int(max(mat.shape[1] - 1, 0)), "p_spa": None, "reason": "too_short", "label": REPEATED_HISTORICAL_TESTING}
        else:
            d = mat.drop(columns=["base"]).sub(mat["base"], axis=0)
            obs = float(d.mean().max())
            rng = np.random.default_rng(17)
            n = len(d)
            block = BLOCK_DAYS
            hits = 0
            arr = d.to_numpy(dtype=float)
            for _ in range(SPA_BOOT):
                starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
                idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
                boot_max = float(arr[idx].mean(axis=0).max())
                if boot_max >= obs - 1e-15:
                    hits += 1
            spa = {
                "status": "COMPUTED",
                "observed_statistic": obs,
                "n_alternatives": int(d.shape[1]),
                "p_spa": hits / SPA_BOOT,
                "n_boot": SPA_BOOT,
                "block": block,
                "label": REPEATED_HISTORICAL_TESTING,
            }
    replace_rows = []
    by_code = {r["code"]: r for r in material_rows}
    train = load_json(ROUND22_ROOT / "model" / "train.json")
    for gate in train.get("gates") or []:
        role = str(gate.get("role") or "")
        if not role.startswith("E5") and not role.startswith("E7") and not role.startswith("E9"):
            continue
        fid = gate.get("factor_id")
        num = role.split("_", 1)[0]
        remove_role = {"E5": "E4", "E7": "E6", "E9": "E8"}[num]
        add_code = gate.get("code")
        rem_code = f"R22_{remove_role}_{fid}"
        if add_code not in by_code or rem_code not in by_code:
            replace_rows.append({"code": add_code, "remove_code": rem_code, "status": "MISSING_PAIR", "delta_sharpe_replace_minus_remove": None})
            continue
        ds_r = by_code[add_code].get("delta_sharpe")
        ds_m = by_code[rem_code].get("delta_sharpe")
        gap = None if ds_r is None or ds_m is None else float(ds_r) - float(ds_m)
        replace_rows.append({"code": add_code, "remove_code": rem_code, "factor_id": fid, "delta_sharpe_replace": ds_r, "delta_sharpe_remove": ds_m, "delta_sharpe_replace_minus_remove": gap, "status": "COMPUTED"})
    any_material = any(r.get("economically_material") for r in material_rows)
    seed = {"status": NOT_TRIGGERED, "reason": "经济门与增量门未同时通过，不另跑种子扫描。模型内五种子仍是冻结配置。", "seeds_inside_model": [17, 42, 73, 101, 202]}
    if any_material:
        seed = {"status": "BLOCKED", "reason": "经济门已过，但本轮没有额外种子扫描产物，不能写成 COMPLETE。", "seeds_inside_model": [17, 42, 73, 101, 202]}
    dump_json(ROUND22_ROOT / "seed" / "seed_robustness.json", seed)
    flat_yearly = []
    for rec in yearly_rows:
        if "full" not in rec:
            flat_yearly.append(rec)
            continue
        row = {"code": rec["code"]}
        for key in ("full", "ex2024", "ex2025", "ex2024_2025", *WINDOWS):
            block = rec.get(key) or {}
            row[f"{key}_sharpe"] = block.get("sharpe")
            row[f"{key}_cagr"] = block.get("cagr")
            row[f"{key}_maxdd"] = block.get("max_drawdown")
            row[f"{key}_n"] = block.get("n")
        flat_yearly.append(row)
    write_csv(ROUND22_ROOT / "stats" / "yearly_results.csv", flat_yearly)
    write_csv(ROUND22_ROOT / "bootstrap" / "bootstrap.csv", boot_rows)
    write_csv(ROUND22_ROOT / "spa" / "spa.csv", [spa])
    write_csv(ROUND22_ROOT / "stats" / "materiality.csv", material_rows)
    write_csv(ROUND22_ROOT / "stats" / "replace_minus_remove.csv", replace_rows)
    payload = {
        "status": "DONE",
        "yearly": yearly_rows,
        "bootstrap": boot_rows,
        "spa": spa,
        "material": material_rows,
        "replace_minus_remove": replace_rows,
        "seed": seed,
        "base_full": base_full,
        "boot_n": BOOT_N,
        "block_days": BLOCK_DAYS,
        "forward": FORWARD_OOS_NOT_AVAILABLE,
        "label": REPEATED_HISTORICAL_TESTING,
        "n_material": int(sum(bool(r.get("economically_material")) for r in material_rows)),
    }
    dump_json(STATS_JSON, payload)
    write_md(
        MD["18"],
        "\n".join(
            [
                "# Round22 · 18 分年与费用",
                "",
                "分桶：2018–2020、2021–2023、2024–2025、2026 YTD，以及去掉 2024、去掉 2025、同时去掉 2024 和 2025。",
                "费用方向用 HISTORICAL_FEE_CORRECTED 相对 E0 的 Sharpe 差，与主口径 ROUND7_FLAT_5BP 的差同号才算不翻转。",
                f"标记 `{REPEATED_HISTORICAL_TESTING}`。短窗只和同窗 v5 比，不拿来单独晋级。",
                "",
            ]
        ),
    )
    write_md(
        MD["19"],
        "\n".join(
            [
                "# Round22 · 19 Bootstrap / SPA",
                "",
                f"Bootstrap N={BOOT_N} block={BLOCK_DAYS}。",
                f"SPA status=`{spa.get('status')}` observed=`{spa.get('observed_statistic')}` n_alternatives=`{spa.get('n_alternatives')}` p=`{spa.get('p_spa')}`。",
                "没有备择时 SPA 为 NOT_IDENTIFIABLE，不写成 DONE。",
                f"种子状态=`{seed.get('status')}`。",
                "Replace 相对 Remove 的 Sharpe 差见 stats/replace_minus_remove.csv。只删旧列变好不算新因子增量。",
                "",
            ]
        ),
    )
    return payload
