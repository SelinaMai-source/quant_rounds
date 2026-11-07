"""D 方向 / E buffer / F residual / DD / K 组合。两模块改预算只允许 primary vs veto。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND9_ROOT, assert_project_result_path, ensure_round9_tree, round9_dir
from .allocator import activity_stats, b01_nav, b03_nav, load_alloc, snap_nav, write_alloc
from .contract import (
    FIVE_FEATURES,
    FLOW_CANDIDATES,
    INTRA_CANDIDATES,
    R9_00,
    SEAL_FAMILY,
    TEN_PANEL,
    WF_PREDICT_YEARS,
)
from .features import attach_panel_columns
from .meta import train_mask, walk_forward
from .registry import get_spec
from .runner import load_summary, run_codes


def _daily() -> pd.DataFrame:
    path = ROUND9_ROOT / "ledger" / "top2_daily_features.parquet"
    daily = pd.read_parquet(path)
    daily["signal_date"] = pd.to_datetime(daily["signal_date"]).dt.normalize()
    return daily.set_index("signal_date").sort_index()


def _oof() -> pd.DataFrame:
    path = ROUND9_ROOT / "models" / "meta_oof.parquet"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    return frame.set_index("signal_date").sort_index()


def run_direction(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    ensure_round9_tree()
    daily = _daily()
    oof = _oof()
    pred = pd.to_numeric(oof.get("c01_pred"), errors="coerce") if not oof.empty else pd.Series(dtype=float)
    y = pd.to_numeric(daily.get("top2_ret"), errors="coerce")
    cases = {}
    for name, mask in {
        "CASE_A": daily["agreement"].eq("A"),
        "CASE_B": daily["sep_bucket"].eq("HIGH"),
        "CASE_C": pd.to_numeric(daily.get("top2_d1_open_gap"), errors="coerce").ge(0.02),
        "CASE_D": daily["agreement"].eq("C") & daily["sep_bucket"].eq("LOW"),
    }.items():
        sub_y = y.loc[mask]
        sub_p = pred.reindex(daily.index).loc[mask] if len(pred) else pd.Series(dtype=float)
        by_year = {}
        for year in WF_PREDICT_YEARS:
            m = mask & (daily.index.year == year)
            yy = y.loc[m]
            by_year[str(year)] = {
                "n": int(m.sum()),
                "mean_ret": float(yy.mean()) if yy.notna().any() else None,
            }
        pos_years = sum(1 for v in by_year.values() if (v.get("mean_ret") or 0) > 0 and (v.get("n") or 0) >= 15)
        cases[name] = {
            "n": int(mask.sum()),
            "mean_ret": float(sub_y.mean()) if sub_y.notna().any() else None,
            "mean_pred": float(sub_p.mean()) if sub_p.notna().any() else None,
            "pos_years_ge15": pos_years,
            "by_year": by_year,
        }
    stable = any(
        cases[k]["pos_years_ge15"] >= 4 and (cases[k]["mean_ret"] or 0) > 0
        for k in ("CASE_A", "CASE_B", "CASE_C")
    )
    skip = not stable
    get_spec("R9_D01").skip = True
    get_spec("R9_D02").skip = True
    get_spec("R9_D03").skip = True
    get_spec("R9_D01").skip_reason = "SKIP_D01"
    get_spec("R9_D02").skip_reason = "SKIP_D02"
    get_spec("R9_D03").skip_reason = "SKIP_D03"
    if not skip:
        get_spec("R9_D01").skip_reason = "SKIP_D01_NO_PRE_REGISTERED_RULE"
        get_spec("R9_D02").skip_reason = "SKIP_D02_NO_PRE_REGISTERED_RULE"
        get_spec("R9_D03").skip_reason = "SKIP_D03_NO_PRE_REGISTERED_RULE"
    payload = {
        "cases": cases,
        "stable_oof": stable,
        "decision": "SKIP_D01/D02/D03",
        "note": "无稳定 OOF 方向规则则 SKIP；即使有方向差异也不另搜规则。",
    }
    path = ROUND9_ROOT / "metrics" / "direction_d00.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    md = ROUND9_ROOT / "04_direction.md"
    assert_project_result_path(md)
    md.write_text(
        f"# 04 — 方向诊断 D00\n\n稳定 OOF：{stable}。结论：**SKIP_D01/D02/D03**。\n\n"
        f"```json\n{json.dumps(cases, ensure_ascii=False, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    from .registry import specs_for_stage

    run_codes([s.code for s in specs_for_stage("direction")], resume=resume, jobs=jobs)
    return payload


def run_execution(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    ensure_round9_tree()
    daily = _daily()
    oof = _oof()
    if oof.empty or "c01_pred" not in oof.columns:
        for code in ("R9_E01", "R9_E02", "R9_E03"):
            get_spec(code).skip = True
            get_spec(code).skip_reason = "NO_C01_OOF"
        path = ROUND9_ROOT / "05_execution.md"
        assert_project_result_path(path)
        path.write_text("# 05 — execution\n\n无 C01 OOF，E 组 SKIP。\n", encoding="utf-8")
        return {"status": "skipped"}
    pred = pd.to_numeric(oof["c01_pred"], errors="coerce").reindex(daily.index)
    base = pd.to_numeric(oof.get("alloc_c01"), errors="coerce").reindex(daily.index)
    if base.isna().all():
        base = load_alloc("R9_C01").reindex(daily.index)
    base = base.fillna(0.0)
    e01 = base.where(pred > 0.001, 0.0)
    e02 = base.where(pred > 0.002, 0.0)
    e03 = base.copy()
    gap = pd.to_numeric(daily.get("top2_d1_open_gap"), errors="coerce")
    te_year = pd.Series(daily.index.year, index=daily.index)
    for year in WF_PREDICT_YEARS:
        tr = train_mask(daily, year)
        te = te_year.eq(year)
        gtr = gap.loc[tr].dropna()
        if len(gtr) < 20:
            e03.loc[te] = base.loc[te].where(pred.loc[te] > 0.001, 0.0)
            continue
        cut20 = float(np.nanpercentile(gtr, 80))
        use20 = gap.ge(cut20).fillna(False) & te
        use10 = te & ~use20
        e03.loc[use20] = base.loc[use20].where(pred.loc[use20] > 0.002, 0.0)
        e03.loc[use10] = base.loc[use10].where(pred.loc[use10] > 0.001, 0.0)
    write_alloc("R9_E01", daily.index, e01.map(snap_nav))
    write_alloc("R9_E02", daily.index, e02.map(snap_nav))
    write_alloc("R9_E03", daily.index, e03.map(snap_nav))
    bt = run_codes(["R9_E01", "R9_E02", "R9_E03"], resume=resume, jobs=jobs)
    from .docs import write_named_md

    extra = (
        "E 只在 C01 已净费用预测之上加 +10/+20bp buffer。E03：训练折 gap 顶 20%→20bp，否则 10bp。\n"
        f"activity E01={activity_stats(e01)} E02={activity_stats(e02)} E03={activity_stats(e03)}\n"
    )
    write_named_md("05_execution.md", "execution", ["R9_E01", "R9_E02", "R9_E03"], extra)
    return {"jobs": {k: v.get("status") for k, v in (bt.get("jobs") or {}).items()}}


def _panel_columns() -> tuple[Any, list[str]]:
    from ..paths import CANDIDATE_PANEL_RELATIVE, stock_path

    candidates = [
        TEN_PANEL / "candidate_panel.parquet",
        stock_path(
            "data/first_board_hermes/aligned_original_candidate_full44_plus_d1_open_gap_v1/candidate_panel.parquet"
        ),
        stock_path(str(CANDIDATE_PANEL_RELATIVE)),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            import pyarrow.parquet as pq

            names = list(pq.ParquetFile(str(path)).schema.names)
            return path, names
        except Exception:
            continue
    return None, []


def _pick_family(names: list[str], candidates: tuple[str, ...]) -> list[str]:
    banned = set(FIVE_FEATURES)
    return [c for c in candidates if c in names and c not in banned][:3]


def run_residual(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    ensure_round9_tree()
    daily = _daily()
    path, names = _panel_columns()
    families = {
        "F01": _pick_family(names, SEAL_FAMILY),
        "F02": _pick_family(names, FLOW_CANDIDATES),
        "F03": _pick_family(names, INTRA_CANDIDATES),
    }
    info: dict[str, Any] = {"panel": str(path) if path else None, "n_cols": len(names), "families": families}
    if path is None:
        for code in ("R9_F01", "R9_F02", "R9_F03", "R9_F04"):
            get_spec(code).skip = True
            get_spec(code).skip_reason = "PANEL_MISSING"
        from .docs import write_named_md

        write_named_md("06_residual.md", "residual", ["R9_F01", "R9_F02", "R9_F03", "R9_F04"], "panel 不存在，跳过。")
        return info
    from .contract import META_FEATURES

    runnable = []
    for tag, cols in families.items():
        spec = get_spec(f"R9_{tag}")
        if not cols:
            spec.skip = True
            spec.skip_reason = "FAMILY_COLUMNS_MISSING"
            continue
        work, used = attach_panel_columns(daily, cols)
        if not used:
            spec.skip = True
            spec.skip_reason = "FAMILY_COLUMNS_MISSING"
            continue
        payload = walk_forward(
            work,
            feat_cols=tuple(list(META_FEATURES) + used),
            dest={"C01": f"R9_{tag}"},
            update_c04_skip=False,
        )
        runnable.append(f"R9_{tag}")
        info[f"{tag}_used"] = used
        info[f"{tag}_activity"] = (payload.get("activity") or {}).get("C01")
    if runnable:
        run_codes(runnable, resume=resume, jobs=jobs)
    c01 = load_summary("R9_C01") or {}
    beaten = []
    for code in runnable:
        s = load_summary(code) or {}
        if s.get("cagr") is None or c01.get("cagr") is None:
            continue
        if float(s["cagr"]) + 1e-12 >= float(c01["cagr"]) and float(s["sharpe"]) + 1e-12 >= float(c01.get("sharpe") or -9):
            beaten.append(code)
    f04 = get_spec("R9_F04")
    if len(beaten) >= 2:
        a = load_alloc(beaten[0])
        b = load_alloc(beaten[1])
        idx = a.index.union(b.index)
        combo = pd.concat([a.reindex(idx), b.reindex(idx)], axis=1).min(axis=1)
        write_alloc("R9_F04", combo.index, combo.map(snap_nav))
        run_codes(["R9_F04"], resume=resume, jobs=jobs)
    else:
        f04.skip = True
        f04.skip_reason = "NEED_TWO_FAMILIES_GE_C01"
    from .docs import write_named_md

    extra = f"真实列 {json.dumps(families, ensure_ascii=False)}。F04 仅当两 family≥C01。beaten={beaten}。\n"
    write_named_md("06_residual.md", "residual", ["R9_F01", "R9_F02", "R9_F03", "R9_F04"], extra)
    pathj = ROUND9_ROOT / "metrics" / "residual_meta.json"
    assert_project_result_path(pathj)
    pathj.write_text(json.dumps({**info, "beaten": beaten}, ensure_ascii=False, indent=2, default=str) + "\n")
    return info


def _episodes(nav: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    n = nav.copy()
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    asset = n.sort_values("trade_date").set_index("trade_date")["total_asset"].astype(float)
    dd = asset / asset.cummax() - 1.0
    episodes = []
    in_dd = False
    start = trough = None
    trough_val = 0.0
    for dt, val in dd.items():
        if val < -1e-12:
            if not in_dd:
                in_dd, start, trough, trough_val = True, dt, dt, val
            elif val < trough_val:
                trough, trough_val = dt, val
        elif in_dd:
            episodes.append({
                "start": str(pd.Timestamp(start).date()),
                "trough": str(pd.Timestamp(trough).date()),
                "end": str(pd.Timestamp(dt).date()),
                "depth": float(trough_val),
            })
            in_dd = False
    if in_dd:
        episodes.append({
            "start": str(pd.Timestamp(start).date()),
            "trough": str(pd.Timestamp(trough).date()),
            "end": str(pd.Timestamp(dd.index[-1]).date()),
            "depth": float(trough_val),
        })
    return sorted(episodes, key=lambda x: x["depth"])[:top_n]


def run_drawdown(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    ensure_round9_tree()
    daily = _daily()
    nav = pd.read_parquet(round9_dir(R9_00) / "nav.parquet")
    episodes = _episodes(nav, 5)
    try:
        parent = load_alloc("R9_C01")
    except FileNotFoundError:
        parent = b01_nav(daily["agreement"].astype(str), daily["sep_bucket"].astype(str))
    parent = parent.reindex(daily.index).fillna(0.0)
    veto = daily["agreement"].eq("C") & daily["sep_bucket"].eq("LOW")
    write_alloc("R9_DD01", daily.index, parent.where(~veto, 0.0).map(snap_nav))
    run_codes(["R9_DD01"], resume=resume, jobs=jobs)
    payload = {"episodes": episodes, "veto_days": int(veto.sum()), "note": "no whole-portfolio market filter"}
    path = ROUND9_ROOT / "metrics" / "drawdown_top5.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    from .docs import write_named_md

    extra = (
        "v3 Top5 回撤，只调 Top2（C∧LOW→0），无全组合市场门。\n\n"
        f"```json\n{json.dumps(episodes, ensure_ascii=False, indent=2)}\n```\n"
    )
    write_named_md("07_drawdown.md", "drawdown", ["R9_DD01"], extra)
    return payload


def _better(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not a or a.get("cagr") is None:
        return False
    if not b or b.get("cagr") is None:
        return True
    return (float(a["sharpe"]), float(a["cagr"])) > (float(b["sharpe"]), float(b["cagr"]))


def _min_alloc(primary: str, veto: str, out_code: str, idx: pd.DatetimeIndex) -> None:
    p = load_alloc(primary).reindex(idx).fillna(0.0)
    v = load_alloc(veto).reindex(idx).fillna(0.0)
    write_alloc(out_code, idx, pd.concat([p, v], axis=1).min(axis=1).map(snap_nav))


def run_combine(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    ensure_round9_tree()
    daily = _daily()
    idx = daily.index
    best_s = None
    for c in ("R9_B01", "R9_B02", "R9_B03"):
        if _better(load_summary(c), load_summary(best_s) if best_s else None):
            best_s = c
    best_m = None
    for c in ("R9_C01", "R9_C02", "R9_C03", "R9_C04"):
        if get_spec(c).skip:
            continue
        alloc_path = ROUND9_ROOT / "ledger" / f"{c}_top2_nav.parquet"
        if alloc_path.is_file() and activity_stats(load_alloc(c)).get("flag") == "LOW_ACTIVITY":
            continue
        if _better(load_summary(c), load_summary(best_m) if best_m else None):
            best_m = c
    e_best = None
    for c in ("R9_E01", "R9_E02", "R9_E03"):
        if _better(load_summary(c), load_summary(e_best) if e_best else None):
            e_best = c
    slots = {"R9_K01": best_s, "R9_K02": best_m, "R9_K03": e_best, "R9_K04": None, "R9_K05": None}
    for code, src in list(slots.items())[:3]:
        spec = get_spec(code)
        if src is None:
            spec.skip = True
            spec.skip_reason = "no_stable_module"
        else:
            spec.skip = True
            spec.skip_reason = f"identical_to_{src}"
    overall = best_m if _better(load_summary(best_m) if best_m else None, load_summary(best_s) if best_s else None) else best_s
    runnable = []
    if overall:
        if not (ROUND9_ROOT / "ledger" / "R9_B03_top2_nav.parquet").is_file():
            write_alloc("R9_B03", idx, b03_nav(daily["agreement"].astype(str), daily["sep_bucket"].astype(str)))
        if overall != "R9_B03":
            _min_alloc(overall, "R9_B03", "R9_K04", idx)
            get_spec("R9_K04").skip = False
            get_spec("R9_K04").skip_reason = ""
            runnable.append("R9_K04")
        else:
            get_spec("R9_K04").skip = True
            get_spec("R9_K04").skip_reason = "identical_to_R9_B03"
        if (ROUND9_ROOT / "ledger" / "R9_DD01_top2_nav.parquet").is_file() and overall != "R9_DD01":
            _min_alloc(overall, "R9_DD01", "R9_K05", idx)
            get_spec("R9_K05").skip = False
            get_spec("R9_K05").skip_reason = ""
            runnable.append("R9_K05")
        else:
            get_spec("R9_K05").skip = True
            get_spec("R9_K05").skip_reason = "no_dd_or_identical"
    else:
        get_spec("R9_K04").skip = True
        get_spec("R9_K04").skip_reason = "no_overall"
        get_spec("R9_K05").skip = True
        get_spec("R9_K05").skip_reason = "no_overall"
    if runnable:
        run_codes(runnable, resume=resume, jobs=jobs)
    payload = {"slots": slots, "overall": overall, "runnable": runnable, "rule": "primary_vs_veto_min"}
    path = ROUND9_ROOT / "metrics" / "combine_slots.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    from .docs import write_named_md

    write_named_md(
        "08_combine.md",
        "combine",
        ["R9_K01", "R9_K02", "R9_K03", "R9_K04", "R9_K05"],
        "最多 5 槽。两模块改预算 = primary vs veto（min），禁止相乘。\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}\n```\n",
    )
    return payload


__all__ = ["run_combine", "run_direction", "run_drawdown", "run_execution", "run_residual"]
