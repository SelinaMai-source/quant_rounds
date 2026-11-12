"""门禁后的消融 / 双置信度 / 仓位 / 邻域 / 时间 / seed / 成本 / 尾部 / bootstrap / stress。"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND3_ROOT, assert_project_result_path, ensure_round3_tree
from ..round2.data import SEEDS
from ..round2.metrics import neighbor_fragility, pareto_front, right_tail_contribution, window_slice_metrics
from .registry import R3Spec, clone_spec, get_spec, spec_from_overlay
from .runner import load_report, run_spec, tracking_row

COMPLETE_YEARS = tuple(range(2018, 2026))
WINDOWS = (
    ("2018-2020", "2018-01-01", "2020-12-31"),
    ("2021-2023", "2021-01-01", "2023-12-31"),
    ("2024-2025", "2024-01-01", "2025-12-31"),
    ("2026YTD", "2026-01-01", "2026-07-24"),
)


def _write_json(rel: str, payload: Any) -> None:
    path = ensure_round3_tree() / rel
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _metrics(code: str) -> dict[str, Any]:
    report = load_report(code) or {}
    return report.get("metrics") or {}


def _row(code: str) -> dict[str, Any]:
    m = _metrics(code)
    return {
        "code": code,
        "cagr": m.get("annualized_return"),
        "sharpe": m.get("sharpe_zero_rate"),
        "max_drawdown": m.get("max_drawdown"),
        "worst_year_return": m.get("worst_complete_year_return") or m.get("worst_year_return"),
        "worst_12m": m.get("worst_rolling_12m"),
        "completed_trades": m.get("completed_trades"),
        "total_return": m.get("total_return"),
        "calmar": m.get("calmar"),
        "low_activity": m.get("low_activity"),
    }


def _delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    def d(key):
        if a.get(key) is None or b.get(key) is None:
            return None
        return float(a[key]) - float(b[key])

    return {
        "d_cagr": d("cagr"),
        "d_sharpe": d("sharpe"),
        "d_mdd": d("max_drawdown"),
        "d_worst_year": d("worst_year_return"),
        "d_worst_12m": d("worst_12m"),
        "d_trades": (
            None if a.get("completed_trades") is None or b.get("completed_trades") is None
            else int(a["completed_trades"]) - int(b["completed_trades"])
        ),
    }


def run_many(specs: list[R3Spec], *, predictions, prepared, meta, resume: bool, run_now: bool) -> dict[str, Any]:
    out = {}
    for spec in specs:
        out[spec.code] = run_spec(
            spec,
            predictions=predictions,
            prepared_market=prepared,
            market_meta=meta,
            resume=resume,
            run_now=run_now,
        )
    return out


def pit_corrected_specs() -> list[R3Spec]:
    specs = [
        spec_from_overlay(
            "R3_G02p",
            "行业拥挤 rolling250 PIT",
            {"kind": "industry_crowd_pit", "half": 0.80, "skip": 0.90},
            source_round2="R2_G02",
            pit_correct_g02=True,
            stage="pit_fix",
            tags=("g02", "pit"),
            uses_g02=True,
        ),
        clone_spec(get_spec("R3_04"), code="R3_04p", pit_correct_g02=True, stage="pit_fix", tags=("t06", "pit")),
        clone_spec(get_spec("R3_05"), code="R3_05p", pit_correct_g02=True, stage="pit_fix", tags=("p13", "pit")),
        clone_spec(get_spec("R3_07"), code="R3_07p", pit_correct_g02=True, stage="pit_fix", tags=("p14", "pit")),
    ]
    return specs


def ablation_specs() -> list[R3Spec]:
    singles = [
        spec_from_overlay("R3_B01a", "上涨家数分档 45/50/60", {"kind": "breadth_tier", "t1": 0.45, "t2": 0.50, "t3": 0.60},
                          source_round2="R2_B01a", stage="ablation", tags=("ablation", "b01a")),
        spec_from_overlay("R3_C02", "seed 分档（=R3_01）", {"kind": "confidence_agree", "mode": "graded"},
                          source_round2="R2_C02_graded", stage="ablation", tags=("ablation", "c02")),
        spec_from_overlay("R3_C03", "分差半仓（=R3_02）", {"kind": "margin", "low_mult": 0.5, "cutoff": 0.30},
                          source_round2="R2_C03_half", stage="ablation", tags=("ablation", "c03")),
        spec_from_overlay(
            "R3_B01a_G02p",
            "上涨家数分档 × 行业拥挤 PIT",
            {"kind": "product", "components": ["R2_B01a", "R2_G02"]},
            pit_correct_g02=True, stage="ablation", tags=("ablation", "pair"), uses_g02=True,
        ),
        spec_from_overlay(
            "R3_B01a_C03_G02p",
            "上涨家数分档 × 分差半仓 × 行业拥挤 PIT",
            {"kind": "product", "components": ["R2_B01a", "R2_C03_half", "R2_G02"]},
            pit_correct_g02=True, stage="ablation", tags=("ablation", "new"), uses_g02=True,
        ),
    ]
    return singles + pit_corrected_specs()


def write_ablation(results: dict[str, Any]) -> dict[str, Any]:
    base = _row("R3_00")
    t06_codes = {
        "B01a": "R3_B01a",
        "C02": "R3_01",
        "G02p": "R3_G02p",
        "B01a×C02": "R3_06",
        "C02×G02p": "R3_05p",
        "B01a×G02p": "R3_B01a_G02p",
        "B01a×C02×G02p": "R3_04p",
    }
    p04_codes = {
        "B01a": "R3_B01a",
        "C03": "R3_02",
        "B01a×C03": "R3_03",
    }
    extra = {"B01a×C03×G02p": "R3_B01a_C03_G02p"}

    def pack(mapping):
        rows = []
        for name, code in mapping.items():
            row = _row(code)
            row["name"] = name
            row.update(_delta(row, base))
            rows.append(row)
        return rows

    payload = {
        "baseline": base,
        "t06": pack(t06_codes),
        "p04": pack(p04_codes),
        "new_candidate": pack(extra),
    }
    _write_json("03_ablation/ablation.json", payload)
    return payload


def dual_specs() -> list[R3Spec]:
    return [
        spec_from_overlay("R3_DUAL_min", "双置信度取 min", {"kind": "dual_conf", "combine": "min"},
                          stage="dual", tags=("dual",)),
        spec_from_overlay("R3_DUAL_intersect", "双置信度相交（乘）", {"kind": "dual_conf", "combine": "intersect"},
                          stage="dual", tags=("dual",)),
        spec_from_overlay("R3_DUAL_mean", "双置信度取均值", {"kind": "dual_conf", "combine": "mean"},
                          stage="dual", tags=("dual",)),
    ]


def write_dual() -> dict[str, Any]:
    rows = [_row(code) for code in ("R3_DUAL_min", "R3_DUAL_intersect", "R3_DUAL_mean", "R3_01", "R3_02")]
    best = max((r for r in rows if r.get("cagr") is not None), key=lambda r: (r["sharpe"] or 0, r["cagr"] or 0), default=None)
    payload = {"rows": rows, "best": best}
    _write_json("04_dual/dual.json", payload)
    return payload


def core_candidates() -> list[str]:
    """不默认 T06。用 shortlist + PIT + 双置信度里账本有效的组。"""
    codes = ["R3_00", "R3_01", "R3_02", "R3_03", "R3_06", "R3_04p", "R3_05p", "R3_07p", "R3_B01a_C03_G02p"]
    codes += ["R3_DUAL_min", "R3_DUAL_intersect", "R3_DUAL_mean"]
    return [c for c in codes if (ROUND3_ROOT / c / "report.json").is_file()]


def pick_size_bases() -> list[str]:
    dual = write_dual()
    best_dual = (dual.get("best") or {}).get("code")
    bases = ["R3_03", "R3_06"]
    if best_dual:
        bases.append(best_dual)
    # 再加一个 PIT T06 对照，不默认它赢
    if (ROUND3_ROOT / "R3_04p" / "report.json").is_file():
        bases.append("R3_04p")
    seen = []
    for b in bases:
        if b not in seen and (ROUND3_ROOT / b / "report.json").is_file():
            seen.append(b)
    return seen[:3] or ["R3_01", "R3_02", "R3_03"]


def size_specs(bases: list[str] | None = None) -> list[R3Spec]:
    bases = bases or pick_size_bases()
    out = []
    for base in bases:
        if base.startswith("R3_DUAL"):
            how = base.split("_")[-1]
            overlay = {"kind": "dual_conf", "combine": how}
            pit = False
            src = None
        elif base in SHORTLIST_CODES():
            spec = get_spec(base) if base in ("R3_00", "R3_01", "R3_02", "R3_03", "R3_04", "R3_05", "R3_06", "R3_07") else None
            if spec is None:
                continue
            overlay = spec.overlay
            pit = spec.pit_correct_g02
            src = spec.source_round2
        elif base.endswith("p") and base.replace("p", "") in ("R3_04", "R3_05", "R3_07"):
            spec = get_spec(base.replace("p", ""))
            overlay = spec.overlay
            pit = True
            src = spec.source_round2
        else:
            report = load_report(base) or {}
            overlay = (json.loads((ROUND3_ROOT / "configs" / f"{base}.json").read_text())["overlay"]
                       if (ROUND3_ROOT / "configs" / f"{base}.json").is_file() else {"kind": "identity"})
            pit = bool(report.get("pit_correct_g02"))
            src = report.get("source_round2")
        for frac, tag in ((0.20, "20"), (0.25, "25"), (0.30, "30")):
            out.append(spec_from_overlay(
                f"{base}_NAV{tag}",
                f"{base} 单票 NAV{tag}%",
                overlay,
                source_round2=src,
                name_budget_nav_fraction=frac,
                pit_correct_g02=pit,
                stage="size",
                tags=("size", base),
                uses_g02=("G02" in json.dumps(overlay)),
            ))
    return out


def SHORTLIST_CODES() -> set[str]:
    return {"R3_00", "R3_01", "R3_02", "R3_03", "R3_04", "R3_05", "R3_06", "R3_07"}


def write_size(bases: list[str]) -> dict[str, Any]:
    rows = []
    for base in bases:
        for tag in ("20", "25", "30"):
            code = f"{base}_NAV{tag}"
            if (ROUND3_ROOT / code / "report.json").is_file():
                row = _row(code)
                row["base"] = base
                row["nav_frac"] = float(tag) / 100.0
                rows.append(row)
    payload = {"bases": bases, "rows": rows}
    _write_json("05_size/size.json", payload)
    return payload


def neighbor_specs() -> list[R3Spec]:
    return [
        spec_from_overlay("R3_NB_B01b", "广度 40/50/60", {"kind": "breadth_tier", "t1": 0.40, "t2": 0.50, "t3": 0.60},
                          market_up_ratio_min=0.40, stage="neighbor", tags=("neighbor", "breadth")),
        spec_from_overlay("R3_NB_B01c", "广度 45/55/65", {"kind": "breadth_tier", "t1": 0.45, "t2": 0.55, "t3": 0.65},
                          stage="neighbor", tags=("neighbor", "breadth")),
        spec_from_overlay("R3_NB_C02_ge4", "C02 严：≥4/5", {"kind": "confidence_agree", "mode": "ge4"},
                          stage="neighbor", tags=("neighbor", "c02")),
        spec_from_overlay("R3_NB_C02_una", "C02 松/更严：5/5", {"kind": "confidence_agree", "mode": "unanimous"},
                          stage="neighbor", tags=("neighbor", "c02")),
        spec_from_overlay("R3_NB_C03_c20", "C03 cutoff 20%", {"kind": "margin", "low_mult": 0.5, "cutoff": 0.20},
                          stage="neighbor", tags=("neighbor", "c03")),
        spec_from_overlay("R3_NB_C03_c40", "C03 cutoff 40%", {"kind": "margin", "low_mult": 0.5, "cutoff": 0.40},
                          stage="neighbor", tags=("neighbor", "c03")),
        spec_from_overlay("R3_NB_G02_lo", "G02 PIT 邻档 70/80", {"kind": "industry_crowd_pit", "half": 0.70, "skip": 0.80},
                          pit_correct_g02=True, stage="neighbor", tags=("neighbor", "g02"), uses_g02=True),
        spec_from_overlay("R3_NB_G02_hi", "G02 PIT 邻档 85/95", {"kind": "industry_crowd_pit", "half": 0.85, "skip": 0.95},
                          pit_correct_g02=True, stage="neighbor", tags=("neighbor", "g02"), uses_g02=True),
    ]


def write_neighbor() -> dict[str, Any]:
    groups = {
        "breadth": {"center": "R3_B01a", "neighbors": ["R3_NB_B01b", "R3_NB_B01c"]},
        "c02": {"center": "R3_01", "neighbors": ["R3_NB_C02_ge4", "R3_NB_C02_una"]},
        "c03": {"center": "R3_02", "neighbors": ["R3_NB_C03_c20", "R3_NB_C03_c40"]},
        "g02": {"center": "R3_G02p", "neighbors": ["R3_NB_G02_lo", "R3_NB_G02_hi"]},
    }
    out = {}
    for name, spec in groups.items():
        center = _row(spec["center"]) if (ROUND3_ROOT / spec["center"] / "report.json").is_file() else {}
        nbs = [_row(c) for c in spec["neighbors"] if (ROUND3_ROOT / c / "report.json").is_file()]
        flag = neighbor_fragility(center, nbs) if center and nbs else "unknown"
        if flag == "fragile":
            flag = "FRAGILE"
        out[name] = {"center": center, "neighbors": nbs, "flag": flag}
    _write_json("06_neighbor/neighbor.json", out)
    return out


def time_specs(bases: list[str]) -> list[R3Spec]:
    out = []
    for base in bases:
        overlay, pit, src = _overlay_of(base)
        out.append(spec_from_overlay(
            f"{base}_drop2024",
            f"{base} 去掉 2024 信号",
            overlay,
            source_round2=src,
            pit_correct_g02=pit,
            skip_years=(2024,),
            stage="time",
            tags=("time", "drop2024"),
            uses_g02=("G02" in json.dumps(overlay)),
        ))
        for year in COMPLETE_YEARS:
            out.append(spec_from_overlay(
                f"{base}_loo{year}",
                f"{base} 去掉 {year} 信号",
                overlay,
                source_round2=src,
                pit_correct_g02=pit,
                skip_years=(year,),
                stage="time",
                tags=("time", "loo"),
                uses_g02=("G02" in json.dumps(overlay)),
            ))
    return out


def _overlay_of(code: str) -> tuple[dict[str, Any], bool, str | None]:
    cfg_path = ROUND3_ROOT / "configs" / f"{code}.json"
    if code in ("R3_00", "R3_01", "R3_02", "R3_03", "R3_04", "R3_05", "R3_06", "R3_07"):
        spec = get_spec(code)
        return spec.overlay, spec.pit_correct_g02, spec.source_round2
    if code.endswith("p") and code[:-1] in ("R3_04", "R3_05", "R3_07"):
        spec = get_spec(code[:-1])
        return spec.overlay, True, spec.source_round2
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return cfg.get("overlay") or {"kind": "identity"}, bool(cfg.get("pit_correct_g02")), cfg.get("source_round2")
    if code.startswith("R3_DUAL"):
        return {"kind": "dual_conf", "combine": code.split("_")[-1]}, False, None
    return {"kind": "identity"}, False, None


def write_time(bases: list[str]) -> dict[str, Any]:
    payload = {"bases": bases, "windows": {}, "yearly": {}, "drop2024": {}, "loo": {}, "year_concentrated": []}
    for base in bases:
        nav_path = ROUND3_ROOT / base / "nav.parquet"
        if not nav_path.is_file():
            continue
        nav = pd.read_parquet(nav_path)
        payload["windows"][base] = {
            name: window_slice_metrics(nav, start, end) for name, start, end in WINDOWS
        }
        m = _metrics(base)
        payload["yearly"][base] = m.get("yearly_returns") or {}
        drop = _row(f"{base}_drop2024") if (ROUND3_ROOT / f"{base}_drop2024" / "report.json").is_file() else None
        payload["drop2024"][base] = drop
        loo_rows = []
        for year in COMPLETE_YEARS:
            code = f"{base}_loo{year}"
            if (ROUND3_ROOT / code / "report.json").is_file():
                row = _row(code)
                row["dropped_year"] = year
                loo_rows.append(row)
        payload["loo"][base] = loo_rows
    # 单年决定排名：若去掉某年后相对顺序相对 Normal 翻转
    ranks_normal = _rank_by_cagr(bases)
    for year in COMPLETE_YEARS:
        codes = [f"{b}_loo{year}" for b in bases if (ROUND3_ROOT / f"{b}_loo{year}" / "report.json").is_file()]
        if len(codes) >= 2:
            ranks = _rank_by_cagr(codes)
            mapped = [c.replace(f"_loo{year}", "") for c in ranks]
            if mapped != [c for c in ranks_normal if c in mapped]:
                payload["year_concentrated"].append({"year": year, "normal": ranks_normal, "loo": mapped})
    if payload["year_concentrated"]:
        payload["flag"] = "YEAR_CONCENTRATED"
    _write_json("07_time/time.json", payload)
    return payload


def _rank_by_cagr(codes: list[str]) -> list[str]:
    rows = [(c, _metrics(c).get("annualized_return")) for c in codes]
    rows = [(c, v) for c, v in rows if v is not None]
    rows.sort(key=lambda x: float(x[1]), reverse=True)
    return [c for c, _ in rows]


def seed_specs(bases: list[str]) -> list[R3Spec]:
    out = []
    for base in bases:
        overlay, pit, src = _overlay_of(base)
        # 只对带 C02 的组做 leave-one-seed
        blob = json.dumps(overlay)
        if "confidence_agree" not in blob and "dual_conf" not in blob and "C02" not in blob:
            if base not in {"R3_01", "R3_06", "R3_04p", "R3_05p"} and not str(overlay.get("kind", "")).startswith("dual"):
                continue
        for seed in SEEDS:
            out.append(spec_from_overlay(
                f"{base}_looSeed{seed}",
                f"{base} 去掉 seed{seed}（比例档）",
                {"kind": "confidence_agree_ratio"},
                source_round2=src,
                pit_correct_g02=pit,
                seed_leave_out=seed,
                seeds=SEEDS,
                stage="seed",
                tags=("seed", "loo"),
            ))
    return out


def write_seed(bases: list[str]) -> dict[str, Any]:
    payload = {"bases": bases, "loo": {}, "diagnostic": {}}
    for base in bases:
        rows = []
        for seed in SEEDS:
            code = f"{base}_looSeed{seed}"
            if (ROUND3_ROOT / code / "report.json").is_file():
                row = _row(code)
                row["dropped_seed"] = seed
                rows.append(row)
        payload["loo"][base] = rows
        trades_path = ROUND3_ROOT / base / "trades.parquet"
        if trades_path.is_file():
            payload["diagnostic"][base] = _seed_trade_diagnostic(pd.read_parquet(trades_path))
    _write_json("08_seed/seed.json", payload)
    return payload


def _seed_trade_diagnostic(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty or "ret" not in trades:
        return {}
    ret = pd.to_numeric(trades["ret"], errors="coerce")
    return {
        "n": int(ret.notna().sum()),
        "win_rate": float((ret > 0).mean()) if ret.notna().any() else None,
        "mean_ret": float(ret.mean()) if ret.notna().any() else None,
        "note": "单 seed 诊断不重打 fs00263 分；只看已有成交收益分布。",
    }


def cost_specs(bases: list[str]) -> list[R3Spec]:
    out = []
    variants = (
        [("slip5", {"extra_slippage_bps": 5.0}), ("slip10", {"extra_slippage_bps": 10.0}),
         ("slip20", {"extra_slippage_bps": 20.0}),
         ("fee15", {"fee_mult": 1.5}), ("fee20", {"fee_mult": 2.0}),
         ("buy0", {"buy_adverse_bps": 0.0}), ("buy10", {"buy_adverse_bps": 10.0}),
         ("buy20", {"buy_adverse_bps": 20.0}), ("buy30", {"buy_adverse_bps": 30.0}),
         ("tol003", {"limit_up_tolerance": 0.003}), ("tol005", {"limit_up_tolerance": 0.005}),
         ("tol007", {"limit_up_tolerance": 0.007})]
    )
    for base in bases:
        overlay, pit, src = _overlay_of(base)
        for tag, ov in variants:
            if tag == "buy0" or tag == "tol005":
                continue  # 与原点重复
            out.append(spec_from_overlay(
                f"{base}_{tag}",
                f"{base} {tag}",
                overlay,
                source_round2=src,
                pit_correct_g02=pit,
                cost_overrides=ov,
                stage="cost",
                tags=("cost", tag),
                uses_g02=("G02" in json.dumps(overlay)),
            ))
    return out


def write_cost(bases: list[str]) -> dict[str, Any]:
    payload = {"bases": bases, "rows": []}
    for base in bases:
        normal = _row(base)
        for tag in ("slip5", "slip10", "slip20", "fee15", "fee20", "buy10", "buy20", "buy30", "tol003", "tol007"):
            code = f"{base}_{tag}"
            if (ROUND3_ROOT / code / "report.json").is_file():
                row = _row(code)
                row["base"] = base
                row["stress"] = tag
                row["still_positive"] = (row.get("cagr") or 0) > 0
                row.update({f"vs_{k}": v for k, v in _delta(row, normal).items()})
                payload["rows"].append(row)
    _write_json("09_cost/cost.json", payload)
    return payload


def write_tail(bases: list[str]) -> dict[str, Any]:
    payload = {"bases": bases, "rows": []}
    base_nav = ROUND3_ROOT / "R3_00" / "nav.parquet"
    base_dd_days = _top_drawdown_days(base_nav, n=10) if base_nav.is_file() else []
    for code in ["R3_00", "R3_01", "R3_02", "R3_03"] + [c for c in bases if c not in {"R3_00", "R3_01", "R3_02", "R3_03"}]:
        trades_path = ROUND3_ROOT / code / "trades.parquet"
        nav_path = ROUND3_ROOT / code / "nav.parquet"
        if not trades_path.is_file():
            continue
        trades = pd.read_parquet(trades_path)
        tail = right_tail_contribution(trades)
        row = {"code": code, **tail}
        if "ret" in trades and len(trades):
            ret = pd.to_numeric(trades["ret"], errors="coerce").dropna().sort_values(ascending=False)
            if len(ret):
                row["without_max_winner_sum"] = float(ret.iloc[1:].sum()) if len(ret) > 1 else 0.0
                row["max_winner"] = float(ret.iloc[0])
        if nav_path.is_file() and base_dd_days:
            nav = pd.read_parquet(nav_path)
            row["dd_cluster_vs_baseline"] = _path_move_on_days(nav, base_dd_days)
        payload["rows"].append(row)
    _write_json("10_tail/tail.json", payload)
    return payload


def _top_drawdown_days(nav_path, n: int = 10) -> list[str]:
    nav = pd.read_parquet(nav_path).sort_values("trade_date")
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    asset = nav["total_asset"].astype(float)
    dd = asset / asset.cummax() - 1.0
    worst = dd.nsmallest(n)
    return [str(pd.Timestamp(nav.loc[i, "trade_date"]).date()) for i in worst.index]


def _path_move_on_days(nav: pd.DataFrame, days: list[str]) -> dict[str, Any]:
    frame = nav.sort_values("trade_date").copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    sub = frame.loc[frame["trade_date"].isin(days)]
    if sub.empty:
        return {}
    rets = sub["total_asset"].astype(float).pct_change().dropna()
    return {"n": int(len(sub)), "sum_ret": float(rets.sum()) if len(rets) else None}


def write_bootstrap(bases: list[str], n_first: int = 500, n_full: int = 2000) -> dict[str, Any]:
    payload = {"block": 20, "n_first": n_first, "n_full": n_full, "note": "统计重采样，不是 OOS", "rows": {}}
    rng = np.random.default_rng(42)
    for base in bases:
        nav_path = ROUND3_ROOT / base / "nav.parquet"
        if not nav_path.is_file():
            continue
        nav = pd.read_parquet(nav_path).sort_values("trade_date")
        daily = nav["total_asset"].astype(float).pct_change().dropna().to_numpy()
        first = _block_bootstrap(daily, 20, n_first, rng)
        full = _block_bootstrap(daily, 20, n_full, rng)
        payload["rows"][base] = {"n500": first, "n2000": full}
    _write_json("11_bootstrap/bootstrap.json", payload)
    return payload


def _block_bootstrap(daily: np.ndarray, block: int, n: int, rng: np.random.Generator) -> dict[str, Any]:
    if len(daily) < block * 2:
        return {}
    blocks = [daily[i:i + block] for i in range(0, len(daily) - block + 1, block)]
    if not blocks:
        return {}
    need = int(math.ceil(len(daily) / block))
    cagrs, sharpes, mdds = [], [], []
    for _ in range(n):
        idx = rng.integers(0, len(blocks), size=need)
        path = np.concatenate([blocks[i] for i in idx])[: len(daily)]
        wealth = np.cumprod(1.0 + path)
        years = max(len(path) / 365.25 * (365.25 / 252), 1 / 365.25)
        # 用交易日近似 years = n/252
        years = max(len(path) / 252.0, 1 / 252.0)
        cagrs.append(float(wealth[-1] ** (1 / years) - 1))
        vol = float(np.std(path, ddof=1))
        sharpes.append(float(np.mean(path) / vol * math.sqrt(252)) if vol > 0 else None)
        peak = np.maximum.accumulate(wealth)
        mdds.append(float((wealth / peak - 1.0).min()))
    sharpes_f = [x for x in sharpes if x is not None]
    return {
        "cagr_p05": float(np.percentile(cagrs, 5)),
        "cagr_p50": float(np.percentile(cagrs, 50)),
        "sharpe_p05": float(np.percentile(sharpes_f, 5)) if sharpes_f else None,
        "sharpe_p50": float(np.percentile(sharpes_f, 50)) if sharpes_f else None,
        "mdd_p05": float(np.percentile(mdds, 5)),
        "mdd_p50": float(np.percentile(mdds, 50)),
    }


def write_stress(bases: list[str]) -> dict[str, Any]:
    scenarios = {
        "normal": lambda b: _row(b),
        "cost_slip20": lambda b: _row(f"{b}_slip20") if (ROUND3_ROOT / f"{b}_slip20" / "report.json").is_file() else None,
        "cost_fee20": lambda b: _row(f"{b}_fee20") if (ROUND3_ROOT / f"{b}_fee20" / "report.json").is_file() else None,
        "open_buy30": lambda b: _row(f"{b}_buy30") if (ROUND3_ROOT / f"{b}_buy30" / "report.json").is_file() else None,
        "size_nav20": lambda b: _row(f"{b}_NAV20") if (ROUND3_ROOT / f"{b}_NAV20" / "report.json").is_file() else None,
        "drop2024": lambda b: _row(f"{b}_drop2024") if (ROUND3_ROOT / f"{b}_drop2024" / "report.json").is_file() else None,
    }
    payload = {"bases": bases, "look_at": "worst_not_best", "rows": []}
    for base in bases:
        row = {"code": base}
        worst_cagr = None
        worst_name = None
        for name, fn in scenarios.items():
            val = fn(base)
            row[name] = val
            if val and val.get("cagr") is not None:
                if worst_cagr is None or float(val["cagr"]) < worst_cagr:
                    worst_cagr = float(val["cagr"])
                    worst_name = name
        # 邻域 / 丢 seed：取该 base 相关最差
        seed_rows = []
        for seed in SEEDS:
            code = f"{base}_looSeed{seed}"
            if (ROUND3_ROOT / code / "report.json").is_file():
                seed_rows.append(_row(code))
        if seed_rows:
            worst_seed = min(seed_rows, key=lambda r: r.get("cagr") or 9)
            row["drop_seed"] = worst_seed
            if worst_seed.get("cagr") is not None and (worst_cagr is None or worst_seed["cagr"] < worst_cagr):
                worst_cagr = float(worst_seed["cagr"])
                worst_name = "drop_seed"
        row["worst_scenario"] = worst_name
        row["worst_cagr"] = worst_cagr
        payload["rows"].append(row)
    _write_json("12_stress/stress.json", payload)
    return payload


def balanced_hard_gate(row: dict[str, Any], *, cost_positive: bool, neighbor_ok: bool, seed_ok: bool, drop2024_ok: bool) -> dict[str, Any]:
    mdd_ok = row.get("max_drawdown") is not None and float(row["max_drawdown"]) >= -0.20
    year_ok = row.get("worst_year_return") is not None and float(row["worst_year_return"]) >= -0.05
    roll_ok = row.get("worst_12m") is not None and float(row["worst_12m"]) >= -0.15
    trades_ok = row.get("completed_trades") is not None and int(row["completed_trades"]) >= 437
    cagr_ok = row.get("cagr") is not None and float(row["cagr"]) >= 0.18
    meets = all([mdd_ok, year_ok, roll_ok, trades_ok, cagr_ok, cost_positive, neighbor_ok, seed_ok, drop2024_ok])
    reasons = []
    if not mdd_ok:
        reasons.append("MaxDD < −20%")
    if not year_ok:
        reasons.append("完整年最差 < −5%")
    if not roll_ok:
        reasons.append("滚动12M < −15%")
    if not trades_ok:
        reasons.append("笔数 < 437")
    if not cagr_ok:
        reasons.append("CAGR < 18%")
    if not cost_positive:
        reasons.append("成本压力后 CAGR≤0")
    if not neighbor_ok:
        reasons.append("邻域崩")
    if not seed_ok:
        reasons.append("去任一 seed 崩")
    if not drop2024_ok:
        reasons.append("去 2024 崩")
    return {"meets": meets, "reasons": reasons}


def collapse(row: dict[str, Any] | None, center: dict[str, Any]) -> bool:
    if not row or row.get("cagr") is None:
        return True
    if float(row["cagr"]) <= 0:
        return True
    if row.get("max_drawdown") is not None and float(row["max_drawdown"]) < -0.30:
        return True
    if center.get("cagr") and float(row["cagr"]) < 0.5 * float(center["cagr"]):
        return True
    return False
