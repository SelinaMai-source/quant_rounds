"""单模块 / 组合 / 稳健性。每模块 Top2；两两≤12、三元≤6。禁止笛卡尔积。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import ROUND4_ROOT, assert_project_result_path, ensure_round4_tree
from ..round2.data import SEEDS
from ..round2.metrics import neighbor_fragility, window_slice_metrics
from .overlays import agreement_frame, cross_section_daily
from .registry import (
    B01A,
    C03_HALF,
    LOW_ACTIVITY_TRADES,
    R4Spec,
    SPECS,
    clone_spec,
    get_spec,
    spec_from_overlay,
    stack,
)
from .runner import load_report, run_many, tracking_row

WINDOWS = (
    ("2018-2020", "2018-01-01", "2020-12-31"),
    ("2021-2023", "2021-01-01", "2023-12-31"),
    ("2024-2025", "2024-01-01", "2025-12-31"),
    ("2026YTD", "2026-01-01", "2026-07-24"),
)
COMPLETE_YEARS = tuple(range(2018, 2026))
V2 = "R4_00_baseline_v2"

CROSS_FEATURES = ("top1_minus_median", "top1_minus_p75", "z", "dispersion", "entropy")


def _write_json(rel: str, payload: Any) -> None:
    path = ensure_round4_tree() / rel
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _row(code: str) -> dict[str, Any]:
    report = load_report(code) or {}
    spec = SPECS.get(code)
    return tracking_row(spec, report) if spec else {
        "code": code,
        **{k: (report.get("metrics") or {}).get(k) for k in (
            "annualized_return", "sharpe_zero_rate", "max_drawdown",
        )},
        "cagr": (report.get("metrics") or {}).get("annualized_return"),
        "sharpe": (report.get("metrics") or {}).get("sharpe_zero_rate"),
        "max_drawdown": (report.get("metrics") or {}).get("max_drawdown"),
        "cagr_ex2024": (report.get("metrics") or {}).get("cagr_ex2024"),
        "loyo_median": (report.get("metrics") or {}).get("loyo_median"),
        "completed_trades": (report.get("metrics") or {}).get("completed_trades"),
        "worst_12m": (report.get("metrics") or {}).get("worst_rolling_12m"),
        "flags": [],
    }


def _v2() -> dict[str, Any]:
    return _row(V2)


def score_year_robust(row: dict[str, Any]) -> float:
    """排序键：去2024 / LOYO 优先，不是单看全样本 CAGR。"""
    ex = row.get("cagr_ex2024")
    lo = row.get("loyo_median")
    sh = row.get("sharpe")
    if row.get("low_activity") or "LOW_ACTIVITY" in (row.get("flags") or []):
        return -99.0
    if report_abandoned(row.get("code") or ""):
        return -99.0
    parts = [float(x) for x in (ex, lo, sh) if x is not None]
    return float(sum(parts) / len(parts)) if parts else -99.0


def report_abandoned(code: str) -> bool:
    report = load_report(code) or {}
    return report.get("status") == "ABANDONED"


def module_specs(module: str) -> list[R4Spec]:
    return [s for s in SPECS.values() if module in s.tags and s.stage in {
        "ecology", "margin", "twodim", "cross", "cvar", "fdiv",
    }]


def run_module(module: str, *, predictions, prepared, meta, resume, run_now) -> dict[str, Any]:
    specs = [s for s in SPECS.values() if s.stage == module or (module == "ecology" and s.stage == "ecology")]
    stage_map = {
        "ecology": "ecology",
        "margin": "margin",
        "twodim": "twodim",
        "cvar": "cvar",
    }
    specs = [s for s in SPECS.values() if s.stage == stage_map.get(module, module)]
    results = run_many(specs, predictions=predictions, prepared=prepared, meta=meta, resume=resume, run_now=run_now)
    rows = [_row(s.code) for s in specs]
    payload = {"module": module, "rows": rows, "top2": _top2(rows)}
    _write_json(f"08_single/{module}.json", payload)
    return payload


def _top2(rows: list[dict[str, Any]]) -> list[str]:
    usable = [r for r in rows if r.get("cagr") is not None and not report_abandoned(r.get("code") or "")]
    usable.sort(key=score_year_robust, reverse=True)
    return [r["code"] for r in usable[:2]]


def write_single_summary() -> dict[str, Any]:
    groups = {"B": [], "C": [], "D": [], "E": [], "F": [], "G": []}
    seen = set()
    for spec in list(SPECS.values()):
        if spec.diagnostic or spec.stage in {"parity", "attr", "pair", "triple", "robust"}:
            continue
        if not (ROUND4_ROOT / spec.code / "report.json").is_file():
            continue
        tag_mod = next((t for t in spec.tags if t in groups), None)
        key = tag_mod or spec.replace_module
        if key in groups:
            groups[key].append(_row(spec.code))
            seen.add(spec.code)
    # 动态组（E/F/组合）以磁盘 config 为准，避免新进程丢掉 SPECS
    for cfg in sorted((ROUND4_ROOT / "configs").glob("R4_*.json")):
        code = cfg.stem
        if code in seen or not (ROUND4_ROOT / code / "report.json").is_file():
            continue
        blob = json.loads(cfg.read_text(encoding="utf-8"))
        tags = tuple(blob.get("tags") or ())
        if blob.get("diagnostic") or blob.get("stage") in {"parity", "attr", "pair", "triple", "robust"}:
            continue
        key = next((t for t in tags if t in groups), blob.get("replace_module"))
        if key in groups:
            if code not in SPECS:
                SPECS[code] = spec_from_overlay(
                    code, blob.get("title_zh") or code, blob.get("overlay") or {},
                    description_zh=blob.get("description_zh") or "",
                    replace_module=blob.get("replace_module"),
                    top_k=int(blob.get("top_k") or 1),
                    rank_name_budget_nav_fractions=(
                        tuple(blob["rank_name_budget_nav_fractions"])
                        if blob.get("rank_name_budget_nav_fractions") else None
                    ),
                    stage=blob.get("stage") or "single",
                    tags=tags,
                )
            groups[key].append(_row(code))
    top = {k: _top2(v) for k, v in groups.items()}
    payload = {"groups": groups, "top2": top, "v2": _v2()}
    _write_json("08_single/single.json", payload)
    return payload


def e_monotonic_diagnostic(predictions: pd.DataFrame) -> dict[str, Any]:
    """十分位诊断：至少 3 个相邻桶单调才回测。"""
    xs = cross_section_daily(predictions)
    trades = pd.read_parquet(ROUND4_ROOT / V2 / "trades.parquet")
    trades["signal_date"] = pd.to_datetime(trades["signal_date"])
    trades["ret"] = pd.to_numeric(trades["ret"], errors="coerce")
    merged = trades.merge(xs.reset_index(), on="signal_date", how="left")
    out = {"features": {}, "pass": []}
    for feat in CROSS_FEATURES:
        col = f"{feat}_p250"
        if col not in merged:
            continue
        p = pd.to_numeric(merged[col], errors="coerce")
        bucket = pd.cut(p, bins=10, labels=False, include_lowest=True)
        means = merged.groupby(bucket)["ret"].mean()
        seq = [float(means.get(i)) for i in range(10) if i in means.index and pd.notna(means.get(i))]
        mono = _has_three_adjacent_mono(seq)
        out["features"][feat] = {
            "decile_mean_ret": {str(i): float(means[i]) for i in means.index if pd.notna(means[i])},
            "monotonic_3adj": mono,
        }
        if mono:
            out["pass"].append(feat)
    _write_json("05_cross/diagnostic.json", out)
    return out


def _has_three_adjacent_mono(seq: list[float]) -> bool:
    if len(seq) < 3:
        return False
    for i in range(len(seq) - 2):
        a, b, c = seq[i], seq[i + 1], seq[i + 2]
        if a < b < c or a > b > c:
            return True
    return False


def e_specs(passed: list[str]) -> list[R4Spec]:
    specs = []
    for feat in passed[:3]:
        specs.append(spec_from_overlay(
            f"R4_E_{feat}",
            f"换 C：截面 {feat} 底 20%×0.5",
            stack(B01A, {"kind": "cross_section", "feature": feat}),
            replace_module="C",
            stage="cross",
            tags=("single", "E"),
        ))
        SPECS[specs[-1].code] = specs[-1]
    if len(passed) >= 2:
        parts = [{"kind": "cross_section", "feature": f} for f in passed[:3]]
        specs.append(spec_from_overlay(
            "R4_E_mean",
            "换 C：截面过关因子等权，底 20%×0.5",
            stack(B01A, {"kind": "combine", "how": "mean", "parts": parts}),
            replace_module="C",
            stage="cross",
            tags=("single", "E"),
        ))
        SPECS[specs[-1].code] = specs[-1]
    return specs


def f_specs() -> list[R4Spec]:
    out = []
    for tag, frac in (("1515", (0.15, 0.15)), ("2010", (0.20, 0.10)), ("2005", (0.20, 0.05))):
        spec = spec_from_overlay(
            f"R4_F_{tag}",
            f"分散 Top2 {frac[0]*100:.0f}/{frac[1]*100:.0f}，买不到留现金",
            stack(B01A, C03_HALF),
            top_k=2,
            rank_name_budget_nav_fractions=frac,
            replace_module="F",
            stage="fdiv",
            tags=("single", "F"),
        )
        SPECS[spec.code] = spec
        out.append(spec)
    return out


def pair_specs(top2: dict[str, list[str]], f_ok: bool) -> list[R4Spec]:
    def ov(code: str) -> dict[str, Any]:
        if code in SPECS:
            return SPECS[code].overlay
        cfg = ROUND4_ROOT / "configs" / f"{code}.json"
        if cfg.is_file():
            return json.loads(cfg.read_text(encoding="utf-8")).get("overlay") or stack(B01A, C03_HALF)
        return stack(B01A, C03_HALF)

    def pick(mod: str, i: int = 0) -> str | None:
        lst = top2.get(mod) or []
        return lst[i] if i < len(lst) else None

    planned: list[tuple[str, str, str | None, str | None, str]] = []
    # 优先：生态+分差 / 生态+二维 / 生态+CVaR / 分差+截面 / 分差+CVaR
    pairs = [
        ("B", 0, "C", 0, "生态+连续分差"),
        ("B", 1, "C", 0, "生态2+连续分差"),
        ("B", 0, "C", 1, "生态+连续分差2"),
        ("B", 0, "D", 0, "生态+二维"),
        ("B", 0, "G", 0, "生态+CVaR"),
        ("C", 0, "E", 0, "分差+截面"),
        ("C", 0, "G", 0, "分差+CVaR"),
        ("B", 0, "E", 0, "生态+截面"),
        ("B", 1, "C", 1, "生态2+分差2"),
        ("D", 0, "C", 0, "二维+分差"),
    ]
    if f_ok:
        pairs += [("F", 0, "B", 0, "分散+生态"), ("F", 0, "C", 0, "分散+分差")]
    seen = set()
    specs = []
    for m1, i1, m2, i2, title in pairs:
        if len(specs) >= 12:
            break
        a, b = pick(m1, i1), pick(m2, i2)
        if not a or not b or a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        oa, ob = ov(a), ov(b)
        # 从 overlay 拆 market/confidence，再 stack
        market = (oa.get("market") if oa.get("kind") == "stack" else oa) or B01A
        conf = (ob.get("confidence") if ob.get("kind") == "stack" else ob) or C03_HALF
        if m1 == "B":
            market = (oa.get("market") if oa.get("kind") == "stack" else oa)
        if m2 in {"C", "D", "E"}:
            conf = (ob.get("confidence") if ob.get("kind") == "stack" else ob)
        if m1 in {"C", "D", "E"} and m2 in {"C", "D", "E"}:
            market = B01A
            conf = {"kind": "combine", "how": "product", "parts": [
                (oa.get("confidence") if oa.get("kind") == "stack" else oa),
                (ob.get("confidence") if ob.get("kind") == "stack" else ob),
            ]}
        cvar = None
        cvar_mode = "product"
        if m2 == "G" or m1 == "G":
            g = ob if m2 == "G" else oa
            cvar = (g.get("cvar") if g.get("kind") == "stack" else {"kind": "cvar_rel"})
            cvar_mode = "min"
            if m1 == "B":
                market = (oa.get("market") if oa.get("kind") == "stack" else oa)
                conf = C03_HALF
            if m1 == "C":
                market = B01A
                conf = (oa.get("confidence") if oa.get("kind") == "stack" else oa)
        top_k = 1
        ranks = None
        fspec = None
        if m1 == "F":
            fspec = SPECS.get(a)
        if m2 == "F":
            fspec = SPECS.get(b)
        if fspec is not None:
            top_k = fspec.top_k
            ranks = fspec.rank_name_budget_nav_fractions
        code = f"R4_P_{len(specs)+1:02d}"
        spec = spec_from_overlay(
            code, title, stack(market, conf, cvar, cvar_mode),
            top_k=top_k, rank_name_budget_nav_fractions=ranks,
            stage="pair", tags=("pair",),
        )
        SPECS[code] = spec
        specs.append(spec)
        planned.append((code, title, a, b, cvar_mode))
    _write_json("09_pair/plan.json", {"pairs": planned, "n": len(specs)})
    return specs


def triple_specs(top2: dict[str, list[str]], f_ok: bool) -> list[R4Spec]:
    def pick(mod: str, i: int = 0) -> str | None:
        lst = top2.get(mod) or []
        return lst[i] if i < len(lst) else None

    def part(code: str, layer: str) -> dict[str, Any]:
        ov = SPECS[code].overlay if code in SPECS else {}
        if ov.get("kind") == "stack":
            return ov.get(layer) or {"kind": "identity"}
        return ov

    planned = []
    specs = []

    def add(code: str, title: str, market, conf, cvar=None, mode="product", top_k=1, ranks=None):
        if len(specs) >= 6:
            return
        spec = spec_from_overlay(
            code, title, stack(market, conf, cvar, mode),
            top_k=top_k, rank_name_budget_nav_fractions=ranks,
            stage="triple", tags=("triple",),
        )
        SPECS[code] = spec
        specs.append(spec)
        planned.append({"code": code, "title": title, "cvar_mode": mode})

    b1, c1, d1, e1, g1 = pick("B"), pick("C"), pick("D"), pick("E"), pick("G")
    b2, c2 = pick("B", 1), pick("C", 1)
    if b1 and c1 and g1:
        add("R4_T01", "生态+分差+CVaR min", part(b1, "market"), part(c1, "confidence"),
            part(g1, "cvar") or {"kind": "cvar_rel"}, "min")
        add("R4_T02", "生态+分差+CVaR 几何再 min", part(b1, "market"), part(c1, "confidence"),
            part(g1, "cvar") or {"kind": "cvar_rel"}, "geom_min")
    if b1 and c1 and d1:
        add("R4_T03", "生态+分差+二维", part(b1, "market"),
            {"kind": "combine", "how": "product", "parts": [part(c1, "confidence"), part(d1, "confidence")]})
    if b1 and c1 and e1:
        add("R4_T04", "生态+分差+截面", part(b1, "market"),
            {"kind": "combine", "how": "product", "parts": [part(c1, "confidence"), part(e1, "confidence")]})
    if b1 and d1 and g1:
        add("R4_T05", "生态+二维+CVaR min", part(b1, "market"), part(d1, "confidence"),
            part(g1, "cvar") or {"kind": "cvar_rel"}, "min")
    if f_ok and b1 and c1:
        f1 = pick("F")
        if f1 and SPECS.get(f1):
            add("R4_T06", "生态+分差+分散", part(b1, "market"), part(c1, "confidence"),
                top_k=SPECS[f1].top_k, ranks=SPECS[f1].rank_name_budget_nav_fractions)
    elif b2 and c1 and g1:
        add("R4_T06", "生态2+分差+CVaR min", part(b2, "market"), part(c1, "confidence"),
            part(g1, "cvar") or {"kind": "cvar_rel"}, "min")
    _write_json("10_triple/plan.json", {"triples": planned, "n": len(specs)})
    return specs


def neighbor_specs(top_codes: list[str]) -> list[R4Spec]:
    """邻域粗档，只对 shortlist。"""
    out = []
    for code in top_codes[:4]:
        spec = SPECS.get(code)
        if not spec:
            continue
        ov = spec.overlay
        market = (ov.get("market") if ov.get("kind") == "stack" else None) or B01A
        conf = (ov.get("confidence") if ov.get("kind") == "stack" else None) or C03_HALF
        if market.get("kind") == "eco_relimit_p250":
            for low, tag in ((0.15, "lo"), (0.25, "hi")):
                nb = spec_from_overlay(
                    f"{code}_nbB{tag}", f"{code} 连板分位邻档 {low:.0%}",
                    stack({**market, "low": low}, conf),
                    stage="robust", tags=("neighbor",),
                )
                SPECS[nb.code] = nb
                out.append(nb)
        if conf.get("kind") == "margin_linear":
            lo = float(conf.get("lo", 0.4))
            alt = max(0.3, lo - 0.1)
            nb = spec_from_overlay(
                f"{code}_nbC", f"{code} 线性邻档 {alt}+",
                stack(market, {**conf, "lo": alt}),
                stage="robust", tags=("neighbor",),
            )
            SPECS[nb.code] = nb
            out.append(nb)
        if conf.get("kind") == "margin_tiers":
            nb = spec_from_overlay(
                f"{code}_nbC", f"{code} 五档邻档 0.35 起",
                stack(market, {"kind": "margin_tiers", "mults": (0.35, 0.50, 0.65, 0.85, 1.00)}),
                stage="robust", tags=("neighbor",),
            )
            SPECS[nb.code] = nb
            out.append(nb)
    return out


def cost_specs(bases: list[str]) -> list[R4Spec]:
    out = []
    variants = (
        ("buy10", {"buy_adverse_bps": 10.0}),
        ("buy20", {"buy_adverse_bps": 20.0}),
        ("fee15", {"fee_mult": 1.5}),
        ("fee20", {"fee_mult": 2.0}),
    )
    for base in bases:
        spec = SPECS.get(base)
        if not spec:
            continue
        for tag, ov in variants:
            child = clone_spec(
                spec, code=f"{base}_{tag}", title_zh=f"{spec.title_zh} {tag}",
                cost_overrides=ov, stage="robust", tags=("cost", tag),
            )
            SPECS[child.code] = child
            out.append(child)
    return out


def time_specs(bases: list[str]) -> list[R4Spec]:
    out = []
    for base in bases:
        spec = SPECS.get(base)
        if not spec:
            continue
        child = clone_spec(spec, code=f"{base}_drop2024", title_zh=f"{spec.title_zh} 去2024",
                           skip_years=(2024,), stage="robust", tags=("time", "drop2024"))
        SPECS[child.code] = child
        out.append(child)
        for year in COMPLETE_YEARS:
            child = clone_spec(
                spec, code=f"{base}_loo{year}", title_zh=f"{spec.title_zh} 去{year}",
                skip_years=(year,), stage="robust", tags=("time", "loo"),
            )
            SPECS[child.code] = child
            out.append(child)
    return out


def seed_specs(bases: list[str]) -> list[R4Spec]:
    """4-seed 比例 leave-one-out：只对用了人数/二维的组。"""
    out = []
    for base in bases:
        spec = SPECS.get(base)
        if not spec:
            continue
        blob = json.dumps(spec.overlay)
        if "twodim" not in blob and "confidence_agree" not in blob:
            continue
        for seed in SEEDS:
            child = clone_spec(
                spec, code=f"{base}_looSeed{seed}", title_zh=f"{spec.title_zh} 去seed{seed}",
                seed_leave_out=seed, seeds=SEEDS, stage="robust", tags=("seed",),
            )
            SPECS[child.code] = child
            out.append(child)
    return out


def write_pair() -> dict[str, Any]:
    rows = []
    for p in sorted((ROUND4_ROOT / "09_pair").glob("..")):
        pass
    for code, spec in SPECS.items():
        if spec.stage == "pair" and (ROUND4_ROOT / code / "report.json").is_file():
            row = _row(code)
            if row.get("completed_trades") is not None and int(row["completed_trades"]) < LOW_ACTIVITY_TRADES:
                row["flags"] = list(row.get("flags") or []) + (["LOW_ACTIVITY"] if "LOW_ACTIVITY" not in (row.get("flags") or []) else [])
            rows.append(row)
    payload = {"rows": rows, "v2": _v2()}
    _write_json("09_pair/pair.json", payload)
    return payload


def write_triple() -> dict[str, Any]:
    rows = [_row(code) for code, spec in SPECS.items()
            if spec.stage == "triple" and (ROUND4_ROOT / code / "report.json").is_file()]
    payload = {"rows": rows, "v2": _v2()}
    _write_json("10_triple/triple.json", payload)
    return payload


def write_robust(bases: list[str]) -> dict[str, Any]:
    payload = {"bases": bases, "windows": {}, "drop2024": {}, "loo": {}, "cost": [], "neighbor": {}}
    for base in bases:
        nav_path = ROUND4_ROOT / base / "nav.parquet"
        if nav_path.is_file():
            nav = pd.read_parquet(nav_path)
            payload["windows"][base] = {
                name: window_slice_metrics(nav, start, end) for name, start, end in WINDOWS
            }
        if (ROUND4_ROOT / f"{base}_drop2024" / "report.json").is_file():
            payload["drop2024"][base] = _row(f"{base}_drop2024")
        loo_rows = []
        for year in COMPLETE_YEARS:
            if (ROUND4_ROOT / f"{base}_loo{year}" / "report.json").is_file():
                row = _row(f"{base}_loo{year}")
                row["dropped_year"] = year
                loo_rows.append(row)
        payload["loo"][base] = loo_rows
        for tag in ("buy10", "buy20", "fee15", "fee20"):
            code = f"{base}_{tag}"
            if (ROUND4_ROOT / code / "report.json").is_file():
                row = _row(code)
                row["base"] = base
                row["stress"] = tag
                payload["cost"].append(row)
        nbs = []
        for p in ROUND4_ROOT.glob(f"{base}_nb*"):
            if (p / "report.json").is_file():
                nbs.append(_row(p.name))
        if nbs:
            payload["neighbor"][base] = {
                "center": _row(base),
                "neighbors": nbs,
                "flag": neighbor_fragility(_row(base), nbs),
            }
    _write_json("11_robust/robust.json", payload)
    return payload


def promotion_gate(row: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    """只推荐、不改锁。"""
    checks = {
        "pit": True,
        "cagr": (row.get("cagr") or -1) >= 0.22,
        "sharpe": (row.get("sharpe") or -1) >= 1.35,
        "mdd": (row.get("max_drawdown") or -1) >= -0.18,
        "worst12m": (row.get("worst_12m") or -1) >= -0.12,
        "trades": (row.get("completed_trades") or 0) >= LOW_ACTIVITY_TRADES,
    }
    ex_ok = False
    if row.get("cagr_ex2024") is not None and v2.get("cagr_ex2024") is not None:
        ex_ok = float(row["cagr_ex2024"]) >= float(v2["cagr_ex2024"]) + 0.01
    lo_ok = False
    if row.get("loyo_median") is not None and v2.get("loyo_median") is not None:
        lo_ok = float(row["loyo_median"]) >= float(v2["loyo_median"]) + 0.01
    worst_ok = False
    if row.get("worst_year_return") is not None and v2.get("worst_year_return") is not None:
        worst_ok = float(row["worst_year_return"]) >= float(v2["worst_year_return"]) + 0.02
    year_ok = bool(ex_ok or lo_ok or worst_ok)
    collapse = False
    if row.get("cagr") is not None and v2.get("cagr") is not None:
        collapse = float(row["cagr"]) < float(v2["cagr"]) - 0.04
    if row.get("sharpe") is not None and v2.get("sharpe") is not None:
        collapse = collapse or float(row["sharpe"]) < float(v2["sharpe"]) - 0.15
    meets = all(checks.values()) and year_ok and not collapse and not row.get("low_activity")
    if "LOW_ACTIVITY" in (row.get("flags") or []):
        meets = False
    return {
        "meets": meets,
        "checks": checks,
        "year_ok": year_ok,
        "ex_ok": ex_ok,
        "lo_ok": lo_ok,
        "worst_ok": worst_ok,
        "collapse": collapse,
    }


def shortlist_codes() -> list[str]:
    single = json.loads((ROUND4_ROOT / "08_single" / "single.json").read_text()) if (ROUND4_ROOT / "08_single" / "single.json").is_file() else {}
    codes = [V2]
    for lst in (single.get("top2") or {}).values():
        codes.extend(lst)
    for rel in ("09_pair/pair.json", "10_triple/triple.json"):
        path = ROUND4_ROOT / rel
        if path.is_file():
            for row in json.loads(path.read_text()).get("rows") or []:
                if row.get("code"):
                    codes.append(row["code"])
    # unique, keep order
    seen = set()
    out = []
    for c in codes:
        if c not in seen and (ROUND4_ROOT / c / "report.json").is_file():
            seen.add(c)
            out.append(c)
    return out
