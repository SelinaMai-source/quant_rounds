"""C：单因子筛选 + Promotion / |ρ|>0.90 + 每族最多 3 → factor_catalog.csv。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND5_ROOT, assert_project_result_path, ensure_round5_tree
from .factors import (
    EXCLUDE,
    FAMILY_COLUMNS,
    NO_NEW_DEPENDENCY,
    build_factor_panel,
    candidate_factor_names,
    family_of,
)
from .registry import LOCKED_FEATURES, MAX_FAMILY_PICK

COMPLETE_YEARS = tuple(range(2018, 2026))


def _rank_ic_day(block: pd.DataFrame, feat: str, y: str = "label_e2_net") -> float | None:
    if feat not in block or y not in block:
        return None
    s = pd.to_numeric(block[feat], errors="coerce")
    t = pd.to_numeric(block[y], errors="coerce")
    mask = s.notna() & t.notna()
    if mask.sum() < 5 or s[mask].nunique() < 2 or t[mask].nunique() < 2:
        return None
    val = s[mask].corr(t[mask], method="spearman")
    if val is None or not np.isfinite(val):
        return None
    return float(val)


def screen_factor(panel: pd.DataFrame, feat: str) -> dict[str, Any]:
    work = panel.loc[panel["trade_date"].dt.year.between(2018, 2025), ["trade_date", feat, "label_e2_net"]].copy()
    yearly = {}
    ics = []
    for year, block in work.groupby(work["trade_date"].dt.year):
        day_ics = []
        for _, day in block.groupby("trade_date"):
            ic = _rank_ic_day(day, feat)
            if ic is not None:
                day_ics.append(ic)
        mu = float(np.mean(day_ics)) if day_ics else None
        yearly[str(int(year))] = mu
        if mu is not None and int(year) in COMPLETE_YEARS:
            ics.append(mu)
    mean = float(np.mean(ics)) if ics else None
    std = float(np.std(ics, ddof=1)) if len(ics) >= 2 else None
    icir = (mean / std) if mean is not None and std not in (None, 0) else None
    signs = [float(np.sign(x)) for x in ics if x is not None and np.isfinite(x)]
    same = int(abs(sum(signs))) if signs else 0
    coverage = float(pd.to_numeric(panel[feat], errors="coerce").notna().mean()) if feat in panel else 0.0
    first = None
    s = pd.to_numeric(panel[feat], errors="coerce") if feat in panel else pd.Series(dtype=float)
    if s.notna().any():
        first = int(panel.loc[s.notna(), "trade_date"].dt.year.min())
    promote = bool(
        mean is not None
        and mean > 0
        and coverage >= 0.70
        and (same >= 4 or (icir is not None and icir >= 0.30))
        and feat not in EXCLUDE
        and feat not in NO_NEW_DEPENDENCY
    )
    return {
        "factor": feat,
        "family": family_of(feat),
        "coverage": coverage,
        "first_year": first,
        "ic_mean": mean,
        "icir": icir,
        "same_sign_years": same,
        "yearly": yearly,
        "promote": promote,
        "exclude": feat in EXCLUDE,
        "no_new_dependency": feat in NO_NEW_DEPENDENCY,
    }


def _spearman_abs(panel: pd.DataFrame, a: str, b: str) -> float | None:
    if a not in panel or b not in panel:
        return None
    x = pd.to_numeric(panel[a], errors="coerce")
    y = pd.to_numeric(panel[b], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 200:
        return None
    return float(x[mask].corr(y[mask], method="spearman"))


def write_univariate(*, resume: bool = True) -> dict[str, Any]:
    dest = ensure_round5_tree() / "03_univariate"
    dest.mkdir(parents=True, exist_ok=True)
    catalog_path = ROUND5_ROOT / "factor_catalog.csv"
    md = ROUND5_ROOT / "03_univariate_factor_screen.md"
    js = dest / "univariate.json"
    if resume and catalog_path.is_file() and md.is_file() and js.is_file():
        return json.loads(js.read_text(encoding="utf-8"))

    panel = build_factor_panel(resume=True)
    names = candidate_factor_names()
    rows = [screen_factor(panel, f) for f in names if f in panel.columns]
    # 冗余：对已 promote 的，与锁 6 或同族已选 |ρ|>0.90 则丢
    selected: dict[str, list[str]] = {k: [] for k in FAMILY_COLUMNS}
    kept = []
    dropped_red = []
    ranked = sorted(
        [r for r in rows if r.get("promote")],
        key=lambda r: (float(r.get("icir") or 0), float(r.get("ic_mean") or 0)),
        reverse=True,
    )
    for row in ranked:
        fam = row.get("family")
        if fam not in selected:
            continue
        if len(selected[fam]) >= MAX_FAMILY_PICK:
            row = dict(row)
            row["promote"] = False
            row["drop_reason"] = "family_cap"
            dropped_red.append(row)
            continue
        redundant = False
        for other in list(LOCKED_FEATURES) + [x for xs in selected.values() for x in xs]:
            rho = _spearman_abs(panel, row["factor"], other)
            if rho is not None and abs(rho) > 0.90:
                redundant = True
                row = dict(row)
                row["promote"] = False
                row["drop_reason"] = f"rho>{0.90} vs {other}"
                row["rho"] = rho
                dropped_red.append(row)
                break
        if redundant:
            continue
        selected[fam].append(row["factor"])
        kept.append(row)

    shortlist = {fam: selected[fam] for fam in FAMILY_COLUMNS}
    catalog_rows = []
    for row in rows:
        rec = {k: row.get(k) for k in (
            "factor", "family", "coverage", "first_year", "ic_mean", "icir",
            "same_sign_years", "promote", "exclude", "no_new_dependency",
        )}
        rec["shortlisted"] = rec["factor"] in {x for xs in selected.values() for x in xs}
        rec["drop_reason"] = next((d.get("drop_reason") for d in dropped_red if d["factor"] == row["factor"]), None)
        catalog_rows.append(rec)
    catalog = pd.DataFrame(catalog_rows).sort_values(["family", "icir"], ascending=[True, False])
    assert_project_result_path(catalog_path)
    catalog.to_csv(catalog_path, index=False)

    payload = {
        "written_at": datetime.now().astimezone().isoformat(),
        "n_screened": len(rows),
        "shortlist": shortlist,
        "kept": kept,
        "dropped": dropped_red,
        "rows": rows,
        "rules": {
            "promote": "IC均值>0 且 覆盖≥70% 且 (同号年≥4 或 ICIR≥0.30)",
            "redundancy": "|ρ|>0.90 vs 锁6或已选",
            "family_cap": MAX_FAMILY_PICK,
        },
    }
    assert_project_result_path(js)
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        "# 03 单因子筛选",
        "",
        f"- 生成时间：{payload['written_at']}",
        f"- 规则：{payload['rules']['promote']}；冗余 {payload['rules']['redundancy']}；每族最多 {MAX_FAMILY_PICK}。",
        f"- 筛了 {payload['n_screened']} 个候选。",
        "",
        "## Shortlist",
        "",
    ]
    for fam, cols in shortlist.items():
        lines.append(f"- **{fam}**：{', '.join(f'`{c}`' for c in cols) or '（无）'}")
    lines += [
        "",
        "## 目录（节选）",
        "",
        "| 因子 | 族 | 覆盖 | IC | ICIR | 同号年 | 晋级 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for rec in catalog_rows:
        if rec.get("exclude") or rec.get("no_new_dependency"):
            continue
        ic = rec.get("ic_mean")
        icir = rec.get("icir")
        ic_s = "—" if ic is None else f"{float(ic):.3f}"
        icir_s = "—" if icir is None else f"{float(icir):.2f}"
        lines.append(
            f"| `{rec['factor']}` | {rec.get('family')} | {float(rec.get('coverage') or 0):.1%} "
            f"| {ic_s} | {icir_s} | {rec.get('same_sign_years')} | {'是' if rec.get('shortlisted') else '否'} |"
        )
    lines += ["", f"全表：`result/round5/factor_catalog.csv`。", ""]
    assert_project_result_path(md)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def load_shortlist() -> dict[str, list[str]]:
    path = ensure_round5_tree() / "03_univariate" / "univariate.json"
    if not path.is_file():
        return {k: [] for k in FAMILY_COLUMNS}
    return json.loads(path.read_text(encoding="utf-8")).get("shortlist") or {k: [] for k in FAMILY_COLUMNS}
