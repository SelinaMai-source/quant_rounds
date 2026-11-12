"""A00/A01 + B/C 代表 + M0/M1/M2。Best = 冻结筛选代表，非独立 OOS。看 NAV 前冻结矩阵。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND20_ROOT
from ..round7.contract import FIVE_FEATURES, TEN_FEATURES
from .common import dump_json, load_json, write_md
from .conditional import COND_JSON, ORTHO_CSV
from .contract import (
    A00,
    A01,
    B01,
    B02,
    B03,
    B04,
    C01,
    C02,
    C03,
    C04,
    D01,
    D02,
    D03,
    DIAGNOSTIC_ONLY,
    F2_NAME,
    MODEL_MD,
    SKIP_DEPENDENCY,
    V5_ANCHOR,
)
from .freeze import require_frozen
from .runner import parse_extras, run_one
from .train import new_columns, probe_lightgbm

MATRIX_JSON = ROUND20_ROOT / "configs" / "model_matrix.json"
MODEL_JSON = ROUND20_ROOT / "model" / "model_selection.json"
MODEL_CSV = ROUND20_ROOT / "model" / "abcd_matrix.csv"


def _members(raw: Any) -> tuple[str, ...]:
    return parse_extras(raw)


def _abs(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return -1.0
    return abs(x) if np.isfinite(x) else -1.0


def _family_score(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for fam, part in frame.groupby("family"):
        rows.append({
            "family": fam,
            "score": float(part[col].map(_abs).mean()) if col in part.columns else -1.0,
            "winner": str(part["winner"].iloc[0]) if "winner" in part.columns else "",
            "members": str(part["members"].iloc[0]) if "members" in part.columns else "",
        })
    return pd.DataFrame(rows)


def freeze_model_matrix(*, resume: bool = True) -> dict[str, Any]:
    """在看模型 NAV 前按冻结规则钉死 A/B/C/D 代表。禁止事后 Best×Best。"""
    require_frozen()
    if MATRIX_JSON.is_file():
        existing = load_json(MATRIX_JSON)
        if existing.get("status") == "FROZEN" and existing.get("frozen_before_model_nav"):
            return existing
    from .combo import REG_CSV, REG_JSON

    freeze = load_json(REG_JSON)
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("combo registry not FROZEN; refuse model matrix")
    selected = list(freeze.get("selected_new_factors") or [])
    registry = pd.read_csv(REG_CSV)
    by_role = {}
    for rec in registry.to_dict(orient="records"):
        by_role.setdefault(str(rec.get("role")), []).append(rec)
    c01 = (by_role.get("single") or [None])[0]
    c02 = (by_role.get("pair") or [None])[0]
    c03 = (by_role.get("triple") or [None])[0]
    c04 = (by_role.get("cross_family") or [None])[0]
    ortho = pd.read_csv(ORTHO_CSV) if ORTHO_CSV.is_file() else pd.DataFrame()
    cond = load_json(COND_JSON)
    leaders = list(cond.get("leaders") or [])

    def _pick_transform(winner: str, score_col: str) -> dict[str, Any] | None:
        if ortho.empty:
            return None
        sub = ortho.loc[ortho["winner"].astype(str).eq(winner)].copy()
        if sub.empty:
            return None
        scored = _family_score(sub, score_col).sort_values(["score", "family"], ascending=[False, True])
        for rec in scored.to_dict(orient="records"):
            extras = _members(rec.get("members"))
            extras = tuple(x for x in extras if x != F2_NAME)
            if not extras:
                continue
            if not new_columns(list(FIVE_FEATURES), extras) and not new_columns(list(TEN_FEATURES), extras):
                continue
            return {"family": rec.get("family"), "extras": extras, "score": rec.get("score"), "winner": winner}
        return None

    b01 = _pick_transform("pruned", "pruned_icir")
    b03 = _pick_transform("ortho", "ortho_icir")
    b04 = _pick_transform("pca", "pca_icir")
    b02 = None
    keep_actions = {"KEEP", "KEEP_AUDIT"}
    ranked = []
    for row in leaders:
        if str(row.get("raw_action") or "") not in keep_actions:
            continue
        name = str(row.get("factor") or "")
        if not name or name == F2_NAME:
            continue
        if name in FIVE_FEATURES or name in TEN_FEATURES:
            continue
        ranked.append((abs(float(row.get("cond_rankicir") or 0.0)), name, row))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    if ranked:
        b02 = {"factor": ranked[0][1], "extras": (ranked[0][1],), "score": ranked[0][0], "action": ranked[0][2].get("raw_action")}

    def _alias_for(extras: tuple[str, ...]) -> str | None:
        want = tuple(sorted(extras))
        for rec in registry.to_dict(orient="records"):
            got = tuple(sorted(_members(rec.get("extras"))))
            if got == want:
                return str(rec["code"])
        return None

    specs: list[dict[str, Any]] = [
        {"code": A00, "role": "exact_v5_replay", "extras": (), "five_extras": (), "ten_extras": (), "model": "locked_v5", "force_retrain": False, "formal": True, "head": "both_90_10"},
        {"code": A01, "role": "force_retrain_original_14_13", "extras": (), "five_extras": (), "ten_extras": (), "model": "M0", "force_retrain": True, "formal": True, "head": "both_90_10"},
    ]

    def _add_b(code: str, role: str, picked: dict[str, Any] | None) -> None:
        extras = tuple(picked.get("extras") or ()) if picked else ()
        spec = {
            "code": code,
            "role": role,
            "extras": extras,
            "five_extras": extras,
            "ten_extras": extras,
            "model": "M0",
            "force_retrain": False,
            "formal": True,
            "head": "both_90_10",
            "frozen_pick": picked,
            "alias_of": _alias_for(extras) if extras else None,
        }
        if not extras:
            spec["status_pre"] = "SKIP_NO_REPRESENTATIVE"
        specs.append(spec)

    _add_b(B01, "pruned_representative", b01)
    _add_b(B02, "cond_ic_representative", b02)
    _add_b(B03, "ortho_representative", b03)
    _add_b(B04, "pca_representative", b04)

    def _add_c(code: str, role: str, rec: dict[str, Any] | None) -> None:
        extras = _members(rec.get("extras") if rec else ())
        specs.append({
            "code": code,
            "role": role,
            "extras": extras,
            "five_extras": extras,
            "ten_extras": extras,
            "model": "M0",
            "force_retrain": False,
            "formal": True,
            "head": "both_90_10",
            "alias_of": rec.get("code") if rec else None,
            "note": "Best = frozen selection-rule representative, not independent OOS",
        })

    _add_c(C01, "registry_best_single", c01)
    _add_c(C02, "registry_best_pair", c02)
    _add_c(C03, "registry_best_triple", c03)
    _add_c(C04, "registry_best_cross_family", c04)

    sel = tuple(selected)
    specs.append({"code": D01, "role": "M0_xgb_selected_factors", "extras": sel, "five_extras": sel, "ten_extras": sel, "model": "M0", "force_retrain": False, "formal": True, "head": "both_90_10"})
    specs.append({"code": D02, "role": "M1_elasticnet_rank_proxy", "extras": sel, "five_extras": sel, "ten_extras": sel, "model": "M1", "force_retrain": False, "formal": True, "head": "both_90_10", "loss_ne_ndcg": True})
    specs.append({"code": D03, "role": "M2_lightgbm_lambdarank", "extras": sel, "five_extras": sel, "ten_extras": sel, "model": "M2", "force_retrain": False, "formal": True, "head": "both_90_10"})

    c01_ex = _members(c01.get("extras") if c01 else ())
    specs.extend([
        {"code": "R20_ATTR_C01_FIVE", "role": "attr_five_only_C01", "extras": c01_ex, "five_extras": c01_ex, "ten_extras": (), "model": "M0", "formal": False, "head": "five_only", "diagnostic": DIAGNOSTIC_ONLY},
        {"code": "R20_ATTR_C01_TEN", "role": "attr_ten_only_C01", "extras": c01_ex, "five_extras": (), "ten_extras": c01_ex, "model": "M0", "formal": False, "head": "ten_only", "diagnostic": DIAGNOSTIC_ONLY},
        {"code": "R20_ATTR_D01_FIVE", "role": "attr_five_only_selected", "extras": sel, "five_extras": sel, "ten_extras": (), "model": "M0", "formal": False, "head": "five_only", "diagnostic": DIAGNOSTIC_ONLY},
        {"code": "R20_ATTR_D01_TEN", "role": "attr_ten_only_selected", "extras": sel, "five_extras": (), "ten_extras": sel, "model": "M0", "formal": False, "head": "ten_only", "diagnostic": DIAGNOSTIC_ONLY},
    ])
    payload = {
        "status": "FROZEN",
        "frozen_before_model_nav": True,
        "post_hoc_best_x_best": False,
        "selected_new_factors": selected,
        "b01": b01,
        "b02": b02,
        "b03": b03,
        "b04": b04,
        "c01": c01.get("code") if c01 else None,
        "c02": c02.get("code") if c02 else None,
        "c03": c03.get("code") if c03 else None,
        "c04": c04.get("code") if c04 else None,
        "lgbm_probe_at_freeze": probe_lightgbm(),
        "specs": specs,
        "note": "C01–C04 = first single/pair/triple/cross_family in frozen registry (ICIR order). B01–B04 from frozen pruned/cond-IC/ortho/PCA rules. Formal candidates keep 90/10. five/ten-only are DIAGNOSTIC_ONLY attribution.",
    }
    dump_json(MATRIX_JSON, payload)
    return payload


def _finalists(rows: list[dict[str, Any]], a00: dict[str, Any]) -> list[dict[str, Any]]:
    a_sh = a00.get("sharpe")
    a_cagr = a00.get("cagr")
    a_dd = a00.get("max_drawdown")
    try:
        a_sh_f = float(a_sh)
        a_cagr_f = float(a_cagr)
        a_dd_f = float(a_dd)
    except (TypeError, ValueError):
        a_sh_f, a_cagr_f, a_dd_f = float(V5_ANCHOR["sharpe"]), float(V5_ANCHOR["cagr"]), float(V5_ANCHOR["max_drawdown"])
    pool = []
    for r in rows:
        if not r.get("formal"):
            continue
        if r.get("code") == A00:
            continue
        if r.get("sharpe") is None:
            continue
        if str(r.get("status") or "").startswith("SKIP") or r.get("status") == SKIP_DEPENDENCY:
            continue
        sh = float(r["sharpe"])
        cagr = float(r.get("cagr") or -999)
        dd = float(r.get("max_drawdown") or 0.0)
        better = sh > a_sh_f + 1e-12 or cagr > a_cagr_f + 1e-12
        competitive = sh >= a_sh_f - 0.02
        dd_ok = dd >= a_dd_f - 0.02
        if better and competitive and dd_ok:
            pool.append(r)
    pool.sort(key=lambda r: (-float(r.get("sharpe") or -999), -float(r.get("cagr") or -999)))
    out = []
    for r in pool[:3]:
        out.append({
            "code": r.get("code"),
            "role": r.get("role"),
            "model": r.get("model"),
            "cagr": r.get("cagr"),
            "sharpe": r.get("sharpe"),
            "max_drawdown": r.get("max_drawdown"),
            "completed_trades": r.get("completed_trades"),
            "nav_end": r.get("nav_end"),
            "note": "not independent OOS; selection-bias disclosed",
        })
    return out


def _parity_a00_a01(a00: dict[str, Any], a01: dict[str, Any]) -> dict[str, Any]:
    from .common import close_enough
    from .contract import NAV_END_ABS_TOL, PARITY_METRIC_ABS

    checks = {
        "cagr": close_enough(a00.get("cagr"), a01.get("cagr"), abs_tol=PARITY_METRIC_ABS),
        "sharpe": close_enough(a00.get("sharpe"), a01.get("sharpe"), abs_tol=PARITY_METRIC_ABS),
        "max_drawdown": close_enough(a00.get("max_drawdown"), a01.get("max_drawdown"), abs_tol=PARITY_METRIC_ABS),
        "completed_trades": (a00.get("completed_trades") is not None and a00.get("completed_trades") == a01.get("completed_trades")),
        "nav_end": close_enough(a00.get("nav_end"), a01.get("nav_end"), abs_tol=NAV_END_ABS_TOL),
    }
    return {
        "agree": bool(all(checks.values())),
        "checks": checks,
        "a00": {k: a00.get(k) for k in ("cagr", "sharpe", "max_drawdown", "completed_trades", "nav_end", "status")},
        "a01": {k: a01.get(k) for k in ("cagr", "sharpe", "max_drawdown", "completed_trades", "nav_end", "status")},
        "note": "A00=locked v5 replay; A01=force_retrain original 14/13 through v5 engine. Rank-invariance at 1.0/1.0 does not imply NAV identity vs locked overlay.",
    }


def run_model_selection(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    require_frozen()
    matrix = freeze_model_matrix(resume=True)
    specs = list(matrix.get("specs") or [])
    from .contract import XGB_N_JOBS
    from .runner import init_panels, prepare_panels

    prepare_panels(resume=True)
    init_panels()
    xgb_jobs = 1 if int(jobs) > 1 else XGB_N_JOBS
    rows: list[dict[str, Any]] = []
    if int(jobs) <= 1:
        for spec in specs:
            if spec.get("status_pre") == "SKIP_NO_REPRESENTATIVE":
                from .runner import skip_payload

                rows.append(skip_payload(spec["code"], spec, "no frozen representative", "SKIP_NO_REPRESENTATIVE"))
                continue
            print(f"[round20/model] start {spec['code']} role={spec.get('role')}", flush=True)
            rows.append(run_one(dict(spec), resume=resume, n_jobs=xgb_jobs))
            print(f"[round20/model] done {spec['code']} status={rows[-1].get('status')} sharpe={rows[-1].get('sharpe')}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .runner import _job, skip_payload

        runnable = []
        for spec in specs:
            if spec.get("status_pre") == "SKIP_NO_REPRESENTATIVE":
                rows.append(skip_payload(spec["code"], spec, "no frozen representative", "SKIP_NO_REPRESENTATIVE"))
            else:
                runnable.append(spec)
        with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
            futs = {pool.submit(_job, dict(spec), xgb_jobs): spec["code"] for spec in runnable}
            extra = [fut.result() for fut in as_completed(futs)]
        order = {spec["code"]: i for i, spec in enumerate(specs)}
        rows.extend(extra)
        rows.sort(key=lambda r: order.get(r.get("code"), 99))

    by_code = {r.get("code"): r for r in rows}
    a00 = by_code.get(A00) or {}
    a01 = by_code.get(A01) or {}
    parity = _parity_a00_a01(a00, a01)
    lgbm = probe_lightgbm()
    d03 = by_code.get(D03) or {}
    m2_status = d03.get("status") if d03 else SKIP_DEPENDENCY
    if not lgbm.get("available"):
        m2_status = SKIP_DEPENDENCY
    finalists = _finalists(rows, a00)
    slim = []
    for r in rows:
        slim.append({
            "code": r.get("code"),
            "role": r.get("role"),
            "formal": r.get("formal"),
            "model": r.get("model") or r.get("model_family"),
            "status": r.get("status"),
            "reason": r.get("reason"),
            "alias_of": r.get("alias_of"),
            "cagr": r.get("cagr"),
            "sharpe": r.get("sharpe"),
            "max_drawdown": r.get("max_drawdown"),
            "completed_trades": r.get("completed_trades"),
            "nav_end": r.get("nav_end"),
            "loss_ne_ndcg": r.get("loss_ne_ndcg"),
            "head": None,
        })
    spec_head = {s["code"]: s.get("head") for s in specs}
    for row in slim:
        row["head"] = spec_head.get(row["code"])
    MODEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(slim).to_csv(MODEL_CSV, index=False)
    payload = {
        "status": "DONE",
        "n_specs": len(specs),
        "n_ran": len(rows),
        "a00_a01_parity": parity,
        "m2_status": m2_status,
        "lgbm_probe": lgbm,
        "finalists": finalists,
        "abcd": [{k: (by_code.get(c) or {}).get(k) for k in (
            "code", "role", "status", "reason", "alias_of", "cagr", "sharpe", "max_drawdown",
            "completed_trades", "nav_end", "model", "loss_ne_ndcg",
        )} | {"code": c} for c in (A00, A01, B01, B02, B03, B04, C01, C02, C03, C04, D01, D02, D03)],
        "rows": slim,
        "csv": str(MODEL_CSV),
        "matrix": str(MATRIX_JSON),
        "auto_promote_current": False,
        "note": "formal candidates 90/10 only; five/ten-only = DIAGNOSTIC_ONLY attribution. D02 loss ≠ NDCG. D03 = LightGBM or SKIP_DEPENDENCY, no silent swap.",
    }
    dump_json(MODEL_JSON, payload)
    lines = [
        "# 09 模型对照 A/B/C/D",
        "",
        "A00 = exact v5 replay（Phase A parity 复制确认）。A01 = 原 14/13 `force_retrain`。",
        "B01–B04 = 冻结 pruned / cond-IC / ortho / PCA 代表。C01–C04 = 冻结 registry 筛选代表（非独立 OOS）。",
        "D01 = XGB + 12 selected；D02 = ElasticNet 排序代理（损失≠NDCG）；D03 = LightGBM LambdaRank 或 SKIP_DEPENDENCY。",
        "正式候选保持 90/10。five-only / ten-only 仅预注册归因。",
        "",
        f"A00 vs A01 agree=`{parity.get('agree')}`",
        f"M2 status=`{m2_status}` lightgbm=`{lgbm}`",
        f"finalists=`{[f.get('code') for f in finalists]}`",
        "",
        "| code | role | formal | model | status | CAGR | Sharpe | MaxDD | trades | alias |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in slim:
        if r["code"] in {A00, A01, B01, B02, B03, B04, C01, C02, C03, C04, D01, D02, D03} or str(r["code"]).startswith("R20_ATTR"):
            lines.append(
                f"| `{r.get('code')}` | {r.get('role')} | {r.get('formal')} | {r.get('model')} | {r.get('status')} | "
                f"{r.get('cagr')} | {r.get('sharpe')} | {r.get('max_drawdown')} | {r.get('completed_trades')} | `{r.get('alias_of')}` |"
            )
    write_md(MODEL_MD, "\n".join(lines))
    return payload


__all__ = [
    "MATRIX_JSON",
    "MODEL_CSV",
    "MODEL_JSON",
    "freeze_model_matrix",
    "run_model_selection",
]
