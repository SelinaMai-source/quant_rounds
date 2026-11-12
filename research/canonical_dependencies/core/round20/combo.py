"""按冻结规则生成 ≤100 组 experiment_registry。不训模型、不看组合 NAV。禁止事后 Best×Best。"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from ..paths import ROUND20_ROOT
from .common import dump_json, load_json, write_md
from .contract import COMBO_BUDGET, COMBO_REG_MD, F1_NAME, F2_NAME, F3_NAME, FIVE_FEATURES, MAX_NEW_FACTORS_IN_MAIN
from .conditional import ORTHO_CSV, run_conditional
from .freeze import require_frozen
from .redundancy import KEEP_CSV, run_redundancy

REG_CSV = ROUND20_ROOT / "combo" / "experiment_registry.csv"
REG_JSON = ROUND20_ROOT / "combo" / "combo_freeze.json"


def freeze_combo_registry(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if REG_CSV.is_file() and REG_JSON.is_file():
        existing = load_json(REG_JSON)
        if existing.get("status") == "FROZEN":
            # 已冻结则永不重写，避免看 NAV 后再生或追加 Best×Best。
            return existing
    leaked = []
    cand_root = ROUND20_ROOT / "candidates"
    if cand_root.is_dir():
        leaked = [str(p) for p in cand_root.rglob("*") if p.suffix in {".parquet", ".json"} and p.is_file()]
    if leaked:
        raise RuntimeError(f"REFUSE_COMBO_REGISTRY_AFTER_CANDIDATE_NAV: {leaked[:8]}")
    run_redundancy(resume=True)
    run_conditional(resume=True)
    keep = pd.read_csv(KEEP_CSV)
    ortho = pd.read_csv(ORTHO_CSV) if ORTHO_CSV.is_file() else pd.DataFrame()
    cand = keep.loc[keep["action"].isin(["KEEP", "KEEP_AUDIT"]) & keep["name"].ne(F2_NAME)].copy()
    cand = cand.loc[~cand["name"].isin(list(FIVE_FEATURES))]
    cand = cand.sort_values(["best_abs_icir", "best_abs_mean", "name"], ascending=[False, False, True])
    # 质量不够就少用，不凑满 12
    quality = cand.loc[(cand["best_abs_icir"] >= 0.20) | (cand["best_abs_mean"] >= 0.03)]
    selected = quality.head(int(MAX_NEW_FACTORS_IN_MAIN))
    selected_names = [str(x) for x in selected["name"].tolist()]
    fam_of = {str(r["name"]): str(r.get("family") or "") for r in keep.to_dict(orient="records")}
    icir_of = {str(r["name"]): float(r.get("best_abs_icir") or 0.0) for r in keep.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []

    def _add(code: str, role: str, extras: tuple[str, ...], note: str) -> None:
        rows.append(
            {
                "code": code,
                "role": role,
                "extras": "|".join(extras),
                "n_extras": len(extras),
                "families": "|".join(sorted({fam_of.get(x, "") for x in extras})),
                "includes_f2": F2_NAME in extras,
                "includes_f1_or_f3": any(x in {F1_NAME, F3_NAME} for x in extras),
                "rebrand_f1_f3": False,
                "trained": False,
                "looked_at_combo_nav": False,
                "note": note,
            }
        )

    budget = dict(COMBO_BUDGET)
    for i, name in enumerate(selected_names[: budget["single"]], start=1):
        tag = "KEEP_AUDIT 不换名" if name in {F1_NAME, F3_NAME} else "single from frozen KEEP"
        _add(f"R20_S{i:02d}", "single", (name,), tag)
    pair_cands = []
    for a, b in combinations(selected_names, 2):
        pair_cands.append((icir_of.get(a, 0.0) + icir_of.get(b, 0.0), a, b))
    pair_cands.sort(key=lambda t: (-t[0], t[1], t[2]))
    for i, (_, a, b) in enumerate(pair_cands[: budget["pair"]], start=1):
        _add(f"R20_P{i:02d}", "pair", (a, b), "frozen pair; not post-hoc Best×Best")
    trip_cands = []
    for a, b, c in combinations(selected_names, 3):
        trip_cands.append((icir_of.get(a, 0.0) + icir_of.get(b, 0.0) + icir_of.get(c, 0.0), a, b, c))
    trip_cands.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    for i, (_, a, b, c) in enumerate(trip_cands[: budget["triple"]], start=1):
        _add(f"R20_T{i:02d}", "triple", (a, b, c), "frozen triple; not post-hoc Best×Best")
    cross = []
    for a, b in combinations(selected_names, 2):
        if fam_of.get(a) and fam_of.get(b) and fam_of[a] != fam_of[b]:
            cross.append((icir_of.get(a, 0.0) + icir_of.get(b, 0.0), a, b))
    cross.sort(key=lambda t: (-t[0], t[1], t[2]))
    for i, (_, a, b) in enumerate(cross[: budget["cross_family"]], start=1):
        _add(f"R20_X{i:02d}", "cross_family", (a, b), "cross-family pair listed before NAV")
    # pruned/ortho/PCA 代表：每个家族 winner 一条，上限 12
    if len(ortho):
        fam_best = (
            ortho.sort_values(["pca_icir", "pruned_icir", "ortho_icir"], ascending=False)
            .groupby("family", as_index=False)
            .head(1)
        )
        i = 0
        for rec in fam_best.to_dict(orient="records"):
            extras = tuple(x for x in str(rec.get("members") or "").split(",") if x)
            if not extras:
                continue
            i += 1
            if i > budget["pruned_ortho_pca"]:
                break
            _add(
                f"R20_R{i:02d}",
                f"family_{rec.get('winner')}",
                extras,
                f"family={rec.get('family')} winner={rec.get('winner')} fold-fit only",
            )
    if len(rows) > budget["max_total"]:
        rows = rows[: budget["max_total"]]
    frame = pd.DataFrame(rows)
    if frame["includes_f2"].any():
        raise RuntimeError("F2 leaked into combo registry")
    REG_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REG_CSV, index=False)
    payload = {
        "status": "FROZEN",
        "frozen_before_combo_nav": True,
        "trained": False,
        "looked_at_combo_nav": False,
        "post_hoc_best_x_best": False,
        "n_selected_new_factors": len(selected_names),
        "selected_new_factors": selected_names,
        "n_registry": int(len(frame)),
        "by_role": frame["role"].value_counts().to_dict() if len(frame) else {},
        "budget": budget,
        "padded_to_quota": False,
        "f2_included": False,
        "f1_f3_rebranded": False,
        "csv": str(REG_CSV),
        "note": "registry 在看正式组合 NAV 之前冻结。本阶段不训练、不读 combo NAV。",
    }
    dump_json(REG_JSON, payload)
    lines = [
        "# 07 组合 registry（看 NAV 前冻结）",
        "",
        "本文件在任何组合训练 / 组合 NAV **之前**冻结。禁止事后 Best×Best。本阶段不训练。",
        f"入选新因子 `{len(selected_names)}`（上限 {MAX_NEW_FACTORS_IN_MAIN}，不凑数）："
        + ", ".join(f"`{x}`" for x in selected_names),
        f"registry 行=`{len(frame)}` / 预算 {budget['max_total']}。F2 未入选。F1/F3 不换名。",
        "",
        "| code | role | extras |",
        "|---|---|---|",
    ]
    for rec in rows[:40]:
        lines.append(f"| `{rec['code']}` | {rec['role']} | `{rec['extras']}` |")
    if len(rows) > 40:
        lines.append(f"| … | {len(rows)-40} more | see csv |")
    lines += ["", f"csv：`{REG_CSV}`", ""]
    write_md(COMBO_REG_MD, "\n".join(lines))
    return payload


def run_combo(*, resume: bool = True, **kwargs: Any) -> dict[str, Any]:
    """冻结 registry（若已冻则原样返回）后跑 walk-forward 训练。禁止事后加组。"""
    freeze = freeze_combo_registry(resume=True)
    from .model import freeze_model_matrix
    from .runner import run_combo_train

    matrix = freeze_model_matrix(resume=True)
    trained = run_combo_train(resume=resume, **kwargs)
    return {
        "status": trained.get("status") or "DONE",
        "frozen": True,
        "trained": True,
        "n_registry": freeze.get("n_registry"),
        "n_tested": trained.get("n_tested"),
        "n_with_metrics": trained.get("n_with_metrics"),
        "n_skip": trained.get("n_skip"),
        "best": trained.get("best"),
        "selected_new_factors": freeze.get("selected_new_factors"),
        "looked_at_combo_nav": True,
        "post_hoc_best_x_best": False,
        "model_matrix_frozen": matrix.get("status") == "FROZEN",
        "csv": trained.get("csv"),
    }


__all__ = ["REG_CSV", "REG_JSON", "freeze_combo_registry", "run_combo"]
