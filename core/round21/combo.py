"""看 NAV 前冻结 interaction registry。Pair≤66 Triple≤36 extras≤12。禁止事后 Best×Best。BLOCKED_PIT 不进正式组。"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from ..paths import ROUND21_ROOT
from .common import dump_json, load_json, sha256_json, write_md
from .conditional import ORTHO_CSV, run_conditional
from .contract import (
    BLOCKED_PIT,
    COMBO_BUDGET,
    COMBO_REG_MD,
    MAX_NEW_FACTORS_IN_MAIN,
    PIT_ELIGIBLE_MODEL,
)
from .freeze import require_frozen
from .redundancy import KEEP_CSV, run_redundancy

REG_CSV = ROUND21_ROOT / "combo" / "experiment_registry.csv"
REG_JSON = ROUND21_ROOT / "combo" / "combo_freeze.json"


def freeze_combo_registry(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if REG_CSV.is_file() and REG_JSON.is_file():
        existing = load_json(REG_JSON)
        if existing.get("status") == "FROZEN":
            return existing
    leaked = []
    cand_root = ROUND21_ROOT / "candidates"
    if cand_root.is_dir():
        leaked = [str(p) for p in cand_root.rglob("*") if p.suffix in {".parquet", ".json"} and p.is_file()]
    if leaked:
        raise RuntimeError(f"REFUSE_COMBO_REGISTRY_AFTER_CANDIDATE_NAV: {leaked[:8]}")
    run_redundancy(resume=True)
    run_conditional(resume=True)
    keep = pd.read_csv(KEEP_CSV)
    ortho = pd.read_csv(ORTHO_CSV) if ORTHO_CSV.is_file() else pd.DataFrame()
    cand = keep.loc[keep["action"].eq("KEEP") & keep["pit_status"].eq("PIT_PASS")].copy()
    cand = cand.sort_values(["best_abs_icir", "best_abs_mean", "name"], ascending=[False, False, True])
    quality = cand.loc[(cand["best_abs_icir"] >= 0.20) | (cand["best_abs_mean"] >= 0.03)]
    selected = quality.head(int(MAX_NEW_FACTORS_IN_MAIN))
    selected_names = [str(x) for x in selected["name"].tolist()]
    fam_of = {str(r["name"]): str(r.get("family") or "") for r in keep.to_dict(orient="records")}
    icir_of = {str(r["name"]): float(r.get("best_abs_icir") or 0.0) for r in keep.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []

    def _add(code: str, role: str, extras: tuple[str, ...], note: str, *, formal: bool = True) -> None:
        rows.append(
            {
                "code": code,
                "role": role,
                "extras": "|".join(extras),
                "n_extras": len(extras),
                "families": "|".join(sorted({fam_of.get(x, "") for x in extras})),
                "includes_blocked_pit": False,
                "formal": formal,
                "model_eligible": PIT_ELIGIBLE_MODEL,
                "trained": False,
                "looked_at_combo_nav": False,
                "note": note,
            }
        )

    budget = dict(COMBO_BUDGET)
    for i, name in enumerate(selected_names[: budget["single"]], start=1):
        _add(f"R21_S{i:02d}", "single", (name,), "single from frozen PIT_PASS KEEP; not post-hoc")
    pair_cands = []
    for a, b in combinations(selected_names, 2):
        pair_cands.append((icir_of.get(a, 0.0) + icir_of.get(b, 0.0), a, b))
    pair_cands.sort(key=lambda t: (-t[0], t[1], t[2]))
    overflow_pairs = max(0, len(pair_cands) - budget["pair"])
    for i, (_, a, b) in enumerate(pair_cands[: budget["pair"]], start=1):
        _add(f"R21_P{i:02d}", "pair", (a, b), "frozen pair; not post-hoc Best×Best")
    trip_cands = []
    for a, b, c in combinations(selected_names, 3):
        trip_cands.append((icir_of.get(a, 0.0) + icir_of.get(b, 0.0) + icir_of.get(c, 0.0), a, b, c))
    trip_cands.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    overflow_trips = max(0, len(trip_cands) - budget["triple"])
    for i, (_, a, b, c) in enumerate(trip_cands[: budget["triple"]], start=1):
        _add(f"R21_T{i:02d}", "triple", (a, b, c), "frozen triple; not post-hoc Best×Best")
    if len(ortho):
        fam_best = (
            ortho.sort_values(["pca_icir", "pruned_icir", "ortho_icir"], ascending=False)
            .groupby("family", as_index=False)
            .head(1)
        )
        i = 0
        for rec in fam_best.to_dict(orient="records"):
            extras = tuple(x for x in str(rec.get("members") or "").split(",") if x)
            extras = tuple(x for x in extras if x in selected_names or x in cand["name"].astype(str).tolist())
            if not extras:
                continue
            i += 1
            if i > budget["ortho_pca_nonlinear"]:
                break
            _add(
                f"R21_R{i:02d}",
                f"family_{rec.get('winner')}",
                extras,
                f"family={rec.get('family')} winner={rec.get('winner')} fold-fit only",
            )
    blocked = keep.loc[keep["pit_status"].eq(BLOCKED_PIT), "name"].astype(str).tolist()
    if any(x in blocked for rec in rows for x in str(rec["extras"]).split("|") if x):
        raise RuntimeError("BLOCKED_PIT leaked into formal combo registry")
    if len(rows) > budget["max_confirm_batch"]:
        overflow_total = len(rows) - budget["max_confirm_batch"]
        rows = rows[: budget["max_confirm_batch"]]
    else:
        overflow_total = overflow_pairs + overflow_trips
    frame = pd.DataFrame(rows)
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
        "blocked_pit_included": False,
        "blocked_pit_ids": blocked,
        "overflow_pairs": overflow_pairs,
        "overflow_triples": overflow_trips,
        "overflow_opens_next_batch": overflow_total > 0,
        "sha256": sha256_json({"selected": selected_names, "rows": rows}),
        "csv": str(REG_CSV),
        "note": "registry 在看正式组合 NAV 之前冻结。本阶段不训练、不读 combo NAV。",
    }
    dump_json(REG_JSON, payload)
    from ..paths import ROUND21_ROOT as _R21

    for name in selected_names:
        dump_json(
            _R21 / "factors" / name / "combination.json",
            {
                "factor_id": name,
                "in_frozen_registry": True,
                "selected_new_factors": selected_names,
                "registry_codes": [r["code"] for r in rows if name in str(r["extras"]).split("|")],
                "blocked_pit": False,
                "looked_at_combo_nav": False,
            },
        )
    for name in [str(x) for x in keep.loc[keep["pit_status"].eq("PIT_PASS"), "name"].tolist() if str(x) not in selected_names]:
        dump_json(
            _R21 / "factors" / name / "combination.json",
            {
                "factor_id": name,
                "in_frozen_registry": False,
                "selected_new_factors": selected_names,
                "reason": "PIT_PASS but below frozen quality bar; still in Add/Replace/Ablation",
                "looked_at_combo_nav": False,
            },
        )
    for name in blocked:
        dump_json(
            _R21 / "factors" / name / "combination.json",
            {
                "factor_id": name,
                "in_frozen_registry": False,
                "blocked_pit": True,
                "status": "BLOCKED_PIT",
                "reason": "BLOCKED_PIT stays out of PIT_ELIGIBLE_MODEL / formal combos",
            },
        )
    lines = [
        "# 07 组合 registry（看 NAV 前冻结）",
        "",
        "本文件在任何组合训练 / 组合 NAV **之前**冻结。禁止事后 Best×Best。本阶段不训练。",
        f"入选新因子 `{len(selected_names)}`（上限 {MAX_NEW_FACTORS_IN_MAIN}，不凑数）："
        + ", ".join(f"`{x}`" for x in selected_names),
        f"registry 行=`{len(frame)}` / 确认批 {budget['max_confirm_batch']}。BLOCKED_PIT 未入选。",
        f"overflow_pairs=`{overflow_pairs}` overflow_triples=`{overflow_trips}` → 下一批次，不在确认批追赢家。",
        "",
        "| code | role | extras |",
        "|---|---|---|",
    ]
    for rec in rows[:50]:
        lines.append(f"| `{rec['code']}` | {rec['role']} | `{rec['extras']}` |")
    if len(rows) > 50:
        lines.append(f"| … | {len(rows)-50} more | see csv |")
    lines += ["", f"csv：`{REG_CSV}`", ""]
    write_md(COMBO_REG_MD, "\n".join(lines))
    return payload


def run_combo(*, resume: bool = True, **kwargs: Any) -> dict[str, Any]:
    """只冻结 registry。训练在 model 阶段，避免看 NAV 后改组。"""
    freeze = freeze_combo_registry(resume=True)
    return {
        "status": "FROZEN",
        "frozen": True,
        "trained": False,
        "n_registry": freeze.get("n_registry"),
        "selected_new_factors": freeze.get("selected_new_factors"),
        "looked_at_combo_nav": False,
        "post_hoc_best_x_best": False,
        "sha256": freeze.get("sha256"),
        "csv": freeze.get("csv"),
    }


__all__ = ["REG_CSV", "REG_JSON", "freeze_combo_registry", "run_combo"]
