"""预注册 6 条经济组合。预算写入 manifest，禁止事后加乘。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_2_ROOT, ROUND21_3_ROOT
from .common import dump_json, load_json, write_md
from .contract import INTERACTION_CAP, MD, PREREGISTERED_INTERACTIONS
from .pit import PIT_JSON
from .train import train_one

INT_JSON = ROUND21_3_ROOT / "interaction" / "interactions.json"


def run_interact(*, resume: bool = True, jobs: int = 4, **_: Any) -> dict[str, Any]:
    if resume and INT_JSON.is_file() and MD["12"].is_file():
        existing = load_json(INT_JSON)
        if existing.get("status") == "DONE":
            return existing
    pit = load_json(PIT_JSON)
    passed = list(pit.get("pit_pass") or [])
    # 预注册 T-L / U-K / V-L 需要 21.2 已 PIT_PASS 的 K/L，不是本轮新 ID
    safe_21_2 = load_json(ROUND21_2_ROOT / "data" / "pit_safe_factors.json")
    for fid in safe_21_2.get("factors") or []:
        fam = str(fid).split("_")[1][0].upper() if "_" in str(fid) else "?"
        if fam in {"K", "L"} and fid not in passed:
            passed.append(fid)
    by_fam: dict[str, list[str]] = {}
    for fid in passed:
        fam = fid.split("_")[1][0].upper() if "_" in fid else "?"
        by_fam.setdefault(fam, []).append(fid)
    planned = []
    for a, b in PREREGISTERED_INTERACTIONS:
        fa = by_fam.get(a) or []
        fb = by_fam.get(b) or []
        if not fa or not fb:
            planned.append({"pair": [a, b], "status": "SKIPPED", "reason": "family missing PIT_PASS"})
            continue
        left, right = fa[0], fb[0]
        if left == right:
            planned.append({"pair": [a, b], "status": "SKIPPED", "reason": "same factor"})
            continue
        name = f"{left}__{right}"
        spec = {
            "code": f"R21_3_E6_{name}",
            "role": "E6_INTERACT",
            "factor_id": name,
            "five_extras": (left, right),
            "ten_extras": (left, right),
            "five_drops": (),
            "ten_drops": (),
            "formal": True,
            "head": "both_90_10",
            "alias_from": None,
            "force_retrain": True,
        }
        planned.append({"pair": [a, b], "left": left, "right": right, "spec": spec, "status": "PLANNED"})
    if len(planned) > INTERACTION_CAP:
        raise RuntimeError("interaction budget exceeded; do not add post hoc")
    rows = []
    for rec in planned:
        spec = rec.get("spec")
        if not spec:
            rows.append(rec)
            continue
        print(f"[round21.3] INTERACT {spec['code']}", flush=True)
        try:
            oof = train_one(spec, resume=True)
            rows.append({**rec, "status": oof.get("status"), "code": spec["code"], "five_oof": oof.get("five_oof"), "ten_oof": oof.get("ten_oof")})
        except Exception as exc:
            rows.append({**rec, "status": "FAILED", "reason": str(exc)})
    payload = {
        "status": "DONE",
        "n_planned": len(planned),
        "n_run": sum(1 for r in rows if r.get("status") in {"OOF", "DONE"}),
        "budget": INTERACTION_CAP,
        "no_post_hoc": True,
        "rows": rows,
    }
    dump_json(INT_JSON, payload)
    write_md(MD["12"], f"# Round21.3 · 12 交互\n\n预注册 {len(PREREGISTERED_INTERACTIONS)} 条，预算 `{INTERACTION_CAP}`。禁止事后加乘。完成=`{payload['n_run']}`。\n")
    return payload
