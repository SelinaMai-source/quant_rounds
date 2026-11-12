"""预注册跨家族交互。上限 120。禁止按 NAV 追加。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND21_2_ROOT
from .common import dump_json, write_md
from .contract import INTERACTION_CAP, MD, PREREGISTERED_INTERACTIONS, family_of
from .panels import pit_safe_factors
from .train import train_one

INT_JSON = ROUND21_2_ROOT / "interaction" / "interaction.json"
INT_CSV = ROUND21_2_ROOT / "interaction_registry.csv"


def run_interact(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    if resume and INT_JSON.is_file() and MD["12"].is_file():
        from .common import load_json

        existing = load_json(INT_JSON)
        if existing.get("status") == "DONE":
            return existing
    factors = pit_safe_factors(resume=True)
    by_fam: dict[str, list[str]] = {}
    for f in factors:
        by_fam.setdefault(family_of(f), []).append(f)
    specs = []
    for a, b in PREREGISTERED_INTERACTIONS:
        fa, fb = by_fam.get(a) or [], by_fam.get(b) or []
        if not fa or not fb:
            continue
        # 每家族取最多 2 个代表，控制预算
        for x in fa[:2]:
            for y in fb[:2]:
                if x == y:
                    continue
                code = f"R21_2_E6_{x}__{y}"
                specs.append(
                    {
                        "code": code,
                        "role": "E6_INTERACTION_ADD",
                        "factor_id": f"{x}+{y}",
                        "five_extras": (x, y),
                        "ten_extras": (x, y),
                        "five_drops": (),
                        "ten_drops": (),
                        "force_retrain": True,
                        "formal": True,
                        "preregistered_pair": (a, b),
                        "alias_from": None,
                    }
                )
                if len(specs) >= INTERACTION_CAP:
                    break
            if len(specs) >= INTERACTION_CAP:
                break
        if len(specs) >= INTERACTION_CAP:
            break
    rows = []
    xgb_jobs = 1 if int(jobs) > 1 else 2
    for spec in specs:
        print(f"[round21.2] INTERACT {spec['code']}", flush=True)
        try:
            rows.append(train_one(spec, resume=resume, n_jobs=xgb_jobs))
        except Exception as exc:
            rows.append({"code": spec["code"], "status": "FAILED", "error": str(exc)})
    INT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "code": r.get("code"),
                "role": r.get("role"),
                "five_oof": (r.get("five_oof") or {}).get("oof_rankic_mean"),
                "ten_oof": (r.get("ten_oof") or {}).get("oof_rankic_mean"),
                "status": r.get("status"),
            }
            for r in rows
        ]
    ).to_csv(INT_CSV, index=False)
    payload = {
        "status": "DONE",
        "n_planned": len(specs),
        "n_run": len(rows),
        "cap": INTERACTION_CAP,
        "preregistered": [list(p) for p in PREREGISTERED_INTERACTIONS],
        "no_nav_driven_append": True,
        "rows": [{"code": r.get("code"), "status": r.get("status")} for r in rows],
    }
    dump_json(INT_JSON, payload)
    write_md(
        MD["12"],
        "\n".join(
            [
                "# Round21.2 · 12 预注册交互",
                "",
                f"计划 N=`{len(specs)}` 上限=`{INTERACTION_CAP}`。只由 PIT-valid 因子构成。不按 NAV 追加。",
                f"机制对：`{list(PREREGISTERED_INTERACTIONS)}`",
                "",
            ]
        ),
    )
    return payload
