"""Seed 仅当预注册 OOF 增量门通过。否则 NOT_TRIGGERED。禁止从 100/300 里挑最好 seed。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_ROOT
from .common import dump_json, load_json, write_md
from .contract import NOT_TRIGGERED, SEED_LADDER, SEED_MD, exp_dir
from .freeze import require_frozen
from .model import MODEL_JSON

SEED_JSON = ROUND21_ROOT / "seed" / "seed_robustness.json"


def run_seed(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and SEED_JSON.is_file() and SEED_MD.is_file():
        existing = load_json(SEED_JSON)
        if existing.get("status") in {"DONE", NOT_TRIGGERED}:
            return existing
    model = load_json(MODEL_JSON)
    passed = list(model.get("increment_pass_codes") or [])
    if not passed:
        payload = {
            "status": NOT_TRIGGERED,
            "reason": "pre-registered OOF increment gate not hit; refuse seed ladder",
            "gate": model.get("increment_gate"),
            "n_increment_pass": 0,
            "do_not_pick_best_from_100_or_300": True,
        }
        dump_json(SEED_JSON, payload)
        write_md(
            SEED_MD,
            "# 10 Seed\n\n"
            f"状态=**{NOT_TRIGGERED}**。预注册 OOF 增量门未过，不跑 seed ladder。\n"
            "禁止从 100/300 个 seed 里挑最好的。\n",
        )
        return payload

    audits = []
    native_degenerate = False
    for code in passed:
        dest = exp_dir(str(code))
        five_dir = dest / "five"
        ten_dir = dest / "ten"
        rec = {"code": code, "five_seeds": [], "ten_seeds": []}
        for head, folder in (("five", five_dir), ("ten", ten_dir)):
            if not folder.is_dir():
                continue
            seeds = sorted(p.name for p in folder.iterdir() if p.is_dir() and p.name.startswith("seed"))
            rec[f"{head}_seeds"] = seeds
            rec[f"{head}_n"] = len(seeds)
        # 多样性：至少 2 个 seed 预测文件
        n5 = rec.get("five_n") or 0
        n10 = rec.get("ten_n") or 0
        rec["native_n"] = min(n5, n10) if n5 and n10 else max(n5, n10)
        rec["degenerate"] = rec["native_n"] < int(SEED_LADDER["S1_native"])
        if rec["degenerate"]:
            native_degenerate = True
        audits.append(rec)

    group_date = None
    if native_degenerate:
        group_date = {
            "status": "TRIGGERED_GROUP_DATE_SPEC_ONLY",
            "note": "native 5 退化。按预注册改 group-date，不从 100/300 挑最好。本轮不扩 100/300。",
            "spec": SEED_LADDER,
        }
    payload = {
        "status": "DONE",
        "triggered": True,
        "codes": passed,
        "native_audit": audits,
        "native_degenerate": native_degenerate,
        "group_date": group_date,
        "do_not_pick_best_from_100_or_300": True,
        "ladder": SEED_LADDER,
    }
    dump_json(SEED_JSON, payload)
    lines = [
        "# 10 Seed",
        "",
        f"增量门通过 `{passed}`。native 5 审计见 json。退化=`{native_degenerate}`。",
        "不从 100/300 里挑最好 seed。",
        "",
    ]
    write_md(SEED_MD, "\n".join(lines))
    return payload


__all__ = ["SEED_JSON", "run_seed"]
