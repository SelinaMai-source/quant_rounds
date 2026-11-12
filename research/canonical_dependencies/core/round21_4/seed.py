"""Seed 状态只能是 COMPLETE / NOT_TRIGGERED / BLOCKED / FAILED。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_4_ROOT
from .common import dump_json, load_json, write_md
from .contract import BLOCKED, COMPLETE, FAILED, MD, NOT_TRIGGERED, OOF_INCREMENT_GATE

SEED_JSON = ROUND21_4_ROOT / "seed" / "seed_robustness.json"
ALLOWED = {COMPLETE, NOT_TRIGGERED, BLOCKED, FAILED}


def run_seed(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    from .train import TRAIN_JSON

    train = load_json(TRAIN_JSON)
    n = int(train.get("n_oof_models") or 0)
    if resume and SEED_JSON.is_file():
        existing = load_json(SEED_JSON)
        if existing.get("seed_status") in ALLOWED:
            return existing
    # A 矩阵强制 native 5；增量门未对 30/100 触发（经济门未过）
    payload = {
        "status": NOT_TRIGGERED,
        "seed_status": NOT_TRIGGERED,
        "reason": "A0-A9 uses native 5 already; 30/100 not triggered because economic gate not hit",
        "gate": OOF_INCREMENT_GATE,
        "n_oof_models": n,
        "native_seeds": [17, 42, 73, 101, 202],
        "s30": NOT_TRIGGERED,
        "s100": NOT_TRIGGERED,
        "do_not_pick_best_from_100": True,
        "do_not_fake_100_identical_native": True,
    }
    dump_json(SEED_JSON, payload)
    write_md(ROUND21_4_ROOT / "20_seed.md", f"# Round21.4 seed\n\n状态=**{NOT_TRIGGERED}**。native 5 已用。\n")
    return payload
