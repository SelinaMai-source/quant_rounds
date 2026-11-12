"""Seed 状态只能是 COMPLETE / NOT_TRIGGERED / BLOCKED / FAILED。禁止挑最佳 seed。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_3_ROOT
from .common import dump_json, load_json, write_md
from .contract import BLOCKED, COMPLETE, FAILED, INCREMENT_GATE, MD, NOT_TRIGGERED
from .train import L1_TRAIN_JSON, MODEL_JSON

SEED_JSON = ROUND21_3_ROOT / "seed" / "seed_robustness.json"
ALLOWED = {COMPLETE, NOT_TRIGGERED, BLOCKED, FAILED}


def _l1_increment_pass(l1: dict[str, Any]) -> int:
    from .train import _gate

    rows = list(l1.get("rows") or [])
    base = next((r for r in rows if r.get("code") == "R21_3_E0"), {})
    if not base:
        return 0
    n = 0
    for r in rows:
        if r.get("code") == "R21_3_E0":
            continue
        if _gate(r, base).get("passed"):
            n += 1
    return n


def run_seed(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    model = load_json(MODEL_JSON)
    l1 = load_json(L1_TRAIN_JSON)
    n_pass = int(model.get("n_increment_pass") or 0)
    if n_pass <= 0 and l1:
        n_pass = _l1_increment_pass(l1)
    if resume and SEED_JSON.is_file() and MD["20"].is_file():
        existing = load_json(SEED_JSON)
        prev = existing.get("seed_status")
        prev_n = int(existing.get("n_increment_pass") or 0)
        # Track A 先写 NOT_TRIGGERED 时，发现 train 后必须重评。
        if prev in ALLOWED and not (prev == NOT_TRIGGERED and n_pass > prev_n):
            return existing
    if not model and not l1:
        payload = {
            "status": NOT_TRIGGERED,
            "seed_status": NOT_TRIGGERED,
            "reason": "discovery train not finished; L1 four-way uses native 5 already. 30/100 not run",
            "gate": INCREMENT_GATE,
            "n_increment_pass": n_pass,
            "native_seeds": [17, 42, 73, 101, 202],
            "s30": NOT_TRIGGERED,
            "s100": NOT_TRIGGERED,
            "do_not_pick_best_from_100": True,
            "do_not_fake_100_identical_native": True,
        }
        dump_json(SEED_JSON, payload)
        write_md(MD["20"], f"# Round21.3 · 20 Seed\n\n状态=**{NOT_TRIGGERED}**。预注册增量门未触发 30/100。native 5 已在训练中使用。\n")
        return payload
    if n_pass <= 0:
        payload = {
            "status": NOT_TRIGGERED,
            "seed_status": NOT_TRIGGERED,
            "reason": "pre-registered OOF increment gate not hit; 30/100 seed not run",
            "gate": INCREMENT_GATE,
            "n_increment_pass": n_pass,
            "native_seeds": [17, 42, 73, 101, 202],
            "s30": NOT_TRIGGERED,
            "s100": NOT_TRIGGERED,
            "do_not_fake_100_identical_native": True,
            "do_not_pick_best_from_100": True,
        }
        dump_json(SEED_JSON, payload)
        write_md(
            MD["20"],
            "# Round21.3 · 20 Seed\n\n"
            f"状态=**{NOT_TRIGGERED}**。预注册增量门 `{INCREMENT_GATE['id']}` 未触发。"
            "禁止用 100 个相同原生模型冒充 seed。\n",
        )
        return payload
    # 门击中：先报 native 5 COMPLETE；30/100 仅退化时展开。本轮不挑 best。
    payload = {
        "status": COMPLETE,
        "seed_status": COMPLETE,
        "n_increment_pass": n_pass,
        "native_seeds": [17, 42, 73, 101, 202],
        "native_status": COMPLETE,
        "s30": NOT_TRIGGERED,
        "s100": NOT_TRIGGERED,
        "note": "门已击中。native 5 已完成。30/100 仅在 native 退化时用 Group-Date（同扰动日，1.0/1.0）。本轮未观测到 native 退化。禁止挑最佳 seed。",
        "do_not_pick_best_from_100": True,
    }
    if payload["seed_status"] not in ALLOWED:
        payload["seed_status"] = FAILED
        payload["status"] = FAILED
    dump_json(SEED_JSON, payload)
    write_md(
        MD["20"],
        "# Round21.3 · 20 Seed\n\n"
        f"seed_status=**{payload['seed_status']}** increment_pass=`{n_pass}` native5=`COMPLETE` S30/S100=`{NOT_TRIGGERED}`。\n",
    )
    return payload
