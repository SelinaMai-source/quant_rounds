"""30/100 seed 仅增量触发，否则 NOT_TRIGGERED。禁止 100 个相同原生模型冒充。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_2_ROOT
from .common import dump_json, load_json, write_md
from .contract import INCREMENT_GATE, MD, NOT_TRIGGERED
from .train import MODEL_JSON

SEED_JSON = ROUND21_2_ROOT / "seed" / "seed_robustness.json"


def run_seed(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and SEED_JSON.is_file() and MD["20"].is_file():
        existing = load_json(SEED_JSON)
        if existing.get("status") in {"DONE", NOT_TRIGGERED}:
            return existing
    model = load_json(MODEL_JSON)
    n_pass = int(model.get("n_increment_pass") or 0)
    passed = n_pass > 0
    if not passed:
        payload = {
            "status": NOT_TRIGGERED,
            "seed_status": NOT_TRIGGERED,
            "reason": "pre-registered OOF increment gate not hit; 30/100 seed not run",
            "gate": INCREMENT_GATE,
            "n_increment_pass": n_pass,
            "do_not_fake_100_identical_native": True,
        }
        dump_json(SEED_JSON, payload)
        write_md(
            MD["20"],
            "# Round21.2 · 20 Seed\n\n"
            f"状态=**{NOT_TRIGGERED}**。预注册增量门 `{INCREMENT_GATE['id']}` 未触发。"
            "禁止用 100 个相同原生模型冒充 seed。\n",
        )
        return payload
    payload = {
        "status": "DONE",
        "seed_status": "GATE_HIT_NATIVE5_ONLY",
        "n_increment_pass": n_pass,
        "native_seeds": [17, 42, 73, 101, 202],
        "s30": NOT_TRIGGERED,
        "s100": NOT_TRIGGERED,
        "note": "门已击中但 30/100 仅在 native5 退化时展开。本轮不挑 best-of-100。",
        "do_not_pick_best_from_100": True,
    }
    dump_json(SEED_JSON, payload)
    write_md(
        MD["20"],
        "# Round21.2 · 20 Seed\n\n"
        f"增量门击中 N=`{n_pass}`。native 5 已在训练中使用。S30/S100=`{NOT_TRIGGERED}`。\n",
    )
    return payload
