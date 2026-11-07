"""Evidence Factory 测试 + evaluate 挂钩。"""

from __future__ import annotations

from typing import Any

from .common import dump_json, write_md
from .contract import C1_LAG1_ID, C4_ID, FACTORY_STEPS_EXTRA, HN2_ID, MD, ROUND22_3_ROOT
from .docs import update_evaluate_factor, update_standard
from .evaluate import evaluate_factor
from .tests import run_tests

FACT_JSON = ROUND22_3_ROOT / "tests" / "factory.json"


def run_factory(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    std = update_standard()
    ev = update_evaluate_factor()
    tests = run_tests()
    ids = [
        C1_LAG1_ID,
        C4_ID,
        HN2_ID,
        "r22_3_fs1_accrual_minus_ocf",
        "r22_3_fs2_ocf_over_assets",
    ]
    packs = []
    for fid in ids:
        try:
            packs.append(evaluate_factor(fid, through="PromotionGate", persist=True))
        except KeyError:
            packs.append({"factor_id": fid, "status": "UNKNOWN_FACTOR"})
    payload = {
        "status": "DONE",
        "n_eval": len(packs),
        "standard": std,
        "evaluate_cli": ev,
        "tests": {"n": tests.get("n"), "n_failed": tests.get("n_failed")},
        "extra_steps": list(FACTORY_STEPS_EXTRA),
    }
    dump_json(FACT_JSON, payload)
    write_md(MD["12"], f"# Round22.3 · 12 Factory\n\nevaluate n=`{len(packs)}` tests_failed=`{tests.get('n_failed')}`。mandatory 禁止 SKIP_IF_MISSING。\n")
    return payload
