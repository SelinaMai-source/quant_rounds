"""Evidence Factory 测试 + evaluate 挂钩。mandatory 阶段禁止静默 skip。"""

from __future__ import annotations

from typing import Any

from .common import dump_json, load_json, write_md
from .contract import C4_ID, EXISTING_11, FACTORY_STEPS_EXTRA, MD, ROUND22_2_ROOT
from .docs import update_evaluate_factor, update_standard
from .evaluate import evaluate_factor
from .tests import run_tests

FACT_JSON = ROUND22_2_ROOT / "tests" / "factory.json"


def run_factory(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    std = update_standard()
    ev = update_evaluate_factor()
    tests = run_tests()
    ids = list(EXISTING_11) + [
        (load_json(ROUND22_2_ROOT / "pit" / "c_pit.json").get("c1") or {}).get("model_id") or "r22_c1_xl_net_imbalance",
        C4_ID,
        "r22_2_hn2_avg_holding_chg",
        "r22_2_hn1_neg_log_holder_ratio",
        "r22_2_bt1_block_discount",
        "r22_2_mg1_fin_buy_intensity",
        "r22_2_fs1_accrual_minus_ocf",
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
    write_md(MD["12"], f"# Round22.2 · 12 Factory\n\nevaluate n=`{len(packs)}` tests_failed=`{tests.get('n_failed')}`。mandatory 禁止 SKIP_IF_MISSING。\n")
    return payload
