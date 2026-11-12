"""工厂固化：STANDARD + evaluate + 16 测试。"""

from __future__ import annotations

from typing import Any

from .common import dump_json, write_md
from .contract import MD
from .docs import update_evaluate_factor, update_standard
from .evaluate import evaluate_factor
from .tests import run_tests
from ..paths import ROUND22_1_ROOT
from .contract import PIT_PASS_IDS
from .freeze import require_frozen_gate

FACTORY_JSON = ROUND22_1_ROOT / "tests" / "factory.json"


def run_factory(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    require_frozen_gate()
    std = update_standard()
    ev = update_evaluate_factor()
    gates = []
    for fid in PIT_PASS_IDS:
        gates.append(evaluate_factor(fid, through="PromotionGate", persist=True))
    tests = run_tests()
    payload = {
        "status": tests.get("status"),
        "standard": std,
        "evaluate_factor": ev,
        "n_tests": tests.get("n"),
        "n_failed": tests.get("n_failed"),
        "n_gates": len(gates),
        "silent_skip_forbidden": True,
    }
    dump_json(FACTORY_JSON, payload)
    write_md(
        MD["09"],
        f"# Round22.1 · 09 工厂与测试\n\nSTANDARD 已追加。evaluate_factor 接入 round22_1。测试 {tests.get('n')} 条，失败 {tests.get('n_failed')}。mandatory 禁止 SKIP_IF_MISSING。\n",
    )
    return payload
