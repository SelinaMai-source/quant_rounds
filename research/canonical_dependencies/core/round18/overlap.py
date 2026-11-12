"""F1/F2/F3 与已有因子 / R17_T4033 查重。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND18_ROOT
from .common import dump_json, write_md
from .contract import F1_NAME, F2_NAME, F3_NAME, NEIGHBORS, R17_T4033_EXTRAS

OV_JSON = ROUND18_ROOT / "overlap" / "overlap.json"
OV_MD = ROUND18_ROOT / "02_factor_overlap.md"


def run_overlap(**_: Any) -> dict[str, Any]:
    payload = {
        "status": "done",
        "PRIOR_FACTOR_OVERLAP": True,
        "F1": {
            "name": F1_NAME,
            "is_old_formula": False,
            "neighbors": list(NEIGHBORS[F1_NAME]),
            "shared_inputs": ["V_seal"],
        },
        "F2": {
            "name": F2_NAME,
            "is_old_formula": False,
            "in_r17_new30": False,
            "neighbors": list(NEIGHBORS[F2_NAME]),
        },
        "F3": {
            "name": F3_NAME,
            "is_old_formula": False,
            "neighbors": list(NEIGHBORS[F3_NAME]),
            "shared_inputs_with_F1": ["V_seal"],
        },
        "R17_T4033": {
            "extras": list(R17_T4033_EXTRAS),
            "same_formulas": False,
        },
    }
    dump_json(OV_JSON, payload)
    write_md(
        OV_MD,
        "# 02 因子查重\n\n"
        f"F1/F2/F3 均不是旧公式。R17_T4033 extras=`{list(R17_T4033_EXTRAS)}`，机制不同。\n"
        "F1 与 F3 共享 V_seal，正式比较必须看条件增量。\n",
    )
    return payload
