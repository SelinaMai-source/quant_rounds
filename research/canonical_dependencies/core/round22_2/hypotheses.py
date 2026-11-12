"""预注册假设卡。看结果前冻结公式，不得为收益改 C。"""

from __future__ import annotations

from typing import Any

from .common import dump_json, write_csv
from .contract import C1_FORMULA_FROZEN, C4_FORMULA_FROZEN, NEW_FACTOR_SPECS, ROUND22_2_ROOT


def write_hypotheses() -> dict[str, Any]:
    rows = [
        {
            "factor_id": "r22_c1_xl_net_imbalance",
            "formula": C1_FORMULA_FROZEN,
            "note": "表无 BUY_VALUE_XL。必须滞后则新版本 r22_2_c1_xl_net_imbalance_lag1",
        },
        {"factor_id": "r22_c4_flow_persist_5", "formula": C4_FORMULA_FROZEN, "note": "已是 D-5..D-1"},
    ]
    for spec in NEW_FACTOR_SPECS:
        rows.append({"factor_id": spec["factor_id"], "formula": spec["formula"], "note": spec.get("lineage")})
    write_csv(ROUND22_2_ROOT / "hypotheses" / "hypotheses.csv", rows)
    dump_json(ROUND22_2_ROOT / "hypotheses" / "hypotheses.json", {"n": len(rows), "rows": rows, "frozen": True})
    return {"n": len(rows)}
