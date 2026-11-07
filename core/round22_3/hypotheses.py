"""预注册假设卡。看结果前冻结公式。"""

from __future__ import annotations

from typing import Any

from .common import dump_json, write_csv
from .contract import (
    C1_FORMULA_FROZEN,
    C4_FORMULA_FROZEN,
    HN2_FORMULA_FROZEN,
    NEW_FACTOR_SPECS,
    REPLAY_GROUPS,
    ROUND22_3_ROOT,
)


def write_hypotheses() -> dict[str, Any]:
    rows = [
        {"factor_id": "r22_c1_xl_net_imbalance", "formula": C1_FORMULA_FROZEN, "note": "滞后版 r22_2_c1_xl_net_imbalance_lag1；2020-09-24 前 NaN"},
        {"factor_id": "r22_c4_flow_persist_5", "formula": C4_FORMULA_FROZEN, "note": "已是 D-5..D-1；2020-09-24 前 NaN；|ρ| vs g1_ret_5 不准降 Replace 门"},
        {"factor_id": "r22_2_hn2_avg_holding_chg", "formula": HN2_FORMULA_FROZEN, "note": "检验是否 Δlog(circ)−Δlog(holders)；缺失不填 0"},
    ]
    for spec in NEW_FACTOR_SPECS:
        if spec["factor_id"] == "r22_2_hn2_avg_holding_chg":
            continue
        rows.append({"factor_id": spec["factor_id"], "formula": spec["formula"], "note": spec.get("lineage")})
    rows.append({"factor_id": "REPLAY", "formula": ",".join(g["code"] for g in REPLAY_GROUPS), "note": "全历史重跑 22.2 正式组 + HN2 Ten/Both"})
    write_csv(ROUND22_3_ROOT / "hypotheses" / "hypotheses.csv", rows)
    dump_json(ROUND22_3_ROOT / "hypotheses" / "hypotheses.json", {"n": len(rows), "rows": rows, "frozen": True})
    return {"n": len(rows)}
