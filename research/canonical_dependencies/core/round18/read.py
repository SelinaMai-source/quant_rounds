"""读 R7–R17 约束与 T4033 对照。完成前不看本轮候选 NAV。"""

from __future__ import annotations

from typing import Any

from ..paths import RESULT_ROOT, ROUND17_ROOT, ROUND18_ROOT
from .common import dump_json, load_json, write_md
from .contract import (
    CURRENT_BASELINE,
    F1_NAME,
    F2_NAME,
    F3_NAME,
    NEIGHBORS,
    OPTION_CANDIDATE,
    PARENT_V4,
    PRIOR_RESEARCH_SELECTION_EXISTS,
    R17_T4033_EXTRAS,
    V4_ANCHOR,
    copy_plan,
)

READ_JSON = ROUND18_ROOT / "read" / "read.json"
READ_MD = ROUND18_ROOT / "00_prior_research.md"


def run_read(**_: Any) -> dict[str, Any]:
    copy_plan()
    tracker = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
    r17_report = ROUND17_ROOT / "ROUND17_LARGE_SCALE_FACTOR_SEARCH_REPORT.md"
    t4033 = load_json(ROUND17_ROOT / "search" / "trials" / "R17_T4033" / "metrics.json")
    extras = list(t4033.get("extras") or R17_T4033_EXTRAS)
    overlap = {
        F1_NAME: {
            "is_old_formula": False,
            "neighbors": list(NEIGHBORS[F1_NAME]),
            "mechanism": "分母是触板后 below-limit 分钟成交额，不是 V_seal/amount 或封板时长",
        },
        F2_NAME: {
            "is_old_formula": False,
            "in_r17_30": False,
            "neighbors": list(NEIGHBORS[F2_NAME]),
            "mechanism": "L+XL 平均单笔规模比，不是净流入",
        },
        F3_NAME: {
            "is_old_formula": False,
            "neighbors": list(NEIGHBORS[F3_NAME]),
            "mechanism": "V_seal/FFMV，不是对成交额也不是总市值/流通市值对数",
        },
    }
    payload = {
        "status": "done",
        "current": CURRENT_BASELINE,
        "parent": PARENT_V4,
        "v4_anchor": V4_ANCHOR,
        "option_candidate_untouched": OPTION_CANDIDATE,
        "prior_research_selection_exists": PRIOR_RESEARCH_SELECTION_EXISTS,
        "r17_report_exists": r17_report.is_file(),
        "tracker_exists": tracker.is_file(),
        "r17_t4033": {
            "trial_id": "R17_T4033",
            "sleeve": t4033.get("sleeve") or "FIVE_ONLY",
            "extras": extras,
            "note": "FIVE_ONLY 增量组合，不是 F1/F2/F3 公式",
            "formula_overlap": False,
        },
        "PRIOR_FACTOR_OVERLAP": overlap,
        "constraints": {
            "subsample": 1.0,
            "colsample": 1.0,
            "no_bagging": True,
            "no_fuse_r16_o01": True,
            "no_auto_promote": True,
            "no_fillna_0": True,
        },
    }
    dump_json(READ_JSON, payload)
    lines = [
        "# 00 先验研究与查重",
        "",
        f"CURRENT=`{CURRENT_BASELINE}` = `{PARENT_V4}`。不融合 `{OPTION_CANDIDATE}`。",
        f"PRIOR_RESEARCH_SELECTION_EXISTS=`{PRIOR_RESEARCH_SELECTION_EXISTS}`。",
        "",
        "## R17_T4033",
        "",
        f"extras=`{extras}`。sleeve=`FIVE_ONLY`。**不是** F1/F2/F3。",
        "",
        "## PRIOR_FACTOR_OVERLAP",
        "",
        f"- F1 `{F1_NAME}`：新公式。近邻 `{NEIGHBORS[F1_NAME]}`。机制不同，仍须检验是否只是封单强度变换。",
        f"- F2 `{F2_NAME}`：不在 R17 30 个新因子中。近邻是净流入，不是平均单笔规模。",
        f"- F3 `{F3_NAME}`：不是 `h5_close_limit_value_to_amount` / `g0_float_mv_log`。与 F1 共享 V_seal，必须做条件增量。",
        "",
        "本轮禁止 bagging、禁止搜 epsilon/择时/B03/退出/超参/种子。",
    ]
    write_md(READ_MD, "\n".join(lines))
    return payload
