"""读契约 / 历史搜索 / Round16.5 handoff。差异写入审计，不以上传文档覆盖代码。"""

from __future__ import annotations

from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND16_5_ROOT, ROUND17_ROOT, stock_path
from ..round7.contract import FIVE_FEATURES, SEEDS, TEN_FEATURES
from .common import dump_json, write_md
from .contract import (
    CONTRACT_BAGGING,
    CONTRACT_DETERMINISTIC,
    CURRENT_BASELINE,
    OPTION_CANDIDATE,
    PARENT_V4,
    V4_ANCHOR,
)

READ_MD = ROUND17_ROOT / "00_prior_read.md"
READ_JSON = ROUND17_ROOT / "read" / "prior_artifacts.json"


def run_read(**_: Any) -> dict[str, Any]:
    from .contract import copy_plan

    copy_plan()
    docs = [
        RESULT_ROOT / "round16_5" / "ROUND17_RESEARCH_HANDOFF.md",
        RESULT_ROOT / "round16" / "ROUND17_RESEARCH_HANDOFF.md",
        PROJECT_ROOT / "docs" / "一进二策略_多seed集成全流程.md",
        RESULT_ROOT / "round7" / "contract.json",
        RESULT_ROOT / "一进二_Round7基线_实验组追踪.md",
    ]
    present = {str(p): p.is_file() for p in docs}
    hist_search = stock_path("runs/first_board_factor_search_5000_v1/registry/trials.json")
    payload = {
        "current": CURRENT_BASELINE,
        "parent": PARENT_V4,
        "option_candidate_not_current": OPTION_CANDIDATE,
        "v4_anchor_fee_corrected": V4_ANCHOR,
        "seeds": list(SEEDS),
        "five_n": len(FIVE_FEATURES),
        "ten_n": len(TEN_FEATURES),
        "contracts": {
            "bagging_isolated": CONTRACT_BAGGING,
            "deterministic": CONTRACT_DETERMINISTIC,
            "round7_invariance_stays_on_1_0": True,
            "round17_must_not_use_invariance": True,
        },
        "do_not_edit": [
            "AGENTS.md",
            ".cursor/rules/no-xgb-subsample.mdc",
            "core/round7/",
            "Stock src/runs",
            "R16_O01 artifacts",
        ],
        "historical_5000_subset_search": str(hist_search),
        "historical_5000_exists": hist_search.is_file(),
        "docs_present": present,
        "r6_a04_note": "Round6 forensic R6_A04 showed bagging-only CAGR lift from randomness; A01 luck is not a promotion reason.",
        "status": "read",
    }
    dump_json(READ_JSON, payload)
    lines = [
        "# 00 Prior read",
        "",
        f"CURRENT=`{CURRENT_BASELINE}` parent=`{PARENT_V4}`。期权候选 `{OPTION_CANDIDATE}` 本轮不改、不融合。",
        f"v4 HISTORICAL_FEE_CORRECTED anchor CAGR=`{V4_ANCHOR['cagr']}` Sharpe=`{V4_ANCHOR['sharpe']}` MaxDD=`{V4_ANCHOR['max_drawdown']}`",
        "",
        "孤立契约 `R17_STOCHASTIC_BAGGING_08_08` 仅 core/round17 + result/round17。",
        "Round7 `assert_seed_rank_invariance` 继续只约束 canonical 1.0/1.0。",
        "搜索是增量发现（锁定 baseline + 1–4 新因子），不是历史 44 子集重放。",
        "",
        f"historical 5000 registry exists=`{hist_search.is_file()}`",
    ]
    write_md(READ_MD, "\n".join(lines))
    return payload
