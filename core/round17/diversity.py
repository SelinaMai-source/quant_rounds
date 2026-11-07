"""原特征 0.8/0.8 五 Seed 预检。退化则记录原因，禁止加噪声或改超参。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND17_ROOT, ensure_round17_tree
from ..round7.contract import FIVE_HEAD, TEN_HEAD
from ..round7.train import _load_panel
from .common import dump_json, load_json, write_md
from .contract import (
    A01,
    CONTRACT_BAGGING,
    DISCOVERY_XGB_N_JOBS,
    SEED_DIVERSITY_DEGENERATE,
    SEEDS,
)
from .train import train_head_open

PRE_MD = ROUND17_ROOT / "02_seed_diversity_preflight.md"
PRE_JSON = ROUND17_ROOT / "preflight" / "preflight.json"


def run_preflight(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    ensure_round17_tree()
    if resume and PRE_JSON.is_file():
        existing = load_json(PRE_JSON)
        if existing.get("status") in {SEED_DIVERSITY_DEGENERATE, "SEED_DIVERSITY_PASS", "PASS"} and existing.get("five"):
            return existing
    reports = {}
    for head in (FIVE_HEAD, TEN_HEAD):
        training, candidates = _load_panel(head)
        out_dir = ROUND17_ROOT / "preflight" / "models" / head.name
        payload = train_head_open(
            training=training,
            candidates=candidates,
            features=head.features,
            target=head.target,
            label_end_column=head.label_end_column,
            rank_n_bins=head.rank_n_bins,
            eval_metric=head.eval_metric,
            subsample=0.8,
            colsample_bytree=0.8,
            first_prediction_year=2018,
            last_prediction_year=2026,
            output_dir=out_dir,
            seeds=SEEDS,
            n_jobs=DISCOVERY_XGB_N_JOBS,
            save_models=False,
            resume=resume,
        )
        reports[head.name] = {k: v for k, v in payload.items() if k not in {"ensemble", "seed_frames"}}
        reports[head.name]["seed_diversity"] = payload["seed_diversity"]
    five_d = reports["five"]["seed_diversity"]
    ten_d = reports["ten"]["seed_diversity"]
    degenerate = bool(five_d.get("degenerate") or ten_d.get("degenerate"))
    status = SEED_DIVERSITY_DEGENERATE if degenerate else "SEED_DIVERSITY_PASS"
    reason = None
    if degenerate:
        reason = (
            "If 0.8/0.8 still yields identical predictions, likely XGBoost hist + this dataset "
            "has no effective row/column subsample diversity (all rows/cols kept or tree structure "
            "insensitive). Documented; hyperparameters not changed; noise not added."
        )
    payload = {
        "status": status,
        "contract": CONTRACT_BAGGING,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seeds": list(SEEDS),
        "used_round7_invariance_assert": False,
        "five": reports["five"],
        "ten": reports["ten"],
        "degenerate": degenerate,
        "technical_reason": reason,
        "a01_code": A01,
        "note": "孤立研究契约，不是生产默认。R6_A04 已证明仅改回 0.8/0.8 可从随机性抬 CAGR，不得把 A01 运气当晋升理由。",
    }
    dump_json(PRE_JSON, payload)
    lines = [
        "# 02 Seed diversity preflight（0.8/0.8）",
        "",
        f"status=`{status}` contract=`{CONTRACT_BAGGING}`",
        f"five unique_hashes=`{five_d.get('unique_prediction_hashes')}` mean_corr=`{five_d.get('mean_pairwise_corr')}` top1_agree=`{five_d.get('top1_mean_agreement')}`",
        f"ten unique_hashes=`{ten_d.get('unique_prediction_hashes')}` mean_corr=`{ten_d.get('mean_pairwise_corr')}` top1_agree=`{ten_d.get('top1_mean_agreement')}`",
        "",
        f"technical_reason={reason}",
        "",
        "Round7 invariance assert 未在本路径调用。",
    ]
    write_md(PRE_MD, "\n".join(lines))
    return payload
