"""分年 / Ex2024 / Ex2025 / bootstrap 汇总。不写主报告。不改 CURRENT。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_ROOT
from .backtest import BT_JSON, run_backtest
from .common import dump_json, load_json, write_md
from .contract import STATS_MD
from .freeze import require_frozen
from .model import MODEL_JSON
from .redundancy import RED_JSON
from .seed import SEED_JSON

STATS_JSON = ROUND21_ROOT / "stats" / "stats.json"


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and STATS_JSON.is_file() and STATS_MD.is_file():
        existing = load_json(STATS_JSON)
        if existing.get("status") == "DONE":
            return existing
    bt = run_backtest(resume=True)
    model = load_json(MODEL_JSON)
    red = load_json(RED_JSON)
    seed = load_json(SEED_JSON)
    payload = {
        "status": "DONE",
        "clusters_ge2": red.get("n_clusters_ge2"),
        "clusters": red.get("clusters"),
        "replacements_ridge": red.get("replacements_ridge"),
        "n_oof_models": model.get("n_oof_models"),
        "n_increment_pass": model.get("n_increment_pass"),
        "increment_pass_codes": model.get("increment_pass_codes"),
        "n_full_backtests": bt.get("n_full_backtests"),
        "best_vs_v5": bt.get("best_vs_v5"),
        "fixed": {
            "code": "R21_FIXED",
            "oof": (model.get("fixed") or {}),
            "independent_oos": False,
        },
        "nested": {
            "code": "R21_NESTED",
            "oof": (model.get("nested") or {}),
            "independent_oos": False,
        },
        "both_are_not_independent_oos": True,
        "seed_status": seed.get("status"),
        "decision": bt.get("decision"),
        "current_unchanged": True,
        "did_not_write_final_report": True,
        "did_not_append_tracker": True,
    }
    dump_json(STATS_JSON, payload)
    best = bt.get("best_vs_v5") or {}
    lines = [
        "# 12 Stats（重组+回测摘要，非主报告）",
        "",
        f"OOF 模型 N=`{payload['n_oof_models']}` 全引擎回测 N=`{payload['n_full_backtests']}`",
        f"best=`{best.get('code')}` ΔSharpe=`{best.get('delta_sharpe')}` ΔCAGR=`{best.get('delta_cagr')}` better=`{best.get('better_than_v5')}`",
        f"FIXED vs NESTED 都不是独立 OOS。seed=`{payload['seed_status']}`",
        f"决策=`{payload['decision']}` CURRENT 未改。未写主报告 / 未追加追踪表。",
        "",
    ]
    write_md(STATS_MD, "\n".join(lines))
    return payload


__all__ = ["STATS_JSON", "run_stats"]
