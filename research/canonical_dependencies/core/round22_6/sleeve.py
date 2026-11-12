"""Round22.6 袖套适配：正式回测仍走 r225241_sleeve，经济门改用 CORRECTED。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.r225241_sleeve import run_round7
from core.research_integrity_v2.cache_identity import cache_valid, write_identity
from core.research_integrity_v2.economic_gate import evaluate_economic_gate
from core.research_integrity_v2.io_v2 import dump_json
from core.research_integrity_v2.ledger import build_lineage_rows, reconcile_nav
from core.round22_6.contract import anchor_copy


def run_formal_sleeve(
    dest: Path,
    scores: pd.DataFrame,
    extra: dict[str, Any],
    *,
    identity: dict[str, Any] | None = None,
    resume: bool = True,
    already_overlayed: bool = False,
    scenario: bool = False,
    counts_as_alpha: bool = True,
    top_changed: bool | None = None,
    fee_same_direction: bool | None = None,
    not_only_year: bool | None = None,
) -> dict[str, Any]:
    dest = Path(dest)
    if identity is not None and resume and cache_valid(dest, identity, require_files=("summary.json", "nav.parquet")):
        from core.research_integrity_v2.io_v2 import load_json

        return load_json(dest / "summary.json")
    obs = run_round7(
        dest,
        scores,
        extra,
        resume=False,
        already_overlayed=already_overlayed,
        scenario=scenario,
        counts_as_alpha=counts_as_alpha,
        apply_v5_gate=False,
    )
    gate = evaluate_economic_gate(
        obs,
        baseline=anchor_copy(),
        top_changed=top_changed,
        fee_same_direction=fee_same_direction,
        not_only_year=not_only_year,
    )
    obs["economic_gate"] = gate
    obs["economically_material"] = bool(gate["CORRECTED"]["economically_material"]) and bool(counts_as_alpha) and not bool(scenario)
    obs["nav_improvement"] = gate["CORRECTED"].get("nav_improvement")
    obs["nav_improvement_ok"] = gate["CORRECTED"].get("nav_improvement_ok")
    obs["delta_terminal_nav_rmb"] = gate["CORRECTED"].get("delta_terminal_nav_rmb")
    obs["relative_terminal_nav_change"] = gate["CORRECTED"].get("relative_terminal_nav_change")
    obs["gate_id"] = gate["CORRECTED"]["gate_id"]
    obs["promotion_decision_source"] = "CORRECTED"
    if (dest / "nav.parquet").is_file():
        nav = pd.read_parquet(dest / "nav.parquet")
        obs["nav_reconcile"] = reconcile_nav(nav)
    if (dest / "overlayed.parquet").is_file():
        overlay = pd.read_parquet(dest / "overlayed.parquet")
        trades = pd.read_parquet(dest / "trades.parquet") if (dest / "trades.parquet").is_file() else None
        lineage = build_lineage_rows(scores=overlay, trades=trades)
        lineage_path = dest / "lineage_signal_to_exit.parquet"
        lineage.to_parquet(lineage_path, index=False)
        obs["lineage_rows"] = int(len(lineage))
        obs["lineage_path"] = str(lineage_path)
    dump_json(dest / "summary.json", obs)
    if identity is not None:
        write_identity(dest, identity)
    return obs
