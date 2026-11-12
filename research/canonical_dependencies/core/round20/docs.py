"""Phase A 终端块。主报告由后续代理写。不写「写入时间」。不改 CURRENT。"""

from __future__ import annotations

from typing import Any

from .common import load_json
from .contract import CONTRACT_MD, CURRENT_ALIAS, CURRENT_BASELINE, INVENTORY_MD, PARITY_MD
from .freeze import FREEZE_JSON, manifest_path, sha256_path
from .inventory import INV_JSON
from .parity import PARITY_JSON


def print_phase_a_block(
    parity: dict[str, Any] | None = None,
    inventory: dict[str, Any] | None = None,
    freeze: dict[str, Any] | None = None,
) -> None:
    parity = parity or load_json(PARITY_JSON)
    inventory = inventory or load_json(INV_JSON)
    freeze = freeze or load_json(FREEZE_JSON)
    obs = parity.get("observed") or {}
    print("", flush=True)
    print("===== ROUND20 PHASE A =====", flush=True)
    print(f"CURRENT={CURRENT_BASELINE} alias={CURRENT_ALIAS} AUTO_PROMOTE=False", flush=True)
    print(
        f"parity={parity.get('status')} formal_promotion_blocked={parity.get('formal_promotion_blocked')}",
        flush=True,
    )
    print(
        f"observed CAGR={obs.get('cagr')} Sharpe={obs.get('sharpe')} "
        f"MaxDD={obs.get('max_drawdown')} trades={obs.get('completed_trades')} "
        f"n_obs={obs.get('n_trading_observations') or obs.get('n_obs')} NAV_end={obs.get('nav_end')}",
        flush=True,
    )
    print(
        f"inventory n_total={inventory.get('n_total')} r17={inventory.get('n_r17_registry')} "
        f"r18={inventory.get('n_r18_f123')} match_plan={inventory.get('counts_match_plan')} "
        f"f2_eligible={inventory.get('f2_promotion_eligible')}",
        flush=True,
    )
    print(f"manifest sha256={freeze.get('sha256')} path={freeze.get('path') or manifest_path()}", flush=True)
    print(f"manifest sidecar={sha256_path()}", flush=True)
    print(f"parity_md={PARITY_MD}", flush=True)
    print(f"inventory_md={INVENTORY_MD}", flush=True)
    print(f"contract_md={CONTRACT_MD}", flush=True)
    print("STOP after Phase A. Do not run IC/shape/combos/seeds/full backtests/main report.", flush=True)
    print("===== END PHASE A =====", flush=True)


def print_phase_bc_block(results: dict[str, Any] | None = None) -> None:
    from .combo import REG_CSV, REG_JSON
    from .conditional import COND_JSON, LOFO_CSV, ORTHO_CSV
    from .contract import CONDITIONAL_MD, IC_MD, PIT_MD, REDUNDANCY_MD, SHAPE_MD
    from .ic import IC_JSON
    from .pit import PIT_JSON
    from .redundancy import KEEP_CSV, RED_JSON
    from .shape import SHAPE_JSON

    results = results or {}
    pit = results.get("pit") if isinstance(results.get("pit"), dict) else load_json(PIT_JSON)
    ic = results.get("ic") if isinstance(results.get("ic"), dict) else load_json(IC_JSON)
    shape = results.get("shape") if isinstance(results.get("shape"), dict) else load_json(SHAPE_JSON)
    red = results.get("redundancy") if isinstance(results.get("redundancy"), dict) else load_json(RED_JSON)
    cond = results.get("conditional") if isinstance(results.get("conditional"), dict) else load_json(COND_JSON)
    combo = results.get("combo") if isinstance(results.get("combo"), dict) else load_json(REG_JSON)
    print("", flush=True)
    print("===== ROUND20 PHASE B+C =====", flush=True)
    print(
        f"PIT n_total={pit.get('n_total')} pass_or_locked={pit.get('n_pass_or_locked')} "
        f"eligible={pit.get('n_promotion_eligible')} f2_eligible={pit.get('f2_promotion_eligible')}",
        flush=True,
    )
    print(
        f"IC daily_rows={ic.get('n_daily_rows')} skip_thin={(ic.get('skip_counts') or {}).get('thin')} "
        f"skip_degen={(ic.get('skip_counts') or {}).get('degenerate')}",
        flush=True,
    )
    print(
        f"shape non_monotonic={shape.get('n_non_monotonic')} forced_q10={shape.get('forced_q10')}",
        flush=True,
    )
    print(
        f"KEEP={red.get('n_keep')} DROP={red.get('n_drop')} clusters_ge2={red.get('n_clusters_ge2')} "
        f"f1_f3_rebranded={red.get('f1_f3_rebranded')}",
        flush=True,
    )
    print(
        f"conditional leaders={len(cond.get('leaders') or [])} "
        f"ortho_winners={cond.get('ortho_winner_counts')} lofo_delete_from_v5={cond.get('lofo_delete_from_v5')}",
        flush=True,
    )
    print(
        f"combo_registry n={combo.get('n_registry')} trained={combo.get('trained')} "
        f"looked_at_nav={combo.get('looked_at_combo_nav')} selected={combo.get('selected_new_factors')}",
        flush=True,
    )
    print(f"pit_md={PIT_MD}", flush=True)
    print(f"ic_md={IC_MD}", flush=True)
    print(f"shape_md={SHAPE_MD}", flush=True)
    print(f"redundancy_md={REDUNDANCY_MD} keep_csv={KEEP_CSV}", flush=True)
    print(f"conditional_md={CONDITIONAL_MD} ortho={ORTHO_CSV} lofo={LOFO_CSV}", flush=True)
    print(f"combo_registry={REG_CSV}", flush=True)
    print("STOP after Phase B+C. Do not run model/seed/full backtests/main report.", flush=True)
    print("===== END PHASE B+C =====", flush=True)


def print_combo_model_block(results: dict[str, Any] | None = None) -> None:
    from .combo import REG_CSV
    from .contract import COMBO_TRAIN_MD, MODEL_MD
    from .model import MODEL_CSV, MODEL_JSON
    from .runner import COMBO_RES_CSV, COMBO_TRAIN_JSON

    results = results or {}
    combo = results.get("combo") if isinstance(results.get("combo"), dict) else load_json(COMBO_TRAIN_JSON)
    model = results.get("model") if isinstance(results.get("model"), dict) else load_json(MODEL_JSON)
    best = (combo or {}).get("best") or {}
    parity = (model or {}).get("a00_a01_parity") or {}
    print("", flush=True)
    print("===== ROUND20 COMBO + MODEL =====", flush=True)
    print(
        f"combos tested N={combo.get('n_tested')} with_metrics={combo.get('n_with_metrics')} "
        f"skip={combo.get('n_skip')} post_hoc_best_x_best={combo.get('post_hoc_best_x_best')}",
        flush=True,
    )
    print(
        f"best combo={best.get('code')} Sharpe={best.get('sharpe')} CAGR={best.get('cagr')} "
        f"MaxDD={best.get('max_drawdown')} trades={best.get('completed_trades')} NAV_end={best.get('nav_end')}",
        flush=True,
    )
    print(
        f"A00 vs A01 agree={parity.get('agree')} M2={model.get('m2_status')} "
        f"finalists={[f.get('code') for f in (model.get('finalists') or [])]}",
        flush=True,
    )
    print(f"combo_registry={REG_CSV}", flush=True)
    print(f"combo_results={COMBO_RES_CSV} md={COMBO_TRAIN_MD}", flush=True)
    print(f"model_csv={MODEL_CSV} md={MODEL_MD}", flush=True)
    print("AUTO_PROMOTE_CURRENT=False. No CURRENT promote.", flush=True)
    print("===== END COMBO + MODEL =====", flush=True)


def print_complete_block(payload: dict[str, Any] | None = None) -> None:
    from .report import print_part38

    print_part38(payload)
