"""Phase A 终端块。主报告由后续代理写。不写「写入时间」。不改 CURRENT。"""

from __future__ import annotations

from typing import Any

from .audit import AUDIT_JSON
from .common import load_json
from .contract import CONTRACT_MD, CURRENT_ALIAS, CURRENT_BASELINE, HYP_MD, PARITY_MD
from .freeze import FREEZE_JSON, manifest_path, sha256_path
from .github import GH_JSON
from .hypotheses import HYP_JSON
from .parity import PARITY_JSON


def print_phase_a_block(results: dict[str, Any] | None = None) -> None:
    results = results or {}
    parity = results.get("parity") if isinstance(results.get("parity"), dict) else load_json(PARITY_JSON)
    github = results.get("github") if isinstance(results.get("github"), dict) else load_json(GH_JSON)
    audit = results.get("audit") if isinstance(results.get("audit"), dict) else load_json(AUDIT_JSON)
    hyp = results.get("hypotheses") if isinstance(results.get("hypotheses"), dict) else load_json(HYP_JSON)
    freeze = results.get("freeze") if isinstance(results.get("freeze"), dict) else load_json(FREEZE_JSON)
    obs = parity.get("observed") or {}
    counts = audit.get("counts") or {}
    print("", flush=True)
    print("===== ROUND21 PHASE A =====", flush=True)
    print(f"CURRENT={CURRENT_BASELINE} alias={CURRENT_ALIAS} AUTO_PROMOTE=False", flush=True)
    print(
        f"parity={parity.get('status')} formal_promotion_blocked={parity.get('formal_promotion_blocked')}",
        flush=True,
    )
    print(
        f"observed CAGR={obs.get('cagr')} Sharpe={obs.get('sharpe')} "
        f"MaxDD={obs.get('max_drawdown')} trades={obs.get('completed_trades')} "
        f"NAV_end={obs.get('nav_end')}",
        flush=True,
    )
    print(
        f"github={github.get('GITHUB_METHOD_STATUS') or github.get('status')} "
        f"SHA={github.get('commit_sha')} exact_pass={github.get('exact_github_reproduction_pass')}",
        flush=True,
    )
    print(
        f"audit hermes={counts.get('hermes_planned_ok')}/{counts.get('hermes_planned_n')} "
        f"min1={counts.get('min1_days_2013')}+{counts.get('min1_days_2022')} "
        f"holder_rows={counts.get('holder_rows')} block_rows={counts.get('block_rows')} "
        f"zyquant350={counts.get('zyquant_350_n')}",
        flush=True,
    )
    print(f"hypotheses N={hyp.get('n')} ids={hyp.get('factor_ids')}", flush=True)
    print(f"manifest sha256={freeze.get('sha256')} path={freeze.get('path') or manifest_path()}", flush=True)
    print(f"manifest sidecar={sha256_path()}", flush=True)
    print(f"parity_md={PARITY_MD}", flush=True)
    print(f"hyp_md={HYP_MD}", flush=True)
    print(f"contract_md={CONTRACT_MD}", flush=True)
    print("STOP after Phase A. Do not construct factor values / evidence / combos / main report.", flush=True)
    print("===== END PHASE A =====", flush=True)


def print_construct_evidence_block(results: dict[str, Any] | None = None) -> None:
    results = results or {}
    from .construct import CONSTRUCT_JSON
    from .evidence import EVIDENCE_JSON
    from .pit import PIT_JSON

    construct = results.get("construct") if isinstance(results.get("construct"), dict) else load_json(CONSTRUCT_JSON)
    pit = results.get("pit") if isinstance(results.get("pit"), dict) else load_json(PIT_JSON)
    ev = results.get("evidence") if isinstance(results.get("evidence"), dict) else load_json(EVIDENCE_JSON)
    print("", flush=True)
    print("===== ROUND21 CONSTRUCT+PIT+EVIDENCE =====", flush=True)
    print(f"computed={construct.get('computed')}", flush=True)
    print(f"blocked_pit={construct.get('blocked_pit') or pit.get('blocked_pit')}", flush=True)
    print(f"blocked_data={construct.get('blocked_data') or pit.get('blocked_data')}", flush=True)
    print(f"PIT_PASS={pit.get('pit_pass')}", flush=True)
    print(f"evidence_packages={ev.get('n_packages')} github={ev.get('github')}", flush=True)
    print(f"best_raw={ev.get('best_raw_rankic_e2')}", flush=True)
    print(f"best_rolling={ev.get('best_rolling_w60_rankic_e2')}", flush=True)
    print(f"best_toptail={ev.get('best_top_tail_e2')}", flush=True)
    print("Did not run Add/Replace/combo/full backtest/seed/main report.", flush=True)
    print("===== END CONSTRUCT+PIT+EVIDENCE =====", flush=True)


def print_recombine_block(results: dict[str, Any] | None = None) -> None:
    results = results or {}
    from .backtest import BT_JSON
    from .combo import REG_JSON
    from .model import MODEL_JSON
    from .redundancy import RED_JSON
    from .seed import SEED_JSON
    from .stats import STATS_JSON

    red = results.get("redundancy") if isinstance(results.get("redundancy"), dict) else load_json(RED_JSON)
    combo = results.get("combo") if isinstance(results.get("combo"), dict) else load_json(REG_JSON)
    model = results.get("model") if isinstance(results.get("model"), dict) else load_json(MODEL_JSON)
    bt = results.get("backtest") if isinstance(results.get("backtest"), dict) else load_json(BT_JSON)
    seed = results.get("seed") if isinstance(results.get("seed"), dict) else load_json(SEED_JSON)
    stats = results.get("stats") if isinstance(results.get("stats"), dict) else load_json(STATS_JSON)
    best = (bt.get("best_vs_v5") or {}) if bt else {}
    print("", flush=True)
    print("===== ROUND21 RECOMBINE+WF+BACKTEST =====", flush=True)
    print(f"clusters_ge2={red.get('n_clusters_ge2')} edges={red.get('n_edges')}", flush=True)
    print(f"replacements_ridge_n={len(red.get('replacements_ridge') or [])}", flush=True)
    print(f"interactions_registry_n={combo.get('n_registry')} selected={combo.get('selected_new_factors')}", flush=True)
    print(f"OOF_models_N={model.get('n_oof_models')} increment_pass={model.get('increment_pass_codes')}", flush=True)
    print(f"full_backtests_N={bt.get('n_full_backtests')}", flush=True)
    print(
        f"best_vs_v5 code={best.get('code')} dSharpe={best.get('delta_sharpe')} "
        f"dCAGR={best.get('delta_cagr')} better={best.get('better_than_v5')}",
        flush=True,
    )
    print("FIXED vs NESTED: both_are_not_independent_oos=True", flush=True)
    print(f"seed={seed.get('status')}", flush=True)
    print(f"decision={bt.get('decision')} CURRENT_unchanged=True", flush=True)
    print("did_not_write_final_report=True did_not_append_tracker=True", flush=True)
    print(f"redundancy={RED_JSON}", flush=True)
    print(f"combo={REG_JSON}", flush=True)
    print(f"model={MODEL_JSON}", flush=True)
    print(f"backtest={BT_JSON}", flush=True)
    print(f"seed={SEED_JSON}", flush=True)
    print(f"stats={STATS_JSON}", flush=True)
    print("===== END RECOMBINE+WF+BACKTEST =====", flush=True)


def print_part39(payload: dict[str, Any] | None = None) -> None:
    from .report import print_part39 as _print

    _print(payload)
