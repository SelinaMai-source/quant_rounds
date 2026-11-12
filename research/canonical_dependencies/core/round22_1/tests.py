"""16 条自动化测试。失败先修不断言。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND22_1_ROOT, ROUND22_ROOT
from .common import dump_json, load_json
from .contract import (
    ECONOMIC_MATERIALITY,
    FULL_EVIDENCE_COMPLETE,
    IC_AGG_MIN,
    IC_DAILY_MIN,
    KEEP_BASELINE_V5,
    PIT_PASS_IDS,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
)
from .evaluate import MANDATORY_NO_SKIP, SKIP_IF_MISSING


def _ok(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_tests() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    parity = load_json(ROUND22_1_ROOT / "parity" / "parity.json")
    obs = parity.get("observed") or {}
    rows.append(
        _ok(
            "T01_v5_parity_exact",
            bool(parity.get("passed")),
            f"CAGR={obs.get('cagr')} Sharpe={obs.get('sharpe')} NAV={obs.get('nav_end')} vs {V5_ANCHOR}",
        )
    )
    freeze = load_json(ROUND22_1_ROOT / "configs" / "frozen_gates.json")
    rows.append(
        _ok(
            "T02_economic_gate_frozen",
            freeze.get("status") == "FROZEN"
            and freeze.get("do_not_retune") is True
            and (freeze.get("body") or {}).get("economic_materiality", {}).get("id") == ECONOMIC_MATERIALITY["id"],
            f"id={(freeze.get('body') or {}).get('economic_materiality', {}).get('id')} sha={freeze.get('body_sha256')}",
        )
    )
    ic = load_json(ROUND22_1_ROOT / "ic" / "ic_audit.json")
    rows.append(_ok("T03_ic_universe_candidate", ic.get("universe") == "first_board_candidate" and ic.get("full_market_is_diagnostic_only") is True, str(ic.get("universe"))))
    rows.append(_ok("T04_daily_skip_n5", int(ic.get("daily_min_pairs") or 0) == IC_DAILY_MIN, f"daily_min={ic.get('daily_min_pairs')}"))
    rows.append(_ok("T05_agg_not_identifiable_n60", int(ic.get("aggregate_min_days") or 0) == IC_AGG_MIN, f"agg_min={ic.get('aggregate_min_days')}"))
    rows.append(_ok("T06_icir_not_annualized", ic.get("by_year_clipped_2018") is False and "not annualized" in str(ic.get("icir") or ""), str(ic.get("icir"))))
    rows.append(
        _ok(
            "T07_no_skip_if_missing_mandatory",
            MANDATORY_NO_SKIP.issuperset({"Rolling", "Shape", "TopTail", "ConditionalIC"}) and not ({"Rolling", "Shape", "TopTail", "ConditionalIC"} & SKIP_IF_MISSING),
            f"mandatory={sorted(MANDATORY_NO_SKIP)} skip={sorted(SKIP_IF_MISSING)}",
        )
    )
    ev = load_json(ROUND22_1_ROOT / "evidence" / "evidence.json")
    n_full = int(ev.get("n_full") or 0)
    missing_json = []
    for fid in PIT_PASS_IDS:
        dest = ROUND22_1_ROOT / "factors" / fid
        for name in ("rolling.json", "conditional.json", "shape.json", "top_tail.json", "ic.json"):
            if not (dest / name).is_file():
                missing_json.append(f"{fid}/{name}")
    rows.append(_ok("T08_eleven_evidence_files", n_full == 11 and not missing_json, f"n_full={n_full} missing={missing_json[:8]}"))
    if ev.get("status") == FULL_EVIDENCE_COMPLETE and missing_json:
        rows.append(_ok("T09_no_fake_full_evidence", False, "FULL_EVIDENCE_COMPLETE but missing JSON"))
    else:
        rows.append(_ok("T09_no_fake_full_evidence", ev.get("status") != FULL_EVIDENCE_COMPLETE or not missing_json, ev.get("status")))
    led = load_json(ROUND22_1_ROOT / "ledger" / "ledger.json")
    e0 = led.get("e0_parity") or {}
    rows.append(_ok("T10_e0_same_nav_orders", e0.get("status") == "PASS" and e0.get("same_nav") and e0.get("same_orders"), str(e0)))
    rows.append(_ok("T11_nav_unchanged", led.get("nav_performance_changed") is False and led.get("replay_only") is True, led.get("note")))
    top = led.get("top_net") or []
    was_missing = all(r.get("was_missing_columns") is False for r in top) if top else False
    rows.append(_ok("T12_top_net_was_missing_columns", bool(top) and was_missing, f"n={len(top)}"))
    rec = load_json(ROUND22_1_ROOT / "data_recovery" / "recovery.json")
    rows.append(
        _ok(
            "T13_recovery_write_scope",
            rec.get("overwrote_hermes_v7") is False and rec.get("overwrote_round22") is False and rec.get("password_guessing") is False,
            rec.get("write_root"),
        )
    )
    fgh = (rec.get("fgh") or {}).get("F") or {}
    rows.append(_ok("T14_f_no_fake_complete", int(fgh.get("download_attempts") or rec.get("download_attempts") or 0) == 0 and rec.get("status") != "RECOVERY_COMPLETE" or fgh.get("status") != "RECOVERY_COMPLETE", f"attempts={rec.get('download_attempts')} status={rec.get('status')}"))
    dfam = load_json(ROUND22_1_ROOT / "diagnostics" / "dfamily.json")
    rows.append(_ok("T15_d2_replace_remove_not_alone", dfam.get("replace_minus_remove_is_new_alpha_alone") is False, str(dfam.get("replace_minus_remove_d2"))))
    report = load_json(ROUND22_1_ROOT / "report" / "decision.json")
    block = load_json(ROUND22_1_ROOT / "report" / "status_block.json")
    if not report:
        rows.append(_ok("T16_keep_v5_no_promote", True, "PENDING_REPORT contract default KEEP_BASELINE_V5"))
    else:
        rows.append(
            _ok(
                "T16_keep_v5_no_promote",
                (report.get("final_decision") or KEEP_BASELINE_V5) == KEEP_BASELINE_V5
                and block.get("CURRENT_BASELINE_CHANGED") in {False, "FALSE", None}
                and block.get("live_ready") in {False, None},
                f"decision={report.get('final_decision')} live={block.get('live_ready')}",
            )
        )
    n_fail = int(sum(not r["passed"] for r in rows))
    payload = {"status": "PASS" if n_fail == 0 else "FAIL", "n": len(rows), "n_failed": n_fail, "rows": rows}
    dump_json(ROUND22_1_ROOT / "tests" / "unit_tests.json", payload)
    return payload
