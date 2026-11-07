"""仅 material bug 时隔离修复并重跑 R18 原公式，命名 R19_B_REPAIRED_*。不覆盖 result/round18。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...paths import ROUND19_ROOT
from ...round18.contract import EXPERIMENTS
from ...round18.factors import load_factor_panel
from ..common import slim_obs
from ..contract import REPAIRED_ROUND18_CANDIDATE
from .support import (
    B00,
    KEYS,
    VERDICT_JSON,
    cand_dir,
    dump_json,
    load_json,
    load_original_panels,
    load_r18_panels,
    tb_dir,
    write_md,
)
from .train import train_both_heads
from .verdict import run_bug_verdict


def _repaired_dir(code: str):
    dest = ROUND19_ROOT / "track_b" / "repair" / str(code)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def rebuild_clean_panels() -> dict[str, pd.DataFrame]:
    """Original 14/13 columns from R7 panels; F columns left-joined. Never overwrite originals."""
    orig = load_original_panels()
    factors = load_factor_panel()
    out = dict(orig)
    r18 = load_r18_panels()
    out["five_frozen"] = r18["five_frozen"]
    out["ten_frozen"] = r18["ten_frozen"]
    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate"):
        frame = orig[key].copy()
        orig_cols = [c for c in frame.columns if c not in KEYS]
        merged = frame.merge(factors, on=KEYS, how="left", suffixes=("", "_fjoin"))
        # drop any accidental overwrite copies
        drop = [c for c in merged.columns if c.endswith("_fjoin") and c[: -len("_fjoin")] in orig_cols]
        if drop:
            merged = merged.drop(columns=drop)
        out[key] = merged
    return out


def run_repair(*, resume: bool = True, **kw: Any) -> dict[str, Any]:
    verdict = load_json(VERDICT_JSON) or run_bug_verdict(resume=resume, **kw)
    dest_json = tb_dir("repair") / "repair.json"
    if not verdict.get("material_bug"):
        payload = {
            "status": "NOT_REQUIRED",
            "r18_rebacktest_required": False,
            "repaired_best": None,
            "PIPELINE_BUG_VERDICT": verdict.get("PIPELINE_BUG_VERDICT"),
            "TRACK_B_DECISION": verdict.get("TRACK_B_DECISION"),
            "note": "No material pipeline bug. Refuse epsilon/formula/head-weight search. R18 originals untouched.",
            "round18_originals_untouched": True,
        }
        dump_json(dest_json, payload)
        write_md(
            tb_dir("repair") / "13_repair.md",
            "# Track B · repair\n\nstatus=`NOT_REQUIRED`\n\n无 material bug，不重跑 R18 矩阵，不搜 epsilon。\n",
        )
        return payload

    print("[round19/track_b] material bug → isolated repair + R19_B_REPAIRED_* matrix", flush=True)
    panels = rebuild_clean_panels()
    # prove repaired original-feature retrain vs B00
    prove_dir = _repaired_dir("R19_B_REPAIRED_B01")
    trained = train_both_heads(
        five_training=panels["five_training"],
        five_candidates=panels["five_candidate"],
        ten_training=panels["ten_training"],
        ten_candidates=panels["ten_candidate"],
        five_extras=(),
        ten_extras=(),
        output_dir=prove_dir,
        resume=resume,
        five_frozen=panels["five_frozen"],
        ten_frozen=panels["ten_frozen"],
        force_retrain=True,
    )
    from ...round18.backtest import run_fused_backtest

    prove_obs = run_fused_backtest(
        prove_dir,
        trained["five_pred"],
        trained["ten_pred"],
        extra={"code": "R19_B_REPAIRED_B01", "role": "repaired_force_retrain_14_13", "formal": False},
        resume=resume,
    )
    b00 = slim_obs((load_json(cand_dir(B00) / "summary.json") or {}).get("observed") or {})
    prove = slim_obs(prove_obs)
    try:
        close = abs(float(prove.get("sharpe") or 0) - float(b00.get("sharpe") or 0)) < 0.05
    except (TypeError, ValueError):
        close = False
    if not close:
        payload = {
            "status": "REPAIR_DID_NOT_RESTORE_B01_B00",
            "r18_rebacktest_required": True,
            "repaired_best": None,
            "prove_B01": prove,
            "B00": b00,
            "note": "Isolated join repair did not restore B01≈B00. Stop. No epsilon/formula search.",
        }
        dump_json(dest_json, payload)
        write_md(tb_dir("repair") / "13_repair.md", f"# repair\n\nstatus=`{payload['status']}`\n")
        return payload

    rows = []
    for spec in EXPERIMENTS:
        code = f"R19_B_REPAIRED_{str(spec['code']).removeprefix('R18_')}"
        dest = _repaired_dir(code)
        print(f"[round19/track_b] repaired {code} role={spec['role']}", flush=True)
        if spec["code"] == "R18_A00":
            # keep locked replay semantics for A00; copy B00 ledger pointers, do not overwrite R18
            src = cand_dir(B00)
            obs = dict(load_json(src / "summary.json") or {})
            obs = obs.get("observed") or obs
            rec = {
                "code": code,
                "src": spec["code"],
                "role": spec["role"],
                "formal": spec.get("formal"),
                "status": "COPIED_B00_LOCKED",
                **{k: obs.get(k) for k in ("cagr", "sharpe", "max_drawdown", "completed_trades")},
            }
            dump_json(dest / "summary.json", rec)
            rows.append(rec)
            continue
        extras_f = list(spec.get("five") or ())
        extras_t = list(spec.get("ten") or ())
        trained = train_both_heads(
            five_training=panels["five_training"],
            five_candidates=panels["five_candidate"],
            ten_training=panels["ten_training"],
            ten_candidates=panels["ten_candidate"],
            five_extras=extras_f,
            ten_extras=extras_t,
            output_dir=dest,
            resume=resume,
            five_frozen=panels["five_frozen"],
            ten_frozen=panels["ten_frozen"],
            force_retrain=not (extras_f or extras_t),
        )
        obs = run_fused_backtest(
            dest,
            trained["five_pred"],
            trained["ten_pred"],
            extra={
                "code": code,
                "role": spec["role"],
                "five_extras": extras_f,
                "ten_extras": extras_t,
                "repaired": True,
                "src": spec["code"],
            },
            resume=resume,
        )
        rec = {
            "code": code,
            "src": spec["code"],
            "role": spec["role"],
            "formal": spec.get("formal"),
            "status": obs.get("status"),
            "cagr": obs.get("cagr"),
            "sharpe": obs.get("sharpe"),
            "max_drawdown": obs.get("max_drawdown"),
            "completed_trades": obs.get("completed_trades"),
            "promotion_label": REPAIRED_ROUND18_CANDIDATE if spec.get("formal") else "DIAGNOSTIC_ONLY",
        }
        dump_json(dest / "summary.json", rec)
        rows.append(rec)

    formal = [r for r in rows if r.get("formal") and r.get("sharpe") is not None]
    best = None
    if formal:
        best = max(formal, key=lambda r: float(r.get("sharpe") or -999))
        best = {**best, "label": REPAIRED_ROUND18_CANDIDATE, "competes_with_o01_for_v5": False}
    payload = {
        "status": "REPAIRED_MATRIX_DONE",
        "r18_rebacktest_required": True,
        "r18_rebacktest_ran": True,
        "rows": rows,
        "repaired_best": best,
        "prove_B01": prove,
        "B00": b00,
        "round18_originals_untouched": True,
        "note": "Original F formulas only. No epsilon/head-weight search. Does not compete with O01 for v5.",
    }
    dump_json(dest_json, payload)
    lines = [
        "# Track B · R19_B_REPAIRED_*",
        "",
        f"status=`{payload['status']}`",
        "",
        "| code | src | Sharpe | CAGR | trades |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('code')} | {r.get('src')} | {r.get('sharpe')} | {r.get('cagr')} | {r.get('completed_trades')} |"
        )
    if best:
        lines += ["", f"repaired_best=`{best.get('code')}` Sharpe=`{best.get('sharpe')}` label=`{REPAIRED_ROUND18_CANDIDATE}`", ""]
    write_md(tb_dir("repair") / "13_repair.md", "\n".join(lines))
    return payload


__all__ = ["rebuild_clean_panels", "run_repair"]
