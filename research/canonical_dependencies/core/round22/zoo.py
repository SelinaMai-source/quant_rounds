"""把 r22_* 写进 factor zoo。只替换本轮行，不改 r21 行。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND22_ROOT
from .common import dump_json, load_json, write_md
from .contract import FACTOR_SPECS, FACTOR_ZOO_ROOT, KEEP_BASELINE_V5, MD

ZOO_JSON = ROUND22_ROOT / "zoo" / "zoo.json"
REG_COLS = [
    "factor_id",
    "family",
    "short",
    "formula",
    "round",
    "values_constructed",
    "pit_status",
    "model_eligible",
    "redundancy_action",
    "cluster",
    "best_abs_icir",
    "best_abs_mean",
    "raw_rankic_e2",
    "zyquant_nearest",
    "zyquant_abs_rho",
    "near_neighbors",
    "promotion_status",
    "promotion_reason",
    "promotion_passed",
    "zoo_status",
    "copied_global_result",
    "package_dir",
    "nearest_legacy_factor",
    "correlation",
    "correlation_stability",
    "removed_factor",
    "replace_result",
    "add_result",
    "remove_result",
    "incremental_value_after_removal",
    "replacement_verdict",
    "conditional_ic_five",
    "best_e3_sharpe",
    "best_e3_cagr",
]


def _assert_zoo(path: Path) -> Path:
    resolved = path.resolve()
    root = FACTOR_ZOO_ROOT.resolve()
    if resolved == root or root in resolved.parents:
        return resolved
    raise RuntimeError(f"refusing to write outside {root}: {resolved}")


def _drop_r22(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "factor_id" not in frame.columns:
        return frame
    mask = frame["factor_id"].astype(str).str.startswith("r22_")
    return frame.loc[~mask].copy()


def run_zoo(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    pit = load_json(ROUND22_ROOT / "pit" / "pit.json")
    corr = load_json(ROUND22_ROOT / "correlation" / "correlation.json")
    ic = load_json(ROUND22_ROOT / "ic" / "ic.json")
    decision = load_json(ROUND22_ROOT / "report" / "decision.json") if (ROUND22_ROOT / "report" / "decision.json").is_file() else {}
    verdict = decision.get("final_decision") or KEEP_BASELINE_V5
    pit_rows = {r["factor_id"]: r for r in (pit.get("rows") or [])}
    corr_rows = {r["factor_id"]: r for r in (corr.get("rows") or [])}
    ic_rows = {}
    for row in ic.get("rows") or []:
        ic_rows.setdefault(row.get("factor_id"), []).append(row)
    abde_cols = set(load_json(ROUND22_ROOT / "construct" / "abde.json").get("columns") or [])
    rows = []
    rejected = []
    for spec in FACTOR_SPECS:
        fid = spec["factor_id"]
        p = pit_rows.get(fid) or {}
        c = corr_rows.get(fid) or {}
        constructed = fid in abde_cols or bool(p.get("n_days"))
        status = p.get("pit_status") or "NOT_RUN"
        action = c.get("five_action") or c.get("ten_action") or ""
        rec = {col: "" for col in REG_COLS}
        rec.update(
            {
                "factor_id": fid,
                "family": spec["family"],
                "short": spec.get("short"),
                "formula": spec.get("formula"),
                "round": "22",
                "values_constructed": bool(constructed and status not in {"DUPLICATE_EXPRESSION"} and not str(status).startswith("BLOCKED")),
                "pit_status": status,
                "model_eligible": bool(p.get("model_eligible")),
                "redundancy_action": action,
                "promotion_status": verdict,
                "promotion_reason": status,
                "promotion_passed": False,
                "zoo_status": "REJECTED" if status != "PIT_PASS" else "RESEARCH",
                "copied_global_result": False,
                "package_dir": str(ROUND22_ROOT / "factors" / fid),
                "nearest_legacy_factor": c.get("five_nearest") or c.get("duplicate_hit") or "",
                "correlation": c.get("five_abs_rho"),
                "replacement_verdict": verdict,
            }
        )
        rows.append(rec)
        if status != "PIT_PASS" or not p.get("model_eligible"):
            rejected.append(
                {
                    "factor_id": fid,
                    "family": spec["family"],
                    "action": status,
                    "pit_status": status,
                    "reason": p.get("reason") or status,
                    "round": "22",
                    "archive": "REJECTED",
                }
            )
    FACTOR_ZOO_ROOT.mkdir(parents=True, exist_ok=True)
    reg_path = _assert_zoo(FACTOR_ZOO_ROOT / "registry.csv")
    prev = pd.read_csv(reg_path) if reg_path.is_file() else pd.DataFrame(columns=REG_COLS)
    kept = _drop_r22(prev)
    merged = pd.concat([kept, pd.DataFrame(rows)], ignore_index=True)
    merged.to_csv(reg_path, index=False)
    rej_path = _assert_zoo(FACTOR_ZOO_ROOT / "rejected_archive.csv")
    prev_rej = pd.read_csv(rej_path) if rej_path.is_file() else pd.DataFrame()
    rej_kept = _drop_r22(prev_rej)
    rej_out = pd.concat([rej_kept, pd.DataFrame(rejected)], ignore_index=True) if rejected else rej_kept
    rej_out.to_csv(rej_path, index=False)
    repl_cols = ["factor_id", "round", "nearest_legacy_factor", "correlation", "replacement_verdict", "pit_status", "model_eligible"]
    repl = pd.DataFrame(rows)[repl_cols]
    local = ROUND22_ROOT / "zoo" / "round22_replacement.csv"
    remote = _assert_zoo(FACTOR_ZOO_ROOT / "round22_replacement.csv")
    repl.to_csv(local, index=False)
    repl.to_csv(remote, index=False)
    side = _assert_zoo(FACTOR_ZOO_ROOT / "round22_registry.yaml")
    side.write_text(
        "round: 22\n"
        f"n: {len(rows)}\n"
        f"rejected: {len(rejected)}\n"
        f"decision: {verdict}\n"
        "note: sidecar only; registry.csv r21 rows were not rewritten\n",
        encoding="utf-8",
    )
    yaml_path = FACTOR_ZOO_ROOT / "registry.yaml"
    if yaml_path.is_file():
        text = yaml_path.read_text(encoding="utf-8")
        if "round: 22\n" not in text and "round: 22 " not in text:
            yaml_path.write_text(text.rstrip() + f"\n# round 22 n={len(rows)} rejected={len(rejected)} decision={verdict}\n", encoding="utf-8")
    n_r21 = int(kept["factor_id"].astype(str).str.startswith("r21_").sum()) if "factor_id" in kept.columns else 0
    payload = {"status": "DONE", "n": len(rows), "n_rejected": len(rejected), "n_r21_kept": n_r21, "verdict": verdict}
    dump_json(ZOO_JSON, payload)
    write_md(MD["20"], f"# Round22 · 20 Zoo\n\n写入 r22 行={len(rows)}。拒绝归档={len(rejected)}。保留的 r21 行={n_r21}。决策 `{verdict}`。没有改写 r21 行。\n")
    return payload
