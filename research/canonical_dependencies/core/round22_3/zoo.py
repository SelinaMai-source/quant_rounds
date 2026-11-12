"""zoo 追加 22.3 行。不改写 r21/r22/r22.1/r22.2 主行。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import dump_json, load_json, write_md
from .contract import FACTOR_ZOO_ROOT, KEEP_BASELINE_V5, MD, NEW_FACTOR_SPECS, ROUND22_3_ROOT

ZOO_JSON = ROUND22_3_ROOT / "zoo" / "zoo.json"
REG_COLS = [
    "factor_id",
    "family",
    "short",
    "formula",
    "round",
    "values_constructed",
    "pit_status",
    "evidence_status",
    "ic_universe",
    "raw_rankic_e2",
    "icir_e2",
    "promotion_status",
    "zoo_status",
    "package_dir",
    "lineage",
    "copied_global_result",
]


def _assert_zoo(path: Path) -> Path:
    resolved = path.resolve()
    root = FACTOR_ZOO_ROOT.resolve()
    if resolved == root or root in resolved.parents:
        return resolved
    raise RuntimeError(f"refusing to write outside {root}: {resolved}")


def run_zoo(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    decision = load_json(ROUND22_3_ROOT / "report" / "decision.json")
    verdict = decision.get("final_decision") or KEEP_BASELINE_V5
    c4 = load_json(ROUND22_3_ROOT / "c4" / "c4.json")
    hn = load_json(ROUND22_3_ROOT / "hn2" / "hn2.json")
    fs = load_json(ROUND22_3_ROOT / "fs" / "fs.json")
    rejected = []
    rows = []

    def add(fid: str, family: str, short: str, formula: str, pit_status: Any, lineage: str = "", constructed: bool = False, zoo_status: str = "R22_3") -> None:
        rec = {c: "" for c in REG_COLS}
        rec.update(
            {
                "factor_id": fid,
                "family": family,
                "short": short,
                "formula": formula,
                "round": "22.3",
                "values_constructed": constructed,
                "pit_status": pit_status,
                "ic_universe": "first_board_candidate",
                "promotion_status": verdict,
                "zoo_status": zoo_status,
                "package_dir": str(ROUND22_3_ROOT / "factors" / fid),
                "lineage": lineage,
                "copied_global_result": False,
            }
        )
        rows.append(rec)
        if str(pit_status) in {"DUPLICATE", "BLOCKED_PIT", "BLOCKED_DATA", "BLOCKED_PERIOD_MISMATCH", "PIT_FAIL_COVERAGE"}:
            rejected.append({"factor_id": fid, "family": family, "reason": pit_status, "round": "22.3", "archive": "REJECTED"})

    add("r22_2_c1_xl_net_imbalance_lag1", "C", "超大单净额滞后", "D-1 NET_FLOW_XL/TURNOVER_VALUE", "LAGGED_SAFE_RESEARCH", "22.2 C1 lag1 full-history replay", True)
    add("r22_c4_flow_persist_5", "C", "净流入符号持续", (c4.get("formula") or {}).get("frozen") or "", (c4.get("formula") or {}).get("pit_status"), "22.2 C4 full-history replay", True)
    add("r22_2_hn2_avg_holding_chg", "HN", "户均持股变化", (hn.get("identity") or {}).get("formula_locked") or "", ((hn.get("pit") or {}).get("status")), "Δlog(circ)-Δlog(holders) check", True)
    for spec in NEW_FACTOR_SPECS:
        if spec["factor_id"].startswith("r22_3_fs"):
            block = (fs.get("fs1") if "fs1" in spec["factor_id"] else fs.get("fs2")) or {}
            add(spec["factor_id"], "FS", spec["short"], spec["formula"], fs.get("status") or block.get("status"), spec.get("lineage") or "", bool(block.get("constructed")))

    local = ROUND22_3_ROOT / "zoo" / "round22_3_evidence.csv"
    pd.DataFrame(rows).to_csv(local, index=False)
    remote = _assert_zoo(FACTOR_ZOO_ROOT / "round22_3_evidence.csv")
    pd.DataFrame(rows).to_csv(remote, index=False)
    if rejected:
        rej_path = _assert_zoo(FACTOR_ZOO_ROOT / "rejected_archive.csv")
        existing = pd.read_csv(rej_path) if rej_path.is_file() else pd.DataFrame()
        add_df = pd.DataFrame(rejected)
        keep = add_df.loc[~add_df["factor_id"].isin(set(existing["factor_id"]))] if not existing.empty and "factor_id" in existing.columns else add_df
        if not keep.empty:
            pd.concat([existing, keep], ignore_index=True).to_csv(rej_path, index=False)
        pd.DataFrame(rejected).to_csv(ROUND22_3_ROOT / "rejected" / "rejected_archive.csv", index=False)
    trial_src = ROUND22_3_ROOT / "model" / "trial_registry.csv"
    if trial_src.is_file():
        remote_trial = _assert_zoo(FACTOR_ZOO_ROOT / "trial_registry.csv")
        add_t = pd.read_csv(trial_src)
        add_t["round"] = "22.3"
        if remote_trial.is_file():
            old = pd.read_csv(remote_trial)
            if "trial_id" in add_t.columns and "trial_id" in old.columns:
                add_t = add_t.loc[~add_t["trial_id"].isin(set(old["trial_id"]))]
            if not add_t.empty:
                pd.concat([old, add_t], ignore_index=True).to_csv(remote_trial, index=False)
        else:
            add_t.to_csv(remote_trial, index=False)
    yaml_path = _assert_zoo(FACTOR_ZOO_ROOT / "round22_3_registry.yaml")
    yaml_path.write_text(f"round: 22.3\nn: {len(rows)}\ndecision: {verdict}\nnote: sidecar rows; prior rounds not rewritten\n", encoding="utf-8")
    main_yaml = FACTOR_ZOO_ROOT / "registry.yaml"
    if main_yaml.is_file():
        text = main_yaml.read_text(encoding="utf-8")
        if "round: 22.3" not in text:
            main_yaml.write_text(text.rstrip() + f"\n# round 22.3 n={len(rows)} decision={verdict}\n", encoding="utf-8")
    payload = {"status": "DONE", "n": len(rows), "n_rejected": len(rejected), "verdict": verdict, "rewrote_prior": False}
    dump_json(ZOO_JSON, payload)
    write_md(MD["19"], f"# Round22.3 · 19 Zoo\n\n追加 {len(rows)} 行。rejected={len(rejected)}。未改写先前主行。决策 `{verdict}`。\n")
    return payload
