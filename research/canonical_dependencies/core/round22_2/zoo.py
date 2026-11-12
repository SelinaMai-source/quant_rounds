"""zoo 追加 22.2 行。DUPLICATE/失败进 rejected archive。不改写 r21/r22/r22.1 主行。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import dump_json, load_json, write_md
from .contract import FACTOR_ZOO_ROOT, KEEP_BASELINE_V5, MD, NEW_FACTOR_SPECS, ROUND22_2_ROOT

ZOO_JSON = ROUND22_2_ROOT / "zoo" / "zoo.json"
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
    decision = load_json(ROUND22_2_ROOT / "report" / "decision.json")
    verdict = decision.get("final_decision") or KEEP_BASELINE_V5
    pit = load_json(ROUND22_2_ROOT / "pit" / "c_pit.json")
    ev = load_json(ROUND22_2_ROOT / "evidence" / "c14.json")
    disc = load_json(ROUND22_2_ROOT / "data_recovery" / "discover.json")
    rejected = []
    rows = []

    def add(fid: str, family: str, short: str, formula: str, pit_status: Any, ic=None, lineage: str = "", constructed: bool = False, zoo_status: str = "R22_2") -> None:
        rec = {c: "" for c in REG_COLS}
        rec.update(
            {
                "factor_id": fid,
                "family": family,
                "short": short,
                "formula": formula,
                "round": "22.2",
                "values_constructed": constructed,
                "pit_status": pit_status,
                "ic_universe": "first_board_candidate",
                "raw_rankic_e2": None if not ic else (ic.get("label_e2_net") or {}).get("mean_2018_2026") or (ic.get("label_e2_net") or {}).get("mean_all"),
                "icir_e2": None if not ic else (ic.get("label_e2_net") or {}).get("icir_all"),
                "promotion_status": verdict,
                "zoo_status": zoo_status,
                "package_dir": str(ROUND22_2_ROOT / "factors" / fid),
                "lineage": lineage,
                "copied_global_result": False,
            }
        )
        rows.append(rec)
        if str(pit_status) in {"DUPLICATE", "BLOCKED_PIT", "BLOCKED_DATA", "PIT_FAIL_COVERAGE", "RESEARCH_ONLY_BACKFILL"}:
            rejected.append({"factor_id": fid, "family": family, "reason": pit_status, "round": "22.2", "archive": "REJECTED"})

    c1 = pit.get("c1") or {}
    c4 = pit.get("c4") or {}
    ic_map = ev.get("ic") or {}
    add(c1.get("model_id") or "r22_c1_xl_net_imbalance", "C", "超大单净额占成交额", c1.get("formula") or "", c1.get("pit_status"), ic_map.get(c1.get("model_id") or ""), "22.1 C1 frozen", True)
    add("r22_c4_flow_persist_5", "C", "净流入符号持续", c4.get("formula") or "", c4.get("pit_status"), ic_map.get("r22_c4_flow_persist_5"), "22.1 C4 frozen", True)
    hn = disc.get("hn") or {}
    add("r22_2_hn1_neg_log_holder_ratio", "HN", "户数对数比符号翻转", (hn.get("hn1") or {}).get("formula") or "", (hn.get("hn1") or {}).get("status"), None, "DUPLICATE -r21_d1", False, "DUPLICATE")
    add("r22_2_hn2_avg_holding_chg", "HN", "户均持股变化", (hn.get("hn2") or {}).get("formula") or "", (hn.get("hn2") or {}).get("status"), None, "new vs r21_d1/d3", bool((hn.get("hn2") or {}).get("values_constructed")))
    bt = disc.get("bt") or {}
    add("r22_2_bt1_block_discount", "BT", "大宗折价", "1-B_SHARE_PRICE/close", bt.get("status"), None, "DUPLICATE r21_e1", False, "DUPLICATE")
    mg = disc.get("mg") or {}
    add("r22_2_mg1_fin_buy_intensity", "MG", "融资买入强度", mg.get("formula") or "", mg.get("status"), None, "DUPLICATE r21_f3", False, "DUPLICATE")
    fs = disc.get("fs") or {}
    add("r22_2_fs1_accrual_minus_ocf", "FS", "应计减经营现金流", " (NI-OCF)/avg_assets ", fs.get("status"), None, "priority below C/HN", False)

    local = ROUND22_2_ROOT / "zoo" / "round22_2_evidence.csv"
    pd.DataFrame(rows).to_csv(local, index=False)
    remote = _assert_zoo(FACTOR_ZOO_ROOT / "round22_2_evidence.csv")
    pd.DataFrame(rows).to_csv(remote, index=False)
    if rejected:
        rej_path = _assert_zoo(FACTOR_ZOO_ROOT / "rejected_archive.csv")
        existing = pd.read_csv(rej_path) if rej_path.is_file() else pd.DataFrame()
        add_df = pd.DataFrame(rejected)
        keep = add_df.loc[~add_df["factor_id"].isin(set(existing["factor_id"])) ] if not existing.empty and "factor_id" in existing.columns else add_df
        if not keep.empty:
            pd.concat([existing, keep], ignore_index=True).to_csv(rej_path, index=False)
        pd.DataFrame(rejected).to_csv(ROUND22_2_ROOT / "rejected" / "rejected_archive.csv", index=False)
    trial_src = ROUND22_2_ROOT / "model" / "trial_registry.csv"
    if trial_src.is_file():
        trial_src.read_text(encoding="utf-8")
        remote_trial = _assert_zoo(FACTOR_ZOO_ROOT / "trial_registry.csv")
        add_t = pd.read_csv(trial_src)
        add_t["round"] = "22.2"
        if remote_trial.is_file():
            old = pd.read_csv(remote_trial)
            if "trial_id" in add_t.columns and "trial_id" in old.columns:
                add_t = add_t.loc[~add_t["trial_id"].isin(set(old["trial_id"]))]
            if not add_t.empty:
                pd.concat([old, add_t], ignore_index=True).to_csv(remote_trial, index=False)
        else:
            add_t.to_csv(remote_trial, index=False)
    yaml_path = _assert_zoo(FACTOR_ZOO_ROOT / "round22_2_registry.yaml")
    yaml_path.write_text(f"round: 22.2\nn: {len(rows)}\ndecision: {verdict}\nnote: sidecar rows; r21/r22/r22.1 not rewritten\n", encoding="utf-8")
    main_yaml = FACTOR_ZOO_ROOT / "registry.yaml"
    if main_yaml.is_file():
        text = main_yaml.read_text(encoding="utf-8")
        if "round: 22.2" not in text:
            main_yaml.write_text(text.rstrip() + f"\n# round 22.2 n={len(rows)} decision={verdict}\n", encoding="utf-8")
    payload = {"status": "DONE", "n": len(rows), "n_rejected": len(rejected), "verdict": verdict, "rewrote_r21": False, "rewrote_r22": False}
    dump_json(ZOO_JSON, payload)
    write_md(MD["19"], f"# Round22.2 · 19 Zoo\n\n追加 {len(rows)} 行。rejected={len(rejected)}。未改写 r21/r22/r22.1 主行。决策 `{verdict}`。\n")
    return payload
