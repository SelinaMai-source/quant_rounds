"""zoo 追加 22.1 证据行。不改写 r21 行，不覆盖 r22 主行。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND22_1_ROOT, ROUND22_ROOT
from .common import dump_json, load_json, write_md
from .contract import FACTOR_SPECS, FACTOR_ZOO_ROOT, KEEP_BASELINE_V5, MD, PIT_PASS_IDS

ZOO_JSON = ROUND22_1_ROOT / "zoo" / "zoo.json"
REG_COLS = [
    "factor_id",
    "family",
    "short",
    "formula",
    "round",
    "values_constructed",
    "pit_status",
    "evidence_status",
    "rolling_status",
    "conditional_status",
    "shape_status",
    "top_tail_status",
    "ic_universe",
    "raw_rankic_e2",
    "icir_e2",
    "promotion_status",
    "zoo_status",
    "package_dir",
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
    decision = load_json(ROUND22_1_ROOT / "report" / "decision.json")
    verdict = decision.get("final_decision") or KEEP_BASELINE_V5
    ic = load_json(ROUND22_1_ROOT / "ic" / "ic.json")
    ic_rows = {r.get("factor_id"): r for r in (ic.get("rows") or [])}
    pit = load_json(ROUND22_ROOT / "pit" / "pit.json")
    pit_rows = {r.get("factor_id"): r for r in (pit.get("rows") or [])}
    rows = []
    for spec in FACTOR_SPECS:
        fid = spec["factor_id"]
        dest = ROUND22_1_ROOT / "factors" / fid
        icr = ic_rows.get(fid) or {}
        e2 = icr.get("label_e2_net") if isinstance(icr.get("label_e2_net"), dict) else {}
        roll = load_json(dest / "rolling.json")
        cond = load_json(dest / "conditional.json")
        shape = load_json(dest / "shape.json")
        tail = load_json(dest / "top_tail.json")
        pkg = load_json(dest / "package.json")
        rec = {c: "" for c in REG_COLS}
        rec.update(
            {
                "factor_id": fid,
                "family": spec["family"],
                "short": spec.get("short"),
                "formula": spec.get("formula"),
                "round": "22.1",
                "values_constructed": fid in PIT_PASS_IDS,
                "pit_status": (pit_rows.get(fid) or {}).get("pit_status"),
                "evidence_status": pkg.get("status"),
                "rolling_status": "OK" if roll else "MISSING",
                "conditional_status": cond.get("status") if cond else "MISSING",
                "shape_status": "OK" if shape else "MISSING",
                "top_tail_status": "OK" if tail else "MISSING",
                "ic_universe": "first_board_candidate" if fid in PIT_PASS_IDS else "",
                "raw_rankic_e2": (e2 or {}).get("mean"),
                "icir_e2": (e2 or {}).get("icir"),
                "promotion_status": verdict,
                "zoo_status": "EVIDENCE_22_1",
                "package_dir": str(dest),
                "copied_global_result": False,
            }
        )
        rows.append(rec)
    local = ROUND22_1_ROOT / "zoo" / "round22_1_evidence.csv"
    pd.DataFrame(rows).to_csv(local, index=False)
    remote = _assert_zoo(FACTOR_ZOO_ROOT / "round22_1_evidence.csv")
    pd.DataFrame(rows).to_csv(remote, index=False)
    yaml_path = _assert_zoo(FACTOR_ZOO_ROOT / "round22_1_registry.yaml")
    yaml_path.write_text(
        "round: 22.1\n"
        f"n: {len(rows)}\n"
        f"n_pit_pass: {len(PIT_PASS_IDS)}\n"
        f"decision: {verdict}\n"
        "note: sidecar evidence rows; r21/r22 registry rows were not rewritten\n",
        encoding="utf-8",
    )
    main_yaml = FACTOR_ZOO_ROOT / "registry.yaml"
    if main_yaml.is_file():
        text = main_yaml.read_text(encoding="utf-8")
        if "round: 22.1" not in text:
            main_yaml.write_text(text.rstrip() + f"\n# round 22.1 n={len(rows)} decision={verdict}\n", encoding="utf-8")
    payload = {"status": "DONE", "n": len(rows), "verdict": verdict, "rewrote_r21": False, "rewrote_r22": False}
    dump_json(ZOO_JSON, payload)
    write_md(MD["12"], f"# Round22.1 · 12 Zoo\n\n追加 22.1 证据行 {len(rows)}。未改写 r21/r22 主行。决策 `{verdict}`。\n")
    return payload
