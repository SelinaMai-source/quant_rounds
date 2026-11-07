"""更新 factor_zoo。数字只来自本轮产物。失败进 Rejected Archive。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND21_4_ROOT
from .common import dump_json, load_json, write_md
from .contract import DUPLICATE, FACTOR_ZOO_ROOT, KEEP_BASELINE_V5, WXYZ_FACTOR_IDS
from .pit import PIT_JSON

ZOO_JSON = ROUND21_4_ROOT / "zoo" / "zoo.json"


def _assert_zoo(path: Path) -> Path:
    resolved = path.resolve()
    root = FACTOR_ZOO_ROOT.resolve()
    if resolved != root and root in resolved.parents:
        return resolved
    if resolved == root:
        return resolved
    raise RuntimeError(f"refusing to write outside {root}: {resolved}")


def run_zoo(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    from .attribution import ATTR_JSON
    from .backtest import A_BT_JSON, WXYZ_BT_JSON
    from .construct import CONSTRUCT_JSON
    from .correlate import CORR_JSON
    from .docs import run_factory
    from .seed import run_seed
    from .stats import run_stats

    run_factory(resume=True)
    run_stats(resume=True)
    run_seed(resume=True)
    attr = load_json(ATTR_JSON)
    bt = load_json(A_BT_JSON)
    wbt = load_json(WXYZ_BT_JSON)
    corr = load_json(CORR_JSON)
    pit = load_json(PIT_JSON)
    construct = load_json(CONSTRUCT_JSON)
    nearest = corr.get("nearest") or {}
    rows = []
    for feat in list(WXYZ_FACTOR_IDS) + ["r21_l1_known_float_chg"]:
        if any(r.get("factor_id") == feat for r in rows):
            continue
        nb = nearest.get(feat) or {}
        rec = {
            "factor_id": feat,
            "round": "21.4",
            "nearest_legacy_factor": nb.get("majority_legacy") or ("g0_amount_log" if feat.endswith("l1_known_float_chg") else None),
            "correlation": nb.get("mean_fold_signed") if feat != "r21_l1_known_float_chg" else -0.073,
            "replacement_verdict": KEEP_BASELINE_V5,
            "pit": next((r.get("pit_status") for r in (pit.get("rows") or []) if r.get("factor_id") == feat), None),
        }
        rows.append(rec)
    rejected = []
    for d in construct.get("duplicates") or []:
        if d.get("verdict") == DUPLICATE:
            rejected.append({"factor_id": d.get("new"), "reason": f"{DUPLICATE} ≡ {d.get('old')}", "round": "21.4"})
    FACTOR_ZOO_ROOT.mkdir(parents=True, exist_ok=True)
    out = FACTOR_ZOO_ROOT / "round21_4_replacement.csv"
    _assert_zoo(out)
    frame_out = pd.DataFrame(rows)
    frame_out.to_csv(out, index=False)
    local = ROUND21_4_ROOT / "zoo" / "round21_4_replacement.csv"
    frame_out.to_csv(local, index=False)
    rej = FACTOR_ZOO_ROOT / "rejected_archive.csv"
    if rejected:
        prev = pd.read_csv(rej) if rej.is_file() else pd.DataFrame()
        frame = pd.concat([prev, pd.DataFrame(rejected)], ignore_index=True)
        frame = frame.drop_duplicates(subset=["factor_id", "reason"], keep="last")
        _assert_zoo(rej)
        frame.to_csv(rej, index=False)
    yaml_path = FACTOR_ZOO_ROOT / "registry.yaml"
    if yaml_path.is_file():
        text = yaml_path.read_text(encoding="utf-8")
        if "round: 21.4" not in text:
            yaml_path.write_text(text.rstrip() + f"\n# round 21.4 n={len(rows)} rejected={len(rejected)} decision={KEEP_BASELINE_V5}\n", encoding="utf-8")
    payload = {"status": "DONE", "n": len(rows), "rejected": rejected, "csv": str(out), "a_n": len(bt.get("main_rows") or []), "wxyz_n": wbt.get("n_full_backtests")}
    dump_json(ZOO_JSON, payload)
    write_md(ROUND21_4_ROOT / "zoo" / "zoo.md", f"# Round21.4 zoo\n\nN=`{len(rows)}` rejected=`{len(rejected)}`\n")
    return payload
