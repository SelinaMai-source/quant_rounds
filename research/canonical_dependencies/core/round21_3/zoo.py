"""更新 factor_zoo。数字只来自本轮产物。失败进 Rejected Archive。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import PROJECT_ROOT, ROUND21_3_ROOT
from .common import dump_json, load_json, write_md
from .contract import CROSS_SECTIONALLY_DEGENERATE, DUPLICATE, FACTOR_ZOO_ROOT, KEEP_BASELINE_V5, MD
from .n1n3 import N1N3_JSON
from .pit import PIT_JSON

ZOO_JSON = ROUND21_3_ROOT / "zoo" / "zoo.json"


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
    from .backtest import BT_JSON
    from .construct import CONSTRUCT_JSON
    from .correlate import CORR_JSON

    attr = load_json(ATTR_JSON)
    bt = load_json(BT_JSON)
    corr = load_json(CORR_JSON)
    pit = load_json(PIT_JSON)
    construct = load_json(CONSTRUCT_JSON)
    n1n3 = load_json(N1N3_JSON)
    nearest = corr.get("nearest") or {}
    rows = []
    for feat in list(pit.get("pit_pass") or []) + list(pit.get("blocked") or []) + list(construct.get("computed") or []):
        if any(r.get("factor_id") == feat for r in rows):
            continue
        nb = nearest.get(feat) or {}
        arow = next((x for x in (attr.get("rows") or []) if x.get("new_factor") == feat), {})
        rec = {
            "factor_id": feat,
            "round": "21.3",
            "nearest_legacy_factor": nb.get("majority_legacy"),
            "correlation": nb.get("mean_fold_signed"),
            "replacement_verdict": arow.get("interpretation") or KEEP_BASELINE_V5,
            "pit": next((r.get("pit_status") for r in (pit.get("rows") or []) if r.get("factor_id") == feat), None),
        }
        rows.append(rec)
    rejected = []
    if n1n3.get("verdict") == CROSS_SECTIONALLY_DEGENERATE:
        rejected.append({"factor_id": "r21_n1_lagged_peer_limit_up", "reason": CROSS_SECTIONALLY_DEGENERATE, "round": "21.3"})
        rejected.append({"factor_id": "r21_n3_peer_failed_spillover", "reason": CROSS_SECTIONALLY_DEGENERATE, "round": "21.3"})
    for d in construct.get("duplicates") or []:
        if d.get("verdict") == DUPLICATE:
            rejected.append({"factor_id": d.get("new"), "reason": f"{DUPLICATE} ≡ {d.get('old')}", "round": "21.3"})
    FACTOR_ZOO_ROOT.mkdir(parents=True, exist_ok=True)
    out = FACTOR_ZOO_ROOT / "round21_3_replacement.csv"
    _assert_zoo(out)
    pd.DataFrame(rows).to_csv(out, index=False)
    rej = FACTOR_ZOO_ROOT / "rejected_archive.csv"
    if rejected:
        prev = pd.read_csv(rej) if rej.is_file() else pd.DataFrame()
        frame = pd.concat([prev, pd.DataFrame(rejected)], ignore_index=True)
        frame = frame.drop_duplicates(subset=["factor_id", "reason"], keep="last")
        _assert_zoo(rej)
        frame.to_csv(rej, index=False)
    payload = {"status": "DONE", "n": len(rows), "rejected": rejected, "csv": str(out)}
    dump_json(ZOO_JSON, payload)
    write_md(ROUND21_3_ROOT / "zoo" / "zoo.md", f"# Round21.3 zoo\n\nN=`{len(rows)}` rejected=`{len(rejected)}`\n")
    return payload
