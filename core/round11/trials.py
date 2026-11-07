"""研究自由度：扫 tracking + round1–10 + replay + registry。禁止 n_trials=4。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import RESULT_ROOT, ROUND11_ROOT
from .common import daily_from_nav, dump_json, stage_dir, write_md

TRIALS_MD = ROUND11_ROOT / "08_research_trial_registry.md"
REGISTRY_CSV = ROUND11_ROOT / "trials" / "research_trial_registry.csv"


def _summaries_under(root: Path) -> list[dict[str, Any]]:
    rows = []
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob("summary.json")):
        if any(part in {"parity", "configs", "metrics"} for part in path.parts):
            # still count experiment dirs that have summary at leaf
            pass
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rel = path.relative_to(RESULT_ROOT)
        nav = path.parent / "nav.parquet"
        rows.append({
            "trial_id": payload.get("code") or path.parent.name,
            "source_round": rel.parts[0] if rel.parts else "",
            "path": str(path.parent),
            "cagr": payload.get("cagr") or payload.get("annualized_return"),
            "sharpe": payload.get("sharpe") or payload.get("sharpe_zero_rate"),
            "max_drawdown": payload.get("max_drawdown"),
            "completed_trades": payload.get("completed_trades"),
            "has_nav": nav.is_file(),
            "has_daily": nav.is_file(),
        })
    return rows


def _registry_csvs() -> list[dict[str, Any]]:
    rows = []
    for path in RESULT_ROOT.rglob("experiment_registry.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        for rec in frame.to_dict("records"):
            rows.append({
                "trial_id": rec.get("experiment_id") or rec.get("code") or rec.get("ID"),
                "source_round": path.parent.name,
                "path": str(path),
                "cagr": rec.get("CAGR") or rec.get("cagr"),
                "sharpe": rec.get("Sharpe") or rec.get("sharpe"),
                "max_drawdown": rec.get("MaxDD") or rec.get("max_drawdown"),
                "has_nav": False,
                "has_daily": False,
                "registry_only": True,
            })
    return rows


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('source_round')}::{row.get('trial_id')}"
        prev = seen.get(key)
        if prev is None or (row.get("has_nav") and not prev.get("has_nav")):
            seen[key] = row
    return list(seen.values())


def run_trials(**_: Any) -> dict[str, Any]:
    dest = stage_dir("trials")
    discovered = _summaries_under(RESULT_ROOT)
    registry_only = _registry_csvs()
    all_rows = _dedupe(discovered + registry_only)
    documented = [r for r in all_rows if r.get("cagr") is not None or r.get("sharpe") is not None]
    with_daily = [r for r in all_rows if r.get("has_nav")]
    # conservative upper: every summary + registry + replay + known skipped searches
    n_doc = int(len(documented))
    n_disc = int(len(all_rows))
    n_upper = int(max(n_disc * 2, n_disc + 50))
    if n_doc <= 4:
        raise RuntimeError("n_trials collapsed to <=4; refuse to treat as multiple-testing universe")
    frame = pd.DataFrame(all_rows)
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REGISTRY_CSV, index=False)
    matrix_meta = []
    for row in with_daily:
        nav_path = Path(row["path"]) / "nav.parquet"
        if not nav_path.is_file():
            continue
        try:
            s = daily_from_nav(pd.read_parquet(nav_path))
        except Exception:
            continue
        matrix_meta.append({
            "trial_id": row["trial_id"],
            "source_round": row["source_round"],
            "n": int(len(s)),
            "sharpe": row.get("sharpe"),
            "nav_path": str(nav_path),
        })
    dump_json(dest / "daily_matrix_index.json", {"n": len(matrix_meta), "rows": matrix_meta})
    payload = {
        "status": "DONE",
        "n_documented": n_doc,
        "n_full_discovered": n_disc,
        "n_conservative_upper": n_upper,
        "n_with_daily": int(len(with_daily)),
        "n_matrix": int(len(matrix_meta)),
        "forbid_n4": True,
        "csv": str(REGISTRY_CSV),
        "note": "禁止把 n_trials=4（仅 v1–v4）当成 multiple-testing 通过。",
        "post_hoc_diagnostic": True,
    }
    dump_json(dest / "trials.json", payload)
    write_md(
        TRIALS_MD,
        "\n".join([
            "# Round 11 · research trial registry",
            "",
            f"- documented = **{n_doc}**",
            f"- full discovered = **{n_disc}**",
            f"- conservative upper = **{n_upper}**",
            f"- 有日收益可对齐 = **{len(with_daily)}**",
            "",
            "禁止 n_trials=4。完整研究宇宙不可恢复则后续 PBO 标 `TRUE_PBO_NOT_FULLY_IDENTIFIABLE`。",
            "",
            f"CSV：`{REGISTRY_CSV}`",
            "",
        ]),
    )
    return payload


def load_trials() -> dict[str, Any]:
    path = ROUND11_ROOT / "trials" / "trials.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


__all__ = ["load_trials", "run_trials"]
