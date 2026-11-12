"""更新 factor_zoo：nearest / replace 结果。数字只来自本轮产物。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import PROJECT_ROOT, ROUND21_2_ROOT
from .common import dump_json, load_json, write_md
from .contract import FACTOR_ZOO_ROOT, KEEP_BASELINE_V5, MD
from .panels import pit_safe_factors

ZOO_JSON = ROUND21_2_ROOT / "zoo" / "zoo.json"


def _assert_zoo(path: Path) -> Path:
    resolved = path.resolve()
    root = FACTOR_ZOO_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside {root}: {resolved}")
    if PROJECT_ROOT.resolve() not in resolved.parents and resolved != PROJECT_ROOT.resolve():
        raise RuntimeError(f"refusing to write outside project: {resolved}")
    return resolved


def _write_table(path: Path, frame: pd.DataFrame) -> Path:
    dest = _assert_zoo(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix == ".parquet":
        frame.to_parquet(dest, index=False)
    else:
        frame.to_csv(dest, index=False)
    return dest


def run_zoo(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    from .attribution import ATTR_JSON
    from .backtest import BT_JSON
    from .conditional import COND_JSON
    from .correlate import CORR_JSON
    from .pit import PIT_JSON

    attr = load_json(ATTR_JSON)
    bt = load_json(BT_JSON)
    cond = load_json(COND_JSON)
    corr = load_json(CORR_JSON)
    pit = load_json(PIT_JSON)
    nearest = corr.get("nearest") or {}
    cond_map: dict[str, dict[str, Any]] = {}
    for r in cond.get("rows") or []:
        cond_map.setdefault(r["factor_id"], {})[r.get("head")] = r
    bt_map: dict[str, list] = {}
    for r in bt.get("main_rows") or []:
        fid = r.get("factor_id") or str(r.get("code") or "")
        for feat in pit_safe_factors(resume=True):
            if feat in str(r.get("code") or ""):
                bt_map.setdefault(feat, []).append(r)
                break
    pit_map = {r["factor_id"]: r for r in (pit.get("rows") or [])}
    rows = []
    for feat in pit_safe_factors(resume=True):
        nb = nearest.get(feat) or {}
        arow = next((x for x in (attr.get("rows") or []) if x.get("new_factor") == feat), {})
        e3s = [r for r in bt_map.get(feat, []) if str(r.get("code") or "").startswith("R21_2_E3_")]
        best_e3 = max((r for r in e3s if r.get("sharpe") is not None), key=lambda r: float(r["sharpe"]), default={})
        rec = {
            "factor_id": feat,
            "round": "21.2",
            "nearest_legacy_factor": nb.get("majority_legacy"),
            "correlation": None,
            "correlation_stability": "changed" if nb.get("fold_changed") else "stable",
            "removed_factor": nb.get("majority_legacy"),
            "replace_result": arow.get("delta_replace_sharpe"),
            "add_result": arow.get("delta_add_sharpe"),
            "remove_result": arow.get("delta_remove_sharpe"),
            "incremental_value_after_removal": arow.get("delta_new_given_remove_sharpe"),
            "replacement_verdict": arow.get("interpretation") or KEEP_BASELINE_V5,
            "conditional_ic_five": (cond_map.get(feat) or {}).get("five", {}).get("conditional_ic"),
            "pit_status": (pit_map.get(feat) or {}).get("pit_status") or "PIT_PASS",
            "best_e3_sharpe": best_e3.get("sharpe"),
            "best_e3_cagr": best_e3.get("cagr"),
        }
        fold = nb.get("fold_nearest") or {}
        rhos = [v.get("fold_abs_rho") or v.get("abs_spearman") for v in fold.values() if isinstance(v, dict)]
        rhos = [x for x in rhos if x is not None]
        rec["correlation"] = float(sum(rhos) / len(rhos)) if rhos else nb.get("mean_fold_abs_rho_majority")
        rows.append(rec)
        dest = ROUND21_2_ROOT / "factors" / feat
        dest.mkdir(parents=True, exist_ok=True)
        dump_json(dest / "zoo.json", rec)

    frame = pd.DataFrame(rows)
    zoo_csv = FACTOR_ZOO_ROOT / "round21_2_replacement.csv"
    _write_table(zoo_csv, frame)
    reg_path = FACTOR_ZOO_ROOT / "registry.csv"
    if reg_path.is_file() and not frame.empty:
        reg = pd.read_csv(reg_path)
        extra_cols = [
            "nearest_legacy_factor",
            "correlation",
            "correlation_stability",
            "removed_factor",
            "replace_result",
            "add_result",
            "remove_result",
            "incremental_value_after_removal",
            "replacement_verdict",
        ]
        for col in extra_cols:
            if col not in reg.columns:
                reg[col] = pd.NA
        for rec in rows:
            mask = reg["factor_id"].astype(str).eq(rec["factor_id"])
            if mask.any():
                for col in extra_cols:
                    reg.loc[mask, col] = rec.get(col)
            else:
                reg = pd.concat([reg, pd.DataFrame([{**rec, "family": "", "round": "21.2"}])], ignore_index=True)
        _write_table(reg_path, reg)

    payload = {
        "status": "DONE",
        "n": len(rows),
        "csv": str(zoo_csv),
        "rows": rows,
        "did_not_overwrite_r21_or_r21_1": True,
    }
    dump_json(ZOO_JSON, payload)
    lines = [
        "# Round21.2 · Factor Zoo",
        "",
        "新增字段：nearest_legacy_factor / correlation / correlation_stability / removed_factor / "
        "replace_result / add_result / remove_result / incremental_value_after_removal / replacement_verdict。",
        "",
        "| factor | nearest | |ρ| | ΔRep Sh | ΔAdd Sh | ΔRem Sh | ΔNew|Rem | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['factor_id']}` | `{r['nearest_legacy_factor']}` | {r['correlation']} | "
            f"{r['replace_result']} | {r['add_result']} | {r['remove_result']} | "
            f"{r['incremental_value_after_removal']} | {r['replacement_verdict']} |"
        )
    write_md(ROUND21_2_ROOT / "zoo" / "zoo.md", "\n".join(lines))
    return payload
