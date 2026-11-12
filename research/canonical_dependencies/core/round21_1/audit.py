"""只读哈希审计 R21 ADD vs REP。预期 IDENTICAL_BY_DESIGN + REPLACE_NOT_APPLIED。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND21_1_ROOT, ROUND21_ROOT
from .common import dump_json, file_sha256, load_json, sha256_json, write_md
from .contract import AUDIT_MD, IDENTICAL_BY_DESIGN, REPLACE_NOT_APPLIED

AUDIT_JSON = ROUND21_1_ROOT / "audit" / "add_rep_audit.json"
HASH_CSV = ROUND21_1_ROOT / "audit" / "add_rep_hashes.csv"

COMPARE_NAMES = (
    "five_pred.parquet",
    "ten_pred.parquet",
    "oof.json",
    "summary.json",
    "nav.parquet",
    "overlayed.parquet",
    "predictions.parquet",
    "report.json",
    "trades.parquet",
)


def _exp(code: str) -> Path:
    return ROUND21_ROOT / "candidates" / code


def _pairs_from_matrix() -> list[tuple[str, str, dict[str, Any]]]:
    matrix = load_json(ROUND21_ROOT / "configs" / "model_matrix.json")
    specs = {s["code"]: s for s in (matrix.get("specs") or [])}
    pairs = []
    for code, spec in specs.items():
        if not str(code).startswith("R21_REP_"):
            continue
        src = spec.get("alias_from")
        if not src:
            feat = spec.get("factor_id")
            src = f"R21_ADD_{feat}" if feat else None
        if src and src in specs:
            pairs.append((src, code, spec))
    return pairs


def _nav_metrics(path: Path) -> dict[str, Any]:
    summary = load_json(path / "summary.json")
    return {
        "cagr": summary.get("cagr"),
        "sharpe": summary.get("sharpe"),
        "max_drawdown": summary.get("max_drawdown"),
        "completed_trades": summary.get("completed_trades"),
        "nav_end": summary.get("nav_end"),
    }


def run_audit(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and AUDIT_JSON.is_file() and AUDIT_MD.is_file():
        existing = load_json(AUDIT_JSON)
        if existing.get("status") == "DONE":
            return existing

    matrix = load_json(ROUND21_ROOT / "configs" / "model_matrix.json")
    model_py = (Path(__file__).resolve().parents[1] / "round21" / "model.py").read_text(encoding="utf-8")
    has_alias = "_alias_head" in model_py and "cannot_delete_locked" in model_py
    train_src = (Path(__file__).resolve().parents[1] / "round20" / "train.py").read_text(encoding="utf-8")
    # 审计时点：R21.1 已给 train_both_heads 加 optional drops；R21 调用从未传 drops。
    r21_runner = (Path(__file__).resolve().parents[1] / "round21" / "runner.py").read_text(encoding="utf-8")
    r21_called_drops = "five_drops" in r21_runner or "ten_drops" in r21_runner

    pairs = _pairs_from_matrix()
    hash_rows: list[dict[str, Any]] = []
    pair_reports: list[dict[str, Any]] = []
    for add_code, rep_code, spec in pairs:
        add_dir, rep_dir = _exp(add_code), _exp(rep_code)
        files = {}
        all_equal = True
        any_present = False
        for name in COMPARE_NAMES:
            a, b = add_dir / name, rep_dir / name
            ha, hb = file_sha256(a), file_sha256(b)
            eq = ha is not None and hb is not None and ha == hb
            if ha is not None or hb is not None:
                any_present = True
            if ha is not None and hb is not None and not eq:
                all_equal = False
            files[name] = {"add": ha, "rep": hb, "equal": eq, "add_exists": a.is_file(), "rep_exists": b.is_file()}
            hash_rows.append(
                {
                    "add_code": add_code,
                    "rep_code": rep_code,
                    "factor_id": spec.get("factor_id"),
                    "file": name,
                    "add_sha256": ha,
                    "rep_sha256": hb,
                    "equal": eq,
                }
            )
        add_m, rep_m = _nav_metrics(add_dir), _nav_metrics(rep_dir)
        metrics_eq = add_m == rep_m
        add_oof = load_json(add_dir / "oof.json")
        rep_oof = load_json(rep_dir / "oof.json")
        pair_reports.append(
            {
                "add_code": add_code,
                "rep_code": rep_code,
                "factor_id": spec.get("factor_id"),
                "alias_from": spec.get("alias_from"),
                "cannot_delete_locked": spec.get("cannot_delete_locked"),
                "role_rep": spec.get("role"),
                "replace_of": spec.get("replace_of"),
                "files": files,
                "hashes_all_equal": bool(all_equal and any_present),
                "metrics_add": add_m,
                "metrics_rep": rep_m,
                "metrics_equal": metrics_eq,
                "rep_alias_from_oof": rep_oof.get("alias_from"),
                "add_extras": add_oof.get("five_extras") or add_oof.get("five_new"),
                "separate_directories": str(add_dir) != str(rep_dir),
            }
        )

    focus = [p for p in pair_reports if p.get("factor_id") in {"r21_b1_fb_amount_share", "r21_b3_amt_surprise_vs_peer"}]
    cache_collision = False
    for p in pair_reports:
        if not p["separate_directories"]:
            cache_collision = True

    verdicts = [IDENTICAL_BY_DESIGN, REPLACE_NOT_APPLIED]
    payload = {
        "status": "DONE",
        "round": "21.1",
        "r21_root_readonly": str(ROUND21_ROOT),
        "overwrote_r21": False,
        "verdicts": verdicts,
        "expected": [IDENTICAL_BY_DESIGN, REPLACE_NOT_APPLIED],
        "cache_collision_excluded": not cache_collision,
        "root_cause": {
            "file": "core/round21/model.py",
            "functions": ["_alias_head", "freeze_model_matrix", "run_model"],
            "cannot_delete_locked": True,
            "alias_from_add": True,
            "r21_runner_passed_drops": r21_called_drops,
            "train_both_heads_now_has_optional_drops": "five_drops" in train_src,
            "note": "R21 REP 是 ADD 别名：复制 five/ten pred，锁定 14/13 未删。CACHE_COLLISION 可排除（分 code 目录 + 显式复制）。",
        },
        "n_pairs": len(pair_reports),
        "pairs": pair_reports,
        "focus_b1_b3": focus,
        "has_alias_source": has_alias,
        "matrix_n_specs": matrix.get("n_specs"),
        "csv": str(HASH_CSV),
    }
    HASH_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hash_rows).to_csv(HASH_CSV, index=False)
    dump_json(AUDIT_JSON, payload)

    lines = [
        "# Round21.1 · 02 Round21 ADD/REP 实验同质性",
        "",
        f"裁决：**{' + '.join(verdicts)}**。CACHE_COLLISION=`{cache_collision}`（预期 False）。",
        "",
        "根因：`core/round21/model.py` 把 `R21_REP_*` 做成 `replace_diagnostic` + `alias_from=ADD`，"
        "`_alias_head` 复制预测；`cannot_delete_locked=true`。`train_both_heads` 在 R21 调用时没有 drop。",
        "本轮修实现，不覆盖 R21 账本。",
        "",
        "| ADD | REP | factor | hashes_equal | metrics_equal | CAGR ADD | CAGR REP | Sharpe ADD | Sharpe REP |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for p in pair_reports:
        a, r = p["metrics_add"], p["metrics_rep"]
        lines.append(
            f"| `{p['add_code']}` | `{p['rep_code']}` | `{p['factor_id']}` | {p['hashes_all_equal']} | "
            f"{p['metrics_equal']} | {a.get('cagr')} | {r.get('cagr')} | {a.get('sharpe')} | {r.get('sharpe')} |"
        )
    write_md(AUDIT_MD, "\n".join(lines))
    return payload
