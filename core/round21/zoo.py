"""Persist research/factor_zoo from Round21 真实产物。禁止编造。禁止复制全局结果冒充单因子。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import PROJECT_ROOT, ROUND21_ROOT
from .common import dump_json, load_json, write_md
from .contract import FACTOR_ZOO_ROOT, FACTORY_STEPS, KEEP_BASELINE_V5
from .evaluate import evaluate_factor
from .freeze import require_frozen
from .hypotheses import HYPOTHESES

ZOO_JSON = ROUND21_ROOT / "zoo" / "zoo.json"
ZOO_MD = ROUND21_ROOT / "13_factor_zoo.md"
FACTORS_DIR = ROUND21_ROOT / "factors"


def _assert_zoo_path(path: Path) -> Path:
    resolved = path.resolve()
    root = FACTOR_ZOO_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside {root}: {resolved}")
    if PROJECT_ROOT.resolve() not in resolved.parents and resolved != PROJECT_ROOT.resolve():
        raise RuntimeError(f"refusing to write outside project: {resolved}")
    return resolved


def _write_text(path: Path, text: str) -> Path:
    dest = _assert_zoo_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    body = text if text.endswith("\n") else text + "\n"
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(dest)
    return dest


def _write_json(path: Path, payload: Any) -> Path:
    return _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _write_table(path: Path, frame: pd.DataFrame) -> Path:
    dest = _assert_zoo_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix == ".parquet":
        frame.to_parquet(dest, index=False)
    else:
        frame.to_csv(dest, index=False)
    return dest


def _rankic_e2(pkg: dict[str, Any]) -> float | None:
    if pkg.get("raw_rankic_e2") is not None:
        try:
            return float(pkg["raw_rankic_e2"])
        except (TypeError, ValueError):
            return None
    ic = ((pkg.get("ic") or {}).get("label_e2_net") or {}).get("rankic") or {}
    try:
        return float(ic["mean"])
    except (TypeError, ValueError, KeyError):
        return None


def _gather_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from .backtest import BT_JSON
    from .combo import REG_CSV, REG_JSON
    from .model import MODEL_JSON
    from .pit import PIT_JSON
    from .redundancy import EDGE_CSV, KEEP_CSV

    pit = load_json(PIT_JSON)
    keep = pd.read_csv(KEEP_CSV) if KEEP_CSV.is_file() else pd.DataFrame()
    keep_map = {str(r["name"]): r for r in keep.to_dict(orient="records")} if not keep.empty else {}
    bt = load_json(BT_JSON)
    main_rows = list(bt.get("main_rows") or [])
    model = load_json(MODEL_JSON)
    combo = load_json(REG_JSON)
    registry = pd.read_csv(REG_CSV) if REG_CSV.is_file() else pd.DataFrame()

    registry_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for card in HYPOTHESES:
        fid = str(card["factor_id"])
        dest = FACTORS_DIR / fid
        pkg = load_json(dest / "package.json")
        pit_one = load_json(dest / "pit.json")
        red_one = load_json(dest / "redundancy.json")
        kd = keep_map.get(fid) or {}
        decision = (red_one.get("decision") or {}) if red_one else {}
        action = str(kd.get("action") or decision.get("action") or "")
        pit_status = str(
            pit_one.get("pit_status")
            or kd.get("pit_status")
            or pkg.get("pit_status")
            or ""
        )
        gate = evaluate_factor(fid, through="PromotionGate", persist=True)
        promo = gate.get("PromotionGate") or {}
        zoo_status = str(promo.get("status") or "SKIPPED")
        if zoo_status not in {"PASS", "RECOMMEND_NEW_ALPHA_CANDIDATE"}:
            zoo_status = zoo_status if zoo_status in {"FAIL", "SKIPPED", "BLOCKED_PIT"} else "REJECTED"
        row = {
            "factor_id": fid,
            "family": card.get("family"),
            "short": card.get("short"),
            "formula": card.get("formula"),
            "round": 21,
            "values_constructed": (dest / "values.parquet").is_file(),
            "pit_status": pit_status,
            "model_eligible": pit_one.get("model_eligible") or kd.get("model_eligible"),
            "redundancy_action": action,
            "cluster": kd.get("cluster") or decision.get("cluster"),
            "best_abs_icir": kd.get("best_abs_icir") or decision.get("best_abs_icir"),
            "best_abs_mean": kd.get("best_abs_mean") or decision.get("best_abs_mean"),
            "raw_rankic_e2": _rankic_e2(pkg),
            "zyquant_nearest": kd.get("zyquant_nearest") or decision.get("zyquant_nearest"),
            "zyquant_abs_rho": kd.get("zyquant_abs_rho") or decision.get("zyquant_abs_rho"),
            "near_neighbors": "|".join(card.get("near_neighbors") or []),
            "promotion_status": promo.get("status"),
            "promotion_reason": promo.get("reason"),
            "promotion_passed": bool(promo.get("passed")),
            "zoo_status": zoo_status,
            "copied_global_result": False,
            "package_dir": str(dest),
        }
        registry_rows.append(row)
        for nb in card.get("near_neighbors") or []:
            lineage_rows.append(
                {
                    "factor_id": fid,
                    "parent_or_neighbor": nb,
                    "relation": "pre_registered_near_neighbor",
                    "round": 21,
                }
            )
        if kd.get("cluster"):
            lineage_rows.append(
                {
                    "factor_id": fid,
                    "parent_or_neighbor": str(kd.get("cluster")),
                    "relation": "spearman_cluster",
                    "round": 21,
                }
            )
        evidence_rows.append(
            {
                "factor_id": fid,
                "family": card.get("family"),
                "pit_status": pit_status,
                "raw_rankic_e2": _rankic_e2(pkg),
                "rolling_w60_rankic_e2": pkg.get("rolling_w60_rankic_e2"),
                "top_tail_e2": pkg.get("top_tail_e2"),
                "github_exact_status": pkg.get("github_exact_status"),
                "github_exact_namespace": pkg.get("github_exact_namespace") or "GITHUB_EXACT",
                "rolling_namespace": pkg.get("rolling_namespace") or "ROUND21_INTERNAL",
                "has_ic": (dest / "ic.json").is_file(),
                "has_rolling": (dest / "rolling.json").is_file(),
                "has_shape": (dest / "shape.json").is_file(),
                "has_top_tail": (dest / "top_tail.json").is_file(),
                "has_incrementality": (dest / "incrementality.json").is_file(),
                "has_portfolio": (dest / "portfolio.json").is_file(),
                "promotion_status": promo.get("status"),
                "promotion_reason": promo.get("reason"),
            }
        )

    similarity_rows: list[dict[str, Any]] = []
    if EDGE_CSV.is_file():
        edges = pd.read_csv(EDGE_CSV)
        for rec in edges.to_dict(orient="records"):
            similarity_rows.append(
                {
                    "a": rec.get("a"),
                    "b": rec.get("b"),
                    "rho": rec.get("rho"),
                    "n_days": rec.get("n_days"),
                    "source": "cluster_edges_spearman",
                    "threshold": 0.85,
                }
            )
    for row in registry_rows:
        if row.get("zyquant_nearest"):
            similarity_rows.append(
                {
                    "a": row["factor_id"],
                    "b": row.get("zyquant_nearest"),
                    "rho": row.get("zyquant_abs_rho"),
                    "n_days": None,
                    "source": "zyquant350_novelty",
                    "threshold": 0.85,
                }
            )

    rejected = [
        {
            "factor_id": r["factor_id"],
            "family": r["family"],
            "action": r["redundancy_action"],
            "pit_status": r["pit_status"],
            "reason": r.get("promotion_reason") or r["redundancy_action"],
            "round": 21,
            "archive": "REJECTED",
        }
        for r in registry_rows
        if str(r.get("redundancy_action") or "").startswith("DROP")
        or str(r.get("pit_status")) in {"BLOCKED_PIT", "PIT_FAIL_COVERAGE", "BLOCKED_DATA"}
    ]

    trial_rows: list[dict[str, Any]] = []
    if not registry.empty:
        for rec in registry.to_dict(orient="records"):
            trial_rows.append(
                {
                    "trial_id": rec.get("code"),
                    "role": rec.get("role"),
                    "extras": rec.get("extras"),
                    "source": "combo_registry_frozen_before_nav",
                    "frozen": True,
                    "independent_oos": False,
                }
            )
    for rec in model.get("rows") or model.get("models") or []:
        if isinstance(rec, dict) and rec.get("code"):
            trial_rows.append(
                {
                    "trial_id": rec.get("code"),
                    "role": rec.get("role"),
                    "extras": rec.get("factor_id") or rec.get("extras"),
                    "source": "walk_forward_oof",
                    "frozen": True,
                    "independent_oos": False,
                    "increment_pass": rec.get("increment_pass")
                    if "increment_pass" in rec
                    else (rec.get("code") in set(model.get("increment_pass_codes") or [])),
                }
            )
    if not any(t.get("source") == "walk_forward_oof" for t in trial_rows):
        for code in model.get("increment_pass_codes") or []:
            trial_rows.append(
                {
                    "trial_id": code,
                    "role": "oof_increment_pass",
                    "extras": None,
                    "source": "walk_forward_oof",
                    "frozen": True,
                    "independent_oos": False,
                    "increment_pass": True,
                }
            )
    for rec in main_rows:
        trial_rows.append(
            {
                "trial_id": rec.get("code"),
                "role": rec.get("table") or "MAIN_ROUND7_FLAT_5BP",
                "extras": None,
                "source": "full_engine_backtest",
                "frozen": True,
                "independent_oos": False,
                "sharpe": rec.get("sharpe"),
                "cagr": rec.get("cagr"),
            }
        )

    families = []
    for card in HYPOTHESES:
        families.append(
            {
                "family": card.get("family"),
                "factor_id": card.get("factor_id"),
                "short": card.get("short"),
                "formula": card.get("formula"),
            }
        )

    extra = {
        "pit_pass": list(pit.get("pit_pass") or []),
        "blocked_pit": list(pit.get("blocked_pit") or []),
        "combo_selected": list(combo.get("selected_new_factors") or []),
        "combo_sha256": combo.get("sha256"),
        "n_combo_registry": combo.get("n_registry"),
    }
    return registry_rows, lineage_rows, evidence_rows, similarity_rows, rejected, trial_rows, families, extra


def persist_zoo() -> dict[str, Any]:
    FACTOR_ZOO_ROOT.mkdir(parents=True, exist_ok=True)
    (
        registry_rows,
        lineage_rows,
        evidence_rows,
        similarity_rows,
        rejected,
        trial_rows,
        families,
        extra,
    ) = _gather_rows()
    registry = pd.DataFrame(registry_rows)
    lineage = pd.DataFrame(lineage_rows)
    evidence = pd.DataFrame(evidence_rows)
    similarity = pd.DataFrame(similarity_rows)
    rejected_df = pd.DataFrame(rejected)
    trials = pd.DataFrame(trial_rows)
    families_df = pd.DataFrame(families)

    files = {
        "registry_parquet": str(_write_table(FACTOR_ZOO_ROOT / "registry.parquet", registry)),
        "registry_csv": str(_write_table(FACTOR_ZOO_ROOT / "registry.csv", registry)),
        "families_csv": str(_write_table(FACTOR_ZOO_ROOT / "families.csv", families_df)),
        "lineage_parquet": str(_write_table(FACTOR_ZOO_ROOT / "lineage.parquet", lineage)),
        "lineage_csv": str(_write_table(FACTOR_ZOO_ROOT / "lineage.csv", lineage)),
        "similarity_csv": str(_write_table(FACTOR_ZOO_ROOT / "similarity.csv", similarity)),
        "evidence_parquet": str(_write_table(FACTOR_ZOO_ROOT / "evidence.parquet", evidence)),
        "evidence_csv": str(_write_table(FACTOR_ZOO_ROOT / "evidence.csv", evidence)),
        "rejected_archive_csv": str(_write_table(FACTOR_ZOO_ROOT / "rejected_archive.csv", rejected_df)),
        "trial_registry_csv": str(_write_table(FACTOR_ZOO_ROOT / "trial_registry.csv", trials)),
    }
    yaml_lines = [
        "# Round21 Factor Zoo（由真实产物落盘，不是假设桩）",
        "round: 21",
        "values_constructed: true",
        f"n_factors: {len(registry_rows)}",
        f"n_rejected: {len(rejected)}",
        f"n_trials: {len(trial_rows)}",
        f"decision: {KEEP_BASELINE_V5}",
        "auto_promote: false",
        "factory_steps:",
    ]
    for step in FACTORY_STEPS:
        yaml_lines.append(f"  - {step}")
    yaml_lines.append("factors:")
    for row in registry_rows:
        yaml_lines += [
            f"- factor_id: {row['factor_id']}",
            f"  family: {row['family']}",
            f"  pit_status: {row['pit_status']}",
            f"  redundancy_action: {row['redundancy_action']}",
            f"  zoo_status: {row['zoo_status']}",
            f"  promotion_status: {row['promotion_status']}",
        ]
    _write_text(FACTOR_ZOO_ROOT / "registry.yaml", "\n".join(yaml_lines))
    files["registry_yaml"] = str(FACTOR_ZOO_ROOT / "registry.yaml")
    _write_json(
        FACTOR_ZOO_ROOT / "zoo_index.json",
        {
            "round": 21,
            "root": str(FACTOR_ZOO_ROOT),
            "files": files,
            "n_registry": len(registry_rows),
            "n_rejected": len(rejected),
            "n_trials": len(trial_rows),
            "decision": KEEP_BASELINE_V5,
            **extra,
        },
    )
    files["zoo_index"] = str(FACTOR_ZOO_ROOT / "zoo_index.json")
    return {
        "status": "DONE",
        "root": str(FACTOR_ZOO_ROOT),
        "files": files,
        "n_registry": len(registry_rows),
        "n_rejected": len(rejected),
        "n_trials": len(trial_rows),
        "n_lineage": len(lineage_rows),
        "n_similarity": len(similarity_rows),
        "n_evidence": len(evidence_rows),
        "decision": KEEP_BASELINE_V5,
        "auto_promote": False,
        **extra,
    }


def run_zoo(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and ZOO_JSON.is_file() and (FACTOR_ZOO_ROOT / "registry.parquet").is_file():
        existing = load_json(ZOO_JSON)
        if existing.get("status") == "DONE":
            return existing
    payload = persist_zoo()
    dump_json(ZOO_JSON, payload)
    write_md(
        ZOO_MD,
        "# 13 Factor Zoo\n\n"
        f"root=`{FACTOR_ZOO_ROOT}` n_registry=`{payload['n_registry']}` "
        f"rejected=`{payload['n_rejected']}` trials=`{payload['n_trials']}`\n"
        f"决策=`{KEEP_BASELINE_V5}` AUTO_PROMOTE=False。PromotionGate 未过门写 SKIPPED/FAIL+原因，不伪造 PASS。\n",
    )
    return payload


__all__ = ["FACTOR_ZOO_ROOT", "ZOO_JSON", "persist_zoo", "run_zoo"]
