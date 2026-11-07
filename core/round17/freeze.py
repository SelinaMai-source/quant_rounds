"""冻结 5000 trial manifest + 选择/晋升公式。生成后禁止自适应加 trial。不得读 2022+ 收益。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND17_ROOT, assert_project_result_path, ensure_round17_tree
from ..round11.contract import source_tree_sha256
from ..round11.freeze import _dump_yaml
from ..stock_readonly import sha256_file
from .common import dump_json, load_json, sha256_json, write_md
from .contract import (
    BUDGET_BOTH,
    BUDGET_FIVE_ONLY,
    BUDGET_TEN_ONLY,
    CONTRACT_BAGGING,
    CONTRACT_DETERMINISTIC,
    FORBIDDEN,
    MANIFEST_NAME,
    MANIFEST_SEED,
    MAXDD_WORSEN_CAP,
    N_TRIALS,
    PROMOTION_GATES,
    PROTOCOL_SOURCE_FILES,
    SELECTION_WEIGHTS,
    SIZE_MIX,
    STAGES,
)
from .factors import FACTOR_SPECS
from .pit import PIT_JSON, valid_factor_names

FREEZE_MD = ROUND17_ROOT / "05_search_manifest.md"
FREEZE_JSON = ROUND17_ROOT / "configs" / "freeze.json"
REG_CSV = ROUND17_ROOT / "search" / "search_registry.csv"
BUDGET_MD = ROUND17_ROOT / "04_compute_budget.md"


def manifest_path():
    return ROUND17_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND17_ROOT / f"{MANIFEST_NAME}.sha256"


def _family(name: str) -> str:
    for spec in FACTOR_SPECS:
        if spec["name"] == name:
            return spec["family"]
    return "unknown"


def _hash_trial(payload: dict[str, Any]) -> str:
    key = {
        "sleeve": payload["sleeve"],
        "extras": list(payload["extras"]),
        "five_extras": list(payload["five_extras"]),
        "ten_extras": list(payload["ten_extras"]),
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def _size_counts(n: int, n_factors: int) -> dict[int, int]:
    raw = {k: int(round(n * v)) for k, v in SIZE_MIX.items()}
    cap = {1: n_factors, 2: n_factors * (n_factors - 1) // 2, 3: n_factors * (n_factors - 1) * (n_factors - 2) // 6, 4: 10**9}
    for k in (1, 2, 3, 4):
        raw[k] = min(raw[k], cap[k])
    delta = n - sum(raw.values())
    for k in (4, 3, 2, 1):
        room = cap[k] - raw[k]
        take = min(max(delta, 0), room)
        raw[k] += take
        delta -= take
    if delta < 0:
        for k in (4, 3, 2, 1):
            drop = min(-delta, raw[k])
            raw[k] -= drop
            delta += drop
    return raw


def _sample_combos(names: list[str], k: int, n: int, rng: np.random.Generator, used: set[tuple[str, ...]]) -> list[tuple[str, ...]]:
    fam = {name: _family(name) for name in names}
    pool = []
    if k <= 2 or (k == 3 and len(names) <= 24):
        for combo in combinations(names, k):
            key = tuple(sorted(combo))
            if key not in used:
                pool.append(key)
        rng.shuffle(pool)
        # prefer cross-family
        pool.sort(key=lambda c: -len({fam[x] for x in c}))
        out = []
        for key in pool:
            if len(out) >= n:
                break
            out.append(key)
            used.add(key)
        return out
    out = []
    guard = 0
    while len(out) < n and guard < n * 80:
        guard += 1
        choice = tuple(sorted(rng.choice(names, size=k, replace=False).tolist()))
        if choice in used:
            continue
        used.add(choice)
        out.append(choice)
    return out


def build_trials(valid: list[str]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(MANIFEST_SEED)
    names = list(valid)
    if len(names) < 1:
        raise RuntimeError("no PIT-valid new factors")
    used: set[tuple[str, ...]] = set()
    trials: list[dict[str, Any]] = []
    seq = 1

    def add_block(sleeve: str, n: int) -> None:
        nonlocal seq
        counts = _size_counts(n, len(names))
        block_used: set[tuple[str, ...]] = set()
        for k, ck in counts.items():
            combos = _sample_combos(names, k, ck, rng, block_used)
            for extras in combos:
                if sleeve == "TEN_ONLY":
                    five_ex, ten_ex = (), extras
                elif sleeve == "FIVE_ONLY":
                    five_ex, ten_ex = extras, ()
                else:
                    five_ex, ten_ex = extras, extras
                rec = {
                    "trial_id": f"R17_T{seq:04d}",
                    "sleeve": sleeve,
                    "extras": list(extras),
                    "five_extras": list(five_ex),
                    "ten_extras": list(ten_ex),
                    "n_new": len(extras),
                    "families": sorted({_family(x) for x in extras}),
                }
                rec["feature_set_hash"] = _hash_trial(rec)
                trials.append(rec)
                seq += 1
                if len([t for t in trials if t["sleeve"] == sleeve]) >= n:
                    break

    add_block("TEN_ONLY", BUDGET_TEN_ONLY)
    add_block("FIVE_ONLY", BUDGET_FIVE_ONLY)
    add_block("BOTH", BUDGET_BOTH)
    if len(trials) != N_TRIALS:
        # pad with additional size-4 BOTH/TEN unique hashes if short
        k = 4
        guard = 0
        block_used = {tuple(t["extras"]) for t in trials}
        while len(trials) < N_TRIALS and guard < 20000:
            guard += 1
            extras = tuple(sorted(rng.choice(names, size=min(k, len(names)), replace=False).tolist()))
            if extras in block_used:
                continue
            block_used.add(extras)
            sleeve = "TEN_ONLY" if len([t for t in trials if t["sleeve"] == "TEN_ONLY"]) < BUDGET_TEN_ONLY else (
                "FIVE_ONLY" if len([t for t in trials if t["sleeve"] == "FIVE_ONLY"]) < BUDGET_FIVE_ONLY else "BOTH"
            )
            five_ex = extras if sleeve in {"FIVE_ONLY", "BOTH"} else ()
            ten_ex = extras if sleeve in {"TEN_ONLY", "BOTH"} else ()
            rec = {
                "trial_id": f"R17_T{len(trials)+1:04d}",
                "sleeve": sleeve,
                "extras": list(extras),
                "five_extras": list(five_ex),
                "ten_extras": list(ten_ex),
                "n_new": len(extras),
                "families": sorted({_family(x) for x in extras}),
            }
            rec["feature_set_hash"] = _hash_trial(rec)
            trials.append(rec)
        for i, rec in enumerate(trials, start=1):
            rec["trial_id"] = f"R17_T{i:04d}"
    hashes = [t["feature_set_hash"] for t in trials]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("duplicate feature_set_hash in manifest")
    if len(trials) != N_TRIALS:
        raise RuntimeError(f"expected {N_TRIALS} trials, got {len(trials)}")
    return trials


def freeze_manifest(*, overwrite: bool = False, **_: Any) -> dict[str, Any]:
    ensure_round17_tree()
    path = manifest_path()
    if path.is_file() and not overwrite:
        existing = load_json(FREEZE_JSON)
        if existing.get("frozen"):
            return existing
        # yaml already there
        digest = sha256_file(path)
        payload = load_json(FREEZE_JSON) or {"frozen": True, "sha256": digest, "path": str(path)}
        if payload.get("frozen"):
            return payload
    valid = valid_factor_names()
    trials = build_trials(valid)
    body = {
        "round": 17,
        "name": MANIFEST_NAME,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "isolated_contract": CONTRACT_BAGGING,
        "deterministic_contract": CONTRACT_DETERMINISTIC,
        "not_production_default": True,
        "r6_a04_warning": "bagging-only CAGR lift from randomness is not a promotion reason",
        "n_trials": N_TRIALS,
        "budgets": {"TEN_ONLY": BUDGET_TEN_ONLY, "FIVE_ONLY": BUDGET_FIVE_ONLY, "BOTH": BUDGET_BOTH},
        "manifest_seed": MANIFEST_SEED,
        "valid_new_factors": valid,
        "discovery_window": ["2018-01-01", "2021-12-31"],
        "discovery_ranking_reads_2022plus": False,
        "selection_score": SELECTION_WEIGHTS,
        "selection_score_formula": "0.40*rankpct(dev_sharpe)+0.25*rankpct(dev_median_year)+0.20*rankpct(dev_worst_year)+0.15*rankpct(dev_OOS_NDCG)",
        "tie_break": "trial_id ascending",
        "maxdd_worsen_cap_vs_a00_same_window": MAXDD_WORSEN_CAP,
        "promotion_gates": PROMOTION_GATES,
        "forbidden": list(FORBIDDEN),
        "protocol_tree": source_tree_sha256(PROTOCOL_SOURCE_FILES),
        "stages": list(STAGES),
        "hyperparams_locked_except_sampling": True,
        "xgb_search_n_jobs": 1,
        "trials": trials,
    }
    text = _dump_yaml(body)
    assert_project_result_path(path)
    path.write_text(text, encoding="utf-8")
    digest = sha256_file(path)
    sha256_path().write_text(digest + "\n", encoding="utf-8")
    frame = pd.DataFrame([{k: v for k, v in t.items() if k != "families"} | {"families": "|".join(t["families"])} for t in trials])
    frame["status"] = "PLANNED"
    REG_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REG_CSV, index=False)
    payload = {
        "frozen": True,
        "sha256": digest,
        "path": str(path),
        "n_trials": len(trials),
        "n_valid_factors": len(valid),
        "budgets": body["budgets"],
        "unique_hashes": len(set(hashes for hashes in [t["feature_set_hash"] for t in trials])),
        "selection_formula": body["selection_score_formula"],
        "promotion_gates": PROMOTION_GATES,
    }
    dump_json(FREEZE_JSON, payload)
    lines = [
        "# 05 冻结 5000 trial manifest",
        "",
        f"sha256=`{digest}` n=`{len(trials)}` valid_factors=`{len(valid)}`",
        "TEN_ONLY=3500 FIVE_ONLY=1000 BOTH=500。发现窗仅 2018–2021 用于排序。冻结后禁止自适应加 trial。",
        "",
        f"selection_score = {body['selection_score_formula']}",
        "并列 trial_id 升序。MaxDD 相对同窗 A00 恶化上限预注册。",
        "",
        "本文件冻结时尚未读取 2022+ 发现收益。",
    ]
    write_md(FREEZE_MD, "\n".join(lines))
    write_budget()
    return payload


def write_budget() -> dict[str, Any]:
    import os

    cpu = os.cpu_count() or 1
    fits = (BUDGET_TEN_ONLY + BUDGET_FIVE_ONLY + BUDGET_BOTH * 2) * 5 * 4
    payload = {
        "cpu": cpu,
        "search_xgb_n_jobs": 1,
        "suggested_jobs": max(1, min(32, cpu - 2)),
        "approx_small_tree_fits_discovery": fits,
        "note": "jobs × n_jobs 不得超过核数。checkpoint 每 trial。不得把 5000 静默缩成 500。",
        "disk": "trial metrics+fused preds+nav; 全量模型搜索期不落盘",
    }
    dump_json(ROUND17_ROOT / "configs" / "compute_budget.json", payload)
    write_md(
        BUDGET_MD,
        "\n".join([
            "# 04 算力预算",
            "",
            f"CPU=`{cpu}` suggested_jobs=`{payload['suggested_jobs']}` xgb_n_jobs=1",
            f"discovery 约 `{fits}` 次小树拟合（3500+1000+500×2 头 ×5 seed ×4 折）。",
            "checkpoint 每 trial、atomic write、`--resume`。jobs×n_jobs 不超过核数。",
        ]),
    )
    return payload


def require_frozen() -> dict[str, Any]:
    payload = load_json(FREEZE_JSON)
    if not payload.get("frozen"):
        raise RuntimeError("manifest not frozen")
    return payload


def load_trials() -> list[dict[str, Any]]:
    require_frozen()
    import yaml

    text = manifest_path().read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except Exception:
        data = None
    if not isinstance(data, dict) or not data.get("trials"):
        # fallback csv
        frame = pd.read_csv(REG_CSV)
        recs = frame.to_dict(orient="records")
        for rec in recs:
            rec["extras"] = _parse_list(rec.get("extras"))
            rec["five_extras"] = _parse_list(rec.get("five_extras"))
            rec["ten_extras"] = _parse_list(rec.get("ten_extras"))
        return recs
    return list(data["trials"])


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            import ast
            return [str(x) for x in ast.literal_eval(s)]
        if not s:
            return []
        return [p for p in s.split("|") if p]
    return []
