"""四路真实 XGB WF。five_drops/ten_drops 真删。禁止 alias ADD 为 Replace。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round17.ensemble import ensemble_rank_pct
from ..round17.train import complete_cases, ranker_config_from_params, rolling_train_predict_open
from ..round20.panel import fold_masks
from ..round20.train import new_columns, train_both_heads, unique_keep
from ..round21.runner import oof_rankic
from .common import dump_json, feature_set_hash, file_sha256, load_json, sha256_json, write_md
from .contract import (
    CLUSTER_SPEARMAN_ABS,
    COLSAMPLE,
    E0_CODE,
    FIVE_ONLY_FEATURE,
    INCREMENT_GATE,
    KEYS,
    PREDICT_YEARS,
    SHARED_13,
    SUBSAMPLE,
    TRAIN_MD,
    WF_MD,
    XGB_N_JOBS,
    exp_dir,
    model_parameters,
)
from .panels import _G, init_heads, pit_safe_factors, prepare_head_panels

from ..paths import ROUND21_1_ROOT

MODEL_JSON = ROUND21_1_ROOT / "model" / "model_selection.json"
MATRIX_JSON = ROUND21_1_ROOT / "configs" / "model_matrix.json"
IDENT_JSON = ROUND21_1_ROOT / "model" / "identifiability.json"
OOF_CSV = ROUND21_1_ROOT / "model" / "oof_increment.csv"


def _fold_map() -> dict[str, Any]:
    return load_json(ROUND21_1_ROOT / "correlation" / "fold_nearest.json")


def _majority_and_folds(feat: str) -> tuple[str | None, dict[str, str], bool, list[str]]:
    nearest = _fold_map().get(feat) or {}
    fold = nearest.get("fold_nearest") or {}
    mapping = {}
    clusters: list[str] = []
    from .correlate import FOLD_CSV

    if FOLD_CSV.is_file():
        frame = pd.read_csv(FOLD_CSV)
        sub = frame.loc[frame["new_factor"].astype(str).eq(feat)]
        for _, row in sub.iterrows():
            y = str(int(row["predict_year"]))
            j = row.get("nearest_legacy")
            if pd.isna(j) or str(j) in {"nan", "None", ""}:
                continue
            mapping[y] = str(j)
            raw = str(row.get("cluster") or "")
            for p in raw.split("|"):
                if p and p not in clusters and p != mapping[y]:
                    clusters.append(p)
    if not mapping:
        for y, rec in fold.items():
            j = (rec or {}).get("nearest_legacy")
            if j:
                mapping[str(y)] = str(j)
    changed = len(set(mapping.values())) > 1
    majority = nearest.get("majority_legacy")
    if not majority and mapping:
        counts: dict[str, int] = {}
        for v in mapping.values():
            counts[v] = counts.get(v, 0) + 1
        majority = max(counts, key=counts.get)
    return majority, mapping, changed, clusters


def _head_rule(legacy: str | None) -> str:
    if legacy == FIVE_ONLY_FEATURE:
        return "five_only"
    if legacy in SHARED_13:
        return "shared_13"
    return "unknown"


def freeze_matrix(*, resume: bool = True) -> dict[str, Any]:
    if resume and MATRIX_JSON.is_file():
        existing = load_json(MATRIX_JSON)
        if existing.get("status") == "FROZEN":
            return existing
    factors = pit_safe_factors(resume=True)
    specs: list[dict[str, Any]] = [
        {
            "code": E0_CODE,
            "role": "E0_CONTROL",
            "factor_id": None,
            "five_extras": (),
            "ten_extras": (),
            "five_drops": (),
            "ten_drops": (),
            "force_retrain": True,
            "formal": True,
            "head": "both_90_10",
            "alias_from": None,
        }
    ]
    for feat in factors:
        majority, mapping, changed, cluster_old = _majority_and_folds(feat)
        rule = _head_rule(majority)
        fold_five = {y: ([j] if j in FIVE_FEATURES else []) for y, j in mapping.items()}
        fold_ten = {y: ([j] if j in TEN_FEATURES else []) for y, j in mapping.items()}
        specs.append(
            {
                "code": f"R21_1_E1_{feat}",
                "role": "E1_REMOVE",
                "factor_id": feat,
                "five_extras": (),
                "ten_extras": (),
                "five_drops": (majority,) if majority in FIVE_FEATURES else (),
                "ten_drops": (majority,) if majority in TEN_FEATURES else (),
                "fold_five_drops": fold_five if changed else None,
                "fold_ten_drops": fold_ten if changed else None,
                "per_fold_drops": changed,
                "nearest_legacy": majority,
                "force_retrain": True,
                "formal": True,
                "head": "both_90_10" if rule != "five_only" else "five_only",
                "alias_from": None,
            }
        )
        specs.append(
            {
                "code": f"R21_1_E2_{feat}",
                "role": "E2_ADD",
                "factor_id": feat,
                "five_extras": (feat,),
                "ten_extras": (feat,),
                "five_drops": (),
                "ten_drops": (),
                "force_retrain": True,
                "formal": True,
                "head": "both_90_10",
                "nearest_legacy": majority,
                "alias_from": None,
            }
        )
        # E3 Both-head（主）
        specs.append(
            {
                "code": f"R21_1_E3_{feat}",
                "role": "E3_REPLACE",
                "factor_id": feat,
                "five_extras": (feat,),
                "ten_extras": (feat,) if rule != "five_only" else (),
                "five_drops": (majority,) if majority in FIVE_FEATURES else (),
                "ten_drops": (majority,) if majority in TEN_FEATURES and rule != "five_only" else (),
                "fold_five_drops": fold_five if changed else None,
                "fold_ten_drops": fold_ten if changed and rule != "five_only" else None,
                "per_fold_drops": changed,
                "nearest_legacy": majority,
                "force_retrain": True,
                "formal": True,
                "head": "both_90_10",
                "head_variant": "both",
                "alias_from": None,
            }
        )
        if rule == "shared_13":
            specs.append(
                {
                    "code": f"R21_1_E3F_{feat}",
                    "role": "E3_REPLACE_FIVE_ONLY",
                    "factor_id": feat,
                    "five_extras": (feat,),
                    "ten_extras": (),
                    "five_drops": (majority,) if majority in FIVE_FEATURES else (),
                    "ten_drops": (),
                    "force_retrain": True,
                    "formal": False,
                    "head": "five_only",
                    "head_variant": "five_only",
                    "nearest_legacy": majority,
                    "q8": True,
                    "alias_from": None,
                }
            )
            specs.append(
                {
                    "code": f"R21_1_E3T_{feat}",
                    "role": "E3_REPLACE_TEN_ONLY",
                    "factor_id": feat,
                    "five_extras": (),
                    "ten_extras": (feat,),
                    "five_drops": (),
                    "ten_drops": (majority,) if majority in TEN_FEATURES else (),
                    "force_retrain": True,
                    "formal": False,
                    "head": "ten_only",
                    "head_variant": "ten_only",
                    "nearest_legacy": majority,
                    "q8": True,
                    "alias_from": None,
                }
            )
        cluster_members = [majority] + [c for c in cluster_old if c != majority] if majority else list(cluster_old)
        cluster_members = [c for c in unique_keep(cluster_members) if c in FIVE_FEATURES or c in TEN_FEATURES]
        trigger = len(cluster_members) >= 2
        # 仅当训练窗 |ρ|>=0.85 且可辨识冗余簇
        if trigger:
            specs.append(
                {
                    "code": f"R21_1_E4_{feat}",
                    "role": "E4_CLUSTER_REMOVE",
                    "factor_id": feat,
                    "five_extras": (),
                    "ten_extras": (),
                    "five_drops": tuple(c for c in cluster_members if c in FIVE_FEATURES),
                    "ten_drops": tuple(c for c in cluster_members if c in TEN_FEATURES),
                    "force_retrain": True,
                    "formal": True,
                    "head": "both_90_10",
                    "cluster": cluster_members,
                    "alias_from": None,
                }
            )
            specs.append(
                {
                    "code": f"R21_1_E5_{feat}",
                    "role": "E5_CLUSTER_REPLACE",
                    "factor_id": feat,
                    "five_extras": (feat,),
                    "ten_extras": (feat,),
                    "five_drops": tuple(c for c in cluster_members if c in FIVE_FEATURES),
                    "ten_drops": tuple(c for c in cluster_members if c in TEN_FEATURES),
                    "force_retrain": True,
                    "formal": True,
                    "head": "both_90_10",
                    "cluster": cluster_members,
                    "alias_from": None,
                }
            )
    payload = {
        "status": "FROZEN",
        "n_specs": len(specs),
        "specs": specs,
        "factors": factors,
        "no_alias_add_as_replace": True,
        "cannot_delete_locked_false": True,
        "fusion": "0.90 ten + 0.10 five",
    }
    dump_json(MATRIX_JSON, payload)
    return payload


def _identity(spec: dict[str, Any], five_feat: Sequence[str], ten_feat: Sequence[str]) -> dict[str, Any]:
    body = {
        "code": spec["code"],
        "role": spec.get("role"),
        "added_five": list(spec.get("five_extras") or ()),
        "added_ten": list(spec.get("ten_extras") or ()),
        "dropped_five": list(spec.get("five_drops") or ()),
        "dropped_ten": list(spec.get("ten_drops") or ()),
        "fold_five_drops": spec.get("fold_five_drops"),
        "fold_ten_drops": spec.get("fold_ten_drops"),
        "head": spec.get("head"),
        "five_features": list(five_feat),
        "ten_features": list(ten_feat),
        "five_feature_set_hash": feature_set_hash(list(five_feat)),
        "ten_feature_set_hash": feature_set_hash(list(ten_feat)),
    }
    body["cache_key"] = sha256_json(body)
    return body


def _write_ident(dest, spec, five_feat, ten_feat, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    ident = _identity(spec, five_feat, ten_feat)
    if extra:
        ident.update(extra)
    dump_json(dest / "feature_list.json", {"five": list(five_feat), "ten": list(ten_feat)})
    dump_json(dest / "dropped_features.json", {"five": list(spec.get("five_drops") or ()), "ten": list(spec.get("ten_drops") or ())})
    dump_json(dest / "added_features.json", {"five": list(spec.get("five_extras") or ()), "ten": list(spec.get("ten_extras") or ())})
    dump_json(dest / "cache_key.json", ident)
    dump_json(dest / "model_config.json", model_parameters(17))
    return ident


def _top12(pred: pd.DataFrame) -> pd.DataFrame:
    work = pred.copy()
    if "signal_date" in work.columns and "trade_date" not in work.columns:
        work["trade_date"] = pd.to_datetime(work["signal_date"])
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["rank"] = work.groupby("trade_date")["score"].rank(ascending=False, method="first")
    top = work.loc[work["rank"].le(2), ["trade_date", "instrument_id", "score", "rank"]].copy()
    return top.sort_values(["trade_date", "rank", "instrument_id"])


def _done(dest, ident: dict[str, Any]) -> dict[str, Any] | None:
    path = dest / "oof.json"
    key_path = dest / "cache_key.json"
    if not path.is_file() or not (dest / "five_pred.parquet").is_file() or not (dest / "ten_pred.parquet").is_file():
        return None
    if key_path.is_file():
        old = load_json(key_path)
        if old.get("cache_key") != ident.get("cache_key"):
            return None
    payload = load_json(path)
    if payload.get("status") in {"OOF", "OOF_SKIP_EXACT_REPLAY"} or payload.get("five_oof"):
        payload["resumed"] = True
        return payload
    return None


def _train_fold_aware(spec: dict[str, Any], *, resume: bool, n_jobs: int) -> dict[str, Any]:
    code = spec["code"]
    dest = exp_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    init_heads()
    five_base = [x for x in unique_keep(list(FIVE_FEATURES) + list(spec.get("five_extras") or ())) if x not in set(spec.get("five_drops") or ())]
    ten_base = [x for x in unique_keep(list(TEN_FEATURES) + list(spec.get("ten_extras") or ())) if x not in set(spec.get("ten_drops") or ())]
    ident = _write_ident(dest, spec, five_base, ten_base, extra={"per_fold": True})
    existing = _done(dest, ident) if resume else None
    if existing is not None:
        return existing

    fold_five = spec.get("fold_five_drops") or {}
    fold_ten = spec.get("fold_ten_drops") or {}

    def _head(name: str, training, candidates, locked, extras, fold_drops, head):
        frames = []
        fold_feat: dict[str, list[str]] = {}
        for seed in SEEDS:
            seed_dir = dest / name / f"seed{seed}"
            seed_path = seed_dir / "predictions.parquet"
            if resume and seed_path.is_file() and (seed_dir / "fold_features.json").is_file():
                prediction = pd.read_parquet(seed_path)
                prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
                frames.append(prediction)
                fold_feat.update(load_json(seed_dir / "fold_features.json"))
                continue
            preds = []
            years = [y for y in PREDICT_YEARS if int(candidates["trade_date"].dt.year.eq(y).sum()) > 0]
            for y in years:
                drops = list(fold_drops.get(str(y)) or spec.get(f"{name}_drops") or spec.get(f"{'five' if name=='five' else 'ten'}_drops") or ())
                features = [x for x in unique_keep([*locked, *extras]) if x not in set(drops)]
                fold_feat[str(y)] = features
                train_m, _ = fold_masks(training, predict_year=y, label=head.target)
                tr_all = training.loc[train_m].copy()
                tr, _ = complete_cases(tr_all, features, require_target=True, target=head.target)
                te = candidates.loc[candidates["trade_date"].dt.year.eq(y)].copy()
                te, _ = complete_cases(te, features, require_target=False, target=head.target)
                if tr.empty or te.empty:
                    continue
                params = model_parameters(int(seed), n_jobs=n_jobs)
                params["eval_metric"] = head.eval_metric
                params["rank_n_bins"] = head.rank_n_bins
                config = ranker_config_from_params(params)
                part, meta = rolling_train_predict_open(
                    tr,
                    te,
                    None,
                    features=features,
                    first_prediction_year=y,
                    last_prediction_year=y,
                    config=config,
                    target=head.target,
                    label_end_column=head.label_end_column,
                    save_models=False,
                    pre_cleaned=True,
                    clean_audit=({}, {}),
                )
                part["feature_set_hash"] = feature_set_hash(features)
                preds.append(part)
            if not preds:
                raise RuntimeError(f"{code} {name} seed{seed} produced no predictions")
            prediction = pd.concat(preds, ignore_index=True)
            seed_dir.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(seed_path, index=False)
            dump_json(seed_dir / "fold_features.json", fold_feat)
            frames.append(prediction)
        ens = ensemble_rank_pct(frames)
        ens.to_parquet(dest / f"{name}_pred.parquet", index=False)
        return ens, fold_feat

    five_extras = list(spec.get("five_extras") or ())
    ten_extras = list(spec.get("ten_extras") or ())
    five, five_folds = _head("five", _G["five_training"], _G["five_candidate"], FIVE_FEATURES, five_extras, fold_five, FIVE_HEAD)
    ten, ten_folds = _head("ten", _G["ten_training"], _G["ten_candidate"], TEN_FEATURES, ten_extras, fold_ten, TEN_HEAD)
    return _finish(spec, dest, five, ten, five_base, ten_base, ident, extra={"five_fold_features": five_folds, "ten_fold_features": ten_folds})


def _persist_pred_artifacts(dest, five: pd.DataFrame, ten: pd.DataFrame, ident: dict[str, Any]) -> dict[str, Any]:
    daily = pd.concat(
        [
            five.assign(head="five"),
            ten.assign(head="ten"),
        ],
        ignore_index=True,
    )
    daily.to_parquet(dest / "daily_prediction.parquet", index=False)
    pred_hash = {
        "five_pred": file_sha256(dest / "five_pred.parquet") if (dest / "five_pred.parquet").is_file() else None,
        "ten_pred": file_sha256(dest / "ten_pred.parquet") if (dest / "ten_pred.parquet").is_file() else None,
        "daily_prediction": file_sha256(dest / "daily_prediction.parquet"),
    }
    model_hash = {
        "cache_key": ident.get("cache_key"),
        "five_feature_set_hash": ident.get("five_feature_set_hash"),
        "ten_feature_set_hash": ident.get("ten_feature_set_hash"),
        "model_config": sha256_json(model_parameters(17)),
        "note": "models not persisted (save_models=False); hash is config+feature_set",
    }
    dump_json(dest / "prediction_hash.json", pred_hash)
    dump_json(dest / "model_hash.json", model_hash)
    return {"prediction_hash": pred_hash, "model_hash": sha256_json(model_hash)}


def backfill_artifacts() -> int:
    n = 0
    root = ROUND21_1_ROOT / "candidates"
    if not root.is_dir():
        return 0
    for dest in sorted(root.iterdir()):
        if not dest.is_dir():
            continue
        five_p, ten_p = dest / "five_pred.parquet", dest / "ten_pred.parquet"
        if not five_p.is_file() or not ten_p.is_file():
            continue
        if (dest / "daily_prediction.parquet").is_file() and (dest / "prediction_hash.json").is_file():
            continue
        ident = load_json(dest / "cache_key.json")
        _persist_pred_artifacts(dest, pd.read_parquet(five_p), pd.read_parquet(ten_p), ident)
        n += 1
    return n


def _finish(spec, dest, five, ten, five_feat, ten_feat, ident, extra=None) -> dict[str, Any]:
    five.to_parquet(dest / "five_pred.parquet", index=False)
    ten.to_parquet(dest / "ten_pred.parquet", index=False)
    hashes = _persist_pred_artifacts(dest, five, ten, ident)
    top = pd.concat(
        [
            _top12(five).assign(head="five"),
            _top12(ten).assign(head="ten"),
        ],
        ignore_index=True,
    )
    top.to_parquet(dest / "daily_top1_top2.parquet", index=False)
    five_oof = oof_rankic(five, _G["five_candidate"], target=FIVE_HEAD.target)
    ten_oof = oof_rankic(ten, _G["ten_candidate"], target=TEN_HEAD.target)
    payload = {
        "code": spec["code"],
        "role": spec.get("role"),
        "status": "OOF",
        "factor_id": spec.get("factor_id"),
        "formal": bool(spec.get("formal", True)),
        "five_extras": list(spec.get("five_extras") or ()),
        "ten_extras": list(spec.get("ten_extras") or ()),
        "five_drops": list(spec.get("five_drops") or ()),
        "ten_drops": list(spec.get("ten_drops") or ()),
        "five_features": list(five_feat),
        "ten_features": list(ten_feat),
        "five_feature_set_hash": feature_set_hash(list(five_feat)),
        "ten_feature_set_hash": feature_set_hash(list(ten_feat)),
        "cache_key": ident.get("cache_key"),
        "alias_from": None,
        "five_oof": five_oof,
        "ten_oof": ten_oof,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "filled_with_zero": False,
        "out_dir": str(dest),
        "looked_at_combo_nav": False,
        **hashes,
        **(extra or {}),
    }
    dump_json(dest / "oof.json", payload)
    dump_json(dest / "summary.json", {**payload, "cagr": None, "sharpe": None, "status": "OOF_ONLY"})
    dump_json(dest / "training_data_hash.json", {"five_hash": ident.get("five_feature_set_hash"), "ten_hash": ident.get("ten_feature_set_hash")})
    return payload


def _reuse_r21_preds(spec: dict[str, Any], dest, five_feat, ten_feat, ident) -> dict[str, Any] | None:
    """仅当特征集合与 R21 ADD/A01 一致时只读复制预测到 round21_1。Replace 永不复用 ADD。"""
    from ..paths import ROUND21_ROOT

    role = spec.get("role")
    src = None
    if role == "E0_CONTROL":
        src = ROUND21_ROOT / "candidates" / "R21_A01"
    elif role == "E2_ADD" and spec.get("factor_id"):
        src = ROUND21_ROOT / "candidates" / f"R21_ADD_{spec['factor_id']}"
    if src is None or not (src / "five_pred.parquet").is_file() or not (src / "ten_pred.parquet").is_file():
        return None
    if spec.get("five_drops") or spec.get("ten_drops") or spec.get("per_fold_drops"):
        return None
    if role == "E3_REPLACE" or str(spec.get("code") or "").startswith("R21_1_E3"):
        return None
    src_oof = load_json(src / "oof.json")
    src_five_ex = tuple(src_oof.get("five_extras") or src_oof.get("five_new") or ())
    src_ten_ex = tuple(src_oof.get("ten_extras") or src_oof.get("ten_new") or ())
    if role == "E2_ADD":
        if src_five_ex != tuple(spec.get("five_extras") or ()) or src_ten_ex != tuple(spec.get("ten_extras") or ()):
            return None
    five = pd.read_parquet(src / "five_pred.parquet")
    ten = pd.read_parquet(src / "ten_pred.parquet")
    print(f"[round21.1] REUSE_R21 {spec['code']} from {src.name} (read-only copy)", flush=True)
    return _finish(spec, dest, five, ten, five_feat, ten_feat, ident, extra={"reused_from": str(src), "r21_overwritten": False})


def train_one(spec: dict[str, Any], *, resume: bool = True, n_jobs: int = XGB_N_JOBS) -> dict[str, Any]:
    if spec.get("alias_from"):
        raise RuntimeError(f"{spec['code']} forbids alias_from; Replace must retrain")
    if spec.get("per_fold_drops"):
        return _train_fold_aware(spec, resume=resume, n_jobs=n_jobs)
    code = spec["code"]
    dest = exp_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    five_feat = [x for x in unique_keep(list(FIVE_FEATURES) + list(spec.get("five_extras") or ())) if x not in set(spec.get("five_drops") or ())]
    ten_feat = [x for x in unique_keep(list(TEN_FEATURES) + list(spec.get("ten_extras") or ())) if x not in set(spec.get("ten_drops") or ())]
    ident = _write_ident(dest, spec, five_feat, ten_feat)
    if resume:
        existing = _done(dest, ident)
        if existing is not None:
            return existing
    init_heads()
    reused = _reuse_r21_preds(spec, dest, five_feat, ten_feat, ident)
    if reused is not None:
        return reused
    trained = train_both_heads(
        five_training=_G["five_training"],
        five_candidates=_G["five_candidate"],
        ten_training=_G["ten_training"],
        ten_candidates=_G["ten_candidate"],
        five_extras=list(spec.get("five_extras") or ()),
        ten_extras=list(spec.get("ten_extras") or ()),
        five_drops=list(spec.get("five_drops") or ()),
        ten_drops=list(spec.get("ten_drops") or ()),
        output_dir=dest,
        resume=resume,
        five_frozen=_G["five_frozen"],
        ten_frozen=_G["ten_frozen"],
        force_retrain=True,
        n_jobs=n_jobs,
    )
    return _finish(spec, dest, trained["five_pred"], trained["ten_pred"], five_feat, ten_feat, ident)


def _assert_ident(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["code"]: r for r in rows}
    e0 = by.get(E0_CODE) or {}
    issues = []
    for r in rows:
        if r.get("code") == E0_CODE:
            continue
        fid = r.get("factor_id")
        if not fid:
            continue
        e1, e2, e3 = by.get(f"R21_1_E1_{fid}"), by.get(f"R21_1_E2_{fid}"), by.get(f"R21_1_E3_{fid}")
        if e1 and e1.get("five_feature_set_hash") == e0.get("five_feature_set_hash") and e1.get("ten_feature_set_hash") == e0.get("ten_feature_set_hash"):
            issues.append({"code": e1["code"], "assert": "E1!=E0", "ok": False})
        elif e1:
            issues.append({"code": e1["code"], "assert": "E1!=E0", "ok": True})
        if e2 and e2.get("five_feature_set_hash") == e0.get("five_feature_set_hash"):
            issues.append({"code": e2["code"], "assert": "E2!=E0", "ok": False})
        elif e2:
            issues.append({"code": e2["code"], "assert": "E2!=E0", "ok": True})
        if e2 and e3 and e2.get("five_feature_set_hash") == e3.get("five_feature_set_hash") and e2.get("ten_feature_set_hash") == e3.get("ten_feature_set_hash"):
            issues.append({"code": e3["code"], "assert": "E3!=E2", "ok": False})
        elif e3:
            issues.append({"code": e3["code"], "assert": "E3!=E2", "ok": True})
        if e3:
            added = set(e3.get("five_extras") or []) | set(e3.get("ten_extras") or [])
            dropped = set(e3.get("five_drops") or []) | set(e3.get("ten_drops") or [])
            ok = fid in added and (not dropped or dropped.isdisjoint({fid}))
            if dropped:
                ok = ok and dropped.isdisjoint(added)
            issues.append({"code": e3["code"], "assert": "E3_has_new_not_old", "ok": bool(ok), "added": list(added), "dropped": list(dropped)})
    return {"n": len(issues), "all_ok": all(i.get("ok") for i in issues), "rows": issues}


def _gate(row: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    gate = INCREMENT_GATE
    bf = (base.get("five_oof") or {}).get("oof_rankic_mean")
    bt = (base.get("ten_oof") or {}).get("oof_rankic_mean")
    rf = (row.get("five_oof") or {}).get("oof_rankic_mean")
    rt = (row.get("ten_oof") or {}).get("oof_rankic_mean")
    d5 = None if rf is None or bf is None else float(rf) - float(bf)
    d10 = None if rt is None or bt is None else float(rt) - float(bt)
    not_worse = rf is not None and bf is not None and rt is not None and bt is not None and rf + gate["eps"] >= bf and rt + gate["eps"] >= bt
    better = (d5 is not None and d5 > gate["min_delta"]) or (d10 is not None and d10 > gate["min_delta"])
    days_ok = int((row.get("five_oof") or {}).get("n_oos_days") or 0) >= gate["min_oos_days"] and int((row.get("ten_oof") or {}).get("n_oos_days") or 0) >= gate["min_oos_days"]
    return {"passed": bool(not_worse and better and days_ok), "delta_five": d5, "delta_ten": d10, "gate_id": gate["id"]}


def run_train(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    if resume and MODEL_JSON.is_file():
        existing = load_json(MODEL_JSON)
        if existing.get("status") == "DONE":
            return existing
    prepare_head_panels(resume=True)
    init_heads()
    matrix = freeze_matrix(resume=True)
    specs = list(matrix.get("specs") or [])
    xgb_jobs = 1 if int(jobs) > 1 else XGB_N_JOBS
    rows: list[dict[str, Any]] = []

    def _run(spec: dict[str, Any]) -> dict[str, Any]:
        print(f"[round21.1] TRAIN {spec['code']} role={spec.get('role')} drops5={spec.get('five_drops')} drops10={spec.get('ten_drops')}", flush=True)
        return train_one(spec, resume=resume, n_jobs=xgb_jobs)

    if int(jobs) <= 1:
        for spec in specs:
            rows.append(_run(spec))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
            futs = {pool.submit(_run, spec): spec["code"] for spec in specs}
            order = {spec["code"]: i for i, spec in enumerate(specs)}
            got = {}
            for fut in as_completed(futs):
                got[futs[fut]] = fut.result()
            rows = [got[c] for c, _ in sorted(order.items(), key=lambda t: t[1]) if c in got]

    backfill_artifacts()
    ident = _assert_ident(rows)
    dump_json(IDENT_JSON, ident)
    if not ident.get("all_ok"):
        print(f"[round21.1] IDENTIFIABILITY issues: {ident}", flush=True)
    by = {r["code"]: r for r in rows}
    base = by.get(E0_CODE) or {}
    increment = []
    for r in rows:
        inc = {"passed": False, "reason": "anchor"} if r.get("code") == E0_CODE else _gate(r, base)
        increment.append({**r, "increment": inc})
        fid = r.get("factor_id")
        if fid:
            dest = ROUND21_1_ROOT / "factors" / fid
            dest.mkdir(parents=True, exist_ok=True)
            prev = load_json(dest / "incrementality.json") or {"factor_id": fid, "vs": E0_CODE, "runs": []}
            runs = [x for x in (prev.get("runs") or []) if x.get("code") != r.get("code")]
            runs.append({"code": r.get("code"), "role": r.get("role"), "oof": {"five": r.get("five_oof"), "ten": r.get("ten_oof")}, "increment": inc})
            dump_json(dest / "incrementality.json", {"factor_id": fid, "vs": E0_CODE, "runs": runs})
    OOF_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "code": r.get("code"),
                "role": r.get("role"),
                "five_oof": (r.get("five_oof") or {}).get("oof_rankic_mean"),
                "ten_oof": (r.get("ten_oof") or {}).get("oof_rankic_mean"),
                "delta_five": (r.get("increment") or {}).get("delta_five"),
                "delta_ten": (r.get("increment") or {}).get("delta_ten"),
                "increment_passed": (r.get("increment") or {}).get("passed"),
            }
            for r in increment
        ]
    ).to_csv(OOF_CSV, index=False)
    payload = {
        "status": "DONE",
        "n_oof_models": len(rows),
        "n_increment_pass": sum(1 for r in increment if (r.get("increment") or {}).get("passed")),
        "identifiability": ident,
        "e0": {"five_oof": base.get("five_oof"), "ten_oof": base.get("ten_oof")},
        "rows": [
            {
                "code": r.get("code"),
                "role": r.get("role"),
                "status": r.get("status"),
                "five_oof": r.get("five_oof"),
                "ten_oof": r.get("ten_oof"),
                "increment": r.get("increment"),
                "five_drops": r.get("five_drops"),
                "ten_drops": r.get("ten_drops"),
                "cache_key": r.get("cache_key"),
            }
            for r in increment
        ],
    }
    dump_json(MODEL_JSON, payload)
    lines = [
        "# Round21.1 · 05 四路真实训练",
        "",
        f"N=`{len(rows)}`。禁止 alias ADD。缓存 key 含 added/dropped/fold/head/feature_set_hash。",
        f"Identifiability all_ok=`{ident.get('all_ok')}`。",
        "",
        "| code | role | five OOF | ten OOF | Δfive | Δten | drops5 | drops10 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in increment:
        inc = r.get("increment") or {}
        lines.append(
            f"| `{r.get('code')}` | {r.get('role')} | {(r.get('five_oof') or {}).get('oof_rankic_mean')} | "
            f"{(r.get('ten_oof') or {}).get('oof_rankic_mean')} | {inc.get('delta_five')} | {inc.get('delta_ten')} | "
            f"{r.get('five_drops')} | {r.get('ten_drops')} |"
        )
    write_md(TRAIN_MD, "\n".join(lines))
    write_md(WF_MD, "# Round21.1 · 08 Walk-Forward\n\n" + "\n".join(lines[2:]))
    return payload
