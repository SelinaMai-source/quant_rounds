"""全历史 2018–2026 重训。C1/C4 在 2020-09-24 前保持 NaN，不填 0，不删股票。规范 3y/1y。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..paths import PREDICTION_START, ROUND21_2_ROOT, ROUND21_3_ROOT, ROUND21_4_ROOT, ROUND22_ROOT, SIGNAL_END, assert_project_result_path
from ..round17.train import model_parameters, ranker_config_from_params, rolling_train_predict_open
from ..round20.train import _invariance, new_columns, slim_panel, unique_keep
from ..round21.runner import oof_rankic
from ..round21_2.panels import _G
from ..stock_readonly import sha256_file
from .common import dump_json, feature_set_hash, file_sha256, load_json, sha256_json, write_csv, write_md
from .contract import (
    C1_LAG1_ID,
    C4_ID,
    C_AVAILABLE_FROM,
    C_PANEL_22_2,
    COLSAMPLE,
    E0_CODE,
    E0_IMPORT_CANDIDATES,
    FIRST_PREDICTION_YEAR,
    FIVE_FEATURES,
    FIVE_HEAD,
    HN2_ID,
    HN2_PANEL_22_2,
    IMPORTED_BY_HASH,
    KEYS,
    LAST_PREDICTION_YEAR,
    MD,
    OOF_INCREMENT_GATE,
    REPLAY_GROUPS,
    ROUND22_3_ROOT,
    SEEDS,
    SUBSAMPLE,
    TEN_FEATURES,
    TEN_HEAD,
    TRAINED_REAL,
    XGB_N_JOBS,
    exp_dir,
)
from .freeze import require_frozen_gate

DATA = ROUND22_3_ROOT / "data"
TRAIN_JSON = ROUND22_3_ROOT / "model" / "train.json"
REG_CSV = ROUND22_3_ROOT / "model" / "experiment_registry.csv"
OOF_CSV = ROUND22_3_ROOT / "model" / "oof_results.csv"
HASH_CSV = ROUND22_3_ROOT / "model" / "hash_ledger.csv"
PANEL_NAMES = (
    "five_training.parquet",
    "five_candidate.parquet",
    "ten_training.parquet",
    "ten_candidate.parquet",
    "five_frozen.parquet",
    "ten_frozen.parquet",
)
C_MASK_COLS = (C1_LAG1_ID, C4_ID, "r22_c1_xl_net_imbalance")


def _copy_base_panels() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    roots = [ROUND21_4_ROOT / "data", ROUND21_3_ROOT / "data", ROUND21_2_ROOT / "data", ROUND22_ROOT / "data"]
    for name in PANEL_NAMES:
        dest = DATA / name
        if dest.is_file() and dest.stat().st_size > 100:
            continue
        src = next((root / name for root in roots if (root / name).is_file()), None)
        if src is None and name.endswith("frozen.parquet"):
            continue
        if src is None:
            raise RuntimeError(f"MISSING_ARTIFACT head panel {name}")
        shutil.copy2(src, dest)


def _mask_c_before_pit(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    cutoff = pd.Timestamp(C_AVAILABLE_FROM)
    for col in C_MASK_COLS:
        if col in work.columns:
            work.loc[work["trade_date"] < cutoff, col] = np.nan
    return work


def _merge_new_factors() -> dict[str, Any]:
    cols: list[str] = []
    frames = []
    audit: dict[str, Any] = {"filled_with_zero": False, "dropped_names": False, "masked_before": C_AVAILABLE_FROM}
    if C_PANEL_22_2.is_file():
        c = pd.read_parquet(C_PANEL_22_2)
        c["trade_date"] = pd.to_datetime(c["trade_date"])
        c["instrument_id"] = c["instrument_id"].astype(str)
        c = _mask_c_before_pit(c)
        frames.append(c)
        dest_c = DATA / "c_factor_panel.parquet"
        keep_c = [c for c in c.columns if c in KEYS or c.startswith("r22_")]
        c[keep_c].drop_duplicates(KEYS).to_parquet(dest_c, index=False)
        audit["c_panel"] = str(dest_c)
        audit["c_n"] = int(len(c))
        for col in C_MASK_COLS:
            if col in c.columns:
                s = pd.to_numeric(c[col], errors="coerce")
                finite = c.loc[s.notna()]
                audit[f"{col}_finite_share"] = float(s.notna().mean())
                audit[f"{col}_min_finite"] = str(pd.Timestamp(finite["trade_date"].min()).date()) if len(finite) else None
    if HN2_PANEL_22_2.is_file():
        hn = pd.read_parquet(HN2_PANEL_22_2)
        hn["trade_date"] = pd.to_datetime(hn["trade_date"])
        hn["instrument_id"] = hn["instrument_id"].astype(str)
        frames.append(hn)
        dest_h = DATA / "hn2_panel.parquet"
        keep_h = [c for c in hn.columns if c in KEYS or c == HN2_ID]
        hn[keep_h].drop_duplicates(KEYS).to_parquet(dest_h, index=False)
        audit["hn2_panel"] = str(dest_h)
        audit["hn2_n"] = int(len(hn))
        if HN2_ID in hn.columns:
            s = pd.to_numeric(hn[HN2_ID], errors="coerce")
            audit["hn2_finite_share"] = float(s.notna().mean())
            audit["hn2_filled_with_zero"] = False
    if not frames:
        return {"cols": cols, **audit}
    add = frames[0]
    for extra in frames[1:]:
        more = [c for c in extra.columns if c not in add.columns and c not in {"ann_dt", "known_at", "available_at", "stale"}]
        if more:
            add = add.merge(extra[KEYS + more], on=KEYS, how="left")
    cols = [c for c in add.columns if c.startswith("r22_") or c.startswith("r22_2_")]
    add = add[KEYS + cols].drop_duplicates(KEYS)
    add = _mask_c_before_pit(add)
    for name in ("five_training.parquet", "five_candidate.parquet", "ten_training.parquet", "ten_candidate.parquet"):
        path = DATA / name
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        n_before = frame["instrument_id"].nunique() if "instrument_id" in frame.columns else 0
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            frame = frame.merge(add[KEYS + missing], on=KEYS, how="left")
        frame = _mask_c_before_pit(frame)
        n_after = frame["instrument_id"].nunique()
        if n_after < n_before:
            raise RuntimeError(f"{name} dropped names {n_before}->{n_after}; refuse")
        frame.to_parquet(path, index=False)
    audit["cols"] = cols
    dump_json(DATA / "merge_audit.json", audit)
    return audit


def _bind() -> None:
    _G.clear()
    for name in PANEL_NAMES:
        path = DATA / name
        if not path.is_file():
            continue
        key = name.replace(".parquet", "")
        frame = pd.read_parquet(path)
        if "trade_date" in frame.columns:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        if "signal_date" in frame.columns:
            frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        if "instrument_id" in frame.columns:
            frame["instrument_id"] = frame["instrument_id"].astype(str)
        _G[key] = frame


def _lists(spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    five = [x for x in unique_keep(list(FIVE_FEATURES) + list(spec.get("five_extras") or ())) if x not in set(spec.get("five_drops") or ())]
    ten = [x for x in unique_keep(list(TEN_FEATURES) + list(spec.get("ten_extras") or ())) if x not in set(spec.get("ten_drops") or ())]
    for col in spec.get("five_drops") or ():
        if col in five:
            raise RuntimeError(f"{spec['code']} failed true delete {col}")
    for col in spec.get("ten_drops") or ():
        if col not in TEN_FEATURES and col not in (spec.get("ten_extras") or ()):
            raise RuntimeError(f"{spec['code']} ten drop {col} is not a Ten column")
        if col in ten:
            raise RuntimeError(f"{spec['code']} failed true ten delete {col}")
    return five, ten


def complete_cases_keep_extras(
    frame: pd.DataFrame,
    features: Sequence[str],
    extra_features: Sequence[str],
    *,
    require_target: bool,
    target: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    extras = [c for c in extra_features if c in features]
    base = [c for c in features if c not in extras]
    needed = [*base, *([target] if require_target else [])]
    missing = sorted({*KEYS, *needed} - set(frame.columns))
    if missing:
        raise ValueError(f"INVALID_FACTOR panel missing required columns: {missing}")
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    present = [c for c in features if c in work.columns]
    work[present] = work[present].replace([np.inf, -np.inf], np.nan)
    mask = work[needed].notna().all(axis=1)
    audit = {
        "rows_before": len(work),
        "rows_dropped_in_memory": int((~mask).sum()),
        "rows_after": int(mask.sum()),
        "require_target": require_target,
        "base_required": list(base),
        "extras_allowed_nan": list(extras),
        "feature_missing_rows": {name: int(work[name].isna().sum()) for name in present},
        "filled_with_zero": False,
        "dropped_names_for_extra_nan": False,
    }
    clean = work.loc[mask].sort_values(KEYS).reset_index(drop=True)
    return clean, audit


def train_head_keep_nan_extras(
    *,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
    features: Sequence[str],
    extras: Sequence[str],
    target: str,
    label_end_column: str,
    rank_n_bins: int,
    eval_metric: str,
    first_prediction_year: int,
    last_prediction_year: int,
    output_dir: Path | None,
    resume: bool,
) -> dict[str, Any]:
    from ..round17.ensemble import assert_seed_diversity, ensemble_rank_pct

    training_clean, train_audit = complete_cases_keep_extras(
        training, features, extras, require_target=True, target=target
    )
    candidates_clean, cand_audit = complete_cases_keep_extras(
        candidates, features, extras, require_target=False, target=target
    )
    frames = []
    seed_reports = []
    for seed in SEEDS:
        seed_dir = None
        seed_path = None
        if output_dir is not None:
            seed_dir = Path(output_dir) / f"seed{seed}"
            seed_path = seed_dir / "predictions.parquet"
            if resume and seed_path is not None and seed_path.is_file():
                prediction = pd.read_parquet(seed_path)
                prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
                prediction = prediction.loc[prediction["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)].copy()
                metadata = json.loads((seed_dir / "training_report.json").read_text(encoding="utf-8"))
                frames.append(prediction)
                seed_reports.append(
                    {
                        "seed": int(seed),
                        "rows": int(len(prediction)),
                        "resumed": True,
                        "folds": metadata.get("folds", []),
                        "prediction_sha256": sha256_file(seed_path) if seed_path.is_file() else None,
                    }
                )
                continue
        params = model_parameters(seed, subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE, n_jobs=XGB_N_JOBS)
        params["eval_metric"] = eval_metric
        params["rank_n_bins"] = rank_n_bins
        config = ranker_config_from_params(params)
        prediction, metadata = rolling_train_predict_open(
            training_clean,
            candidates_clean,
            seed_dir,
            features=features,
            first_prediction_year=first_prediction_year,
            last_prediction_year=last_prediction_year,
            config=config,
            target=target,
            label_end_column=label_end_column,
            save_models=False,
            pre_cleaned=True,
            clean_audit=(train_audit, cand_audit),
        )
        prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
        prediction = prediction.loc[prediction["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)].copy()
        frames.append(prediction)
        seed_reports.append(
            {
                "seed": int(seed),
                "rows": int(len(prediction)),
                "resumed": False,
                "folds": metadata.get("folds", []),
                "prediction_sha256": sha256_file(seed_path) if seed_path is not None and seed_path.is_file() else None,
            }
        )
    diversity = assert_seed_diversity(frames)
    ensemble = ensemble_rank_pct(frames)
    payload = {
        "status": "trained_oos",
        "features": list(features),
        "target": target,
        "label_end_column": label_end_column,
        "subsample": float(SUBSAMPLE),
        "colsample_bytree": float(COLSAMPLE),
        "seeds": list(SEEDS),
        "seed_diversity": diversity,
        "rows": int(len(ensemble)),
        "days": int(ensemble["signal_date"].nunique()) if len(ensemble) else 0,
        "seed_reports": seed_reports,
        "ensemble": ensemble,
        "filled_with_zero": False,
        "nan_extras_kept": list(extras),
        "train_clean_audit": train_audit,
        "candidate_clean_audit": cand_audit,
        "first_prediction_year": int(first_prediction_year),
        "last_prediction_year": int(last_prediction_year),
        "walk_forward": "exact prior 3 calendar years train, next 1 calendar year predict",
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        assert_project_result_path(root)
        ens_path = root / "predictions.parquet"
        ensemble.to_parquet(ens_path, index=False)
        slim = {k: v for k, v in payload.items() if k not in {"ensemble"}}
        slim["prediction_sha256"] = sha256_file(ens_path)
        (root / "ensemble_report.json").write_text(
            json.dumps(slim, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return payload


def _import_e0(spec: dict[str, Any], five: list[str], ten: list[str]) -> str | None:
    if spec["code"] != E0_CODE:
        return None
    if list(five) != list(FIVE_FEATURES) or list(ten) != list(TEN_FEATURES):
        return None
    dest = exp_dir(spec["code"])
    dest.mkdir(parents=True, exist_ok=True)
    for src in E0_IMPORT_CANDIDATES:
        feat = load_json(src / "feature_list.json")
        if feat and (list(feat.get("five") or []) != list(five) or list(feat.get("ten") or []) != list(ten)):
            continue
        if not (src / "five_pred.parquet").is_file() or not (src / "ten_pred.parquet").is_file():
            continue
        shutil.copy2(src / "five_pred.parquet", dest / "five_pred.parquet")
        shutil.copy2(src / "ten_pred.parquet", dest / "ten_pred.parquet")
        dump_json(
            dest / "imported_by_hash.json",
            {
                "source": str(src),
                "five_feature_set_hash": feature_set_hash(five),
                "ten_feature_set_hash": feature_set_hash(ten),
                "reason": "feature hash matches v5; full-history import; not alias ADD",
                "window": "2018-2026",
            },
        )
        return str(src)
    return None


def _finish(spec: dict[str, Any], five: list[str], ten: list[str], origin: str) -> dict[str, Any]:
    dest = exp_dir(spec["code"])
    five_pred = pd.read_parquet(dest / "five_pred.parquet")
    ten_pred = pd.read_parquet(dest / "ten_pred.parquet")
    ident = {
        "code": spec["code"],
        "five_features": five,
        "ten_features": ten,
        "five_feature_set_hash": feature_set_hash(five),
        "ten_feature_set_hash": feature_set_hash(ten),
        "five_drops": list(spec.get("five_drops") or ()),
        "ten_drops": list(spec.get("ten_drops") or ()),
        "five_extras": list(spec.get("five_extras") or ()),
        "ten_extras": list(spec.get("ten_extras") or ()),
        "window": "2018-2026",
        "first_prediction_year": FIRST_PREDICTION_YEAR,
        "last_prediction_year": LAST_PREDICTION_YEAR,
    }
    ident["cache_key"] = sha256_json(ident)
    dump_json(dest / "feature_list.json", {"five": five, "ten": ten})
    dump_json(dest / "cache_key.json", ident)
    dump_json(dest / "model_hash.json", {"cache_key": ident["cache_key"], "five_feature_set_hash": ident["five_feature_set_hash"], "ten_feature_set_hash": ident["ten_feature_set_hash"]})
    dump_json(dest / "prediction_hash.json", {"five_pred": file_sha256(dest / "five_pred.parquet"), "ten_pred": file_sha256(dest / "ten_pred.parquet")})
    five_oof = oof_rankic(five_pred, _G["five_candidate"], target=FIVE_HEAD.target)
    ten_oof = oof_rankic(ten_pred, _G["ten_candidate"], target=TEN_HEAD.target)
    payload = {
        "code": spec["code"],
        "role": spec.get("role"),
        "factor_id": spec.get("factor_id"),
        "r222": spec.get("r222"),
        "status": "OOF",
        "origin": origin,
        "formal": True,
        **ident,
        "five_oof": five_oof,
        "ten_oof": ten_oof,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "filled_with_zero": False,
        "alias_from": None,
        "compare_to": "R16_O01",
        "paste_2023_preds": False,
    }
    dump_json(dest / "oof.json", payload)
    return payload


def _train_both(spec: dict[str, Any], five: list[str], ten: list[str], dest: Path, *, resume: bool) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)

    def _head(name: str, training, candidates, features, extras, head):
        extra_cols = [head.target, head.label_end_column]
        tr = slim_panel(training, [*KEYS, *features, *extra_cols])
        ca = slim_panel(candidates, [*KEYS, *features, head.target])
        payload = train_head_keep_nan_extras(
            training=tr,
            candidates=ca,
            features=features,
            extras=extras,
            target=head.target,
            label_end_column=head.label_end_column,
            rank_n_bins=head.rank_n_bins,
            eval_metric=head.eval_metric,
            first_prediction_year=FIRST_PREDICTION_YEAR,
            last_prediction_year=LAST_PREDICTION_YEAR,
            output_dir=dest / name,
            resume=resume,
        )
        payload["force_retrain"] = True
        payload["extras"] = list(extras)
        payload["features"] = list(features)
        _invariance(payload)
        return payload["ensemble"], payload

    five_pred, five_meta = _head("five", _G["five_training"], _G["five_candidate"], five, spec.get("five_extras") or (), FIVE_HEAD)
    ten_pred, ten_meta = _head("ten", _G["ten_training"], _G["ten_candidate"], ten, spec.get("ten_extras") or (), TEN_HEAD)
    return {
        "five_pred": five_pred,
        "ten_pred": ten_pred,
        "five": {k: v for k, v in five_meta.items() if k not in {"ensemble", "seed_frames"}},
        "ten": {k: v for k, v in ten_meta.items() if k not in {"ensemble", "seed_frames"}},
        "five_new": new_columns(FIVE_FEATURES, spec.get("five_extras") or ()),
        "ten_new": new_columns(TEN_FEATURES, spec.get("ten_extras") or ()),
        "status": "trained_oos",
    }


def train_one(spec: dict[str, Any], *, resume: bool = True) -> dict[str, Any]:
    if spec.get("alias_from"):
        raise RuntimeError("alias ADD is forbidden")
    five, ten = _lists(spec)
    dest = exp_dir(spec["code"])
    if resume and (dest / "oof.json").is_file() and (dest / "five_pred.parquet").is_file():
        existing = load_json(dest / "oof.json")
        if existing.get("five_feature_set_hash") == feature_set_hash(five) and existing.get("ten_feature_set_hash") == feature_set_hash(ten):
            if existing.get("first_prediction_year") == FIRST_PREDICTION_YEAR:
                existing["resumed"] = True
                return existing
    src = _import_e0(spec, five, ten)
    if src:
        _bind()
        return _finish(spec, five, ten, IMPORTED_BY_HASH)
    _bind()
    missing_five = [c for c in five if c not in _G["five_training"].columns]
    missing_ten = [c for c in ten if c not in _G["ten_training"].columns]
    if missing_five or missing_ten:
        raise RuntimeError(f"{spec['code']} missing columns five={missing_five} ten={missing_ten}")
    print(f"[r223 train] {spec['code']} pred={FIRST_PREDICTION_YEAR}-{LAST_PREDICTION_YEAR} extras_nan_kept", flush=True)
    trained = _train_both(spec, five, ten, dest, resume=resume)
    if trained.get("five_pred") is None or trained.get("ten_pred") is None:
        raise RuntimeError(f"{spec['code']} train returned empty predictions")
    trained["five_pred"].to_parquet(dest / "five_pred.parquet", index=False)
    trained["ten_pred"].to_parquet(dest / "ten_pred.parquet", index=False)
    return _finish(spec, five, ten, TRAINED_REAL)


def _oof_gate(row: dict[str, Any], e0: dict[str, Any]) -> dict[str, Any]:
    eps = float(OOF_INCREMENT_GATE["eps"])
    five = (row.get("five_oof") or {}).get("oof_rankic_mean")
    ten = (row.get("ten_oof") or {}).get("oof_rankic_mean")
    base_five = (e0.get("five_oof") or {}).get("oof_rankic_mean")
    base_ten = (e0.get("ten_oof") or {}).get("oof_rankic_mean")
    if None in {five, ten, base_five, base_ten}:
        return {"passed": False, "reason": "missing_oof"}
    d5 = float(five) - float(base_five)
    d10 = float(ten) - float(base_ten)
    n5 = int((row.get("five_oof") or {}).get("n_oos_days") or 0)
    n10 = int((row.get("ten_oof") or {}).get("n_oos_days") or 0)
    not_worse = d5 >= -eps and d10 >= -eps
    better = d5 > eps or d10 > eps
    enough = n5 >= int(OOF_INCREMENT_GATE["min_oos_days"]) and n10 >= int(OOF_INCREMENT_GATE["min_oos_days"])
    return {"passed": bool(not_worse and better and enough), "delta_five": d5, "delta_ten": d10, "n5": n5, "n10": n10}


def run_train(*, resume: bool = True, jobs: int = 4, **_: Any) -> dict[str, Any]:
    del jobs
    require_frozen_gate()
    if resume and TRAIN_JSON.is_file():
        existing = load_json(TRAIN_JSON)
        if existing.get("status") in {"DONE"} and existing.get("sample") == "full_history":
            return existing
    _copy_base_panels()
    merge = _merge_new_factors()
    _bind()
    specs = [dict(g) for g in REPLAY_GROUPS]
    trained = []
    for spec in specs:
        trained.append(train_one(spec, resume=resume))
    e0 = next(r for r in trained if r["code"] == E0_CODE)
    if e0.get("origin") == IMPORTED_BY_HASH:
        pred = pd.read_parquet(exp_dir(E0_CODE) / "five_pred.parquet")
        col = "signal_date" if "signal_date" in pred.columns else "trade_date"
        years = pd.to_datetime(pred[col]).dt.year
        if int(years.min()) > 2018:
            raise RuntimeError("E0 imported preds start after 2018; refuse paste-2023-onto-2018")
    gates = []
    for row in trained:
        gate = {"code": row["code"], "role": row.get("role"), "factor_id": row.get("factor_id"), "origin": row.get("origin"), "r222": row.get("r222")}
        if row["code"] == E0_CODE:
            gate["oof_gate"] = {"passed": False, "reason": "control"}
        else:
            gate["oof_gate"] = _oof_gate(row, e0)
        gate["five_oof"] = (row.get("five_oof") or {}).get("oof_rankic_mean")
        gate["ten_oof"] = (row.get("ten_oof") or {}).get("oof_rankic_mean")
        gates.append(gate)
    ledger = [
        {
            "code": g["code"],
            "role": g.get("role"),
            "factor_id": g.get("factor_id"),
            "r222": g.get("r222"),
            "origin": g.get("origin"),
            "oof_pass": bool((g.get("oof_gate") or {}).get("passed")),
            "formal": True,
            "window": "2018-2026",
        }
        for g in gates
    ]
    write_csv(REG_CSV, ledger)
    write_csv(OOF_CSV, gates)
    write_csv(
        ROUND22_3_ROOT / "model" / "trial_registry.csv",
        [{"trial_id": g["code"], "role": g.get("role"), "factor_id": g.get("factor_id"), "origin": g.get("origin"), "oof_pass": (g.get("oof_gate") or {}).get("passed"), "round": "22.3"} for g in gates],
    )
    hashes = [
        {
            "code": row["code"],
            "origin": row.get("origin"),
            "five_feature_set_hash": row.get("five_feature_set_hash"),
            "ten_feature_set_hash": row.get("ten_feature_set_hash"),
            "prediction_five": (load_json(exp_dir(row["code"]) / "prediction_hash.json") or {}).get("five_pred"),
            "prediction_ten": (load_json(exp_dir(row["code"]) / "prediction_hash.json") or {}).get("ten_pred"),
        }
        for row in trained
    ]
    write_csv(HASH_CSV, hashes)
    payload = {
        "status": "DONE",
        "sample": "full_history",
        "window": "2018-01-02–2026-07-24",
        "first_prediction_year": FIRST_PREDICTION_YEAR,
        "last_prediction_year": LAST_PREDICTION_YEAR,
        "walk_forward": "2015-2017→2018, …, 2023-2025→2026 (canonical 3y/1y from 2018)",
        "n_specs": len(trained),
        "n_imported": int(sum(r.get("origin") == IMPORTED_BY_HASH for r in trained)),
        "n_trained_real": int(sum(r.get("origin") == TRAINED_REAL for r in trained)),
        "n_oof_pass": int(sum(bool((g.get("oof_gate") or {}).get("passed")) for g in gates)),
        "e0_origin": e0.get("origin"),
        "compare_mixed_full_v5": False,
        "compare_to": "R16_O01",
        "paste_2023_preds": False,
        "filled_with_zero": False,
        "dropped_names": False,
        "c_available_from": C_AVAILABLE_FROM,
        "merge": merge,
        "alias_add": False,
        "gates": gates,
    }
    dump_json(TRAIN_JSON, payload)
    write_md(
        MD["05"],
        f"# Round22.3 · 05 全历史真训\n\nE0 origin=`{e0.get('origin')}` sample=`full_history` 折=`{FIRST_PREDICTION_YEAR}-{LAST_PREDICTION_YEAR}`。真训={payload['n_trained_real']} 导入={payload['n_imported']}。C1/C4 在 `{C_AVAILABLE_FROM}` 前 NaN，不填 0，不删股票。禁止把 2023+ 预测贴到 2018 NAV。\n",
    )
    write_md(MD["13"], f"# Round22.3 · 13 E0–E9\n\n排队={payload['n_specs']}。对照全历史 v5 / R16_O01，不对照 R222_E0。\n")
    write_md(MD["14"], f"# Round22.3 · 14 OOF\n\n对照 `{E0_CODE}`。入围={payload['n_oof_pass']}。有 E3 必有 E1。\n")
    return payload
