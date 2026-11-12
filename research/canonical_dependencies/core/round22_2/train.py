"""E0–E9。真删 five_drops/ten_drops。禁止 alias ADD。非全历史则重训 matched v5。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND21_2_ROOT, ROUND21_3_ROOT, ROUND21_4_ROOT, ROUND22_ROOT
from ..round17.train import train_head_open
from ..round20.train import _invariance, new_columns, slim_panel, unique_keep
from ..round21.runner import oof_rankic
from ..round21_2.panels import _G
from .common import dump_json, feature_set_hash, file_sha256, load_json, sha256_json, write_csv, write_md
from .contract import (
    C4_ID,
    COLSAMPLE,
    E0_CODE,
    E0_IMPORT_CANDIDATES,
    EXPLORATORY_DIAGNOSTIC,
    FIVE_FEATURES,
    FIVE_HEAD,
    FULL_HISTORY_STRICT_PIT,
    IMPORTED_BY_HASH,
    INSUFFICIENT_HISTORY,
    KEYS,
    MD,
    MIN_TRAIN_YEARS,
    NON_CANONICAL_EXPLORATORY,
    NOT_APPLICABLE,
    OOF_INCREMENT_GATE,
    ROUND22_2_ROOT,
    SEEDS,
    SUBPERIOD_STRICT_PIT,
    SUBSAMPLE,
    TEN_FEATURES,
    TEN_HEAD,
    TRAINED_REAL,
    XGB_N_JOBS,
    exp_dir,
)
from .freeze import require_frozen_gate

DATA = ROUND22_2_ROOT / "data"
TRAIN_JSON = ROUND22_2_ROOT / "model" / "train.json"
REG_CSV = ROUND22_2_ROOT / "model" / "experiment_registry.csv"
OOF_CSV = ROUND22_2_ROOT / "model" / "oof_results.csv"
HASH_CSV = ROUND22_2_ROOT / "model" / "hash_ledger.csv"
PANEL_NAMES = (
    "five_training.parquet",
    "five_candidate.parquet",
    "ten_training.parquet",
    "ten_candidate.parquet",
    "five_frozen.parquet",
    "ten_frozen.parquet",
)


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


def _merge_new_factors() -> list[str]:
    cols: list[str] = []
    frames = []
    c_path = DATA / "c_factor_panel.parquet"
    if c_path.is_file():
        frames.append(pd.read_parquet(c_path))
    hn = ROUND22_2_ROOT / "data_recovery" / "hn2_panel.parquet"
    if hn.is_file():
        frames.append(pd.read_parquet(hn))
    if not frames:
        return cols
    add = frames[0]
    add["trade_date"] = pd.to_datetime(add["trade_date"])
    add["instrument_id"] = add["instrument_id"].astype(str)
    for extra in frames[1:]:
        extra["trade_date"] = pd.to_datetime(extra["trade_date"])
        extra["instrument_id"] = extra["instrument_id"].astype(str)
        more = [c for c in extra.columns if c not in add.columns and c not in {"ann_dt", "known_at", "available_at", "stale"}]
        if more:
            add = add.merge(extra[["trade_date", "instrument_id", *more]], on=["trade_date", "instrument_id"], how="left")
    cols = [c for c in add.columns if c.startswith("r22_") or c.startswith("r22_2_")]
    add = add[["trade_date", "instrument_id", *cols]].drop_duplicates(["trade_date", "instrument_id"])
    for name in ("five_training.parquet", "five_candidate.parquet", "ten_training.parquet", "ten_candidate.parquet"):
        path = DATA / name
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            frame = frame.merge(add[["trade_date", "instrument_id", *missing]], on=["trade_date", "instrument_id"], how="left")
            frame.to_parquet(path, index=False)
    return cols


def _filter_matched(dates: pd.Series | None) -> None:
    if dates is None or dates.empty:
        return
    keep = set(pd.to_datetime(dates).dt.normalize())
    for name in PANEL_NAMES:
        path = DATA / name
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        col = "trade_date" if "trade_date" in frame.columns else ("signal_date" if "signal_date" in frame.columns else None)
        if col is None:
            continue
        frame[col] = pd.to_datetime(frame[col])
        frame = frame.loc[frame[col].dt.normalize().isin(keep)].copy()
        frame.to_parquet(path, index=False)


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


def _import_e0(spec: dict[str, Any], five: list[str], ten: list[str], *, allow: bool) -> str | None:
    if spec["code"] != E0_CODE or not allow:
        return None
    if list(five) != list(FIVE_FEATURES) or list(ten) != list(TEN_FEATURES):
        return None
    dest = exp_dir(spec["code"])
    dest.mkdir(parents=True, exist_ok=True)
    for src in E0_IMPORT_CANDIDATES:
        feat = load_json(src / "feature_list.json")
        if list(feat.get("five") or []) != list(five) or list(feat.get("ten") or []) != list(ten):
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
                "reason": "feature hash matches v5; not alias ADD; full-history only",
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
        "status": "OOF",
        "origin": origin,
        "formal": bool(spec.get("formal", True)),
        **ident,
        "five_oof": five_oof,
        "ten_oof": ten_oof,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "filled_with_zero": False,
        "alias_from": None,
    }
    dump_json(dest / "oof.json", payload)
    return payload


def _panel_year_span() -> tuple[int, int]:
    years: list[int] = []
    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate"):
        frame = _G.get(key)
        if frame is None or "trade_date" not in frame.columns:
            continue
        y = pd.to_datetime(frame["trade_date"]).dt.year.dropna()
        if len(y):
            years.append(int(y.min()))
            years.append(int(y.max()))
    if not years:
        return 2018, 2026
    return min(years), max(years)


def _train_both_matched(spec: dict[str, Any], five: list[str], ten: list[str], dest, *, first_year: int, last_year: int, resume: bool) -> dict[str, Any]:
    """Walk-forward 首年随 matched 样本平移，禁止对空的 2018 折硬训。"""
    dest.mkdir(parents=True, exist_ok=True)

    def _head(name: str, training, candidates, features, head):
        extra_cols = [head.target, head.label_end_column]
        tr = slim_panel(training, [*KEYS, *features, *extra_cols])
        ca = slim_panel(candidates, [*KEYS, *features, head.target])
        payload = train_head_open(
            training=tr,
            candidates=ca,
            features=features,
            target=head.target,
            label_end_column=head.label_end_column,
            rank_n_bins=head.rank_n_bins,
            eval_metric=head.eval_metric,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE,
            first_prediction_year=int(first_year),
            last_prediction_year=int(last_year),
            output_dir=dest / name,
            seeds=SEEDS,
            n_jobs=XGB_N_JOBS,
            save_models=False,
            resume=resume,
        )
        payload["force_retrain"] = True
        payload["extras"] = list(spec.get(f"{name}_extras") or ())
        payload["drops"] = list(spec.get(f"{name}_drops") or ())
        payload["features"] = list(features)
        payload["first_prediction_year"] = int(first_year)
        payload["last_prediction_year"] = int(last_year)
        _invariance(payload)
        return payload["ensemble"], payload

    five_pred, five_meta = _head("five", _G["five_training"], _G["five_candidate"], five, FIVE_HEAD)
    ten_pred, ten_meta = _head("ten", _G["ten_training"], _G["ten_candidate"], ten, TEN_HEAD)
    return {
        "five_pred": five_pred,
        "ten_pred": ten_pred,
        "five": {k: v for k, v in five_meta.items() if k not in {"ensemble", "seed_frames"}},
        "ten": {k: v for k, v in ten_meta.items() if k not in {"ensemble", "seed_frames"}},
        "five_new": new_columns(FIVE_FEATURES, spec.get("five_extras") or ()),
        "ten_new": new_columns(TEN_FEATURES, spec.get("ten_extras") or ()),
        "status": "trained_oos",
    }


def train_one(spec: dict[str, Any], *, resume: bool = True, allow_import_e0: bool = False, first_year: int = 2018, last_year: int = 2026) -> dict[str, Any]:
    if spec.get("alias_from"):
        raise RuntimeError("alias ADD is forbidden")
    five, ten = _lists(spec)
    dest = exp_dir(spec["code"])
    if resume and (dest / "oof.json").is_file() and (dest / "five_pred.parquet").is_file():
        existing = load_json(dest / "oof.json")
        if existing.get("five_feature_set_hash") == feature_set_hash(five) and existing.get("ten_feature_set_hash") == feature_set_hash(ten):
            existing["resumed"] = True
            return existing
    src = _import_e0(spec, five, ten, allow=allow_import_e0)
    if src:
        return _finish(spec, five, ten, IMPORTED_BY_HASH)
    _bind()
    missing_five = [c for c in five if c not in _G["five_training"].columns]
    missing_ten = [c for c in ten if c not in _G["ten_training"].columns]
    if missing_five or missing_ten:
        raise RuntimeError(f"{spec['code']} missing columns five={missing_five} ten={missing_ten}")
    trained = _train_both_matched(spec, five, ten, dest, first_year=first_year, last_year=last_year, resume=resume)
    if trained.get("five_pred") is None or trained.get("ten_pred") is None:
        raise RuntimeError(f"{spec['code']} train returned empty predictions: {trained.get('status')}")
    trained["five_pred"].to_parquet(dest / "five_pred.parquet", index=False)
    trained["ten_pred"].to_parquet(dest / "ten_pred.parquet", index=False)
    return _finish(spec, five, ten, TRAINED_REAL)


def _spec(code: str, role: str, fid: str | None, five_extras=(), ten_extras=(), five_drops=(), ten_drops=(), formal: bool = True) -> dict[str, Any]:
    return {
        "code": code,
        "role": role,
        "factor_id": fid,
        "five_extras": tuple(five_extras),
        "ten_extras": tuple(ten_extras),
        "five_drops": tuple(five_drops),
        "ten_drops": tuple(ten_drops),
        "formal": formal,
        "alias_from": None,
    }


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


def _eligible_from_pit(pit: dict[str, Any], ev: dict[str, Any], disc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    blocked = {"RESEARCH_ONLY_BACKFILL", "BLOCKED_PIT", "BLOCKED_DATA", "BLOCKED"}
    c1 = pit.get("c1") or {}
    c4 = pit.get("c4") or {}
    if c1.get("pit_status") not in blocked and c1.get("model_id"):
        rows.append(
            {
                "factor_id": c1["model_id"],
                "family": "C",
                "five_nearest": (ev.get("c1_five") or {}).get("nearest"),
                "five_abs_rho": (ev.get("c1_five") or {}).get("abs_rho"),
                "five_action": (ev.get("c1_five") or {}).get("action") or "ADD_ONLY",
                "ten_nearest": (ev.get("c1_ten") or {}).get("nearest"),
                "ten_abs_rho": (ev.get("c1_ten") or {}).get("abs_rho"),
                "ten_action": (ev.get("c1_ten") or {}).get("action") or "ADD_ONLY",
                "model_eligible": True,
            }
        )
    if c4.get("pit_status") not in blocked:
        rows.append(
            {
                "factor_id": C4_ID,
                "family": "C",
                "five_nearest": (ev.get("c4_five") or {}).get("nearest"),
                "five_abs_rho": (ev.get("c4_five") or {}).get("abs_rho"),
                "five_action": (ev.get("c4_five") or {}).get("action") or "ADD_ONLY",
                "ten_nearest": (ev.get("c4_ten") or {}).get("nearest"),
                "ten_abs_rho": (ev.get("c4_ten") or {}).get("abs_rho"),
                "ten_action": (ev.get("c4_ten") or {}).get("action") or "ADD_ONLY",
                "model_eligible": True,
            }
        )
    hn2 = (disc.get("hn") or {}).get("hn2") or {}
    if hn2.get("status") == "PIT_PASS" and hn2.get("model_eligible"):
        rows.append(
            {
                "factor_id": "r22_2_hn2_avg_holding_chg",
                "family": "HN",
                "five_nearest": None,
                "five_abs_rho": None,
                "five_action": "ADD_ONLY",
                "ten_nearest": None,
                "ten_abs_rho": None,
                "ten_action": "ADD_ONLY",
                "model_eligible": True,
            }
        )
    return rows


def _registry(corr_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [_spec(E0_CODE, "E0_CONTROL", None)]
    ledger = []
    for row in corr_rows:
        fid = row["factor_id"]
        base = {"factor_id": fid, "five_nearest": row.get("five_nearest"), "ten_nearest": row.get("ten_nearest"), "five_abs_rho": row.get("five_abs_rho"), "ten_abs_rho": row.get("ten_abs_rho")}
        if not row.get("model_eligible"):
            continue
        specs += [
            _spec(f"R222_E1_{fid}", "E1_FIVE_ADD", fid, five_extras=(fid,)),
            _spec(f"R222_E2_{fid}", "E2_TEN_ADD", fid, ten_extras=(fid,)),
            _spec(f"R222_E3_{fid}", "E3_BOTH_ADD", fid, five_extras=(fid,), ten_extras=(fid,)),
        ]
        for role in ("E1", "E2", "E3"):
            ledger.append({**base, "role": role, "code": f"R222_{role}_{fid}", "status": "QUEUED", "formal": True})
        if row.get("five_action") == "REMOVE_REPLACE_ALLOWED" and row.get("five_nearest"):
            drop = (row["five_nearest"],)
            specs += [
                _spec(f"R222_E4_{fid}", "E4_FIVE_REMOVE", fid, five_drops=drop),
                _spec(f"R222_E5_{fid}", "E5_FIVE_REPLACE", fid, five_extras=(fid,), five_drops=drop),
            ]
            ledger += [
                {**base, "role": "E4", "code": f"R222_E4_{fid}", "status": "QUEUED", "formal": True},
                {**base, "role": "E5", "code": f"R222_E5_{fid}", "status": "QUEUED", "formal": True},
            ]
        else:
            status = EXPLORATORY_DIAGNOSTIC if row.get("five_action") == EXPLORATORY_DIAGNOSTIC else NOT_APPLICABLE
            ledger += [
                {**base, "role": "E4", "code": "", "status": status, "formal": False},
                {**base, "role": "E5", "code": "", "status": status, "formal": False},
            ]
        if row.get("ten_action") == "REMOVE_REPLACE_ALLOWED" and row.get("ten_nearest"):
            drop = (row["ten_nearest"],)
            specs += [
                _spec(f"R222_E6_{fid}", "E6_TEN_REMOVE", fid, ten_drops=drop),
                _spec(f"R222_E7_{fid}", "E7_TEN_REPLACE", fid, ten_extras=(fid,), ten_drops=drop),
            ]
            ledger += [
                {**base, "role": "E6", "code": f"R222_E6_{fid}", "status": "QUEUED", "formal": True},
                {**base, "role": "E7", "code": f"R222_E7_{fid}", "status": "QUEUED", "formal": True},
            ]
        else:
            ledger += [
                {**base, "role": "E6", "code": "", "status": NOT_APPLICABLE, "formal": False},
                {**base, "role": "E7", "code": "", "status": NOT_APPLICABLE, "formal": False},
            ]
    return specs, ledger


def run_train(*, resume: bool = True, jobs: int = 4, **_: Any) -> dict[str, Any]:
    del jobs
    require_frozen_gate()
    if resume and TRAIN_JSON.is_file():
        existing = load_json(TRAIN_JSON)
        if existing.get("status") in {"DONE", "PARTIAL", INSUFFICIENT_HISTORY, NON_CANONICAL_EXPLORATORY}:
            return existing
    pit = load_json(ROUND22_2_ROOT / "pit" / "c_pit.json")
    ev = load_json(ROUND22_2_ROOT / "evidence" / "c14.json")
    disc = load_json(ROUND22_2_ROOT / "data_recovery" / "discover.json")
    matched = load_json(ROUND22_2_ROOT / "matched" / "matched_sample.json")
    union = pit.get("status")
    allow_import = union == FULL_HISTORY_STRICT_PIT
    both = (matched.get("both") or pit.get("matched_both") or {})
    c1m = matched.get("c1") or pit.get("matched_c1") or {}
    if (both.get("n_dates") or 0) == 0 and c1m.get("n_dates"):
        both = {**c1m, "path": c1m.get("dates_path"), "insufficient_history": c1m.get("insufficient_history")}
    if both.get("insufficient_history") and union not in {FULL_HISTORY_STRICT_PIT}:
        payload = {
            "status": INSUFFICIENT_HISTORY,
            "reason": "matched train window < 3y; refuse formal compare to 2018-2026 v5",
            "n_trained_real": 0,
            "n_formal": 0,
            "compare_mixed_full_v5": False,
        }
        dump_json(TRAIN_JSON, payload)
        write_md(MD["05"], f"# Round22.2 · 05 真训\n\n`{INSUFFICIENT_HISTORY}`。不足 3 年训练。不与 2018–2026 v5 Sharpe 混表。\n")
        write_md(MD["13"], f"# Round22.2 · 13 E0–E9\n\n未排队。原因 `{INSUFFICIENT_HISTORY}`。\n")
        write_md(MD["14"], f"# Round22.2 · 14 OOF\n\n未跑。`{INSUFFICIENT_HISTORY}`。\n")
        return payload
    if union in {"RESEARCH_ONLY_BACKFILL", "BLOCKED_PIT", "BLOCKED"}:
        payload = {
            "status": NON_CANONICAL_EXPLORATORY if union == "RESEARCH_ONLY_BACKFILL" else "BLOCKED",
            "reason": f"C PIT `{union}`；不进正式全历史模型",
            "n_trained_real": 0,
            "n_formal": 0,
        }
        dump_json(TRAIN_JSON, payload)
        write_md(MD["05"], f"# Round22.2 · 05 真训\n\nC PIT `{union}`，不训正式模型。\n")
        write_md(MD["13"], "# Round22.2 · 13 E0–E9\n\n未排队。\n")
        write_md(MD["14"], "# Round22.2 · 14 OOF\n\n未跑。\n")
        return payload
    _copy_base_panels()
    _merge_new_factors()
    dates = None
    if union != FULL_HISTORY_STRICT_PIT:
        path = both.get("path") or (ROUND22_2_ROOT / "matched" / "matched_dates_c1_c4.parquet")
        if Path(str(path)).is_file():
            dates = pd.to_datetime(pd.read_parquet(path)["trade_date"])
            _filter_matched(dates)
    _bind()
    min_year, max_year = _panel_year_span()
    first_year = 2018 if allow_import else min_year + MIN_TRAIN_YEARS
    last_year = max_year
    if first_year > last_year:
        payload = {
            "status": INSUFFICIENT_HISTORY,
            "reason": f"matched years {min_year}-{max_year} cannot form 3y walk-forward",
            "n_trained_real": 0,
            "n_formal": 0,
            "compare_mixed_full_v5": False,
        }
        dump_json(TRAIN_JSON, payload)
        write_md(MD["05"], f"# Round22.2 · 05 真训\n\n`{INSUFFICIENT_HISTORY}`。{payload['reason']}。\n")
        return payload
    rows = _eligible_from_pit(pit, ev, disc)
    specs, ledger = _registry(rows)
    trained = []
    for spec in specs:
        print(f"[r222 train] {spec['code']} pred={first_year}-{last_year}", flush=True)
        trained.append(
            train_one(
                spec,
                resume=resume,
                allow_import_e0=allow_import and spec["code"] == E0_CODE,
                first_year=first_year,
                last_year=last_year,
            )
        )
    e0 = next(r for r in trained if r["code"] == E0_CODE)
    if allow_import and e0.get("origin") != IMPORTED_BY_HASH:
        # 全历史才要求导入；找不到源则诚实真训 matched/full
        pass
    if not allow_import and e0.get("origin") == IMPORTED_BY_HASH:
        raise RuntimeError("matched sample must retrain E0; refuse imported full-history preds")
    gates = []
    for row in trained:
        gate = {"code": row["code"], "role": row.get("role"), "factor_id": row.get("factor_id"), "origin": row.get("origin")}
        if row["code"] == E0_CODE:
            gate["oof_gate"] = {"passed": False, "reason": "control"}
        else:
            gate["oof_gate"] = _oof_gate(row, e0)
        gate["five_oof"] = (row.get("five_oof") or {}).get("oof_rankic_mean")
        gate["ten_oof"] = (row.get("ten_oof") or {}).get("oof_rankic_mean")
        gates.append(gate)
    by_code = {g["code"]: g for g in gates}
    for item in ledger:
        if item.get("code") in by_code:
            item["status"] = "OOF"
            item["oof_pass"] = bool((by_code[item["code"]].get("oof_gate") or {}).get("passed"))
    # 联合 Add：C1 与 C4 都过增量门才预注册，禁止失败后无限组合
    c1_id = (pit.get("c1") or {}).get("model_id")
    c1_pass = any(g.get("factor_id") == c1_id and (g.get("oof_gate") or {}).get("passed") for g in gates)
    c4_pass = any(g.get("factor_id") == C4_ID and (g.get("oof_gate") or {}).get("passed") for g in gates)
    joint = None
    if c1_pass and c4_pass:
        extras = (c1_id, C4_ID)
        joint_specs = [
            _spec("R222_E1_c1_c4_joint", "E1_FIVE_ADD_JOINT", "c1_c4", five_extras=extras),
            _spec("R222_E2_c1_c4_joint", "E2_TEN_ADD_JOINT", "c1_c4", ten_extras=extras),
            _spec("R222_E3_c1_c4_joint", "E3_BOTH_ADD_JOINT", "c1_c4", five_extras=extras, ten_extras=extras),
        ]
        joint = joint_specs[-1]
        print("[r222 train] c1_c4 joint E1/E2/E3 pre-registered (E3 requires E1)", flush=True)
        for jspec in joint_specs:
            jrow = train_one(jspec, resume=resume, allow_import_e0=False, first_year=first_year, last_year=last_year)
            trained.append(jrow)
            jg = {
                "code": jrow["code"],
                "role": jrow.get("role"),
                "factor_id": "c1_c4",
                "origin": jrow.get("origin"),
                "oof_gate": _oof_gate(jrow, e0),
                "five_oof": (jrow.get("five_oof") or {}).get("oof_rankic_mean"),
                "ten_oof": (jrow.get("ten_oof") or {}).get("oof_rankic_mean"),
                "pre_registered": True,
            }
            gates.append(jg)
    write_csv(REG_CSV, ledger)
    write_csv(OOF_CSV, gates)
    write_csv(
        ROUND22_2_ROOT / "model" / "trial_registry.csv",
        [
            {
                "trial_id": g.get("code"),
                "role": g.get("role"),
                "factor_id": g.get("factor_id"),
                "origin": g.get("origin"),
                "oof_pass": (g.get("oof_gate") or {}).get("passed"),
                "round": "22.2",
            }
            for g in gates
        ],
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
        "sample": "full_history" if allow_import else "matched_pit",
        "pit_status": union,
        "n_specs": len(trained),
        "n_imported": int(sum(r.get("origin") == IMPORTED_BY_HASH for r in trained)),
        "n_trained_real": int(sum(r.get("origin") == TRAINED_REAL for r in trained)),
        "n_oof_pass": int(sum(bool((g.get("oof_gate") or {}).get("passed")) for g in gates)),
        "e0_origin": e0.get("origin"),
        "first_prediction_year": first_year,
        "last_prediction_year": last_year,
        "panel_year_span": [min_year, max_year],
        "joint_pre_registered": bool(joint),
        "compare_mixed_full_v5": False,
        "alias_add": False,
        "gates": gates,
    }
    dump_json(TRAIN_JSON, payload)
    write_md(MD["05"], f"# Round22.2 · 05 真训\n\nE0 origin=`{e0.get('origin')}` sample=`{payload['sample']}`。真训={payload['n_trained_real']} 导入={payload['n_imported']}。禁止 alias ADD。\n")
    write_md(MD["13"], f"# Round22.2 · 13 E0–E9\n\n排队={payload['n_specs']}。Replace 仅当训练折 |ρ|≥0.60。联合 Add 预注册=`{payload['joint_pre_registered']}`。\n")
    write_md(MD["14"], f"# Round22.2 · 14 OOF\n\n对照 `{E0_CODE}`。入围={payload['n_oof_pass']}。有 E3 必有 E1。\n")
    return payload
