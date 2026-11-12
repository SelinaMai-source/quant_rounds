"""E0–E9。E0 仅在特征哈希与 v5 一致时 IMPORTED_BY_HASH。Add 真训。禁止 alias ADD。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND21_2_ROOT, ROUND21_3_ROOT, ROUND21_4_ROOT, ROUND22_ROOT
from ..round20.train import train_both_heads, unique_keep
from ..round21.runner import oof_rankic
from ..round21_2.panels import _G
from .common import dump_json, feature_set_hash, file_sha256, load_json, sha256_json, write_csv, write_md
from .contract import (
    COLSAMPLE,
    COMPOSITE_OF_KNOWN_FACTORS,
    E0_CODE,
    E0_IMPORT_CANDIDATES,
    EXPLORATORY_DIAGNOSTIC,
    FIVE_FEATURES,
    FIVE_HEAD,
    IMPORTED_BY_HASH,
    MD,
    NOT_APPLICABLE,
    OOF_INCREMENT_GATE,
    SUBSAMPLE,
    TEN_FEATURES,
    TEN_HEAD,
    TRAINED_REAL,
    XGB_N_JOBS,
    exp_dir,
    model_parameters,
)
from .freeze import require_frozen_gate

DATA = ROUND22_ROOT / "data"
TRAIN_JSON = ROUND22_ROOT / "model" / "train.json"
REG_CSV = ROUND22_ROOT / "model" / "experiment_registry.csv"
OOF_CSV = ROUND22_ROOT / "model" / "oof_results.csv"
HASH_CSV = ROUND22_ROOT / "model" / "hash_ledger.csv"
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
    roots = [ROUND21_4_ROOT / "data", ROUND21_3_ROOT / "data", ROUND21_2_ROOT / "data"]
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


def _merge_factors() -> list[str]:
    panel = pd.read_parquet(ROUND22_ROOT / "data" / "factor_panel.parquet")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    cols = [c for c in panel.columns if c.startswith("r22_") and c != "r22_e_status" and not c.startswith("r22_aux")]
    add = panel[["trade_date", "instrument_id", *cols]].drop_duplicates(["trade_date", "instrument_id"])
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


def _bind() -> None:
    _copy_base_panels()
    _merge_factors()
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


def _import_e0(spec: dict[str, Any], five: list[str], ten: list[str]) -> str | None:
    if spec["code"] != E0_CODE:
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
                "reason": "feature hash matches v5; not alias ADD",
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
    dump_json(dest / "model_config.json", model_parameters(17))
    dump_json(dest / "model_hash.json", {"cache_key": ident["cache_key"], "five_feature_set_hash": ident["five_feature_set_hash"], "ten_feature_set_hash": ident["ten_feature_set_hash"]})
    dump_json(dest / "prediction_hash.json", {"five_pred": file_sha256(dest / "five_pred.parquet"), "ten_pred": file_sha256(dest / "ten_pred.parquet")})
    five_oof = oof_rankic(five_pred, _G["five_candidate"], target=FIVE_HEAD.target)
    ten_oof = oof_rankic(ten_pred, _G["ten_candidate"], target=TEN_HEAD.target)
    payload = {
        "code": spec["code"],
        "role": spec.get("role"),
        "factor_id": spec.get("factor_id"),
        "profile": spec.get("profile", "A"),
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


def train_one(spec: dict[str, Any], *, resume: bool = True) -> dict[str, Any]:
    if spec.get("alias_from"):
        raise RuntimeError("alias ADD is forbidden")
    five, ten = _lists(spec)
    dest = exp_dir(spec["code"])
    if resume and (dest / "oof.json").is_file() and (dest / "five_pred.parquet").is_file():
        existing = load_json(dest / "oof.json")
        if existing.get("five_feature_set_hash") == feature_set_hash(five) and existing.get("ten_feature_set_hash") == feature_set_hash(ten):
            existing["resumed"] = True
            return existing
    src = _import_e0(spec, five, ten)
    if src:
        return _finish(spec, five, ten, IMPORTED_BY_HASH)
    _bind()
    missing_five = [c for c in five if c not in _G["five_training"].columns]
    missing_ten = [c for c in ten if c not in _G["ten_training"].columns]
    if missing_five or missing_ten:
        raise RuntimeError(f"{spec['code']} missing columns five={missing_five} ten={missing_ten}")
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
        five_frozen=_G.get("five_frozen"),
        ten_frozen=_G.get("ten_frozen"),
        force_retrain=True,
        n_jobs=XGB_N_JOBS,
    )
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
        "profile": "A",
        "five_extras": tuple(five_extras),
        "ten_extras": tuple(ten_extras),
        "five_drops": tuple(five_drops),
        "ten_drops": tuple(ten_drops),
        "formal": formal,
        "alias_from": None,
    }


def _registry(corr_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [_spec(E0_CODE, "E0_CONTROL", None)]
    ledger = []
    for row in corr_rows:
        fid = row["factor_id"]
        eligible = bool(row.get("model_eligible")) and row.get("profile") == "A" and row.get("label") != "blocked"
        base = {"factor_id": fid, "profile": row.get("profile"), "five_nearest": row.get("five_nearest"), "ten_nearest": row.get("ten_nearest"), "five_abs_rho": row.get("five_abs_rho"), "ten_abs_rho": row.get("ten_abs_rho")}
        if not eligible:
            for role, action in (
                ("E1", "NOT_TRAINED"),
                ("E2", "NOT_TRAINED"),
                ("E3", "NOT_TRAINED"),
                ("E4", row.get("five_action") or NOT_APPLICABLE),
                ("E5", row.get("five_action") or NOT_APPLICABLE),
                ("E6", row.get("ten_action") or NOT_APPLICABLE),
                ("E7", row.get("ten_action") or NOT_APPLICABLE),
                ("E8", NOT_APPLICABLE),
                ("E9", NOT_APPLICABLE),
            ):
                ledger.append({**base, "role": role, "code": "", "status": action if action != "REMOVE_REPLACE_ALLOWED" else "NOT_TRAINED", "formal": False})
            continue
        specs += [
            _spec(f"R22_E1_{fid}", "E1_FIVE_ADD", fid, five_extras=(fid,)),
            _spec(f"R22_E2_{fid}", "E2_TEN_ADD", fid, ten_extras=(fid,)),
            _spec(f"R22_E3_{fid}", "E3_BOTH_ADD", fid, five_extras=(fid,), ten_extras=(fid,)),
        ]
        for role in ("E1", "E2", "E3"):
            ledger.append({**base, "role": role, "code": f"R22_{role}_{fid}", "status": "QUEUED", "formal": True})
        if row.get("five_action") == "REMOVE_REPLACE_ALLOWED" and row.get("five_nearest"):
            drop = (row["five_nearest"],)
            specs += [
                _spec(f"R22_E4_{fid}", "E4_FIVE_REMOVE", fid, five_drops=drop),
                _spec(f"R22_E5_{fid}", "E5_FIVE_REPLACE", fid, five_extras=(fid,), five_drops=drop),
            ]
            ledger += [
                {**base, "role": "E4", "code": f"R22_E4_{fid}", "status": "QUEUED", "formal": True},
                {**base, "role": "E5", "code": f"R22_E5_{fid}", "status": "QUEUED", "formal": True},
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
                _spec(f"R22_E6_{fid}", "E6_TEN_REMOVE", fid, ten_drops=drop),
                _spec(f"R22_E7_{fid}", "E7_TEN_REPLACE", fid, ten_extras=(fid,), ten_drops=drop),
            ]
            ledger += [
                {**base, "role": "E6", "code": f"R22_E6_{fid}", "status": "QUEUED", "formal": True},
                {**base, "role": "E7", "code": f"R22_E7_{fid}", "status": "QUEUED", "formal": True},
            ]
        else:
            status = EXPLORATORY_DIAGNOSTIC if row.get("ten_action") == EXPLORATORY_DIAGNOSTIC else NOT_APPLICABLE
            ledger += [
                {**base, "role": "E6", "code": "", "status": status, "formal": False},
                {**base, "role": "E7", "code": "", "status": status, "formal": False},
            ]
        if row.get("five_action") == "REMOVE_REPLACE_ALLOWED" and row.get("ten_action") == "REMOVE_REPLACE_ALLOWED":
            specs += [
                _spec(f"R22_E8_{fid}", "E8_BOTH_REMOVE", fid, five_drops=(row["five_nearest"],), ten_drops=(row["ten_nearest"],)),
                _spec(f"R22_E9_{fid}", "E9_BOTH_REPLACE", fid, five_extras=(fid,), ten_extras=(fid,), five_drops=(row["five_nearest"],), ten_drops=(row["ten_nearest"],)),
            ]
            ledger += [
                {**base, "role": "E8", "code": f"R22_E8_{fid}", "status": "QUEUED", "formal": True},
                {**base, "role": "E9", "code": f"R22_E9_{fid}", "status": "QUEUED", "formal": True},
            ]
        else:
            ledger += [
                {**base, "role": "E8", "code": "", "status": NOT_APPLICABLE, "formal": False},
                {**base, "role": "E9", "code": "", "status": NOT_APPLICABLE, "formal": False},
            ]
        if row.get("label") == COMPOSITE_OF_KNOWN_FACTORS:
            for item in ledger:
                if item.get("factor_id") == fid:
                    item["note"] = COMPOSITE_OF_KNOWN_FACTORS
    return specs, ledger


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
        if existing.get("status") == "DONE":
            return existing
    corr = load_json(ROUND22_ROOT / "correlation" / "correlation.json")
    specs, ledger = _registry(corr.get("rows") or [])
    _bind()
    trained = []
    for spec in specs:
        print(f"[r22 train] {spec['code']}", flush=True)
        row = train_one(spec, resume=resume)
        trained.append(row)
    e0 = next(r for r in trained if r["code"] == E0_CODE)
    if e0.get("origin") != IMPORTED_BY_HASH:
        raise RuntimeError("E0 must be IMPORTED_BY_HASH after the v5 feature hash matches")
    if e0.get("five_feature_set_hash") != feature_set_hash(list(FIVE_FEATURES)):
        raise RuntimeError("E0 five hash is not v5")
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
            item["origin"] = by_code[item["code"]].get("origin")
    # 同级：E3 入围则 E1 也保留袖套资格标记，真正回测在 backtest。
    write_csv(REG_CSV, ledger)
    write_csv(OOF_CSV, gates)
    hashes = []
    for row in trained:
        hashes.append(
            {
                "code": row["code"],
                "origin": row.get("origin"),
                "five_feature_set_hash": row.get("five_feature_set_hash"),
                "ten_feature_set_hash": row.get("ten_feature_set_hash"),
                "prediction_five": (load_json(exp_dir(row["code"]) / "prediction_hash.json") or {}).get("five_pred"),
                "prediction_ten": (load_json(exp_dir(row["code"]) / "prediction_hash.json") or {}).get("ten_pred"),
            }
        )
    write_csv(HASH_CSV, hashes)
    payload = {
        "status": "DONE",
        "n_specs": len(specs),
        "n_imported": int(sum(r.get("origin") == IMPORTED_BY_HASH for r in trained)),
        "n_trained_real": int(sum(r.get("origin") == TRAINED_REAL for r in trained)),
        "n_oof_pass": int(sum(bool((g.get("oof_gate") or {}).get("passed")) for g in gates)),
        "e0_origin": e0.get("origin"),
    }
    dump_json(TRAIN_JSON, payload | {"gates": gates})
    write_md(MD["14"], f"# Round22 · 14 实验注册\n\nE0–E9 规则已在冻结门中。实际排队={payload['n_specs']}。导入={payload['n_imported']}。真训={payload['n_trained_real']}。OOF 入围={payload['n_oof_pass']}。\n")
    write_md(MD["15"], f"# Round22 · 15 OOF Add\n\n对照 `{E0_CODE}`。两头都不差且至少一头严格更好，并且两头 OOS 天数 ≥ {OOF_INCREMENT_GATE['min_oos_days']}。入围数={payload['n_oof_pass']}。\n")
    write_md(MD["16"], "# Round22 · 16 Remove / Replace\n\n只有训练折 |ρ|≥0.60 的头才排队 E4–E9。低相关删除保持 EXPLORATORY_DIAGNOSTIC，没有正式袖套。E5 与 E4 成对，E9 与 E8 成对。\n")
    return payload
