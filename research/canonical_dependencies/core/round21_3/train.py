"""四路真实 XGB WF。复用 train_both_heads 的 five_drops/ten_drops。禁止 alias ADD。"""

from __future__ import annotations

import shutil
from typing import Any, Sequence

import pandas as pd

from ..paths import ROUND21_2_ROOT, ROUND21_3_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from ..round20.train import train_both_heads, unique_keep
from ..round21.runner import oof_rankic
from ..round21_2.panels import _G, init_heads, prepare_head_panels
from .common import dump_json, feature_set_hash, file_sha256, load_json, sha256_json, write_md
from .contract import (
    COLSAMPLE,
    E0_CODE,
    INCREMENT_GATE,
    L1_E0,
    L1_E3,
    L1_FACTOR,
    L1_SPECS,
    MD,
    NEW_FACTOR_IDS,
    SUBSAMPLE,
    XGB_N_JOBS,
    exp_dir,
    family_of,
    model_parameters,
)

MODEL_JSON = ROUND21_3_ROOT / "model" / "model_selection.json"
MATRIX_JSON = ROUND21_3_ROOT / "configs" / "model_matrix.json"
L1_MATRIX_JSON = ROUND21_3_ROOT / "configs" / "l1_model_matrix.json"
OOF_CSV = ROUND21_3_ROOT / "OOF_model_results.csv"
L1_TRAIN_JSON = ROUND21_3_ROOT / "model" / "l1_train.json"


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


def _write_ident(dest, spec, five_feat, ten_feat) -> dict[str, Any]:
    ident = _identity(spec, five_feat, ten_feat)
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
    return work.loc[work["rank"].le(2), ["trade_date", "instrument_id", "score", "rank"]].sort_values(["trade_date", "rank"])


def _done(dest, ident: dict[str, Any]) -> dict[str, Any] | None:
    path = dest / "oof.json"
    key_path = dest / "cache_key.json"
    if not path.is_file() or not (dest / "five_pred.parquet").is_file() or not (dest / "ten_pred.parquet").is_file():
        return None
    if key_path.is_file() and load_json(key_path).get("cache_key") != ident.get("cache_key"):
        return None
    payload = load_json(path)
    if payload.get("status") == "OOF" or payload.get("five_oof"):
        payload["resumed"] = True
        return payload
    return None


def _maybe_import_21_2(spec: dict[str, Any], dest, five_feat, ten_feat, ident) -> bool:
    """只读 21.2 已训预测：feature_set_hash 一致才拷入 21.3。不是 alias ADD。"""
    old_code = str(spec["code"]).replace("R21_3_", "R21_2_")
    src = ROUND21_2_ROOT / "candidates" / old_code
    if not (src / "five_pred.parquet").is_file() or not (src / "ten_pred.parquet").is_file():
        return False
    src_feat = load_json(src / "feature_list.json")
    if not src_feat:
        return False
    if list(src_feat.get("five") or []) != list(five_feat):
        return False
    if list(src_feat.get("ten") or []) != list(ten_feat):
        return False
    for name in ("five_pred.parquet", "ten_pred.parquet"):
        shutil.copy2(src / name, dest / name)
    dump_json(
        dest / "imported_from_21_2.json",
        {
            "source": str(src),
            "reason": "identical five/ten feature lists after true drops; not alias ADD",
            "five_feature_set_hash": ident.get("five_feature_set_hash"),
            "ten_feature_set_hash": ident.get("ten_feature_set_hash"),
        },
    )
    return True


def _finish(spec, dest, five, ten, five_feat, ten_feat, ident) -> dict[str, Any]:
    five.to_parquet(dest / "five_pred.parquet", index=False)
    ten.to_parquet(dest / "ten_pred.parquet", index=False)
    daily = pd.concat([five.assign(head="five"), ten.assign(head="ten")], ignore_index=True)
    daily.to_parquet(dest / "daily_prediction.parquet", index=False)
    top = pd.concat([_top12(five).assign(head="five"), _top12(ten).assign(head="ten")], ignore_index=True)
    top.to_parquet(dest / "daily_top1_top2.parquet", index=False)
    dump_json(
        dest / "prediction_hash.json",
        {
            "five_pred": file_sha256(dest / "five_pred.parquet"),
            "ten_pred": file_sha256(dest / "ten_pred.parquet"),
        },
    )
    dump_json(
        dest / "model_hash.json",
        {
            "cache_key": ident.get("cache_key"),
            "five_feature_set_hash": ident.get("five_feature_set_hash"),
            "ten_feature_set_hash": ident.get("ten_feature_set_hash"),
        },
    )
    init_heads()
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
    }
    dump_json(dest / "oof.json", payload)
    return payload


def train_one(spec: dict[str, Any], *, resume: bool = True, n_jobs: int = XGB_N_JOBS) -> dict[str, Any]:
    if spec.get("alias_from"):
        raise RuntimeError(f"{spec['code']} forbids alias_from; Replace must retrain")
    dest = exp_dir(spec["code"])
    dest.mkdir(parents=True, exist_ok=True)
    five_feat = [x for x in unique_keep(list(FIVE_FEATURES) + list(spec.get("five_extras") or ())) if x not in set(spec.get("five_drops") or ())]
    ten_feat = [x for x in unique_keep(list(TEN_FEATURES) + list(spec.get("ten_extras") or ())) if x not in set(spec.get("ten_drops") or ())]
    ident = _write_ident(dest, spec, five_feat, ten_feat)
    if resume:
        existing = _done(dest, ident)
        if existing is not None:
            return existing
        if _maybe_import_21_2(spec, dest, five_feat, ten_feat, ident):
            five = pd.read_parquet(dest / "five_pred.parquet")
            ten = pd.read_parquet(dest / "ten_pred.parquet")
            return _finish(spec, dest, five, ten, five_feat, ten_feat, ident)
    prepare_head_panels(resume=True)
    init_heads()
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


def _assert_e3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["code"]: r for r in rows}
    e0 = by.get(L1_E0) or {}
    e3 = by.get(L1_E3) or {}
    e0_five = set(e0.get("five_features") or [])
    e3_five = set(e3.get("five_features") or [])
    e0_ten = set(e0.get("ten_features") or [])
    e3_ten = set(e3.get("ten_features") or [])
    expected_five = (e0_five - {"g0_amount_log"}) | {L1_FACTOR}
    expected_ten = e0_ten | {L1_FACTOR}
    ok = e3_five == expected_five and e3_ten == expected_ten
    return {
        "assert": "E3 = E0 - {g0_amount_log} + {L1}",
        "passed": bool(ok),
        "e0_five": sorted(e0_five),
        "e3_five": sorted(e3_five),
        "expected_five": sorted(expected_five),
        "e0_ten": sorted(e0_ten),
        "e3_ten": sorted(e3_ten),
        "expected_ten": sorted(expected_ten),
    }


def freeze_l1_matrix() -> dict[str, Any]:
    payload = {
        "status": "FROZEN",
        "n_specs": len(L1_SPECS),
        "specs": list(L1_SPECS),
        "no_alias_add_as_replace": True,
        "fusion": "0.90 ten + 0.10 five",
        "five_only_drop": "g0_amount_log",
    }
    dump_json(L1_MATRIX_JSON, payload)
    return payload


def _run_specs(specs: list[dict[str, Any]], *, resume: bool, jobs: int) -> list[dict[str, Any]]:
    xgb_jobs = 1 if int(jobs) > 1 else XGB_N_JOBS
    rows: list[dict[str, Any]] = []

    def _run(spec: dict[str, Any]) -> dict[str, Any]:
        print(f"[round21.3] TRAIN {spec['code']} role={spec.get('role')} drops5={spec.get('five_drops')} drops10={spec.get('ten_drops')}", flush=True)
        return train_one(spec, resume=resume, n_jobs=xgb_jobs)

    if int(jobs) <= 1:
        for spec in specs:
            rows.append(_run(spec))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=min(int(jobs), len(specs))) as pool:
            futs = {pool.submit(_run, spec): spec["code"] for spec in specs}
            order = {spec["code"]: i for i, spec in enumerate(specs)}
            got = {futs[fut]: fut.result() for fut in as_completed(futs)}
            rows = [got[c] for c, _ in sorted(order.items(), key=lambda t: t[1]) if c in got]
    return rows


def run_l1_train(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    if resume and L1_TRAIN_JSON.is_file():
        existing = load_json(L1_TRAIN_JSON)
        if existing.get("status") == "DONE":
            return existing
    prepare_head_panels(resume=True)
    init_heads()
    freeze_l1_matrix()
    rows = _run_specs(list(L1_SPECS), resume=resume, jobs=jobs)
    assertion = _assert_e3(rows)
    if not assertion.get("passed"):
        raise RuntimeError(f"E3 feature-set assertion failed: {assertion}")
    hashes = {r["code"]: {"five": r.get("five_feature_set_hash"), "ten": r.get("ten_feature_set_hash"), "cache": r.get("cache_key")} for r in rows}
    payload = {
        "status": "DONE",
        "n_oof_models": len(rows),
        "e3_assert": assertion,
        "hashes": hashes,
        "no_alias_add": True,
        "rows": rows,
    }
    dump_json(L1_TRAIN_JSON, payload)
    write_md(
        MD["14"],
        "# Round21.3 · 14 L1 四路训练\n\n"
        f"N=`{len(rows)}` E3断言=`{assertion.get('passed')}` 禁止 alias ADD。g0_amount_log 仅 Five 删除。\n",
    )
    return payload


def _pit_safe() -> list[str]:
    from .pit import PIT_JSON

    pit = load_json(PIT_JSON)
    return list(pit.get("pit_pass") or [])


def _discovery_factors() -> list[str]:
    from .contract import RESEARCH_ONLY
    from .pit import PIT_JSON

    pit = load_json(PIT_JSON)
    ids = list(pit.get("pit_pass") or [])
    for row in pit.get("rows") or []:
        if row.get("pit_status") == RESEARCH_ONLY and row.get("factor_id"):
            ids.append(row["factor_id"])
    return [f for f in dict.fromkeys(ids) if f and f != L1_FACTOR]


def freeze_discovery_matrix(*, resume: bool = True) -> dict[str, Any]:
    if resume and MATRIX_JSON.is_file():
        existing = load_json(MATRIX_JSON)
        if existing.get("discovery_frozen"):
            return existing
    from .correlate import FOLD_JSON

    fold = load_json(FOLD_JSON) if FOLD_JSON.is_file() else {}
    nearest = fold.get("nearest") or {}
    # 正式四路只对 PIT_PASS；RESEARCH_ONLY / DUPLICATE 不进 discovery 训练
    factors = [f for f in _pit_safe() if f != L1_FACTOR]
    specs = [dict(s) for s in L1_SPECS if s["code"] == E0_CODE]
    for feat in factors:
        nb = nearest.get(feat) or {}
        majority = nb.get("majority_legacy")
        five_drop = (majority,) if majority in FIVE_FEATURES else ()
        ten_drop = (majority,) if majority in TEN_FEATURES else ()
        five_only = majority == "g0_amount_log"
        specs.append(
            {
                "code": f"R21_3_E1_{feat}",
                "role": "E1_REMOVE",
                "factor_id": feat,
                "five_extras": (),
                "ten_extras": (),
                "five_drops": five_drop,
                "ten_drops": ten_drop if not five_only else (),
                "formal": True,
                "head": "both_90_10",
                "nearest_legacy": majority,
                "alias_from": None,
                "force_retrain": True,
            }
        )
        specs.append(
            {
                "code": f"R21_3_E2_{feat}",
                "role": "E2_ADD",
                "factor_id": feat,
                "five_extras": (feat,),
                "ten_extras": (feat,),
                "five_drops": (),
                "ten_drops": (),
                "formal": True,
                "head": "both_90_10",
                "nearest_legacy": majority,
                "alias_from": None,
                "force_retrain": True,
            }
        )
        specs.append(
            {
                "code": f"R21_3_E3_{feat}",
                "role": "E3_REPLACE",
                "factor_id": feat,
                "five_extras": (feat,),
                "ten_extras": (feat,) if not five_only else (),
                "five_drops": five_drop,
                "ten_drops": ten_drop if not five_only else (),
                "formal": True,
                "head": "both_90_10",
                "nearest_legacy": majority,
                "alias_from": None,
                "force_retrain": True,
            }
        )
    payload = {
        "status": "FROZEN",
        "discovery_frozen": True,
        "n_specs": len(specs),
        "specs": specs,
        "factors": factors,
        "no_alias_add_as_replace": True,
        "fusion": "0.90 ten + 0.10 five",
    }
    dump_json(MATRIX_JSON, payload)
    return payload


def run_train(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    if resume and MODEL_JSON.is_file():
        existing = load_json(MODEL_JSON)
        if existing.get("status") == "DONE":
            return existing
    prepare_head_panels(resume=True)
    init_heads()
    try:
        from .panels_local import prepare_discovery_panels

        prepare_discovery_panels(resume=True)
        for key in ("five_training", "five_candidate", "ten_training", "ten_candidate"):
            p = ROUND21_3_ROOT / "data" / f"{key}.parquet"
            if p.is_file():
                _G[key] = pd.read_parquet(p)
                _G[key]["trade_date"] = pd.to_datetime(_G[key]["trade_date"])
                _G[key]["instrument_id"] = _G[key]["instrument_id"].astype(str)
    except Exception as exc:
        print(f"[round21.3] discovery panel merge skipped: {exc}", flush=True)
    matrix = freeze_discovery_matrix(resume=True)
    l1_skip = {s["code"] for s in L1_SPECS if s["code"] != E0_CODE}
    need = []
    for spec in matrix.get("specs") or []:
        if spec.get("code") in l1_skip:
            continue
        if spec.get("factor_id") == L1_FACTOR:
            continue
        need.append(spec)
    if not need:
        raise RuntimeError("discovery train: no T/U/V specs after skipping L1")
    rows = _run_specs(need, resume=resume, jobs=jobs)
    by = {r["code"]: r for r in rows}
    base = by.get(E0_CODE) or {}
    increment = []
    for r in rows:
        inc = {"passed": False, "reason": "anchor"} if r.get("code") == E0_CODE else _gate(r, base)
        increment.append({**r, "increment": inc})
    OOF_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "code": r.get("code"),
                "role": r.get("role"),
                "factor_id": r.get("factor_id"),
                "five_oof": (r.get("five_oof") or {}).get("oof_rankic_mean"),
                "ten_oof": (r.get("ten_oof") or {}).get("oof_rankic_mean"),
                "delta_five": (r.get("increment") or {}).get("delta_five"),
                "delta_ten": (r.get("increment") or {}).get("delta_ten"),
                "increment_passed": (r.get("increment") or {}).get("passed"),
                "five_drops": r.get("five_drops"),
                "ten_drops": r.get("ten_drops"),
                "cache_key": r.get("cache_key"),
                "family": family_of(str(r.get("factor_id") or "")),
            }
            for r in increment
        ]
    ).to_csv(OOF_CSV, index=False)
    payload = {
        "status": "DONE",
        "n_oof_models": len(rows),
        "n_increment_pass": sum(1 for r in increment if (r.get("increment") or {}).get("passed")),
        "e0": {"five_oof": base.get("five_oof"), "ten_oof": base.get("ten_oof")},
        "rows": increment,
        "no_alias_add": True,
    }
    dump_json(MODEL_JSON, payload)
    write_md(MD["13"], "# Round21.3 · 13 Nested Walk-Forward\n\n3y 训 / 1y 测。相关、簇、变换均在训练折内。FIXED 与 NESTED 都标 REPEATED_HISTORICAL_TESTING。\n")
    return payload
