"""四路真实 XGB WF。five_drops/ten_drops 真删。禁止 alias ADD。"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from ..round20.train import train_both_heads, unique_keep
from ..round21.runner import oof_rankic
from .common import dump_json, feature_set_hash, file_sha256, load_json, sha256_json, write_md
from .contract import (
    CLUSTER_SPEARMAN_ABS,
    COLSAMPLE,
    E0_CODE,
    FIVE_ONLY_FEATURE,
    INCREMENT_GATE,
    MD,
    PREDICT_YEARS,
    SHARED_13,
    SUBSAMPLE,
    XGB_N_JOBS,
    exp_dir,
    model_parameters,
)
from .panels import _G, init_heads, pit_safe_factors, prepare_head_panels

from ..paths import ROUND21_2_ROOT

MODEL_JSON = ROUND21_2_ROOT / "model" / "model_selection.json"
MATRIX_JSON = ROUND21_2_ROOT / "configs" / "model_matrix.json"
OOF_CSV = ROUND21_2_ROOT / "OOF_model_results.csv"


def _fold_map() -> dict[str, Any]:
    return load_json(ROUND21_2_ROOT / "correlation" / "fold_nearest.json")


def _majority_and_folds(feat: str) -> tuple[str | None, dict[str, str], bool, list[str]]:
    nearest = _fold_map().get(feat) or {}
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
                "code": f"R21_2_E1_{feat}",
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
                "head": "both_90_10",
                "alias_from": None,
            }
        )
        specs.append(
            {
                "code": f"R21_2_E2_{feat}",
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
        specs.append(
            {
                "code": f"R21_2_E3_{feat}",
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
                "alias_from": None,
            }
        )
        cluster_members = [majority] + [c for c in cluster_old if c != majority] if majority else list(cluster_old)
        cluster_members = [c for c in unique_keep(cluster_members) if c in FIVE_FEATURES or c in TEN_FEATURES]
        if len(cluster_members) >= 2:
            specs.append(
                {
                    "code": f"R21_2_E4_{feat}",
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
                    "code": f"R21_2_E5_{feat}",
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
    dump_json(dest / "model_hash.json", {"cache_key": ident.get("cache_key"), "five_feature_set_hash": ident.get("five_feature_set_hash"), "ten_feature_set_hash": ident.get("ten_feature_set_hash")})
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
    init_heads()
    if spec.get("per_fold_drops"):
        from ..round17.ensemble import ensemble_rank_pct
        from ..round17.train import complete_cases, ranker_config_from_params, rolling_train_predict_open
        from ..round20.panel import fold_masks

        def _head(name, training, candidates, locked, extras, fold_drops, head):
            frames = []
            for seed in (17, 42, 73, 101, 202):
                preds = []
                years = [y for y in PREDICT_YEARS if int(candidates["trade_date"].dt.year.eq(y).sum()) > 0]
                for y in years:
                    drops = list(fold_drops.get(str(y)) or spec.get(f"{name}_drops") or ())
                    features = [x for x in unique_keep([*locked, *extras]) if x not in set(drops)]
                    train_m, _ = fold_masks(training, predict_year=y, label=head.target)
                    tr, _ = complete_cases(training.loc[train_m], features, require_target=True, target=head.target)
                    te, _ = complete_cases(candidates.loc[candidates["trade_date"].dt.year.eq(y)], features, require_target=False, target=head.target)
                    if tr.empty or te.empty:
                        continue
                    params = model_parameters(int(seed), n_jobs=n_jobs)
                    params["eval_metric"] = head.eval_metric
                    params["rank_n_bins"] = head.rank_n_bins
                    part, _ = rolling_train_predict_open(
                        tr, te, None, features=features, first_prediction_year=y, last_prediction_year=y,
                        config=ranker_config_from_params(params), target=head.target, label_end_column=head.label_end_column,
                        save_models=False, pre_cleaned=True, clean_audit=({}, {}),
                    )
                    preds.append(part)
                if preds:
                    frames.append(pd.concat(preds, ignore_index=True))
            return ensemble_rank_pct(frames)

        five = _head("five", _G["five_training"], _G["five_candidate"], FIVE_FEATURES, list(spec.get("five_extras") or ()), spec.get("fold_five_drops") or {}, FIVE_HEAD)
        ten = _head("ten", _G["ten_training"], _G["ten_candidate"], TEN_FEATURES, list(spec.get("ten_extras") or ()), spec.get("fold_ten_drops") or {}, TEN_HEAD)
        return _finish(spec, dest, five, ten, five_feat, ten_feat, ident)
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
        print(f"[round21.2] TRAIN {spec['code']} role={spec.get('role')} drops5={spec.get('five_drops')} drops10={spec.get('ten_drops')}", flush=True)
        return train_one(spec, resume=resume, n_jobs=xgb_jobs)

    if int(jobs) <= 1:
        for spec in specs:
            rows.append(_run(spec))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
            futs = {pool.submit(_run, spec): spec["code"] for spec in specs}
            order = {spec["code"]: i for i, spec in enumerate(specs)}
            got = {futs[fut]: fut.result() for fut in as_completed(futs)}
            rows = [got[c] for c, _ in sorted(order.items(), key=lambda t: t[1]) if c in got]
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
    }
    dump_json(MODEL_JSON, payload)
    lines = ["# Round21.2 · 14 模型增量 / 四路训练", "", f"N=`{len(rows)}` increment_pass=`{payload['n_increment_pass']}` 禁止 alias ADD。", ""]
    write_md(MD["14"], "\n".join(lines))
    write_md(MD["13"], "# Round21.2 · 13 Nested Walk-Forward\n\n3y 训 / 1y 测。相关、簇、变换均在训练折内。FIXED 与 NESTED 都不是独立 OOS。\n")
    try:
        from .conditional import run_conditional

        run_conditional(resume=True)
    except Exception as exc:
        write_md(MD["10"], f"# Round21.2 · 10 Conditional IC\n\nPARTIAL: {exc}\n")
    return payload
