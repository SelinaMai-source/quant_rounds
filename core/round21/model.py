"""FIXED_FACTOR_SET_REPLAY vs NESTED_WALK_FORWARD_FEATURE_SELECTION。二者都不是独立 OOS。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..paths import ROUND21_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round17.ensemble import ensemble_rank_pct
from ..round17.train import complete_cases, ranker_config_from_params, rolling_train_predict_open
from ..round20.metrics import rank_ic
from ..round20.panel import fold_masks
from ..round20.train import unique_keep
from .common import dump_json, finite_or_none, load_json, write_md
from .combo import REG_JSON, freeze_combo_registry
from .conditional import COND_CSV
from .contract import (
    A00,
    A01,
    CLUSTER_SPEARMAN_ABS,
    COLSAMPLE,
    FIXED_CODE,
    INCREMENT_GATE,
    MAX_NEW_FACTORS_IN_MAIN,
    MODEL_MD,
    NESTED_CODE,
    SUBSAMPLE,
    XGB_N_JOBS,
    exp_dir,
    model_parameters,
)
from .freeze import require_frozen
from .library import build_research_panel
from .panel import KEYS, ic_window
from .pit import run_pit
from .redundancy import KEEP_CSV
from .runner import init_panels, oof_rankic, prepare_panels, skip_payload, train_oof

MODEL_JSON = ROUND21_ROOT / "model" / "model_selection.json"
MATRIX_JSON = ROUND21_ROOT / "configs" / "model_matrix.json"
OOF_CSV = ROUND21_ROOT / "model" / "oof_increment.csv"
FACTORS_DIR = ROUND21_ROOT / "factors"


def _select_fixed_extras() -> list[str]:
    freeze = freeze_combo_registry(resume=True)
    selected = list(freeze.get("selected_new_factors") or [])
    if selected:
        return selected[: MAX_NEW_FACTORS_IN_MAIN]
    keep = pd.read_csv(KEEP_CSV)
    cand = keep.loc[keep["action"].eq("KEEP") & keep["pit_status"].eq("PIT_PASS")]
    cand = cand.sort_values(["best_abs_icir", "best_abs_mean", "name"], ascending=[False, False, True])
    return [str(x) for x in cand["name"].tolist()[: MAX_NEW_FACTORS_IN_MAIN]]


def _pit_pass() -> list[str]:
    pit = run_pit(resume=True)
    return list(pit.get("pit_pass") or [])


def _cluster_old_for(feat: str) -> list[str]:
    red = load_json(ROUND21_ROOT / "redundancy" / "redundancy.json")
    locked = set(FIVE_FEATURES) | set(TEN_FEATURES)
    out = []
    for cl in red.get("clusters") or []:
        members = cl.get("members") or []
        if feat not in members:
            continue
        for m in members:
            if m != feat and m in locked:
                out.append(m)
    return unique_keep(out)


def freeze_model_matrix(*, resume: bool = True) -> dict[str, Any]:
    require_frozen()
    if MATRIX_JSON.is_file():
        existing = load_json(MATRIX_JSON)
        if existing.get("status") == "FROZEN" and existing.get("frozen_before_model_nav"):
            return existing
    freeze = freeze_combo_registry(resume=True)
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("combo registry not FROZEN")
    selected = _select_fixed_extras()
    pit_pass = _pit_pass()
    specs: list[dict[str, Any]] = [
        {"code": A00, "role": "exact_v5_replay", "extras": (), "five_extras": (), "ten_extras": (), "force_retrain": False, "formal": True, "head": "both_90_10"},
        {"code": A01, "role": "force_retrain_original_14_13", "extras": (), "five_extras": (), "ten_extras": (), "force_retrain": True, "formal": True, "head": "both_90_10"},
        {"code": FIXED_CODE, "role": "FIXED_FACTOR_SET_REPLAY", "extras": tuple(selected), "five_extras": tuple(selected), "ten_extras": tuple(selected), "force_retrain": False, "formal": True, "head": "both_90_10", "independent_oos": False},
        {"code": NESTED_CODE, "role": "NESTED_WALK_FORWARD_FEATURE_SELECTION", "extras": tuple(pit_pass), "five_extras": tuple(pit_pass), "ten_extras": tuple(pit_pass), "force_retrain": False, "formal": True, "head": "both_90_10", "independent_oos": False, "nested": True},
    ]
    for feat in pit_pass:
        extras = (feat,)
        specs.append({"code": f"R21_ADD_{feat}", "role": "add_both", "extras": extras, "five_extras": extras, "ten_extras": extras, "force_retrain": False, "formal": True, "head": "both_90_10", "factor_id": feat})
        specs.append({"code": f"R21_ADD5_{feat}", "role": "add_five_only", "extras": extras, "five_extras": extras, "ten_extras": (), "force_retrain": False, "formal": False, "head": "five_only", "factor_id": feat, "alias_from": f"R21_ADD_{feat}"})
        specs.append({"code": f"R21_ADD10_{feat}", "role": "add_ten_only", "extras": extras, "five_extras": (), "ten_extras": extras, "force_retrain": False, "formal": False, "head": "ten_only", "factor_id": feat, "alias_from": f"R21_ADD_{feat}"})
        olds = _cluster_old_for(feat)
        if olds:
            specs.append({
                "code": f"R21_REP_{feat}",
                "role": "replace_diagnostic",
                "extras": extras,
                "five_extras": extras,
                "ten_extras": extras,
                "force_retrain": False,
                "formal": False,
                "head": "both_90_10",
                "factor_id": feat,
                "replace_of": olds,
                "cannot_delete_locked": True,
                "alias_from": f"R21_ADD_{feat}",
                "note": "锁定 14/13 不删；XGB Replace 与 Add 同列，对照见簇内 Ridge ONAR",
            })
    for feat in selected:
        rest = tuple(x for x in selected if x != feat)
        specs.append({
            "code": f"R21_ABL_{feat}",
            "role": "ablation_fixed",
            "extras": rest,
            "five_extras": rest,
            "ten_extras": rest,
            "force_retrain": bool(not rest),
            "formal": False,
            "head": "both_90_10",
            "factor_id": feat,
        })
    payload = {
        "status": "FROZEN",
        "frozen_before_model_nav": True,
        "post_hoc_best_x_best": False,
        "increment_gate": INCREMENT_GATE,
        "selected_fixed": selected,
        "pit_pass": pit_pass,
        "n_specs": len(specs),
        "specs": specs,
        "both_are_not_independent_oos": True,
    }
    dump_json(MATRIX_JSON, payload)
    return payload


def _alias_head(code: str, src_code: str, *, five_from: str, ten_from: str, spec: dict[str, Any]) -> dict[str, Any]:
    dest = exp_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    src = exp_dir(src_code)
    frozen_five = ROUND21_ROOT / "data" / "five_frozen.parquet"
    frozen_ten = ROUND21_ROOT / "data" / "ten_frozen.parquet"
    five_p = src / "five_pred.parquet" if five_from == "src" else frozen_five
    ten_p = src / "ten_pred.parquet" if ten_from == "src" else frozen_ten
    if not five_p.is_file() or not ten_p.is_file():
        return skip_payload(code, spec, f"missing alias source {src_code}", "SKIP_NO_OOF")
    five = pd.read_parquet(five_p)
    ten = pd.read_parquet(ten_p)
    five.to_parquet(dest / "five_pred.parquet", index=False)
    ten.to_parquet(dest / "ten_pred.parquet", index=False)
    init_panels()
    from .runner import _G

    five_oof = oof_rankic(five, _G["five_candidate"], target=FIVE_HEAD.target)
    ten_oof = oof_rankic(ten, _G["ten_candidate"], target=TEN_HEAD.target)
    payload = {
        "code": code,
        "role": spec.get("role"),
        "status": "OOF",
        "alias_from": src_code,
        "head": spec.get("head"),
        "five_oof": five_oof,
        "ten_oof": ten_oof,
        "five_extras": list(spec.get("five_extras") or ()),
        "ten_extras": list(spec.get("ten_extras") or ()),
        "looked_at_combo_nav": False,
        "independent_oos": False,
        "out_dir": str(dest),
    }
    dump_json(dest / "oof.json", payload)
    dump_json(dest / "summary.json", {**payload, "status": "OOF_ONLY"})
    return payload


def _nested_select(train: pd.DataFrame, *, extras: Sequence[str], locked: Sequence[str], label: str) -> list[str]:
    ranked = []
    for name in extras:
        if name not in train.columns:
            continue
        ric = rank_ic(train[name], train[label])
        if ric is None:
            continue
        ranked.append((abs(float(ric)), float(ric), name))
    ranked.sort(reverse=True)
    chosen: list[str] = []
    controls = list(locked)
    for _, _, name in ranked:
        ok = True
        for c in [*controls, *chosen]:
            if c not in train.columns:
                continue
            rho = pd.to_numeric(train[name], errors="coerce").corr(pd.to_numeric(train[c], errors="coerce"), method="spearman")
            if rho is not None and np.isfinite(rho) and abs(float(rho)) >= CLUSTER_SPEARMAN_ABS:
                ok = False
                break
        if ok:
            chosen.append(name)
        if len(chosen) >= MAX_NEW_FACTORS_IN_MAIN:
            break
    return chosen


def train_nested(*, resume: bool = True, n_jobs: int = XGB_N_JOBS) -> dict[str, Any]:
    dest = exp_dir(NESTED_CODE)
    dest.mkdir(parents=True, exist_ok=True)
    oof_path = dest / "oof.json"
    if resume and oof_path.is_file() and (dest / "five_pred.parquet").is_file():
        return load_json(oof_path)
    init_panels()
    from .runner import _G

    pit_pass = _pit_pass()
    fold_sel: dict[str, Any] = {"five": {}, "ten": {}}

    def _head(name: str, training: pd.DataFrame, candidates: pd.DataFrame, locked: Sequence[str], head) -> pd.DataFrame:
        frames = []
        for seed in SEEDS:
            seed_dir = dest / name / f"seed{seed}"
            seed_path = seed_dir / "predictions.parquet"
            if resume and seed_path.is_file():
                prediction = pd.read_parquet(seed_path)
                prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
                frames.append(prediction)
                continue
            preds = []
            years = sorted(int(y) for y in candidates["trade_date"].dt.year.dropna().unique() if 2018 <= int(y) <= 2026)
            for y in years:
                train_m, _ = fold_masks(training, predict_year=y, label=head.target)
                tr_all = training.loc[train_m].copy()
                chosen = _nested_select(tr_all, extras=pit_pass, locked=locked, label=head.target)
                fold_sel[name][str(y)] = chosen
                features = unique_keep([*locked, *chosen])
                tr, _ = complete_cases(tr_all, features, require_target=True, target=head.target)
                te = candidates.loc[candidates["trade_date"].dt.year.eq(y)].copy()
                te, _ = complete_cases(te, features, require_target=False, target=head.target)
                if tr.empty or te.empty:
                    continue
                params = model_parameters(int(seed), n_jobs=n_jobs)
                params["eval_metric"] = head.eval_metric
                params["rank_n_bins"] = head.rank_n_bins
                config = ranker_config_from_params(params)
                # 单年预测：复用 rolling 入口但只喂当年候选
                part, _meta = rolling_train_predict_open(
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
                preds.append(part)
            if not preds:
                raise RuntimeError(f"nested {name} seed{seed} produced no predictions")
            prediction = pd.concat(preds, ignore_index=True)
            seed_dir.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(seed_path, index=False)
            frames.append(prediction)
        ens = ensemble_rank_pct(frames)
        ens.to_parquet(dest / f"{name}_pred.parquet", index=False)
        return ens

    five = _head("five", _G["five_training"], _G["five_candidate"], FIVE_FEATURES, FIVE_HEAD)
    ten = _head("ten", _G["ten_training"], _G["ten_candidate"], TEN_FEATURES, TEN_HEAD)
    payload = {
        "code": NESTED_CODE,
        "role": "NESTED_WALK_FORWARD_FEATURE_SELECTION",
        "status": "OOF",
        "independent_oos": False,
        "both_are_not_independent_oos": True,
        "fold_selection": fold_sel,
        "five_oof": oof_rankic(five, _G["five_candidate"], target=FIVE_HEAD.target),
        "ten_oof": oof_rankic(ten, _G["ten_candidate"], target=TEN_HEAD.target),
        "pit_pass_pool": pit_pass,
        "looked_at_combo_nav": False,
        "out_dir": str(dest),
        "note": "每折仅用训练窗选因子。NESTED 与 FIXED 都不是独立 OOS。",
    }
    dump_json(dest / "oof.json", payload)
    dump_json(dest / "summary.json", {**payload, "status": "OOF_ONLY"})
    return payload


def _gate(row: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    gate = INCREMENT_GATE
    bf = finite_or_none((base.get("five_oof") or {}).get("oof_rankic_mean"))
    bt = finite_or_none((base.get("ten_oof") or {}).get("oof_rankic_mean"))
    rf = finite_or_none((row.get("five_oof") or {}).get("oof_rankic_mean"))
    rt = finite_or_none((row.get("ten_oof") or {}).get("oof_rankic_mean"))
    nf = int((row.get("five_oof") or {}).get("n_oos_days") or 0)
    nt = int((row.get("ten_oof") or {}).get("n_oos_days") or 0)
    eps = float(gate["eps"])
    d5 = None if rf is None or bf is None else rf - bf
    d10 = None if rt is None or bt is None else rt - bt
    not_worse = True
    if gate["not_worse_both_heads"]:
        if rf is None or bf is None or rt is None or bt is None:
            not_worse = False
        else:
            not_worse = rf + eps >= bf and rt + eps >= bt
    better = False
    if gate["strict_better_any_head"]:
        better = (d5 is not None and d5 > float(gate["min_delta"])) or (d10 is not None and d10 > float(gate["min_delta"]))
    days_ok = nf >= int(gate["min_oos_days"]) and nt >= int(gate["min_oos_days"])
    passed = bool(not_worse and better and days_ok)
    return {
        "passed": passed,
        "delta_five": d5,
        "delta_ten": d10,
        "not_worse_both": not_worse,
        "strict_better_any": better,
        "days_ok": days_ok,
        "gate_id": gate["id"],
    }


def run_model(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and MODEL_JSON.is_file():
        existing = load_json(MODEL_JSON)
        if existing.get("status") == "DONE":
            return existing
    prepare_panels(resume=True)
    matrix = freeze_model_matrix(resume=True)
    specs = list(matrix.get("specs") or [])
    xgb_jobs = 1 if int(jobs) > 1 else XGB_N_JOBS
    rows: list[dict[str, Any]] = []

    def _run(spec: dict[str, Any]) -> dict[str, Any]:
        print(f"[round21] OOF {spec['code']} role={spec.get('role')}", flush=True)
        if spec.get("code") == NESTED_CODE or spec.get("nested"):
            return train_nested(resume=resume, n_jobs=xgb_jobs)
        if spec.get("alias_from") and spec.get("role") == "add_five_only":
            return _alias_head(spec["code"], spec["alias_from"], five_from="src", ten_from="frozen", spec=spec)
        if spec.get("alias_from") and spec.get("role") == "add_ten_only":
            return _alias_head(spec["code"], spec["alias_from"], five_from="frozen", ten_from="src", spec=spec)
        if spec.get("alias_from") and spec.get("role") == "replace_diagnostic":
            return _alias_head(spec["code"], spec["alias_from"], five_from="src", ten_from="src", spec=spec)
        return train_oof(spec, resume=resume, n_jobs=xgb_jobs)

    # 先训非 alias，再 alias（五/十 only 依赖 ADD both）
    primary = [s for s in specs if not s.get("alias_from")]
    aliases = [s for s in specs if s.get("alias_from")]
    if int(jobs) <= 1:
        for spec in primary:
            rows.append(_run(spec))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
            futs = {pool.submit(_run, spec): spec["code"] for spec in primary}
            order = {spec["code"]: i for i, spec in enumerate(primary)}
            got = {}
            for fut in as_completed(futs):
                got[futs[fut]] = fut.result()
            rows = [got[c] for c, _ in sorted(order.items(), key=lambda t: t[1]) if c in got]
    for spec in aliases:
        rows.append(_run(spec))

    by_code = {str(r.get("code")): r for r in rows}
    base = by_code.get(A01) or {}
    increment_rows = []
    for r in rows:
        if r.get("code") in {A00, A01}:
            increment_rows.append({**r, "increment": {"passed": False, "reason": "anchor"}})
            continue
        inc = _gate(r, base)
        increment_rows.append({**r, "increment": inc})
        fid = None
        for spec in specs:
            if spec["code"] == r.get("code"):
                fid = spec.get("factor_id")
                break
        if fid:
            prev = load_json(FACTORS_DIR / fid / "incrementality.json") or {"factor_id": fid, "vs": A01, "runs": []}
            runs = list(prev.get("runs") or [])
            runs.append({"code": r.get("code"), "role": r.get("role"), "oof": {"five": r.get("five_oof"), "ten": r.get("ten_oof")}, "increment": inc})
            dump_json(FACTORS_DIR / fid / "incrementality.json", {"factor_id": fid, "vs": A01, "runs": runs})
    passed = [r for r in increment_rows if (r.get("increment") or {}).get("passed")]
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
            for r in increment_rows
        ]
    ).to_csv(OOF_CSV, index=False)

    payload = {
        "status": "DONE",
        "n_oof_models": len(rows),
        "n_increment_pass": len(passed),
        "increment_pass_codes": [r.get("code") for r in passed],
        "increment_gate": INCREMENT_GATE,
        "a01": {"five_oof": base.get("five_oof"), "ten_oof": base.get("ten_oof")},
        "fixed": by_code.get(FIXED_CODE),
        "nested": by_code.get(NESTED_CODE),
        "both_are_not_independent_oos": True,
        "looked_at_combo_nav": False,
        "csv": str(OOF_CSV),
        "rows": [
            {
                "code": r.get("code"),
                "role": r.get("role"),
                "status": r.get("status"),
                "five_oof": r.get("five_oof"),
                "ten_oof": r.get("ten_oof"),
                "increment": r.get("increment"),
            }
            for r in increment_rows
        ],
    }
    dump_json(MODEL_JSON, payload)
    lines = [
        "# 08 Walk-Forward OOF（未看组合 NAV）",
        "",
        f"`FIXED_FACTOR_SET_REPLAY` 与 `NESTED_WALK_FORWARD_FEATURE_SELECTION` **都不是独立 OOS**。",
        f"OOF 模型 N=`{len(rows)}`。预注册增量门 `{INCREMENT_GATE['id']}` 通过=`{len(passed)}`。",
        f"A01 five=`{(base.get('five_oof') or {}).get('oof_rankic_mean')}` ten=`{(base.get('ten_oof') or {}).get('oof_rankic_mean')}`",
        "",
        "| code | role | five OOF | ten OOF | Δfive | Δten | gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in increment_rows:
        inc = r.get("increment") or {}
        lines.append(
            f"| `{r.get('code')}` | {r.get('role')} | {(r.get('five_oof') or {}).get('oof_rankic_mean')} | "
            f"{(r.get('ten_oof') or {}).get('oof_rankic_mean')} | {inc.get('delta_five')} | {inc.get('delta_ten')} | {inc.get('passed')} |"
        )
    write_md(MODEL_MD, "\n".join(lines))
    return payload


__all__ = ["MATRIX_JSON", "MODEL_JSON", "OOF_CSV", "freeze_model_matrix", "run_model", "train_nested"]
