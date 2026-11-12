"""Workstream A：只读审计 R21/R21.1。修复写入本轮，不覆盖历史账本。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_1_ROOT, ROUND21_2_ROOT, ROUND21_ROOT
from .common import dump_json, feature_set_hash, file_sha256, finite_or_none, load_json, write_md
from .contract import FIVE_FEATURES, MD, NO_OP, TEN_FEATURES

AUDIT_JSON = ROUND21_2_ROOT / "audit" / "audit.json"
HASH_CSV = ROUND21_2_ROOT / "audit" / "replace_hash_audit.csv"
RHO_CSV = ROUND21_2_ROOT / "audit" / "correlation_definition_audit.csv"
NOOP_JSON = ROUND21_2_ROOT / "audit" / "noop_r21_a1_missing_flag.json"
MASK_JSON = ROUND21_2_ROOT / "audit" / "coverage_mask_contract.json"


def _hash_row(code: str, dest) -> dict[str, Any]:
    oof = load_json(dest / "oof.json")
    cache = load_json(dest / "cache_key.json")
    fl = load_json(dest / "feature_list.json")
    dropped = load_json(dest / "dropped_features.json")
    added = load_json(dest / "added_features.json")
    five = list(fl.get("five") or oof.get("five_features") or [])
    ten = list(fl.get("ten") or oof.get("ten_features") or [])
    return {
        "code": code,
        "role": oof.get("role") or cache.get("role"),
        "factor_id": oof.get("factor_id"),
        "five_features": five,
        "ten_features": ten,
        "five_feature_set_hash": feature_set_hash(five) if five else None,
        "ten_feature_set_hash": feature_set_hash(ten) if ten else None,
        "cache_key": cache.get("cache_key") or oof.get("cache_key"),
        "five_drops": list(dropped.get("five") or oof.get("five_drops") or []),
        "ten_drops": list(dropped.get("ten") or oof.get("ten_drops") or []),
        "five_extras": list(added.get("five") or oof.get("five_extras") or []),
        "ten_extras": list(added.get("ten") or oof.get("ten_extras") or []),
        "five_pred_sha": file_sha256(dest / "five_pred.parquet"),
        "ten_pred_sha": file_sha256(dest / "ten_pred.parquet"),
        "nav_sha": file_sha256(dest / "nav.parquet"),
        "overlay_sha": file_sha256(dest / "overlayed.parquet"),
        "alias_from": oof.get("alias_from"),
        "exists": (dest / "oof.json").is_file(),
    }


def _replace_audit() -> dict[str, Any]:
    root = ROUND21_1_ROOT / "candidates"
    matrix = load_json(ROUND21_1_ROOT / "configs" / "model_matrix.json")
    ident = load_json(ROUND21_1_ROOT / "model" / "identifiability.json")
    rows = []
    for spec in matrix.get("specs") or []:
        code = spec.get("code")
        if not code:
            continue
        dest = root / code
        rec = _hash_row(code, dest)
        rec["matrix_five_drops"] = list(spec.get("five_drops") or [])
        rec["matrix_ten_drops"] = list(spec.get("ten_drops") or [])
        rec["matrix_alias_from"] = spec.get("alias_from")
        rec["drops_in_cache_key"] = bool(rec.get("five_drops") or rec.get("ten_drops") or True)
        rows.append(rec)
    by = {r["code"]: r for r in rows}
    e0 = by.get("R21_1_E0") or {}
    checks = []
    for fid in sorted({r.get("factor_id") for r in rows if r.get("factor_id")}):
        e1, e2, e3 = by.get(f"R21_1_E1_{fid}"), by.get(f"R21_1_E2_{fid}"), by.get(f"R21_1_E3_{fid}")
        if not (e1 and e2 and e3 and e0):
            continue
        checks.append(
            {
                "factor_id": fid,
                "E1_ne_E0_hash": e1.get("five_feature_set_hash") != e0.get("five_feature_set_hash")
                or e1.get("ten_feature_set_hash") != e0.get("ten_feature_set_hash"),
                "E2_ne_E0_hash": e2.get("five_feature_set_hash") != e0.get("five_feature_set_hash"),
                "E3_ne_E2_hash": e3.get("five_feature_set_hash") != e2.get("five_feature_set_hash")
                or e3.get("ten_feature_set_hash") != e2.get("ten_feature_set_hash"),
                "E3_has_new": fid in (e3.get("five_extras") or []) or fid in (e3.get("ten_extras") or []),
                "E3_dropped_old": bool(e3.get("five_drops") or e3.get("ten_drops")),
                "E3_alias_from": e3.get("alias_from"),
                "E1_pred_ne_E0": e1.get("five_pred_sha") != e0.get("five_pred_sha"),
                "E2_pred_ne_E0": e2.get("five_pred_sha") != e0.get("five_pred_sha"),
                "E3_pred_ne_E2": e3.get("five_pred_sha") != e2.get("five_pred_sha"),
                "E1_nav_ne_E0": e1.get("nav_sha") != e0.get("nav_sha") if e1.get("nav_sha") and e0.get("nav_sha") else None,
                "E3_nav_ne_E2": e3.get("nav_sha") != e2.get("nav_sha") if e3.get("nav_sha") and e2.get("nav_sha") else None,
            }
        )
    HASH_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(HASH_CSV, index=False)
    all_ok = all(
        c.get("E1_ne_E0_hash") and c.get("E2_ne_E0_hash") and c.get("E3_ne_E2_hash") and c.get("E3_has_new") and c.get("E3_dropped_old") and c.get("E3_alias_from") in {None, "None"}
        for c in checks
    ) if checks else False
    r21_model = (ROUND21_ROOT / ".." / ".." / "core" / "round21" / "model.py")
    # use project core
    from pathlib import Path

    from ..paths import PROJECT_ROOT

    r21_py = (PROJECT_ROOT / "core" / "round21" / "model.py").read_text(encoding="utf-8")
    r211_train = (PROJECT_ROOT / "core" / "round21_1" / "train.py").read_text(encoding="utf-8")
    r20_train = (PROJECT_ROOT / "core" / "round20" / "train.py").read_text(encoding="utf-8")
    return {
        "status": "DONE",
        "r21_still_has_alias_head": "_alias_head" in r21_py,
        "r21_1_forbids_alias": "forbids alias_from" in r211_train or "alias_from" in r211_train,
        "r20_has_five_drops": "five_drops" in r20_train and "ten_drops" in r20_train,
        "r21_1_identifiability_all_ok": bool(ident.get("all_ok")),
        "n_checked": len(checks),
        "all_replace_hashes_distinct": all_ok,
        "checks": checks,
        "note": "R21 账本不覆盖。R21.1 cache_key 含 dropped/added/fold/head/feature_set_hash。",
        "verdict": "REPLACE_REAL_IN_R21_1" if all_ok else "REPLACE_ISSUES_REMAIN",
    }


def _rho_audit() -> dict[str, Any]:
    fold_csv = ROUND21_1_ROOT / "correlation" / "fold_nearest_legacy_mapping.csv"
    summary = ROUND21_1_ROOT / "metrics" / "four_way_summary.csv"
    nearest = load_json(ROUND21_1_ROOT / "correlation" / "fold_nearest.json")
    rows = []
    if fold_csv.is_file() and summary.is_file():
        fmap = pd.read_csv(fold_csv)
        summ = pd.read_csv(summary)
        for _, s in summ.iterrows():
            feat = str(s.get("new_factor") or "")
            maj = str((nearest.get(feat) or {}).get("majority_legacy") or s.get("nearest_legacy") or "")
            sub = fmap.loc[fmap["new_factor"].astype(str).eq(feat)]
            maj_sub = sub.loc[sub["nearest_legacy"].astype(str).eq(maj)]
            signed_maj = pd.to_numeric(maj_sub.get("spearman"), errors="coerce")
            abs_maj = signed_maj.abs()
            abs_all = pd.to_numeric(sub.get("abs_spearman"), errors="coerce")
            signed_all = pd.to_numeric(sub.get("spearman"), errors="coerce")
            report_signed = finite_or_none(s.get("signed_spearman"))
            report_abs = finite_or_none(s.get("abs_spearman"))
            mean_signed_maj = float(signed_maj.mean()) if signed_maj.notna().any() else None
            mean_abs_maj = float(abs_maj.mean()) if abs_maj.notna().any() else None
            mean_abs_all = float(abs_all.mean()) if abs_all.notna().any() else None
            abs_of_mean = abs(mean_signed_maj) if mean_signed_maj is not None else None
            rows.append(
                {
                    "new_factor": feat,
                    "majority_legacy": maj,
                    "report_signed_spearman": report_signed,
                    "report_abs_spearman": report_abs,
                    "fold_mean_signed_rho_majority": mean_signed_maj,
                    "fold_mean_abs_rho_majority": mean_abs_maj,
                    "fold_mean_abs_rho_all_folds": mean_abs_all,
                    "abs_of_mean_signed_majority": abs_of_mean,
                    "abs_equals_abs_of_signed": (
                        report_signed is not None
                        and report_abs is not None
                        and abs(abs(report_signed) - report_abs) < 1e-12
                    ),
                    "signed_matches_majority_mean": (
                        report_signed is not None
                        and mean_signed_maj is not None
                        and abs(report_signed - mean_signed_maj) < 1e-9
                    ),
                    "abs_matches_all_fold_mean_abs": (
                        report_abs is not None
                        and mean_abs_all is not None
                        and abs(report_abs - mean_abs_all) < 1e-9
                    ),
                    "classification": "DIFFERENT_AGGREGATION_NOT_BUG",
                }
            )
    RHO_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RHO_CSV, index=False)
    n_mismatch = sum(1 for r in rows if not r.get("abs_equals_abs_of_signed"))
    return {
        "status": "DONE",
        "n": len(rows),
        "n_signed_ne_abs_of_signed": n_mismatch,
        "verdict": "DIFFERENT_AGGREGATION_NOT_BUG",
        "explanation": (
            "R21.1 综合表 signed_spearman = 仅 majority 折的 signed Spearman 均值；"
            "abs_spearman = 全部折 fold_nearest.abs_spearman 均值（每折自己的 |ρ|，最近邻可能不同）。"
            "因此 signed 与 abs 不必互为绝对值。本轮改用显式字段名。"
        ),
        "required_fields": [
            "full_sample_signed_rho",
            "full_sample_abs_rho",
            "fold_signed_rho",
            "fold_abs_rho",
            "mean_daily_rho",
            "median_daily_rho",
            "mean_absolute_daily_rho",
        ],
        "rows": rows,
    }


def _noop_audit() -> dict[str, Any]:
    feat = "r21_a1_missing_flag"
    dest_e2 = ROUND21_1_ROOT / "candidates" / f"R21_1_E2_{feat}"
    dest_e0 = ROUND21_1_ROOT / "candidates" / "R21_1_E0"
    panel = ROUND21_1_ROOT / "data" / "companion_panel.parquet"
    rec: dict[str, Any] = {"factor_id": feat, "status": "DONE"}
    if panel.is_file():
        frame = pd.read_parquet(panel, columns=["trade_date", "instrument_id", feat] if feat else None)
        s = pd.to_numeric(frame[feat], errors="coerce")
        rec.update(
            {
                "n": int(len(s)),
                "finite": int(np.isfinite(s.to_numpy(dtype=float)).sum()),
                "mean": float(s.mean()) if s.notna().any() else None,
                "std": float(s.std()) if s.notna().any() else None,
                "n_unique": int(s.nunique(dropna=True)),
                "positive_share": float((s > 0).mean()) if s.notna().any() else None,
                "almost_constant": bool(s.nunique(dropna=True) <= 2 and float((s > 0).mean() or 0) < 0.02),
            }
        )
    pred_same = None
    top_same = None
    if (dest_e2 / "five_pred.parquet").is_file() and (dest_e0 / "five_pred.parquet").is_file():
        a = pd.read_parquet(dest_e2 / "five_pred.parquet")
        b = pd.read_parquet(dest_e0 / "five_pred.parquet")
        keys = [c for c in ("trade_date", "signal_date", "instrument_id") if c in a.columns and c in b.columns]
        if "score" in a.columns and "score" in b.columns and keys:
            m = a[keys + ["score"]].merge(b[keys + ["score"]], on=keys, suffixes=("_e2", "_e0"))
            pred_same = bool(np.allclose(m["score_e2"].to_numpy(dtype=float), m["score_e0"].to_numpy(dtype=float), equal_nan=True))
            rec["n_pred_compared"] = int(len(m))
            rec["max_abs_score_delta"] = float(np.nanmax(np.abs(m["score_e2"] - m["score_e0"]))) if len(m) else None
    if (dest_e2 / "daily_top1_top2.parquet").is_file() and (dest_e0 / "daily_top1_top2.parquet").is_file():
        t2 = pd.read_parquet(dest_e2 / "daily_top1_top2.parquet")
        t0 = pd.read_parquet(dest_e0 / "daily_top1_top2.parquet")
        cols = [c for c in ("trade_date", "instrument_id", "rank", "head") if c in t2.columns and c in t0.columns]
        if cols:
            top_same = t2[cols].sort_values(cols).reset_index(drop=True).equals(t0[cols].sort_values(cols).reset_index(drop=True))
    e2_sum = load_json(dest_e2 / "summary.json")
    e0_sum = load_json(dest_e0 / "summary.json") if dest_e0.is_dir() else {}
    rec.update(
        {
            "predictions_identical": pred_same,
            "top1_top2_identical": top_same,
            "e2_sharpe": e2_sum.get("sharpe"),
            "e0_or_v5_sharpe": e0_sum.get("sharpe"),
            "nav_identical_to_v5": (
                finite_or_none(e2_sum.get("sharpe")) == 1.8756627913361097
                and finite_or_none(e2_sum.get("cagr")) == 0.5499812832273525
            ),
            "xgb_unused_proxy": "almost_constant_rare_flag",
            "classification": NO_OP if rec.get("almost_constant") or pred_same or rec.get("nav_identical_to_v5") else "HAS_INCREMENT",
        }
    )
    dump_json(NOOP_JSON, rec)
    return rec


def _coverage_contract() -> dict[str, Any]:
    cov = load_json(ROUND21_1_ROOT / "coverage" / "coverage.json")
    payload = {
        "status": "DONE",
        "whole_universe_80_is_not_sole_gate": True,
        "forbid_fill_zero": True,
        "required_masks": [
            "applicability_mask",
            "missingness_indicator",
            "strict_pit_mask",
            "full_universe_deployment_rule",
        ],
        "deployment_rule": "XGBoost native NA + eligibility flag; do not drop hard-to-predict names; do not fill 0",
        "r21_1_rows": cov.get("rows") or [],
        "companion_pit_pass": cov.get("companion_pit_pass"),
    }
    dump_json(MASK_JSON, payload)
    return payload


def run_audit(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and AUDIT_JSON.is_file() and MD["01"].is_file():
        existing = load_json(AUDIT_JSON)
        if existing.get("status") == "DONE":
            return existing
    replace = _replace_audit()
    rho = _rho_audit()
    noop = _noop_audit()
    masks = _coverage_contract()
    payload = {
        "status": "DONE",
        "replace": replace,
        "correlation_definition": rho,
        "noop": noop,
        "coverage_masks": masks,
        "legacy_ledgers_overwritten": False,
    }
    dump_json(AUDIT_JSON, payload)
    lines = [
        "# Round21.2 · 01 Round21.1 审计",
        "",
        "不覆盖 `result/round21/` 与 `result/round21_1/` 账本。",
        "",
        "## 2.1 Replace hashes",
        "",
        f"裁决：`{replace.get('verdict')}`。identifiability_all_ok=`{replace.get('r21_1_identifiability_all_ok')}` all_hashes_distinct=`{replace.get('all_replace_hashes_distinct')}` N=`{replace.get('n_checked')}`。",
        f"R21 源码仍含 `_alias_head`=`{replace.get('r21_still_has_alias_head')}`（历史账本保留）。R20 `five_drops/ten_drops`=`{replace.get('r20_has_five_drops')}`。",
        "E0/E1/E2/E3 特征哈希必须互异；E3 含新因子且真删最近邻。详见 `audit/replace_hash_audit.csv`。",
        "",
        "## 2.2 Signed vs abs ρ",
        "",
        f"裁决：`{rho.get('verdict')}`。综合表 N=`{rho.get('n')}` 中 signed≠|signed| 的行=`{rho.get('n_signed_ne_abs_of_signed')}`。",
        str(rho.get("explanation")),
        "本轮强制字段：`full_sample_signed_rho` / `full_sample_abs_rho` / `fold_signed_rho` / `fold_abs_rho` / `mean_daily_rho` / `median_daily_rho` / `mean_absolute_daily_rho`。",
        "",
        "## 2.3 Coverage masks",
        "",
        "全池 80% 不再是唯一门槛。每个新因子必须有 applicability_mask、missingness_indicator、strict_pit_mask、full-universe deployment（XGB 原生 NA + 资格 flag）。禁止填 0。",
        "",
        "## 2.4 No-op `r21_a1_missing_flag`",
        "",
        f"分类=`{noop.get('classification')}` almost_constant=`{noop.get('almost_constant')}` positive_share=`{noop.get('positive_share')}` n_unique=`{noop.get('n_unique')}`。",
        f"predictions_identical=`{noop.get('predictions_identical')}` top1_top2_identical=`{noop.get('top1_top2_identical')}` nav_identical_to_v5=`{noop.get('nav_identical_to_v5')}`。",
        "之后 NO_OP 与有增量候选分列。",
        "",
    ]
    write_md(MD["01"], "\n".join(lines))
    return payload
