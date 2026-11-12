"""Round 23.3 主流程。"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.r225241_sleeve import apply_b03_o01, run_round7
from core.research_integrity_v2.io_v2 import dump_json, load_json, load_stage, save_stage, write_df, write_text
from core.research_integrity_v2.stats_v2 import paired_block_bootstrap_v2, spa_hansen_or_blocked
from core.research_integrity_v2.time_contract import shanghai_now
from core.round11.common import daily_from_nav
from core.round22_6.sleeve import run_formal_sleeve
from core.round23_2.lgbm_train import train_lgbm_ranker_trunc
from core.round23_3.blend import blend_ten, blend_xgb_lgbm
from core.round23_3.contract import (
    AUDIT,
    DEFAULT_W_LGBM,
    DEFAULT_W_TEN,
    E0,
    E4_MIN_PAIRS,
    E4_TRIGGER_DELTA,
    LGBM_WEIGHTS,
    PANEL,
    ROOT,
    TEN_PANEL,
    TEN_TRAIN,
    TEN_WEIGHTS,
    TRAIN,
    anchor_copy,
)
from core.round23_3.inner_adapter import run_windowed_sleeve
from core.round23_3.selection import evaluate_inner_configs
from core.round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from core.round22_3.contract import FIVE_FEATURES as F5, TEN_FEATURES as T13


def _skip(name: str, resume: bool) -> dict[str, Any] | None:
    return load_stage(ROOT, name) if resume else None


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _v5_fused() -> pd.DataFrame:
    from core.round17.backtest import fuse_heads

    five = pd.read_parquet(E0 / "five_pred.parquet")
    ten = pd.read_parquet(E0 / "ten_pred.parquet")
    return fuse_heads(five, ten)


def run_source_audit(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("source_audit", resume)
    if cached:
        return cached
    AUDIT.mkdir(parents=True, exist_ok=True)
    baseline_audit = Path("/data1/stock-mxy/research_publish/wt_baseline_v5/baseline_v5_audit")
    copied = []
    for name in (
        "factor_registry.csv",
        "factor_formulas.md",
        "feature_order.json",
        "factor_source_lineage.json",
        "factor_pit_contract.md",
        "SECOND_TO_THIRD_FACTOR_MIGRATION_PLAN.md",
        "second_board_coverage_matrix.csv",
    ):
        src = baseline_audit / name
        if not src.is_file():
            src = baseline_audit / "data_catalog" / name
        if src.is_file():
            dst = AUDIT / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    # feature order vs code
    order = load_json(AUDIT / "feature_order.json") if (AUDIT / "feature_order.json").is_file() else {}
    five_code = list(FIVE_FEATURES)
    ten_code = list(TEN_FEATURES)
    five_audit = list(order.get("five14") or [])
    ten_audit = list(order.get("ten13") or [])
    mismatch = {
        "five_match": five_code == five_audit,
        "ten_match": ten_code == ten_audit,
        "ten_has_g0_amount_log": "g0_amount_log" in ten_code,
        "BASELINE_SOURCE_MISMATCH": (five_code != five_audit) or (ten_code != ten_audit),
        "five_code": five_code,
        "ten_code": ten_code,
        "five_audit": five_audit,
        "ten_audit": ten_audit,
        "source_of_truth": "core/round7/contract.py FIVE_FEATURES/TEN_FEATURES",
    }
    dump_json(AUDIT / "feature_order_check.json", mismatch)
    # hashes of E0 preds
    hashes = {}
    for p in (E0 / "five_pred.parquet", E0 / "ten_pred.parquet"):
        if p.is_file():
            hashes[p.name] = _sha_file(p)
    branch_audit = ROOT / "_audit" / "SOURCE_BRANCH_AUDIT.json"
    payload = {
        "status": "DONE",
        "copied": copied,
        "feature_order_check": mismatch,
        "e0_hashes": hashes,
        "branch_audit_path": str(branch_audit) if branch_audit.is_file() else None,
        "anchor": anchor_copy(),
    }
    dump_json(ROOT / "source_audit.json", payload)
    return save_stage(ROOT, "source_audit", payload)


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("parity", resume)
    if cached:
        return cached
    v5 = _v5_fused()
    five = pd.read_parquet(E0 / "five_pred.parquet")
    ten = pd.read_parquet(E0 / "ten_pred.parquet")
    t090 = blend_ten(five, ten, 0.90)
    a = v5.sort_values(["signal_date", "instrument_id"]).reset_index(drop=True)
    b = t090.sort_values(["signal_date", "instrument_id"]).reset_index(drop=True)
    same_keys = a[["signal_date", "instrument_id"]].equals(b[["signal_date", "instrument_id"]])
    score_max_diff = float(np.nanmax(np.abs(a["score"] - b["score"]))) if same_keys and len(a) == len(b) else None
    zero_w, _ = blend_xgb_lgbm(v5, v5, 0.0)
    identity_zero = zero_w.equals(v5) or (
        len(zero_w) == len(v5)
        and float(
            np.nanmax(
                np.abs(
                    zero_w.sort_values(["signal_date", "instrument_id"])["score"].to_numpy()
                    - v5.sort_values(["signal_date", "instrument_id"])["score"].to_numpy()
                )
            )
        )
        <= 1e-15
    )
    s_v5 = run_formal_sleeve(
        ROOT / "parity" / "V5",
        v5,
        {"code": "R233_PARITY_V5", "round": "23.3", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        counts_as_alpha=False,
        top_changed=False,
        fee_same_direction=True,
        not_only_year=True,
    )
    s_90 = run_formal_sleeve(
        ROOT / "parity" / "T090",
        t090,
        {"code": "R233_PARITY_T090", "round": "23.3", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        counts_as_alpha=False,
        top_changed=False,
        fee_same_direction=True,
        not_only_year=True,
    )
    s_z = run_formal_sleeve(
        ROOT / "parity" / "L000",
        zero_w,
        {"code": "R233_PARITY_L000", "round": "23.3", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        counts_as_alpha=False,
        top_changed=False,
        fee_same_direction=True,
        not_only_year=True,
    )
    rows = [
        {"check": "feature_order_five14", "pass": list(F5) == list(FIVE_FEATURES)},
        {"check": "feature_order_ten13", "pass": list(T13) == list(TEN_FEATURES) and "g0_amount_log" not in TEN_FEATURES},
        {"check": "t090_keys", "pass": same_keys},
        {"check": "t090_score_max_diff", "pass": score_max_diff is not None and score_max_diff <= 1e-12, "value": score_max_diff},
        {
            "check": "t090_nav",
            "pass": abs(float(s_v5["nav_end"]) - float(s_90["nav_end"])) <= max(0.01, 1e-8 * float(s_v5["nav_end"])),
            "v5": s_v5.get("nav_end"),
            "cand": s_90.get("nav_end"),
        },
        {"check": "l000_identity", "pass": identity_zero},
        {
            "check": "l000_nav",
            "pass": abs(float(s_v5["nav_end"]) - float(s_z["nav_end"])) <= max(0.01, 1e-8 * float(s_v5["nav_end"])),
            "v5": s_v5.get("nav_end"),
            "cand": s_z.get("nav_end"),
        },
        {
            "check": "v5_anchor_sharpe",
            "pass": abs(float(s_v5["sharpe"]) - float(anchor_copy()["sharpe"])) < 1e-8,
            "value": s_v5.get("sharpe"),
        },
    ]
    write_df(ROOT / "parity_checks.csv", pd.DataFrame(rows))
    status = "PASS" if all(bool(r["pass"]) for r in rows) else "FAIL"
    payload = {"status": status, "rows": rows, "v5": s_v5, "t090": s_90, "l000": s_z}
    dump_json(ROOT / "parity.json", payload)
    return save_stage(ROOT, "parity", payload)


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (ROOT / "frozen_manifest.json").is_file():
        return load_json(ROOT / "frozen_manifest.json")
    body = {
        "round": "23.3",
        "frozen_before_new_nav_promotion": True,
        "freeze_timestamp": shanghai_now().isoformat(),
        "anchor": anchor_copy(),
        "ten_weights": list(TEN_WEIGHTS),
        "lgbm_weights": list(LGBM_WEIGHTS),
        "default_w_ten": DEFAULT_W_TEN,
        "default_w_lgbm": DEFAULT_W_LGBM,
        "inner_boot_n": 5000,
        "inner_block": 20,
        "current_baseline_changed": False,
        "auto_promote": False,
        "production_config_changed": False,
        "experiments": ["E1", "E2", "E3_conditional", "E4"],
    }
    dump_json(ROOT / "frozen_manifest.json", body)
    return save_stage(ROOT, "freeze", {**body, "status": "FROZEN"})


def run_base_oof(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    cached = _skip("base_oof", resume)
    if cached:
        return cached
    # XGB = 冻结 v5 OOF（E0）
    five_meta = {"source": str(E0 / "five_pred.parquet"), "sha256": _sha_file(E0 / "five_pred.parquet"), "head": "five", "model": "XGBRanker_v5"}
    ten_meta = {"source": str(E0 / "ten_pred.parquet"), "sha256": _sha_file(E0 / "ten_pred.parquet"), "head": "ten", "model": "XGBRanker_v5"}
    five_lgbm = train_lgbm_ranker_trunc(
        training=pd.read_parquet(TRAIN),
        candidates=pd.read_parquet(PANEL),
        features=list(FIVE_FEATURES),
        target=FIVE_HEAD.target,
        label_end_column=FIVE_HEAD.label_end_column,
        n_bins=FIVE_HEAD.rank_n_bins,
        eval_at=[1],
        truncation_level=4,
        output_dir=ROOT / "oof" / "lgbm_five_trunc4",
        resume=resume,
    )
    ten_lgbm = train_lgbm_ranker_trunc(
        training=pd.read_parquet(TEN_TRAIN),
        candidates=pd.read_parquet(TEN_PANEL),
        features=list(TEN_FEATURES),
        target=TEN_HEAD.target,
        label_end_column=TEN_HEAD.label_end_column,
        n_bins=TEN_HEAD.rank_n_bins,
        eval_at=[2],
        truncation_level=5,
        output_dir=ROOT / "oof" / "lgbm_ten_trunc5",
        resume=resume,
    )
    rows = [
        {"model": "xgb_five", **five_meta},
        {"model": "xgb_ten", **ten_meta},
        {"model": "lgbm_five", "rows": five_lgbm.get("rows"), "trunc": 4, "eval_at": [1]},
        {"model": "lgbm_ten", "rows": ten_lgbm.get("rows"), "trunc": 5, "eval_at": [2]},
    ]
    write_df(ROOT / "oof_model_registry.csv", pd.DataFrame(rows))
    payload = {"status": "DONE", "xgb_five": five_meta, "xgb_ten": ten_meta, "lgbm_five": five_lgbm, "lgbm_ten": ten_lgbm}
    dump_json(ROOT / "base_oof.json", payload)
    return save_stage(ROOT, "base_oof", payload)


def run_inner_fix(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    """用半年窗冒烟：必须远小于全历史 NAV 日。"""
    cached = _skip("inner_fix", resume)
    if cached:
        return cached
    v5 = _v5_fused()
    out = run_windowed_sleeve(
        ROOT / "inner_fix" / "smoke_2019H1",
        v5,
        signal_start="2019-01-01",
        signal_end="2019-06-30",
        backtest_start="2019-01-01",
        backtest_end="2019-06-30",
        extra={"code": "R233_INNER_SMOKE", "round": "23.3"},
        resume=resume,
    )
    ok = (not out.get("window_fail")) and int(out.get("n_nav_days") or 0) <= 200
    payload = {"status": "PASS" if ok else "FAIL", "smoke": out}
    dump_json(ROOT / "inner_fix.json", payload)
    if not ok:
        raise RuntimeError(f"Inner replay window fix FAILED: n_nav_days={out.get('n_nav_days')}")
    return save_stage(ROOT, "inner_fix", payload)


def _nested_scores(score_map: dict[str, pd.DataFrame], selections: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    parts = []
    for rec in selections.itertuples(index=False):
        key = rec.chosen
        year = int(getattr(rec, year_col))
        part = score_map[key]
        part = part.loc[pd.to_datetime(part["signal_date"]).dt.year.eq(year)]
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["signal_date", "instrument_id", "score"])


def run_e1(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("e1", resume)
    if cached:
        return cached
    parity = load_json(ROOT / "parity.json")
    if parity.get("status") != "PASS":
        return save_stage(ROOT, "e1", {"status": "BLOCKED_PARITY_FAIL"})
    five = pd.read_parquet(E0 / "five_pred.parquet")
    ten = pd.read_parquet(E0 / "ten_pred.parquet")
    score_map = {f"t_{w:.2f}": blend_ten(five, ten, w) for w in TEN_WEIGHTS}
    # fixed configs formal
    fixed_rows = []
    for w, code in ((0.75, "E1_T075"), (0.90, "E1_T090"), (1.00, "E1_T100")):
        key = f"t_{w:.2f}"
        summary = run_formal_sleeve(
            ROOT / "e1" / code,
            score_map[key],
            {"code": f"R233_{code}", "round": "23.3", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
            resume=resume,
            counts_as_alpha=(code != "E1_T090"),
            top_changed=None if code != "E1_T090" else False,
            fee_same_direction=True,
            not_only_year=True,
        )
        fixed_rows.append({"experiment": code, "w_ten": w, **{k: summary.get(k) for k in ("sharpe", "cagr", "nav_end", "max_drawdown", "completed_trades", "economically_material")}})
    # nested inner select
    weight_rows = []
    fold_all = []
    obj_rows = []
    for year in range(2018, 2027):
        configs = [(f"t_{w:.2f}", score_map[f"t_{w:.2f}"]) for w in TEN_WEIGHTS]
        # 注意：E1 使用 Outer OOF 在 Y-1 上的预测——这些预测来自训练至 Y-2 的折，不含 Y-1 标签泄漏到特征；
        # 权重选择的标签/NAV 评价仅在 Inner 块内。
        sel = evaluate_inner_configs(
            configs,
            year=year,
            default_key="t_0.90",
            dest_root=ROOT / "inner" / "e1",
            resume=resume,
            model_fit_cutoff=f"{year-2}-12-31",
            model_hash="xgb_v5_oof",
            prediction_hash="e0_five_ten",
        )
        weight_rows.append({"year": year, "family": "E1", "chosen": sel["chosen"], "reason": sel["reason"]})
        fold_all.extend(sel["fold_rows"])
        for r in sel["records"]:
            obj_rows.append({"family": "E1", **r})
        write_df(ROOT / "inner" / "e1" / f"year_{year}_records.csv", pd.DataFrame(sel["records"]))
    nested = _nested_scores(score_map, pd.DataFrame(weight_rows))
    nest_summary = run_formal_sleeve(
        ROOT / "e1" / "E1_NESTED_SELECTED",
        nested,
        {"code": "R233_E1_NESTED", "round": "23.3", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    fixed_rows.append({"experiment": "E1_NESTED_SELECTED", "w_ten": None, **{k: nest_summary.get(k) for k in ("sharpe", "cagr", "nav_end", "max_drawdown", "completed_trades", "economically_material")}})
    write_df(ROOT / "e1_results.csv", pd.DataFrame(fixed_rows))
    write_df(ROOT / "selected_weights.csv", pd.DataFrame(weight_rows))
    write_df(ROOT / "inner_fold_manifest.csv", pd.DataFrame(fold_all))
    write_df(ROOT / "inner_objectives.csv", pd.DataFrame(obj_rows))
    payload = {"status": "DONE", "fixed": fixed_rows, "nested": nest_summary, "weights": weight_rows}
    dump_json(ROOT / "e1.json", payload)
    return save_stage(ROOT, "e1", payload)


def run_e2(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("e2", resume)
    if cached:
        return cached
    parity = load_json(ROOT / "parity.json")
    if parity.get("status") != "PASS":
        return save_stage(ROOT, "e2", {"status": "BLOCKED_PARITY_FAIL"})
    v5 = _v5_fused()
    from core.round17.backtest import fuse_heads

    lgbm = fuse_heads(
        pd.read_parquet(ROOT / "oof" / "lgbm_five_trunc4" / "predictions.parquet"),
        pd.read_parquet(ROOT / "oof" / "lgbm_ten_trunc5" / "predictions.parquet"),
    )
    score_map = {}
    fallback_all = []
    for w in LGBM_WEIGHTS:
        scores, fb = blend_xgb_lgbm(v5, lgbm, w)
        score_map[f"l_{w:.2f}"] = scores
        if len(fb):
            fb = fb.copy()
            fb["w_lgbm"] = w
            fallback_all.append(fb)
    if fallback_all:
        write_df(ROOT / "lgbm_fallback_days.csv", pd.concat(fallback_all, ignore_index=True))
    fixed_rows = []
    for w, code in ((0.00, "E2_L000"), (0.10, "E2_L010"), (0.25, "E2_L025"), (0.50, "E2_L050")):
        key = f"l_{w:.2f}"
        summary = run_formal_sleeve(
            ROOT / "e2" / code,
            score_map[key],
            {"code": f"R233_{code}", "round": "23.3", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
            resume=resume,
            counts_as_alpha=(code != "E2_L000"),
            top_changed=False if code == "E2_L000" else None,
            fee_same_direction=True,
            not_only_year=True,
        )
        fixed_rows.append({"experiment": code, "w_lgbm": w, **{k: summary.get(k) for k in ("sharpe", "cagr", "nav_end", "max_drawdown", "completed_trades", "economically_material")}})
    weight_rows = []
    obj_rows = []
    fold_all = []
    for year in range(2018, 2027):
        configs = [(f"l_{w:.2f}", score_map[f"l_{w:.2f}"]) for w in LGBM_WEIGHTS]
        sel = evaluate_inner_configs(
            configs,
            year=year,
            default_key="l_0.00",
            dest_root=ROOT / "inner" / "e2",
            resume=resume,
            model_fit_cutoff=f"{year-2}-12-31",
            model_hash="xgb_v5+lgbm_oof",
            prediction_hash="blend_e2",
        )
        weight_rows.append({"year": year, "family": "E2", "chosen": sel["chosen"], "reason": sel["reason"]})
        fold_all.extend(sel["fold_rows"])
        for r in sel["records"]:
            obj_rows.append({"family": "E2", **r})
        write_df(ROOT / "inner" / "e2" / f"year_{year}_records.csv", pd.DataFrame(sel["records"]))
    nested = _nested_scores(score_map, pd.DataFrame(weight_rows))
    nest_summary = run_formal_sleeve(
        ROOT / "e2" / "E2_NESTED_SELECTED",
        nested,
        {"code": "R233_E2_NESTED", "round": "23.3", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    fixed_rows.append({"experiment": "E2_NESTED_SELECTED", "w_lgbm": None, **{k: nest_summary.get(k) for k in ("sharpe", "cagr", "nav_end", "max_drawdown", "completed_trades", "economically_material")}})
    # append selected weights
    prev = pd.read_csv(ROOT / "selected_weights.csv") if (ROOT / "selected_weights.csv").is_file() else pd.DataFrame()
    write_df(ROOT / "selected_weights.csv", pd.concat([prev, pd.DataFrame(weight_rows)], ignore_index=True))
    write_df(ROOT / "e2_results.csv", pd.DataFrame(fixed_rows))
    if (ROOT / "inner_objectives.csv").is_file():
        write_df(ROOT / "inner_objectives.csv", pd.concat([pd.read_csv(ROOT / "inner_objectives.csv"), pd.DataFrame(obj_rows)], ignore_index=True))
    else:
        write_df(ROOT / "inner_objectives.csv", pd.DataFrame(obj_rows))
    if (ROOT / "inner_fold_manifest.csv").is_file():
        write_df(ROOT / "inner_fold_manifest.csv", pd.concat([pd.read_csv(ROOT / "inner_fold_manifest.csv"), pd.DataFrame(fold_all)], ignore_index=True))
    payload = {"status": "DONE", "fixed": fixed_rows, "nested": nest_summary, "weights": weight_rows}
    dump_json(ROOT / "e2.json", payload)
    return save_stage(ROOT, "e2", payload)


def run_cond_inc(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("cond_inc", resume)
    if cached:
        return cached
    v5 = _v5_fused()
    from core.round17.backtest import fuse_heads
    from core.r225241_stats import daily_spearman

    lgbm = fuse_heads(
        pd.read_parquet(ROOT / "oof" / "lgbm_five_trunc4" / "predictions.parquet"),
        pd.read_parquet(ROOT / "oof" / "lgbm_ten_trunc5" / "predictions.parquet"),
    )
    both = v5.rename(columns={"score": "xgb"}).merge(lgbm.rename(columns={"score": "lgbm"}), on=["signal_date", "instrument_id"], how="inner")
    # top agreement
    def topk(df, col, k=1):
        x = df.sort_values(["signal_date", col, "instrument_id"], ascending=[True, False, True])
        return x.groupby("signal_date").head(k)

    t1x = topk(both, "xgb", 1)
    t1l = topk(both, "lgbm", 1)
    agree1 = float((t1x.set_index("signal_date")["instrument_id"] == t1l.set_index("signal_date")["instrument_id"]).mean())
    # conditional increment trigger: any nested economically_material
    e1 = load_json(ROOT / "e1.json")
    e2 = load_json(ROOT / "e2.json")
    e1_mat = bool((e1.get("nested") or {}).get("economically_material"))
    e2_mat = bool((e2.get("nested") or {}).get("economically_material"))
    # also check fixed non-defaults
    for block in (e1.get("fixed") or [], e2.get("fixed") or []):
        for row in block:
            if row.get("experiment") not in {"E1_T090", "E2_L000"} and row.get("economically_material"):
                if "E1" in str(row.get("experiment")):
                    e1_mat = True
                if "E2" in str(row.get("experiment")):
                    e2_mat = True
    rows = [
        {
            "pair": "xgb_vs_lgbm",
            "daily_pred_corr": daily_spearman(both, "xgb", "lgbm"),
            "top1_agreement": agree1,
            "e1_conditional_increment": e1_mat,
            "e2_conditional_increment": e2_mat,
            "e3_trigger": bool(e1_mat or e2_mat),
        }
    ]
    write_df(ROOT / "conditional_increment.csv", pd.DataFrame(rows))
    payload = {"status": "DONE", "e3_trigger": bool(e1_mat or e2_mat), "rows": rows}
    dump_json(ROOT / "cond_inc.json", payload)
    return save_stage(ROOT, "cond_inc", payload)


def run_e3(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("e3", resume)
    if cached:
        return cached
    cond = load_json(ROOT / "cond_inc.json")
    if not cond.get("e3_trigger"):
        write_df(ROOT / "stacking_results.csv", pd.DataFrame([{"experiment": "E3", "status": "NOT_TRIGGERED"}]))
        return save_stage(ROOT, "e3", {"status": "NOT_TRIGGERED"})
    # Meta OOF：需要足够历史真实 OOF；用逐年 walk-forward 的 head ranks
    five = pd.read_parquet(E0 / "five_pred.parquet").rename(columns={"score": "xgb_five"})
    ten = pd.read_parquet(E0 / "ten_pred.parquet").rename(columns={"score": "xgb_ten"})
    lf = pd.read_parquet(ROOT / "oof" / "lgbm_five_trunc4" / "predictions.parquet").rename(columns={"score": "lgb_five"})
    lt = pd.read_parquet(ROOT / "oof" / "lgbm_ten_trunc5" / "predictions.parquet").rename(columns={"score": "lgb_ten"})
    keys = ["signal_date", "instrument_id"]
    panel = five
    for other in (ten, lf, lt):
        panel = panel.merge(other, on=keys, how="inner")
    if panel["signal_date"].dt.year.nunique() < 4:
        write_df(ROOT / "stacking_results.csv", pd.DataFrame([{"experiment": "E3", "status": "BLOCKED_META_OOF"}]))
        return save_stage(ROOT, "e3", {"status": "BLOCKED_META_OOF"})
    # labels
    labels = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", "label_e2_net"])
    labels = labels.rename(columns={"trade_date": "signal_date"})
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    panel = panel.merge(labels, on=keys, how="left")
    for col in ("xgb_five", "xgb_ten", "lgb_five", "lgb_ten"):
        panel[f"{col}_rank"] = panel.groupby("signal_date")[col].rank(method="average", pct=True)
    panel["xgb_lgb_disagreement"] = (panel["xgb_five_rank"] - panel["lgb_five_rank"]).abs()
    panel["five_ten_disagreement"] = (panel["xgb_five_rank"] - panel["xgb_ten_rank"]).abs()
    base_cols = ["xgb_five_rank", "xgb_ten_rank", "lgb_five_rank", "lgb_ten_rank"]
    from sklearn.linear_model import Ridge
    from scipy.optimize import nnls

    parts_nnls = []
    parts_ridge = []
    meta_rows = []
    years = sorted(panel["signal_date"].dt.year.unique())
    for year in years:
        if year < 2018:
            continue
        train = panel.loc[panel["signal_date"].dt.year.between(year - 3, year - 1)].dropna(subset=["label_e2_net"])
        test = panel.loc[panel["signal_date"].dt.year.eq(year)].copy()
        if len(train) < 500 or test.empty:
            continue
        Xtr = train[base_cols].to_numpy(dtype=float)
        ytr = train["label_e2_net"].to_numpy(dtype=float)
        # NNLS only on four base ranks
        w, _ = nnls(Xtr, ytr)
        if w.sum() <= 0:
            w = np.ones(4) / 4.0
        else:
            w = w / w.sum()
        Xt = test[base_cols].to_numpy(dtype=float)
        test_nn = test[keys].copy()
        test_nn["score"] = Xt @ w
        parts_nnls.append(test_nn)
        # Ridge with disagreement features; standardize on train only
        feat = base_cols + ["xgb_lgb_disagreement", "five_ten_disagreement"]
        mu = train[feat].mean()
        sd = train[feat].std().replace(0, 1.0)
        Ztr = ((train[feat] - mu) / sd).to_numpy(dtype=float)
        Zte = ((test[feat] - mu) / sd).to_numpy(dtype=float)
        ridge = Ridge(alpha=10.0)
        ridge.fit(Ztr, ytr)
        test_rg = test[keys].copy()
        test_rg["score"] = ridge.predict(Zte)
        parts_ridge.append(test_rg)
        meta_rows.append({"year": year, "nnls_w": w.tolist(), "train_rows": int(len(train)), "test_rows": int(len(test)), "model_fit_cutoff": f"{year-1}-12-31"})
    if not parts_nnls:
        write_df(ROOT / "stacking_results.csv", pd.DataFrame([{"experiment": "E3", "status": "BLOCKED_META_OOF"}]))
        return save_stage(ROOT, "e3", {"status": "BLOCKED_META_OOF"})
    nnls_scores = pd.concat(parts_nnls, ignore_index=True)
    ridge_scores = pd.concat(parts_ridge, ignore_index=True)
    s_nn = run_formal_sleeve(ROOT / "e3" / "E3_NNLS", nnls_scores, {"code": "R233_E3_NNLS", "round": "23.3"}, resume=resume, top_changed=None, fee_same_direction=True, not_only_year=True)
    s_rg = run_formal_sleeve(ROOT / "e3" / "E3_RIDGE", ridge_scores, {"code": "R233_E3_RIDGE", "round": "23.3"}, resume=resume, top_changed=None, fee_same_direction=True, not_only_year=True)
    rows = [
        {"experiment": "E3_NNLS", "status": "ran", **{k: s_nn.get(k) for k in ("sharpe", "cagr", "nav_end", "economically_material")}},
        {"experiment": "E3_RIDGE", "status": "ran", **{k: s_rg.get(k) for k in ("sharpe", "cagr", "nav_end", "economically_material")}},
    ]
    write_df(ROOT / "stacking_results.csv", pd.DataFrame(rows))
    write_df(ROOT / "e3_meta_folds.csv", pd.DataFrame(meta_rows))
    payload = {"status": "DONE", "nnls": s_nn, "ridge": s_rg}
    dump_json(ROOT / "e3.json", payload)
    return save_stage(ROOT, "e3", payload)


def run_e4(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("e4", resume)
    if cached:
        return cached
    v5 = _v5_fused()
    overlay, _ = apply_b03_o01(v5)
    # base
    s_base = run_formal_sleeve(ROOT / "e4" / "E4_BASE", overlay, {"code": "R233_E4_BASE", "round": "23.3"}, resume=resume, already_overlayed=True, counts_as_alpha=False, top_changed=False, fee_same_direction=True, not_only_year=True)
    # build top2/top3 pairs after same eligibility
    work = overlay.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort").copy()
    work["rk"] = work.groupby("signal_date").cumcount() + 1
    top3 = work.loc[work["rk"].le(3)].copy()
    # labels for pair delta
    labels = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", "label_e2_net"])
    labels = labels.rename(columns={"trade_date": "signal_date"})
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    top3 = top3.merge(labels, on=["signal_date", "instrument_id"], how="left")
    pairs = []
    for day, block in top3.groupby("signal_date"):
        block = block.sort_values("rk")
        if len(block) < 3:
            continue
        id1, id2, id3 = block["instrument_id"].tolist()[:3]
        s1, s2, s3 = block["score"].tolist()[:3]
        y2 = block["label_e2_net"].iloc[1]
        y3 = block["label_e2_net"].iloc[2]
        if not (np.isfinite(y2) and np.isfinite(y3)):
            continue
        pairs.append(
            {
                "signal_date": day,
                "id1": id1,
                "trained_pair_id2": id2,
                "trained_pair_id3": id3,
                "score_gap_23": float(s2 - s3),
                "label_delta_swap": float(y3 - y2),
            }
        )
    pair_df = pd.DataFrame(pairs)
    if len(pair_df) < E4_MIN_PAIRS:
        write_df(
            ROOT / "challenger_pair_attribution.csv",
            pd.DataFrame([{"status": "BLOCKED_INSUFFICIENT_PAIRS", "n_pairs": len(pair_df)}]),
        )
        return save_stage(ROOT, "e4", {"status": "BLOCKED_INSUFFICIENT_PAIRS", "n_pairs": int(len(pair_df)), "base": s_base})
    # disagreement feature from lgbm if available
    from core.round17.backtest import fuse_heads

    lgbm = fuse_heads(
        pd.read_parquet(ROOT / "oof" / "lgbm_five_trunc4" / "predictions.parquet"),
        pd.read_parquet(ROOT / "oof" / "lgbm_ten_trunc5" / "predictions.parquet"),
    )
    lmap = lgbm.set_index(["signal_date", "instrument_id"])["score"]

    def _lgb(day, iid):
        try:
            return float(lmap.loc[(pd.Timestamp(day), iid)])
        except Exception:
            return np.nan

    pair_df["dis_gap"] = [
        abs((_lgb(r.signal_date, r.trained_pair_id2) if np.isfinite(_lgb(r.signal_date, r.trained_pair_id2)) else 0.0) - (_lgb(r.signal_date, r.trained_pair_id3) if np.isfinite(_lgb(r.signal_date, r.trained_pair_id3)) else 0.0))
        for r in pair_df.itertuples(index=False)
    ]
    from sklearn.linear_model import Ridge

    exec_rows = []
    score_only_parts = []
    dis_parts = []
    for year in range(2018, 2027):
        train = pair_df.loc[pd.to_datetime(pair_df["signal_date"]).dt.year.between(year - 3, year - 1)]
        test = pair_df.loc[pd.to_datetime(pair_df["signal_date"]).dt.year.eq(year)]
        if len(train) < 50 or test.empty:
            continue
        # score only
        ridge1 = Ridge(alpha=10.0)
        ridge1.fit(train[["score_gap_23"]].to_numpy(), train["label_delta_swap"].to_numpy())
        pred1 = ridge1.predict(test[["score_gap_23"]].to_numpy())
        ridge2 = Ridge(alpha=10.0)
        ridge2.fit(train[["score_gap_23", "dis_gap"]].to_numpy(), train["label_delta_swap"].to_numpy())
        pred2 = ridge2.predict(test[["score_gap_23", "dis_gap"]].to_numpy())
        for i, rec in enumerate(test.itertuples(index=False)):
            exec_rows.append(
                {
                    "signal_date": str(pd.Timestamp(rec.signal_date).date()),
                    "trained_pair_id2": rec.trained_pair_id2,
                    "trained_pair_id3": rec.trained_pair_id3,
                    "pred_score_only": float(pred1[i]),
                    "pred_disagreement": float(pred2[i]),
                }
            )
    exec_df = pd.DataFrame(exec_rows)
    # apply swaps with assert same ids
    def apply_swap(pred_col: str) -> pd.DataFrame:
        scores = overlay.copy()
        scores["signal_date"] = pd.to_datetime(scores["signal_date"])
        out_parts = []
        n_swap = 0
        n_skip = 0
        for day, block in scores.groupby("signal_date", sort=True):
            block = block.sort_values(["score", "instrument_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
            day_exec = exec_df.loc[exec_df["signal_date"] == str(pd.Timestamp(day).date())]
            if day_exec.empty or len(block) < 3:
                out_parts.append(block)
                continue
            row = day_exec.iloc[0]
            id2, id3 = block.loc[1, "instrument_id"], block.loc[2, "instrument_id"]
            assert row["trained_pair_id2"] == id2 and row["trained_pair_id3"] == id3, "CH1 pair id mismatch"
            if float(row[pred_col]) > E4_TRIGGER_DELTA:
                # swap ranks by score nudge
                s2, s3 = float(block.loc[1, "score"]), float(block.loc[2, "score"])
                block.loc[1, "score"], block.loc[2, "score"] = s3 - 1e-12, s2 + 1e-12
                n_swap += 1
            else:
                n_skip += 1
            out_parts.append(block)
        out = pd.concat(out_parts, ignore_index=True)
        out.attrs["n_swap"] = n_swap
        out.attrs["n_skip"] = n_skip
        return out

    so = apply_swap("pred_score_only")
    dg = apply_swap("pred_disagreement")
    s_so = run_formal_sleeve(ROOT / "e4" / "E4_SCORE_ONLY", so, {"code": "R233_E4_SCORE_ONLY", "round": "23.3", "n_swap": so.attrs.get("n_swap")}, resume=resume, already_overlayed=True, top_changed=None, fee_same_direction=True, not_only_year=True)
    s_dg = run_formal_sleeve(ROOT / "e4" / "E4_DISAGREEMENT", dg, {"code": "R233_E4_DISAGREEMENT", "round": "23.3", "n_swap": dg.attrs.get("n_swap")}, resume=resume, already_overlayed=True, top_changed=None, fee_same_direction=True, not_only_year=True)
    write_df(ROOT / "challenger_pair_attribution.csv", exec_df)
    rows = [
        {"experiment": "E4_BASE", **{k: s_base.get(k) for k in ("sharpe", "cagr", "nav_end", "economically_material")}},
        {"experiment": "E4_SCORE_ONLY", **{k: s_so.get(k) for k in ("sharpe", "cagr", "nav_end", "economically_material")}},
        {"experiment": "E4_DISAGREEMENT", **{k: s_dg.get(k) for k in ("sharpe", "cagr", "nav_end", "economically_material")}},
    ]
    write_df(ROOT / "e4_results.csv", pd.DataFrame(rows))
    payload = {"status": "DONE", "n_pairs": int(len(pair_df)), "rows": rows}
    dump_json(ROOT / "e4.json", payload)
    return save_stage(ROOT, "e4", payload)


def run_outer_bt(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    """汇总已有正式袖套到 model_matrix。"""
    cached = _skip("outer_bt", resume)
    if cached:
        return cached
    parts = []
    for name in ("e1_results.csv", "e2_results.csv", "stacking_results.csv", "e4_results.csv"):
        path = ROOT / name
        if path.is_file():
            df = pd.read_csv(path)
            df["source"] = name
            parts.append(df)
    mat = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    write_df(ROOT / "model_matrix.csv", mat)
    # top1/top2 pnl attribution vs v5 for nested if present
    rows = []
    v5_nav = ROOT / "parity" / "V5" / "nav.parquet"
    if v5_nav.is_file():
        base = daily_from_nav(pd.read_parquet(v5_nav))
        for code, path in (
            ("E1_NESTED_SELECTED", ROOT / "e1" / "E1_NESTED_SELECTED" / "nav.parquet"),
            ("E2_NESTED_SELECTED", ROOT / "e2" / "E2_NESTED_SELECTED" / "nav.parquet"),
        ):
            if path.is_file():
                cand = daily_from_nav(pd.read_parquet(path))
                both = pd.concat([cand.rename("c"), base.rename("b")], axis=1, join="inner")
                rows.append({"code": code, "mean_delta": float((both["c"] - both["b"]).mean()), "n": int(len(both))})
    write_df(ROOT / "top1_top2_realized_pnl.csv", pd.DataFrame(rows) if rows else pd.DataFrame([{"status": "NO_NESTED"}]))
    return save_stage(ROOT, "outer_bt", {"status": "DONE", "n": int(len(mat))})


def run_stress(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("stress", resume)
    if cached:
        return cached
    v5 = _v5_fused()
    rows = []
    s0 = run_formal_sleeve(ROOT / "exec" / "S0", v5, {"code": "R233_S0", "round": "23.3"}, resume=resume, counts_as_alpha=False, top_changed=False, fee_same_direction=True, not_only_year=True)
    rows.append({"scenario": "S0", **{k: s0.get(k) for k in ("sharpe", "cagr", "nav_end", "completed_trades")}})
    for name, bps in (("S1", 10), ("S2", 20)):
        obs = run_round7(
            ROOT / "exec" / name,
            v5,
            {"code": f"R233_{name}", "round": "23.3", "scope": "EXECUTION_SCENARIO"},
            resume=resume,
            scenario=True,
            counts_as_alpha=False,
            apply_v5_gate=False,
            validation_hooks={"buy_extra_slippage_bps": bps},
        )
        rows.append({"scenario": name, "extra_buy_slippage_bps": bps, **{k: obs.get(k) for k in ("sharpe", "cagr", "nav_end", "completed_trades")}})
    # S3 fill-layer cancel, no re-rank
    overlay, _ = apply_b03_o01(v5)
    from core.r225241_sleeve import market as load_market

    prepared = load_market()["prepared"]
    calendar = pd.DatetimeIndex(prepared.calendar)
    pos = {pd.Timestamp(d): i for i, d in enumerate(calendar)}
    work = overlay.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort").copy()
    work["rk"] = work.groupby("signal_date").cumcount() + 1
    top = work.loc[work["rk"].le(2)].copy()
    top["entry_date"] = top["signal_date"].map(
        lambda day: calendar[pos[pd.Timestamp(day)] + 1] if pd.Timestamp(day) in pos and pos[pd.Timestamp(day)] + 1 < len(calendar) else pd.NaT
    )
    lookup = prepared.market_lookup
    sealed = []
    for rec in top.itertuples(index=False):
        flag = False
        if pd.notna(rec.entry_date):
            try:
                bar = lookup.loc[(pd.Timestamp(rec.entry_date), rec.instrument_id)]
                limit_up = float(bar["limit_up"]) if np.isfinite(bar["limit_up"]) and 0 < float(bar["limit_up"]) < 99_999 else np.nan
                flag = bool(np.isfinite(limit_up) and np.isfinite(bar["open"]) and float(bar["open"]) + 1e-6 >= limit_up - 0.005)
            except KeyError:
                flag = False
        sealed.append(flag)
    top["open_limit_up"] = sealed
    n_cancel = int(top["open_limit_up"].sum())
    s3 = top.loc[~top["open_limit_up"], [c for c in ("signal_date", "instrument_id", "score", "allocation_multiplier") if c in top.columns]].copy()
    if "allocation_multiplier" not in s3.columns:
        s3["allocation_multiplier"] = 1.0
    s3_out = run_formal_sleeve(
        ROOT / "exec" / "S3",
        s3,
        {"code": "R233_S3", "round": "23.3", "n_cancelled": n_cancel, "no_rerank_no_replacement": True},
        resume=resume,
        already_overlayed=True,
        scenario=True,
        counts_as_alpha=False,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    rows.append({"scenario": "S3", "n_cancelled": n_cancel, **{k: s3_out.get(k) for k in ("sharpe", "cagr", "nav_end", "completed_trades")}})
    write_df(ROOT / "execution_sensitivity.csv", pd.DataFrame(rows))
    return save_stage(ROOT, "stress", {"status": "DONE", "rows": rows})


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("stats", resume)
    if cached:
        return cached
    series = {"V5": daily_from_nav(pd.read_parquet(ROOT / "parity" / "V5" / "nav.parquet"))}
    for code, path in (
        ("E1_NESTED", ROOT / "e1" / "E1_NESTED_SELECTED" / "nav.parquet"),
        ("E2_NESTED", ROOT / "e2" / "E2_NESTED_SELECTED" / "nav.parquet"),
        ("E1_T075", ROOT / "e1" / "E1_T075" / "nav.parquet"),
        ("E2_L025", ROOT / "e2" / "E2_L025" / "nav.parquet"),
    ):
        if path.is_file():
            series[code] = daily_from_nav(pd.read_parquet(path))
    boots = {}
    for name, s in series.items():
        if name == "V5":
            continue
        boots[name] = paired_block_bootstrap_v2(s, series["V5"], n_boot=5000, block=20)
    panel = pd.DataFrame(series)
    spa = spa_hansen_or_blocked(panel.dropna(how="any"), baseline="V5") if not panel.empty else {"status": "NOT_IDENTIFIABLE"}
    rows = [{"name": k, "status": v.get("status"), **{kk: v.get(kk) for kk in ("delta_mean", "delta_sharpe", "ci_low", "ci_high")}} for k, v in boots.items()]
    rows.append({"name": "SPA", "status": spa.get("status")})
    write_df(ROOT / "bootstrap_spa.csv", pd.DataFrame(rows))
    dump_json(ROOT / "statistics.json", {"bootstrap": boots, "spa": spa})
    return save_stage(ROOT, "stats", {"status": "DONE", "spa": spa.get("status")})


def run_tests(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    import unittest
    from core.round23_3 import tests_round23_3 as tmod

    suite = unittest.defaultTestLoader.loadTestsFromModule(tmod)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # also inner window check
    inner = load_json(ROOT / "inner_fix.json")
    parity = load_json(ROOT / "parity.json")
    payload = {
        "status": "PASS" if result.wasSuccessful() and inner.get("status") == "PASS" and parity.get("status") == "PASS" else "FAIL",
        "unittest_failures": len(result.failures) + len(result.errors),
        "tests_run": result.testsRun,
        "inner_fix": inner.get("status"),
        "parity": parity.get("status"),
        "production_config_changed": False,
    }
    dump_json(ROOT / "tests_summary.json", payload)
    return save_stage(ROOT, "tests", payload)


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    mat = pd.read_csv(ROOT / "model_matrix.csv") if (ROOT / "model_matrix.csv").is_file() else pd.DataFrame()
    material = False
    best = None
    if len(mat) and "economically_material" in mat.columns:
        hit = mat.loc[mat["economically_material"].astype(str).isin(["True", "true", "1"])]
        material = len(hit) > 0
        if len(mat) and "sharpe" in mat.columns:
            # best matched candidate among non-baseline identities
            cand = mat.dropna(subset=["sharpe"]).copy()
            if len(cand):
                best_row = cand.sort_values("sharpe", ascending=False).iloc[0]
                best = best_row.to_dict()
    decision = "KEEP_BASELINE_V5" if not material else "V6_RESEARCH_CANDIDATE_NOT_PROMOTED"
    anchor = anchor_copy()
    delta_sharpe = None
    delta_cagr = None
    if best and best.get("sharpe") is not None:
        delta_sharpe = float(best["sharpe"]) - float(anchor["sharpe"])
        if best.get("cagr") is not None:
            delta_cagr = float(best["cagr"]) - float(anchor["cagr"])
    payload = {
        "decision": decision,
        "best_matched_candidate": best,
        "delta_sharpe": delta_sharpe,
        "delta_cagr": delta_cagr,
        "economic_gate": material,
        "current_baseline_changed": False,
        "auto_promote": False,
        "production_config_changed": False,
        "parity": load_json(ROOT / "parity.json").get("status"),
        "e3": (load_json(ROOT / "e3.json") if (ROOT / "e3.json").is_file() else load_stage(ROOT, "e3") or {}).get("status"),
    }
    dump_json(ROOT / "decision.json", payload)
    # all trials
    parts = []
    for name in ("model_matrix.csv", "execution_sensitivity.csv", "bootstrap_spa.csv"):
        path = ROOT / name
        if path.is_file():
            df = pd.read_csv(path)
            df["source_table"] = name
            parts.append(df)
    if parts:
        write_df(ROOT / "all_trials.csv", pd.concat(parts, ignore_index=True, sort=False))
    write_text(
        ROOT / "FINAL_DECISION.md",
        "\n".join(
            [
                "# Round23.3 最终裁决",
                "",
                f"- Parity：`{payload['parity']}`",
                f"- 决策：`{decision}`",
                f"- 最佳匹配候选：`{(best or {}).get('experiment')}`",
                f"- ΔSharpe：`{delta_sharpe}`",
                f"- ΔCAGR：`{delta_cagr}`",
                f"- 经济门：`{material}`",
                "- CURRENT_BASELINE_CHANGED=false",
                "- PRODUCTION_CONFIG_CHANGED=false",
                "- AUTO_PROMOTE=false",
                "",
            ]
        ),
    )
    return save_stage(ROOT, "decision", payload)


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dec = load_json(ROOT / "decision.json")
    parity = load_json(ROOT / "parity.json")
    inner = load_json(ROOT / "inner_fix.json")
    e1 = load_json(ROOT / "e1.json") if (ROOT / "e1.json").is_file() else {}
    e2 = load_json(ROOT / "e2.json") if (ROOT / "e2.json").is_file() else {}
    e3 = load_stage(ROOT, "e3") or {}
    e4 = load_json(ROOT / "e4.json") if (ROOT / "e4.json").is_file() else {}
    lines = [
        "# Round23.3 研究报告",
        "",
        "## 结论",
        f"决策：**{dec.get('decision')}**。Parity=`{parity.get('status')}`。Inner 窗修复=`{inner.get('status')}`。",
        "本轮只在冻结 baseline_v5 上检验 E1/E2/E3/E4 的条件增量；不自动升级生产配置。",
        "",
        "## 修复",
        "- Inner Replay 适配器传入 backtest_start/end 与 signal_start/end；半年块若出现全历史 NAV 日则 FAIL。",
        "- 默认权重不再因 distance=0 永远获胜；仅非默认且过全部门槛者竞争。",
        "- w_lgbm=0 直接返回冻结 v5 分数；LGBM 缺失按日回退并记账，不静默缩小候选池。",
        "",
        "## E1 Five/Ten 权重",
        f"- 状态：`{e1.get('status')}`；详见 `e1_results.csv` / `selected_weights.csv`。",
        "",
        "## E2 XGB/LGBM",
        f"- 状态：`{e2.get('status')}`；详见 `e2_results.csv`。",
        "",
        "## E3 Stacking",
        f"- 状态：`{e3.get('status')}`；无条件增量则 NOT_TRIGGERED。",
        "",
        "## E4 局部纠错",
        f"- 状态：`{e4.get('status')}`；CH1 要求 trained_pair_id2/3 与执行一致。",
        "",
        "## 统计与执行压力",
        "- Bootstrap 5000 × block20 见 `bootstrap_spa.csv`。",
        "- S0–S3 见 `execution_sensitivity.csv`（S3 成交层取消，不重排不递补）。",
        "",
        "## 冻结标志",
        "- CURRENT_BASELINE_CHANGED=FALSE",
        "- PRODUCTION_CONFIG_CHANGED=FALSE",
        "- AUTO_PROMOTE=FALSE",
        "",
    ]
    write_text(ROOT / "ROUND23_3_RESEARCH_REPORT.md", "\n".join(lines))
    return save_stage(ROOT, "report", {"status": "DONE"})


STAGES = {
    "source_audit": run_source_audit,
    "freeze": run_freeze,
    "parity": run_parity,
    "base_oof": run_base_oof,
    "inner_fix": run_inner_fix,
    "e1": run_e1,
    "e2": run_e2,
    "cond_inc": run_cond_inc,
    "e3": run_e3,
    "e4": run_e4,
    "outer_bt": run_outer_bt,
    "stress": run_stress,
    "stats": run_stats,
    "tests": run_tests,
    "decision": run_decision,
    "report": run_report,
}


def run_all(*, resume: bool = True, jobs: int = 2, **kwargs: Any) -> dict[str, Any]:
    out = {}
    for name, fn in STAGES.items():
        print(f"[round23_3] stage={name}", flush=True)
        out[name] = fn(resume=resume, jobs=jobs, **kwargs)
    return out
