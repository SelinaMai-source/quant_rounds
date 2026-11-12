"""Round 22.6 主流程。不扩因子库，不改 C4 公式，不自动晋级 CURRENT。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.r225241_models import train_xgb_ranker
from core.r225241_stats import daily_spearman
from core.research_integrity_v2.adapters import fee_components, source_hashes
from core.research_integrity_v2.cache_identity import file_sha256
from core.research_integrity_v2.economic_gate import freeze_economic_gate
from core.research_integrity_v2.io_v2 import dump_json, load_json, load_stage, save_stage, sha256_json, write_df, write_text
from core.research_integrity_v2.merge_rank import explicit_merge, rank_unique
from core.research_integrity_v2.stats_v2 import paired_block_bootstrap_v2, spa_hansen_or_blocked
from core.research_integrity_v2.tests_failure_cases import run_unit_tests
from core.research_integrity_v2.time_contract import LEGACY_OPEN_INFORMATION_LANE, shanghai_now
from core.round11.common import daily_from_nav
from core.round22_6.contract import (
    C4,
    C4_FORMULA,
    CHALLENGER_BUFFER,
    CHALLENGER_MIN_TRAIN_PAIR_DAYS,
    E0,
    FIVE_FEATURES,
    FS2,
    FS2_FORMULA,
    FS2_PANEL,
    HYPOTHESES,
    PANEL,
    PERM_SEEDS,
    RIDGE_ALPHA,
    ROOT,
    R22_3,
    R22_5,
    TEN_FEATURES,
    TEN_PANEL,
    TEN_TRAIN,
    TRAIN,
    anchor_copy,
)
from core.round22_6.sleeve import run_formal_sleeve
from core.round7.contract import FIVE_HEAD, TEN_HEAD


def _skip(name: str, resume: bool) -> dict[str, Any] | None:
    return load_stage(ROOT, name) if resume else None


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (ROOT / "frozen_manifest.json").is_file():
        return load_json(ROOT / "frozen_manifest.json")
    body = {
        "round": "22.6",
        "frozen_before_new_nav": True,
        "freeze_timestamp": shanghai_now().isoformat(),
        "anchor": anchor_copy(),
        "hypotheses": list(HYPOTHESES),
        "hypotheses_locked": True,
        "fs2_formula": FS2_FORMULA,
        "fs2_is_not_cash_conversion_change": True,
        "c4_formula_unchanged": C4_FORMULA,
        "c1_archived_no_extension": True,
        "permutation_seeds": list(PERM_SEEDS),
        "challenger_buffer": CHALLENGER_BUFFER,
        "challenger_min_train_pair_days": CHALLENGER_MIN_TRAIN_PAIR_DAYS,
        "ridge_alpha": RIDGE_ALPHA,
        "information_lane": LEGACY_OPEN_INFORMATION_LANE,
        "no_new_factors": True,
        "no_xgb_retune": True,
        "current_baseline_changed": False,
        "auto_promote": False,
        "source_hashes": source_hashes(),
        "fee_components": fee_components(),
    }
    dump_json(ROOT / "frozen_manifest.json", body)
    freeze_economic_gate(ROOT / "economic_gate_freeze.json")
    return save_stage(ROOT, "freeze", {**body, "status": "FROZEN"})


def _attach_fs2(frame: pd.DataFrame, fs: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["instrument_id"] = out["instrument_id"].astype(str)
    extra = fs[["trade_date", "instrument_id", FS2]].copy()
    out = out.drop(columns=[FS2], errors="ignore")
    merged = explicit_merge(out, extra, on=["trade_date", "instrument_id"], how="left", validate="one_to_one")
    return merged


def _load_fs2_pit() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not FS2_PANEL.is_file():
        return pd.DataFrame(), {"status": "MISSING_ARTIFACT", "pit_pass": False}
    fs = pd.read_parquet(FS2_PANEL)
    fs["trade_date"] = pd.to_datetime(fs["trade_date"])
    fs["known_at"] = pd.to_datetime(fs["known_at"])
    fs["instrument_id"] = fs["instrument_id"].astype(str)
    meta = {
        "formula": FS2_FORMULA,
        "panel_path": str(FS2_PANEL),
        "panel_sha256": file_sha256(FS2_PANEL),
        "columns": list(fs.columns),
    }
    for col in ("report_id", "period", "statement_basis", "revision", "unit", "average_assets", "operating_cash_flow"):
        meta[f"has_{col}"] = col in fs.columns
    future = int((fs["known_at"] > fs["trade_date"]).sum())
    fs = fs.loc[fs["known_at"].notna() & fs["known_at"].le(fs["trade_date"])].copy()
    fs = fs.sort_values("known_at").groupby(["trade_date", "instrument_id"], as_index=False).tail(1)
    meta.update(
        {
            "future_known_at_rows_before_filter": future,
            "asof": "known_at<=trade_date then latest known_at",
            "filled_with_zero": False,
            "pit_pass": future == 0 and bool(fs[FS2].notna().any()),
            "n_rows": int(len(fs)),
        }
    )
    return fs, meta


def _fold_slices(year: int, label_end: pd.Series, trade_date: pd.Series) -> pd.Series:
    dates = pd.to_datetime(trade_date)
    end = pd.to_datetime(label_end)
    return dates.dt.year.between(year - 3, year - 1) & end.dt.year.le(year - 1)


def _ridge_residual_ic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    x_cols: list[str],
    factor: str,
    label: str,
) -> dict[str, Any]:
    use_cols = list(x_cols)
    tr = train.dropna(subset=[label]).copy()
    if len(tr) < 50 or test.empty:
        return {"status": "NOT_IDENTIFIABLE", "reason": "insufficient_rows"}
    x_tr = tr[use_cols].to_numpy(dtype=float)
    f_tr = pd.to_numeric(tr[factor], errors="coerce").to_numpy(dtype=float)
    y_tr = pd.to_numeric(tr[label], errors="coerce").to_numpy(dtype=float)
    miss_f = ~np.isfinite(f_tr)
    f_tr = np.where(miss_f, np.nan, f_tr)
    pipe_x = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    x_design = pipe_x.fit_transform(x_tr)
    # factor residual model
    f_imp = SimpleImputer(strategy="median").fit_transform(f_tr.reshape(-1, 1)).ravel()
    ridge_f = Ridge(alpha=RIDGE_ALPHA).fit(x_design, f_imp)
    ridge_y = Ridge(alpha=RIDGE_ALPHA).fit(x_design, y_tr)
    te = test.copy()
    x_te = pipe_x.transform(te[use_cols].to_numpy(dtype=float))
    f_te = pd.to_numeric(te[factor], errors="coerce").to_numpy(dtype=float)
    y_te = pd.to_numeric(te[label], errors="coerce").to_numpy(dtype=float)
    f_te_imp = np.where(np.isfinite(f_te), f_te, np.nanmedian(f_tr[np.isfinite(f_tr)]) if np.isfinite(f_tr).any() else 0.0)
    u = f_te_imp - ridge_f.predict(x_te)
    v = y_te - ridge_y.predict(x_te)
    te = te.copy()
    te["u_resid"] = u
    te["v_resid"] = v
    te["raw_factor"] = f_te
    te["raw_label"] = y_te
    daily = []
    for day, block in te.groupby("trade_date", sort=True):
        sub = block.dropna(subset=["u_resid", "v_resid"])
        if len(sub) < 5:
            daily.append({"signal_date": str(pd.Timestamp(day).date()), "status": "NOT_IDENTIFIABLE", "n": int(len(sub))})
            continue
        if float(sub["u_resid"].std(ddof=0) or 0) == 0 or float(sub["v_resid"].std(ddof=0) or 0) == 0:
            daily.append({"signal_date": str(pd.Timestamp(day).date()), "status": "NOT_IDENTIFIABLE", "reason": "constant", "n": int(len(sub))})
            continue
        pear = float(np.corrcoef(sub["u_resid"], sub["v_resid"])[0, 1])
        spear = float(sub["u_resid"].corr(sub["v_resid"], method="spearman"))
        raw = float(pd.Series(sub["raw_factor"]).corr(pd.Series(sub["raw_label"]), method="spearman")) if sub["raw_factor"].notna().sum() >= 5 else None
        resid_to_raw = float(pd.Series(sub["u_resid"]).corr(pd.Series(sub["raw_label"]), method="spearman")) if sub["raw_label"].notna().sum() >= 5 else None
        daily.append(
            {
                "signal_date": str(pd.Timestamp(day).date()),
                "status": "OK",
                "n": int(len(sub)),
                "pearson_uv": pear,
                "spearman_uv": spear,
                "raw_rank_ic": raw,
                "residual_factor_to_raw_label_IC": resid_to_raw,
            }
        )
    ok = [d for d in daily if d.get("status") == "OK"]
    return {
        "status": "OK" if ok else "NOT_IDENTIFIABLE",
        "diagnostic_name": "LINEAR_PARTIAL_INFORMATION_DIAGNOSTIC",
        "n_days_ok": len(ok),
        "mean_spearman_uv": float(np.mean([d["spearman_uv"] for d in ok])) if ok else None,
        "mean_pearson_uv": float(np.mean([d["pearson_uv"] for d in ok])) if ok else None,
        "mean_raw_rank_ic": float(np.nanmean([d["raw_rank_ic"] for d in ok if d.get("raw_rank_ic") is not None])) if ok else None,
        "daily": daily,
    }


def run_partial_ic(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("partial_ic", resume)
    if cached:
        return cached
    fs, pit = _load_fs2_pit()
    five = _attach_fs2(pd.read_parquet(PANEL), fs) if pit.get("pit_pass") else pd.DataFrame()
    ten = _attach_fs2(pd.read_parquet(TEN_PANEL), fs) if pit.get("pit_pass") else pd.DataFrame()
    five_train = _attach_fs2(pd.read_parquet(TRAIN), fs) if pit.get("pit_pass") else pd.DataFrame()
    ten_train = _attach_fs2(pd.read_parquet(TEN_TRAIN), fs) if pit.get("pit_pass") else pd.DataFrame()
    rows = []
    summaries = {}
    if pit.get("pit_pass"):
        for head, panel, train, feats, label, label_end in (
            ("five", five, five_train, list(FIVE_FEATURES), FIVE_HEAD.target, FIVE_HEAD.label_end_column),
            ("ten", ten, ten_train, list(TEN_FEATURES), TEN_HEAD.target, TEN_HEAD.label_end_column),
        ):
            panel = panel.copy()
            train = train.copy()
            panel["trade_date"] = pd.to_datetime(panel["trade_date"])
            train["trade_date"] = pd.to_datetime(train["trade_date"])
            train[label_end] = pd.to_datetime(train[label_end])
            year_rows = []
            for year in range(2018, 2027):
                tr_mask = _fold_slices(year, train[label_end], train["trade_date"])
                tr = train.loc[tr_mask]
                te = panel.loc[panel["trade_date"].dt.year.eq(year)]
                diag = _ridge_residual_ic(tr, te, x_cols=feats, factor=FS2, label=label)
                year_rows.append({"head": head, "predict_year": year, **{k: diag.get(k) for k in ("status", "n_days_ok", "mean_spearman_uv", "mean_pearson_uv", "mean_raw_rank_ic")}})
                for day in diag.get("daily") or []:
                    rows.append({"head": head, "predict_year": year, **day})
            summaries[head] = {
                "years": year_rows,
                "raw_rank_ic_full": daily_spearman(panel.rename(columns={"trade_date": "signal_date"}), FS2, label),
                "note": "raw_rank_ic 不是 conditional_ic；条件诊断名为 LINEAR_PARTIAL_INFORMATION_DIAGNOSTIC",
            }
    payload = {
        "pit": pit,
        "diagnostic_name": "LINEAR_PARTIAL_INFORMATION_DIAGNOSTIC",
        "renamed_from_misuse": "daily_spearman_formerly_called_conditional_ic",
        "summaries": summaries,
        "status": "DONE" if pit.get("pit_pass") else "BLOCKED",
    }
    write_df(ROOT / "raw_vs_partial_ic.csv", pd.DataFrame(rows))
    dump_json(ROOT / "fs2_partial_information.json", payload)
    return save_stage(ROOT, "partial_ic", payload)


def _fuse(five: pd.DataFrame, ten: pd.DataFrame) -> pd.DataFrame:
    from core.round17.backtest import fuse_heads

    return fuse_heads(five, ten)


def _tops(pred: pd.DataFrame, k: int) -> pd.DataFrame:
    ranked, _ = rank_unique(pred)
    return ranked.loc[ranked["raw_rank"].le(k), ["signal_date", "instrument_id", "raw_rank", "score"]].rename(columns={"raw_rank": "rk"})


def _delta_topk(base_pred: pd.DataFrame, cand_pred: pd.DataFrame, labels: pd.DataFrame, label_col: str) -> dict[str, Any]:
    lab = labels.copy()
    lab["signal_date"] = pd.to_datetime(lab["trade_date"] if "trade_date" in lab.columns else lab["signal_date"])
    out = {}
    for k, name in ((1, "top1"), (2, "top2")):
        a = _tops(base_pred, k)
        b = _tops(cand_pred, k)
        a_m = a.merge(lab[["signal_date", "instrument_id", label_col]], on=["signal_date", "instrument_id"], how="left")
        b_m = b.merge(lab[["signal_date", "instrument_id", label_col]], on=["signal_date", "instrument_id"], how="left")
        base_mean = float(pd.to_numeric(a_m[label_col], errors="coerce").mean())
        cand_mean = float(pd.to_numeric(b_m[label_col], errors="coerce").mean())
        out[name] = {
            "base_mean": base_mean,
            "cand_mean": cand_mean,
            "delta": cand_mean - base_mean,
            "n": int(len(b_m)),
        }
    return out


def run_fs2_heads(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    cached = _skip("fs2_heads", resume)
    if cached:
        return cached
    fs, pit = _load_fs2_pit()
    if not pit.get("pit_pass"):
        payload = {"status": "BLOCKED", "reason": "FS2_PIT", "pit": pit}
        return save_stage(ROOT, "fs2_heads", payload)
    five = _attach_fs2(pd.read_parquet(PANEL), fs)
    ten = _attach_fs2(pd.read_parquet(TEN_PANEL), fs)
    five_train = _attach_fs2(pd.read_parquet(TRAIN), fs)
    ten_train = _attach_fs2(pd.read_parquet(TEN_TRAIN), fs)
    # F0 = frozen v5
    five_v5 = pd.read_parquet(E0 / "five_pred.parquet")
    ten_v5 = pd.read_parquet(E0 / "ten_pred.parquet")
    for frame in (five_v5, ten_v5):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    f0 = run_formal_sleeve(
        ROOT / "heads" / "F0_v5",
        _fuse(five_v5, ten_v5),
        {"code": "R226_F0_V5", "round": "22.6", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        top_changed=False,
        fee_same_direction=True,
        not_only_year=True,
        counts_as_alpha=False,
    )
    results_by_head: dict[str, Any] = {}
    # F1 Five+FS2, Ten frozen
    train_xgb_ranker(
        training=five_train,
        candidates=five,
        features=list(FIVE_FEATURES) + [FS2],
        extras=[FS2],
        target=FIVE_HEAD.target,
        label_end_column=FIVE_HEAD.label_end_column,
        n_bins=FIVE_HEAD.rank_n_bins,
        eval_metric=FIVE_HEAD.eval_metric,
        output_dir=ROOT / "fs2" / "five_add",
        seed=17,
        n_jobs=jobs,
        resume=resume,
    )
    five_f1 = pd.read_parquet(ROOT / "fs2" / "five_add" / "predictions.parquet")
    five_f1["signal_date"] = pd.to_datetime(five_f1["signal_date"])
    f1 = run_formal_sleeve(
        ROOT / "heads" / "F1_five_fs2",
        _fuse(five_f1, ten_v5),
        {"code": "R226_F1_FIVE_FS2", "round": "22.6", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    results_by_head["five"] = {
        "code": "R226_F1_FIVE_FS2",
        "summary": f1,
        "delta_vs_f0": {
            "delta_sharpe": None if f1.get("sharpe") is None else float(f1["sharpe"]) - float(f0["sharpe"]),
            "delta_cagr": None if f1.get("cagr") is None else float(f1["cagr"]) - float(f0["cagr"]),
            "delta_maxdd": None if f1.get("max_drawdown") is None else float(f1["max_drawdown"]) - float(f0["max_drawdown"]),
            "nav_improvement": f1.get("nav_improvement"),
        },
        "topk_delta": _delta_topk(five_v5, five_f1, five, FIVE_HEAD.target),
    }
    # F2 Ten+FS2, Five frozen — NO early break
    train_xgb_ranker(
        training=ten_train,
        candidates=ten,
        features=list(TEN_FEATURES) + [FS2],
        extras=[FS2],
        target=TEN_HEAD.target,
        label_end_column=TEN_HEAD.label_end_column,
        n_bins=TEN_HEAD.rank_n_bins,
        eval_metric=TEN_HEAD.eval_metric,
        output_dir=ROOT / "fs2" / "ten_add",
        seed=17,
        n_jobs=jobs,
        resume=resume,
    )
    ten_f2 = pd.read_parquet(ROOT / "fs2" / "ten_add" / "predictions.parquet")
    ten_f2["signal_date"] = pd.to_datetime(ten_f2["signal_date"])
    f2 = run_formal_sleeve(
        ROOT / "heads" / "F2_ten_fs2",
        _fuse(five_v5, ten_f2),
        {"code": "R226_F2_TEN_FS2", "round": "22.6", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    results_by_head["ten"] = {
        "code": "R226_F2_TEN_FS2",
        "summary": f2,
        "delta_vs_f0": {
            "delta_sharpe": None if f2.get("sharpe") is None else float(f2["sharpe"]) - float(f0["sharpe"]),
            "delta_cagr": None if f2.get("cagr") is None else float(f2["cagr"]) - float(f0["cagr"]),
            "delta_maxdd": None if f2.get("max_drawdown") is None else float(f2["max_drawdown"]) - float(f0["max_drawdown"]),
            "nav_improvement": f2.get("nav_improvement"),
        },
        "topk_delta": _delta_topk(ten_v5, ten_f2, ten, TEN_HEAD.target),
    }
    improved = any(
        bool((results_by_head[h]["summary"] or {}).get("economically_material")) for h in ("five", "ten")
    )
    verdict = "FS2_DIRECT_ADD_CANDIDATE" if improved else "FS2_DIRECT_ADD_NOT_SUPPORTED"
    rows = []
    for code, summary, head in (
        ("R226_F0_V5", f0, "F0"),
        ("R226_F1_FIVE_FS2", f1, "F1"),
        ("R226_F2_TEN_FS2", f2, "F2"),
    ):
        rows.append(
            {
                "code": code,
                "head": head,
                "sharpe": summary.get("sharpe"),
                "cagr": summary.get("cagr"),
                "max_drawdown": summary.get("max_drawdown"),
                "nav_end": summary.get("nav_end"),
                "nav_improvement": summary.get("nav_improvement"),
                "economically_material": summary.get("economically_material"),
                "completed_trades": summary.get("completed_trades"),
            }
        )
    write_df(ROOT / "fs2_head_backtests.csv", pd.DataFrame(rows))
    oof_rows = []
    for head, path in (("five", ROOT / "fs2" / "five_add" / "predictions.parquet"), ("ten", ROOT / "fs2" / "ten_add" / "predictions.parquet")):
        pred = pd.read_parquet(path)
        oof_rows.append({"head": head, "rows": int(len(pred)), "path": str(path), "sha256": file_sha256(path)})
    write_df(ROOT / "fs2_paired_oof.csv", pd.DataFrame(oof_rows))
    payload = {
        "status": "DONE",
        "verdict": verdict,
        "f0": f0,
        "results_by_head": {
            "five": {k: results_by_head["five"][k] for k in ("code", "delta_vs_f0", "topk_delta")},
            "ten": {k: results_by_head["ten"][k] for k in ("code", "delta_vs_f0", "topk_delta")},
        },
        "f1_economically_material": f1.get("economically_material"),
        "f2_economically_material": f2.get("economically_material"),
        "no_early_break": True,
        "screening_not_own_mean_gt0": True,
    }
    dump_json(ROOT / "fs2_heads.json", payload)
    return save_stage(ROOT, "fs2_heads", payload)


def _permute_c4_within_date(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    rng = np.random.default_rng(seed)
    values = pd.to_numeric(out[C4], errors="coerce").to_numpy(copy=True)
    for day, idx in out.groupby("trade_date", sort=False).groups.items():
        index = list(idx)
        finite = [i for i in index if np.isfinite(values[i])]
        if len(finite) < 2:
            continue
        shuffled = values[finite].copy()
        rng.shuffle(shuffled)
        values[finite] = shuffled
    out[C4] = values
    # ensure other five features untouched; score/allocation not present here
    return out


def run_c4_placebo(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    cached = _skip("c4_placebo", resume)
    if cached:
        return cached
    five = pd.read_parquet(PANEL)
    five_train = pd.read_parquet(TRAIN)
    ten_v5 = pd.read_parquet(E0 / "ten_pred.parquet")
    ten_v5["signal_date"] = pd.to_datetime(ten_v5["signal_date"])
    # rename old score/allocation shuffle
    old_null = load_json(R22_5 / "null_recheck.json") if (R22_5 / "null_recheck.json").is_file() else {}
    portfolio_randomization = {
        "name": "PORTFOLIO_SCORE_ALLOCATION_RANDOMIZATION",
        "not_c4_feature_evidence": True,
        "source_round22_5": old_null.get("permutations"),
        "frozen_shuffle_sharpe_note": "旧打乱最终分数/仓位实验保留并改名，不再当作 C4 因子级证据",
    }
    feats = list(FIVE_FEATURES) + ([C4] if C4 not in FIVE_FEATURES else [])
    # true C4 sleeve from historical E1 if present else retrain
    true_pred_path = R22_3 / "candidates" / f"R223_E1_{C4}" / "five_pred.parquet"
    if true_pred_path.is_file():
        five_true = pd.read_parquet(true_pred_path)
    else:
        train_xgb_ranker(
            training=five_train,
            candidates=five,
            features=feats,
            extras=[C4],
            target=FIVE_HEAD.target,
            label_end_column=FIVE_HEAD.label_end_column,
            n_bins=FIVE_HEAD.rank_n_bins,
            eval_metric=FIVE_HEAD.eval_metric,
            output_dir=ROOT / "c4" / "true",
            seed=17,
            n_jobs=jobs,
            resume=resume,
        )
        five_true = pd.read_parquet(ROOT / "c4" / "true" / "predictions.parquet")
    five_true["signal_date"] = pd.to_datetime(five_true["signal_date"])
    true_sleeve = run_formal_sleeve(
        ROOT / "c4" / "TRUE_C4",
        _fuse(five_true, ten_v5),
        {"code": "R226_C4_TRUE", "round": "22.6", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
        counts_as_alpha=False,
    )
    rows = [{"code": "R226_C4_TRUE", "kind": "TRUE_C4", "seed": None, "sharpe": true_sleeve.get("sharpe"), "cagr": true_sleeve.get("cagr"), "nav_end": true_sleeve.get("nav_end"), "nav_improvement": true_sleeve.get("nav_improvement")}]
    v5 = load_json(E0 / "summary.json")
    rows.append({"code": "R16_O01", "kind": "V5", "seed": None, "sharpe": v5.get("sharpe"), "cagr": v5.get("cagr"), "nav_end": v5.get("nav_end"), "nav_improvement": 0.0})
    for seed in PERM_SEEDS:
        train_p = _permute_c4_within_date(five_train, seed)
        cand_p = _permute_c4_within_date(five, seed)
        # assert other features identical
        for col in FIVE_FEATURES:
            if col == C4:
                continue
            if not np.allclose(
                pd.to_numeric(five[col], errors="coerce").fillna(0).to_numpy(),
                pd.to_numeric(cand_p[col], errors="coerce").fillna(0).to_numpy(),
                equal_nan=True,
            ):
                raise RuntimeError(f"placebo altered non-C4 feature {col}")
        train_xgb_ranker(
            training=train_p,
            candidates=cand_p,
            features=feats,
            extras=[C4],
            target=FIVE_HEAD.target,
            label_end_column=FIVE_HEAD.label_end_column,
            n_bins=FIVE_HEAD.rank_n_bins,
            eval_metric=FIVE_HEAD.eval_metric,
            output_dir=ROOT / "c4" / f"placebo_{seed}",
            seed=17,
            n_jobs=jobs,
            resume=resume,
        )
        pred = pd.read_parquet(ROOT / "c4" / f"placebo_{seed}" / "predictions.parquet")
        pred["signal_date"] = pd.to_datetime(pred["signal_date"])
        sleeve = run_formal_sleeve(
            ROOT / "c4" / f"PLACEBO_{seed}",
            _fuse(pred, ten_v5),
            {"code": f"R226_C4_PLACEBO_{seed}", "round": "22.6", "scope": "DIAGNOSTIC_ONLY", "seed": seed},
            resume=resume,
            scenario=False,
            counts_as_alpha=False,
            top_changed=None,
            fee_same_direction=True,
            not_only_year=True,
        )
        rows.append({"code": f"R226_C4_PLACEBO_{seed}", "kind": "FEATURE_PLACEBO", "seed": seed, "sharpe": sleeve.get("sharpe"), "cagr": sleeve.get("cagr"), "nav_end": sleeve.get("nav_end"), "nav_improvement": sleeve.get("nav_improvement")})
    write_df(ROOT / "c4_feature_placebo.csv", pd.DataFrame(rows))
    payload = {
        "status": "DONE",
        "c4_formula_unchanged": C4_FORMULA,
        "placebo": "within-date finite C4 only",
        "seeds": list(PERM_SEEDS),
        "no_pseudo_exact_pvalue": True,
        "portfolio_score_allocation_randomization": portfolio_randomization,
        "rows": rows,
    }
    dump_json(ROOT / "c4_placebo.json", payload)
    return save_stage(ROOT, "c4_placebo", payload)


def _pair_day_features(work: pd.DataFrame, fs_map: pd.DataFrame) -> pd.DataFrame:
    """从冻结 v5 OOF 融合分构造每日 rank2/3 配对。"""
    ranked, _ = rank_unique(work.rename(columns={"blend": "score"})[["signal_date", "instrument_id", "score", "pct_five", "pct_ten", "five", "ten"]])
    ranked = ranked.rename(columns={"score": "blend"})
    rows = []
    fs_map = fs_map.copy()
    fs_map["trade_date"] = pd.to_datetime(fs_map["trade_date"])
    fs_map["instrument_id"] = fs_map["instrument_id"].astype(str)
    for day, block in ranked.groupby("signal_date", sort=True):
        block = block.sort_values("raw_rank")
        if len(block) < 3:
            continue
        r2 = block.iloc[1]
        r3 = block.iloc[2]
        fs2 = fs_map.loc[(fs_map["trade_date"] == pd.Timestamp(day))]
        f2 = fs2.loc[fs2["instrument_id"] == r2["instrument_id"], FS2]
        f3 = fs2.loc[fs2["instrument_id"] == r3["instrument_id"], FS2]
        fs2_2 = float(f2.iloc[0]) if len(f2) and np.isfinite(f2.iloc[0]) else np.nan
        fs2_3 = float(f3.iloc[0]) if len(f3) and np.isfinite(f3.iloc[0]) else np.nan
        rows.append(
            {
                "signal_date": pd.Timestamp(day),
                "id2": r2["instrument_id"],
                "id3": r3["instrument_id"],
                "five_pct_diff": float(r3["pct_five"] - r2["pct_five"]),
                "ten_pct_diff": float(r3["pct_ten"] - r2["pct_ten"]),
                "blend_diff": float(r3["blend"] - r2["blend"]),
                "fs2_diff": fs2_3 - fs2_2 if np.isfinite(fs2_2) and np.isfinite(fs2_3) else np.nan,
                "fs2_2": fs2_2,
                "fs2_3": fs2_3,
                "fs2_2_miss": int(not np.isfinite(fs2_2)),
                "fs2_3_miss": int(not np.isfinite(fs2_3)),
                "score2": float(r2["blend"]),
                "score3": float(r3["blend"]),
            }
        )
    return pd.DataFrame(rows)


def run_challenger(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("challenger", resume)
    if cached:
        return cached
    fs, pit = _load_fs2_pit()
    five = pd.read_parquet(E0 / "five_pred.parquet").rename(columns={"score": "five"})
    ten = pd.read_parquet(E0 / "ten_pred.parquet").rename(columns={"score": "ten"})
    for frame in (five, ten):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    labels = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", "label_e2_net", "actual_exit_date"])
    labels["signal_date"] = pd.to_datetime(labels["trade_date"])
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    work = explicit_merge(five, ten, on=["signal_date", "instrument_id"], how="inner", validate="one_to_one")
    work["pct_five"] = work.groupby("signal_date")["five"].rank(method="average", pct=True)
    work["pct_ten"] = work.groupby("signal_date")["ten"].rank(method="average", pct=True)
    work["blend"] = 0.90 * work["pct_ten"] + 0.10 * work["pct_five"]
    pairs = _pair_day_features(work, fs if pit.get("pit_pass") else pd.DataFrame(columns=["trade_date", "instrument_id", FS2]))
    # action payoff from matured labels; unfilled -> 0 with flag (here label NA => UNFILLED/ IMMATURE)
    lab = labels.rename(columns={"instrument_id": "id"})
    p2 = pairs.merge(lab.rename(columns={"id": "id2", "label_e2_net": "pay2", "actual_exit_date": "exit2"}), on=["signal_date", "id2"], how="left")
    p2 = p2.merge(lab.rename(columns={"id": "id3", "label_e2_net": "pay3", "actual_exit_date": "exit3"}), on=["signal_date", "id3"], how="left")
    p2["action2"] = np.where(p2["exit2"].notna(), pd.to_numeric(p2["pay2"], errors="coerce"), 0.0)
    p2["action3"] = np.where(p2["exit3"].notna(), pd.to_numeric(p2["pay3"], errors="coerce"), 0.0)
    p2["flag2"] = np.where(p2["exit2"].isna(), "UNFILLED_OR_IMMATURE_ACTION", "FILLED")
    p2["flag3"] = np.where(p2["exit3"].isna(), "UNFILLED_OR_IMMATURE_ACTION", "FILLED")
    p2["y"] = p2["action3"] - p2["action2"]
    p2["label_available_at"] = pd.concat([pd.to_datetime(p2["exit2"]), pd.to_datetime(p2["exit3"])], axis=1).max(axis=1)
    mature = p2["exit2"].notna() & p2["exit3"].notna()
    write_df(ROOT / "challenger_pairs_manifest.csv", p2)
    # meta OOF by year using prior mature pairs only
    ch1_feats = ["five_pct_diff", "ten_pct_diff", "blend_diff"]
    ch2_feats = ch1_feats + ["fs2_diff", "fs2_2_miss", "fs2_3_miss"]
    oof_rows = []
    swap_ch1: dict[pd.Timestamp, bool] = {}
    swap_ch2: dict[pd.Timestamp, bool] = {}
    for year in range(2018, 2027):
        train = p2.loc[mature & p2["signal_date"].dt.year.between(year - 3, year - 1) & p2["label_available_at"].dt.year.le(year - 1)].copy()
        train = train.loc[pd.to_numeric(train["y"], errors="coerce").notna()].copy()
        test = p2.loc[p2["signal_date"].dt.year.eq(year)].copy()
        for tag, feats in (("CH1", ch1_feats), ("CH2", ch2_feats)):
            fallback = True
            pred = np.zeros(len(test), dtype=float)
            if int(train["signal_date"].nunique()) >= CHALLENGER_MIN_TRAIN_PAIR_DAYS and len(test):
                x_tr = train[feats].to_numpy(dtype=float)
                y_tr = pd.to_numeric(train["y"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(y_tr)
                if int(ok.sum()) >= CHALLENGER_MIN_TRAIN_PAIR_DAYS:
                    pipe = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
                        ]
                    )
                    pipe.fit(x_tr[ok], y_tr[ok])
                    pred = pipe.predict(test[feats].to_numpy(dtype=float))
                    fallback = False
                else:
                    fallback = True
            for i, (idx, row) in enumerate(test.iterrows()):
                day = pd.Timestamp(row["signal_date"])
                inputs_ok = True
                if tag == "CH2":
                    inputs_ok = bool(np.isfinite(row["fs2_diff"])) or (int(row["fs2_2_miss"]) + int(row["fs2_3_miss"]) > 0)
                    # 缺失指示可用；若两只都缺且 diff 非有限，仍允许模型用 miss 指示，但要求 feats 可计算
                    inputs_ok = True
                do_swap = (not fallback) and inputs_ok and float(pred[i]) > CHALLENGER_BUFFER
                if tag == "CH1":
                    swap_ch1[day] = bool(do_swap)
                else:
                    swap_ch2[day] = bool(do_swap)
                oof_rows.append(
                    {
                        "signal_date": str(day.date()),
                        "year": year,
                        "model": tag,
                        "pred_delta": float(pred[i]),
                        "actual_delta": float(row["y"]) if bool(mature.loc[idx]) else None,
                        "swap": bool(do_swap),
                        "fallback_v5": fallback,
                        "train_pair_days": int(train["signal_date"].nunique()),
                    }
                )
    write_df(ROOT / "challenger_oof.csv", pd.DataFrame(oof_rows))

    def _apply_swaps(base: pd.DataFrame, swap_map: dict) -> pd.DataFrame:
        out = base.copy()
        out["signal_date"] = pd.to_datetime(out["signal_date"])
        for day, do in swap_map.items():
            if not do:
                continue
            mask = out["signal_date"].eq(pd.Timestamp(day))
            block = out.loc[mask].sort_values(["score", "instrument_id"], ascending=[False, True], kind="mergesort")
            if len(block) < 3:
                continue
            ids = list(block.index)
            i2, i3 = ids[1], ids[2]
            s2, s3 = float(out.loc[i2, "score"]), float(out.loc[i3, "score"])
            if s2 == s3:
                continue
            out.loc[i2, "score"], out.loc[i3, "score"] = s3, s2
        return out

    base_scores = _fuse(five.rename(columns={"five": "score"}), ten.rename(columns={"ten": "score"}))
    # evaluate on intersection active dates for fair capital compare
    active_dates = sorted({d for d, v in swap_ch1.items() if True} | {d for d, v in swap_ch2.items() if True})
    # CH0/CH1/CH2 full replay with same capital via formal sleeve (full history; also report active-period separately)
    ch0 = run_formal_sleeve(
        ROOT / "challenger" / "CH0",
        base_scores,
        {"code": "R226_CH0", "round": "22.6", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        top_changed=False,
        fee_same_direction=True,
        not_only_year=True,
        counts_as_alpha=False,
    )
    ch1_scores = _apply_swaps(base_scores, swap_ch1)
    ch1 = run_formal_sleeve(
        ROOT / "challenger" / "CH1",
        ch1_scores,
        {"code": "R226_CH1", "round": "22.6", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    ch2_scores = _apply_swaps(base_scores, swap_ch2)
    ch2 = run_formal_sleeve(
        ROOT / "challenger" / "CH2",
        ch2_scores,
        {"code": "R226_CH2", "round": "22.6", "scope": "FULL_HISTORY_FIRST_TO_SECOND"},
        resume=resume,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    # attribution
    attr = []
    for name, summary in (("CH0", ch0), ("CH1", ch1), ("CH2", ch2)):
        attr.append(
            {
                "code": name,
                "sharpe": summary.get("sharpe"),
                "cagr": summary.get("cagr"),
                "nav_end": summary.get("nav_end"),
                "nav_improvement_vs_ch0": None if summary.get("nav_end") is None else float(summary["nav_end"]) / float(ch0["nav_end"]) - 1.0,
                "economically_material": summary.get("economically_material"),
                "n_swap_days": int(sum(1 for v in (swap_ch1 if name == "CH1" else swap_ch2 if name == "CH2" else {}).values() if v)),
            }
        )
    write_df(ROOT / "challenger_trade_attribution.csv", pd.DataFrame(attr))
    ch1_better = (ch1.get("sharpe") or -np.inf) > (ch0.get("sharpe") or -np.inf) and (ch1.get("nav_improvement") or -1) > 0
    ch2_vs_ch0 = (ch2.get("sharpe") or -np.inf) > (ch0.get("sharpe") or -np.inf) and (ch2.get("nav_end") or 0) > (ch0.get("nav_end") or 0)
    ch2_vs_ch1 = (ch2.get("sharpe") or -np.inf) > (ch1.get("sharpe") or -np.inf) and (ch2.get("nav_end") or 0) > (ch1.get("nav_end") or 0)
    if ch2_vs_ch0 and ch2_vs_ch1:
        verdict = "LOCAL_FS2_INCREMENT_CANDIDATE"
    elif ch1_better and not ch2_vs_ch1:
        verdict = "LOCAL_REORDER_EFFECT_ONLY"
    else:
        verdict = "CHALLENGER_NO_INCREMENT"
    payload = {
        "status": "DONE",
        "verdict": verdict,
        "ch0": {k: ch0.get(k) for k in ("sharpe", "cagr", "nav_end", "completed_trades")},
        "ch1": {k: ch1.get(k) for k in ("sharpe", "cagr", "nav_end", "completed_trades", "nav_improvement")},
        "ch2": {k: ch2.get(k) for k in ("sharpe", "cagr", "nav_end", "completed_trades", "nav_improvement")},
        "ch2_vs_ch0": ch2_vs_ch0,
        "ch2_vs_ch1": ch2_vs_ch1,
        "buffer": CHALLENGER_BUFFER,
        "min_train_pair_days": CHALLENGER_MIN_TRAIN_PAIR_DAYS,
        "active_dates": len(active_dates),
    }
    dump_json(ROOT / "challenger.json", payload)
    return save_stage(ROOT, "challenger", payload)


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("stats", resume)
    if cached:
        return cached
    series = {}
    for code, path in (
        ("F0", ROOT / "heads" / "F0_v5" / "nav.parquet"),
        ("F1", ROOT / "heads" / "F1_five_fs2" / "nav.parquet"),
        ("F2", ROOT / "heads" / "F2_ten_fs2" / "nav.parquet"),
        ("CH0", ROOT / "challenger" / "CH0" / "nav.parquet"),
        ("CH1", ROOT / "challenger" / "CH1" / "nav.parquet"),
        ("CH2", ROOT / "challenger" / "CH2" / "nav.parquet"),
    ):
        if path.is_file():
            series[code] = daily_from_nav(pd.read_parquet(path))
    boots = {}
    for name in ("F1", "F2", "CH1", "CH2"):
        if name in series and "F0" in series:
            base = "CH0" if name.startswith("CH") and "CH0" in series else "F0"
            boots[name] = paired_block_bootstrap_v2(series[name], series[base])
    panel = pd.DataFrame({k: v for k, v in series.items() if k in ("F0", "F1", "F2")}).dropna(how="any")
    spa = spa_hansen_or_blocked(panel, baseline="F0") if not panel.empty else {"status": "BLOCKED_STAT_TEST", "reason": "empty"}
    payload = {"bootstrap": boots, "spa_full_history_first_to_second": spa}
    dump_json(ROOT / "statistics.json", payload)
    return save_stage(ROOT, "stats", payload)


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    heads = load_json(ROOT / "fs2_heads.json")
    chal = load_json(ROOT / "challenger.json")
    tests = run_unit_tests(ROOT / "_unit_tests_tmp")
    dump_json(ROOT / "unit_tests.json", tests)
    fs2_verdict = heads.get("verdict") or "NOT_EVALUATED"
    chal_verdict = chal.get("verdict") or "NOT_EVALUATED"
    decision = "KEEP_BASELINE_V5"
    if fs2_verdict == "FS2_DIRECT_ADD_CANDIDATE" or chal_verdict == "LOCAL_FS2_INCREMENT_CANDIDATE":
        decision = "V6_RESEARCH_CANDIDATE_NOT_PROMOTED"
    payload = {
        "decision": decision,
        "fs2_verdict": fs2_verdict,
        "challenger_verdict": chal_verdict,
        "current_baseline_changed": False,
        "auto_promote": False,
        "production_config_changed": False,
        "tests": {"passed": tests["passed"], "failed": tests["failed"], "total": tests["total"]},
        "f1_delta_sharpe": ((heads.get("results_by_head") or {}).get("five") or {}).get("delta_vs_f0", {}).get("delta_sharpe"),
        "f2_delta_sharpe": ((heads.get("results_by_head") or {}).get("ten") or {}).get("delta_vs_f0", {}).get("delta_sharpe"),
        "challenger_vs_baseline": chal.get("ch2_vs_ch0"),
        "challenger_vs_score_only": chal.get("ch2_vs_ch1"),
    }
    dump_json(ROOT / "decision.json", payload)
    write_text(
        ROOT / "FINAL_DECISION.md",
        "\n".join(
            [
                "# Round22.6 最终裁决",
                "",
                f"- FS2 直接加头：`{fs2_verdict}`",
                f"- Challenger：`{chal_verdict}`",
                f"- 晋级建议：`{decision}`",
                "- CURRENT_BASELINE_CHANGED=false",
                "- AUTO_PROMOTE=false",
                "- PRODUCTION_CONFIG_CHANGED=false",
                "- 经济门使用 CORRECTED；缺证据不得默认通过。",
                "",
            ]
        ),
    )
    # all trials
    trials = []
    for csv_name in ("fs2_head_backtests.csv", "c4_feature_placebo.csv", "challenger_trade_attribution.csv"):
        path = ROOT / csv_name
        if path.is_file():
            part = pd.read_csv(path)
            part["source_table"] = csv_name
            trials.append(part)
    if trials:
        write_df(ROOT / "all_trials.csv", pd.concat(trials, ignore_index=True, sort=False))
    return save_stage(ROOT, "decision", payload)


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dec = load_json(ROOT / "decision.json")
    heads = load_json(ROOT / "fs2_heads.json")
    partial = load_json(ROOT / "fs2_partial_information.json")
    chal = load_json(ROOT / "challenger.json")
    fee = fee_components()
    lines = [
        "# Round22.6 研究报告",
        "",
        "## 1. 相对旧代码修了什么",
        "- 经济门改为 CORRECTED：有符号 nav_improvement；缺费用/Top 证据为 NOT_EVALUATED，不得默认 True。",
        "- 旧 daily_spearman 误称 conditional_ic → 现名 raw_rank_ic；真条件诊断为 LINEAR_PARTIAL_INFORMATION_DIAGNOSTIC（折内 Ridge α=10 残差）。",
        "- FS2 的 Five/Ten 完整袖套都跑完，无因 F1 提前 break。",
        "- C4 feature-level placebo 只打乱 C4；旧分数/仓位打乱改名为 PORTFOLIO_SCORE_ALLOCATION_RANDOMIZATION。",
        "- 缓存与正式摘要绑定完整实验身份；账本容差 max(0.01, 1e-8·NAV)。",
        "",
        "## 2. 哪些旧结论仍成立 / 撤回",
        "- C1 继续归档，不扩展。",
        "- C4 公式未改。",
        "- ROUND7_FLAT_5BP 名称保留，但真实费用分项已记录，不可理解为总成本仅 5bp。",
        f"- 费用分项：{fee}",
        "",
        "## 3. 训练与回测计数",
        f"- FS2 heads verdict=`{heads.get('verdict')}`",
        f"- Challenger verdict=`{chal.get('verdict')}`",
        f"- 单测 passed={dec.get('tests',{}).get('passed')} failed={dec.get('tests',{}).get('failed')}",
        "",
        "## 4. 实验数值",
        f"- F1/F2 明细见 `fs2_head_backtests.csv`；partial IC 见 `raw_vs_partial_ic.csv`。",
        f"- F1 ΔSharpe=`{dec.get('f1_delta_sharpe')}` F2 ΔSharpe=`{dec.get('f2_delta_sharpe')}`",
        f"- Challenger vs CH0=`{dec.get('challenger_vs_baseline')}` vs CH1=`{dec.get('challenger_vs_score_only')}`",
        "",
        "## 5. 基线与信息契约",
        f"- 锁定基线 R16_O01；信息车道 `{LEGACY_OPEN_INFORMATION_LANE}`（含 d1_open_gap）。",
        f"- PIT=`{(partial.get('pit') or {}).get('pit_pass')}`",
        "",
        "## 6. 是否可替换 baseline",
        f"- 决策：`{dec.get('decision')}`；CURRENT 未改。",
        "",
        "## 7. 下一步（仅基于本轮）",
        "- 不把 Challenger 最佳结果自动带入 R23.2。",
        "- R23.2 从冻结 v5 端点一致性开始。",
        "",
        "## 8. 文件",
        f"- 根目录 `{ROOT}`",
        f"- frozen_manifest sha 见文件；decision=`{sha256_json(dec)}`",
        "",
    ]
    write_text(ROOT / "ROUND22_6_RESEARCH_REPORT.md", "\n".join(lines))
    return save_stage(ROOT, "report", {"status": "DONE", "path": str(ROOT / "ROUND22_6_RESEARCH_REPORT.md")})


STAGES = {
    "freeze": run_freeze,
    "partial_ic": run_partial_ic,
    "fs2_heads": run_fs2_heads,
    "c4_placebo": run_c4_placebo,
    "challenger": run_challenger,
    "stats": run_stats,
    "decision": run_decision,
    "report": run_report,
}


def run_all(*, resume: bool = True, jobs: int = 2, **kwargs: Any) -> dict[str, Any]:
    out = {}
    for name, fn in STAGES.items():
        print(f"[round22_6] stage={name}", flush=True)
        out[name] = fn(resume=resume, jobs=jobs, **kwargs)
    return out
