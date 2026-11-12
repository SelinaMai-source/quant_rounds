"""六层 parity + join/label/fold/feature-order/missing 审计。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from ...round11.parity import compare_ledgers
from ...round17.train import complete_cases
from ..common import finite_or_none
from ..contract import PARITY_ABS_NAV, PARITY_METRIC_ABS, SIX_LAYER_PARITY, V4_ANCHOR
from .retrain import run_fresh_retrain
from .support import (
    B00,
    B01,
    JOIN_AUDIT_CSV,
    KEYS,
    PARITY_JSON,
    R18_A01_FIVE_REPORT,
    R18_A01_FIVE_SEED17,
    R7_FIVE_REPORT,
    R7_FIVE_SEED17,
    R7_TEN_REPORT,
    R7_TEN_SEED17,
    add_rank,
    cand_dir,
    dump_json,
    load_b01_preds,
    load_json,
    load_original_panels,
    load_overlay,
    load_r18_panels,
    load_r7_preds,
    pred_keys,
    status_of,
    tb_dir,
    write_md,
    year_of,
)


def _ensure_retrain(**kw: Any) -> dict[str, Any]:
    path = tb_dir("retrain") / "fresh_retrain.json"
    if path.is_file():
        return load_json(path)
    return run_fresh_retrain(**kw)


def _key_diff(base: pd.DataFrame, other: pd.DataFrame, keys: list[str]) -> dict[str, Any]:
    b = base[keys].drop_duplicates()
    o = other[keys].drop_duplicates()
    b_i = b.set_index(keys)
    o_i = o.set_index(keys)
    missing = int((~b_i.index.isin(o_i.index)).sum())
    extra = int((~o_i.index.isin(b_i.index)).sum())
    return {
        "baseline_rows": int(len(base)),
        "other_rows": int(len(other)),
        "baseline_keys": int(len(b)),
        "other_keys": int(len(o)),
        "n_missing": missing,
        "n_extra": extra,
        "n_dup_baseline": int(len(base) - len(b)),
        "n_dup_other": int(len(other) - len(o)),
    }


def _col_mismatch(base: pd.DataFrame, other: pd.DataFrame, keys: list[str], cols: list[str]) -> dict[str, int]:
    use = [c for c in cols if c in base.columns and c in other.columns and c not in keys]
    if not use:
        return {c: 0 if c in keys else -1 for c in cols}
    left = base[keys + use].copy()
    right = other[keys + [c for c in use]].copy()
    for k in keys:
        if "date" in k:
            left[k] = pd.to_datetime(left[k])
            right[k] = pd.to_datetime(right[k])
        elif k == "instrument_id":
            left[k] = left[k].astype(str)
            right[k] = right[k].astype(str)
    merged = left.merge(right, on=keys, how="inner", suffixes=("_b", "_o"))
    out: dict[str, int] = {}
    for c in use:
        a = merged[f"{c}_b"]
        b = merged[f"{c}_o"]
        if pd.api.types.is_datetime64_any_dtype(a) or pd.api.types.is_datetime64_any_dtype(b):
            miss = int((pd.to_datetime(a).fillna(pd.Timestamp(0)) != pd.to_datetime(b).fillna(pd.Timestamp(0))).sum())
        else:
            an = pd.to_numeric(a, errors="coerce")
            bn = pd.to_numeric(b, errors="coerce")
            if an.notna().any() or bn.notna().any():
                both = an.notna() & bn.notna()
                close = np.zeros(len(merged), dtype=bool)
                if both.any():
                    close[both.to_numpy()] = np.isclose(an[both], bn[both], rtol=0, atol=1e-12, equal_nan=False)
                na_both = an.isna() & bn.isna()
                miss = int((~(close | na_both.to_numpy())).sum())
            else:
                miss = int((a.astype(str).fillna("__NA__") != b.astype(str).fillna("__NA__")).sum())
        out[c] = miss
    for c in cols:
        out.setdefault(c, -1)
    return out


def build_join_audit() -> pd.DataFrame:
    r7 = load_r7_preds()
    b01 = load_b01_preds()
    orig = load_original_panels()
    r18 = load_r18_panels()
    rows: list[dict[str, Any]] = []

    for head in ("five", "ten"):
        base = pred_keys(r7[head])
        alt = pred_keys(b01[head])
        base["year"] = year_of(base["signal_date"])
        alt["year"] = year_of(alt["signal_date"])
        years = sorted(set(base["year"]).union(set(alt["year"])))
        for year in years:
            b = base.loc[base["year"].eq(year)]
            a = alt.loc[alt["year"].eq(year)]
            diff = _key_diff(b, a, ["signal_date", "instrument_id"])
            rows.append({
                "layer": "prediction",
                "head": head,
                "fold": int(year),
                "date": "",
                "baseline_rows": diff["baseline_rows"],
                "retrain_rows": diff["other_rows"],
                "n_missing": diff["n_missing"],
                "n_extra": diff["n_extra"],
                "n_dup_baseline": diff["n_dup_baseline"],
                "n_dup_retrain": diff["n_dup_other"],
                "orig_feature_mismatch": "",
                "label_mismatch": "",
                "group_mismatch": "",
                "note": "R7 locked vs R19_B01 force retrain",
            })
        # date-level prediction key counts
        b_cnt = base.groupby("signal_date").size()
        a_cnt = alt.groupby("signal_date").size()
        idx = b_cnt.index.union(a_cnt.index)
        for dt in idx:
            br = int(b_cnt.get(dt, 0))
            ar = int(a_cnt.get(dt, 0))
            if br == ar:
                continue
            rows.append({
                "layer": "prediction_date",
                "head": head,
                "fold": int(pd.Timestamp(dt).year),
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "baseline_rows": br,
                "retrain_rows": ar,
                "n_missing": max(br - ar, 0),
                "n_extra": max(ar - br, 0),
                "n_dup_baseline": 0,
                "n_dup_retrain": 0,
                "orig_feature_mismatch": "",
                "label_mismatch": "",
                "group_mismatch": "",
                "note": "date-level row-count mismatch only",
            })

        features = list(FIVE_FEATURES if head == "five" else TEN_FEATURES)
        target = FIVE_HEAD.target if head == "five" else TEN_HEAD.target
        label_end = FIVE_HEAD.label_end_column if head == "five" else TEN_HEAD.label_end_column
        for kind in ("training", "candidate"):
            o = orig[f"{head}_{kind}"]
            m = r18[f"{head}_{kind}"]
            o["year"] = year_of(o["trade_date"])
            m["year"] = year_of(m["trade_date"])
            years = sorted(set(o["year"]).union(set(m["year"])))
            for year in years:
                ob = o.loc[o["year"].eq(year)]
                mb = m.loc[m["year"].eq(year)]
                diff = _key_diff(ob, mb, KEYS)
                feat_m = _col_mismatch(ob, mb, KEYS, features)
                labs = [c for c in (target, label_end) if c in ob.columns and c in mb.columns]
                lab_m = _col_mismatch(ob, mb, KEYS, labs) if labs else {}
                rows.append({
                    "layer": f"panel_{kind}",
                    "head": head,
                    "fold": int(year),
                    "date": "",
                    "baseline_rows": diff["baseline_rows"],
                    "retrain_rows": diff["other_rows"],
                    "n_missing": diff["n_missing"],
                    "n_extra": diff["n_extra"],
                    "n_dup_baseline": diff["n_dup_baseline"],
                    "n_dup_retrain": diff["n_dup_other"],
                    "orig_feature_mismatch": int(sum(v for v in feat_m.values() if v > 0)),
                    "label_mismatch": int(sum(v for v in lab_m.values() if v > 0)),
                    "group_mismatch": int(diff["n_missing"] + diff["n_extra"] + diff["n_dup_baseline"] + diff["n_dup_other"]),
                    "note": "original R7 panel vs R18 merged panel; group=trade_date qid",
                })
    return pd.DataFrame(rows)


def run_join_audit(**kw: Any) -> dict[str, Any]:
    _ensure_retrain(**kw)
    frame = build_join_audit()
    dest = JOIN_AUDIT_CSV
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    pred = frame.loc[frame["layer"].eq("prediction")]
    summary = {
        "status": "done",
        "path": str(dest),
        "n_rows": int(len(frame)),
        "prediction_missing_total": int(pd.to_numeric(pred["n_missing"], errors="coerce").fillna(0).sum()),
        "prediction_extra_total": int(pd.to_numeric(pred["n_extra"], errors="coerce").fillna(0).sum()),
        "panel_feature_mismatch_total": int(
            pd.to_numeric(frame.loc[frame["layer"].str.startswith("panel"), "orig_feature_mismatch"], errors="coerce")
            .fillna(0).sum()
        ),
        "panel_label_mismatch_total": int(
            pd.to_numeric(frame.loc[frame["layer"].str.startswith("panel"), "label_mismatch"], errors="coerce")
            .fillna(0).sum()
        ),
    }
    dump_json(tb_dir("audit") / "join_audit.json", summary)
    write_md(
        tb_dir("audit") / "03_join_audit.md",
        "# Track B · join_audit\n\n"
        f"path=`{dest}` rows=`{len(frame)}`\n\n"
        f"- prediction missing keys=`{summary['prediction_missing_total']}` extra=`{summary['prediction_extra_total']}`\n"
        f"- panel orig-feature mismatch=`{summary['panel_feature_mismatch_total']}` "
        f"label mismatch=`{summary['panel_label_mismatch_total']}`\n",
    )
    return summary


def run_labels(**kw: Any) -> dict[str, Any]:
    _ensure_retrain(**kw)
    orig = load_original_panels()
    r18 = load_r18_panels()
    rows = []
    for head, target, label_end in (
        ("five", FIVE_HEAD.target, FIVE_HEAD.label_end_column),
        ("ten", TEN_HEAD.target, TEN_HEAD.label_end_column),
    ):
        for kind in ("training", "candidate"):
            o = orig[f"{head}_{kind}"]
            m = r18[f"{head}_{kind}"]
            cols = [c for c in (target, label_end) if c in o.columns]
            mm = _col_mismatch(o, m, KEYS, cols)
            rows.append({
                "head": head,
                "panel": kind,
                "columns": cols,
                "mismatches": mm,
                "n_mismatch": int(sum(v for v in mm.values() if v > 0)),
                "baseline_rows": int(len(o)),
                "r18_rows": int(len(m)),
            })
    ok = all(r["n_mismatch"] == 0 for r in rows)
    payload = {"status": status_of(ok), "passed": ok, "rows": rows, "note": "label / next2_date vs original panel"}
    dump_json(tb_dir("audit") / "labels.json", payload)
    write_md(tb_dir("audit") / "04_labels.md", f"# labels\n\nstatus=`{payload['status']}` rows=`{rows}`\n")
    return payload


def run_folds(**kw: Any) -> dict[str, Any]:
    _ensure_retrain(**kw)
    r7_five = load_json(R7_FIVE_SEED17)
    r7_ten = load_json(R7_TEN_SEED17)
    b01_five = load_json(cand_dir(B01) / "five" / "seed17" / "training_report.json")
    b01_ten = load_json(cand_dir(B01) / "ten" / "seed17" / "training_report.json")
    a01_five = load_json(R18_A01_FIVE_SEED17)

    def _folds(report: dict[str, Any]) -> list[dict[str, Any]]:
        return list(report.get("folds") or [])

    def _cmp(name: str, locked: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> dict[str, Any]:
        by_l = {int(f["predict_year"]): f for f in locked}
        by_f = {int(f["predict_year"]): f for f in fresh}
        years = sorted(set(by_l) | set(by_f))
        recs = []
        for y in years:
            a = by_l.get(y) or {}
            b = by_f.get(y) or {}
            recs.append({
                "predict_year": y,
                "train_years_locked": a.get("train_years"),
                "train_years_fresh": b.get("train_years"),
                "train_rows_locked": a.get("train_rows") or a.get("train_rows_available"),
                "train_rows_fresh": b.get("train_rows"),
                "candidate_rows_locked": a.get("candidate_rows_scored"),
                "candidate_rows_fresh": b.get("candidate_rows_scored"),
                "eval_rows_locked": a.get("evaluation_rows"),
                "eval_rows_fresh": b.get("evaluation_rows"),
                "train_years_match": a.get("train_years") == b.get("train_years"),
            })
        years_ok = all(r["train_years_match"] for r in recs if r["train_years_locked"] and r["train_years_fresh"])
        return {"head": name, "years_match": years_ok, "folds": recs}

    five = _cmp("five", _folds(r7_five), _folds(b01_five))
    ten = _cmp("ten", _folds(r7_ten), _folds(b01_ten))
    a01_vs_r7 = _cmp("five_a01_vs_r7", _folds(r7_five), _folds(a01_five))
    ok = bool(five["years_match"] and ten["years_match"])
    payload = {
        "status": status_of(ok),
        "passed": ok,
        "five": five,
        "ten": ten,
        "r18_a01_vs_r7_five": a01_vs_r7,
        "note": "walk-forward = prior 3 calendar years → next year; B01 must keep the same windows.",
    }
    dump_json(tb_dir("audit") / "folds.json", payload)
    write_md(tb_dir("audit") / "05_folds.md", f"# folds\n\nstatus=`{payload['status']}` years_match five/ten=`{five['years_match']}`/`{ten['years_match']}`\n")
    return payload


def run_feature_order(**kw: Any) -> dict[str, Any]:
    _ensure_retrain(**kw)
    five = load_json(cand_dir(B01) / "five" / "ensemble_report.json")
    ten = load_json(cand_dir(B01) / "ten" / "ensemble_report.json")
    five_used = list(five.get("features") or [])
    ten_used = list(ten.get("features") or [])
    five_ok = five_used == list(FIVE_FEATURES)
    ten_ok = ten_used == list(TEN_FEATURES)
    extras_five = [c for c in five_used if c not in FIVE_FEATURES]
    extras_ten = [c for c in ten_used if c not in TEN_FEATURES]
    payload = {
        "status": status_of(five_ok and ten_ok and not extras_five and not extras_ten),
        "passed": bool(five_ok and ten_ok and not extras_five and not extras_ten),
        "five_contract": list(FIVE_FEATURES),
        "five_used": five_used,
        "five_order_match": five_ok,
        "five_extras": extras_five,
        "ten_contract": list(TEN_FEATURES),
        "ten_used": ten_used,
        "ten_order_match": ten_ok,
        "ten_extras": extras_ten,
        "note": "B01 must be original 14/13 order, no F1/F2/F3.",
    }
    dump_json(tb_dir("audit") / "feature_order.json", payload)
    write_md(
        tb_dir("audit") / "06_feature_order.md",
        f"# feature order\n\nstatus=`{payload['status']}` five_ok=`{five_ok}` ten_ok=`{ten_ok}` extras=`{extras_five+extras_ten}`\n",
    )
    return payload


def run_missing(**kw: Any) -> dict[str, Any]:
    _ensure_retrain(**kw)
    orig = load_original_panels()
    r18 = load_r18_panels()
    r7_five = load_json(R7_FIVE_SEED17)
    b01_five = load_json(cand_dir(B01) / "five" / "seed17" / "training_report.json")
    a01_five = load_json(R18_A01_FIVE_SEED17)
    rows = []
    for head, feats, target in (
        ("five", list(FIVE_FEATURES), FIVE_HEAD.target),
        ("ten", list(TEN_FEATURES), TEN_HEAD.target),
    ):
        for src_name, src in (("original", orig), ("r18_merged", r18)):
            for kind, req in (("training", True), ("candidate", False)):
                audit, = (complete_cases(src[f"{head}_{kind}"], feats, require_target=req, target=target)[1],)
                rows.append({
                    "source": src_name,
                    "head": head,
                    "panel": kind,
                    **audit,
                })
    r7_drop = (r7_five.get("candidate_clean_audit") or {}).get("rows_dropped_in_memory")
    b01_drop = (b01_five.get("candidate_clean_audit") or {}).get("rows_dropped_in_memory")
    a01_drop = (a01_five.get("candidate_clean_audit") or {}).get("rows_dropped_in_memory")
    filled = any(r.get("filled_with_zero") for r in rows) or bool(
        (b01_five.get("filled_with_zero") if isinstance(b01_five, dict) else False)
    )
    orig_vs_r18 = []
    for head, feats, target in (("five", list(FIVE_FEATURES), FIVE_HEAD.target), ("ten", list(TEN_FEATURES), TEN_HEAD.target)):
        o_c, o_a = complete_cases(orig[f"{head}_candidate"], feats, require_target=False, target=target)
        m_c, m_a = complete_cases(r18[f"{head}_candidate"], feats, require_target=False, target=target)
        orig_vs_r18.append({
            "head": head,
            "original_after": o_a.get("rows_after"),
            "r18_after": m_a.get("rows_after"),
            "delta": (m_a.get("rows_after") or 0) - (o_a.get("rows_after") or 0),
        })
    same_on_original_feats = all(r["delta"] == 0 for r in orig_vs_r18)
    payload = {
        "status": "FAIL" if filled else ("PASS" if same_on_original_feats else "FAIL"),
        "passed": (not filled) and same_on_original_feats,
        "filled_with_zero": filled,
        "complete_cases": rows,
        "r7_five_candidate_dropped": r7_drop,
        "b01_five_candidate_dropped": b01_drop,
        "r18_a01_five_candidate_dropped": a01_drop,
        "a01_extra_drop_vs_r7": None if r7_drop is None or a01_drop is None else int(a01_drop) - int(r7_drop),
        "original_vs_r18_on_14_13": orig_vs_r18,
        "note": (
            "A01 extra drop vs R7 is complete_cases on F1, not fillna(0). "
            "B01 uses original 14/13 so that extra drop must not apply."
        ),
    }
    dump_json(tb_dir("audit") / "missing.json", payload)
    write_md(
        tb_dir("audit") / "07_missing.md",
        "# missing\n\n"
        f"status=`{payload['status']}` filled_with_zero=`{filled}`\n\n"
        f"- R7 five candidate drop=`{r7_drop}` B01=`{b01_drop}` A01=`{a01_drop}` A01-R7=`{payload['a01_extra_drop_vs_r7']}`\n"
        f"- original vs R18 merged complete_cases on 14/13: `{orig_vs_r18}`\n",
    )
    return payload


def _score_compare(base: pd.DataFrame, other: pd.DataFrame) -> dict[str, Any]:
    a = pred_keys(base)[["signal_date", "instrument_id", "score"]]
    b = pred_keys(other)[["signal_date", "instrument_id", "score"]]
    keys = ["signal_date", "instrument_id"]
    inner = a.merge(b, on=keys, how="inner", suffixes=("_b00", "_b01"))
    diff = _key_diff(a, b, keys)
    if inner.empty:
        return {**diff, "score_corr": None, "score_mae": None, "top1_agree": None, "top2_overlap": None}
    corr = finite_or_none(inner["score_b00"].corr(inner["score_b01"], method="spearman"))
    mae = finite_or_none((inner["score_b00"] - inner["score_b01"]).abs().mean())
    t1a = add_rank(a)
    t1b = add_rank(b)
    j1 = t1a.loc[t1a["_rank"].eq(1), ["signal_date", "instrument_id"]].merge(
        t1b.loc[t1b["_rank"].eq(1), ["signal_date", "instrument_id"]],
        on="signal_date",
        how="inner",
        suffixes=("_b00", "_b01"),
    )
    top1 = float((j1["instrument_id_b00"] == j1["instrument_id_b01"]).mean()) if len(j1) else None
    t2a = t1a.loc[t1a["_rank"] <= 2, keys]
    t2b = t1b.loc[t1b["_rank"] <= 2, keys]
    both = t2a.merge(t2b, on=keys, how="inner")
    top2 = float(len(both) / max(len(t2a), 1))
    return {
        **diff,
        "n_inner": int(len(inner)),
        "score_corr": corr,
        "score_mae": mae,
        "top1_agree": top1,
        "top2_overlap": top2,
        "identical_scores": bool(mae is not None and mae <= 1e-15),
    }


def run_six_layer(**kw: Any) -> dict[str, Any]:
    _ensure_retrain(**kw)
    join = run_join_audit(**kw) if not JOIN_AUDIT_CSV.is_file() else load_json(tb_dir("audit") / "join_audit.json")
    labels = load_json(tb_dir("audit") / "labels.json") or run_labels(**kw)
    folds = load_json(tb_dir("audit") / "folds.json") or run_folds(**kw)
    feats = load_json(tb_dir("audit") / "feature_order.json") or run_feature_order(**kw)
    miss = load_json(tb_dir("audit") / "missing.json") or run_missing(**kw)
    r7 = load_r7_preds()
    b01 = load_b01_preds()
    five_cmp = _score_compare(r7["five"], b01["five"])
    ten_cmp = _score_compare(r7["ten"], b01["ten"])
    r7_five_n = int(load_json(R7_FIVE_REPORT).get("rows") or len(r7["five"]))
    r7_ten_n = int(load_json(R7_TEN_REPORT).get("rows") or len(r7["ten"]))
    a01_five_n = (load_json(R18_A01_FIVE_REPORT) or {}).get("rows")

    b00 = load_json(cand_dir(B00) / "summary.json")
    b01s = load_json(cand_dir(B01) / "summary.json")
    obs0 = b00.get("observed") or {}
    obs1 = b01s.get("observed") or {}
    ledger = {"passed": False, "status": "MISSING"}
    t0 = cand_dir(B00) / "trades.parquet"
    t1 = cand_dir(B01) / "trades.parquet"
    n0 = cand_dir(B00) / "nav.parquet"
    n1 = cand_dir(B01) / "nav.parquet"
    if t0.is_file() and t1.is_file() and n0.is_file() and n1.is_file():
        ledger = compare_ledgers(
            pd.read_parquet(t1),
            pd.read_parquet(t0),
            pd.read_parquet(n1),
            pd.read_parquet(n0),
            target_n_trades=int(V4_ANCHOR["completed_trades"]),
        )

    ov0 = load_overlay(B00)
    ov1 = load_overlay(B01)
    top1 = _score_compare(ov0, ov1)
    alloc0 = add_rank(ov0)
    alloc1 = add_rank(ov1)
    jalloc = alloc0.merge(alloc1, on=["signal_date", "instrument_id"], how="inner", suffixes=("_b00", "_b01"))
    alloc_agree = None
    if "allocation_multiplier_b00" in jalloc.columns and len(jalloc):
        a = pd.to_numeric(jalloc["allocation_multiplier_b00"], errors="coerce")
        b = pd.to_numeric(jalloc["allocation_multiplier_b01"], errors="coerce")
        alloc_agree = float(np.isclose(a, b, rtol=0, atol=1e-12).mean())

    metric_close = False
    try:
        metric_close = (
            abs(float(obs1.get("cagr") or 0) - float(obs0.get("cagr") or 0)) < PARITY_METRIC_ABS
            and abs(float(obs1.get("sharpe") or 0) - float(obs0.get("sharpe") or 0)) < PARITY_METRIC_ABS
            and abs(float(obs1.get("max_drawdown") or 0) - float(obs0.get("max_drawdown") or 0)) < PARITY_METRIC_ABS
            and int(obs1.get("completed_trades") or -1) == int(obs0.get("completed_trades") or -2)
        )
    except (TypeError, ValueError):
        metric_close = False

    xgb_env = bool(
        five_cmp.get("identical_scores") is False
        or ten_cmp.get("identical_scores") is False
    )
    # Level 1
    l1_ok = (
        join.get("prediction_missing_total", 1) == 0
        and join.get("prediction_extra_total", 1) == 0
        and int(five_cmp.get("n_dup_baseline") or 0) == 0
        and int(five_cmp.get("n_dup_other") or 0) == 0
    )
    # Level 2
    l2_ok = bool(labels.get("passed"))
    # Level 3: 45256 vs B01; record A01 45255 as the known extra-drop
    l3_ok = int(b01s.get("five_rows") or 0) == r7_five_n and int(b01s.get("ten_rows") or 0) == r7_ten_n
    # Level 4
    l4_ok = bool(feats.get("passed")) and bool(miss.get("passed")) and not miss.get("filled_with_zero")
    # Level 5: OOF identity is NOT required; record XGB env. Identifiable always.
    l5_ident = five_cmp.get("score_corr") is not None
    l5_pass = bool(five_cmp.get("identical_scores") and ten_cmp.get("identical_scores"))
    l5_status = "PASS" if l5_pass else ("XGB_ENV_DIFF" if l5_ident else "NOT_IDENTIFIABLE")
    # Level 6
    l6_pass = bool(metric_close and ledger.get("passed"))
    l6_status = "PASS" if l6_pass else ("XGB_ENV_DIFF" if xgb_env and l1_ok and l4_ok else "FAIL")

    layers = [
        {
            "level": 1,
            "name": "join_keys",
            "status": status_of(l1_ok),
            "passed": l1_ok,
            "detail": {
                "join": join,
                "five": {k: five_cmp.get(k) for k in ("baseline_rows", "other_rows", "n_missing", "n_extra")},
                "ten": {k: ten_cmp.get(k) for k in ("baseline_rows", "other_rows", "n_missing", "n_extra")},
            },
        },
        {
            "level": 2,
            "name": "labels_groups",
            "status": labels.get("status") or status_of(l2_ok),
            "passed": l2_ok,
            "detail": labels,
        },
        {
            "level": 3,
            "name": "prediction_rows",
            "status": status_of(l3_ok),
            "passed": l3_ok,
            "detail": {
                "r7_five": r7_five_n,
                "r7_ten": r7_ten_n,
                "b01_five": b01s.get("five_rows"),
                "b01_ten": b01s.get("ten_rows"),
                "r18_a01_five": a01_five_n,
                "note": "R7 five 45256 vs A01 45255 is the F1 complete_cases extra drop, not B01.",
            },
        },
        {
            "level": 4,
            "name": "feature_order_and_missing",
            "status": status_of(l4_ok),
            "passed": l4_ok,
            "detail": {"feature_order": feats, "missing": {
                "status": miss.get("status"),
                "filled_with_zero": miss.get("filled_with_zero"),
                "a01_extra_drop_vs_r7": miss.get("a01_extra_drop_vs_r7"),
                "original_vs_r18_on_14_13": miss.get("original_vs_r18_on_14_13"),
            }},
        },
        {
            "level": 5,
            "name": "oof_scores",
            "status": l5_status,
            "passed": l5_pass,
            "xgb_env_diff": (not l5_pass) and l5_ident,
            "detail": {"five": five_cmp, "ten": ten_cmp},
            "note": "XGBoost/环境差异不得当成可忽略 PASS。",
        },
        {
            "level": 6,
            "name": "canonical_ledger",
            "status": l6_status,
            "passed": l6_pass,
            "xgb_env_diff": l6_status == "XGB_ENV_DIFF",
            "detail": {
                "b00": obs0,
                "b01": obs1,
                "metric_close": metric_close,
                "abs_tol": PARITY_METRIC_ABS,
                "nav_tol": PARITY_ABS_NAV,
                "ledger": {k: ledger.get(k) for k in ("passed", "status", "n_local", "n_anchor") if k in ledger or True},
                "overlay_top1_agree": top1.get("top1_agree"),
                "overlay_top2_overlap": top1.get("top2_overlap"),
                "allocation_multiplier_agree": alloc_agree,
            },
            "note": "禁止只比 CAGR。",
        },
    ]
    # keep ledger dump smaller
    layers[5]["detail"]["ledger"] = {
        "passed": ledger.get("passed"),
        "status": ledger.get("status"),
        "n_local": ledger.get("n_local"),
        "n_anchor": ledger.get("n_anchor"),
        "n_matched": ledger.get("n_matched"),
    }
    payload = {
        "status": "done",
        "layers": layers,
        "six_layer_contract": [dict(x) for x in SIX_LAYER_PARITY],
        "xgb_env_diff_recorded": bool(xgb_env),
        "xgb_env_not_ignorable_pass": True,
        "all_pass": all(x.get("status") == "PASS" for x in layers),
    }
    dump_json(PARITY_JSON, payload)
    lines = [
        "# Track B · 六层 parity",
        "",
        "不得只用 CAGR。XGBoost/环境差异不得标可忽略 PASS。",
        "",
        "| level | name | status |",
        "|---|---|---|",
    ]
    for layer in layers:
        lines.append(f"| {layer['level']} | {layer['name']} | `{layer['status']}` |")
    lines += [
        "",
        f"- B00 CAGR=`{obs0.get('cagr')}` Sharpe=`{obs0.get('sharpe')}` trades=`{obs0.get('completed_trades')}`",
        f"- B01 CAGR=`{obs1.get('cagr')}` Sharpe=`{obs1.get('sharpe')}` trades=`{obs1.get('completed_trades')}`",
        f"- five Top1 agree=`{five_cmp.get('top1_agree')}` score_corr=`{five_cmp.get('score_corr')}`",
        f"- ten Top1 agree=`{ten_cmp.get('top1_agree')}` score_corr=`{ten_cmp.get('score_corr')}`",
        f"- overlay Top1 agree=`{top1.get('top1_agree')}` alloc agree=`{alloc_agree}`",
        f"- R7 five rows=`{r7_five_n}` B01=`{b01s.get('five_rows')}` A01=`{a01_five_n}`",
        "",
    ]
    write_md(tb_dir("parity") / "02_six_layer.md", "\n".join(lines))
    return payload


def run_parity(**kw: Any) -> dict[str, Any]:
    # ensure audit pieces exist so six-layer can embed them
    run_labels(**kw)
    run_folds(**kw)
    run_feature_order(**kw)
    run_missing(**kw)
    run_join_audit(**kw)
    return run_six_layer(**kw)


__all__ = [
    "run_feature_order",
    "run_folds",
    "run_join_audit",
    "run_labels",
    "run_missing",
    "run_parity",
    "run_six_layer",
]
