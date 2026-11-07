"""a-equity-control：C2 原 V03 重放；C3 预注册 activity-match（对照不得晋级）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...round15.runner import apply_top2_veto
from ...round16_5.features import equity_panel, ridge_fit, ridge_predict, standardize_train
from ...round16_5.placebo import _breadth_bucket
from ..common import dump_json, load_json, write_md
from ...round16_5.contract import MATCHING_CONSTRAINT_FAILURE
from ..contract import (
    ACTIVITY_MATCHING_PASS,
    ACTIVITY_NOT_FULLY_MATCHED,
    EX_POST_DIAGNOSTIC_ONLY,
    V03_FEATURES,
    V03_RIDGE_ALPHA,
)
from .engine import cached_overlay_v4, o01_signal_dates, run_overlayed
from .support import (
    A_C2,
    A_C3,
    EQ_JSON,
    EVENTS_JSON,
    control_dir,
    idx_dates,
    overlayed_path,
    pack_obs_from_dir,
    planned_budget_by_date,
    require_c0,
    resume_ok,
    skip_result,
    ta_md,
    top2,
    ymd_list,
)


def _jaccard(a, b) -> float | None:
    a = set(idx_dates(a))
    b = set(idx_dates(b))
    if not a and not b:
        return None
    return float(len(a & b) / len(a | b))


def _c0_top2_labels() -> pd.DataFrame:
    t = top2(pd.read_parquet(require_c0() / "trades.parquet"))
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["actual_exit_date"] = pd.to_datetime(t["actual_exit_date"]).dt.normalize()
    t["buy_cost"] = pd.to_numeric(t["buy_cost"], errors="coerce")
    t["net"] = pd.to_numeric(t.get("net"), errors="coerce")
    t["ret_net"] = t["net"] / t["buy_cost"].replace(0, np.nan)
    g = t.groupby("signal_date").agg(
        top2_net=("net", "sum"),
        top2_ret=("ret_net", "mean"),
        actual_exit_date=("actual_exit_date", "max"),
        n_top2=("instrument_id", "size"),
        budget=("target_name_budget", "sum"),
    )
    return g.reset_index()


def _cell_series(idx: pd.DatetimeIndex) -> pd.Series:
    buckets = _breadth_bucket(idx)
    years = pd.Series(idx.year.astype(str), index=idx)
    return years.astype(str) + "|" + buckets.astype(str)


def _original_v03_veto(eligible: pd.DatetimeIndex, o01_elig: pd.DatetimeIndex, m: pd.DataFrame) -> dict[str, Any]:
    """原 V03：训练窗 o01_elig 计数定分位，测试年 eligible 上阈值。保留 ACTIVITY_NOT_FULLY_MATCHED。"""
    feat_cols = list(V03_FEATURES)
    scores = pd.Series(index=pd.DatetimeIndex(m["signal_date"]), dtype=float)
    thresholds = {}
    years = pd.to_datetime(m["signal_date"]).dt.year
    for test_year in sorted(set(years)):
        train_mask = years < test_year
        test_mask = years == test_year
        mature = pd.to_datetime(m["actual_exit_date"]).dt.year < test_year
        use = train_mask & mature
        if int(use.sum()) < 40 or int(test_mask.sum()) < 5:
            continue
        Xtr_raw = m.loc[use, feat_cols].to_numpy(float)
        ytr = (-pd.to_numeric(m.loc[use, "top2_net"], errors="coerce")).to_numpy(float)
        ytr = np.where(np.isfinite(ytr), ytr, 0.0)
        Xte_raw = m.loc[test_mask, feat_cols].to_numpy(float)
        Xtr, Xte, _, _ = standardize_train(Xtr_raw, Xte_raw)
        beta = ridge_fit(Xtr, ytr, alpha=V03_RIDGE_ALPHA)
        pred_tr = ridge_predict(Xtr, beta)
        pred_te = ridge_predict(Xte, beta)
        train_dates = idx_dates(m.loc[use, "signal_date"])
        n_target = int(train_dates.isin(o01_elig).sum())
        if n_target <= 0:
            n_target = max(1, int(round(0.1 * len(pred_tr))))
        q = 1.0 - min(0.99, max(0.01, n_target / max(len(pred_tr), 1)))
        thr = float(np.quantile(pred_tr, q))
        thresholds[int(test_year)] = {"threshold": thr, "n_target_train": n_target, "n_train": int(use.sum()), "q": q}
        scores.loc[idx_dates(m.loc[test_mask, "signal_date"])] = pred_te
    veto = []
    by_year = {}
    for test_year, rec in thresholds.items():
        year_idx = scores.index[scores.index.year == test_year]
        year_idx = year_idx.intersection(eligible) if len(eligible) else year_idx
        sc = scores.reindex(year_idx).dropna()
        picked = sc.index[sc >= rec["threshold"]]
        n_o01 = int((o01_elig.year == test_year).sum())
        by_year[int(test_year)] = {
            "n_v03": int(len(picked)),
            "n_o01": n_o01,
            "n_eligible": int((eligible.year == test_year).sum()) if len(eligible) else int(len(year_idx)),
        }
        veto.extend(list(picked))
    veto_idx = idx_dates(veto)
    matched = True
    for rec in by_year.values():
        if rec["n_o01"] > 0 and abs(rec["n_v03"] - rec["n_o01"]) > max(2, 0.25 * rec["n_o01"]):
            matched = False
    return {
        "veto": veto_idx,
        "by_year": by_year,
        "thresholds": thresholds,
        "activity_matched": bool(matched),
        "scores": scores,
    }


def _score_ridge(m: pd.DataFrame) -> pd.Series:
    feat_cols = list(V03_FEATURES)
    scores = pd.Series(index=idx_dates(m["signal_date"]), dtype=float)
    years = pd.to_datetime(m["signal_date"]).dt.year
    fallback = pd.to_numeric(m.set_index(idx_dates(m["signal_date"]))["index_RV20"], errors="coerce")
    for test_year in sorted(set(years)):
        train_mask = years < test_year
        test_mask = years == test_year
        mature = pd.to_datetime(m["actual_exit_date"]).dt.year < test_year
        use = train_mask & mature
        test_idx = idx_dates(m.loc[test_mask, "signal_date"])
        if int(test_mask.sum()) < 1:
            continue
        if int(use.sum()) < 20:
            scores.loc[test_idx] = fallback.reindex(test_idx)
            continue
        Xtr_raw = m.loc[use, feat_cols].to_numpy(float)
        ytr = (-pd.to_numeric(m.loc[use, "top2_net"], errors="coerce")).to_numpy(float)
        ytr = np.where(np.isfinite(ytr), ytr, 0.0)
        Xte_raw = m.loc[test_mask, feat_cols].to_numpy(float)
        Xtr, Xte, _, _ = standardize_train(Xtr_raw, Xte_raw)
        beta = ridge_fit(Xtr, ytr, alpha=V03_RIDGE_ALPHA)
        pred_te = ridge_predict(Xte, beta)
        scores.loc[test_idx] = pred_te
    return scores


def _pick_cell(scores: pd.Series, pool: pd.DatetimeIndex, nneed: int, budgets: pd.Series, target_budget: float | None) -> list:
    if nneed <= 0 or len(pool) == 0:
        return []
    sc = scores.reindex(pool)
    sc = sc.fillna(sc.min() - 1.0 if sc.notna().any() else 0.0).sort_values(ascending=False)
    n = min(int(nneed), len(sc))
    if n <= 0:
        return []
    cand_n = min(max(n * 3, n), len(sc))
    cand = sc.head(cand_n)
    picked = list(cand.head(n).index)
    if target_budget is None or n == len(cand):
        return picked
    bud = budgets.reindex(cand.index).fillna(0.0)
    best = list(picked)
    best_err = abs(float(bud.reindex(best).sum()) - float(target_budget))
    unselected = [x for x in cand.index if x not in set(best)]
    for _ in range(40):
        improved = False
        for i, a in enumerate(list(best)):
            for b in unselected:
                trial = list(best)
                trial[i] = b
                err = abs(float(bud.reindex(trial).sum()) - float(target_budget))
                if err + 1e-9 < best_err:
                    best = trial
                    best_err = err
                    unselected = [x for x in cand.index if x not in set(best)]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _matched_c3_veto(eligible: pd.DatetimeIndex, treat: pd.DatetimeIndex, scores: pd.Series, budgets: pd.Series) -> dict[str, Any]:
    cell = _cell_series(eligible)
    treat_cell = _cell_series(treat) if len(treat) else pd.Series(dtype=str)
    cell_n = {str(k): int(v) for k, v in treat_cell.value_counts().items()} if len(treat_cell) else {}
    year_n = {int(y): int((treat.year == int(y)).sum()) for y in sorted(set(eligible.year))}
    secondary_ok = True
    for key, nneed in cell_n.items():
        if nneed > int((cell == key).sum()):
            secondary_ok = False
            break
    layer = "year_x_b01a" if secondary_ok else "year_only"
    treat_budget = budgets.reindex(treat).fillna(0.0)
    picked: list = []
    if layer == "year_x_b01a":
        for key, nneed in cell_n.items():
            pool = eligible[cell.eq(key)]
            t_pool = treat[treat_cell.eq(key)] if len(treat) else pool[:0]
            target_b = float(treat_budget.reindex(t_pool).sum()) if len(t_pool) else None
            picked.extend(_pick_cell(scores, pool, int(nneed), budgets, target_b))
    else:
        for year, nneed in year_n.items():
            pool = eligible[eligible.year == int(year)]
            t_pool = treat[treat.year == int(year)]
            target_b = float(treat_budget.reindex(t_pool).sum()) if len(t_pool) else None
            picked.extend(_pick_cell(scores, pool, int(nneed), budgets, target_b))
    veto = idx_dates(picked)
    by_year = {}
    year_match = True
    for y in sorted(set(eligible.year) | set(treat.year) | set(veto.year)):
        n_c3 = int((veto.year == int(y)).sum())
        n_o01 = int((treat.year == int(y)).sum())
        by_year[int(y)] = {"n_c3": n_c3, "n_o01": n_o01, "n_eligible": int((eligible.year == int(y)).sum())}
        if n_c3 != n_o01:
            year_match = False
    cell_match = True
    by_cell = {}
    if layer == "year_x_b01a" and cell_n:
        vcell = _cell_series(veto) if len(veto) else pd.Series(dtype=str)
        for key, nneed in cell_n.items():
            got = int((vcell == key).sum()) if len(vcell) else 0
            by_cell[key] = {"n_c3": got, "n_o01": int(nneed)}
            if got != int(nneed):
                cell_match = False
    budget_year = {}
    budget_ok = True
    for y, rec in by_year.items():
        o_sum = float(treat_budget.reindex(treat[treat.year == y]).sum()) if rec["n_o01"] else 0.0
        c_sum = float(budgets.reindex(veto[veto.year == y]).fillna(0.0).sum()) if rec["n_c3"] else 0.0
        rel = None if abs(o_sum) < 1e-9 else abs(c_sum - o_sum) / abs(o_sum)
        budget_year[int(y)] = {"o01_planned_budget": o_sum, "c3_planned_budget": c_sum, "rel_abs_err": rel}
        if rec["n_o01"] > 0 and rel is not None and rel > 0.5:
            budget_ok = False
    matching_pass = bool(year_match and (cell_match if layer == "year_x_b01a" else True) and int(len(veto)) == int(len(treat)))
    return {
        "veto": veto,
        "layer": layer,
        "secondary_ok": secondary_ok,
        "matching_status": ACTIVITY_MATCHING_PASS if matching_pass else (
            MATCHING_CONSTRAINT_FAILURE if not secondary_ok else "ACTIVITY_NOT_FULLY_MATCHED"
        ),
        "activity_matched": matching_pass,
        "year_match": year_match,
        "cell_match": cell_match,
        "budget_ok": budget_ok,
        "by_year": by_year,
        "by_cell": by_cell,
        "year_n": year_n,
        "cell_n": cell_n,
        "budget_year": budget_year,
    }


def run_equity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    existing = resume_ok(EQ_JSON) if resume else None
    if existing is not None:
        return existing
    events = load_json(EVENTS_JSON)
    eligible = idx_dates(events.get("eligible_dates") or [])
    planned = idx_dates(events.get("planned_treat_dates") or [])
    if len(eligible) == 0:
        payload = skip_result("a-equity-control", "ELIGIBLE_DATES_MISSING", status="BLOCKED")
        dump_json(EQ_JSON, payload)
        return payload
    try:
        labels = _c0_top2_labels()
        panel = equity_panel()
        o01_dates = idx_dates(o01_signal_dates())
        t0 = pd.read_parquet(require_c0() / "trades.parquet")
    except Exception as exc:
        payload = skip_result("a-equity-control", f"{type(exc).__name__}:{exc}")
        dump_json(EQ_JSON, payload)
        return payload
    labels["signal_date"] = pd.to_datetime(labels["signal_date"]).dt.normalize()
    feat_cols = list(V03_FEATURES)
    m = labels.merge(panel.reset_index().rename(columns={"index": "signal_date"}), on="signal_date", how="left")
    m = m.sort_values("signal_date")
    o01_elig = eligible.intersection(o01_dates)
    if len(planned):
        o01_elig = planned
    budgets = planned_budget_by_date(top2(t0))

    orig = _original_v03_veto(eligible, o01_elig, m)
    c2_dest = control_dir(A_C2)
    ov2 = apply_top2_veto(cached_overlay_v4(), orig["veto"])
    overlayed_path(A_C2).parent.mkdir(parents=True, exist_ok=True)
    ov2.to_parquet(overlayed_path(A_C2), index=False)
    ran2 = run_overlayed(
        c2_dest,
        ov2,
        extra={"code": A_C2, "round": 19, "role": "original_v03_replay", "not_candidate": True},
        resume=resume,
        write=True,
    )
    jac = _jaccard(orig["veto"], o01_elig)
    c2_flag = None if orig["activity_matched"] else ACTIVITY_NOT_FULLY_MATCHED

    scores = _score_ridge(m)
    matched = _matched_c3_veto(eligible, o01_elig, scores, budgets)
    c3_dest = control_dir(A_C3)
    ov3 = apply_top2_veto(cached_overlay_v4(), matched["veto"])
    ov3.to_parquet(overlayed_path(A_C3), index=False)
    ran3 = run_overlayed(
        c3_dest,
        ov3,
        extra={
            "code": A_C3,
            "round": 19,
            "role": "activity_matched_equity_only_control",
            "not_candidate": True,
            "formal": False,
        },
        resume=resume,
        write=True,
    )
    payload = {
        "status": "DONE",
        "features": feat_cols,
        "alpha": V03_RIDGE_ALPHA,
        "no_future_fill_as_input": True,
        "ex_post_fill_match_label": EX_POST_DIAGNOSTIC_ONLY,
        "c2": {
            "code": A_C2,
            "not_candidate": True,
            "keep_flag": ACTIVITY_NOT_FULLY_MATCHED,
            "n_veto": int(len(orig["veto"])),
            "n_o01_eligible": int(len(o01_elig)),
            "jaccard": jac,
            "overlap": int(len(set(orig["veto"]) & set(o01_elig))),
            "activity_matched": bool(orig["activity_matched"]),
            "activity_flag": c2_flag,
            "by_year": orig["by_year"],
            "observed": ran2.get("observed") or pack_obs_from_dir(c2_dest),
            "veto_dates": ymd_list(orig["veto"]),
            "out_dir": str(c2_dest),
        },
        "c3": {
            "code": A_C3,
            "not_candidate": True,
            "formal": False,
            "n_veto": int(len(matched["veto"])),
            "n_o01_eligible": int(len(o01_elig)),
            "jaccard": _jaccard(matched["veto"], o01_elig),
            "matching_status": matched["matching_status"],
            "activity_matched": bool(matched["activity_matched"]),
            "year_match": matched["year_match"],
            "cell_match": matched["cell_match"],
            "budget_ok": matched["budget_ok"],
            "layer": matched["layer"],
            "by_year": matched["by_year"],
            "by_cell": matched["by_cell"],
            "budget_year": matched["budget_year"],
            "observed": ran3.get("observed") or pack_obs_from_dir(c3_dest),
            "veto_dates": ymd_list(matched["veto"]),
            "out_dir": str(c3_dest),
        },
    }
    dump_json(EQ_JSON, payload)
    write_md(
        ta_md("equity.md"),
        "\n".join(
            [
                "# Round19 Track A · a-equity-control",
                "",
                f"C2 原 V03 n_veto=`{len(orig['veto'])}` Jaccard=`{jac}` flag=`{c2_flag}` Sharpe=`{(payload['c2']['observed'] or {}).get('sharpe')}`。不得晋级。",
                f"C3 匹配 `{matched['matching_status']}` n_veto=`{len(matched['veto'])}` year_match=`{matched['year_match']}` cell_match=`{matched['cell_match']}` Sharpe=`{(payload['c3']['observed'] or {}).get('sharpe')}`。不得晋级。",
                f"特征=`{feat_cols}` Ridge α=`{V03_RIDGE_ALPHA}`。禁止用未来成交/盈亏选日。",
                "",
            ]
        ),
    )
    return payload
