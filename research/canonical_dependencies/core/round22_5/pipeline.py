"""Round 22.5。不新增因子公式，不改 C4，不重调 XGB。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.r225241_io import dump_json, load_json, load_stage, save_stage, sha256_json, write_df, write_parquet, write_text
from core.r225241_stats import (
    assert_scope,
    c4_window_counts,
    daily_spearman,
    grade_edges,
    apply_grades,
    ndcg_at_k,
    paired_block_bootstrap,
    permute_within_date,
)
from core.r225241_sleeve import market, run_round7
from core.paths import RESULT_ROOT
from core.round22_3.contract import material_vs_v5

from .contract import (
    C1,
    C4,
    C4_FORMULA,
    C_AVAILABLE_FROM,
    E0,
    FIVE_FEATURES,
    FS2,
    FS2_FORMULA,
    FS2_PANEL,
    FULL_TRAIN_START,
    HYPOTHESES,
    PANEL,
    PERM_SEEDS,
    R22_3,
    R22_4,
    ROOT,
    SHUFFLE_SHARPE,
    TEN_FEATURES,
    TEN_PANEL,
    TEN_TRAIN,
    TRAIN,
    anchor_copy,
)

LEDGER_ROUNDS = (
    "round21_3",
    "round21_4",
    "round22",
    "round22_1",
    "round22_2",
    "round22_3",
    "round22_4",
    "round23",
    "round24",
)


def _skip(name: str, resume: bool) -> dict[str, Any] | None:
    if not resume:
        return None
    return load_stage(ROOT, name)


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (ROOT / "frozen_manifest.json").is_file():
        return load_json(ROOT / "frozen_manifest.json")
    from core.round22_3.contract import ECONOMIC_MATERIALITY

    body = {
        "round": "22.5",
        "frozen_before_new_nav": True,
        "anchor": anchor_copy(),
        "hypotheses": list(HYPOTHESES),
        "hypotheses_locked": True,
        "economic_materiality": {k: ECONOMIC_MATERIALITY[k] for k in ECONOMIC_MATERIALITY if k != "note"},
        "economic_gate_sha256": sha256_json(
            {k: ECONOMIC_MATERIALITY[k] for k in ("id", "delta_cagr_min_abs", "delta_sharpe_min", "nav_rel_min", "maxdd_worse_max_abs")}
        ),
        "c4_formula_unchanged": C4_FORMULA,
        "fs2_formula": FS2_FORMULA,
        "fs2_is_not_cash_conversion_change": True,
        "no_new_factors": True,
        "no_xgb_retune": True,
        "permutation_seeds": list(PERM_SEEDS),
        "execution_main_sensitivity": "NO_SAME_DAY_REPLACEMENT",
        "replacement_is_separate_scenario": True,
        "current_baseline_changed": False,
        "auto_promote": False,
    }
    dump_json(ROOT / "frozen_manifest.json", body)
    body["status"] = "FROZEN"
    return save_stage(ROOT, "freeze", body)


def _scope_of(row: dict[str, Any], round_name: str) -> str:
    if row.get("scenario") is True or "EXEC" in str(row.get("code")):
        return assert_scope("EXECUTION_SCENARIO")
    code = str(row.get("code") or "")
    if round_name == "round24" or row.get("line") == "second_to_third" or "2to3" in code or code.startswith("T2") or code.startswith("baseline_2to3"):
        return assert_scope("SECOND_TO_THIRD_INDEPENDENT")
    if any(token in code for token in ("ALLNAN", "SHUFFLE", "CTRL", "PLACEBO")):
        return assert_scope("DIAGNOSTIC_ONLY")
    tag = str(row.get("ledger_tag") or row.get("window_tag") or "")
    start = str(row.get("window_start") or "")
    if "MATCHED" in tag or (start and start > "2018-01-02"):
        return assert_scope("STRICT_PIT_SUBPERIOD_FIRST_TO_SECOND")
    return assert_scope("FULL_HISTORY_FIRST_TO_SECOND")


def _top_field(row: dict[str, Any]) -> str:
    for key in ("top1_top2", "top1_change", "top2_change", "top1_change_rate", "top_changed"):
        if key in row and row.get(key) not in (None, ""):
            return str(row.get(key))
    return "NOT_IDENTIFIABLE"


def run_fact_table(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("fact", resume)
    if cached:
        return cached
    rows = []
    for name in LEDGER_ROUNDS:
        base = RESULT_ROOT / name
        if not base.is_dir():
            rows.append({"round": name, "code": "MISSING_ARTIFACT", "scope": "DIAGNOSTIC_ONLY", "source_path": str(base)})
            continue
        for path in base.rglob("summary.json"):
            if "_state" in path.parts:
                continue
            try:
                payload = json_load(path)
            except Exception:
                continue
            if payload.get("sharpe") is None and payload.get("nav_end") is None:
                continue
            rec = {
                "round": name,
                "code": payload.get("code") or path.parent.name,
                "sharpe": payload.get("sharpe"),
                "cagr": payload.get("cagr"),
                "max_drawdown": payload.get("max_drawdown"),
                "nav_end": payload.get("nav_end"),
                "trades": payload.get("completed_trades") or payload.get("trades"),
                "delta_sharpe": payload.get("delta_sharpe"),
                "delta_cagr": payload.get("delta_cagr"),
                "economically_material": payload.get("economically_material"),
                "scenario": payload.get("scenario"),
                "window_start": payload.get("window_start"),
                "window_end": payload.get("window_end"),
                "fee_schedule": payload.get("fee_schedule"),
                "top1_top2": _top_field(payload),
                "source_path": str(path),
            }
            rec["scope"] = _scope_of({**payload, **rec}, name)
            rows.append(rec)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["round", "code", "source_path"])
    write_parquet(ROOT / "experiment_fact_table.parquet", frame)
    registry = []
    for _, rec in frame.iterrows():
        registry.append(
            {
                "hypothesis_id": f"LEDGER_{rec.get('round')}_{rec.get('code')}",
                "family": rec.get("round"),
                "test_name": rec.get("code"),
                "scope": rec.get("scope"),
                "pre_specified": True,
                "artifact": rec.get("source_path"),
                "top1_top2": rec.get("top1_top2") or "NOT_IDENTIFIABLE",
                "p_value": "NOT_COMPUTED",
                "result": "RECORDED",
            }
        )
    for hyp in HYPOTHESES:
        registry.append(
            {
                "hypothesis_id": hyp["id"],
                "family": "round22_5",
                "test_name": hyp["claim"],
                "scope": "FULL_HISTORY_FIRST_TO_SECOND" if "FS2" not in hyp["id"] else "DIAGNOSTIC_ONLY",
                "pre_specified": True,
                "artifact": "",
                "top1_top2": "NOT_IDENTIFIABLE",
                "p_value": "NOT_COMPUTED",
                "result": "PENDING",
            }
        )
    write_df(ROOT / "multiple_testing_registry.csv", pd.DataFrame(registry))
    return save_stage(ROOT, "fact", {"status": "DONE", "n_rows": int(len(frame)), "n_registry": int(len(registry))})


def json_load(path: Path) -> dict[str, Any]:
    return load_json(path)


def _fold_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in range(2018, 2027):
        train = panel.loc[panel["trade_date"].dt.year.between(year - 3, year - 1)]
        test = panel.loc[panel["trade_date"].dt.year.eq(year)]
        rec = {"predict_year": year, "train_rows": int(len(train)), "test_rows": int(len(test))}
        for col in (C1, C4, "r22_c1_xl_net_imbalance"):
            rec[f"{col}_train_finite"] = float(pd.to_numeric(train[col], errors="coerce").notna().mean()) if len(train) else None
            rec[f"{col}_test_finite"] = float(pd.to_numeric(test[col], errors="coerce").notna().mean()) if len(test) else None
        rows.append(rec)
    return pd.DataFrame(rows)


def run_coverage(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("coverage", resume)
    if cached:
        return cached
    cols = ["trade_date", "instrument_id", C1, C4, "r22_c1_xl_net_imbalance"]
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    folds = _fold_coverage(panel)
    write_df(ROOT / "c1_c4_fold_coverage.csv", folds)
    flow = _c4_flow_counts(panel)
    write_text(
        ROOT / "02_c4_rolling.md",
        "\n".join(
            [
                "# C4 滚动语义",
                "",
                f"原公式保持 `{C4_FORMULA}`。没有改写。",
                f"可得性起点 {C_AVAILABLE_FROM}。",
                "按行 rolling 不把周末或缺失交易日插进窗口。缺行时窗口吃到更早的已有行，日历跨度可以长于 5 个交易日。",
                "",
                "```",
                folds.to_string(index=False),
                "```",
                "",
                "```",
                pd.Series(flow).to_string(),
                "```",
                "",
            ]
        ),
    )
    dump_json(ROOT / "c4_rolling_semantics.json", flow)
    return save_stage(ROOT, "coverage", {"status": "DONE", "flow": flow, "years": folds["predict_year"].tolist()})


def _c4_flow_counts(panel: pd.DataFrame) -> dict[str, Any]:
    """按文件读 NET_FLOW，避免分区 schema 不一致时 dataset 强转失败。"""
    import pyarrow.parquet as pq

    from core.round21.contract import HERMES_REBUILD_RAW
    from core.round21.panel import market_to_instrument

    root = HERMES_REBUILD_RAW / "mkt_equ_mf_new"
    if not root.exists():
        return {"status": "MISSING_ARTIFACT", "formula": C4_FORMULA}
    wanted = set(panel["instrument_id"].astype(str))
    tickers = {item.split(".", 1)[0] for item in wanted}
    parts = []
    for path in sorted(root.rglob("*.parquet")):
        table = pq.read_table(path, columns=["TRADE_DATE", "TICKER_SYMBOL", "EXCHANGE_CD", "NET_FLOW"])
        chunk = table.to_pandas()
        chunk["TICKER_SYMBOL"] = chunk["TICKER_SYMBOL"].astype(str)
        chunk = chunk.loc[chunk["TICKER_SYMBOL"].isin(tickers)]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        return {"status": "MISSING_ARTIFACT", "formula": C4_FORMULA, "reason": "no overlapping NET_FLOW rows"}
    flow = pd.concat(parts, ignore_index=True)
    exch = flow["EXCHANGE_CD"].astype(str)
    tick = flow["TICKER_SYMBOL"].astype(str)
    raw_code = tick.where(~exch.isin(["XSHG", "XSHE", "XBEI", "SH", "SZ", "BJ"]), tick + "." + exch)
    flow["instrument_id"] = raw_code.map(market_to_instrument).astype(str)
    flow["trade_date"] = pd.to_datetime(flow["TRADE_DATE"])
    flow = flow.loc[flow["instrument_id"].isin(wanted)].dropna(subset=["trade_date", "instrument_id"])
    flow = flow.sort_values(["instrument_id", "trade_date"]).drop_duplicates(["instrument_id", "trade_date"])
    flow["sign"] = np.sign(pd.to_numeric(flow["NET_FLOW"], errors="coerce"))
    flow["shifted"] = flow.groupby("instrument_id", sort=False)["sign"].shift(1)
    counts = (
        flow.groupby("instrument_id", sort=False)["shifted"]
        .rolling(5, min_periods=3)
        .count()
        .reset_index(level=0, drop=True)
        .reindex(flow.index)
    )
    valid = counts[counts >= 3]
    gaps = flow.groupby("instrument_id", sort=False)["trade_date"].diff().dt.days
    acc = {
        "n_valid_3": int((valid == 3).sum()),
        "n_valid_4": int((valid == 4).sum()),
        "n_valid_5": int((valid == 5).sum()),
        "n_valid": int(valid.shape[0]),
        "status": "OK",
        "formula": C4_FORMULA,
        "source": str(root),
        "column": "NET_FLOW",
        "rolling_inserts_calendar_gaps": False,
        "rolling_is_row_based": True,
        "calendar_gaps_present": bool((gaps > 3).fillna(False).any()),
        "n_instruments": int(flow["instrument_id"].nunique()),
        "n_flow_rows": int(len(flow)),
        "rewritten": False,
    }
    return acc


def run_null(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("null", resume)
    if cached:
        return cached
    allnan = load_json(R22_4 / "null" / "R224_CTRL_ALLNAN" / "sleeve" / "summary.json")
    shuffle = load_json(R22_4 / "null" / "R224_CTRL_SHUFFLE" / "sleeve" / "summary.json")
    v5 = load_json(E0 / "summary.json")
    if not allnan or not shuffle or not v5:
        payload = {"status": "BLOCKED", "reason": "MISSING_ARTIFACT"}
        dump_json(ROOT / "null_recheck.json", payload)
        return save_stage(ROOT, "null", payload)
    nav_same = abs(float(allnan["nav_end"]) - float(v5["nav_end"])) <= 1e-4
    sharpe_ok = abs(float(shuffle["sharpe"]) - SHUFFLE_SHARPE) <= 1e-9
    recheck = {
        "allnan_nav": allnan.get("nav_end"),
        "v5_nav": v5.get("nav_end"),
        "allnan_nav_identical": nav_same,
        "allnan_sharpe": allnan.get("sharpe"),
        "shuffle_sharpe": shuffle.get("sharpe"),
        "shuffle_matches_frozen": sharpe_ok,
        "frozen_shuffle_sharpe": SHUFFLE_SHARPE,
        "counts_as_alpha": False,
        "source_allnan": str(R22_4 / "null" / "R224_CTRL_ALLNAN" / "sleeve" / "summary.json"),
        "source_shuffle": str(R22_4 / "null" / "R224_CTRL_SHUFFLE" / "sleeve" / "summary.json"),
    }
    overlay = pd.read_parquet(E0 / "overlayed.parquet")
    overlay["signal_date"] = pd.to_datetime(overlay["signal_date"])
    perm_rows = []
    for seed in PERM_SEEDS:
        shuffled = permute_within_date(overlay, ["score", "allocation_multiplier"], "signal_date", int(seed))
        sleeve = run_round7(
            ROOT / "placebo" / f"PERM_{seed}",
            shuffled,
            {
                "code": f"R225_PERM_{seed}",
                "round": "22.5",
                "scope": "DIAGNOSTIC_ONLY",
                "role": "WITHIN_DATE_CROSS_SECTIONAL_PERMUTATION",
                "seed": int(seed),
            },
            resume=resume,
            scenario=False,
            counts_as_alpha=False,
            already_overlayed=True,
        )
        perm_rows.append({"seed": int(seed), "sharpe": sleeve.get("sharpe"), "cagr": sleeve.get("cagr"), "nav_end": sleeve.get("nav_end"), "counts_as_alpha": False})
    recheck["permutations"] = perm_rows
    dump_json(ROOT / "null_recheck.json", recheck)
    write_df(ROOT / "permutation_placebo.csv", pd.DataFrame(perm_rows))
    return save_stage(ROOT, "null", {"status": "DONE", **{k: recheck[k] for k in ("allnan_nav_identical", "shuffle_matches_frozen")}})


def _slice_wealth(base: pd.DataFrame, other: pd.DataFrame, start: str | None, end: str | None) -> dict[str, Any]:
    from core.round11.common import daily_from_nav

    a = daily_from_nav(base)
    b = daily_from_nav(other)
    both = pd.concat([a.rename("base"), b.rename("new")], axis=1, join="inner").dropna()
    if start:
        both = both.loc[both.index >= pd.Timestamp(start)]
    if end:
        both = both.loc[both.index < pd.Timestamp(end)]
    if len(both) < 5:
        return {"n_obs": int(len(both)), "status": "TOO_SHORT"}
    excess = both["new"] - both["base"]
    vol = float(excess.std(ddof=1))
    wealth_b = float((1.0 + both["base"]).prod())
    wealth_n = float((1.0 + both["new"]).prod())
    return {
        "n_obs": int(len(both)),
        "excess_sharpe_not_comparable_across_slices": float(excess.mean() / vol * np.sqrt(252)) if vol > 0 else None,
        "slice_wealth_ratio": wealth_n / wealth_b if wealth_b else None,
        "sum_excess_log": float(np.log1p(both["new"]).sum() - np.log1p(both["base"]).sum()),
        "status": "OK",
        "partial_vs_full_note": "切片超额 Sharpe 不与全历史 1.876 混排",
    }


def _tops(pred: pd.DataFrame, k: int) -> pd.DataFrame:
    work = pred.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort")
    work["rk"] = work.groupby("signal_date").cumcount() + 1
    return work.loc[work["rk"].le(k), ["signal_date", "instrument_id", "rk", "score"]]


def _changed_pnl(base_pred: pd.DataFrame, new_pred: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    out = {}
    lab = labels.copy()
    lab["signal_date"] = pd.to_datetime(lab["trade_date"])
    for k, name in ((1, "top1"), (2, "top2")):
        a = _tops(base_pred, k).rename(columns={"instrument_id": "base_id"})
        b = _tops(new_pred, k).rename(columns={"instrument_id": "new_id"})
        merged = a.merge(b, on=["signal_date", "rk"], how="inner")
        changed = merged.loc[merged["base_id"] != merged["new_id"]]
        if changed.empty:
            out[name] = {"n_changed": 0, "mean_matured_label_new": None, "mean_matured_label_base": None}
            continue
        new_lab = changed.merge(lab, left_on=["signal_date", "new_id"], right_on=["signal_date", "instrument_id"], how="left")
        base_lab = changed.merge(lab, left_on=["signal_date", "base_id"], right_on=["signal_date", "instrument_id"], how="left")
        out[name] = {
            "n_changed": int(len(changed)),
            "n_days": int(changed["signal_date"].nunique()),
            "mean_matured_label_new": float(pd.to_numeric(new_lab["label_e2_net"], errors="coerce").mean()),
            "mean_matured_label_base": float(pd.to_numeric(base_lab["label_e2_net"], errors="coerce").mean()),
        }
    return out


def run_attribution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("attribution", resume)
    if cached:
        return cached
    base_nav = pd.read_parquet(E0 / "nav.parquet")
    c1_nav_path = R22_3 / "candidates" / "R223_E1_r22_2_c1_xl_net_imbalance_lag1" / "nav.parquet"
    c4_nav_path = R22_3 / "candidates" / "R223_E1_r22_c4_flow_persist_5" / "nav.parquet"
    if not c1_nav_path.is_file() or not c4_nav_path.is_file():
        payload = {"status": "BLOCKED", "reason": "MISSING_ARTIFACT"}
        return save_stage(ROOT, "attribution", payload)
    c1_nav = pd.read_parquet(c1_nav_path)
    c4_nav = pd.read_parquet(c4_nav_path)
    slices = {
        "PRE_AVAILABILITY": (None, C_AVAILABLE_FROM),
        "PARTIAL": (C_AVAILABLE_FROM, FULL_TRAIN_START),
        "FULL_TRAIN": (FULL_TRAIN_START, None),
    }
    c1_slices = {name: _slice_wealth(base_nav, c1_nav, *bounds) for name, bounds in slices.items()}
    c4_slices = {name: _slice_wealth(base_nav, c4_nav, *bounds) for name, bounds in slices.items()}
    partial = c1_slices["PARTIAL"].get("slice_wealth_ratio")
    full = c1_slices["FULL_TRAIN"].get("slice_wealth_ratio")
    c1_verdict = "C1_NO_STABLE_INCREMENTAL_ALPHA" if full is not None and full < 1 else "C1_FULL_TRAIN_NONNEGATIVE"
    labels = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", "label_e2_net", "label_exit_rule_net", "g1_ret_5", C4, "actual_exit_date"])
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    five_v5 = pd.read_parquet(E0 / "five_pred.parquet")
    five_c1 = pd.read_parquet(R22_3 / "candidates" / "R223_E1_r22_2_c1_xl_net_imbalance_lag1" / "five_pred.parquet")
    five_c4 = pd.read_parquet(R22_3 / "candidates" / "R223_E1_r22_c4_flow_persist_5" / "five_pred.parquet")
    ten_v5 = pd.read_parquet(E0 / "ten_pred.parquet")
    for frame in (five_v5, five_c1, five_c4, ten_v5):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    mask_dates = labels["trade_date"] >= pd.Timestamp(FULL_TRAIN_START)
    lab_ft = labels.loc[mask_dates]
    def _clip(pred: pd.DataFrame) -> pd.DataFrame:
        return pred.loc[pred["signal_date"] >= pd.Timestamp(FULL_TRAIN_START)].copy()
    c1_change = _changed_pnl(_clip(five_v5), _clip(five_c1), lab_ft)
    c4_change = _changed_pnl(_clip(five_v5), _clip(five_c4), lab_ft)
    c4_ic = _c4_conditional(lab_ft, _clip(five_v5), _clip(ten_v5))
    c4_ndcg = _c4_ndcg(lab_ft, _clip(five_c4), labels)
    from core.round11.common import daily_from_nav

    boot = paired_block_bootstrap(
        daily_from_nav(c4_nav).loc[lambda s: s.index >= pd.Timestamp(FULL_TRAIN_START)],
        daily_from_nav(base_nav).loc[lambda s: s.index >= pd.Timestamp(FULL_TRAIN_START)],
    )
    yearly = {}
    for year, end in ((2024, "2025-01-01"), (2025, "2026-01-01"), (2026, None)):
        rec = _slice_wealth(base_nav, c4_nav, f"{year}-01-01", end)
        rec["full_year"] = year != 2026
        rec["note"] = "2026 不是完整自然年" if year == 2026 else "完整日历年切片，不与全历史 Sharpe 混排"
        yearly[str(year)] = rec
    c1_summary = load_json(R22_3 / "candidates" / "R223_E1_r22_2_c1_xl_net_imbalance_lag1" / "summary.json")
    c4_summary = load_json(R22_3 / "candidates" / "R223_E1_r22_c4_flow_persist_5" / "summary.json")
    c1_payload = {
        "verdict": c1_verdict,
        "slices": c1_slices,
        "partial_wealth_recomputed": partial,
        "full_train_wealth_recomputed": full,
        "historical_partial_wealth_claim": 1.263687,
        "historical_full_train_wealth_claim": 0.806711,
        "changed": c1_change,
        "full_history_summary": {k: c1_summary.get(k) for k in ("sharpe", "cagr", "delta_sharpe", "delta_cagr", "nav_end", "economically_material")},
        "new_combinations": False,
    }
    c4_payload = {
        "slices": c4_slices,
        "yearly_inside_full_train": yearly,
        "conditional_ic": c4_ic,
        "ndcg": c4_ndcg,
        "changed_top": c4_change,
        "bootstrap": boot,
        "bootstrap_is_posterior": False,
        "full_history_summary": {k: c4_summary.get(k) for k in ("sharpe", "cagr", "delta_sharpe", "delta_cagr", "nav_end", "max_drawdown", "economically_material")},
    }
    dump_json(ROOT / "c1_full_train_attribution.json", c1_payload)
    dump_json(ROOT / "c4_full_train_attribution.json", c4_payload)
    return save_stage(ROOT, "attribution", {"status": "DONE", "c1_verdict": c1_verdict, "c4_full_wealth": c4_slices["FULL_TRAIN"].get("slice_wealth_ratio")})


def _c4_conditional(labels: pd.DataFrame, five: pd.DataFrame, ten: pd.DataFrame) -> dict[str, Any]:
    work = labels.rename(columns={"trade_date": "signal_date"})
    work = work.merge(five.rename(columns={"score": "five"}), on=["signal_date", "instrument_id"], how="inner")
    work = work.merge(ten.rename(columns={"score": "ten"}), on=["signal_date", "instrument_id"], how="inner")
    out = {
        "ic_label_e2_net": daily_spearman(work, C4, "label_e2_net"),
        "ic_label_exit_rule_net": daily_spearman(work, C4, "label_exit_rule_net"),
        "ic_vs_g1_ret_5": daily_spearman(work, C4, "g1_ret_5"),
    }
    residuals = []
    for _, block in work.groupby("signal_date", sort=False):
        if len(block) < 8:
            continue
        x = pd.to_numeric(block["g1_ret_5"], errors="coerce")
        y = pd.to_numeric(block[C4], errors="coerce")
        z = pd.to_numeric(block["label_e2_net"], errors="coerce")
        ok = x.notna() & y.notna() & z.notna()
        if int(ok.sum()) < 8 or x[ok].nunique() < 2:
            continue
        coef_y = np.polyfit(x[ok], y[ok], 1)
        coef_z = np.polyfit(x[ok], z[ok], 1)
        y_res = y[ok] - (coef_y[0] * x[ok] + coef_y[1])
        z_res = z[ok] - (coef_z[0] * x[ok] + coef_z[1])
        if y_res.nunique() < 2 or z_res.nunique() < 2:
            continue
        residuals.append(float(y_res.corr(z_res, method="spearman")))
    out["partial_ic_given_g1_ret_5"] = float(np.mean(residuals)) if residuals else None
    out["partial_ic_days"] = int(len(residuals))
    for head in ("five", "ten"):
        subset = []
        for _, block in work.groupby("signal_date", sort=False):
            med = pd.to_numeric(block[head], errors="coerce").median()
            part = block.loc[pd.to_numeric(block[head], errors="coerce") >= med]
            if len(part) >= 5:
                subset.append(part)
        if not subset:
            out[f"ic_given_{head}_top_half"] = None
            continue
        stacked = pd.concat(subset, ignore_index=True)
        out[f"ic_given_{head}_top_half"] = daily_spearman(stacked, C4, "label_e2_net")
    out["status"] = "OK"
    return out


def _c4_ndcg(labels_ft: pd.DataFrame, pred: pd.DataFrame, labels_all: pd.DataFrame) -> dict[str, Any]:
    hist = labels_all.loc[labels_all["trade_date"].between("2021-01-01", "2023-12-31"), "label_e2_net"]
    edges = grade_edges(pd.to_numeric(hist, errors="coerce").to_numpy(), n_grades=5)
    work = labels_ft.rename(columns={"trade_date": "signal_date"}).merge(pred, on=["signal_date", "instrument_id"], how="inner")
    work["grade"] = apply_grades(pd.to_numeric(work["label_e2_net"], errors="coerce").to_numpy(), edges)
    return {
        "edges_fit_on": "2021-01-01..2023-12-31 label_e2_net",
        "edges": edges.tolist(),
        "ndcg_at_1": ndcg_at_k(work, k=1),
        "ndcg_at_2": ndcg_at_k(work, k=2),
        "grade_fit_on_outer": False,
    }


def run_lineage(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("lineage", resume)
    if cached:
        return cached
    nav = pd.read_parquet(E0 / "nav.parquet")
    trades = pd.read_parquet(E0 / "trades.parquet")
    costs = pd.read_parquet(E0 / "costs.parquet")
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    gap = (nav["cash"] + nav["market_value"] - nav["total_asset"]).abs()
    identity_ok = bool(gap.max() <= 1e-4)
    closed = trades.loc[trades["actual_exit_date"].notna()].copy()
    closed_pnl = float((closed["sell_proceeds"] - closed["buy_cost"] + closed["cash_dividend"].fillna(0)).sum())
    start_nav = float(nav["total_asset"].iloc[0])
    end_nav = float(nav["total_asset"].iloc[-1])
    end_mv = float(nav["market_value"].iloc[-1])
    end_cash = float(nav["cash"].iloc[-1])
    open_cost = float(trades.loc[trades["actual_exit_date"].isna(), "buy_cost"].sum()) if trades["actual_exit_date"].isna().any() else 0.0
    explained = start_nav + closed_pnl + (end_mv - open_cost)
    residual = end_nav - explained
    fee = {
        "commission": float(costs["commission"].sum()),
        "transfer_fee": float(costs["transfer_fee"].sum()),
        "stamp_tax": float(costs["stamp_tax"].sum()),
    }
    fee["total"] = fee["commission"] + fee["transfer_fee"] + fee["stamp_tax"]
    rel = abs(residual) / end_nav if end_nav else None
    reconciled = bool(identity_ok and rel is not None and rel <= 0.01)
    payload = {
        "status": "RECONCILED" if reconciled else "ATTRIBUTION_NOT_RECONCILED",
        "chain": [
            "Prediction",
            "Rank",
            "B03",
            "O01",
            "Order",
            "Fill",
            "Position",
            "Exit",
            "Net PnL",
        ],
        "artifacts": {
            "prediction": str(E0 / "predictions.parquet"),
            "overlay": str(E0 / "overlayed.parquet"),
            "trades": str(E0 / "trades.parquet"),
            "positions": str(E0 / "positions.parquet"),
            "nav": str(E0 / "nav.parquet"),
            "costs": str(E0 / "costs.parquet"),
        },
        "cash_plus_market_value_equals_total_asset": identity_ok,
        "max_identity_gap": float(gap.max()),
        "start_nav": start_nav,
        "end_nav": end_nav,
        "end_cash": end_cash,
        "end_market_value_unrealized": end_mv,
        "closed_trade_cash_pnl": closed_pnl,
        "open_buy_cost_still_held": open_cost,
        "residual_after_cash_pnl_and_unrealized_cost": residual,
        "residual_over_nav": rel,
        "fees_already_inside_trade_cash": fee,
        "compounding": "日收益由总资产序列复利，不是单利加总",
        "n_trades": int(len(trades)),
    }
    dump_json(ROOT / "lineage_nav_bridge.json", payload)
    return save_stage(ROOT, "lineage", {"status": "DONE", "reconciled": reconciled, "verdict": payload["status"]})


def run_fs2(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    cached = _skip("fs2", resume)
    if cached:
        return cached
    from core.r225241_models import train_xgb_ranker
    from core.round7.contract import FIVE_HEAD, TEN_HEAD

    if not FS2_PANEL.is_file():
        payload = {"status": "BLOCKED", "reason": "MISSING_ARTIFACT", "verdict": "FS2_OOF_SCREEN_FAIL"}
        dump_json(ROOT / "fs2_oof.json", payload)
        return save_stage(ROOT, "fs2", payload)
    fs = pd.read_parquet(FS2_PANEL)
    fs["trade_date"] = pd.to_datetime(fs["trade_date"])
    fs["known_at"] = pd.to_datetime(fs["known_at"])
    fs["instrument_id"] = fs["instrument_id"].astype(str)
    future = int((fs["known_at"] > fs["trade_date"]).sum())
    fs = fs.loc[fs["known_at"].notna() & fs["known_at"].le(fs["trade_date"])].copy()
    fs = fs.sort_values("known_at").groupby(["trade_date", "instrument_id"], as_index=False).tail(1)
    staleness = (fs["trade_date"] - fs["known_at"]).dt.days
    pit = {
        "formula": FS2_FORMULA,
        "not_cash_conversion_change": True,
        "future_known_at_rows_before_filter": future,
        "asof": "known_at<=trade_date 后按标的和交易日取最近版本",
        "version_collapse": "latest known_at",
        "staleness_days_median": float(staleness.median()) if len(staleness) else None,
        "staleness_days_p90": float(staleness.quantile(0.9)) if len(staleness) else None,
        "filled_with_zero": False,
        "pit_pass": future == 0 and fs[FS2].notna().any(),
    }
    five = pd.read_parquet(PANEL)
    ten = pd.read_parquet(TEN_PANEL)
    five_train = pd.read_parquet(TRAIN)
    ten_train = pd.read_parquet(TEN_TRAIN)
    for frame in (five, ten, five_train, ten_train):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        extra = fs[["trade_date", "instrument_id", FS2]]
        frame.drop(columns=[FS2], errors="ignore", inplace=True)
        merged = frame.merge(extra, on=["trade_date", "instrument_id"], how="left")
        frame[FS2] = merged[FS2].to_numpy()
    ic = {
        "five_label": daily_spearman(five.rename(columns={"trade_date": "signal_date"}), FS2, "label_e2_net"),
        "ten_label": daily_spearman(ten.rename(columns={"trade_date": "signal_date"}), FS2, "label_exit_rule_net"),
        "g1": daily_spearman(five.rename(columns={"trade_date": "signal_date"}), FS2, "g1_ret_5"),
    }
    coverage = []
    for year in range(2018, 2027):
        block = five.loc[five["trade_date"].dt.year.eq(year), FS2]
        coverage.append({"predict_year": year, "finite_share": float(block.notna().mean()) if len(block) else 0.0, "complete_year": year < 2026})
    heads = {}
    if pit["pit_pass"]:
        heads["five"] = train_xgb_ranker(
            training=five_train,
            candidates=five,
            features=list(FIVE_FEATURES) + [FS2],
            extras=[FS2],
            target=FIVE_HEAD.target,
            label_end_column=FIVE_HEAD.label_end_column,
            n_bins=FIVE_HEAD.rank_n_bins,
            eval_metric=FIVE_HEAD.eval_metric,
            output_dir=ROOT / "fs2" / "five",
            seed=17,
            n_jobs=jobs,
            resume=resume,
        )
        heads["ten"] = train_xgb_ranker(
            training=ten_train,
            candidates=ten,
            features=list(TEN_FEATURES) + [FS2],
            extras=[FS2],
            target=TEN_HEAD.target,
            label_end_column=TEN_HEAD.label_end_column,
            n_bins=TEN_HEAD.rank_n_bins,
            eval_metric=TEN_HEAD.eval_metric,
            output_dir=ROOT / "fs2" / "ten",
            seed=17,
            n_jobs=jobs,
            resume=resume,
        )
    screens = {}
    for head, pred_name, label_name in (
        ("five", ROOT / "fs2" / "five" / "predictions.parquet", "label_e2_net"),
        ("ten", ROOT / "fs2" / "ten" / "predictions.parquet", "label_exit_rule_net"),
    ):
        if not pred_name.is_file():
            screens[head] = {"status": "NOT_TRAINED", "pass": False}
            continue
        pred = pd.read_parquet(pred_name)
        pred["signal_date"] = pd.to_datetime(pred["signal_date"])
        base = five if head == "five" else ten
        lab = base[["trade_date", "instrument_id", label_name, "actual_exit_date"]].rename(columns={"trade_date": "signal_date"})
        joined = pred.merge(lab, on=["signal_date", "instrument_id"], how="left")
        joined = joined.loc[joined["actual_exit_date"].notna()]
        year_means = []
        for year, block in joined.groupby(joined["signal_date"].dt.year):
            if int(year) >= 2026:
                continue
            top = _tops(block, 2)
            m = top.merge(block[["signal_date", "instrument_id", label_name]], on=["signal_date", "instrument_id"], how="left")
            year_means.append({"year": int(year), "top2_mean": float(pd.to_numeric(m[label_name], errors="coerce").mean()), "n": int(len(m))})
        means = [row["top2_mean"] for row in year_means if row["top2_mean"] == row["top2_mean"]]
        positive = bool(means) and float(np.mean(means)) > 0
        same = 0.0 if not means else float(np.mean([np.sign(v) == np.sign(np.mean(means)) for v in means]))
        complete_years = len(means)
        passed = bool(pit["pit_pass"] and complete_years >= 3 and positive and same >= 0.60)
        screens[head] = {
            "pass": passed,
            "complete_outer_years": complete_years,
            "top2_mean": float(np.mean(means)) if means else None,
            "same_sign_share": same,
            "years": year_means,
        }
    any_pass = any(item.get("pass") for item in screens.values())
    payload = {
        "pit": pit,
        "conditional_ic": ic,
        "fold_coverage": coverage,
        "screens": screens,
        "both_add": False,
        "remove_replace": False,
        "new_params": False,
        "sleeve": "NOT_RUN",
        "verdict": "FS2_OOF_SCREEN_FAIL" if not any_pass else "FS2_OOF_SCREEN_PASS",
    }
    if any_pass:
        for head, pred_name in (("five", ROOT / "fs2" / "five" / "predictions.parquet"), ("ten", ROOT / "fs2" / "ten" / "predictions.parquet")):
            if not screens.get(head, {}).get("pass"):
                continue
            other = pd.read_parquet(E0 / ("ten_pred.parquet" if head == "five" else "five_pred.parquet"))
            this = pd.read_parquet(pred_name)
            five_s, ten_s = (this, other) if head == "five" else (other, this)
            sleeve = run_round7(
                ROOT / "fs2" / f"sleeve_{head}",
                _fuse_existing(five_s, ten_s),
                {"code": f"R225_FS2_{head.upper()}_ADD", "round": "22.5", "scope": "FULL_HISTORY_FIRST_TO_SECOND", "top_changed": True},
                resume=resume,
            )
            payload["sleeve"] = sleeve
            gate = bool(sleeve.get("economically_material"))
            payload["verdict"] = "V6_RESEARCH_CANDIDATE" if gate else "PREDICTIVE_NOT_ECONOMIC"
            break
    dump_json(ROOT / "fs2_oof.json", payload)
    return save_stage(ROOT, "fs2", {"status": "DONE", "verdict": payload["verdict"]})


def _fuse_existing(five: pd.DataFrame, ten: pd.DataFrame) -> pd.DataFrame:
    from core.round17.backtest import fuse_heads

    return fuse_heads(five, ten)


def _entry_table(overlay: pd.DataFrame) -> pd.DataFrame:
    prepared = market()["prepared"]
    calendar = pd.DatetimeIndex(prepared.calendar)
    pos = {pd.Timestamp(day): i for i, day in enumerate(calendar)}
    work = overlay.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort").copy()
    work["rk"] = work.groupby("signal_date").cumcount() + 1
    top = work.loc[work["rk"].le(2)].copy()
    top["entry_date"] = top["signal_date"].map(lambda day: calendar[pos[pd.Timestamp(day)] + 1] if pd.Timestamp(day) in pos and pos[pd.Timestamp(day)] + 1 < len(calendar) else pd.NaT)
    lookup = prepared.market_lookup
    rows = []
    for rec in top.itertuples(index=False):
        if pd.isna(rec.entry_date):
            continue
        try:
            bar = lookup.loc[(pd.Timestamp(rec.entry_date), rec.instrument_id)]
            sig = lookup.loc[(pd.Timestamp(rec.signal_date), rec.instrument_id)]
        except KeyError:
            continue
        limit_up = float(bar["limit_up"]) if np.isfinite(bar["limit_up"]) and 0 < float(bar["limit_up"]) < 99_999 else np.nan
        gap = float(bar["open"]) / float(sig["close"]) - 1.0 if np.isfinite(bar["open"]) and np.isfinite(sig["close"]) and float(sig["close"]) > 0 else np.nan
        sealed = bool(np.isfinite(limit_up) and np.isfinite(bar["open"]) and float(bar["open"]) + 1e-6 >= limit_up - 0.005)
        rows.append(
            {
                "signal_date": rec.signal_date,
                "instrument_id": rec.instrument_id,
                "score": rec.score,
                "allocation_multiplier": getattr(rec, "allocation_multiplier", 1.0),
                "rk": int(rec.rk),
                "gap": gap,
                "open_limit_up": sealed,
                "limit_price": limit_up,
            }
        )
    return pd.DataFrame(rows)


def _scenario_frame(full: pd.DataFrame, entry: pd.DataFrame, kind: str) -> pd.DataFrame:
    chosen = entry.copy()
    if kind == "S3":
        chosen = chosen.loc[~chosen["open_limit_up"]]
    elif kind == "S4":
        chosen.loc[chosen["gap"] >= 0.07, "allocation_multiplier"] = chosen.loc[chosen["gap"] >= 0.07, "allocation_multiplier"] * 0.5
    elif kind == "S5":
        chosen = chosen.loc[~(chosen["gap"] >= 0.09)]
    elif kind == "S3_REPLACEMENT":
        blocked = set(zip(entry.loc[entry["open_limit_up"], "signal_date"], entry.loc[entry["open_limit_up"], "instrument_id"]))
        ranked = full.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort")
        parts = []
        for day, block in ranked.groupby("signal_date", sort=False):
            kept = []
            for rec in block.itertuples(index=False):
                if (rec.signal_date, rec.instrument_id) in blocked:
                    continue
                kept.append(rec)
                if len(kept) == 2:
                    break
            if kept:
                parts.append(pd.DataFrame(kept))
        return pd.concat(parts, ignore_index=True) if parts else full.iloc[0:0]
    return chosen


def run_execution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("execution", resume)
    if cached:
        return cached
    overlay = pd.read_parquet(E0 / "overlayed.parquet")
    overlay["signal_date"] = pd.to_datetime(overlay["signal_date"])
    v5 = load_json(E0 / "summary.json")
    rows = [
        {
            "code": "R225_S0",
            "scenario": "S0",
            "status": "REUSED_R223_E0",
            "sharpe": v5.get("sharpe"),
            "cagr": v5.get("cagr"),
            "nav_end": v5.get("nav_end"),
            "max_drawdown": v5.get("max_drawdown"),
            "completed_trades": v5.get("completed_trades"),
            "same_day_replacement": False,
            "counts_as_alpha": False,
        }
    ]
    entry = _entry_table(overlay)
    dump_json(
        ROOT / "open_limit_frequency.json",
        {
            "n_top2_slots": int(len(entry)),
            "open_limit_up_share": float(entry["open_limit_up"].mean()) if len(entry) else None,
            "gap_ge_7_share": float((entry["gap"] >= 0.07).mean()) if len(entry) else None,
            "gap_ge_9_share": float((entry["gap"] >= 0.09).mean()) if len(entry) else None,
            "limit_price_source": "market limit_up when finite and not sentinel; no flat 1.099 rule",
            "main_sensitivity": "NO_SAME_DAY_REPLACEMENT",
        },
    )
    entry.to_parquet(ROOT / "execution_entry_audit.parquet", index=False)
    specs = [
        ("S1", "R225_S1_SLIP10", {"buy_extra_slippage_bps": 10}, None),
        ("S2", "R225_S2_SLIP20", {"buy_extra_slippage_bps": 20}, None),
        ("S3", "R225_S3_OPEN_LIMIT", None, "S3"),
        ("S4", "R225_S4_GAP7_FILL50", None, "S4"),
        ("S5", "R225_S5_GAP9_FILL0", None, "S5"),
        ("S3_REPLACEMENT", "R225_S3R_REPLACEMENT", None, "S3_REPLACEMENT"),
    ]
    for scenario, code, hooks, kind in specs:
        frame = overlay if kind is None else _scenario_frame(overlay, entry, kind)
        sleeve = run_round7(
            ROOT / "execution" / code,
            frame,
            {
                "code": code,
                "round": "22.5",
                "scope": "EXECUTION_SCENARIO",
                "scenario_name": scenario,
                "same_day_replacement": kind == "S3_REPLACEMENT",
                "top_changed": True,
            },
            resume=resume,
            validation_hooks=hooks,
            scenario=True,
            counts_as_alpha=False,
            already_overlayed=True,
        )
        rows.append(
            {
                "code": code,
                "scenario": scenario,
                "status": sleeve.get("status"),
                "sharpe": sleeve.get("sharpe"),
                "cagr": sleeve.get("cagr"),
                "nav_end": sleeve.get("nav_end"),
                "max_drawdown": sleeve.get("max_drawdown"),
                "completed_trades": sleeve.get("completed_trades"),
                "same_day_replacement": kind == "S3_REPLACEMENT",
                "counts_as_alpha": False,
            }
        )
    write_df(ROOT / "execution_stress.csv", pd.DataFrame(rows))
    return save_stage(ROOT, "execution", {"status": "DONE", "n": len(rows)})


def _hypothesis_results() -> list[dict[str, Any]]:
    nulls = load_json(ROOT / "null_recheck.json")
    c1 = load_json(ROOT / "c1_full_train_attribution.json")
    c4 = load_json(ROOT / "c4_full_train_attribution.json")
    lineage = load_json(ROOT / "lineage_nav_bridge.json")
    fs2 = load_json(ROOT / "fs2_oof.json")
    rolling = load_json(ROOT / "c4_rolling_semantics.json")
    c4_summary = c4.get("full_history_summary") or {}
    results = []
    mapping = {
        "H225_C1_FULL_TRAIN_INCREMENT": "FAIL" if c1.get("verdict") == "C1_NO_STABLE_INCREMENTAL_ALPHA" or not (c4_summary and False) else "FAIL",
        "H225_C4_ECONOMIC": "PASS" if c4_summary.get("economically_material") else "FAIL",
        "H225_ALLNAN_INVARIANT": "PASS" if nulls.get("allnan_nav_identical") else "FAIL",
        "H225_SHUFFLE_RECHECK": "PASS" if nulls.get("shuffle_matches_frozen") else "FAIL",
        "H225_PERM_NOT_ALPHA": "PASS" if nulls.get("permutations") and all(not row.get("counts_as_alpha", True) for row in nulls["permutations"]) else "FAIL",
        "H225_FS2_OOF_SCREEN": "PASS"
        if any((item or {}).get("pass") for item in (fs2.get("screens") or {}).values())
        or fs2.get("verdict") in {"FS2_OOF_SCREEN_PASS", "PREDICTIVE_NOT_ECONOMIC", "V6_RESEARCH_CANDIDATE"}
        else "FAIL",
        "H225_NAV_RECONCILED": "PASS" if lineage.get("status") == "RECONCILED" else "FAIL",
        "H225_C4_ROLLING_ROWWISE": "PASS" if rolling.get("rolling_is_row_based") and rolling.get("rewritten") is False else ("BLOCKED" if rolling.get("status") == "MISSING_ARTIFACT" else "FAIL"),
    }
    if c1.get("verdict") != "C1_NO_STABLE_INCREMENTAL_ALPHA" and (c1.get("full_history_summary") or {}).get("economically_material"):
        mapping["H225_C1_FULL_TRAIN_INCREMENT"] = "PASS"
    elif c1.get("verdict") == "C1_NO_STABLE_INCREMENTAL_ALPHA":
        mapping["H225_C1_FULL_TRAIN_INCREMENT"] = "FAIL"
    else:
        mapping["H225_C1_FULL_TRAIN_INCREMENT"] = "FAIL"
    for hyp in HYPOTHESES:
        results.append({"id": hyp["id"], "claim": hyp["claim"], "result": mapping[hyp["id"]], "artifact": hyp["id"]})
    return results


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("decision", resume)
    if cached and (ROOT / "FINAL_DECISION.md").is_file() and (ROOT / "ROUND22_5_RESEARCH_REPORT.md").is_file():
        return cached
    results = _hypothesis_results()
    c1 = load_json(ROOT / "c1_full_train_attribution.json")
    c4 = load_json(ROOT / "c4_full_train_attribution.json")
    fs2 = load_json(ROOT / "fs2_oof.json")
    c4_sum = c4.get("full_history_summary") or {}
    c4_decision = "WEAK_RESEARCH_SIGNAL_ONLY"
    if c4_sum.get("economically_material"):
        c4_decision = "PASS_ECONOMIC_GATE"
    elif c4_sum.get("delta_sharpe") is not None and float(c4_sum["delta_sharpe"]) <= 0:
        c4_decision = "C4_NO_POSITIVE_INCREMENT"
    decision = {
        "c1": "ARCHIVE_" + str(c1.get("verdict")),
        "c4": c4_decision,
        "fs2": fs2.get("verdict"),
        "promote": False,
        "live_ready": False,
        "forward": "FORWARD_OOS_NOT_AVAILABLE",
        "current_baseline_changed": False,
        "auto_promote": False,
        "next_round_baseline": "baseline_v5=R16_O01",
        "hypotheses": results,
    }
    if fs2.get("verdict") == "V6_RESEARCH_CANDIDATE":
        decision["promote_recommendation"] = "V6_RESEARCH_CANDIDATE"
    else:
        decision["promote_recommendation"] = "KEEP_BASELINE_V5"
    dump_json(ROOT / "decision.json", decision)
    write_df(ROOT / "hypotheses.csv", pd.DataFrame(results))
    _write_reports(decision)
    return save_stage(ROOT, "decision", {"status": "DONE", "decision": decision["promote_recommendation"]})


def _write_reports(decision: dict[str, Any]) -> None:
    c1 = load_json(ROOT / "c1_full_train_attribution.json")
    c4 = load_json(ROOT / "c4_full_train_attribution.json")
    nulls = load_json(ROOT / "null_recheck.json")
    lineage = load_json(ROOT / "lineage_nav_bridge.json")
    fs2 = load_json(ROOT / "fs2_oof.json")
    rolling = load_json(ROOT / "c4_rolling_semantics.json")
    freq = load_json(ROOT / "open_limit_frequency.json")
    lines = [
        "# Round22.5 研究报告",
        "",
        f"决策：**{decision['promote_recommendation']}**。不自动升 CURRENT。C1={decision['c1']}。C4={decision['c4']}。FS2={decision['fs2']}。",
        "",
        "本轮没有新因子、没有新的 C 公式、没有重调 XGB。经济门仍是 `R21_3_ECONOMIC_MATERIALITY_V1`。短窗 Sharpe 不与 v5 的 1.8756627913361097 混排。",
        "",
        "## 锁定锚点",
        "",
        "baseline_v5=R16_O01，2018-01-02～2026-07-24，Sharpe=1.8756627913361097，CAGR=0.5499812832273525，MaxDD=-0.24056327715394388，NAV=12750089.355724689，笔数=1107，资金=300000。",
        "",
        "## 事实表",
        "",
        "真实账本来自 Round21.3 到 Round24 的 summary.json。缺 Top1/Top2 的行记为 NOT_IDENTIFIABLE。见 `experiment_fact_table.parquet` 与 `multiple_testing_registry.csv`。",
        "",
        "## C1 / C4 覆盖与滚动",
        "",
        f"C4 公式未改：`{C4_FORMULA}`。按行滚动，不插入日历空档。n_valid_3={rolling.get('n_valid_3')}，n_valid_4={rolling.get('n_valid_4')}，n_valid_5={rolling.get('n_valid_5')}。",
        "",
        "## 空值、打乱、截面置换",
        "",
        f"ALLNAN 与 v5 净值相同：{nulls.get('allnan_nav_identical')}。SHUFFLE Sharpe={nulls.get('shuffle_sharpe')}，对齐冻结值 {SHUFFLE_SHARPE}：{nulls.get('shuffle_matches_frozen')}。",
        "日内截面置换种子 17/42/73 是诊断，不是 alpha。",
        "",
        "```",
        pd.DataFrame(nulls.get("permutations") or []).to_string(index=False),
        "```",
        "",
        "## C1 FULL_TRAIN",
        "",
        f"重算 PARTIAL 财富比={c1.get('partial_wealth_recomputed')}，FULL_TRAIN 财富比={c1.get('full_train_wealth_recomputed')}。历史引用 1.263687 与 0.806711 只作核对，不代替重算。裁决：{c1.get('verdict')}。没有新的 C1 组合。",
        "",
        "## C4 FULL_TRAIN",
        "",
        "2024、2025 分开报。2026 不是完整年。条件 IC、NDCG@1/@2、变更后的 Top1/Top2 成熟标签、块长 20 的 5000 次配对自助都在 `c4_full_train_attribution.json`。自助区间不是后验。",
        "",
        f"全历史摘要：{c4.get('full_history_summary')}",
        "",
        f"自助：{c4.get('bootstrap')}",
        "",
        "## 净值桥",
        "",
        f"状态={lineage.get('status')}。现金+市值=总资产：{lineage.get('cash_plus_market_value_equals_total_asset')}。费用已含在成交现金里，总额={((lineage.get('fees_already_inside_trade_cash') or {}).get('total'))}。残差/净值={lineage.get('residual_over_nav')}。终点净值同时包含未实现市值和复利后的总资产。",
        "",
        "## FS2",
        "",
        f"公式是经营现金流/平均资产，不是现金转换变化。裁决={fs2.get('verdict')}。PIT={fs2.get('pit')}。筛选={fs2.get('screens')}。没有 Remove/Replace，没有双头同时加，没有新参数。",
        "",
        "## 执行压力",
        "",
        "S0 复用 R223_E0。S1 +10bp 买滑点，S2 +20bp。S3 用真实涨停价把开盘封死记为未成交。S4 缺口≥7% 成交 50%。S5 缺口≥9% 成交 0%。S4/S5 是压力，不是板块规则。主敏感性没有当日替补。替补只在 S3_REPLACEMENT。",
        "",
        f"开盘封死频率：{freq}",
        "",
        "## 假设",
        "",
        "```",
        pd.DataFrame(decision["hypotheses"]).to_string(index=False),
        "```",
        "",
        "CURRENT 未改。AUTO_PROMOTE=false。没有独立 Forward，因此 LIVE_READY=false。",
        "",
    ]
    write_text(ROOT / "ROUND22_5_RESEARCH_REPORT.md", "\n".join(lines))
    final = [
        "# Round22.5 最终裁决",
        "",
        f"- C1：{decision['c1']}",
        f"- C4：{decision['c4']}",
        f"- FS2：{decision['fs2']}",
        f"- 晋级建议：{decision['promote_recommendation']}",
        "- CURRENT_BASELINE_CHANGED=false",
        "- AUTO_PROMOTE=false",
        "- LIVE_READY=false",
        "- 经济门没有放宽，执行压力不能代替经济门。",
        "",
    ]
    write_text(ROOT / "FINAL_DECISION.md", "\n".join(final))


def run_all(*, resume: bool = True, jobs: int = 2) -> dict[str, Any]:
    stages = [
        run_freeze,
        run_fact_table,
        run_coverage,
        run_null,
        run_attribution,
        run_lineage,
        run_fs2,
        run_execution,
        run_decision,
    ]
    out = {}
    for fn in stages:
        print(f"[22.5] {fn.__name__}", flush=True)
        out[fn.__name__] = fn(resume=resume, jobs=jobs)
    return out
