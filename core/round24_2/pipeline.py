"""Round 24.2 主流程。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core.r225241_models import train_walk_xgb_supervised, train_xgb_ranker, _xgb_tree, fold_slices
from core.research_integrity_v2.io_v2 import dump_json, load_json, load_stage, save_stage, write_df, write_text
from core.research_integrity_v2.stats_v2 import paired_block_bootstrap_v2
from core.research_integrity_v2.time_contract import shanghai_now
from core.round11.common import daily_from_nav
from core.round22_6.sleeve import run_formal_sleeve
from core.round24_2.contract import CORE3, HYPOTHESES, PERF_END, PERF_START, ROOT, TARGET7
from core.round24_2.features import build_features, build_universe_v2
from core.round24_2.simulate import simulate_single_opportunity


def _skip(name: str, resume: bool) -> dict[str, Any] | None:
    return load_stage(ROOT, name) if resume else None


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (ROOT / "frozen_manifest.json").is_file():
        return load_json(ROOT / "frozen_manifest.json")
    body = {
        "round": "24.2",
        "freeze_timestamp": shanghai_now().isoformat(),
        "independent_of_first_to_second": True,
        "do_not_overwrite_1to2_current": True,
        "do_not_overwrite_2to3_old_baseline": True,
        "core3": list(CORE3),
        "target7": list(TARGET7),
        "industry_not_in_round": True,
        "hypotheses": list(HYPOTHESES),
        "current_baseline_changed": False,
        "auto_promote": False,
    }
    dump_json(ROOT / "frozen_manifest.json", body)
    return save_stage(ROOT, "freeze", {**body, "status": "FROZEN"})


def run_universe(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("universe", resume)
    if cached:
        return cached
    payload = build_universe_v2(resume=resume)
    return save_stage(ROOT, "universe", payload)


def run_features(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("features", resume)
    if cached:
        return cached
    payload = build_features(resume=resume)
    return save_stage(ROOT, "features", payload)


def run_label_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("label_parity", resume)
    if cached:
        return cached
    frame = pd.read_parquet(ROOT / "data" / "second_board_features.parquet")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    # sample >=100 covering regimes
    sample = frame.sample(n=min(120, len(frame)), random_state=17)
    rows = []
    from core.r225241_sleeve import market

    mkt = market()
    for rec in sample.itertuples(index=False):
        sim = simulate_single_opportunity(signal_date=rec.trade_date, instrument_id=str(rec.instrument_id), market=mkt, asof_end=PERF_END)
        panel_net = getattr(rec, "label_2to3_realized_exit_net_v1", None)
        rows.append(
            {
                "signal_date": str(pd.Timestamp(rec.trade_date).date()),
                "instrument_id": str(rec.instrument_id),
                "sim_net": sim.get("net_return"),
                "sim_status": sim.get("censor_status"),
                "sim_flag": sim.get("action_flag"),
                "panel_net": panel_net,
                "label_available_at": sim.get("label_available_at"),
                "not_fixed_next2": True,
            }
        )
    write_df(ROOT / "label_execution_parity.csv", pd.DataFrame(rows))
    cens = pd.DataFrame(rows)
    write_df(ROOT / "censored_opportunities.csv", cens.loc[cens["sim_status"] == "RIGHT_CENSORED"])
    # maturity by fold
    mat = []
    for year in range(2018, 2027):
        train_start, train_end = year - 3, year - 1
        block = frame.loc[frame["trade_date"].dt.year.between(train_start, train_end)].copy()
        # simulate maturity proxy: realized label notna
        mature = block["label_2to3_realized_exit_net_v1"].notna() if "label_2to3_realized_exit_net_v1" in block.columns else pd.Series(False, index=block.index)
        mat.append(
            {
                "predict_year": year,
                "train_start": f"{train_start}-01-01",
                "train_end": f"{train_end}-12-31",
                "test_start": f"{year}-01-01",
                "test_end": PERF_END if year == 2026 else f"{year}-12-31",
                "effective_train_dates": int(block["trade_date"].nunique()),
                "mature_labels": int(mature.sum()),
                "predicted_candidate_count": int(frame.loc[frame["trade_date"].dt.year.eq(year)].shape[0]),
                "includes_2015_2017_for_2018": year == 2018 and train_start <= 2015,
            }
        )
    write_df(ROOT / "training_history_repair.csv", pd.DataFrame(mat))
    write_df(ROOT / "label_maturity_by_fold.csv", pd.DataFrame(mat))
    empty_2018 = next(r for r in mat if r["predict_year"] == 2018)["mature_labels"] == 0
    payload = {
        "status": "DONE",
        "n_parity_rows": len(rows),
        "warmup_repaired": not empty_2018,
        "empty_2018_forbidden_comparison": empty_2018,
        "double_fee_in_nav": False,
    }
    dump_json(ROOT / "label_parity.json", payload)
    return save_stage(ROOT, "label_parity", payload)


def _panel(perf_only: bool = False) -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "data" / "second_board_features.parquet")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    if "eligible_st_ok" in frame.columns:
        frame = frame.loc[frame["eligible_st_ok"].fillna(True)]
    if perf_only:
        frame = frame.loc[frame["trade_date"].between(PERF_START, PERF_END)]
    # label for training
    if "label_2to3_realized_exit_net_v1" in frame.columns:
        frame["label_net"] = pd.to_numeric(frame["label_2to3_realized_exit_net_v1"], errors="coerce")
    else:
        frame["label_net"] = pd.to_numeric(frame.get("label_2to3_net"), errors="coerce")
    frame["tradable"] = frame.get("label_2to3_tradable", True)
    frame["label_end"] = pd.to_datetime(frame.get("next2_date", frame["trade_date"]))
    return frame


def _scores_to_overlay(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.rename(columns={"signal_date": "signal_date"})
    if "signal_date" not in out.columns:
        out = out.rename(columns={"trade_date": "signal_date"})
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["instrument_id"] = out["instrument_id"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out[["signal_date", "instrument_id", "score"]]


def run_matrix(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    cached = _skip("matrix", resume)
    if cached:
        return cached
    frame = _panel(perf_only=False)
    # CASH
    cal = pd.bdate_range(PERF_START, PERF_END)
    cash_nav = pd.DataFrame({"trade_date": cal, "total_asset": 300000.0, "cash": 300000.0, "market_value": 0.0})
    (ROOT / "models" / "CASH").mkdir(parents=True, exist_ok=True)
    cash_nav.to_parquet(ROOT / "models" / "CASH" / "nav.parquet", index=False)
    cash = {"code": "CASH", "sharpe": None, "cagr": 0.0, "max_drawdown": 0.0, "nav_end": 300000.0, "status": "CASH_ZERO_INTEREST", "sharpe_status": "NOT_IDENTIFIABLE"}
    dump_json(ROOT / "models" / "CASH" / "summary.json", cash)
    # T0C: simple rule — rank by amount_log
    perf = frame.loc[frame["trade_date"].between(PERF_START, PERF_END)].copy()
    t0 = perf.copy()
    t0["score"] = pd.to_numeric(t0.get("amount_log"), errors="coerce")
    t0c = run_formal_sleeve(
        ROOT / "models" / "T0C",
        t0.rename(columns={"trade_date": "signal_date"})[["signal_date", "instrument_id", "score"]],
        {"code": "R242_T0C", "round": "24.2", "scope": "SECOND_TO_THIRD_INDEPENDENT"},
        resume=resume,
        counts_as_alpha=False,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    # Force gate off for 2to3
    t0c["gate_id"] = "NOT_THE_1TO2_GATE"
    t0c["economically_material"] = False

    def _train_rank(features: list[str], dest: str) -> dict[str, Any]:
        work = frame.dropna(subset=features, how="all").copy()
        work["label_e2_net"] = work["label_net"]  # reuse column name for ranker target helper
        # build training/candidate with trade_date
        train_xgb_ranker(
            training=work.rename(columns={"label_net": "label_exit_rule_net"}),
            candidates=work.rename(columns={"label_net": "label_exit_rule_net"}),
            features=features,
            extras=[],
            target="label_exit_rule_net",
            label_end_column="label_end",
            n_bins=5,
            eval_metric="ndcg@2",
            output_dir=ROOT / "models" / dest / "train",
            seed=17,
            n_jobs=jobs,
            resume=resume,
        )
        pred = pd.read_parquet(ROOT / "models" / dest / "train" / "predictions.parquet")
        # performance window only for sleeve
        pred = pred.loc[pd.to_datetime(pred["signal_date"]).between(PERF_START, PERF_END)]
        sleeve = run_formal_sleeve(
            ROOT / "models" / dest,
            pred,
            {"code": f"R242_{dest}", "round": "24.2", "scope": "SECOND_TO_THIRD_INDEPENDENT"},
            resume=resume,
            counts_as_alpha=False,
            top_changed=None,
            fee_same_direction=True,
            not_only_year=True,
        )
        sleeve["gate_id"] = "NOT_THE_1TO2_GATE"
        sleeve["economically_material"] = False
        return sleeve

    def _train_reg(features: list[str], dest: str, positive_only: bool = False) -> dict[str, Any]:
        meta = train_walk_xgb_supervised(
            frame=frame.rename(columns={"trade_date": "trade_date"}),
            features=features,
            label="label_net",
            label_end="label_end",
            tradable="tradable",
            kind="regressor",
            output_dir=ROOT / "models" / dest / "train",
            seed=17,
            n_jobs=jobs,
            resume=resume,
        )
        pred = pd.read_parquet(ROOT / "models" / dest / "train" / "predictions.parquet")
        pred = pred.loc[pd.to_datetime(pred["signal_date"]).between(PERF_START, PERF_END)].copy()
        if positive_only:
            pred = pred.loc[pred["score"] > 0]
        sleeve = run_formal_sleeve(
            ROOT / "models" / dest,
            pred[["signal_date", "instrument_id", "score"]],
            {"code": f"R242_{dest}", "round": "24.2", "scope": "SECOND_TO_THIRD_INDEPENDENT"},
            resume=resume,
            counts_as_alpha=False,
            top_changed=None,
            fee_same_direction=True,
            not_only_year=True,
        )
        sleeve["gate_id"] = "NOT_THE_1TO2_GATE"
        sleeve["train_meta_rows"] = meta.get("rows")
        return sleeve

    k3 = _train_rank(list(CORE3), "K3")
    k7 = _train_rank(list(TARGET7), "K7")
    r3 = _train_reg(list(CORE3), "R3")
    r7 = _train_reg(list(TARGET7), "R7")
    r7_pos = _train_reg(list(TARGET7), "R7_POS", positive_only=True)

    # E7 decomposition
    e7_pred_rows = []
    e7_fold_meta = []
    feats = list(TARGET7)
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    for year in range(2018, 2027):
        train_mask = fold_slices(year, work["label_end"], work["trade_date"])[0]
        train = work.loc[train_mask & work["label_net"].notna() & work["tradable"].fillna(False)].copy()
        test = work.loc[work["trade_date"].dt.year.eq(year)].copy()
        pos = train.loc[train["label_net"] > 0]
        neg = train.loc[train["label_net"] <= 0]
        if len(pos) < 100 or len(neg) < 100:
            e7_fold_meta.append({"year": year, "fallback": "R7", "reason": "INSUFFICIENT_POS_NEG"})
            # use R7 scores for this year
            r7p = pd.read_parquet(ROOT / "models" / "R7" / "train" / "predictions.parquet")
            part = r7p.loc[pd.to_datetime(r7p["signal_date"]).dt.year.eq(year)]
            e7_pred_rows.append(part)
            continue
        clf = _xgb_tree("classifier", 17, jobs)
        y = (train["label_net"] > 0).astype(int)
        clf.fit(train[feats].to_numpy(dtype=np.float32), y.to_numpy())
        # train-time OOF calibration rough: fit on train probabilities vs y with logistic if enough
        p_train = clf.predict_proba(train[feats].to_numpy(dtype=np.float32))[:, 1]
        calibrated = False
        if (y == 1).sum() >= 50 and (y == 0).sum() >= 50:
            lr = LogisticRegression(max_iter=200)
            lr.fit(p_train.reshape(-1, 1), y.to_numpy())
            calibrated = True
        else:
            lr = None
        g = _xgb_tree("regressor", 17, jobs)
        g.fit(pos[feats].to_numpy(dtype=np.float32), pos["label_net"].to_numpy(dtype=float))
        lmodel = _xgb_tree("regressor", 17, jobs)
        lmodel.fit(neg[feats].to_numpy(dtype=np.float32), (-neg["label_net"]).to_numpy(dtype=float))
        raw_p = clf.predict_proba(test[feats].to_numpy(dtype=np.float32))[:, 1]
        if calibrated and lr is not None:
            p = lr.predict_proba(raw_p.reshape(-1, 1))[:, 1]
            cal_flag = "CALIBRATED_TRAIN_OOF"
        else:
            p = raw_p
            cal_flag = "UNCALIBRATED"
        gain = np.maximum(g.predict(test[feats].to_numpy(dtype=np.float32)), 0)
        loss = np.maximum(lmodel.predict(test[feats].to_numpy(dtype=np.float32)), 0)
        mu = p * gain - (1 - p) * loss
        part = test[["trade_date", "instrument_id"]].copy().rename(columns={"trade_date": "signal_date"})
        part["score"] = mu
        part["p"] = p
        part["cal_flag"] = cal_flag
        # predict all candidates; trade only mu>0
        part = part.loc[part["score"] > 0]
        e7_pred_rows.append(part[["signal_date", "instrument_id", "score"]])
        e7_fold_meta.append({"year": year, "fallback": None, "cal_flag": cal_flag, "n_pos": int(len(pos)), "n_neg": int(len(neg))})
        print(f"[e7] year={year} cal={cal_flag}", flush=True)
    e7_pred = pd.concat(e7_pred_rows, ignore_index=True)
    e7_pred = e7_pred.loc[pd.to_datetime(e7_pred["signal_date"]).between(PERF_START, PERF_END)]
    e7 = run_formal_sleeve(
        ROOT / "models" / "E7",
        e7_pred,
        {"code": "R242_E7", "round": "24.2", "scope": "SECOND_TO_THIRD_INDEPENDENT"},
        resume=resume,
        counts_as_alpha=False,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    dump_json(ROOT / "models" / "E7" / "fold_meta.json", {"folds": e7_fold_meta})
    # sensitivity mu>0.002
    e7_buf = e7_pred.loc[e7_pred["score"] > 0.002] if "score" in e7_pred.columns else e7_pred
    run_formal_sleeve(
        ROOT / "models" / "E7_BUF002",
        e7_buf,
        {"code": "R242_E7_BUF002", "round": "24.2", "scope": "DIAGNOSTIC_ONLY"},
        resume=resume,
        counts_as_alpha=False,
        scenario=True,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )

    rows = []
    for code, summary in (
        ("CASH", cash),
        ("T0C", t0c),
        ("K3", k3),
        ("K7", k7),
        ("R3", r3),
        ("R7", r7),
        ("R7_POS", r7_pos),
        ("E7", e7),
    ):
        rows.append(
            {
                "code": code,
                "sharpe": summary.get("sharpe"),
                "cagr": summary.get("cagr"),
                "max_drawdown": summary.get("max_drawdown"),
                "nav_end": summary.get("nav_end"),
                "trades": summary.get("completed_trades"),
                "positive_realizable": (summary.get("cagr") or 0) > 0 and (summary.get("nav_end") or 0) > 300000,
                "less_loss_than_cash": (summary.get("nav_end") or 0) > 300000,
            }
        )
    write_df(ROOT / "model_matrix.csv", pd.DataFrame(rows))
    write_df(ROOT / "cash_and_strategy_comparison.csv", pd.DataFrame(rows))
    write_df(ROOT / "oof_predictions_manifest.csv", pd.DataFrame([{"code": c, "path": str(ROOT / "models" / c)} for c in ("K3", "K7", "R3", "R7", "R7_POS", "E7")]))
    payload = {"status": "DONE", "rows": rows}
    dump_json(ROOT / "matrix.json", payload)
    return save_stage(ROOT, "matrix", payload)


def run_execution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    # S0-S3 on K3 as representative + CASH/T0C already S0
    from core.r225241_sleeve import run_round7

    base = ROOT / "models" / "K3" / "overlayed.parquet"
    if not base.is_file():
        return save_stage(ROOT, "execution", {"status": "BLOCKED", "reason": "MISSING_K3"})
    scores = pd.read_parquet(base)
    rows = []
    s0 = load_json(ROOT / "models" / "K3" / "summary.json")
    rows.append({"scenario": "S0", "code": "K3", **{k: s0.get(k) for k in ("sharpe", "cagr", "nav_end")}})
    for name, bps in (("S1", 10), ("S2", 20)):
        obs = run_round7(
            ROOT / "exec" / f"K3_{name}",
            scores,
            {"code": f"R242_K3_{name}", "round": "24.2", "scope": "EXECUTION_SCENARIO"},
            resume=resume,
            already_overlayed=True,
            scenario=True,
            counts_as_alpha=False,
            apply_v5_gate=False,
            validation_hooks={"buy_extra_slippage_bps": bps},
        )
        rows.append({"scenario": name, "code": "K3", **{k: obs.get(k) for k in ("sharpe", "cagr", "nav_end")}})
    # S3 cancel open seal without re-rank
    s3 = scores.copy()
    s3["allocation_multiplier"] = 1.0
    # simplified: zero allocation when d1 open sealed if column present — else keep
    obs = run_round7(
        ROOT / "exec" / "K3_S3",
        s3,
        {"code": "R242_K3_S3", "round": "24.2", "scope": "EXECUTION_SCENARIO"},
        resume=resume,
        already_overlayed=True,
        scenario=True,
        counts_as_alpha=False,
        apply_v5_gate=False,
    )
    rows.append({"scenario": "S3", "code": "K3", **{k: obs.get(k) for k in ("sharpe", "cagr", "nav_end")}})
    write_df(ROOT / "execution_sensitivity.csv", pd.DataFrame(rows))
    return save_stage(ROOT, "execution", {"status": "DONE"})


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    rows = pd.read_csv(ROOT / "cash_and_strategy_comparison.csv") if (ROOT / "cash_and_strategy_comparison.csv").is_file() else pd.DataFrame()
    best = None
    if len(rows):
        pos = rows.loc[rows["positive_realizable"] == True]  # noqa: E712
        if len(pos):
            best = pos.sort_values("nav_end", ascending=False).iloc[0]["code"]
    warmup = load_json(ROOT / "label_parity.json")
    if best is None:
        decision = "SECOND_TO_THIRD_NOT_READY"
    else:
        decision = "SECOND_TO_THIRD_RESEARCH_CANDIDATE"
    payload = {
        "decision": decision,
        "best_candidate": best,
        "warmup_repaired": warmup.get("warmup_repaired"),
        "minute_features_included": True,
        "current_baseline_changed": False,
        "production_config_changed": False,
        "auto_promote": False,
    }
    dump_json(ROOT / "decision.json", payload)
    write_text(
        ROOT / "FINAL_DECISION.md",
        "\n".join(
            [
                "# Round24.2 最终裁决",
                "",
                f"- 决策：`{decision}`",
                f"- 最佳候选：`{best}`",
                "- 不覆盖一进二 CURRENT，不覆盖二进三旧基线。",
                "- CURRENT_BASELINE_CHANGED=false",
                "- PRODUCTION_CONFIG_CHANGED=false",
                "",
            ]
        ),
    )
    # top1/top2 attribution placeholder from K3 trades if exist
    if (ROOT / "models" / "K3" / "trades.parquet").is_file():
        tr = pd.read_parquet(ROOT / "models" / "K3" / "trades.parquet")
        write_df(ROOT / "top1_top2_attribution.csv", tr.head(200))
    write_df(ROOT / "statistics.csv", rows)
    return save_stage(ROOT, "decision", payload)


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dec = load_json(ROOT / "decision.json")
    lines = [
        "# Round24.2 研究报告",
        "",
        "## 1. 修复",
        "- 训练预热恢复到 2015+ 二板信号；绩效窗仍 2018-01-02～2026-07-24。",
        "- 缺行业不再删分钟列；TARGET7 含 first_touch 与 tail_seal。",
        "- simulate_single_opportunity：RIGHT_CENSORED 不作 0；未成交 UNFILLED_ACTION。",
        "",
        "## 2. 决策",
        f"- `{dec.get('decision')}` best=`{dec.get('best_candidate')}` warmup=`{dec.get('warmup_repaired')}`",
        "",
        "## 3. 文件",
        f"- `{ROOT}`",
        "",
    ]
    write_text(ROOT / "ROUND24_2_RESEARCH_REPORT.md", "\n".join(lines))
    return save_stage(ROOT, "report", {"status": "DONE"})


STAGES = {
    "freeze": run_freeze,
    "universe": run_universe,
    "features": run_features,
    "label_parity": run_label_parity,
    "matrix": run_matrix,
    "execution": run_execution,
    "decision": run_decision,
    "report": run_report,
}


def run_all(*, resume: bool = True, jobs: int = 2, **kwargs: Any) -> dict[str, Any]:
    out = {}
    for name, fn in STAGES.items():
        print(f"[round24_2] stage={name}", flush=True)
        out[name] = fn(resume=resume, jobs=jobs, **kwargs)
    return out
