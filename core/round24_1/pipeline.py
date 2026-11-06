"""Round 24.1。先交易与否，再排序。不微调一进二模型。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.r225241_io import dump_json, load_json, load_stage, save_stage, write_df, write_text
from core.r225241_models import train_walk_xgb_supervised
from core.r225241_sleeve import run_round7
from core.r225241_stats import daily_spearman, paired_block_bootstrap
from core.round11.common import daily_from_nav

from .contract import (
    ABSTAIN,
    GATE,
    HYPOTHESES,
    MIN_INNER_DAYS,
    MIN_NAMES_FOR_RANKIC,
    ROOT,
    ROUND24,
)
from .features import audit_board_state, build_features


def _skip(name: str, resume: bool) -> dict[str, Any] | None:
    return load_stage(ROOT, name) if resume else None


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (ROOT / "frozen_manifest.json").is_file():
        return load_json(ROOT / "frozen_manifest.json")
    body = {
        "round": "24.1",
        "independent_of_first_to_second": True,
        "do_not_finetune_1to2": True,
        "do_not_replace_baseline_2to3_v0": True,
        "compare_cagr_to_first_board": False,
        "abstention_thresholds": list(ABSTAIN),
        "default_threshold_if_inner_days_below_120": 0.50,
        "threshold_selected_on": "inner mean realized net of selected tradable names",
        "never_select_threshold_on_outer_sharpe": True,
        "rankic_min_names": MIN_NAMES_FOR_RANKIC,
        "gate": GATE,
        "label_cost_not_recharged_in_nav": True,
        "current_baseline_changed": False,
        "hypotheses": list(HYPOTHESES),
    }
    dump_json(ROOT / "frozen_manifest.json", body)
    write_text(ROOT / "00_protocol.md", "# Round24.1 协议\n\nD 日二板收盘后决策，D+1 尝试三板。不覆盖 baseline_2to3_v0，也不和一进二 CAGR 比较。\n")
    return save_stage(ROOT, "freeze", {**body, "status": "FROZEN"})


def run_build(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("build", resume)
    if cached:
        return cached
    audit = audit_board_state()
    dump_json(ROOT / "board_audit.json", audit)
    if audit.get("rebuild_required"):
        payload = {"status": "BLOCKED", "reason": "BOARD_STATE_MISMATCH", "audit": audit, "feature_set": "NOT_BUILT"}
        dump_json(ROOT / "data" / "feature_status.json", payload)
        return save_stage(ROOT, "build", payload)
    features = build_features(resume=resume)
    payload = {"status": "DONE", "audit": audit, **features}
    return save_stage(ROOT, "build", payload)


def _panel() -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "data" / "second_board_features.parquet")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["next2_date"] = pd.to_datetime(frame["next2_date"])
    return frame


def _rankic(frame: pd.DataFrame, score: str, label: str) -> dict[str, Any]:
    vals = []
    thin = 0
    for _, block in frame.groupby("trade_date", sort=False):
        names = int(block["instrument_id"].astype(str).nunique()) if "instrument_id" in block.columns else int(len(block))
        if names < MIN_NAMES_FOR_RANKIC:
            thin += 1
            continue
        left = pd.to_numeric(block[score], errors="coerce")
        right = pd.to_numeric(block[label], errors="coerce")
        ok = left.notna() & right.notna()
        if int(ok.sum()) < MIN_NAMES_FOR_RANKIC or left[ok].nunique() < 2 or right[ok].nunique() < 2:
            continue
        vals.append(float(left[ok].corr(right[ok], method="spearman")))
    if not vals:
        return {"status": "NOT_IDENTIFIABLE", "thin_days": thin, "rankic_days": 0}
    return {"status": "OK", "rankic_mean": float(np.mean(vals)), "rankic_days": len(vals), "thin_days": thin}


def _inner_threshold(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    from core.r225241_models import _xgb_tree

    chosen = {}
    reports = []
    for year in range(2018, 2027):
        inner_start = pd.Timestamp(year - 1, 7, 1)
        inner_end = pd.Timestamp(year - 1, 12, 31)
        train = frame.loc[frame["trade_date"].between(pd.Timestamp(year - 3, 1, 1), inner_start - pd.Timedelta(days=1))]
        train = train.loc[train["label_2to3_tradable"].fillna(False) & train["label_2to3_realized_exit_net_v1"].notna() & train["next2_date"].le(inner_start)]
        inner = frame.loc[frame["trade_date"].between(inner_start, inner_end)]
        n_days = int(inner["trade_date"].nunique())
        if n_days < MIN_INNER_DAYS or len(train) < 50:
            chosen[year] = 0.50
            reports.append({"year": year, "threshold": 0.50, "reason": "DEFAULT_0.50", "n_days": n_days})
            continue
        model = _xgb_tree("classifier", 17, 2)
        y = (pd.to_numeric(train["label_2to3_realized_exit_net_v1"], errors="coerce") > 0).astype(int)
        model.fit(train[features].to_numpy(dtype=np.float32), y.to_numpy())
        scored = inner.copy()
        scored["p"] = model.predict_proba(scored[features].to_numpy(dtype=np.float32))[:, 1]
        best_thr, best_u = 0.50, -np.inf
        detail = []
        for thr in (*ABSTAIN, None):
            part = scored if thr is None else scored.loc[scored["p"] >= thr]
            part = part.loc[part["label_2to3_tradable"].fillna(False) & part["label_2to3_realized_exit_net_v1"].notna() & part["next2_date"].le(inner_end)]
            util = float(pd.to_numeric(part["label_2to3_realized_exit_net_v1"], errors="coerce").mean()) if len(part) else None
            detail.append(
                {
                    "threshold": "ALWAYS_TRADE_ELIGIBLE" if thr is None else thr,
                    "mean_realized_net": util,
                    "n": int(len(part)),
                    "day_share": float(part["trade_date"].nunique() / n_days) if n_days else None,
                }
            )
            if util is not None and util > best_u:
                best_u, best_thr = util, thr
        chosen[year] = "ALWAYS" if best_thr is None else float(best_thr)
        reports.append({"year": year, "threshold": chosen[year], "reason": "INNER_MEAN_NET", "grid": detail, "n_days": n_days})
    return {"by_year": {str(k): v for k, v in chosen.items()}, "reports": reports}


def _apply_threshold(pred: pd.DataFrame, thresholds: dict[str, Any], score_col: str) -> pd.DataFrame:
    work = pred.copy()
    work["year"] = work["signal_date"].dt.year
    parts = []
    for year, block in work.groupby("year"):
        raw = thresholds.get(str(int(year)), 0.50)
        if raw == "ALWAYS":
            kept = block.copy()
        else:
            kept = block.loc[pd.to_numeric(block["p"], errors="coerce") >= float(raw)].copy()
        kept["score"] = kept[score_col]
        parts.append(kept[["signal_date", "instrument_id", "score"]])
    if not parts:
        return work.iloc[0:0][["signal_date", "instrument_id"]].assign(score=np.nan)
    return pd.concat(parts, ignore_index=True)


def run_models(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    cached = _skip("models", resume)
    if cached:
        return cached
    if not (ROOT / "data" / "second_board_features.parquet").is_file():
        return save_stage(ROOT, "models", {"status": "BLOCKED", "reason": "MISSING_FEATURES"})
    status = load_json(ROOT / "data" / "feature_status.json")
    features = list(status["features"])
    frame = _panel()
    frame["tradable_flag"] = frame["label_2to3_tradable"].fillna(False).astype(bool)
    cls = train_walk_xgb_supervised(
        frame=frame,
        features=features,
        label="label_2to3_realized_exit_net_v1",
        label_end="next2_date",
        tradable="tradable_flag",
        kind="classifier",
        output_dir=ROOT / "models" / "classifier",
        n_jobs=jobs,
        resume=resume,
    )
    reg = train_walk_xgb_supervised(
        frame=frame,
        features=features,
        label="label_2to3_realized_exit_net_v1",
        label_end="next2_date",
        tradable="tradable_flag",
        kind="regressor",
        output_dir=ROOT / "models" / "regressor",
        n_jobs=jobs,
        resume=resume,
    )
    thresholds = _inner_threshold(frame, features)
    dump_json(ROOT / "abstention.json", thresholds)
    pred_c = pd.read_parquet(ROOT / "models" / "classifier" / "predictions.parquet")
    pred_r = pd.read_parquet(ROOT / "models" / "regressor" / "predictions.parquet")
    pred_c["signal_date"] = pd.to_datetime(pred_c["signal_date"])
    pred_r["signal_date"] = pd.to_datetime(pred_r["signal_date"])
    pred_c = pred_c.rename(columns={"score": "p"})
    merged = pred_c.merge(pred_r[["signal_date", "instrument_id", "score"]].rename(columns={"score": "reg"}), on=["signal_date", "instrument_id"], how="left")
    a1 = pred_c.rename(columns={"p": "score"})[["signal_date", "instrument_id", "score"]]
    a2 = pred_r[["signal_date", "instrument_id", "score"]]
    a3 = _apply_threshold(merged, thresholds["by_year"], "p")
    a4 = _apply_threshold(merged, thresholds["by_year"], "reg")
    rankic = _rankic(frame.merge(a1, left_on=["trade_date", "instrument_id"], right_on=["signal_date", "instrument_id"], how="left"), "score", "label_2to3_realized_exit_net_v1")
    controls = _control_rows()
    sleeves = list(controls)
    for code, scores in (
        ("A1_P_RANK", a1),
        ("A2_REGRESSOR", a2),
        ("A3_ABSTAIN_P", a3),
        ("A4_ABSTAIN_REG", a4),
    ):
        if scores.empty or scores["score"].notna().sum() == 0:
            sleeve = {"code": code, "status": "NO_ORDERS", "sharpe": None, "cagr": None, "max_drawdown": None, "nav_end": None, "new_backtest": False}
            dump_json(ROOT / "candidates" / code / "summary.json", sleeve)
        else:
            sleeve = run_round7(
                ROOT / "candidates" / code,
                scores,
                {"code": code, "round": "24.1", "scope": "SECOND_TO_THIRD_INDEPENDENT", "compare_to": "baseline_2to3_v0", "feature_set": status.get("feature_set")},
                resume=resume,
                apply_v5_gate=False,
                counts_as_alpha=True,
                already_overlayed=True,
            )
        sleeves.append(sleeve)
    payload = {
        "status": "DONE",
        "classifier": {k: cls.get(k) for k in ("rows", "score_is_fill_probability", "untradable_coded_as_loss")},
        "regressor": {k: reg.get(k) for k in ("rows", "immature_coded_as_loss")},
        "feature_set": status.get("feature_set"),
        "features": features,
        "rankic": rankic,
        "sleeves": sleeves,
        "seal_rate": float(frame["label_2to3_seal"].mean()),
        "tradable_rate": float(frame["label_2to3_tradable"].mean()),
        "open_limit_frequency": float(frame["open_limit_up_entry"].mean()) if "open_limit_up_entry" in frame.columns else None,
        "label_fee_double_charged_in_nav": False,
    }
    dump_json(ROOT / "models.json", payload)
    write_df(ROOT / "sleeve_metrics.csv", pd.DataFrame(sleeves))
    return save_stage(ROOT, "models", {"status": "DONE", "n_sleeves": len(sleeves), "feature_set": status.get("feature_set")})


def _control_rows() -> list[dict[str, Any]]:
    rows = []
    for code, path in (
        ("T0_REUSED", ROUND24 / "candidates" / "baseline_2to3_v0" / "summary.json"),
        ("T2_REUSED", ROUND24 / "candidates" / "T2_RETRAIN" / "summary.json"),
    ):
        payload = load_json(path)
        if not payload:
            rows.append({"code": code, "status": "MISSING_ARTIFACT"})
            continue
        rows.append(
            {
                "code": code,
                "status": "REUSED_ROUND24_ARTIFACT",
                "sharpe": payload.get("sharpe"),
                "cagr": payload.get("cagr"),
                "max_drawdown": payload.get("max_drawdown"),
                "nav_end": payload.get("nav_end"),
                "completed_trades": payload.get("completed_trades"),
                "source": str(path),
                "new_backtest": False,
            }
        )
    return rows


def run_execution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("execution", resume)
    if cached:
        return cached
    from core.round22_5.pipeline import _entry_table, _scenario_frame

    path = ROOT / "candidates" / "A4_ABSTAIN_REG"
    if not (path / "overlayed.parquet").is_file():
        return save_stage(ROOT, "execution", {"status": "BLOCKED", "reason": "MISSING_A4"})
    overlay = pd.read_parquet(path / "overlayed.parquet")
    overlay["signal_date"] = pd.to_datetime(overlay["signal_date"])
    entry = _entry_table(overlay) if len(overlay) else pd.DataFrame()
    base = load_json(path / "summary.json")
    rows = [{"code": "A4_ABSTAIN_REG", "scenario": "S0", "sharpe": base.get("sharpe"), "cagr": base.get("cagr"), "max_drawdown": base.get("max_drawdown"), "nav_end": base.get("nav_end")}]
    if len(overlay) == 0:
        rows.append({"status": "NO_ORDERS"})
    else:
        for scenario, suffix, hooks, kind in (
            ("S1", "SLIP10", {"buy_extra_slippage_bps": 10}, None),
            ("S2", "SLIP20", {"buy_extra_slippage_bps": 20}, None),
            ("S3", "OPENLIM", None, "S3"),
            ("S4", "GAP7", None, "S4"),
            ("S5", "GAP9", None, "S5"),
        ):
            frame = overlay if kind is None else _scenario_frame(overlay, entry, kind)
            ran = run_round7(
                ROOT / "execution" / suffix,
                frame,
                {"code": f"A4_{suffix}", "round": "24.1", "scope": "EXECUTION_SCENARIO", "compare_to": "baseline_2to3_v0"},
                resume=resume,
                validation_hooks=hooks,
                scenario=True,
                counts_as_alpha=False,
                apply_v5_gate=False,
                already_overlayed=True,
            )
            rows.append({"code": ran.get("code"), "scenario": scenario, "sharpe": ran.get("sharpe"), "cagr": ran.get("cagr"), "max_drawdown": ran.get("max_drawdown"), "nav_end": ran.get("nav_end")})
    write_df(ROOT / "execution_stress.csv", pd.DataFrame(rows))
    t0 = pd.read_parquet(ROUND24 / "candidates" / "baseline_2to3_v0" / "nav.parquet")
    t2 = pd.read_parquet(ROUND24 / "candidates" / "T2_RETRAIN" / "nav.parquet")
    a4 = pd.read_parquet(path / "nav.parquet")
    boot = {
        "vs_t0": paired_block_bootstrap(daily_from_nav(a4), daily_from_nav(t0)),
        "vs_t2": paired_block_bootstrap(daily_from_nav(a4), daily_from_nav(t2)),
        "is_posterior": False,
    }
    dump_json(ROOT / "paired_bootstrap.json", boot)
    return save_stage(ROOT, "execution", {"status": "DONE", "n": len(rows)})


def _gate(row: dict[str, Any], boot: dict[str, Any], exec_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sharpe = row.get("sharpe")
    cagr = row.get("cagr")
    dd = row.get("max_drawdown")
    level = sharpe is not None and cagr is not None and dd is not None and float(cagr) > GATE["cagr_min"] and float(sharpe) > GATE["sharpe_min"] and float(dd) > GATE["maxdd_floor"]
    inc_t0 = (boot.get("vs_t0") or {}).get("mean_excess")
    inc_t2 = (boot.get("vs_t2") or {}).get("mean_excess")
    increment = inc_t0 is not None and inc_t2 is not None and float(inc_t0) > 0 and float(inc_t2) > 0
    s3 = next((item for item in exec_rows if item.get("scenario") == "S3"), {})
    s5 = next((item for item in exec_rows if item.get("scenario") == "S5"), {})
    catastrophic = False
    for item in (s3, s5):
        if item.get("max_drawdown") is not None and float(item["max_drawdown"]) <= GATE["catastrophic_maxdd"]:
            catastrophic = True
        if item.get("cagr") is not None and float(item["cagr"]) < GATE["catastrophic_cagr"]:
            catastrophic = True
    s0_cagr = row.get("cagr")
    s3_cagr = s3.get("cagr")
    execution_dependent = s0_cagr is not None and s3_cagr is not None and float(s0_cagr) > 0 and float(s3_cagr) <= 0
    fee_robust = True
    s1 = next((item for item in exec_rows if item.get("scenario") == "S1"), {})
    if s1.get("cagr") is not None and float(s1["cagr"]) <= 0 and s0_cagr is not None and float(s0_cagr) > 0:
        fee_robust = False
    passed = bool(level and increment and fee_robust and not catastrophic and not execution_dependent)
    return {
        "level": bool(level),
        "paired_increment": bool(increment),
        "fee_robust": fee_robust,
        "catastrophic": catastrophic,
        "execution_dependent": execution_dependent,
        "passed": passed,
    }


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    models = load_json(ROOT / "models.json")
    boot = load_json(ROOT / "paired_bootstrap.json")
    exec_rows = pd.read_csv(ROOT / "execution_stress.csv").to_dict(orient="records") if (ROOT / "execution_stress.csv").is_file() else []
    sleeves = models.get("sleeves") or []
    a4 = next((row for row in sleeves if row.get("code") == "A4_ABSTAIN_REG"), {})
    gate = _gate(a4, boot, exec_rows)
    others = []
    for row in sleeves:
        if row.get("code") in {"A1_P_RANK", "A2_REGRESSOR", "A3_ABSTAIN_P"} and row.get("sharpe") is not None:
            others.append({"code": row.get("code"), "sharpe": row.get("sharpe"), "cagr": row.get("cagr"), "max_drawdown": row.get("max_drawdown"), "promoted_by_outer_sharpe": False})
    final = "SECOND_TO_THIRD_RESEARCH_CANDIDATE" if gate["passed"] else "SECOND_TO_THIRD_RESEARCH_ONLY"
    if gate["execution_dependent"]:
        final = "EXECUTION_DEPENDENT"
    hypotheses = [
        {"id": "H241_BOARD", "result": "PASS" if load_json(ROOT / "board_audit.json").get("board_ok") else "FAIL"},
        {"id": "H241_NO_DOUBLE_FEE", "result": "PASS" if models.get("label_fee_double_charged_in_nav") is False else "FAIL"},
        {"id": "H241_FEATURES", "result": "PASS" if models.get("feature_set") in {"FULL", "REDUCED"} else "BLOCKED"},
        {"id": "H241_STAGE", "result": "PASS" if gate["passed"] else "FAIL"},
    ]
    decision = {
        "final": final,
        "gate": gate,
        "feature_set": models.get("feature_set"),
        "baseline_2to3_replaced": False,
        "compare_to_1to2_cagr": False,
        "live_ready": False,
        "current_baseline_changed": False,
        "hypotheses": hypotheses,
        "other_arms_not_ranked_by_sharpe": others,
    }
    dump_json(ROOT / "decision.json", decision)
    write_df(ROOT / "hypotheses.csv", pd.DataFrame(hypotheses))
    t0 = next((row for row in sleeves if row.get("code") == "T0_REUSED"), {})
    t2 = next((row for row in sleeves if row.get("code") == "T2_REUSED"), {})
    write_text(
        ROOT / "ROUND24_1_RESEARCH_REPORT.md",
        "\n".join(
            [
                "# Round24.1 二进三研究报告",
                "",
                f"决策：**{final}**。这是独立策略线。不替换 baseline_2to3_v0，也不和一进二的 CAGR 或 Sharpe 比高低。",
                "",
                f"板块审计：{load_json(ROOT / 'board_audit.json')}",
                "",
                f"特征集={models.get('feature_set')}，列={models.get('features')}。缺失没有填 0，FULL 和 REDUCED 没有混在同一张结果里。",
                "",
                "标签 `label_2to3_net` 仍是诊断：close(D+2)/open(D+1)-1-0.001。组合净值走 ROUND7_FLAT_5BP，执行器不读取标签，因此训练成本没有在净值里再扣一次。",
                "模型训练用的是模型无关的 `label_2to3_realized_exit_net_v1`：可交易才有标签，不可交易或未成熟不是 0。卖出沿用涨停顺延，不是事后挑最赚钱的卖点。",
                "",
                f"T0 产物 Sharpe={t0.get('sharpe')} CAGR={t0.get('cagr')} MaxDD={t0.get('max_drawdown')}。",
                f"T2 产物 Sharpe={t2.get('sharpe')} CAGR={t2.get('cagr')} MaxDD={t2.get('max_drawdown')}。",
                "上面两个数字来自既有产物文件。若和口头记忆不一致，以文件为准。",
                "",
                f"第三板封板率={models.get('seal_rate')}，可交易比例={models.get('tradable_rate')}，开盘封死频率={models.get('open_limit_frequency')}。这三件事分开报。",
                f"RankIC：{models.get('rankic')}。不足 5 个名字的日子不做截面 RankIC。",
                "",
                f"A4 门：{gate}",
                "",
                "分类器输出不是成交概率。没有订单簿，不能把 P 说成排队成交概率。没有 D+1 分钟队列时，开盘封死压力只是压力，不是核验过的排队。",
                "",
                "```",
                pd.DataFrame(hypotheses).to_string(index=False),
                "```",
                "",
            ]
        ),
    )
    return save_stage(ROOT, "decision", {"status": "DONE", "final": final})


def run_all(*, resume: bool = True, jobs: int = 2) -> dict[str, Any]:
    out = {}
    for fn in (run_freeze, run_build, run_models, run_execution, run_decision):
        print(f"[24.1] {fn.__name__}", flush=True)
        out[fn.__name__] = fn(resume=resume, jobs=jobs)
    return out
