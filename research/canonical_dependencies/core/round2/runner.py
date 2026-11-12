"""Round 2 回测入口：复用 local_executor，产物只写 result/round2/{code}/。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..local_executor import (
    baseline_fixed_sleeve_config,
    run_local_fixed_sleeve_backtest,
    write_backtest_artifacts,
)
from ..metrics import LOCKED_BASELINE
from ..paths import (
    BACKTEST_VARIANT,
    CAPACITY,
    ENSEMBLE_METHOD,
    MARKET_END,
    PREDICTION_START,
    SIGNAL_END,
    TRIAL_ID,
    assert_project_result_path,
    ensure_round2_tree,
    round2_dir,
)
from ..stock_readonly import (
    ensemble_dir,
    load_fs00263_ensemble_predictions,
    load_prepared_market,
    sha256_file,
)
from .data import (
    load_all_seed_predictions,
    load_candidate_features,
    load_money_flow_daily_agg,
    load_regime_frame,
    seed_top1_agreement,
)
from .metrics import attach_round2_metrics
from .overlays import (
    apply_date_multiplier,
    breadth_tier_multiplier,
    combine_multipliers,
    confidence_agreement_multiplier,
    down_high_multiplier,
    fail_high_multiplier,
    margin_multiplier,
    overlay_summary,
    probability_multiplier,
    regime_score_five,
    suc_con_weak_multiplier,
    tertile_multiplier,
)
from .registry import R2Spec, get_experiment, register_combo
from .risk_model import (
    build_top1_feature_table,
    persist_model_note,
    walk_forward_proba,
)

_CACHE: dict[str, Any] = {}


def _cached(key: str, factory):
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def regime_frame() -> pd.DataFrame:
    return _cached("regime", load_regime_frame)


def agreement_frame(predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    def _build():
        ens = predictions if predictions is not None else load_fs00263_ensemble_predictions()
        return seed_top1_agreement(ens, load_all_seed_predictions())

    return _cached("agree", _build)


def build_date_multiplier(spec: R2Spec, predictions: pd.DataFrame) -> tuple[pd.Series, dict[str, Any], dict[str, Any] | None]:
    """Return (date_mult, overlay_meta, overlay_hooks_or_None)."""
    kind = (spec.overlay or {}).get("kind")
    regime = regime_frame()
    hooks: dict[str, Any] | None = None
    meta: dict[str, Any] = {"kind": kind}

    if kind is None or spec.read_existing_ledger:
        idx = pd.DatetimeIndex(pd.to_datetime(predictions["signal_date"].unique())).sort_values()
        return pd.Series(1.0, index=idx), {"kind": "identity"}, None

    if kind == "breadth_tier":
        series = breadth_tier_multiplier(
            regime["up_ratio"],
            spec.overlay["t1"], spec.overlay["t2"], spec.overlay["t3"],
        )
    elif kind == "up_num_tertile":
        series = tertile_multiplier(
            regime["up_num_p250"],
            low=spec.overlay["low"], mid=spec.overlay["mid"], high=spec.overlay["high"],
        )
    elif kind == "fail_high":
        series = fail_high_multiplier(
            regime["fail_ratio_p250"], spec.overlay["threshold"], spec.overlay["high_mult"]
        )
    elif kind == "suc_con_weak":
        series = suc_con_weak_multiplier(
            regime["suc_ratio_p250"], regime["con_ratio_p250"],
            weak=spec.overlay["weak"], very_weak=spec.overlay["very_weak"],
        )
    elif kind == "down_high":
        series = down_high_multiplier(
            regime["down_count_p250"], spec.overlay["threshold"], spec.overlay["high_mult"]
        )
    elif kind == "regime_score":
        up_mid = 1.0 - (regime["up_num_p250"] - 0.5).abs() * 2.0
        parts = {
            "breadth": regime["up_ratio"].clip(0, 1),
            "inv_fail": 1.0 - regime["fail_ratio"].clip(0, 1),
            "suc": regime["suc_ratio"].clip(0, 1),
            "con": regime["con_ratio"].clip(0, 1),
            "inv_down": 1.0 - regime["down_pressure"].clip(0, 1),
            "up_mid": up_mid.clip(0, 1),
        }
        score, series = regime_score_five(parts, hard=bool(spec.overlay.get("hard")))
        meta["regime_score_mean"] = float(score.mean())
    elif kind == "confidence_agree":
        agree = agreement_frame(predictions)
        series = confidence_agreement_multiplier(agree["agree_n"], mode=spec.overlay["mode"])
    elif kind == "margin":
        agree = agreement_frame(predictions)
        series = margin_multiplier(
            agree["margin_p250"],
            low_mult=spec.overlay["low_mult"],
            cutoff=spec.overlay["cutoff"],
        )
    elif kind == "nav_dd":
        idx = pd.DatetimeIndex(pd.to_datetime(predictions["signal_date"].unique())).sort_values()
        series = pd.Series(1.0, index=idx)
        hooks = {"nav_dd": {
            "soft": spec.overlay["soft"],
            "hard": spec.overlay["hard"],
            "soft_mult": spec.overlay["soft_mult"],
        }}
    elif kind == "exit_regime":
        idx = pd.DatetimeIndex(pd.to_datetime(predictions["signal_date"].unique())).sort_values()
        series = pd.Series(1.0, index=idx)
        bad = regime.index[
            (regime["fail_ratio_p250"] >= 0.80)
            | (regime["down_count_p250"] >= 0.80)
            | (regime["up_ratio"] < 0.40)
        ]
        hooks = {"early_exit": "regime_deteriorate", "regime_bad_dates": [str(x) for x in bad]}
    elif kind == "exit_low_conf":
        idx = pd.DatetimeIndex(pd.to_datetime(predictions["signal_date"].unique())).sort_values()
        series = pd.Series(1.0, index=idx)
        agree = agreement_frame(predictions)
        low = agree.index[agree["agree_n"] <= 2]
        hooks = {"early_exit": "low_conf_no_con", "low_conf_dates": [str(x) for x in low]}
    elif kind == "exit_ma5":
        idx = pd.DatetimeIndex(pd.to_datetime(predictions["signal_date"].unique())).sort_values()
        series = pd.Series(1.0, index=idx)
        hooks = {"ma5_buffer": float(spec.overlay["buffer"])}
    elif kind in {"f01_logit", "f01_xgb", "f03_continue"}:
        table = _cached("top1_features", build_top1_feature_table)
        if kind == "f03_continue":
            proba = walk_forward_proba(table, label_col="label_d1_touch", threshold=None, model="logit")
            series = probability_multiplier(proba, skip_above=0.70, half_above=0.55, invert=True)
            persist_model_note(spec.code, {"kind": kind, "label": "label_d1_touch", "n": int(proba.notna().sum())})
        else:
            model = "logit" if kind == "f01_logit" else "xgb"
            proba = walk_forward_proba(table, label_col="label_e2_net", threshold=-0.05, model=model)
            if proba.empty:
                raise RuntimeError(f"{spec.code}: xgboost unavailable or empty proba")
            series = probability_multiplier(proba, skip_above=0.55, half_above=0.40, invert=False)
            persist_model_note(spec.code, {"kind": kind, "label": "label_e2_net<=-5%", "n": int(proba.notna().sum())})
    elif kind == "money_flow_weak":
        mf = _cached("money_flow", load_money_flow_daily_agg)
        p = mf["main_net_p250"]
        series = pd.Series(1.0, index=p.index, dtype=float)
        series = series.where(~p.le(spec.overlay["weak"]), 0.5)
        series = series.where(~p.le(spec.overlay["very_weak"]), 0.0)
    elif kind == "industry_crowd":
        ens = predictions.copy()
        ens["signal_date"] = pd.to_datetime(ens["signal_date"])
        top1 = (
            ens.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True])
            .groupby("signal_date", sort=True).head(1)
        )
        panel = _cached("panel_ind", lambda: load_candidate_features(
            ["trade_date", "instrument_id", "ind_first_board_count_log"]
        ))
        merged = top1.merge(
            panel, left_on=["signal_date", "instrument_id"],
            right_on=["trade_date", "instrument_id"], how="left",
        )
        merged["ind_p"] = merged["ind_first_board_count_log"].rank(pct=True)
        series = pd.Series(1.0, index=pd.to_datetime(merged["signal_date"]), dtype=float)
        half_mask = merged["ind_p"].ge(spec.overlay["half"]).to_numpy()
        skip_mask = merged["ind_p"].ge(spec.overlay["skip"]).to_numpy()
        values = series.to_numpy(copy=True)
        values[half_mask] = 0.5
        values[skip_mask] = 0.0
        series = pd.Series(values, index=pd.to_datetime(merged["signal_date"]))
    elif kind == "down_pressure":
        p = regime["down_pressure_p250"]
        series = pd.Series(1.0, index=p.index, dtype=float)
        series = series.where(~p.ge(spec.overlay["half"]), 0.5)
        series = series.where(~p.ge(spec.overlay["skip"]), 0.0)
    elif kind == "product":
        product = None
        hooks_acc: dict[str, Any] = {}
        for child_code in spec.overlay["components"]:
            child = get_experiment(child_code)
            child_series, child_meta, child_hooks = build_date_multiplier(child, predictions)
            product = child_series if product is None else _align_min_product(product, child_series)
            if child_hooks:
                hooks_acc.update(child_hooks)
            meta.setdefault("components", []).append({"code": child_code, **child_meta})
        series = product if product is not None else pd.Series(dtype=float)
        hooks = hooks_acc or None
    else:
        raise ValueError(f"unknown overlay kind: {kind}")

    series.index = pd.to_datetime(series.index)
    meta.update(overlay_summary(series))
    return series, meta, hooks


def _align_min_product(a: pd.Series, b: pd.Series) -> pd.Series:
    idx = a.index.union(b.index)
    aa = a.reindex(idx).fillna(1.0)
    bb = b.reindex(idx).fillna(1.0)
    out = pd.Series([combine_multipliers(x, y) for x, y in zip(aa, bb)], index=idx)
    return out


def _copy_locked_alias(code: str) -> dict[str, Any]:
    """R2_B00 / R2_D05：读锁定账本，不重跑 Stock。优先用课题重建副本。"""
    dest = round2_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")
    source = round2_dir("00_audit") / "locked_reconstruct"
    if not (source / "report.json").is_file():
        raise RuntimeError(
            "locked reconstruct missing; run P0 attribution first "
            f"(expected {source / 'report.json'})"
        )
    for name in ("report.json", "nav.parquet", "trades.parquet", "rejections.parquet",
                 "positions.parquet", "sleeve_states.parquet", "cashflows.parquet", "costs.parquet"):
        src = source / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            assert_project_result_path(dest / name)
    report = json.loads((dest / "report.json").read_text(encoding="utf-8"))
    nav = pd.read_parquet(dest / "nav.parquet") if (dest / "nav.parquet").is_file() else pd.DataFrame()
    trades = pd.read_parquet(dest / "trades.parquet") if (dest / "trades.parquet").is_file() else pd.DataFrame()
    metrics = report.setdefault("metrics", {})
    attach_round2_metrics(nav, trades, metrics)
    report["experiment_code"] = code
    report["alias_of"] = "R1_00_baseline"
    report["note"] = "锁定 baseline 别名；不重跑 Stock。数字来自课题侧重建/拷贝。"
    (dest / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    summary = tracking_row_from_report(code, report)
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {"status": "aliased_locked", "code": code, "report": report, "summary": summary}


def load_result_report(code: str) -> dict[str, Any] | None:
    path = round2_dir(code) / "report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tracking_row_from_report(code: str, report: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = get_experiment(code)
    report = report or load_result_report(code)
    ov = spec.overlay or {}
    row = {
        "code": spec.code,
        "module": spec.module,
        "title_zh": spec.title_zh,
        "regime": ov.get("kind") if spec.module in {"B", "P", "T", "H"} else "",
        "confidence": ov.get("kind") if spec.module == "C" else "",
        "risk": ov.get("kind") if spec.module in {"F", "G"} else "",
        "position": spec.name_budget_nav_fraction,
        "exit": spec.exit_mode if spec.module == "E" or spec.exit_mode != "close_from_d2" else spec.exit_mode,
        "cagr": None,
        "sharpe": None,
        "sortino": None,
        "calmar": None,
        "max_drawdown": None,
        "worst_year_return": None,
        "worst_12m": None,
        "completed_trades": None,
        "status": "待跑",
    }
    if spec.skip:
        row["status"] = f"跳过：{spec.skip_reason}"
        return row
    if spec.diagnostic_only:
        row["status"] = "诊断-不回测"
        return row
    if report is None:
        return row
    m = report.get("metrics") or {}
    row.update({
        "cagr": m.get("annualized_return"),
        "sharpe": m.get("sharpe_zero_rate"),
        "sortino": m.get("sortino_zero_rate"),
        "calmar": m.get("calmar"),
        "max_drawdown": m.get("max_drawdown"),
        "worst_year_return": m.get("worst_year_return"),
        "worst_12m": m.get("worst_rolling_12m"),
        "completed_trades": m.get("completed_trades"),
        "status": report.get("status") or "已跑",
        "p0_maxdd_ok": m.get("p0_maxdd_ok"),
        "near_zero_trade_curve": m.get("near_zero_trade_curve"),
    })
    return row


def run_experiment(
    code: str,
    *,
    run_now: bool = False,
    predictions: Any | None = None,
    prepared_market: Any | None = None,
    market_meta: dict[str, Any] | None = None,
    start_date=PREDICTION_START,
    end_date=MARKET_END,
) -> dict[str, Any]:
    ensure_round2_tree()
    spec = get_experiment(code)
    if spec.skip:
        dest = round2_dir(spec.code)
        dest.mkdir(parents=True, exist_ok=True)
        note = {"code": spec.code, "status": "skipped", "reason": spec.skip_reason}
        assert_project_result_path(dest / "summary.json")
        (dest / "summary.json").write_text(
            json.dumps(note, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return {"status": "skipped", "code": spec.code, "reason": spec.skip_reason}
    if spec.diagnostic_only:
        return {"status": "diagnostic_only", "code": spec.code}
    if spec.read_existing_ledger:
        existing = load_result_report(spec.code)
        if existing is not None and not run_now:
            return {"status": "loaded", "code": spec.code, "report": existing,
                    "summary": tracking_row_from_report(spec.code, existing)}
        return _copy_locked_alias(spec.code)

    existing = load_result_report(spec.code)
    if not run_now:
        if existing is not None:
            return {
                "status": "loaded",
                "code": spec.code,
                "report": existing,
                "summary": tracking_row_from_report(spec.code, existing),
            }
        return {
            "status": "pending",
            "code": spec.code,
            "summary": tracking_row_from_report(spec.code),
            "note": "RUN_NOW=False 且 result/round2 尚无账本；全样本请 nohup 跑。",
        }

    if predictions is None:
        predictions = load_fs00263_ensemble_predictions(start=start_date, signal_end=SIGNAL_END)
    if prepared_market is None:
        loaded = load_prepared_market(warmup_ma5=True)
        prepared_market = loaded["prepared"]
        market_meta = loaded

    date_mult, overlay_meta, hooks = build_date_multiplier(spec, predictions)
    overlayed = apply_date_multiplier(predictions, date_mult)
    merged = overlayed.merge(
        predictions[["signal_date", "instrument_id", "score"]],
        on=["signal_date", "instrument_id"],
        suffixes=("", "_src"),
    )
    if merged.empty or not (merged["score"] == merged["score_src"]).all():
        raise RuntimeError(f"{spec.code}: score mutated")

    stock_exits = {"close_from_d2", "next_open_after_non_limit_close", "ma5_next_open"}
    exit_mode = spec.exit_mode if spec.exit_mode in stock_exits else "close_from_d2"
    config = baseline_fixed_sleeve_config(exit_mode=exit_mode)
    from dataclasses import replace as _replace
    config = _replace(
        config,
        name_budget_nav_fraction=spec.name_budget_nav_fraction,
        market_up_ratio_min=spec.market_up_ratio_min,
    )
    result = run_local_fixed_sleeve_backtest(
        overlayed,
        prepared_market,
        config=config,
        start_date=start_date,
        end_date=end_date,
        overlay_hooks=hooks,
    )
    metrics = result["report"].setdefault("metrics", {})
    attach_round2_metrics(result["nav"], result["trades"], metrics)
    acceptance = {
        "max_live_sleeves_lte_2": metrics.get("max_live_sleeves", 0) <= 2,
        "no_negative_sleeve_cash": metrics.get("minimum_sleeve_cash", 0) >= -1e-8,
        "asset_identity": metrics.get("asset_identity_max_abs", 0) <= 1e-6,
        "score_unmutated": True,
    }
    extra = {
        "status": "completed_and_validated",
        "experiment_code": spec.code,
        "trial_id": TRIAL_ID,
        "capacity": CAPACITY,
        "seed": "ensemble",
        "ensemble": ENSEMBLE_METHOD,
        "variant": BACKTEST_VARIANT,
        "overlay": overlay_meta,
        "overlay_hooks": hooks,
        "name_budget_nav_fraction": spec.name_budget_nav_fraction,
        "market_up_ratio_min": spec.market_up_ratio_min,
        "predictions_file": str(ensemble_dir() / "predictions.parquet"),
        "predictions_sha256": sha256_file(ensemble_dir() / "predictions.parquet"),
        "acceptance": acceptance,
        "evidence_role": "round2_historical_development",
        "note": (
            "Alpha 锁定 fs00263；Risk 只改 allocation_multiplier / gate / exit override。"
            "产物在课题 result/round2/，未写 Stock/runs。不是 final OOS。"
        ),
        "locked_baseline_ref": {
            "code": LOCKED_BASELINE["code"],
            "sharpe": LOCKED_BASELINE["sharpe"],
            "max_drawdown": LOCKED_BASELINE["max_drawdown"],
        },
    }
    if market_meta:
        extra["snapshot_id"] = market_meta.get("snapshot_id")
        extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")
    dest = round2_dir(spec.code)
    written = write_backtest_artifacts(result, dest, extra_report=extra)
    # also drop compact copies under shared folders
    shared_metrics = ensure_round2_tree() / "metrics" / f"{spec.code}.json"
    assert_project_result_path(shared_metrics)
    shared_metrics.write_text(
        json.dumps(written["report"].get("metrics") or {}, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for name in ("nav", "trades"):
        src = dest / f"{name}.parquet"
        if src.is_file():
            target = ensure_round2_tree() / name / f"{spec.code}.parquet"
            shutil.copy2(src, target)
            assert_project_result_path(target)
    cfg_path = ensure_round2_tree() / "configs" / f"{spec.code}.json"
    assert_project_result_path(cfg_path)
    cfg_path.write_text(
        json.dumps({
            "code": spec.code,
            "overlay": spec.overlay,
            "exit_mode": spec.exit_mode,
            "name_budget_nav_fraction": spec.name_budget_nav_fraction,
            "market_up_ratio_min": spec.market_up_ratio_min,
            "overlay_meta": overlay_meta,
        }, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    summary = tracking_row_from_report(spec.code, written["report"])
    summary_path = dest / "summary.json"
    assert_project_result_path(summary_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {
        "status": "ran",
        "code": spec.code,
        "out_dir": written["out_dir"],
        "report": written["report"],
        "summary": summary,
        "overlay": overlay_meta,
    }


def add_combo_and_run(
    code: str,
    components: list[str],
    *,
    module: str,
    title_zh: str,
    run_now: bool,
    predictions=None,
    prepared_market=None,
    market_meta=None,
) -> dict[str, Any]:
    children = [get_experiment(c) for c in components]
    nav_frac = min(c.name_budget_nav_fraction for c in children)
    market_min = min(c.market_up_ratio_min for c in children)
    exit_modes = [c.exit_mode for c in children if c.exit_mode != "close_from_d2"]
    exit_mode = exit_modes[0] if exit_modes else "close_from_d2"
    needs_loop = any(c.needs_overlay_loop for c in children)
    from .registry import EXPERIMENTS
    if code not in EXPERIMENTS:
        register_combo(
            code,
            module=module,
            title_zh=title_zh,
            description_zh=" × ".join(components),
            overlay={"kind": "product", "components": components},
            exit_mode=exit_mode,
            name_budget_nav_fraction=nav_frac,
            market_up_ratio_min=market_min,
            needs_overlay_loop=needs_loop,
        )
    return run_experiment(
        code,
        run_now=run_now,
        predictions=predictions,
        prepared_market=prepared_market,
        market_meta=market_meta,
    )
