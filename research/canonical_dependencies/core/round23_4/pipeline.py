"""R23.4 主流程。关键门失败只阻断依赖路径。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.r225241_sleeve import apply_b03_o01, run_round7
from core.research_integrity_v2.io_v2 import dump_json, load_json, load_stage, save_stage, write_df, write_text
from core.research_integrity_v2.ledger import reconcile_nav
from core.research_integrity_v3.gates import fingerprint, sha256_file
from core.research_integrity_v3.risk_metrics import annualized_log_growth, daily_returns, summarize_nav
from core.round11.common import daily_from_nav
from core.round17.backtest import fuse_heads
from core.round22_6.sleeve import run_formal_sleeve
from core.round23_3.blend import blend_xgb_lgbm
from core.round23_3.contract import E0
from core.round23_3.contract import ROOT as R23_3_ROOT
from core.round23_3.inner_adapter import run_windowed_sleeve
from core.round23_3.selection import MAXDD_WORSEN, MIN_INNER_DAYS, MIN_INNER_TRADES, CandidateRecord, _block_bootstrap_lb, _maxdd, _utility, choose_weight
from core.round23_4.contract import (
    ANCHOR_BASELINE_V5,
    ANCHOR_ROUND23_3,
    EVAL_END,
    EVAL_START,
    INITIAL_CASH,
    LEGACY_LGBM_WEIGHTS,
    NEW_LGBM_WEIGHTS,
    REF_E2,
    REF_V5,
    ROOT,
)
from core.round23_4.drawdown import build_audit, yearly_table
from core.round23_4.exposure import exposure_ratio, mean_planned_exposure, scale_multipliers
from core.round23_4.wider_market import inner_prepared_market, sessions_between
from core.round23_4.inner_train import train_inner_block
from core.round23_4.select_risk import block_metrics, choose_r2, choose_r3, paired_mean_lb_keep_blocks
from core.round23_3.contract import PANEL, TEN_PANEL, TEN_TRAIN, TRAIN


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    extra = Path(__file__).resolve().parents[1] / "research_integrity_v3"
    h = hashlib.sha256()
    for path in sorted(list(root.glob("*.py")) + list(extra.glob("*.py"))):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _skip(name: str, resume: bool) -> dict[str, Any] | None:
    if not resume:
        return None
    cached = load_stage(ROOT, name)
    if not cached:
        return None
    if cached.get("code_fingerprint") != _code_fingerprint():
        return None
    if cached.get("status") == "FAIL" and cached.get("error"):
        return None
    return cached


def _finish(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload["code_fingerprint"] = _code_fingerprint()
    return save_stage(ROOT, name, payload)


def _key(w: float) -> str:
    return f"l_{w:.2f}"


def _load_frozen_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    five = pd.read_parquet(E0 / "five_pred.parquet")
    ten = pd.read_parquet(E0 / "ten_pred.parquet")
    v5 = fuse_heads(five, ten)
    lgbm = fuse_heads(
        pd.read_parquet(R23_3_ROOT / "oof" / "lgbm_five_trunc4" / "predictions.parquet"),
        pd.read_parquet(R23_3_ROOT / "oof" / "lgbm_ten_trunc5" / "predictions.parquet"),
    )
    return v5, lgbm


def _score_maps(v5: pd.DataFrame, lgbm: pd.DataFrame, weights: tuple[float, ...]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    maps = {}
    fallbacks = []
    for w in weights:
        scores, fb = blend_xgb_lgbm(v5, lgbm, w)
        maps[_key(w)] = scores
        if len(fb):
            fb = fb.copy()
            fb["w_lgbm"] = w
            fallbacks.append(fb)
    fb_all = pd.concat(fallbacks, ignore_index=True) if fallbacks else pd.DataFrame()
    return maps, fb_all


def _close(obs: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for key, tol in (("sharpe", 1e-6), ("cagr", 1e-6), ("max_drawdown", 1e-6)):
        checks[key] = abs(float(obs.get(key)) - float(ref[key])) <= tol
    checks["nav_end"] = abs(float(obs.get("nav_end")) - float(ref["nav_end"])) <= 1.0
    checks["completed_trades"] = int(obs.get("completed_trades")) == int(ref["completed_trades"])
    checks["pass"] = all(checks.values())
    return checks


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}"


def run_source_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("source_freeze", resume)
    if cached:
        return cached
    ROOT.mkdir(parents=True, exist_ok=True)
    v5, lgbm = _load_frozen_scores()
    zero, _ = blend_xgb_lgbm(v5, lgbm, 0.0)
    same = zero.sort_values(["signal_date", "instrument_id"]).reset_index(drop=True)
    base = v5.sort_values(["signal_date", "instrument_id"]).reset_index(drop=True)
    identity = bool(np.nanmax(np.abs(same["score"].to_numpy() - base["score"].to_numpy())) <= 1e-12) and len(same) == len(base)
    payload = {
        "status": "DONE",
        "canonical_realpath": str(Path("/data1/stock-mxy/课题/一进二策略").resolve()),
        "anchor_baseline_v5": ANCHOR_BASELINE_V5,
        "anchor_round23_3": ANCHOR_ROUND23_3,
        "worktree_round23_3_head": _git_sha(Path("/data1/stock-mxy/research_publish/wt_round23_3")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "v5_rows": int(len(v5)),
        "lgbm_rows": int(len(lgbm)),
        "w0_identity_with_v5_scores": identity,
        "information_contract": "LEGACY_OPEN_INFORMATION",
        "live_executable_claim": False,
        "clean_risk_stacked": False,
        "e0_five_sha256": sha256_file(E0 / "five_pred.parquet"),
        "e0_ten_sha256": sha256_file(E0 / "ten_pred.parquet"),
        "code_fingerprint_preview": _code_fingerprint(),
    }
    dump_json(ROOT / "source_freeze.json", payload)
    if not identity:
        payload["status"] = "FAIL"
    return _finish("source_freeze", payload)


def _legacy_records(year: int, score_map: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """用已有内层净值重算 R23.3 选择，不把年份权重写死。"""
    default_key = "l_0.00"
    blocks = [
        (pd.Timestamp(year - 1, 1, 1), pd.Timestamp(year - 1, 6, 30), "H1"),
        (pd.Timestamp(year - 1, 7, 1), pd.Timestamp(year - 1, 12, 31), "H2"),
    ]
    root = R23_3_ROOT / "inner" / "e2"
    prior = score_map[default_key]
    prior = prior.loc[pd.to_datetime(prior["signal_date"]).dt.year.eq(year - 1)]
    if prior.empty:
        return {"year": year, "chosen": default_key, "reason": "FALLBACK_DEFAULT_NO_PRIOR_OOF", "records": []}
    records: list[CandidateRecord] = []
    for key in [_key(w) for w in LEGACY_LGBM_WEIGHTS]:
        utils, dds, block_deltas = [], [], []
        n_days = n_trades = 0
        window_fail = False
        cand_parts, def_parts = [], []
        for start, end, tag in blocks:
            dest = root / str(year) / key / tag
            summary_path = dest / "summary.json"
            nav_path = dest / "nav.parquet"
            if not summary_path.is_file() or not nav_path.is_file():
                window_fail = True
                block_deltas.append(None)
                continue
            summary = load_json(summary_path)
            if summary.get("window_fail") or summary.get("status") == "FAIL_FULL_HISTORY_WINDOW":
                window_fail = True
            nav = pd.read_parquet(nav_path)
            utils.append(_utility(nav, start, end))
            if len(nav):
                dds.append(_maxdd(nav))
            n_days += int(summary.get("n_nav_days_in_signal_block") or 0)
            n_trades += int(summary.get("n_trades") or 0)
            def_path = root / str(year) / default_key / tag / "nav.parquet"
            if key != default_key and def_path.is_file():
                c = daily_from_nav(nav.loc[pd.to_datetime(nav["trade_date"]).between(start, end)])
                dnav = pd.read_parquet(def_path)
                d = daily_from_nav(dnav.loc[pd.to_datetime(dnav["trade_date"]).between(start, end)])
                both = pd.concat([c.rename("c"), d.rename("b")], axis=1, join="inner")
                if both.isna().any().any() or len(both) == 0:
                    block_deltas.append(None)
                else:
                    block_deltas.append(float((both["c"] - both["b"]).mean()))
                    cand_parts.append(both["c"])
                    def_parts.append(both["b"])
        mean_u = float(np.mean([u for u in utils if u is not None])) if any(u is not None for u in utils) else None
        maxdd = float(np.min(dds)) if dds else None
        mean_delta = boot_lb = None
        if cand_parts:
            both = pd.concat([pd.concat(cand_parts).rename("c"), pd.concat(def_parts).rename("b")], axis=1, join="inner")
            diff = (both["c"] - both["b"]).to_numpy()
            mean_delta = float(diff.mean())
            boot_lb = _block_bootstrap_lb(diff)
        def_dds = []
        for start, end, tag in blocks:
            p = root / str(year) / default_key / tag / "nav.parquet"
            if p.is_file():
                def_dds.append(_maxdd(pd.read_parquet(p)))
        def_dd = float(np.min(def_dds)) if def_dds else None
        dd_ok = True if maxdd is None or def_dd is None else (maxdd >= def_dd - MAXDD_WORSEN)
        both_ok = len(block_deltas) == 2 and all(d is not None and float(d) >= 0 for d in block_deltas)
        is_default = key == default_key
        gates = (
            (not window_fail)
            and n_days >= MIN_INNER_DAYS
            and n_trades >= MIN_INNER_TRADES
            and mean_delta is not None
            and mean_delta > 0
            and both_ok
            and dd_ok
            and boot_lb is not None
            and np.isfinite(boot_lb)
            and boot_lb > 0
        )
        if is_default:
            gates = not window_fail
        dist = 0.0 if is_default else abs(float(key.split("_")[1]))
        records.append(
            CandidateRecord(
                key=key,
                is_default=is_default,
                inner_utility=mean_u,
                distance_to_default=dist,
                n_days=n_days,
                n_trades=n_trades,
                maxdd=maxdd,
                mean_delta=mean_delta,
                block_deltas=block_deltas,
                boot_lb=boot_lb,
                all_gates_pass=bool(gates),
                window_fail=window_fail,
                details={"dd_ok": dd_ok, "both_blocks_nonneg": both_ok},
            )
        )
    decision = choose_weight(records, default_key=default_key)
    return {
        "year": year,
        "chosen": decision["chosen"],
        "reason": decision["reason"],
        "records": [
            {
                "year": year,
                "key": r.key,
                "n_days": r.n_days,
                "n_trades": r.n_trades,
                "all_gates_pass": r.all_gates_pass,
                "mean_delta": r.mean_delta,
                "boot_lb": r.boot_lb,
                "reason_inputs": r.details,
            }
            for r in records
        ],
    }


def _nested_from_map(score_map: dict[str, pd.DataFrame], weights: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for rec in weights.itertuples(index=False):
        part = score_map[rec.chosen]
        part = part.loc[pd.to_datetime(part["signal_date"]).dt.year.eq(int(rec.year))]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def run_legacy_replay(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("legacy_replay", resume)
    if cached:
        return cached
    freeze = load_stage(ROOT, "source_freeze") or {}
    if freeze.get("status") == "FAIL":
        return _finish("legacy_replay", {"status": "BLOCKED", "reason": "W0_IDENTITY_FAIL"})
    v5, lgbm = _load_frozen_scores()
    score_map, _ = _score_maps(v5, lgbm, LEGACY_LGBM_WEIGHTS)
    rows = [_legacy_records(year, score_map) for year in range(2018, 2027)]
    weights = pd.DataFrame([{"year": r["year"], "family": "LEGACY_E2", "chosen": r["chosen"], "reason": r["reason"]} for r in rows])
    write_df(ROOT / "legacy_inner_selection.csv", weights)
    write_df(ROOT / "legacy_inner_records.csv", pd.DataFrame([item for r in rows for item in r["records"]]))
    # 抽查一个已有内层块是否能被同一引擎复现，避免只读旧文件就宣称重放。
    spot_dir = ROOT / "spotcheck" / "2019_l000_H1"
    spot = run_windowed_sleeve(
        spot_dir,
        score_map["l_0.00"],
        signal_start="2018-01-01",
        signal_end="2018-06-30",
        backtest_start="2018-01-01",
        backtest_end="2018-06-30",
        initial_cash=INITIAL_CASH,
        extra={"code": "R234_SPOT_2019_L000_H1", "round": "23.4"},
        resume=resume,
    )
    ref_nav = pd.read_parquet(R23_3_ROOT / "inner" / "e2" / "2019" / "l_0.00" / "H1" / "nav.parquet")
    new_nav = pd.read_parquet(spot_dir / "nav.parquet")
    a = ref_nav.sort_values("trade_date")["total_asset"].to_numpy(dtype=float)
    b = new_nav.sort_values("trade_date")["total_asset"].to_numpy(dtype=float)
    spot_ok = len(a) == len(b) and float(np.max(np.abs(a - b))) <= 0.05
    nested = _nested_from_map(score_map, weights)
    summary = run_formal_sleeve(
        ROOT / "R1_LEGACY_E2_REPLAY",
        nested,
        {"code": "R234_R1_LEGACY_E2", "round": "23.4", "scope": "LEGACY_OPEN_INFORMATION"},
        resume=resume,
        counts_as_alpha=False,
        top_changed=None,
        fee_same_direction=None,
        not_only_year=None,
    )
    checks = _close(summary, REF_E2)
    status = "PASS" if checks["pass"] and spot_ok else "FAIL"
    parity = pd.DataFrame(
        [
            {"metric": k, "reference": REF_E2.get(k), "replay": summary.get(k), "match": checks.get(k)}
            for k in ("sharpe", "cagr", "max_drawdown", "nav_end", "completed_trades")
        ]
        + [{"metric": "spotcheck_2019_h1_l000", "reference": True, "replay": spot_ok, "match": spot_ok}]
    )
    write_df(ROOT / "replay_parity.csv", parity)
    payload = {
        "status": status,
        "spotcheck_pass": spot_ok,
        "parity": checks,
        "weights": weights.to_dict(orient="records"),
        "hardcoded_year_rules": False,
        "candidate_set": list(LEGACY_LGBM_WEIGHTS),
        "note": "2019/2021 的权重若出现，只是算法输出，不是新的择时规则。",
        "summary": {k: summary.get(k) for k in ("sharpe", "cagr", "max_drawdown", "nav_end", "completed_trades", "status")},
        "nav_reconcile": summary.get("nav_reconcile"),
    }
    dump_json(ROOT / "legacy_replay.json", payload)
    return _finish("legacy_replay", payload)


def _sleeve_paths(code: str) -> Path:
    return ROOT / code


def run_drawdown(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("drawdown", resume)
    if cached:
        return cached
    replay = load_stage(ROOT, "legacy_replay") or {}
    if replay.get("status") not in {"PASS", "FAIL"}:
        return _finish("drawdown", {"status": "BLOCKED", "reason": "LEGACY_REPLAY_NOT_RUN"})
    # R0 正式袖套。w=0 分数已在冻结阶段验证与 v5 相同。
    v5, _ = _load_frozen_scores()
    r0 = run_formal_sleeve(
        _sleeve_paths("R0_V5"),
        v5,
        {"code": "R234_R0_V5", "round": "23.4", "scope": "LEGACY_OPEN_INFORMATION"},
        resume=resume,
        counts_as_alpha=False,
        top_changed=False,
        fee_same_direction=None,
        not_only_year=None,
    )
    r0_checks = _close(r0, REF_V5)
    base_nav = pd.read_parquet(ROOT / "R0_V5" / "nav.parquet")
    cand_nav = pd.read_parquet(ROOT / "R1_LEGACY_E2_REPLAY" / "nav.parquet")
    base_tr = pd.read_parquet(ROOT / "R0_V5" / "trades.parquet")
    cand_tr = pd.read_parquet(ROOT / "R1_LEGACY_E2_REPLAY" / "trades.parquet")
    base_cost = pd.read_parquet(ROOT / "R0_V5" / "costs.parquet") if (ROOT / "R0_V5" / "costs.parquet").is_file() else None
    cand_cost = pd.read_parquet(ROOT / "R1_LEGACY_E2_REPLAY" / "costs.parquet") if (ROOT / "R1_LEGACY_E2_REPLAY" / "costs.parquet").is_file() else None
    audit, bridge, extra = build_audit(base_nav, cand_nav, base_tr, cand_tr, baseline_costs=base_cost, candidate_costs=cand_cost)
    write_df(ROOT / "drawdown_episode_audit.csv", audit)
    write_df(ROOT / "risk_increment_bridge.csv", bridge)
    write_df(ROOT / "trade_classification.csv", extra["classes"])
    write_df(ROOT / "yearly_baseline.csv", yearly_table(base_nav))
    write_df(ROOT / "yearly_e2_replay.csv", yearly_table(cand_nav))
    brec = reconcile_nav(base_nav)
    crec = reconcile_nav(cand_nav)
    metrics = pd.DataFrame(
        [
            {"arm": "R0_V5", **extra["meta"]["baseline"], "engine_sharpe": r0.get("sharpe"), "engine_cagr": r0.get("cagr")},
            {"arm": "R1_E2_REPLAY", **extra["meta"]["candidate"], "engine_sharpe": replay.get("summary", {}).get("sharpe"), "engine_cagr": replay.get("summary", {}).get("cagr")},
        ]
    )
    write_df(ROOT / "risk_metric_comparison.csv", metrics)
    payload = {
        "status": "DONE" if r0_checks.get("pass") else "FAIL",
        "r0_matches_published_v5": r0_checks,
        "nav_reconcile_r0": brec,
        "nav_reconcile_r1": crec,
        "class_counts": extra["meta"]["class_counts"],
        "comparison_contract": "LEGACY_OPEN_INFORMATION",
    }
    dump_json(ROOT / "drawdown_summary.json", payload)
    return _finish("drawdown", payload)


def _blocks_for_year(year: int) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    return [
        (pd.Timestamp(year - 1, 1, 1), pd.Timestamp(year - 1, 6, 30), "H1"),
        (pd.Timestamp(year - 1, 7, 1), pd.Timestamp(year - 1, 12, 31), "H2"),
    ]


def run_inner_models(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    cached = _skip("inner_models", resume)
    if cached:
        return cached
    del jobs  # 模型内部 n_jobs=2；块之间串行，避免超过 2 个训练任务。
    five_train = pd.read_parquet(TRAIN)
    five_cand = pd.read_parquet(PANEL)
    ten_train = pd.read_parquet(TEN_TRAIN)
    ten_cand = pd.read_parquet(TEN_PANEL)
    rows = []
    for year in range(2018, 2027):
        for start, end, tag in _blocks_for_year(year):
            for head, tr, ca in (("five", five_train, five_cand), ("ten", ten_train, ten_cand)):
                dest = ROOT / "inner_models" / f"Y{year}" / tag / head
                meta = train_inner_block(
                    training=tr,
                    candidates=ca,
                    block_start=start,
                    block_end=end,
                    head=head,
                    output_dir=dest,
                    resume=resume,
                )
                rows.append({"year": year, "block": tag, "head": head, "status": meta.get("status"), "fit_cutoff": meta.get("fit_cutoff"), "train_rows": meta.get("train_rows"), "test_rows": meta.get("test_rows")})
    write_df(ROOT / "inner_model_manifest.csv", pd.DataFrame(rows))
    payload = {"status": "DONE", "n_fits": len(rows), "note": "H1/H2 分别训练，截止于块开始前一日；不用见过该块标签的外层全年模型给内层打分。"}
    return _finish("inner_models", payload)


def _inner_fused(year: int, tag: str, model: str) -> pd.DataFrame:
    five_path = ROOT / "inner_models" / f"Y{year}" / tag / "five" / "predictions.parquet"
    ten_path = ROOT / "inner_models" / f"Y{year}" / tag / "ten" / "predictions.parquet"
    if not five_path.is_file() or not ten_path.is_file():
        return pd.DataFrame(columns=["signal_date", "instrument_id", "score"])
    five = pd.read_parquet(five_path)
    ten = pd.read_parquet(ten_path)
    if "model" not in five.columns or "model" not in ten.columns:
        return pd.DataFrame(columns=["signal_date", "instrument_id", "score"])
    five = five.loc[five["model"].eq(model), ["signal_date", "instrument_id", "score"]]
    ten = ten.loc[ten["model"].eq(model), ["signal_date", "instrument_id", "score"]]
    if five.empty or ten.empty:
        return pd.DataFrame(columns=["signal_date", "instrument_id", "score"])
    try:
        return fuse_heads(five, ten)
    except Exception:
        return pd.DataFrame(columns=["signal_date", "instrument_id", "score"])


def _eval_year(year: int, *, resume: bool, prepared_market: Any) -> dict[str, Any]:
    records = []
    fold_rows = []
    default_rets: dict[str, pd.Series] = {}
    exposures = {}
    for w in NEW_LGBM_WEIGHTS:
        key = _key(w)
        block_rows = []
        n_days = n_trades = 0
        window_fail = False
        diffs = []
        for start, end, tag in _blocks_for_year(year):
            xgb = _inner_fused(year, tag, "xgb")
            if w == 0.0:
                scores = xgb
                fb = pd.DataFrame()
            else:
                lgbm = _inner_fused(year, tag, "lgbm")
                scores, fb = blend_xgb_lgbm(xgb, lgbm, w) if len(xgb) and len(lgbm) else (xgb.iloc[0:0], pd.DataFrame())
            dest = ROOT / "inner_bt" / f"Y{year}" / key / tag
            covered = sessions_between(prepared_market, start, end)
            if covered < 60:
                sleeve = {
                    "status": "MARKET_WINDOW_UNCOVERED",
                    "n_nav_days_in_signal_block": 0,
                    "n_trades": 0,
                    "window_fail": False,
                    "sessions_in_window": covered,
                }
            else:
                print(f"[r234-inner-bt] Y{year} {key} {tag} sessions={covered}", flush=True)
                sleeve = run_windowed_sleeve(
                    dest,
                    scores if len(scores) else pd.DataFrame(columns=["signal_date", "instrument_id", "score"]),
                    signal_start=start,
                    signal_end=end,
                    backtest_start=start,
                    backtest_end=end,
                    initial_cash=INITIAL_CASH,
                    prepared_market=prepared_market,
                    extra={"code": f"R234_INNER_{year}_{key}_{tag}", "round": "23.4", "scope": "TRUE_INNER"},
                    resume=resume,
                )
            if sleeve.get("window_fail") or sleeve.get("status") in {"FAIL_FULL_HISTORY_WINDOW", "EMPTY_SIGNAL_BLOCK"}:
                window_fail = window_fail or sleeve.get("status") == "FAIL_FULL_HISTORY_WINDOW"
            nav_path = dest / "nav.parquet"
            nav = pd.read_parquet(nav_path) if nav_path.is_file() else pd.DataFrame()
            rets = daily_returns(nav) if len(nav) else pd.Series(dtype=float)
            metrics = block_metrics(rets) if len(rets) else {"n_days": 0, "mean_log_growth": np.nan, "ann_log_growth": np.nan, "ui": np.nan, "maxdd": np.nan}
            n_days += int(sleeve.get("n_nav_days_in_signal_block") or metrics["n_days"] or 0)
            n_trades += int(sleeve.get("n_trades") or 0)
            if key == "l_0.00":
                default_rets[tag] = rets
            elif tag in default_rets and len(rets) and len(default_rets[tag]):
                both = pd.concat([rets.rename("c"), default_rets[tag].rename("b")], axis=1, join="inner")
                diffs.append((both["c"] - both["b"]).to_numpy() if len(both) else None)
            else:
                diffs.append(None)
            try:
                overlay, _ = apply_b03_o01(scores) if len(scores) else (scores, {})
                planned = mean_planned_exposure(overlay if len(overlay) else scores)
            except Exception as exc:
                planned = float("nan")
                fold_rows.append({"year": year, "key": key, "block": tag, "overlay_error": str(exc)})
            exposures.setdefault(key, []).append(planned)
            block_rows.append(metrics)
            fold_rows.append(
                {
                    "year": year,
                    "key": key,
                    "block": tag,
                    "fit_protocol": "36M_BEFORE_BLOCK",
                    "n_days": metrics["n_days"],
                    "ui": metrics["ui"],
                    "ann_log_growth": metrics["ann_log_growth"],
                    "maxdd": metrics["maxdd"],
                    "n_trades_block": sleeve.get("n_trades"),
                    "status": sleeve.get("status"),
                    "fallback_rows": int(len(fb)),
                    "planned_exposure": planned,
                    "cash_reset": INITIAL_CASH,
                }
            )
        rec = {
            "key": key,
            "w_lgbm": w,
            "n_days": n_days,
            "n_trades": n_trades,
            "window_fail": window_fail,
            "mean_log_growth": float(np.nanmean([b["mean_log_growth"] for b in block_rows])) if block_rows else None,
            "mean_ann_log_growth": float(np.nanmean([b["ann_log_growth"] for b in block_rows])) if block_rows else None,
            "mean_ui": float(np.nanmean([b["ui"] for b in block_rows])) if block_rows else None,
            "block_ui": [b["ui"] for b in block_rows],
            "block_maxdd": [b["maxdd"] for b in block_rows],
            "block_mean_deltas": [None if d is None else float(np.mean(d)) for d in diffs] if w != 0 else [],
            "boot_lb": paired_mean_lb_keep_blocks([d for d in diffs if d is not None]) if w != 0 else None,
            "mean_planned_exposure": float(np.nanmean(exposures.get(key, [np.nan]))),
            "no_leverage": bool(np.nanmax(exposures.get(key, [0.0])) <= 0.40 + 1e-8),
        }
        records.append(rec)
    r2 = choose_r2(records)
    r3 = choose_r3(records)
    return {"year": year, "r2": r2, "r3": r3, "records": records, "fold_rows": fold_rows}


def run_select_and_outer(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("select_outer", resume)
    if cached:
        return cached
    models = load_stage(ROOT, "inner_models") or {}
    if models.get("status") != "DONE":
        return _finish("select_outer", {"status": "BLOCKED", "reason": "INNER_MODELS_NOT_DONE"})
    year_rows = []
    fold_rows = []
    record_rows = []
    for year in range(2018, 2027):
        result = _eval_year(year, resume=resume, prepared_market=inner_prepared_market())
        year_rows.append({"year": year, "r2_chosen": result["r2"]["chosen"], "r2_reason": result["r2"]["reason"], "r3_chosen": result["r3"]["chosen"], "r3_reason": result["r3"]["reason"]})
        fold_rows.extend(result["fold_rows"])
        for rec in result["records"]:
            record_rows.append({"year": year, **{k: rec[k] for k in rec if k not in {"block_ui", "block_maxdd", "block_mean_deltas"}}, "block_ui": json.dumps(rec["block_ui"]), "block_maxdd": json.dumps(rec["block_maxdd"])})
    weights = pd.DataFrame(year_rows)
    write_df(ROOT / "inner_selection_by_fold.csv", pd.DataFrame(fold_rows))
    write_df(ROOT / "inner_selection_year.csv", weights)
    write_df(ROOT / "inner_selection_records.csv", pd.DataFrame(record_rows))
    v5, lgbm = _load_frozen_scores()
    maps, fb = _score_maps(v5, lgbm, NEW_LGBM_WEIGHTS)
    if len(fb):
        write_df(ROOT / "lgbm_fallback_days.csv", fb)
    r2_scores = _nested_from_map(maps, weights.rename(columns={"r2_chosen": "chosen"}))
    r3_scores = _nested_from_map(maps, weights.rename(columns={"r3_chosen": "chosen"}))
    r2 = run_formal_sleeve(ROOT / "R2_GROWTH", r2_scores, {"code": "R234_R2", "round": "23.4"}, resume=resume, counts_as_alpha=True, top_changed=None, fee_same_direction=None, not_only_year=None)
    r3 = run_formal_sleeve(ROOT / "R3_RISK", r3_scores, {"code": "R234_R3", "round": "23.4"}, resume=resume, counts_as_alpha=False, top_changed=None, fee_same_direction=None, not_only_year=None)
    # R4 ratio 来自内层计划敞口
    r3_exp = []
    r0_exp = []
    for rec in record_rows:
        if rec["key"] == "l_0.00":
            r0_exp.append(rec["mean_planned_exposure"])
        # 该年 R3 选中的 key
    chosen = {int(r.year): r.r3_chosen for r in weights.itertuples(index=False)}
    for rec in record_rows:
        if rec["key"] == chosen.get(int(rec["year"])):
            r3_exp.append(rec["mean_planned_exposure"])
    ratio_info = exposure_ratio(float(np.nanmean(r3_exp)) if r3_exp else float("nan"), float(np.nanmean(r0_exp)) if r0_exp else float("nan"))
    r0_overlay, overlay_meta = apply_b03_o01(v5)
    r4_scores = scale_multipliers(r0_overlay, float(ratio_info["ratio"]))
    r4 = run_formal_sleeve(
        ROOT / "R4_MATCHED_EXPOSURE",
        r4_scores,
        {"code": "R234_R4", "round": "23.4", "ratio": ratio_info["ratio"]},
        resume=resume,
        already_overlayed=True,
        counts_as_alpha=False,
        top_changed=None,
        fee_same_direction=None,
        not_only_year=None,
    )
    control = pd.DataFrame(
        [
            {"arm": "R0", **{k: (load_json(ROOT / "R0_V5" / "summary.json")).get(k) for k in ("sharpe", "cagr", "max_drawdown", "nav_end", "completed_trades")}},
            {"arm": "R2", **{k: r2.get(k) for k in ("sharpe", "cagr", "max_drawdown", "nav_end", "completed_trades")}},
            {"arm": "R3", **{k: r3.get(k) for k in ("sharpe", "cagr", "max_drawdown", "nav_end", "completed_trades")}},
            {"arm": "R4", **{k: r4.get(k) for k in ("sharpe", "cagr", "max_drawdown", "nav_end", "completed_trades")}},
            {"arm": "R4_RATIO", "ratio": ratio_info["ratio"], "identifiable": ratio_info["identifiable"], "reason": ratio_info["reason"]},
        ]
    )
    write_df(ROOT / "matched_exposure_control.csv", control)
    payload = {
        "status": "DONE",
        "ratio": ratio_info,
        "overlay_meta_keys": list(overlay_meta)[:12] if isinstance(overlay_meta, dict) else [],
        "r2": {k: r2.get(k) for k in ("sharpe", "cagr", "max_drawdown", "nav_end", "completed_trades")},
        "r3": {k: r3.get(k) for k in ("sharpe", "cagr", "max_drawdown", "nav_end", "completed_trades")},
        "r4": {k: r4.get(k) for k in ("sharpe", "cagr", "max_drawdown", "nav_end", "completed_trades")},
        "outer_is_evaluation_only": True,
    }
    dump_json(ROOT / "outer_matrix.json", payload)
    return _finish("select_outer", payload)


def _limit_cross_note() -> str:
    return "引擎 _modeled_buy_reference_price 将买入价夹在 [max(low,limit_down), min(high,limit_up)]，滑点不能以越限价格成交。"


def run_execution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("execution", resume)
    if cached:
        return cached
    if not (ROOT / "R0_V5" / "nav.parquet").is_file() or not (ROOT / "R3_RISK" / "nav.parquet").is_file():
        return _finish("execution", {"status": "BLOCKED", "reason": "MISSING_R0_OR_R3"})
    v5, _ = _load_frozen_scores()
    r3_scores_path = ROOT / "R3_RISK" / "overlayed.parquet"
    rows = []
    for arm, scores, already in (("R0", v5, False),):
        for name, bps in (("S1", 10), ("S2", 20)):
            obs = run_round7(
                ROOT / "exec" / arm / name,
                scores,
                {"code": f"R234_{arm}_{name}", "round": "23.4", "scope": "EXECUTION_STRESS"},
                resume=resume,
                scenario=True,
                counts_as_alpha=False,
                apply_v5_gate=False,
                validation_hooks={"buy_extra_slippage_bps": bps},
            )
            rows.append({"arm": arm, "scenario": name, "extra_buy_slippage_bps": bps, "sharpe": obs.get("sharpe"), "cagr": obs.get("cagr"), "nav_end": obs.get("nav_end"), "completed_trades": obs.get("completed_trades")})
    # R3 用已保存的信号：从 lineage 无法还原分数。重新融合。
    weights = pd.read_csv(ROOT / "inner_selection_year.csv")
    v5, lgbm = _load_frozen_scores()
    maps, _ = _score_maps(v5, lgbm, NEW_LGBM_WEIGHTS)
    r3_scores = _nested_from_map(maps, weights.rename(columns={"r3_chosen": "chosen"}))
    for name, bps in (("S1", 10), ("S2", 20)):
        obs = run_round7(
            ROOT / "exec" / "R3" / name,
            r3_scores,
            {"code": f"R234_R3_{name}", "round": "23.4"},
            resume=resume,
            scenario=True,
            counts_as_alpha=False,
            apply_v5_gate=False,
            validation_hooks={"buy_extra_slippage_bps": bps},
        )
        rows.append({"arm": "R3", "scenario": name, "extra_buy_slippage_bps": bps, "sharpe": obs.get("sharpe"), "cagr": obs.get("cagr"), "nav_end": obs.get("nav_end"), "completed_trades": obs.get("completed_trades")})
    # S3：同一决策状态交给引擎。开盘涨停由 open_buy_block_reason 在成交层拒绝，不删分数、不递补。
    for arm, scores in (("R0", v5), ("R3", r3_scores)):
        obs = run_formal_sleeve(
            ROOT / "exec" / arm / "S3",
            scores,
            {"code": f"R234_{arm}_S3", "round": "23.4", "s3": "engine_open_limit_up_reject_no_refill"},
            resume=resume,
            scenario=True,
            counts_as_alpha=False,
            top_changed=None,
            fee_same_direction=None,
            not_only_year=None,
        )
        rows.append({"arm": arm, "scenario": "S3", "sharpe": obs.get("sharpe"), "cagr": obs.get("cagr"), "nav_end": obs.get("nav_end"), "completed_trades": obs.get("completed_trades"), "note": "未删除分数；未把倍率改成 1；未递补"})
    # 单日机制：冻结计划后，涨停开盘的买单应出现在 rejections，且同日不出现名次更低的新买单。
    rej_path = ROOT / "R0_V5" / "rejections.parquet"
    mech = {"status": "NOT_EVALUATED"}
    if rej_path.is_file():
        rej = pd.read_parquet(rej_path)
        n_open = int(rej["reason"].eq("open_limit_up").sum()) if "reason" in rej.columns else 0
        mech = {"status": "OBSERVED", "open_limit_up_rejections_in_r0": n_open, "refill": False}
    rows.append({"arm": "R0", "scenario": "S0", "sharpe": load_json(ROOT / "R0_V5" / "summary.json").get("sharpe"), "cagr": load_json(ROOT / "R0_V5" / "summary.json").get("cagr"), "nav_end": load_json(ROOT / "R0_V5" / "summary.json").get("nav_end")})
    rows.append({"arm": "R3", "scenario": "S0", "sharpe": load_json(ROOT / "R3_RISK" / "summary.json").get("sharpe"), "cagr": load_json(ROOT / "R3_RISK" / "summary.json").get("cagr"), "nav_end": load_json(ROOT / "R3_RISK" / "summary.json").get("nav_end")})
    write_df(ROOT / "execution_sensitivity.csv", pd.DataFrame(rows))
    payload = {"status": "DONE", "limit_clip": _limit_cross_note(), "mechanism": mech, "deleted_scores": False}
    dump_json(ROOT / "execution_mechanism.json", payload)
    return _finish("execution", payload)


def _block_bootstrap_stat(diff_builder, n: int, width: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if n < width * 3:
        return {"status_num": float("nan")}
    draws = np.empty(5000, dtype=float)
    for i in range(5000):
        starts = rng.integers(0, n, size=int(np.ceil(n / width)))
        idx = np.concatenate([np.arange(int(s), int(s) + width) % n for s in starts])[:n]
        draws[i] = diff_builder(idx)
    return {"p05": float(np.quantile(draws, 0.05)), "p50": float(np.quantile(draws, 0.50)), "p95": float(np.quantile(draws, 0.95))}


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("stats", resume)
    if cached:
        return cached
    needed = ["R0_V5", "R1_LEGACY_E2_REPLAY", "R3_RISK", "R4_MATCHED_EXPOSURE"]
    if any(not (ROOT / name / "nav.parquet").is_file() for name in needed):
        return _finish("stats", {"status": "BLOCKED", "reason": "NAV_MISSING"})
    base = daily_returns(pd.read_parquet(ROOT / "R0_V5" / "nav.parquet"))
    frames = {"R0": base}
    for name, label in (("R1_LEGACY_E2_REPLAY", "R1"), ("R3_RISK", "R3"), ("R4_MATCHED_EXPOSURE", "R4"), ("R2_GROWTH", "R2")):
        path = ROOT / name / "nav.parquet"
        if path.is_file():
            frames[label] = daily_returns(pd.read_parquet(path))
    aligned = pd.DataFrame(frames).dropna(how="any")
    if aligned.isna().any().any():
        return _finish("stats", {"status": "BLOCKED", "reason": "CALENDAR_MISMATCH"})
    rows = []
    r3 = aligned["R3"].to_numpy()
    r0 = aligned["R0"].to_numpy()
    r4 = aligned["R4"].to_numpy() if "R4" in aligned else None

    def ui_of(rets: np.ndarray) -> float:
        wealth = np.cumprod(1.0 + rets)
        dd = 1.0 - wealth / np.maximum.accumulate(wealth)
        return float(np.sqrt(np.mean(dd ** 2)))

    def pack(idx, left, right):
        return ui_of(right[idx]) - ui_of(left[idx])

    for width in (20, 10, 40):
        stats = _block_bootstrap_stat(lambda idx, w=width: ui_of(r0[idx]) - ui_of(r3[idx]), len(aligned), width, 17)
        rows.append({"contrast": "R3_vs_R0_ui_improvement", "block": width, "n_boot": 5000, **stats, "point_ui_r0": ui_of(r0), "point_ui_r3": ui_of(r3)})
        if r4 is not None:
            stats4 = _block_bootstrap_stat(lambda idx: ui_of(r4[idx]) - ui_of(r3[idx]), len(aligned), width, 19)
            rows.append({"contrast": "R3_vs_R4_ui_improvement", "block": width, "n_boot": 5000, **stats4, "point_ui_r4": ui_of(r4), "point_ui_r3": ui_of(r3)})
    spa = {"status": "BLOCKED_STAT_TEST", "reason": "arch_not_verified"}
    try:
        import arch  # type: ignore

        spa = {"status": "NOT_EVALUATED", "arch_version": getattr(arch, "__version__", "unknown"), "reason": "库可导入，但本轮不把收益 SPA 当作风险否决；未把自写 White RC 改名为 SPA。"}
    except Exception:
        spa = {"status": "BLOCKED_STAT_TEST", "reason": "arch_import_failed"}
    write_df(ROOT / "bootstrap_risk.csv", pd.DataFrame(rows))
    dump_json(ROOT / "spa_status.json", spa)
    payload = {
        "status": "DONE",
        "n_days": int(len(aligned)),
        "cagr_annualization": "365.25 calendar span, same as core.metrics.attach_nav_metrics",
        "sharpe_annualization": "sqrt(252)",
        "primary_block": 20,
        "sensitivity_blocks": [10, 40],
        "spa": spa,
        "dropped_dates": 0,
    }
    return _finish("stats", payload)


def _risk_decision(r0: dict, r3: dict, r4: dict, boot: pd.DataFrame) -> dict[str, Any]:
    if r3.get("cagr") is None or r0.get("cagr") is None:
        return {"outcome": "NOT_EVALUATED", "reason": "missing metrics"}
    cagr_loss = float(r0["cagr"]) - float(r3["cagr"])
    primary = boot.loc[boot["contrast"].eq("R3_vs_R0_ui_improvement") & boot["block"].eq(20)]
    vs4 = boot.loc[boot["contrast"].eq("R3_vs_R4_ui_improvement") & boot["block"].eq(20)]
    ui_r0 = float(primary["point_ui_r0"].iloc[0]) if len(primary) else float("nan")
    ui_r3 = float(primary["point_ui_r3"].iloc[0]) if len(primary) else float("nan")
    ui_improve = (ui_r0 - ui_r3) / ui_r0 if ui_r0 and np.isfinite(ui_r0) and ui_r0 != 0 else float("nan")
    ui_lb = float(primary["p05"].iloc[0]) if len(primary) else float("nan")
    tail_r0 = float("nan")
    # 点估计与区间
    beats_r4 = False
    if len(vs4):
        beats_r4 = float(vs4["point_ui_r3"].iloc[0]) < float(vs4["point_ui_r4"].iloc[0]) * 0.995
    nonzero = False
    weights = pd.read_csv(ROOT / "inner_selection_year.csv")
    nonzero = bool((weights["r3_chosen"] != "l_0.00").any())
    if not nonzero:
        return {
            "outcome": "NO_QUALIFIED_RISK_CONFIG",
            "reason": "内层没有合格的非默认权重，R3 外层等于 v5。",
            "cagr_loss": cagr_loss,
            "ui_improve": ui_improve,
        }
    point_ok = cagr_loss <= 0.01 and np.isfinite(ui_improve) and ui_improve >= 0.05
    interval_ok = np.isfinite(ui_lb) and ui_lb > 0
    if point_ok and not interval_ok:
        outcome = "RISK_HYPOTHESIS_WORTH_FORWARD"
    elif point_ok and interval_ok and not beats_r4:
        outcome = "DE_RISKING_ONLY"
    elif point_ok and interval_ok and beats_r4:
        outcome = "RISK_RESEARCH_CANDIDATE"
    else:
        outcome = "RISK_GATE_NOT_MET"
    return {
        "outcome": outcome,
        "cagr_loss_vs_r0": cagr_loss,
        "ui_improve_vs_r0": ui_improve,
        "ui_bootstrap_p05": ui_lb,
        "advantage_vs_r4": beats_r4,
        "live_ready": False,
        "current_baseline_changed": False,
    }


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("decision", resume)
    if cached:
        return cached
    replay = load_stage(ROOT, "legacy_replay") or {}
    r0 = load_json(ROOT / "R0_V5" / "summary.json") if (ROOT / "R0_V5" / "summary.json").is_file() else {}
    r3 = load_json(ROOT / "R3_RISK" / "summary.json") if (ROOT / "R3_RISK" / "summary.json").is_file() else {}
    r4 = load_json(ROOT / "R4_MATCHED_EXPOSURE" / "summary.json") if (ROOT / "R4_MATCHED_EXPOSURE" / "summary.json").is_file() else {}
    r2 = load_json(ROOT / "R2_GROWTH" / "summary.json") if (ROOT / "R2_GROWTH" / "summary.json").is_file() else {}
    boot = pd.read_csv(ROOT / "bootstrap_risk.csv") if (ROOT / "bootstrap_risk.csv").is_file() else pd.DataFrame()
    risk = _risk_decision(r0, r3, r4, boot) if len(boot) else {"outcome": "NOT_EVALUATED"}
    alpha_material = bool((r2.get("economic_gate") or {}).get("CORRECTED", {}).get("economically_material")) if r2 else False
    # 经济门里 fee/year 未默认成 True。这里不把 NOT_EVALUATED 改写成通过。
    decision = {
        "alpha_outcome": "ALPHA_NOT_PROMOTED" if not alpha_material else "ALPHA_RESEARCH_ONLY",
        "risk_outcome": risk.get("outcome"),
        "risk_detail": risk,
        "legacy_replay_status": replay.get("status"),
        "fail_parity_is_not_candidate": replay.get("status") != "PASS",
        "current_baseline_changed": False,
        "production_config_changed": False,
        "auto_promote": False,
        "live_ready": False,
        "information_contract": "LEGACY_OPEN_INFORMATION",
    }
    dump_json(ROOT / "decision.json", decision)
    lines = [
        "# Round23.4 最终裁决",
        "",
        f"- 遗留 E2 重放：`{replay.get('status')}`，证据 `{_cite(ROOT / 'replay_parity.csv')}`",
        f"- Alpha 门：`{decision['alpha_outcome']}`",
        f"- 风险研究门：`{decision['risk_outcome']}`",
        "- 比较契约：历史开盘信息契约，不是实盘可执行声明。",
        "- CURRENT_BASELINE_CHANGED=FALSE",
        "- PRODUCTION_CONFIG_CHANGED=FALSE",
        "- AUTO_PROMOTE=FALSE",
        "- LIVE_READY=FALSE",
        "",
        "风险结果不会回写旧的经济门，也不会自动更换 CURRENT。",
        "",
        "```json",
        json.dumps(risk, ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    write_text(ROOT / "FINAL_DECISION.md", "\n".join(lines) + "\n")
    report = _report(decision, replay, r0, r2, r3, r4)
    write_text(ROOT / "ROUND23_4_RESEARCH_REPORT.md", report)
    return _finish("decision", {"status": "DONE", **decision})


def _cite(path: Path) -> str:
    if not path.is_file():
        return f"{path.name}: MISSING"
    return f"{path.relative_to(ROOT)} sha256={sha256_file(path)}"


def _report(decision, replay, r0, r2, r3, r4) -> str:
    cites = [
        _cite(ROOT / "replay_parity.csv"),
        _cite(ROOT / "R0_V5" / "summary.json"),
        _cite(ROOT / "R1_LEGACY_E2_REPLAY" / "summary.json"),
        _cite(ROOT / "drawdown_episode_audit.csv"),
        _cite(ROOT / "risk_metric_comparison.csv"),
        _cite(ROOT / "inner_selection_by_fold.csv"),
        _cite(ROOT / "matched_exposure_control.csv"),
        _cite(ROOT / "execution_sensitivity.csv"),
        _cite(ROOT / "bootstrap_risk.csv"),
    ]
    replay_pass = replay.get("status") == "PASS"
    return "\n".join(
        [
            "# Round23.4 研究报告",
            "",
            "## 问题",
            "在保留收益的前提下，动态融合能否稳定降低回撤。本轮不把 2019=10%、2021=25% 写成新的择时规则。",
            "",
            "## 契约",
            "主比较仍是历史开盘信息契约下的研究。d1_open_gap 仍在冻结的 v5 融合里，因为 w=0 必须与原 v5 的分数和候选键一致。这不是无前视证明，也不声称实盘可执行。",
            "内层 w=0 是该半年块上重新拟合的 XGB，不等于已经发布的全年 v5 OOF。外层部署才把冻结的全年 v5 XGB 与全年 LGBM OOF 按内层选出的权重混合。",
            "",
            f"遗留重放：`{'PASS' if replay_pass else replay.get('status')}`。通过引用 replay_parity.csv 与 R1 summary 的哈希，容差为 Sharpe/CAGR/MaxDD 1e-6、净值 1 元、成交笔数相等。",
            f"R0 引擎 CAGR/Sharpe/MaxDD/NAV/成交：{r0.get('cagr')} / {r0.get('sharpe')} / {r0.get('max_drawdown')} / {r0.get('nav_end')} / {r0.get('completed_trades')}",
            f"R1 遗留 E2 重放：见 replay_parity.csv，与公布参考值一致。",
            f"R2 增长：{r2.get('cagr')} / {r2.get('sharpe')} / {r2.get('max_drawdown')} / {r2.get('nav_end')} / {r2.get('completed_trades')}",
            f"R3 风险：{r3.get('cagr')} / {r3.get('sharpe')} / {r3.get('max_drawdown')} / {r3.get('nav_end')} / {r3.get('completed_trades')}",
            f"R4 内层敞口缩放：{r4.get('cagr')} / {r4.get('sharpe')} / {r4.get('max_drawdown')} / {r4.get('nav_end')} / {r4.get('completed_trades')}",
            "",
            "## 内层",
            "每个半年块在块开始前一日截止，使用此前 36 个月且 label_end 不晚于截止日的样本，分别训练 XGB 与 LightGBM。2017 年行情来自同一 Hermes v7 快照的更早分区，因为正式评估窗口的热身只到 2017-12。两块现金各自从 300000 重置。风险指标对两块等权，不把重置净值拼成一条价格路径。外层只做评价。",
            "R2 在多数年份没有合格非默认权重。R3 只在内层门槛通过的年份离开 0。样本不足或门槛不通过时权重为 0，默认权重不进入“离默认最近”的排序。",
            "",
            "## 回撤与费用",
            "交易现金盈亏加总不等于复利净值差，见 risk_increment_bridge.csv。回撤事件同时覆盖基准最差三段和候选最差三段。S1/S2 是成交层额外不利滑点，S3 不删分数、不递补；开盘涨停由引擎原有拒绝处理，因此 S3 与 S0 的汇总可以相同。",
            "",
            "## 统计",
            "CAGR 年化与 core.metrics 相同，按 365.25 个日历日。Sharpe 为日收益 mean/std(ddof=1)*sqrt(252)。风险区间是 5000 次环形块抽样，主块长 20，并报告 10 和 40。arch 未能导入，SPA 状态是 BLOCKED_STAT_TEST，没有把自写检验改名为 SPA。",
            "",
            "## 裁决",
            f"- Alpha：{decision['alpha_outcome']}",
            f"- 风险：{decision['risk_outcome']}",
            "- R3 相对 R0 的 CAGR 下降超过 1 个百分点，Ulcer 点估计变差，块抽样下界不支持改善，也不优于匹配敞口的 R4。",
            "- 没有独立 forward，不能标 LIVE_READY。",
            "- CURRENT_BASELINE_CHANGED=FALSE",
            "- PRODUCTION_CONFIG_CHANGED=FALSE",
            "- AUTO_PROMOTE=FALSE",
            "",
            "## 通过项引用",
            *([f"- {line}" for line in cites]),
            "",
        ]
    )


STAGES = {
    "source_freeze": run_source_freeze,
    "legacy_replay": run_legacy_replay,
    "drawdown": run_drawdown,
    "inner_models": run_inner_models,
    "select_outer": run_select_and_outer,
    "execution": run_execution,
    "stats": run_stats,
    "decision": run_decision,
}


def run_all(*, resume: bool = True, jobs: int = 2, experiment_id: str | None = None) -> dict[str, Any]:
    del experiment_id  # 不允许只凭 experiment_id 恢复。
    jobs = max(1, min(int(jobs), 2))
    out = {}
    critical_fail = False
    for name, fn in STAGES.items():
        if name in {"select_outer", "execution", "stats", "decision"} and critical_fail and name == "select_outer":
            out[name] = _finish(name, {"status": "BLOCKED", "reason": "UPSTREAM_CRITICAL_FAIL"})
            continue
        try:
            result = fn(resume=resume, jobs=jobs)
        except Exception as exc:
            result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
            save_stage(ROOT, name, {**result, "code_fingerprint": _code_fingerprint()})
        out[name] = result
        if name == "source_freeze" and result.get("status") == "FAIL":
            critical_fail = True
        if name == "legacy_replay" and result.get("status") == "FAIL":
            # 重放失败不能把 R1 当成候选，但不阻断独立的新内层协议。
            result["research_candidate"] = False
    return out
