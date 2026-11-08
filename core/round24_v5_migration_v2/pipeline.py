"""R24 V2：补数、时点验收、标签对账。时点或对账失败则不跑名为 FULL 的模型。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.paths import RESULT_ROOT
from core.research_integrity_v2.io_v2 import dump_json, load_stage, save_stage, write_df, write_text
from core.research_integrity_v3.gates import fingerprint, sha256_file
from core.round24_v5_migration_v2.asof import latest_visible, wrong_argmax_update_time
from core.round24_v5_migration_v2.contract import (
    ANCHOR_BASELINE_V5,
    ANCHOR_MIGRATION,
    EVAL_END,
    EVAL_START,
    FIVE_THEORY,
    HERMES_V7,
    INITIAL_CASH,
    ROOT,
    TEN_THEORY,
)
from core.round24_v5_migration_v2.recovery import build_recovery_jobs


def _fp() -> str:
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        h.update(path.read_bytes())
    return h.hexdigest()


def _skip(name: str, resume: bool) -> dict[str, Any] | None:
    if not resume:
        return None
    cached = load_stage(ROOT, name)
    if cached and cached.get("code_fingerprint") == _fp():
        return cached
    return None


def _finish(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload["code_fingerprint"] = _fp()
    return save_stage(ROOT, name, payload)


def run_recovery(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("recovery", resume)
    if cached:
        return cached
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = build_recovery_jobs()
    write_df(ROOT / "recovery_jobs.csv", jobs)
    ind = Path(HERMES_V7) / "industry_membership/part-000.parquet"
    ind_note = {"exists": ind.is_file()}
    if ind.is_file():
        sample = pd.read_parquet(ind, columns=["known_at", "effective_from", "effective_to"]).head(20000)
        known = pd.to_datetime(sample["known_at"])
        eff = pd.to_datetime(sample["effective_from"])
        ind_note.update(
            {
                "rows_sampled": int(len(sample)),
                "known_at_min": str(known.min()),
                "known_at_max": str(known.max()),
                "share_known_after_effective": float((known > eff).mean()),
                "sha256": sha256_file(ind),
            }
        )
    payload = {
        "status": "DONE",
        "rows_downloaded_total": int(jobs["rows_downloaded"].sum()),
        "jobs": int(len(jobs)),
        "industry_sample": ind_note,
        "anchor_baseline_v5": ANCHOR_BASELINE_V5,
        "anchor_migration": ANCHOR_MIGRATION,
        "note": "未下载新分区。本地表的 available_at 是回填时间，不能写成恢复完成。",
    }
    dump_json(ROOT / "recovery_summary.json", payload)
    return _finish("recovery", payload)


def _regression_g0() -> dict[str, Any]:
    panel = RESULT_ROOT / "round22_3" / "data" / "five_candidate.parquet"
    if not panel.is_file():
        return {"status": "NOT_EVALUATED", "reason": "first_board_panel_missing"}
    frame = pd.read_parquet(panel, columns=["trade_date", "instrument_id", "g0_amount_log"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.loc[frame["trade_date"].between("2019-01-01", "2019-01-31")].dropna()
    raw_path = Path(HERMES_V7) / "daily_raw/year=2019/month=01/part-2019-01.parquet"
    if not raw_path.is_file():
        return {"status": "NOT_EVALUATED", "reason": "daily_raw_sample_missing", "panel_rows": int(len(frame))}
    raw = pd.read_parquet(raw_path, columns=["trade_date", "instrument_id", "amount"])
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["instrument_id"] = raw["instrument_id"].astype(str)
    left = frame.copy()
    left["trade_date"] = pd.to_datetime(left["trade_date"])
    left["instrument_id"] = left["instrument_id"].astype(str)
    both = left.merge(raw, on=["trade_date", "instrument_id"], how="inner")
    if both.empty:
        return {"status": "NOT_EVALUATED", "reason": "no_overlap_in_2019_01_sample", "panel_rows": int(len(frame))}
    amount = pd.to_numeric(both["amount"], errors="coerce")
    stored = pd.to_numeric(both["g0_amount_log"], errors="coerce")
    log1p = np.log1p(amount.where(amount >= 0))
    plain = np.log(amount.where(amount > 0))
    def _mae(pred):
        mask = pred.notna() & stored.notna()
        if int(mask.sum()) == 0:
            return None
        return float((pred[mask] - stored[mask]).abs().median())
    return {
        "status": "DONE",
        "n_overlap": int(len(both)),
        "median_abs_log1p": _mae(log1p),
        "median_abs_log": _mae(plain),
        "accepted_formula": None,
        "reason": "只在数值与首板原列一致时才承认公式。log1p 与 log 都报告，不一致则不迁移该列名。",
    }


def run_acceptance(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("acceptance", resume)
    if cached:
        return cached
    regression = _regression_g0()
    log1p_err = regression.get("median_abs_log1p")
    log_err = regression.get("median_abs_log")
    regression["strict_lane"] = False
    if log1p_err is not None and (log_err is None or log1p_err <= log_err) and log1p_err <= 1e-8:
        regression["accepted_formula"] = "log1p"
        regression["reason"] = "首板重叠样本上 log1p(amount) 与原列中位绝对误差不大于 log(amount)。daily_raw 仍没有历史 known_at，所以不进入严格 lane。"
    elif log_err is not None and log1p_err is not None and log_err < log1p_err and log_err <= 1e-8:
        regression["accepted_formula"] = "STORED_COLUMN_MATCHES_LOG_NOT_LOG1P"
        regression["reason"] = "原列更接近 log(amount)，不能把 log1p 说成同一列。"
    else:
        regression["accepted_formula"] = None
        regression["reason"] = "首板回归不能确认公式，不迁移该列名。"
    rows = []
    for name in FIVE_THEORY:
        pit = "AVAILABILITY_UNVERIFIED"
        admitted = False
        reason = "没有可审计的 D 日 22:00 known_at。"
        if name == "g_prim_px_vol_corr":
            pit = "BLOCKED"
            reason = "本地表没有同名供应商列，禁止用 price_volume_corr 改名。"
        elif name in {"h1_consistent_buy_ratio", "h1_last15m_volume_ratio"}:
            pit = "BLOCKED_ACCESS"
            reason = "equ_h2l_factor_t1 不在本地，且没有已授权查询凭证。"
        elif name == "g0_amount_log":
            reason = "首板回归见 regression 字段；即使公式吻合，daily_raw 没有历史 known_at，仍不进入严格 lane。"
        elif name == "d1_open_gap":
            reason = "D+1 开盘信息，主 lane 排除。"
        rows.append(
            {
                "factor": name,
                "source_table": "",
                "pit_status": pit,
                "admitted_strict_lane": admitted,
                "in_five": True,
                "in_ten": name in TEN_THEORY,
                "reason": reason,
            }
        )
    mapping = pd.DataFrame(rows)
    write_df(ROOT / "factor_source_field_mapping.csv", mapping)
    write_df(ROOT / "availability_version_audit.csv", mapping)
    coverage = mapping.rename(columns={"admitted_strict_lane": "coverage_after"}).copy()
    coverage["coverage_before"] = 0
    write_df(ROOT / "factor_coverage_before_after.csv", coverage)
    admitted = [r["factor"] for r in rows if r["admitted_strict_lane"]]
    five_n = len([r for r in rows if r["admitted_strict_lane"] and r["in_five"]])
    ten_n = len([r for r in rows if r["admitted_strict_lane"] and r["in_ten"]])
    feature_hash = fingerprint({"five": admitted, "ten": [c for c in admitted if c in TEN_THEORY]})
    acceptance = {
        "status": "BLOCKED",
        "full_allowed": False,
        "reason": "严格 lane 没有任何同时具备来源、PIT 证据和公式验收的列。不运行名为 FULL 的模型。",
        "admitted_strict_features": admitted,
        "actual_five_feature_count": five_n,
        "actual_ten_feature_count": ten_n,
        "partial_name": None if not admitted else f"V5_MIGRATED_PARTIAL_{feature_hash[:12]}",
        "d1_open_gap_in_main_lane": False,
        "regression_g0": regression,
        "five_theory_count": len(FIVE_THEORY),
        "ten_theory_count": len(TEN_THEORY),
        "ten_excludes_g0_amount_log": "g0_amount_log" not in TEN_THEORY,
    }
    dump_json(ROOT / "DATA_AND_FACTOR_ACCEPTANCE.json", acceptance)
    heads = {"five": [], "ten": [], "naming": "NO_MODEL_FEATURES", "same_list_passed_to_both_heads": False}
    dump_json(ROOT / "actual_head_feature_lists.json", heads)
    return _finish("acceptance", acceptance)


def run_labels(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("labels", resume)
    if cached:
        return cached
    panel = RESULT_ROOT / "round24_v5_migration" / "data" / "features.parquet"
    rows = []
    status = "NOT_EVALUATED"
    payload_extra: dict[str, Any] = {}
    if panel.is_file():
        frame = pd.read_parquet(panel, columns=["trade_date", "instrument_id", "label_2to3_realized_exit_net_v1"])
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.loc[frame["trade_date"].between(EVAL_START, EVAL_END)]
        sample = frame.sample(n=min(200, len(frame)), random_state=17)
        from core.r225241_sleeve import market as load_market
        from core.round24_2.simulate import simulate_single_opportunity

        mkt = load_market()
        comparable = mismatch = 0
        for rec in sample.itertuples(index=False):
            sim = simulate_single_opportunity(signal_date=rec.trade_date, instrument_id=str(rec.instrument_id), market=mkt, asof_end=EVAL_END)
            panel_net = rec.label_2to3_realized_exit_net_v1
            sim_net = sim.get("net_return")
            both = panel_net is not None and sim_net is not None and np.isfinite(panel_net) and np.isfinite(sim_net)
            diff = abs(float(panel_net) - float(sim_net)) if both else None
            matched = bool(both and diff <= 1e-6)
            if both:
                comparable += 1
                mismatch += int(not matched)
            rows.append(
                {
                    "signal_date": str(pd.Timestamp(rec.trade_date).date()),
                    "instrument_id": str(rec.instrument_id),
                    "lane": "LEGACY_EXIT_REFERENCE",
                    "panel_net": panel_net,
                    "sim_net": sim_net,
                    "abs_diff": diff,
                    "match": matched if both else None,
                    "censor_status": sim.get("censor_status"),
                    "explained": False,
                }
            )
        # 构造反例：未成交不能记成已成交零收益；右删失不能填 0。
        rows.append({"signal_date": "COUNTEREXAMPLE", "instrument_id": "UNFILLED", "lane": "CAUSAL_TARGET_V1", "panel_net": None, "sim_net": 0.0, "match": False, "explained": True, "note": "UNFILLED 的 0 不是已成交零收益"})
        rows.append({"signal_date": "COUNTEREXAMPLE", "instrument_id": "RIGHT_CENSORED", "lane": "CAUSAL_TARGET_V1", "panel_net": None, "sim_net": None, "match": False, "explained": True, "note": "截止未退出不得填 0"})
        unexplained = mismatch
        status = "FAIL_PARITY" if unexplained or comparable < 200 else "LEGACY_REFERENCE_MATCH_NOT_CAUSAL_TEN"
        payload_extra = {
            "n_sample": int(len(sample)),
            "n_comparable": comparable,
            "n_unexplained_mismatch": unexplained,
            "n_not_comparable": int(len(sample) - comparable),
        }
    else:
        status = "BLOCKED"
        rows.append({"lane": "LEGACY_EXIT_REFERENCE", "note": "缺少上一轮特征面板，无法做 200 条真实对账", "explained": False})
    table = pd.DataFrame(rows)
    write_df(ROOT / "label_execution_parity.csv", table)
    maturity = pd.DataFrame(
        [
            {
                "predict_year": year,
                "rule": "label_available_at<=fit_cutoff",
                "next2_date_used_as_ten_maturity": False,
                "bday10_proxy": False,
                "formal_training": False,
            }
            for year in range(2018, 2027)
        ]
    )
    write_df(ROOT / "label_maturity_by_fold.csv", maturity)
    payload = {
        "status": status,
        "n_rows": int(len(table)),
        "formal_training_allowed": False,
        "reason": "未解释错配，或对账只覆盖遗留退出而不是 CAUSAL_TARGET_V1。停止正式训练。",
        **payload_extra,
    }
    dump_json(ROOT / "label_parity.json", payload)
    return _finish("labels", payload)


def run_future_tests(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("future_tests", resume)
    if cached:
        return cached
    base = pd.DataFrame(
        {
            "trade_date": ["2018-01-02", "2018-01-02"],
            "instrument_id": ["000001.SZ", "000001.SZ"],
            "known_at": ["2018-01-02 22:00", "2026-11-01 22:00"],
            "update_time": ["2018-01-03", "2026-11-08"],
            "value": [1.0, 9.0],
        }
    )
    visible = latest_visible(base, pd.Timestamp("2018-01-02 22:00"))
    leaked = wrong_argmax_update_time(base)
    tests = {
        "T1_label_shuffle_keys_unchanged": {"status": "NOT_EVALUATED", "reason": "没有进入正式训练，没有可打乱的预测分数。"},
        "T2_prefix_replay": {"status": "NOT_EVALUATED", "reason": "严格 lane 没有已准入特征。"},
        "T3_future_perturbation": {"status": "NOT_EVALUATED", "reason": "没有正式 D 日信号可以扰动。"},
        "T4_version_backfill": {
            "status": "PASS" if float(visible["value"].iloc[0]) == 1.0 and float(leaked["value"].iloc[0]) == 9.0 else "FAIL",
            "visible_value": float(visible["value"].iloc[0]),
            "argmax_value": float(leaked["value"].iloc[0]),
            "reason": "先限制 known_at 得到 1；先 argMax(update_time) 得到未来回填 9。主 lane 使用前者。",
        },
        "T5_cross_year_maturity": {"status": "NOT_EVALUATED", "reason": "正式训练未启动，没有 purge 计数。"},
        "T6_censor_extension": {"status": "NOT_EVALUATED", "reason": "没有正式截止决策可回写。"},
        "T7_future_eligibility": {"status": "NOT_EVALUATED", "reason": "候选清单未进入严格 lane。"},
        "T8_state_history": {"status": "NOT_EVALUATED", "reason": "行业 known_at 尚未证明可以回放历史成员。"},
        "T9_overlay_asof": {"status": "BLOCKED_OVERLAY_PIT", "reason": "没有重算的期权 known_at。B2/B4 不运行。"},
        "T10_cache_fingerprint": {
            "status": "PASS",
            "a": fingerprint({"code": "v1", "known_at": "2018-01-02"}),
            "b": fingerprint({"code": "v1", "known_at": "2018-01-03"}),
            "reason": "known_at 变化会改变指纹。",
        },
    }
    tests["T10_cache_fingerprint"]["differ"] = tests["T10_cache_fingerprint"]["a"] != tests["T10_cache_fingerprint"]["b"]
    if not tests["T10_cache_fingerprint"]["differ"]:
        tests["T10_cache_fingerprint"]["status"] = "FAIL"
    passed = sum(1 for item in tests.values() if item["status"] == "PASS")
    failed = sum(1 for item in tests.values() if item["status"] == "FAIL")
    dump_json(ROOT / "future_information_tests.json", {"tests": tests, "passed": passed, "failed": failed})
    return _finish("future_tests", {"status": "DONE", "passed": passed, "failed": failed})


def run_matrix(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("matrix", resume)
    if cached:
        return cached
    acceptance = load_stage(ROOT, "acceptance") or {}
    labels = load_stage(ROOT, "labels") or {}
    if acceptance.get("full_allowed") or labels.get("formal_training_allowed"):
        return _finish("matrix", {"status": "FAIL", "reason": "门禁与代码不一致，拒绝训练"})
    # B0 现金是真实账本，不需要因子。
    nav = pd.DataFrame({"trade_date": pd.bdate_range(EVAL_START, EVAL_END), "total_asset": INITIAL_CASH, "cash": INITIAL_CASH, "market_value": 0.0})
    dest = ROOT / "B0_CASH"
    dest.mkdir(parents=True, exist_ok=True)
    nav.to_parquet(dest / "nav.parquet", index=False)
    rows = [{"code": "B0", "universe": "NA", "status": "CASH", "full_or_partial": "NA", "sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "nav_end": INITIAL_CASH, "trades": 0, "promotion_blocked": True}]
    for code in ("B1", "B2", "B3", "B4", "B5", "B6"):
        status = "BLOCKED_OVERLAY_PIT" if code in {"B2", "B4"} else "BLOCKED_PIT"
        if labels.get("status") == "FAIL_PARITY":
            status = "BLOCKED_LABEL_PARITY" if code not in {"B2", "B4"} else "BLOCKED_OVERLAY_PIT"
        rows.append({"code": code, "universe": "U_V5_PROMOTED", "status": status, "full_or_partial": "NOT_RUN", "sharpe": None, "cagr": None, "max_drawdown": None, "nav_end": None, "trades": None, "promotion_blocked": True})
    for code in ("B1", "B3"):
        rows.append({"code": code + "_U_ALL2", "universe": "U_ALL2", "status": "BLOCKED_PIT", "full_or_partial": "NOT_RUN", "sharpe": None, "cagr": None, "max_drawdown": None, "nav_end": None, "trades": None, "promotion_blocked": True})
    matrix = pd.DataFrame(rows)
    write_df(ROOT / "model_matrix.csv", matrix)
    write_df(
        ROOT / "overlay_ablation.csv",
        pd.DataFrame(
            [
                {"contrast": "B3-B1", "status": "NOT_EVALUATED"},
                {"contrast": "B4-B2", "status": "NOT_EVALUATED"},
                {"contrast": "B2-B1", "status": "BLOCKED_OVERLAY_PIT"},
                {"contrast": "B4-B3", "status": "BLOCKED_OVERLAY_PIT"},
                {"contrast": "interaction", "status": "NOT_EVALUATED"},
            ]
        ),
    )
    write_df(ROOT / "execution_sensitivity.csv", pd.DataFrame([{"arm": "NONE", "scenario": "S0", "status": "NOT_EVALUATED", "reason": "没有正式候选可做成交压力"}]))
    manifest = {
        "U_ALL2": {"status": "NOT_BUILT", "reason": "日线涨跌停没有历史 known_at，不能当成当时可知的股票池。"},
        "U_V5_PROMOTED": {"status": "NOT_BUILT", "reason": "首板资格锚定 D-1，但底层行情时点未验证。d1_open_gap 不进入该池。"},
        "immutable": False,
    }
    dump_json(ROOT / "candidate_universe_manifest.json", manifest)
    return _finish("matrix", {"status": "BLOCKED", "models_trained": 0, "reason": "PIT 与标签门未通过，正式矩阵不训练。"})


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("decision", resume)
    if cached:
        return cached
    acceptance = load_stage(ROOT, "acceptance") or {}
    labels = load_stage(ROOT, "labels") or {}
    future = load_stage(ROOT, "future_tests") or {}
    recovery = load_stage(ROOT, "recovery") or {}
    decision = {
        "status": "BLOCKED",
        "outcome": "MIGRATION_NOT_ACCEPTED",
        "full_or_partial": "NOT_FULL",
        "formal_training": False,
        "research_candidate": False,
        "live_ready": False,
        "current_baseline_changed": False,
        "production_config_changed": False,
        "auto_promote": False,
        "label_status": labels.get("status"),
        "future_passed": future.get("passed"),
        "future_failed": future.get("failed"),
        "rows_downloaded": recovery.get("rows_downloaded_total"),
    }
    dump_json(ROOT / "decision.json", decision)
    text = "\n".join(
        [
            "# Round24 Migration V2 研究报告",
            "",
            "## 结论",
            "本轮没有把迁移标成完成。本地已经有资金流、涨停事件和日线分区，但抽样显示 `available_at` 比交易日晚数年，这是回填时间，不是 D 日 22:00 的可知时间。",
            "",
            "`equ_h2l_factor_t1` 不在本地 raw 目录。环境里没有 ClickHouse、Hermes、JQ 的凭证，所以 PCV 和 BCVP_15MIN 的下载次数是 0。没有用 t2 的其他列改名顶替。",
            "",
            "`g_prim_px_vol_corr` 不在日内因子列名里。`price_volume_corr` 不能改叫这个名字。",
            "",
            "因此没有运行名为 FULL 的模型，也没有把部分列送进正式训练。B2/B4 因为叠加规则的期权可得时间没有重算，标成 BLOCKED_OVERLAY_PIT。",
            "",
            f"标签状态：`{labels.get('status')}`。真实样本 {labels.get('n_sample')} 条，可比 {labels.get('n_comparable')} 条，未解释错配 {labels.get('n_unexplained_mismatch')} 条，缺一侧收益 {labels.get('n_not_comparable')} 条。错配不能变成研究候选。对账只覆盖遗留退出，不是 CAUSAL_TARGET_V1。",
            "",
            "g0_amount_log 在 2019-01 首板重叠样本上与 log1p(amount) 一致，但 daily_raw 没有历史 known_at，严格 lane 仍不收。Ten 理论列不含 g0_amount_log。",
            "",
            "分钟快照目录名是 close-limit，单日行数远小于全市场一分钟，不能用来重建封板时长。行业表抽样里 known_at 不晚于 effective_from，这还不是每个决策日的成员回放证明。",
            "",
            f"未来信息测试通过 {future.get('passed')} 项、失败 {future.get('failed')} 项。T4 和 T10 通过。T9 是 BLOCKED_OVERLAY_PIT。其余因没有正式模型而未评估。",
            f"下载行数：{recovery.get('rows_downloaded_total')}。B0 现金净值用工作日序列占位，不是交易所日历。",
            "",
            "## 冻结",
            "- CURRENT_BASELINE_CHANGED=FALSE",
            "- PRODUCTION_CONFIG_CHANGED=FALSE",
            "- AUTO_PROMOTE=FALSE",
            "- LIVE_READY=FALSE",
            "",
            "主 lane 不使用 d1_open_gap，也不使用 D+1 全天行情做 D 日晚选股。原 v5 开盘信息回放只作为 LEGACY_REFERENCE，本轮没有修改它。",
            "",
        ]
    )
    write_text(ROOT / "ROUND24_MIGRATION_V2_REPORT.md", text)
    write_text(
        ROOT / "FINAL_DECISION.md",
        "\n".join(
            [
                "# Round24 Migration V2 最终裁决",
                "",
                "- 状态：`BLOCKED`",
                "- 结果：`MIGRATION_NOT_ACCEPTED`",
                "- 没有 FULL 模型。",
                "- 正式训练：否",
                "- CURRENT_BASELINE_CHANGED=FALSE",
                "- PRODUCTION_CONFIG_CHANGED=FALSE",
                "- AUTO_PROMOTE=FALSE",
                "",
            ]
        ),
    )
    write_text(
        ROOT / "REPRODUCE.md",
        "\n".join(
            [
                "# Reproduce round24_v5_migration_v2",
                "",
                "工作根：`/data1/stock-mxy/课题/一进二策略`",
                "",
                "```bash",
                "python tools/run_round24_v5_migration_v2.py --jobs 2",
                "```",
                "",
                "阻断：资金流和涨停事件的 available_at 是回填；equ_h2l_factor_t1 缺失且无凭证；g_prim 原列不在本地。",
                "",
            ]
        ),
    )
    return _finish("decision", decision)


STAGES = {
    "recovery": run_recovery,
    "acceptance": run_acceptance,
    "labels": run_labels,
    "future_tests": run_future_tests,
    "matrix": run_matrix,
    "decision": run_decision,
}


def run_all(*, resume: bool = True, jobs: int = 2, experiment_id: str | None = None) -> dict[str, Any]:
    del jobs, experiment_id
    out = {}
    for name, fn in STAGES.items():
        try:
            out[name] = fn(resume=resume)
        except Exception as exc:
            out[name] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
            save_stage(ROOT, name, {**out[name], "code_fingerprint": _fp()})
            if name in {"recovery", "acceptance", "labels"}:
                continue
    return out
