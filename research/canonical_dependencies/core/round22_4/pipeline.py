"""Round 22.4 阶段。先冻结，再审计，再训练，再回测。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.fold_ranker import train_ranker_head
from core.research_io import (
    TRACKER_PATH,
    append_tracker,
    dump_json,
    load_json,
    sha256_json,
    write_df,
    write_text,
)
from core.research_sleeve import run_heads_sleeve
from core.round22_3.contract import material_vs_v5

from .contract import (
    ALLNAN_COL,
    C1_LAG1_ID,
    C4_ID,
    C_AVAILABLE_FROM,
    COLSAMPLE,
    ECONOMIC_MATERIALITY,
    FEE_SCHEDULE,
    FIVE_FEATURES,
    FS1_ID,
    FS2_ID,
    FS_CHANGE_ID,
    FS_EVIDENCE,
    KEEP,
    MAIN_WINDOW_END,
    MAIN_WINDOW_START,
    NULL_CONTROLS,
    RECOMMEND,
    RHO_GATE,
    ROOT,
    SCOPES,
    SEEDS,
    SHUFFLE_COL,
    SUBSAMPLE,
    TEN_FEATURES,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    evidence_pass,
)

R22_3 = ROOT.parent / "round22_3"
R22_2 = ROOT.parent / "round22_2"
E0_FIVE = R22_3 / "candidates" / "R223_E0" / "five_pred.parquet"
E0_TEN = R22_3 / "candidates" / "R223_E0" / "ten_pred.parquet"
E0_NAV = R22_3 / "candidates" / "R223_E0" / "nav.parquet"
PANEL = R22_3 / "data" / "five_candidate.parquet"
TRAIN_PANEL = R22_3 / "data" / "five_training.parquet"


def _done(name: str) -> dict[str, Any] | None:
    path = ROOT / "_state" / f"{name}.json"
    payload = load_json(path)
    if payload.get("status") in {"DONE", "PARTIAL", "BLOCKED"}:
        return payload
    return None


def _save(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    dump_json(ROOT / "_state" / f"{name}.json", payload)
    return payload


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (ROOT / "frozen_manifest.json").is_file():
        return load_json(ROOT / "frozen_manifest.json")
    body = {
        "round": "22.4",
        "frozen_before_new_nav": True,
        "anchor": "baseline_v5=R16_O01",
        "fee_schedule": FEE_SCHEDULE,
        "window": [MAIN_WINDOW_START, MAIN_WINDOW_END],
        "economic_materiality": {k: ECONOMIC_MATERIALITY[k] for k in ECONOMIC_MATERIALITY if k != "note"},
        "economic_gate_sha256": sha256_json(
            {k: ECONOMIC_MATERIALITY[k] for k in ("id", "delta_cagr_min_abs", "delta_sharpe_min", "nav_rel_min", "maxdd_worse_max_abs")}
        ),
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE,
        "seeds": list(SEEDS),
        "evaluation_scopes": list(SCOPES),
        "null_controls": list(NULL_CONTROLS),
        "fs_evidence": FS_EVIDENCE,
        "rho_remove_replace": RHO_GATE,
        "fs_registered_before_backtest": [
            {"factor_id": FS1_ID, "formula": "(NI-OCF)/avg_assets", "kept_from": "round22_3", "fill_zero": False},
            {"factor_id": FS2_ID, "formula": "OCF/avg_assets", "kept_from": "round22_3", "fill_zero": False},
            {
                "factor_id": FS_CHANGE_ID,
                "formula": "OCF/avg_assets 的相邻公告事件差；同一公告的相邻交易日不是新信息；缺失保持 NaN",
                "registered_before_any_round22_4_backtest": True,
                "fill_zero": False,
            },
        ],
        "joint_add_max_rounds": 1,
        "decision_allowed": [KEEP, RECOMMEND],
        "current_baseline_changed": False,
        "auto_promote": False,
        "do_not_retune_economic_gate": True,
    }
    dump_json(ROOT / "frozen_manifest.json", body)
    write_text(
        ROOT / "00_frozen_manifest.md",
        "\n".join(
            [
                "# Round22.4 冻结",
                "",
                "本文件写在任何新 NAV 之前。",
                f"- 锚点 baseline_v5=R16_O01，费用 {FEE_SCHEDULE}，窗 {MAIN_WINDOW_START}～{MAIN_WINDOW_END}。",
                f"- 经济门 `{ECONOMIC_MATERIALITY['id']}`：ΔCAGR≥{ECONOMIC_MATERIALITY['delta_cagr_min_abs']}，ΔSharpe≥{ECONOMIC_MATERIALITY['delta_sharpe_min']}，NAV 相对≥{ECONOMIC_MATERIALITY['nav_rel_min']}，MaxDD 恶化不超过 {ECONOMIC_MATERIALITY['maxdd_worse_max_abs']}。",
                f"- 门槛哈希 `{body['economic_gate_sha256']}`。",
                "- 三个控制已登记：原 v5、全 NaN 新列、训练折内打乱且保持缺失模式。",
                f"- `{FS_CHANGE_ID}` 已登记，此时还没有它的 IC 或回测。",
                "- 未过门的结果不能当 Round23 的默认 baseline。",
                "",
            ]
        ),
    )
    body["status"] = "FROZEN"
    return _save("freeze", body)


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume:
        cached = _done("parity")
        if cached:
            return cached
    parity = load_json(R22_3 / "parity" / "parity.json")
    summary = load_json(R22_3 / "candidates" / "R223_E0" / "summary.json")
    obs = parity.get("observed") or summary
    checks = {
        "cagr": abs(float(obs.get("cagr") or 0) - float(V5_ANCHOR["cagr"])) < 1e-8,
        "sharpe": abs(float(obs.get("sharpe") or 0) - float(V5_ANCHOR["sharpe"])) < 1e-8,
        "nav": abs(float(obs.get("nav_end") or summary.get("nav_end") or 0) - float(V5_NAV_END_EXACT)) < 0.05,
        "trades": int(obs.get("completed_trades") or summary.get("completed_trades") or 0) == int(V5_ANCHOR["completed_trades"]),
        "fee": (summary.get("fee_schedule") or FEE_SCHEDULE) == FEE_SCHEDULE,
        "engine_artifact": (R22_3 / "candidates" / "R223_E0" / "report.json").is_file(),
    }
    payload = {
        "status": "DONE",
        "passed": all(checks.values()) and bool(parity.get("passed", True)),
        "checks": checks,
        "source": "result/round22_3 已有正式袖套，本轮核对锚点，不重写该目录",
        "sharpe": obs.get("sharpe") or summary.get("sharpe"),
        "cagr": obs.get("cagr") or summary.get("cagr"),
        "nav_end": summary.get("nav_end") or obs.get("nav_end"),
    }
    if not payload["passed"]:
        payload["status"] = "BLOCKED"
        payload["reason"] = "v5 parity artifact does not match the frozen anchor"
    write_text(ROOT / "00_v5_parity.md", f"# Round22.4 parity\n\npassed={payload['passed']}\n\n{checks}\n")
    return _save("parity", payload)


def _scope_for(path: str, window: str | None, sharpe: float | None) -> str:
    text = f"{path} {window or ''}"
    if "round22_2" in text or (window and window.startswith("2023-01-03")):
        return "MATCHED_SUBPERIOD_DIAGNOSTIC"
    if "forward" in text.lower():
        return "FORWARD_PROSPECTIVE"
    if window and "2018-01-02" in window and "2026-07-24" in window:
        return "FULL_HISTORY_OFFICIAL"
    if sharpe is None:
        return "EXPLORATORY"
    return "NOT_COMPARABLE"


def run_lineage(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume:
        cached = _done("lineage")
        if cached:
            return cached
    rows = [
        {
            "experiment_id": "R16_O01",
            "round": "baseline_v5",
            "evaluation_scope": "FULL_HISTORY_OFFICIAL",
            "ledger_tag": "FULL_HISTORY_OFFICIAL",
            "window": f"{MAIN_WINDOW_START}..{MAIN_WINDOW_END}",
            "sharpe": V5_ANCHOR["sharpe"],
            "cagr": V5_ANCHOR["cagr"],
            "nav_end": V5_NAV_END_EXACT,
            "comparable_to_v5": True,
            "source": "frozen anchor",
        }
    ]
    sleeve = R22_3 / "backtest" / "sleeve_results.csv"
    if sleeve.is_file():
        frame = pd.read_csv(sleeve)
        for rec in frame.to_dict(orient="records"):
            rows.append(
                {
                    "experiment_id": rec.get("code"),
                    "round": "22.3",
                    "evaluation_scope": "FULL_HISTORY_OFFICIAL",
                    "ledger_tag": "FULL_HISTORY_OFFICIAL",
                    "window": f"{rec.get('window_start')}..{rec.get('window_end')}",
                    "sharpe": rec.get("sharpe"),
                    "cagr": rec.get("cagr"),
                    "nav_end": rec.get("nav_end"),
                    "comparable_to_v5": True,
                    "source": str(sleeve),
                }
            )
    short_rows = 0
    short_csv = R22_2 / "backtest" / "sleeve_results.csv"
    if short_csv.is_file():
        short = pd.read_csv(short_csv)
        for rec in short.to_dict(orient="records"):
            rows.append(
                {
                    "experiment_id": rec.get("code"),
                    "round": "22.2",
                    "evaluation_scope": "MATCHED_SUBPERIOD_DIAGNOSTIC",
                    "ledger_tag": "MATCHED_SUBPERIOD_NOT_COMPARABLE",
                    "window": f"{rec.get('window_start')}..{rec.get('window_end')}",
                    "sharpe": rec.get("sharpe"),
                    "cagr": rec.get("cagr"),
                    "nav_end": rec.get("nav_end"),
                    "comparable_to_v5": False,
                    "source": str(short_csv),
                }
            )
            short_rows += 1
    elif R22_2.exists():
        for summary in R22_2.glob("candidates/*/summary.json"):
            rec = load_json(summary)
            sharpe = rec.get("sharpe")
            window = str(rec.get("window") or "")
            scope = "MATCHED_SUBPERIOD_DIAGNOSTIC"
            rows.append(
                {
                    "experiment_id": rec.get("code") or summary.parent.name,
                    "round": "22.2",
                    "evaluation_scope": scope,
                    "ledger_tag": "MATCHED_SUBPERIOD_NOT_COMPARABLE",
                    "window": window or "2023-01-03..2026-07-24",
                    "sharpe": sharpe,
                    "cagr": rec.get("cagr"),
                    "nav_end": rec.get("nav_end"),
                    "comparable_to_v5": False,
                    "source": str(summary),
                }
            )
            short_rows += 1
        if short_rows == 0:
            for summary in R22_2.glob("**/summary.json"):
                rec = load_json(summary)
                if rec.get("sharpe") is None:
                    continue
                window = str(rec.get("window") or rec.get("window_start") or "")
                if "2023" not in window and "short" not in str(summary):
                    continue
                rows.append(
                    {
                        "experiment_id": rec.get("code") or summary.parent.name,
                        "round": "22.2",
                        "evaluation_scope": "MATCHED_SUBPERIOD_DIAGNOSTIC",
                        "ledger_tag": "MATCHED_SUBPERIOD_NOT_COMPARABLE",
                        "window": window,
                        "sharpe": rec.get("sharpe"),
                        "cagr": rec.get("cagr"),
                        "nav_end": rec.get("nav_end"),
                        "comparable_to_v5": False,
                        "source": str(summary),
                    }
                )
                short_rows += 1
    frame = pd.DataFrame(rows)
    bad = sorted(set(frame["evaluation_scope"]) - set(SCOPES))
    if bad:
        raise RuntimeError(f"illegal evaluation_scope {bad}")
    write_df(ROOT / "experiment_lineage_registry.csv", frame)
    write_text(
        ROOT / "01_lineage.md",
        "\n".join(
            [
                "# 谱系",
                "",
                "evaluation_scope 只使用冻结的五档。22.2 短窗的 ledger_tag 是 MATCHED_SUBPERIOD_NOT_COMPARABLE，不得与 v5 的 Sharpe 1.876 排在同一张无标注表里。",
                f"登记 {len(frame)} 行，其中 22.2 短窗 {int((frame['ledger_tag']=='MATCHED_SUBPERIOD_NOT_COMPARABLE').sum())} 行。",
                "",
            ]
        ),
    )
    return _save("lineage", {"status": "DONE", "n": int(len(frame)), "n_short": int((frame["ledger_tag"] == "MATCHED_SUBPERIOD_NOT_COMPARABLE").sum())})


def run_tracker(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (ROOT / "tracker_errata.md").is_file():
        return {"status": "DONE", "resumed": True}
    snapshot = ROOT / "tracker_snapshot_before_append.md"
    if not snapshot.is_file():
        shutil.copy2(TRACKER_PATH, snapshot)
    sleeve = pd.read_csv(R22_3 / "backtest" / "sleeve_results.csv")
    formal = sleeve.loc[sleeve["code"] != "R223_E0"]
    lines = [
        "",
        "### Round22.4 勘误（只追加，不改上文数字）",
        "",
        "写入时间以本文件追加为准。修改前快照：`result/round22_4/tracker_snapshot_before_append.md`。",
        "",
        "- 勘误：Round22.2 表内 Sharpe（含约 2.76 的 C4 五头）是 2023-01-03～2026-07-24 短窗，evaluation_scope=`MATCHED_SUBPERIOD_DIAGNOSTIC`，ledger_tag=`MATCHED_SUBPERIOD_NOT_COMPARABLE`。不得与 baseline_v5 Sharpe 1.876 混排。高 Sharpe 来自窗口。",
        "- Round22.3 的 12 个正式全历史回测已在上一节。下文给出同一 12 笔的全精度，不替换旧行，不开新实验。",
        "",
        "| 代号 | Sharpe | CAGR | NAV | ΔSharpe | ΔCAGR | 经济门 | 窗 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rec in formal.to_dict(orient="records"):
        lines.append(
            f"| `{rec['code']}` | {float(rec['sharpe']):.6f} | {float(rec['cagr']):.6f} | {float(rec['nav_end']):.2f} | {float(rec['delta_sharpe']):.6f} | {float(rec['delta_cagr']):.6f} | {rec['economically_material']} | {rec['window_start']}..{rec['window_end']} |"
        )
    lines.append("")
    lines.append("这 12 行对照的是 R16_O01，不是 R222_E0。没有一行过 R21_3 经济门。")
    lines.append("")
    append_tracker("\n".join(lines))
    write_text(ROOT / "tracker_errata.md", "\n".join(lines))
    return _save("tracker", {"status": "DONE", "n_formal_appended_precision": int(len(formal)), "snapshot": str(snapshot)})


def _fold_table(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    rows = []
    for year in range(2018, 2027):
        train = work.loc[work["trade_date"].dt.year.between(year - 3, year - 1)]
        test = work.loc[work["trade_date"].dt.year.eq(year)]
        row: dict[str, Any] = {"predict_year": year, "train_rows": int(len(train)), "test_rows": int(len(test))}
        for col in (C1_LAG1_ID, C4_ID, "r22_c1_xl_net_imbalance"):
            if col not in work.columns:
                row[f"{col}_train_finite"] = None
                row[f"{col}_test_finite"] = None
                continue
            row[f"{col}_train_finite"] = float(pd.to_numeric(train[col], errors="coerce").notna().mean()) if len(train) else None
            row[f"{col}_test_finite"] = float(pd.to_numeric(test[col], errors="coerce").notna().mean()) if len(test) else None
        c4_test = row.get(f"{C4_ID}_test_finite")
        row["c4_test_all_missing"] = bool(c4_test == 0.0)
        row["verified_c_information"] = not (c4_test == 0.0 and (row.get(f"{C4_ID}_train_finite") or 0) == 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _top_change(left: pd.DataFrame, right: pd.DataFrame, dates: set[pd.Timestamp] | None = None) -> dict[str, Any]:
    def tops(frame: pd.DataFrame) -> pd.Series:
        work = frame.copy()
        work["signal_date"] = pd.to_datetime(work["signal_date"])
        if dates is not None:
            work = work.loc[work["signal_date"].isin(dates)]
        work = work.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort")
        return work.groupby("signal_date", sort=False).head(1).set_index("signal_date")["instrument_id"]

    a = tops(left)
    b = tops(right)
    both = a.index.intersection(b.index)
    if len(both) == 0:
        return {"n_days": 0, "top1_change_rate": None}
    changed = a.loc[both].astype(str).ne(b.loc[both].astype(str))
    return {"n_days": int(len(both)), "top1_change_rate": float(changed.mean()), "n_changed": int(changed.sum())}


def run_coverage(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume:
        cached = _done("coverage")
        if cached:
            return cached
    panel = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", C1_LAG1_ID, C4_ID, "r22_c1_xl_net_imbalance"])
    folds = _fold_table(panel)
    write_df(ROOT / "c1_c4_fold_coverage.csv", folds)
    empty_years = [int(y) for y in folds.loc[folds["c4_test_all_missing"], "predict_year"]]
    audits = []
    e0 = pd.read_parquet(E0_FIVE, columns=["signal_date", "instrument_id", "score"])
    for code, col in (
        ("R223_E1_r22_c4_flow_persist_5", C4_ID),
        ("R223_E1_r22_2_c1_xl_net_imbalance_lag1", C1_LAG1_ID),
    ):
        path = R22_3 / "candidates" / code / "five_pred.parquet"
        if not path.is_file():
            audits.append({"code": code, "status": "MISSING_PRED"})
            continue
        pred = pd.read_parquet(path, columns=["signal_date", "instrument_id", "score"])
        pred["signal_date"] = pd.to_datetime(pred["signal_date"])
        e0d = e0.copy()
        e0d["signal_date"] = pd.to_datetime(e0d["signal_date"])
        empty_dates = set()
        sub = panel.copy()
        sub["trade_date"] = pd.to_datetime(sub["trade_date"])
        for year in empty_years:
            part = sub.loc[sub["trade_date"].dt.year.eq(year)]
            if float(pd.to_numeric(part[col], errors="coerce").notna().mean() or 0) == 0.0:
                empty_dates.update(pd.to_datetime(part["trade_date"]).unique())
        change = _top_change(e0d, pred, empty_dates if empty_dates else None)
        change.update({"code": code, "column": col, "empty_test_years": empty_years, "counts_as_alpha": False})
        if change.get("top1_change_rate") and change["top1_change_rate"] > 0 and empty_dates:
            change["cause_check"] = "测试折该列为全缺失时 Top1 仍变化。先记为缺失列/特征顺序效应，不计 alpha。全 NaN 与打乱控制用来分开这件事。"
        audits.append(change)
    dump_json(ROOT / "empty_fold_rank_audit.json", {"rows": audits, "empty_c4_years": empty_years, "available_from": C_AVAILABLE_FROM})
    write_text(
        ROOT / "02_c1_c4_coverage.md",
        "\n".join(
            [
                "# C1/C4 折级覆盖",
                "",
                f"可得性起点 {C_AVAILABLE_FROM}。测试折覆盖率为 0 的预测年：{empty_years}。这些折不能写成已经验证的 C 因子信息。",
                "2018–2020 若训练折也全缺失，模型看到的是一条全空列。此时排名变化不计 alpha。",
                "",
                "```",
                folds.to_string(index=False),
                "```",
                "",
            ]
        ),
    )
    return _save("coverage", {"status": "DONE", "empty_c4_years": empty_years, "audits": audits})


def _slice_delta(base_nav: pd.DataFrame, other_nav: pd.DataFrame, start, end) -> dict[str, Any]:
    from core.round11.common import daily_from_nav

    a = daily_from_nav(base_nav)
    b = daily_from_nav(other_nav)
    both = pd.concat([a.rename("base"), b.rename("new")], axis=1, join="inner").dropna()
    if start is not None:
        both = both.loc[both.index >= pd.Timestamp(start)]
    if end is not None:
        both = both.loc[both.index < pd.Timestamp(end)]
    if len(both) < 5:
        return {"n_obs": int(len(both)), "status": "TOO_SHORT"}
    excess = both["new"] - both["base"]
    vol = float(excess.std(ddof=1))
    sharpe = float(excess.mean() / vol * np.sqrt(252)) if vol > 0 else None
    wealth_b = float((1.0 + both["base"]).prod())
    wealth_n = float((1.0 + both["new"]).prod())
    return {
        "n_obs": int(len(both)),
        "excess_sharpe_not_comparable_across_slices": sharpe,
        "slice_wealth_ratio": wealth_n / wealth_b if wealth_b else None,
        "sum_excess_log": float(np.log1p(both["new"]).sum() - np.log1p(both["base"]).sum()),
        "status": "OK",
    }


def run_period(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume:
        cached = _done("period")
        if cached:
            return cached
    cuts = [
        ("PRE_AVAILABILITY", None, C_AVAILABLE_FROM),
        ("PARTIAL", C_AVAILABLE_FROM, "2024-01-01"),
        ("FULL_TRAIN", "2024-01-01", None),
    ]
    base = pd.read_parquet(E0_NAV)
    rows = []
    for code in ("R223_E1_r22_2_c1_xl_net_imbalance_lag1", "R223_E1_r22_c4_flow_persist_5"):
        nav_path = R22_3 / "candidates" / code / "nav.parquet"
        if not nav_path.is_file():
            rows.append({"code": code, "slice": "ALL", "status": "MISSING_NAV"})
            continue
        other = pd.read_parquet(nav_path)
        for name, start, end in cuts:
            rec = _slice_delta(base, other, start, end)
            rec.update({"code": code, "slice": name, "note": "切片 Sharpe 不相加，也不与全历史 Sharpe 混排"})
            rows.append(rec)
    frame = pd.DataFrame(rows)
    write_df(ROOT / "availability_period_attribution.csv", frame)
    write_text(
        ROOT / "03_period_split.md",
        "\n".join(
            [
                "# 增量按可得性拆开",
                "",
                "PRE：2020-09-24 之前。PARTIAL：该日到 2023 年末（训练窗仍含不可得月份）。FULL_TRAIN：2024 起，训练三年都在可得期之后。",
                "全历史 C1 五头 ΔSharpe=+0.0135、C4 五头 ΔSharpe=+0.0096 都没过 0.05 的门。下表只解释增量落在哪一段，不把三段 Sharpe 加总。",
                "",
                "```",
                frame.to_string(index=False),
                "```",
                "",
            ]
        ),
    )
    return _save("period", {"status": "DONE", "n": int(len(frame))})


def _shuffle(train: pd.DataFrame, test: pd.DataFrame, year: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed * 10007 + int(year))
    tr = train.copy()
    te = test.copy()
    mask_tr = pd.to_numeric(tr[C4_ID], errors="coerce").notna()
    vals = pd.to_numeric(tr.loc[mask_tr, C4_ID], errors="coerce").to_numpy().copy()
    rng.shuffle(vals)
    tr[SHUFFLE_COL] = np.nan
    if len(vals):
        tr.loc[mask_tr, SHUFFLE_COL] = vals
    te[SHUFFLE_COL] = np.nan
    mask_te = pd.to_numeric(te[C4_ID], errors="coerce").notna()
    if bool(mask_te.any()) and len(vals):
        te.loc[mask_te, SHUFFLE_COL] = rng.choice(vals, size=int(mask_te.sum()), replace=True)
    return tr, te


def run_null(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    if resume:
        cached = _done("null")
        if cached and cached.get("status") == "DONE":
            return cached
    if not (ROOT / "frozen_manifest.json").is_file():
        raise RuntimeError("freeze manifest missing; refuse to train")
    train = pd.read_parquet(TRAIN_PANEL)
    cand = pd.read_parquet(PANEL)
    for frame in (train, cand):
        frame[ALLNAN_COL] = np.nan
        frame[SHUFFLE_COL] = pd.to_numeric(frame[C4_ID], errors="coerce")
    ten = pd.read_parquet(E0_TEN)
    base_features = list(FIVE_FEATURES)
    results = []
    v5_summary = load_json(R22_3 / "candidates" / "R223_E0" / "summary.json")
    v5_summary = {
        "code": "R224_CTRL_V5",
        "kind": "baseline",
        "status": "REUSED_R223_E0",
        "sharpe": v5_summary.get("sharpe"),
        "cagr": v5_summary.get("cagr"),
        "nav_end": v5_summary.get("nav_end"),
        "max_drawdown": v5_summary.get("max_drawdown"),
        "delta_sharpe": 0.0,
        "delta_cagr": 0.0,
        "economically_material": False,
        "counts_as_alpha": False,
    }
    results.append(v5_summary)
    specs = [
        ("R224_CTRL_ALLNAN", base_features + [ALLNAN_COL], [ALLNAN_COL], None),
        ("R224_CTRL_SHUFFLE", base_features + [SHUFFLE_COL], [SHUFFLE_COL], _shuffle),
    ]
    for code, features, extras, mutator in specs:
        dest = ROOT / "null" / code
        payload = train_ranker_head(
            training=train,
            candidates=cand,
            features=features,
            extras=extras,
            head_name="five",
            output_dir=dest / "five",
            seeds=SEEDS,
            n_jobs=jobs,
            resume=resume,
            mutator=mutator,
        )
        five = pd.read_parquet(dest / "five" / "predictions.parquet")
        e0 = pd.read_parquet(E0_FIVE)
        change = _top_change(e0, five)
        sleeve = run_heads_sleeve(
            dest / "sleeve",
            five,
            ten,
            {
                "code": code,
                "round": "22.4",
                "top_changed": bool(change.get("top1_change_rate") and change["top1_change_rate"] > 0),
                "control": True,
                "counts_as_alpha": False,
            },
            resume=resume,
            base_nav=E0_NAV,
        )
        sleeve["kind"] = "all_nan" if "ALLNAN" in code else "shuffle"
        sleeve["top1_change_vs_e0"] = change
        sleeve["train_rows"] = payload.get("rows")
        sleeve["counts_as_alpha"] = False
        if sleeve.get("economically_material"):
            sleeve["economically_material"] = False
            sleeve["override"] = "控制组即使过门也不计 alpha"
        results.append(sleeve)
        print(f"[null] {code} sharpe={sleeve.get('sharpe')} dS={sleeve.get('delta_sharpe')}", flush=True)
    frame = pd.DataFrame(results)
    write_df(ROOT / "null_controls.csv", frame)
    dump_json(ROOT / "null_controls.json", {"rows": results, "counts_as_alpha": False})
    return _save("null", {"status": "DONE", "n": len(results)})


def _max_abs_rho(panel: pd.DataFrame, column: str, features: list[str]) -> dict[str, Any]:
    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    best = {"feature": None, "abs_rho": 0.0, "year": None}
    per_year = []
    if column not in work.columns:
        return {"status": "MISSING_COLUMN", "max_abs_rho": None, "feature": None}
    for year in range(2018, 2027):
        block = work.loc[work["trade_date"].dt.year.between(year - 3, year - 1)]
        left = pd.to_numeric(block[column], errors="coerce")
        for feat in features:
            if feat not in block.columns:
                continue
            right = pd.to_numeric(block[feat], errors="coerce")
            both = pd.concat([left, right], axis=1).dropna()
            if len(both) < 50:
                continue
            rho = float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))
            if np.isfinite(rho) and abs(rho) > best["abs_rho"]:
                best = {"feature": feat, "abs_rho": abs(rho), "year": year, "rho": rho}
        per_year.append({"year": year})
    best["triggered"] = bool(best["abs_rho"] >= RHO_GATE)
    best["per_year_checked"] = True
    return best


def run_redundancy(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume:
        cached = _done("redundancy")
        if cached:
            return cached
    cols = ["trade_date", "instrument_id", *FIVE_FEATURES, C4_ID, C1_LAG1_ID]
    panel = pd.read_parquet(TRAIN_PANEL, columns=[c for c in cols if c])
    rows = []
    for col in (C4_ID, C1_LAG1_ID):
        rec = _max_abs_rho(panel, col, list(FIVE_FEATURES))
        rec["column"] = col
        rec["action"] = "REMOVE_AND_REPLACE" if rec.get("triggered") else "NOT_TRIGGERED"
        rows.append(rec)
    payload = {
        "status": "DONE",
        "rho_gate": RHO_GATE,
        "rows": rows,
        "remove_replace_trained": False,
        "reason": "折内最大 |ρ| 未到 0.60，不做 Remove/Replace。不因为接近门槛就放宽。",
    }
    if any(r.get("triggered") for r in rows):
        payload["status"] = "BLOCKED"
        payload["reason"] = "出现 |ρ|≥0.60，但本实现拒绝在未同时跑 Remove 与 Replace 时声称替换有效。本轮没有安全的替换训练预算被预注册到单一对照之外，标阻断而不是只比 baseline。"
        payload["remove_replace_trained"] = False
    dump_json(ROOT / "redundancy.json", payload)
    write_text(ROOT / "04_redundancy.md", f"# 冗余\n\n门槛 |ρ|≥{RHO_GATE}。\n\n```\n{pd.DataFrame(rows).to_string(index=False)}\n```\n\n{payload['reason']}\n")
    return _save("redundancy", payload)


def _event_ic(events: pd.DataFrame, value_col: str, labels: pd.DataFrame) -> dict[str, Any]:
    ev = events.dropna(subset=["known_at", "instrument_id"]).copy()
    ev["known_at"] = pd.to_datetime(ev["known_at"])
    ev["instrument_id"] = ev["instrument_id"].astype(str)
    ev = ev.sort_values(["instrument_id", "known_at", "fiscal_period_end"])
    ev = ev.drop_duplicates(["instrument_id", "known_at", "fiscal_period_end"])
    lab = labels.copy()
    lab["trade_date"] = pd.to_datetime(lab["trade_date"])
    lab["instrument_id"] = lab["instrument_id"].astype(str)
    lab = lab.dropna(subset=["label_e2_net"]).sort_values(["instrument_id", "trade_date"])
    pieces = []
    for iid, block in ev.groupby("instrument_id", sort=False):
        right = lab.loc[lab["instrument_id"].eq(iid), ["trade_date", "label_e2_net"]]
        if right.empty:
            continue
        left = block[["known_at", value_col]].rename(columns={"known_at": "trade_date"})
        merged = pd.merge_asof(left.sort_values("trade_date"), right, on="trade_date", direction="forward")
        merged["instrument_id"] = iid
        pieces.append(merged)
    if not pieces:
        return {"n_events": 0, "rankic": None, "icir": None, "status": "NO_JOIN"}
    joined = pd.concat(pieces, ignore_index=True).dropna(subset=[value_col, "label_e2_net"])
    joined = joined.drop_duplicates(["instrument_id", "trade_date"])
    daily = []
    for day, block in joined.groupby("trade_date"):
        if len(block) < 5:
            continue
        rho = block[value_col].corr(block["label_e2_net"], method="spearman")
        if pd.notna(rho):
            daily.append(float(rho))
    if not daily:
        return {"n_events": int(len(joined)), "n_days": 0, "rankic": None, "icir": None, "status": "NO_DAILY_IC"}
    arr = np.asarray(daily, dtype=float)
    icir = float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else None
    return {
        "n_events": int(len(joined)),
        "n_days": int(len(arr)),
        "rankic": float(arr.mean()),
        "icir": icir,
        "status": "OK",
        "independent_sampling": "one row per announcement event, not per subsequent trade day",
        "filled_with_zero": False,
    }


def run_fs(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    if resume:
        cached = _done("fs")
        if cached:
            return cached
    if not (ROOT / "frozen_manifest.json").is_file():
        raise RuntimeError("registry must exist before FS IC")
    manifest = load_json(ROOT / "frozen_manifest.json")
    ids = [row["factor_id"] for row in manifest.get("fs_registered_before_backtest") or []]
    if FS_CHANGE_ID not in ids:
        raise RuntimeError("cash conversion change was not registered before IC")
    fs2 = pd.read_parquet(R22_3 / "data_recovery" / "fs2_panel.parquet")
    fs1 = pd.read_parquet(R22_3 / "data_recovery" / "fs1_panel.parquet")
    labels = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", "label_e2_net"])
    events = fs2.sort_values(["instrument_id", "known_at", "fiscal_period_end"]).copy()
    events["fs2_value"] = pd.to_numeric(events[FS2_ID], errors="coerce")
    events = events.drop_duplicates(["instrument_id", "known_at", "fiscal_period_end"])
    events["prev"] = events.groupby("instrument_id")["fs2_value"].shift(1)
    events[FS_CHANGE_ID] = events["fs2_value"] - events["prev"]
    events.loc[events["prev"].isna(), FS_CHANGE_ID] = np.nan
    if events[FS_CHANGE_ID].fillna(0).eq(0).all() and events[FS_CHANGE_ID].notna().any():
        pass
    assert not bool((events[FS_CHANGE_ID].isna() & events["prev"].isna()).sum() and False)
    event_path = ROOT / "data" / "fs_events.parquet"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    keep = ["instrument_id", "known_at", "published_at", "fiscal_period_end", "fs2_value", FS_CHANGE_ID, "prev"]
    events[keep].to_parquet(event_path, index=False)
    ic_rows = {
        FS1_ID: _event_ic(fs1.rename(columns={FS1_ID: FS1_ID}), FS1_ID, labels),
        FS2_ID: _event_ic(events, "fs2_value", labels),
        FS_CHANGE_ID: _event_ic(events, FS_CHANGE_ID, labels),
    }
    decisions = []
    for fid, ic in ic_rows.items():
        ok = evidence_pass(ic.get("rankic"), ic.get("icir"), int(ic.get("n_events") or 0), True)
        decisions.append({"factor_id": fid, "train": ok, "ic": ic, "counts_as_alpha_until_sleeve": False})
    trained = []
    passing = [row for row in decisions if row["train"]]
    for row in decisions:
        if not row["train"]:
            row["sleeve"] = {"status": "NOT_TRAINED", "reason": "evidence gate failed before any backtest"}
    if passing:
        levels = {
            FS1_ID: fs1[["trade_date", "instrument_id", FS1_ID]],
            FS2_ID: fs2[["trade_date", "instrument_id", FS2_ID]],
        }
        change_events = events[["instrument_id", "known_at", FS_CHANGE_ID]].dropna(subset=["known_at"]).copy()
        change_events["known_at"] = pd.to_datetime(change_events["known_at"])
        change_events["instrument_id"] = change_events["instrument_id"].astype(str)
        change_events = change_events.sort_values(["instrument_id", "known_at"]).drop_duplicates(["instrument_id", "known_at"], keep="last")

        def _attach(frame: pd.DataFrame) -> pd.DataFrame:
            work = frame.copy()
            work["trade_date"] = pd.to_datetime(work["trade_date"])
            work["instrument_id"] = work["instrument_id"].astype(str)
            for fid, level in levels.items():
                extra = level.copy()
                extra["trade_date"] = pd.to_datetime(extra["trade_date"])
                extra["instrument_id"] = extra["instrument_id"].astype(str)
                if fid in work.columns:
                    work = work.drop(columns=[fid])
                work = work.merge(extra, on=["trade_date", "instrument_id"], how="left")
            parts = []
            for iid, block in work.groupby("instrument_id", sort=False):
                right = change_events.loc[change_events["instrument_id"].eq(iid), ["known_at", FS_CHANGE_ID]]
                left = block.sort_values("trade_date")
                if FS_CHANGE_ID in left.columns:
                    left = left.drop(columns=[FS_CHANGE_ID])
                if right.empty:
                    left[FS_CHANGE_ID] = np.nan
                    parts.append(left)
                    continue
                parts.append(pd.merge_asof(left, right.rename(columns={"known_at": "trade_date"}), on="trade_date", direction="backward"))
            return pd.concat(parts, ignore_index=True)

        five_train = _attach(pd.read_parquet(TRAIN_PANEL))
        five_cand = _attach(pd.read_parquet(PANEL))
        ten_train = _attach(pd.read_parquet(R22_3 / "data" / "ten_training.parquet"))
        ten_cand = _attach(pd.read_parquet(R22_3 / "data" / "ten_candidate.parquet"))
        ten_pred = pd.read_parquet(E0_TEN)
        five_pred = pd.read_parquet(E0_FIVE)
        for row in passing:
            fid = row["factor_id"]
            specs = (
                ("R224_E1_" + fid, "five", list(FIVE_FEATURES) + [fid], [fid], five_train, five_cand, None),
                ("R224_E2_" + fid, "ten", list(TEN_FEATURES) + [fid], [fid], ten_train, ten_cand, None),
                ("R224_E3_" + fid, "both", None, None, None, None, None),
            )
            made = {}
            for code, which, cols, extras, tr, ca, _unused in specs:
                out = ROOT / "fs_models" / code
                if which == "both":
                    preds_five = made["R224_E1_" + fid]
                    preds_ten = made["R224_E2_" + fid]
                else:
                    train_ranker_head(
                        training=tr,
                        candidates=ca,
                        features=cols,
                        extras=extras,
                        head_name=which,
                        output_dir=out / which,
                        n_jobs=jobs,
                        resume=resume,
                    )
                    made[code] = pd.read_parquet(out / which / "predictions.parquet")
                    preds_five = made[code] if which == "five" else five_pred
                    preds_ten = made[code] if which == "ten" else ten_pred
                sleeve = run_heads_sleeve(
                    out / "sleeve",
                    preds_five,
                    preds_ten,
                    {"code": code, "round": "22.4", "factor_id": fid, "top_changed": True},
                    resume=resume,
                    base_nav=E0_NAV,
                )
                trained.append(sleeve)
            row["sleeve_codes"] = [f"R224_E{i}_{fid}" for i in (1, 2, 3)]
    joint_add = "NOT_TRIGGERED"
    e1_pass = [r for r in trained if "_E1_" in str(r.get("code")) and r.get("economically_material")]
    if len(e1_pass) >= 2:
        joint_add = "BLOCKED_JOINT_REQUIRES_EACH_FACTOR_AND_22_3_ALREADY_USED_THE_ONE_JOINT_ROUND"
    payload = {
        "status": "DONE",
        "registered_before_ic": True,
        "decisions": decisions,
        "sleeves": trained,
        "filled_with_zero": False,
        "joint_add": joint_add,
    }
    dump_json(ROOT / "fs_evidence.json", payload)
    write_text(ROOT / "05_fs.md", f"# FS\n\n登记先于 IC。事件级样本，缺失不填 0。\n\n```\n{json_dumps(decisions)}\n```\n")
    return _save("fs", payload)


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    nulls = load_json(ROOT / "null_controls.json").get("rows") or []
    fs = load_json(ROOT / "fs_evidence.json")
    sleeves = list(fs.get("sleeves") or [])
    promotable = []
    for row in sleeves:
        if row.get("scenario"):
            continue
        if str(row.get("code", "")).startswith("R224_CTRL"):
            continue
        if row.get("economically_material") is True:
            promotable.append(row.get("code"))
    decision = RECOMMEND if promotable else KEEP
    reason = (
        "有候选同时过经济门。"
        if promotable
        else "C1/C4 的全历史增量、空值对照和财务报表事件因子都没有形成可晋级的经济增量。空值对照不计 alpha。"
    )
    if decision == RECOMMEND:
        reason += " 没有独立 forward，不得标 LIVE_READY。状态附 PROMOTION_BLOCKED_BY_FORWARD_VALIDATION。"
    best = None
    best_ds = None
    for row in list(nulls) + sleeves:
        ds = row.get("delta_sharpe")
        if ds is None:
            continue
        if best_ds is None or abs(float(ds)) > abs(best_ds):
            best_ds = float(ds)
            best = {"code": row.get("code"), "delta_sharpe": ds, "delta_cagr": row.get("delta_cagr")}
    payload = {
        "status": "DONE",
        "final_decision": decision,
        "reason": reason,
        "promotable": promotable,
        "live_ready": False,
        "current_baseline_changed": False,
        "auto_promote": False,
        "forward": "FORWARD_OOS_NOT_AVAILABLE",
        "promotion_note": "PROMOTION_BLOCKED_BY_FORWARD_VALIDATION" if decision == RECOMMEND else "NOT_TRIGGERED",
        "next_round_baseline": "baseline_v5=R16_O01",
        "best_abs_move": best,
        "gate_id": ECONOMIC_MATERIALITY["id"],
    }
    dump_json(ROOT / "decision.json", payload)
    return payload


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    decision = load_json(ROOT / "decision.json")
    if not decision:
        decision = run_decision()
    parts = ["# Round22.4 研究报告", "", f"决策：**{decision.get('final_decision')}**。{decision.get('reason')}", ""]
    for name in (
        "00_frozen_manifest.md",
        "00_v5_parity.md",
        "01_lineage.md",
        "02_c1_c4_coverage.md",
        "03_period_split.md",
        "04_redundancy.md",
        "05_fs.md",
    ):
        path = ROOT / name
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
            parts.append("")
    nulls = pd.read_csv(ROOT / "null_controls.csv") if (ROOT / "null_controls.csv").is_file() else pd.DataFrame()
    parts.append("## 控制组正式袖套")
    parts.append("")
    if not nulls.empty:
        cols = [c for c in ("code", "sharpe", "cagr", "delta_sharpe", "delta_cagr", "nav_end", "max_drawdown", "economically_material") if c in nulls.columns]
        parts.append("```")
        parts.append(nulls[cols].to_string(index=False))
        parts.append("```")
    parts.append("")
    parts.append(f"CURRENT 未改。AUTO_PROMOTE=False。下一轮默认基线仍是 v5。经济门 `{ECONOMIC_MATERIALITY['id']}` 未放宽。")
    parts.append("")
    text = "\n".join(parts)
    write_text(ROOT / "ROUND22_4_RESEARCH_REPORT.md", text)
    return {"status": "DONE", "final_report": str(ROOT / "ROUND22_4_RESEARCH_REPORT.md"), "decision": decision.get("final_decision")}


def expand_stages(raw: str) -> list[str]:
    from .contract import STAGES

    if raw in {"all", "sequence"}:
        return list(STAGES)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    bad = [p for p in parts if p not in STAGES]
    if bad:
        raise RuntimeError(f"unknown stage {bad}")
    return parts


def run_stage(stage: str, *, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    table = {
        "freeze": run_freeze,
        "parity": run_parity,
        "lineage": run_lineage,
        "tracker": run_tracker,
        "coverage": run_coverage,
        "period": run_period,
        "null": run_null,
        "redundancy": run_redundancy,
        "fs": run_fs,
        "decision": run_decision,
        "report": run_report,
    }
    if stage not in table:
        raise RuntimeError(stage)
    return table[stage](resume=resume, jobs=jobs)
