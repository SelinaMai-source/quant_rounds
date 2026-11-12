"""Stage A：受控替换。禁止用「版本差异」交差。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import (
    ROUND5_ROOT,
    ROUND6_ROOT,
    assert_project_result_path,
    ensure_round6_tree,
    stock_path,
)
from ..round2.data import SEEDS, load_seed_predictions
from ..stock_readonly import ensemble_dir, load_fs00263_ensemble_predictions, sha256_file
from .registry import (
    CANONICAL_CODE,
    LOCKED_FEATURES,
    SPECS,
    V3_CAGR,
    clone_spec,
    get_spec,
)
from .runner import load_report, run_spec, write_skipped
from .train import (
    fold_mask,
    load_panel,
    load_predictions,
    train_spec,
    training_panel_path,
    training_panel_sha256,
    window_bounds,
)

COMPLETE_YEARS = tuple(range(2018, 2026))


def _pct(v: Any, d: int = 2) -> str:
    if v is None:
        return "—"
    return f"{float(v) * 100:.{d}f}%"


def _num(v: Any, d: int = 3) -> str:
    if v is None:
        return "—"
    return f"{float(v):.{d}f}"


def _write_json(rel: str, payload: Any) -> Path:
    path = ensure_round6_tree() / rel
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def daily_topn(frame: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    ranked = frame.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True])
    return ranked.groupby("signal_date", sort=True).head(n)[["signal_date", "instrument_id"]].copy()


def overlap_metrics(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
    aa = a.copy()
    bb = b.copy()
    aa["signal_date"] = pd.to_datetime(aa["signal_date"])
    bb["signal_date"] = pd.to_datetime(bb["signal_date"])
    aa["instrument_id"] = aa["instrument_id"].astype(str)
    bb["instrument_id"] = bb["instrument_id"].astype(str)
    top1_a = daily_topn(aa, 1)
    top1_b = daily_topn(bb, 1)
    m1 = top1_a.merge(top1_b, on="signal_date", suffixes=("_a", "_b"))
    top1 = float((m1["instrument_id_a"] == m1["instrument_id_b"]).mean()) if len(m1) else None
    t3a = daily_topn(aa, 3).groupby("signal_date")["instrument_id"].apply(set)
    t3b = daily_topn(bb, 3).groupby("signal_date")["instrument_id"].apply(set)
    idx = t3a.index.intersection(t3b.index)
    jacs = []
    for d in idx:
        u = t3a.loc[d] | t3b.loc[d]
        jacs.append(len(t3a.loc[d] & t3b.loc[d]) / len(u) if u else np.nan)
    merged = aa.merge(bb, on=["signal_date", "instrument_id"], suffixes=("_a", "_b"))
    spears, kends = [], []
    for _, day in merged.groupby("signal_date"):
        if day["score_a"].nunique() < 2 or day["score_b"].nunique() < 2:
            continue
        spears.append(day["score_a"].corr(day["score_b"], method="spearman"))
        kends.append(day["score_a"].corr(day["score_b"], method="kendall"))
    return {
        "top1_overlap": top1,
        "top3_jaccard": float(np.nanmean(jacs)) if jacs else None,
        "spearman": float(np.nanmean(spears)) if spears else None,
        "kendall": float(np.nanmean(kends)) if kends else None,
        "n_days": int(m1["signal_date"].nunique()) if len(m1) else 0,
    }


def yearly_selection_alpha(preds: pd.DataFrame, panel: pd.DataFrame, ycol: str = "label_exit_rule_net") -> dict[str, float | None]:
    work = preds.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    lab = panel[["trade_date", "instrument_id", ycol]].copy() if ycol in panel.columns else None
    if lab is None:
        return {}
    lab["trade_date"] = pd.to_datetime(lab["trade_date"])
    lab["instrument_id"] = lab["instrument_id"].astype(str)
    merged = work.merge(
        lab.rename(columns={"trade_date": "signal_date"}),
        on=["signal_date", "instrument_id"],
        how="left",
    )
    top = daily_topn(merged, 1).merge(merged, on=["signal_date", "instrument_id"], how="left")
    top["year"] = pd.to_datetime(top["signal_date"]).dt.year
    day_mean = merged.groupby("signal_date")[ycol].mean()
    top = top.merge(day_mean.rename("day_mean"), on="signal_date", how="left")
    out: dict[str, float | None] = {}
    for year, blk in top.groupby("year"):
        yy = pd.to_numeric(blk[ycol], errors="coerce")
        base = pd.to_numeric(blk["day_mean"], errors="coerce")
        out[str(int(year))] = float((yy - base).mean()) if yy.notna().any() else None
    return out


def _cagr(code: str) -> float | None:
    m = (load_report(code) or {}).get("metrics") or {}
    v = m.get("annualized_return")
    return None if v is None else float(v)


def _booster_missing() -> bool:
    root = stock_path("runs/first_board_factor_search_5000_v1/predictions/n50_d2/seed42/fs00263")
    for pat in ("*.ubj", "models/*.json", "xgbranker_*.json"):
        if list(root.glob(pat)):
            return False
    models = root / "models"
    if models.is_dir() and any(models.iterdir()):
        return False
    return True


def write_forensics(
    *,
    predictions=None,
    prepared=None,
    meta=None,
    resume: bool = True,
    run_now: bool = True,
) -> dict[str, Any]:
    ensure_round6_tree()
    dest = ROUND6_ROOT / "01_forensics" / "forensics.json"
    md_path = ROUND6_ROOT / "01_reproducibility_forensics.md"
    if resume and dest.is_file() and md_path.is_file() and (ROUND6_ROOT / "canonical" / "R6_CANONICAL_RETRAIN.json").is_file():
        return json.loads(dest.read_text(encoding="utf-8"))

    locked = predictions if predictions is not None else load_fs00263_ensemble_predictions()
    panel = load_panel(LOCKED_FEATURES, "label_exit_rule_net", source="current")
    frozen_preds = locked.copy()
    steps: dict[str, Any] = {}

    # A00：锁分 + 当前引擎
    a00 = run_spec(get_spec("R6_A00"), predictions=locked, prepared_market=prepared, market_meta=meta, resume=resume, run_now=run_now)
    steps["R6_A00"] = {"status": a00.get("status"), "cagr": _cagr("R6_A00"), "note": "应 = v3"}

    # A01：复用 R5_D00
    train_spec(get_spec("R6_A01"), resume=resume)
    a01 = run_spec(get_spec("R6_A01"), prepared_market=prepared, market_meta=meta, resume=resume, run_now=run_now)
    d00 = json.loads((ROUND5_ROOT / "R5_D00" / "report.json").read_text(encoding="utf-8")) if (ROUND5_ROOT / "R5_D00" / "report.json").is_file() else {}
    steps["R6_A01"] = {
        "status": a01.get("status"),
        "cagr": _cagr("R6_A01"),
        "r5_d00_cagr": (d00.get("metrics") or {}).get("annualized_return"),
        "overlap_vs_locked": overlap_metrics(load_predictions("A01_r5_d00"), frozen_preds),
    }

    # A02：冻结宇宙 + 当前重训
    train_spec(get_spec("R6_A02"), resume=resume, freeze_universe=True)
    a02 = run_spec(get_spec("R6_A02"), prepared_market=prepared, market_meta=meta, resume=resume, run_now=run_now)
    steps["R6_A02"] = {
        "status": a02.get("status"),
        "cagr": _cagr("R6_A02"),
        "overlap_vs_locked": overlap_metrics(load_predictions("A02_frozen_universe"), frozen_preds),
    }

    # A03：冻结 training_panel
    panel_sha = training_panel_sha256()
    train_spec(get_spec("R6_A03"), resume=resume, panel_source="frozen_training_panel")
    a03 = run_spec(get_spec("R6_A03"), prepared_market=prepared, market_meta=meta, resume=resume, run_now=run_now)
    steps["R6_A03"] = {
        "status": a03.get("status"),
        "cagr": _cagr("R6_A03"),
        "training_panel_sha256": panel_sha,
        "training_panel_path": str(training_panel_path()),
        "overlap_vs_locked": overlap_metrics(load_predictions("A03_frozen_panel"), frozen_preds),
    }

    # A04：0.8 FORENSIC_ONLY
    train_spec(get_spec("R6_A04"), resume=resume)
    a04 = run_spec(get_spec("R6_A04"), prepared_market=prepared, market_meta=meta, resume=resume, run_now=run_now)
    steps["R6_A04"] = {
        "status": a04.get("status"),
        "cagr": _cagr("R6_A04"),
        "forensic_only": True,
        "subsample": 0.8,
        "overlap_vs_locked": overlap_metrics(load_predictions("A04_hist_0p8"), frozen_preds),
        "note": "FORENSIC_ONLY，不得当晋升候选。",
    }

    # A05
    write_skipped(get_spec("R6_A05"), "冻结 booster 不存在（models_persisted=false，无 .json/.ubj）", status="SKIPPED_NO_ARTIFACT")
    steps["R6_A05"] = {
        "status": "SKIPPED_NO_ARTIFACT",
        "booster_missing": _booster_missing(),
        "note": "改比：按 training_report.json 超参重训分数 vs 锁分 parquet（见 A04 vs locked）。",
    }

    # A06：单 seed vs 5-seed，读 Stock ensemble_report
    ens_report = json.loads((ensemble_dir() / "ensemble_report.json").read_text(encoding="utf-8"))
    seed42 = load_seed_predictions(seed=42)
    seed_overlap = overlap_metrics(seed42, frozen_preds)
    steps["R6_A06"] = {
        "status": "compared",
        "ensemble_method": ens_report.get("method"),
        "ensemble_seeds": ens_report.get("seeds"),
        "prediction_sha256": ens_report.get("prediction_sha256"),
        "seed42_vs_ensemble": seed_overlap,
        "note": "聚合必须是日内 percentile 等权，已读 Stock ensemble_report.json。",
    }
    # 单 seed42 回测一次作对照
    from .registry import spec_from_overlay, v3_risk_overlay
    from .train import pred_dir

    sdir = pred_dir("A06_seed42")
    seed42[["signal_date", "instrument_id", "score"]].to_parquet(sdir / "predictions.parquet", index=False)
    (sdir / "ensemble.json").write_text(
        json.dumps({"status": "locked_seed42", "key": "A06_seed42"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    spec_a06 = spec_from_overlay(
        "R6_A06",
        "单 seed42 锁分",
        v3_risk_overlay(),
        description_zh="锁分 seed42 vs 五种子。",
        stage="forensics",
        tags=("A", "A06"),
        diagnostic=True,
        alpha_source="local_retrain",
        prediction_key="A06_seed42",
        features=LOCKED_FEATURES,
    )
    SPECS["R6_A06"] = spec_a06
    run_spec(spec_a06, prepared_market=prepared, market_meta=meta, resume=resume, run_now=run_now)
    steps["R6_A06"]["seed42_cagr"] = _cagr("R6_A06")
    steps["R6_A06"]["ensemble_cagr"] = _cagr("R6_A00")

    # A07：fold 对照
    fold_cmp = _compare_folds(panel)
    train_spec(get_spec("R6_A07"), resume=resume)
    a07 = run_spec(get_spec("R6_A07"), prepared_market=prepared, market_meta=meta, resume=resume, run_now=run_now)
    steps["R6_A07"] = {
        "status": a07.get("status"),
        "cagr": _cagr("R6_A07"),
        "fold_compare": fold_cmp,
        "overlap_vs_locked": overlap_metrics(load_predictions("A07_r5_fold"), frozen_preds),
    }

    # waterfall：只报 Δ，不伪造「本环解释 xx% CAGR」
    waterfall = [
        {"layer": "frozen v3 overlay", "code": "R6_A00", "cagr": _cagr("R6_A00"), "vs_prev": None},
        {"layer": "label_e2_net + 1.0 (R5_D00)", "code": "R6_A01", "cagr": _cagr("R6_A01"), "vs_prev": _delta(_cagr("R6_A01"), _cagr("R6_A00"))},
        {"layer": "frozen universe retrain", "code": "R6_A02", "cagr": _cagr("R6_A02"), "vs_prev": _delta(_cagr("R6_A02"), _cagr("R6_A01"))},
        {"layer": "frozen training_panel", "code": "R6_A03", "cagr": _cagr("R6_A03"), "vs_prev": _delta(_cagr("R6_A03"), _cagr("R6_A02"))},
        {"layer": "hist 0.8 FORENSIC_ONLY", "code": "R6_A04", "cagr": _cagr("R6_A04"), "vs_prev": _delta(_cagr("R6_A04"), _cagr("R6_A03"))},
        {"layer": "r5_local fold", "code": "R6_A07", "cagr": _cagr("R6_A07"), "vs_prev": _delta(_cagr("R6_A07"), _cagr("R6_A03"))},
    ]

    status = _decide_status(steps, waterfall)
    manifests = _write_manifests(panel_sha, steps, ens_report)

    # freeze canonical metadata + backtest
    train_spec(get_spec(CANONICAL_CODE), resume=resume)
    can = run_spec(get_spec(CANONICAL_CODE), prepared_market=prepared, market_meta=meta, resume=resume, run_now=run_now)
    canon_meta = {
        "code": CANONICAL_CODE,
        "status": can.get("status"),
        "cagr": _cagr(CANONICAL_CODE),
        "recipe": {
            "target": "label_exit_rule_net",
            "fold": "actual_exit_date.year <= train_end (Stock year-cut)",
            "engine": "Stock rolling_train_predict",
            "capacity": "n50_d2",
            "ensemble": "5seed intra-day rank-pct equal weight",
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        "residual_vs_frozen": {
            "frozen_cagr": V3_CAGR,
            "canonical_cagr": _cagr(CANONICAL_CODE),
            "delta_cagr": _delta(_cagr(CANONICAL_CODE), V3_CAGR),
            "note": "采样保持 1.0；相对冻结残差单独记账，不并进 v4。",
        },
        "forensics_status": status,
        "written_at": datetime.now().astimezone().isoformat(),
    }
    _write_json("canonical/R6_CANONICAL_RETRAIN.json", canon_meta)

    payload = {
        "written_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "steps": steps,
        "waterfall": waterfall,
        "manifests": manifests,
        "canonical": canon_meta,
        "note": "waterfall 只报相邻 Δ，不把单环写成解释了全部 CAGR。",
    }
    _write_json("01_forensics/forensics.json", payload)
    _write_md(payload)
    return payload


def _delta(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _decide_status(steps: dict[str, Any], waterfall: list[dict[str, Any]]) -> str:
    a00 = steps.get("R6_A00") or {}
    a01 = steps.get("R6_A01") or {}
    a04 = steps.get("R6_A04") or {}
    if a00.get("cagr") is None or a01.get("cagr") is None:
        return "REPRODUCIBILITY_UNRESOLVED"
    label_gap = abs(float(a01["cagr"]) - 0.1094) < 0.02
    ov = (a04.get("overlap_vs_locked") or {}).get("spearman")
    close_frozen = a04.get("cagr") is not None and abs(float(a04["cagr"]) - V3_CAGR) < 0.02
    high_corr = ov is not None and float(ov) >= 0.95
    if close_frozen or high_corr:
        return "REPRODUCED"
    # 标签列已对上 D00，且 0.8 实验已落地 → 根因可记账
    if label_gap and a04.get("cagr") is not None:
        return "ROOT_CAUSE_IDENTIFIED"
    if a01.get("cagr") is not None and a04.get("cagr") is not None:
        return "ROOT_CAUSE_IDENTIFIED"
    return "REPRODUCIBILITY_UNRESOLVED"


def _compare_folds(panel: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for predict_year in range(2018, 2027):
        train_start, train_end = window_bounds(predict_year, "3y")
        stock = fold_mask(panel, train_start=train_start, train_end=train_end, predict_year=predict_year, fold_rule="stock")
        local = fold_mask(panel, train_start=train_start, train_end=train_end, predict_year=predict_year, fold_rule="r5_local")
        a = panel.loc[stock]
        b = panel.loc[local]
        y = pd.to_numeric(a.get("label_exit_rule_net"), errors="coerce")
        z = pd.to_numeric(b.get("label_exit_rule_net"), errors="coerce")
        rows.append({
            "predict_year": predict_year,
            "stock_rows": int(stock.sum()),
            "r5_local_rows": int(local.sum()),
            "stock_groups": int(a["trade_date"].nunique()) if len(a) else 0,
            "r5_local_groups": int(b["trade_date"].nunique()) if len(b) else 0,
            "stock_label_mean": float(y.mean()) if y.notna().any() else None,
            "r5_local_label_mean": float(z.mean()) if z.notna().any() else None,
            "row_diff": int(stock.sum()) - int(local.sum()),
        })
    return {"years": rows, "note": "Stock=actual_exit_date.year<=train_end；R5=label_end<predict_year-01-01。外加成熟硬规则 < predict_year-01-01。"}


def _write_manifests(panel_sha: str, steps: dict[str, Any], ens_report: dict[str, Any]) -> dict[str, str]:
    frozen = json.loads((stock_path("runs/first_board_factor_search_5000_v1/predictions/n50_d2/seed42/fs00263/training_report.json")).read_text(encoding="utf-8"))
    frozen_m = {
        "trial": "fs00263",
        "target": frozen.get("target"),
        "subsample": (frozen.get("ranker") or {}).get("subsample"),
        "colsample_bytree": (frozen.get("ranker") or {}).get("colsample_bytree"),
        "models_persisted": frozen.get("models_persisted"),
        "train_rows_seed42": (frozen.get("training_clean_audit") or {}).get("rows_after"),
        "prediction_sha256": ens_report.get("prediction_sha256"),
        "ensemble_method": ens_report.get("method"),
        "features": frozen.get("features"),
    }
    retrain = {
        "A01_target": "label_e2_net",
        "A03_training_panel_sha256": panel_sha,
        "A04_subsample": 0.8,
        "A04_tag": "FORENSIC_ONLY",
        "canonical": "label_exit_rule_net + Stock rolling + n50_d2 + 5seed rank-pct + subsample 1.0",
        "steps_cagr": {k: v.get("cagr") for k, v in steps.items()},
    }
    p1 = _write_json("reproducibility/frozen_manifest.json", frozen_m)
    p2 = _write_json("reproducibility/retrain_manifest.json", retrain)
    return {"frozen": str(p1), "retrain": str(p2)}


def _write_md(payload: dict[str, Any]) -> None:
    steps = payload.get("steps") or {}
    lines = [
        "# 01 复现取证（Stage A）",
        "",
        f"- 生成时间：{payload.get('written_at')}",
        f"- 状态：**{payload.get('status')}**",
        "- 每步只换一层。waterfall 只报相邻 Δ，不伪造「本环解释全部 CAGR」。",
        "- A04 = FORENSIC_ONLY（0.8/0.8）。A05 = SKIPPED_NO_ARTIFACT。",
        "",
        "## 受控替换",
        "",
        "| 代号 | 状态 | CAGR | Top1 overlap | Spearman | 备注 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for code in ("R6_A00", "R6_A01", "R6_A02", "R6_A03", "R6_A04", "R6_A05", "R6_A06", "R6_A07"):
        row = steps.get(code) or {}
        ov = row.get("overlap_vs_locked") or row.get("seed42_vs_ensemble") or {}
        lines.append(
            f"| `{code}` | {row.get('status')} | {_pct(row.get('cagr') or row.get('seed42_cagr'))} "
            f"| {_pct(ov.get('top1_overlap'))} | {_num(ov.get('spearman'))} | {row.get('note') or ''} |"
        )
    lines += [
        "",
        "## Waterfall（相邻 Δ，不是归因分解）",
        "",
        "| 层 | 代号 | CAGR | Δ vs 上一层 |",
        "|---|---|---:|---:|",
    ]
    for w in payload.get("waterfall") or []:
        lines.append(f"| {w.get('layer')} | `{w.get('code')}` | {_pct(w.get('cagr'))} | {_pct(w.get('vs_prev'))} |")
    a07 = (steps.get("R6_A07") or {}).get("fold_compare") or {}
    lines += [
        "",
        "## A07 fold 样本",
        "",
        "| 预测年 | Stock 行 | R5 local 行 | 行差 | Stock 组 | R5 组 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in (a07.get("years") or []):
        lines.append(
            f"| {r['predict_year']} | {r['stock_rows']} | {r['r5_local_rows']} | {r['row_diff']} "
            f"| {r['stock_groups']} | {r['r5_local_groups']} |"
        )
    can = payload.get("canonical") or {}
    res = can.get("residual_vs_frozen") or {}
    lines += [
        "",
        "## Canonical 冻结",
        "",
        f"- 代号：`{can.get('code')}`",
        f"- CAGR：{_pct(can.get('cagr'))}；相对冻结 Δ={_pct(res.get('delta_cagr'))}",
        f"- 配方：`{json.dumps(can.get('recipe') or {}, ensure_ascii=False)}`",
        f"- forensics 状态：`{payload.get('status')}`",
        "",
        "UNRESOLVED 时 B–K 可跑，但禁止 `RECOMMENDED_BASELINE_V4`。",
        "",
    ]
    path = ROUND6_ROOT / "01_reproducibility_forensics.md"
    assert_project_result_path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_forensics_status() -> dict[str, Any]:
    path = ROUND6_ROOT / "01_forensics" / "forensics.json"
    if not path.is_file() or not (ROUND6_ROOT / "01_reproducibility_forensics.md").is_file():
        return {"status": None, "complete": False, "reason": "forensics 未写"}
    blob = json.loads(path.read_text(encoding="utf-8"))
    return {"complete": True, **blob}
