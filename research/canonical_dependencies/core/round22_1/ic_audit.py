"""Track A：IC 宇宙审计。宇宙=一进二 candidate_panel，不是全市场。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND22_1_ROOT, ROUND22_ROOT
from ..round21.panel import load_candidate_keys
from .common import dump_json, load_json, write_csv, write_md
from .contract import (
    IC_AGG_MIN,
    IC_DAILY_MIN,
    IC_THRESHOLDS,
    KEYS,
    LABELS,
    MD,
    NOT_IDENTIFIABLE,
    PIT_PASS_IDS,
)
from .freeze import require_frozen_gate

IC_JSON = ROUND22_1_ROOT / "ic" / "ic_audit.json"


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    if int(x.nunique(dropna=True)) < 2 or int(y.nunique(dropna=True)) < 2:
        return None
    rho = x.corr(y, method="spearman")
    if rho is None or not np.isfinite(rho):
        return None
    return float(rho)


def _daily_rows(frame: pd.DataFrame, factor: str, label: str, *, universe: str) -> list[dict[str, Any]]:
    work = frame[["trade_date", factor, label]].copy()
    work[factor] = pd.to_numeric(work[factor], errors="coerce")
    work[label] = pd.to_numeric(work[label], errors="coerce")
    rows: list[dict[str, Any]] = []
    for day, block in work.groupby("trade_date", sort=True):
        n_candidates = int(len(block))
        n_factor_valid = int(block[factor].notna().sum())
        n_label_valid = int(block[label].notna().sum())
        sub = block.dropna(subset=[factor, label])
        n_pairs = int(len(sub))
        n_unique_factor = int(sub[factor].nunique()) if n_pairs else 0
        n_unique_label = int(sub[label].nunique()) if n_pairs else 0
        rho = _spearman(sub[factor], sub[label]) if n_pairs >= 2 else None
        if n_pairs < IC_DAILY_MIN:
            eligibility = "SKIP_THIN_DAY"
        elif rho is None:
            eligibility = "SKIP_DEGENERATE"
        else:
            eligibility = "ELIGIBLE"
        rows.append(
            {
                "trade_date": str(pd.Timestamp(day).date()),
                "year": int(pd.Timestamp(day).year),
                "factor_id": factor,
                "label": label,
                "universe": universe,
                "n_candidates": n_candidates,
                "n_factor_valid": n_factor_valid,
                "n_label_valid": n_label_valid,
                "n_pairs": n_pairs,
                "n_unique_factor": n_unique_factor,
                "n_unique_label": n_unique_label,
                "rho": rho,
                "eligibility": eligibility,
            }
        )
    return rows


def _aggregate(daily: pd.DataFrame, *, min_days: int) -> dict[str, Any]:
    elig = daily.loc[daily["eligibility"].eq("ELIGIBLE")].copy()
    arr = pd.to_numeric(elig["rho"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) < int(min_days):
        return {
            "status": NOT_IDENTIFIABLE,
            "n_days": int(len(arr)),
            "mean": None,
            "std": None,
            "icir": None,
            "by_year": {},
            "min_days": int(min_days),
            "nan_treated_as_zero": False,
        }
    std = float(arr.std(ddof=1)) if len(arr) >= 2 else None
    mean = float(arr.mean())
    icir = float(mean / std) if std is not None and std > 0 else None
    by_year: dict[str, float] = {}
    for y, block in elig.groupby("year"):
        vals = pd.to_numeric(block["rho"], errors="coerce").dropna()
        if len(vals):
            by_year[str(int(y))] = float(vals.mean())
    return {
        "status": "OK",
        "n_days": int(len(arr)),
        "mean": mean,
        "std": std,
        "icir": icir,
        "icir_annualized": False,
        "icir_ddof": 1,
        "by_year": by_year,
        "by_year_clipped_2018": False,
        "min_days": int(min_days),
        "nan_treated_as_zero": False,
    }


def run_ic_audit(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and IC_JSON.is_file() and MD["01"].is_file():
        existing = load_json(IC_JSON)
        if existing.get("status") in {"DONE", "COMPLETE"}:
            return existing
    panel_path = ROUND22_ROOT / "data" / "factor_panel.parquet"
    if not panel_path.is_file():
        payload = {"status": "BLOCKED_DATA", "reason": f"missing read-only panel {panel_path}"}
        dump_json(IC_JSON, payload)
        return payload
    panel = pd.read_parquet(panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    labels = load_candidate_keys()
    keep = [c for c in [*KEYS, *LABELS, "next2_date", "actual_exit_date"] if c in labels.columns]
    cand = labels[keep].drop_duplicates(KEYS)
    cand["trade_date"] = pd.to_datetime(cand["trade_date"])
    cand["instrument_id"] = cand["instrument_id"].astype(str)
    base = panel.merge(cand, on=KEYS, how="inner")
    # 全市场对照：因子面板里未与候选键对齐的行（若有）
    market = panel.merge(cand[KEYS], on=KEYS, how="left", indicator=True)
    market_only = market.loc[market["_merge"].eq("left_only")].drop(columns=["_merge"], errors="ignore")

    daily_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    for fid in PIT_PASS_IDS:
        dest = ROUND22_1_ROOT / "factors" / fid
        dest.mkdir(parents=True, exist_ok=True)
        if fid not in base.columns:
            rec = {"factor_id": fid, "status": "MISSING_COLUMN"}
            summaries.append(rec)
            dump_json(dest / "ic.json", rec)
            continue
        rec: dict[str, Any] = {
            "factor_id": fid,
            "universe": "first_board_candidate",
            "universe_note": "一进二 candidate_panel 当日截面，不是全市场",
            "daily_skip": f"n_pairs < {IC_DAILY_MIN} → skip day, do not record IC=0",
            "aggregate_rule": f"eligible IC days < {IC_AGG_MIN} → {NOT_IDENTIFIABLE}",
            "diagnostic_matrix_error": "Round22 诊断矩阵误写每日横截面 n<60 标 NOT_IDENTIFIABLE；代码实际是日 n<5 跳过、累计日<60 才不可识别",
        }
        for label in LABELS:
            if label not in base.columns:
                rec[label] = {"status": NOT_IDENTIFIABLE, "reason": "label missing"}
                continue
            drows = _daily_rows(base, fid, label, universe="first_board_candidate")
            daily_rows.extend(drows)
            dframe = pd.DataFrame(drows)
            ic = _aggregate(dframe, min_days=IC_AGG_MIN)
            rec[label] = ic
            dump_json(dest / f"ic_{label}.json", ic)
            for thr in IC_THRESHOLDS:
                sens = _aggregate(dframe, min_days=int(thr))
                sensitivity.append(
                    {
                        "factor_id": fid,
                        "label": label,
                        "universe": "first_board_candidate",
                        "min_eligible_days": int(thr),
                        "status": sens["status"],
                        "n_days": sens["n_days"],
                        "mean": sens["mean"],
                        "icir": sens["icir"],
                        "selected_as_best": False,
                    }
                )
            if len(market_only) and fid in market_only.columns and label in cand.columns:
                # 全市场对照：用因子面板全部行（含非候选）对同一标签，仅诊断
                m = panel.merge(cand[[*KEYS, label]], on=KEYS, how="left")
                md = pd.DataFrame(_daily_rows(m, fid, label, universe="full_panel_diagnostic"))
                magg = _aggregate(md, min_days=IC_AGG_MIN)
                rec[f"{label}_full_panel_diagnostic"] = magg
                sensitivity.append(
                    {
                        "factor_id": fid,
                        "label": label,
                        "universe": "full_panel_diagnostic",
                        "min_eligible_days": IC_AGG_MIN,
                        "status": magg["status"],
                        "n_days": magg["n_days"],
                        "mean": magg["mean"],
                        "icir": magg["icir"],
                        "selected_as_best": False,
                        "not_primary": True,
                    }
                )
        rec["status"] = "DONE"
        summaries.append(rec)
        dump_json(dest / "ic.json", rec)

    daily_df = pd.DataFrame(daily_rows)
    write_csv(ROUND22_1_ROOT / "ic" / "ic_universe_audit.csv", daily_rows)
    write_csv(ROUND22_1_ROOT / "ic" / "ic_threshold_sensitivity.csv", sensitivity)
    if len(daily_df):
        daily_df.to_parquet(ROUND22_1_ROOT / "ic" / "daily_ic_input.parquet", index=False)

    # 标签成熟
    mat_rows = []
    for label, end_col in (("label_e2_net", "next2_date"), ("label_exit_rule_net", "actual_exit_date")):
        if label not in cand.columns:
            mat_rows.append({"label": label, "status": "MISSING"})
            continue
        end = pd.to_datetime(cand[end_col], errors="coerce") if end_col in cand.columns else pd.Series(pd.NaT, index=cand.index)
        years = pd.to_datetime(cand["trade_date"]).dt.year
        mature = end.notna() & cand[label].notna()
        mat_rows.append(
            {
                "label": label,
                "end_col": end_col,
                "n": int(len(cand)),
                "n_label_finite": int(cand[label].notna().sum()),
                "n_end_finite": int(end.notna().sum()),
                "n_mature": int(mature.sum()),
                "share_mature": float(mature.mean()) if len(cand) else None,
                "min_trade_date": str(pd.Timestamp(cand["trade_date"].min()).date()),
                "max_trade_date": str(pd.Timestamp(cand["trade_date"].max()).date()),
                "n_2013_2017": int(((years >= 2013) & (years <= 2017)).sum()),
                "n_2018_plus": int((years >= 2018).sum()),
                "ic_includes_2013_2017": bool(((years >= 2013) & (years <= 2017)).any()),
                "train_cutoff_note": "训练折成熟规则：predict_year-3..predict_year-1 且 end_col.year<=train_end",
            }
        )
    write_csv(ROUND22_1_ROOT / "ic" / "label_maturity_audit.csv", mat_rows)
    write_md(MD["16"], "# Round22.1 · 16 标签成熟\n\n训练 cutoff 前 `label_e2_net`/`label_exit_rule_net` 按 next2_date / actual_exit_date 计成熟。IC 日表未裁 2013–2017；by_year 全量保留。\n")

    payload = {
        "status": "DONE",
        "universe": "first_board_candidate",
        "full_market_is_diagnostic_only": True,
        "daily_min_pairs": IC_DAILY_MIN,
        "aggregate_min_days": IC_AGG_MIN,
        "icir": "mean(daily_ic)/std(ddof=1), not annualized",
        "by_year_clipped_2018": False,
        "nan_as_zero": False,
        "n_factors": len(summaries),
        "rows": summaries,
        "n_daily_rows": int(len(daily_rows)),
        "maturity": mat_rows,
    }
    dump_json(IC_JSON, payload)
    dump_json(ROUND22_1_ROOT / "ic" / "ic.json", {"status": "DONE", "n": len(summaries), "rows": summaries})
    write_csv(
        ROUND22_1_ROOT / "ic" / "ic_results.csv",
        [
            {
                "factor_id": r["factor_id"],
                "universe": r.get("universe"),
                "e2_mean": (r.get("label_e2_net") or {}).get("mean") if isinstance(r.get("label_e2_net"), dict) else None,
                "e2_icir": (r.get("label_e2_net") or {}).get("icir") if isinstance(r.get("label_e2_net"), dict) else None,
                "e2_days": (r.get("label_e2_net") or {}).get("n_days") if isinstance(r.get("label_e2_net"), dict) else None,
                "e2_status": (r.get("label_e2_net") or {}).get("status") if isinstance(r.get("label_e2_net"), dict) else None,
                "exit_mean": (r.get("label_exit_rule_net") or {}).get("mean") if isinstance(r.get("label_exit_rule_net"), dict) else None,
                "exit_icir": (r.get("label_exit_rule_net") or {}).get("icir") if isinstance(r.get("label_exit_rule_net"), dict) else None,
                "exit_days": (r.get("label_exit_rule_net") or {}).get("n_days") if isinstance(r.get("label_exit_rule_net"), dict) else None,
                "exit_status": (r.get("label_exit_rule_net") or {}).get("status") if isinstance(r.get("label_exit_rule_net"), dict) else None,
            }
            for r in summaries
        ],
    )
    write_md(
        MD["01"],
        "\n".join(
            [
                "# Round22.1 · 01 IC 口径审计",
                "",
                "主宇宙=`first_board_candidate`（一进二 `candidate_panel` 当日截面）。全市场只作对照诊断，不进主列。",
                f"日规则：有效对 `< {IC_DAILY_MIN}` 跳过该日，NaN 不当 0。",
                f"聚合：累计有效 IC 日 `< {IC_AGG_MIN}` → `{NOT_IDENTIFIABLE}`。",
                "Round22 诊断矩阵误写「每日横截面 n<60 标 NOT_IDENTIFIABLE」。`core/round22/evidence.py` `_daily_ic` 实际是日 n<5 跳过、累计日<60 才不可识别。",
                "ICIR=`mean(daily_ic)/std(ddof=1)`，未年化。`by_year` 未裁到 2018+。",
                "门槛 `{5,10,20,60}` 是预注册敏感性，不是选最佳。",
                f"日级输入表 `{ROUND22_1_ROOT / 'ic' / 'ic_universe_audit.csv'}`。",
                "",
            ]
        ),
    )
    return payload
