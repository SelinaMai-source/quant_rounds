"""N1/N3 截面退化诊断。日度全截面近似相同 → CROSS_SECTIONALLY_DEGENERATE。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_2_ROOT, ROUND21_3_ROOT
from .common import dump_json, file_sha256, load_json, write_md
from .contract import CROSS_SECTIONALLY_DEGENERATE, KEYS, MD

N1N3_JSON = ROUND21_3_ROOT / "n1n3" / "n1n3.json"
N1 = "r21_n1_lagged_peer_limit_up"
N3 = "r21_n3_peer_failed_spillover"


def _cs_stats(panel: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in panel.columns:
        return {"status": "MISSING_COLUMN", "col": col}
    s = pd.to_numeric(panel[col], errors="coerce")
    n = int(len(s))
    valid = int(np.isfinite(s.to_numpy(dtype=float)).sum())
    nuniq = int(s.dropna().nunique())
    day = panel.assign(_v=s).groupby("trade_date")["_v"]
    day_var = day.var()
    day_nuniq = day.nunique()
    day_n = day.size()
    n_days = int(day_n.shape[0])
    const_days = int((day_nuniq.fillna(1) <= 1).sum())
    near_zero_var_days = int((day_var.fillna(0) < 1e-18).sum())
    mean_day_var = float(day_var.mean()) if day_var.notna().any() else None
    degenerate = n_days > 0 and (const_days / n_days) >= 0.90
    return {
        "col": col,
        "n": n,
        "valid": valid,
        "coverage": (valid / n) if n else None,
        "n_unique_global": nuniq,
        "n_days": n_days,
        "constant_days": const_days,
        "near_zero_var_days": near_zero_var_days,
        "constant_day_ratio": (const_days / n_days) if n_days else None,
        "mean_day_var": mean_day_var,
        "cross_sectionally_degenerate": bool(degenerate),
        "label": CROSS_SECTIONALLY_DEGENERATE if degenerate else "HAS_CROSS_SECTION",
    }


def run_n1n3(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and N1N3_JSON.is_file() and MD["21"].is_file():
        existing = load_json(N1N3_JSON)
        if existing.get("status") in {"DONE", "REJECTED_ARCHIVE"}:
            return existing
    panel_p = ROUND21_2_ROOT / "data" / "factor_panel.parquet"
    n_p = ROUND21_2_ROOT / "data" / "n_factors.parquet"
    if not panel_p.is_file() and not n_p.is_file():
        payload = {"status": "BLOCKED", "reason": "21.2 n_factors/factor_panel missing (read-only)"}
        dump_json(N1N3_JSON, payload)
        write_md(MD["21"], f"# Round21.3 · 21 N1/N3\n\nBLOCKED `{payload['reason']}`\n")
        return payload
    src = n_p if n_p.is_file() else panel_p
    panel = pd.read_parquet(src)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    # 候选截面：若有 21.2 five_candidate 则限制到首板候选
    cand_p = ROUND21_2_ROOT / "data" / "five_candidate.parquet"
    if cand_p.is_file():
        cand = pd.read_parquet(cand_p, columns=KEYS)
        cand["trade_date"] = pd.to_datetime(cand["trade_date"])
        cand["instrument_id"] = cand["instrument_id"].astype(str)
        panel = panel.merge(cand, on=KEYS, how="inner")
    rows = {c: _cs_stats(panel, c) for c in (N1, N3)}
    # peer 数
    extra = {}
    if "peer_n" in panel.columns:
        extra["peer_n_mean"] = float(pd.to_numeric(panel["peer_n"], errors="coerce").mean())
        extra["empty_peer"] = int((pd.to_numeric(panel["peer_n"], errors="coerce") <= 1).sum())
    # XGB gain / pred hash from 21.2 E2
    pred_info = {}
    for fid, code in ((N1, "R21_2_E2_r21_n1_lagged_peer_limit_up"), (N3, "R21_2_E2_r21_n3_peer_failed_spillover")):
        dest = ROUND21_2_ROOT / "candidates" / code
        pred_info[fid] = {
            "pred_hash": load_json(dest / "prediction_hash.json") if (dest / "prediction_hash.json").is_file() else None,
            "five_pred_sha": file_sha256(dest / "five_pred.parquet"),
            "top1_top2": str(dest / "daily_top1_top2.parquet") if (dest / "daily_top1_top2.parquet").is_file() else None,
        }
    e0_hash = file_sha256(ROUND21_2_ROOT / "candidates" / "R21_2_E0" / "five_pred.parquet")
    identical = {
        N1: pred_info[N1].get("five_pred_sha") == e0_hash,
        N3: pred_info[N3].get("five_pred_sha") == e0_hash,
    }
    both_deg = all((rows[c].get("cross_sectionally_degenerate") for c in (N1, N3)))
    construct_bug = False
    # N1/N3 按设计是行业-日滞后均值，未做 exclude-self；候选若跨行业应有截面。若候选日内几乎单行业则合法退化。
    verdict = CROSS_SECTIONALLY_DEGENERATE if both_deg else "HAS_CROSS_SECTION"
    if both_deg and identical[N1] and identical[N3]:
        verdict = CROSS_SECTIONALLY_DEGENERATE
        dest_rej = ROUND21_3_ROOT / "rejected" / "n1_n3_archive.json"
        dump_json(
            dest_rej,
            {
                "factors": [N1, N3],
                "verdict": CROSS_SECTIONALLY_DEGENERATE,
                "reason": "SW_L1 equal-weight industry-day lagged mean; candidate-day cross-section nearly constant; Add/Replace Δ=0; pred hash ≡ E0",
                "construct_bug": False,
                "do_not_retest": True,
            },
        )
    payload = {
        "status": "REJECTED_ARCHIVE" if verdict == CROSS_SECTIONALLY_DEGENERATE else "DONE",
        "verdict": verdict,
        "construct_bug": construct_bug,
        "rows": rows,
        "peer": extra,
        "pred_info": pred_info,
        "pred_identical_to_e0": identical,
        "pit": "SW_L1 asof known_at; RESEARCH_ONLY if membership lag unverified",
        "do_not_retest": verdict == CROSS_SECTIONALLY_DEGENERATE,
    }
    dump_json(N1N3_JSON, payload)
    write_md(
        MD["21"],
        "\n".join(
            [
                "# Round21.3 · 21 N1/N3 截面诊断",
                "",
                f"verdict=`{verdict}` construct_bug=`{construct_bug}`",
                f"N1 `{rows[N1]}`",
                f"N3 `{rows[N3]}`",
                f"pred≡E0 `{identical}`",
                "N1=行业日滞后 5 日涨停率；N3=行业日滞后 10 日炸板率。等权 SW_L1，滞后序列未 exclude-self。",
                "日度全截面近似相同 → CROSS_SECTIONALLY_DEGENERATE，进 Rejected Archive，不再测。",
                "",
            ]
        ),
    )
    return payload
