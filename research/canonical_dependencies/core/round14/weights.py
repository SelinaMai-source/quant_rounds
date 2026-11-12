"""A01/B01/B02/B03/C01 目标权重 + CTRL_EXP。身份锁。禁止笛卡尔积。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND14_ROOT, assert_project_result_path, ensure_round14_tree
from ..round8.overlays import MIN_KEEP
from ..round9.features import build_daily_frame
from ..round13.weights import extract_top_weights, inject_weights
from .common import dump_json, load_json, write_md
from .contract import (
    A01,
    B01,
    B02,
    B03,
    C01,
    CANDIDATE_IDS,
    FAIL_ALPHA_MUTATION,
    RANK_NAV,
    WANT_TOP2_EPS,
    ctrl_code,
    overlayed_path,
)
from .freeze import require_frozen
from .risk import load_risk_panel
from .runner import build_version_overlayed
from .tests import run_unit_tests

WEIGHTS_PARQUET = ROUND14_ROOT / "daily_target_weights.parquet"
WEIGHTS_JSON = ROUND14_ROOT / "weights" / "weights_audit.json"
WEIGHTS_MD = ROUND14_ROOT / "02_weights.md"
FRAC = {1: float(RANK_NAV[0]), 2: float(RANK_NAV[1])}


def _row(date, inst, rank, w, source: str) -> dict[str, Any]:
    frac = FRAC[int(rank)]
    if not np.isfinite(w) or w <= 0:
        w = 0.0
    mult = float(np.clip(w / frac, MIN_KEEP, 1.0)) if frac > 0 else MIN_KEEP
    if w <= 0:
        mult = MIN_KEEP
        w = frac * MIN_KEEP
    return {
        "signal_date": date,
        "instrument_id": str(inst),
        "engine_rank": int(rank),
        "target_nav_fraction": float(w),
        "allocation_multiplier": float(mult),
        "source": source,
    }


def _a01_w2(agree: str, p: float, w2_v4: float) -> float:
    if not np.isfinite(p):
        return float(w2_v4)
    p = float(np.clip(p, 0.0, 1.0))
    a = str(agree)
    if a == "C":
        return 0.0
    if a == "A":
        if p <= 0.20:
            return 0.0
        if p <= 0.50:
            return 0.05 * (p - 0.20) / 0.30
        return 0.05 + 0.05 * (p - 0.50) / 0.50
    if a == "B":
        if p <= 0.50:
            return 0.0
        return 0.05 * (p - 0.50) / 0.50
    return float(w2_v4)


def _cap_down(base: float, numer: float, denom: float) -> float:
    if not (np.isfinite(numer) and np.isfinite(denom) and denom > 0):
        return float(base)
    return float(base) * float(min(1.0, numer / denom))


def _mrc_from_weights(w1: float, w2: float, s1: float, s2: float, rho: float, rho_ok: bool) -> float:
    from .risk import _sigma_mrc

    pack = _sigma_mrc(w1, w2, s1, s2, rho, rho_ok)
    return float(pack["MRC2"])


def build_candidate_weights(v4_top: pd.DataFrame, panel: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    v4 = v4_top.copy()
    v4["signal_date"] = pd.to_datetime(v4["signal_date"]).dt.normalize()
    wide_rows = []
    for day, g in v4.groupby("signal_date", sort=False):
        g = g.sort_values("engine_rank")
        rec: dict[str, Any] = {"signal_date": pd.Timestamp(day).normalize()}
        for _, r in g.iterrows():
            rk = int(r["engine_rank"])
            rec[f"id{rk}"] = str(r["instrument_id"])
            rec[f"w{rk}"] = float(r["target_nav_fraction"])
        wide_rows.append(rec)
    wide = pd.DataFrame(wide_rows)
    p = panel.copy()
    p["signal_date"] = pd.to_datetime(p["signal_date"]).dt.normalize()
    drop_p = [c for c in ("id1", "id2", "w1_v4", "w2_v4") if c in p.columns]
    p = p.drop(columns=drop_p)
    g = geo.copy()
    if "signal_date" not in g.columns:
        g = g.reset_index()
    g["signal_date"] = pd.to_datetime(g["signal_date"]).dt.normalize()
    keep_geo = [c for c in ("signal_date", "agreement", "top2_minus_median_p250") if c in g.columns]
    merged = wide.merge(p, on="signal_date", how="left").merge(g[keep_geo], on="signal_date", how="left")
    if "id1" not in merged.columns:
        raise RuntimeError(f"missing id1 after merge; columns={list(merged.columns)}")
    merged["w1"] = pd.to_numeric(merged.get("w1"), errors="coerce").fillna(0.0)
    merged["w2"] = pd.to_numeric(merged.get("w2"), errors="coerce").fillna(0.0)
    merged["id1"] = merged["id1"].astype(str)
    merged["id2"] = merged.get("id2", pd.Series("", index=merged.index)).astype(str)

    c01_mrc = []
    for rec in merged.itertuples(index=False):
        agree = getattr(rec, "agreement", "")
        pval = float(getattr(rec, "top2_minus_median_p250", np.nan)) if pd.notna(getattr(rec, "top2_minus_median_p250", np.nan)) else float("nan")
        w2_s = _a01_w2(str(agree), pval, float(rec.w2))
        s1 = float(getattr(rec, "sigma1_20", np.nan))
        s2 = float(getattr(rec, "sigma2_20", np.nan))
        rho = float(getattr(rec, "rho12_60", np.nan))
        rho_ok = bool(getattr(rec, "rho_available", False))
        c01_mrc.append(_mrc_from_weights(float(rec.w1), w2_s, s1, s2, rho, rho_ok))
    merged["w2_smooth"] = [
        _a01_w2(
            str(getattr(rec, "agreement", "")),
            float(getattr(rec, "top2_minus_median_p250", np.nan)) if pd.notna(getattr(rec, "top2_minus_median_p250", np.nan)) else float("nan"),
            float(rec.w2),
        )
        for rec in merged.itertuples(index=False)
    ]
    merged["C01_MRC2"] = c01_mrc
    from .risk import _rolling_median
    from .contract import REF_MIN_OBS, REF_SIGNAL_DAYS

    merged["median_C01_MRC2_250"] = _rolling_median(
        np.asarray(merged["C01_MRC2"], dtype=float), REF_SIGNAL_DAYS, REF_MIN_OBS
    )

    rows: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    for rec in merged.itertuples(index=False):
        date = rec.signal_date
        id1, id2 = str(rec.id1), str(rec.id2)
        w1_v4, w2_v4 = float(rec.w1), float(rec.w2)
        w2_s = float(rec.w2_smooth)
        fb_s2 = bool(getattr(rec, "fallback_sigma2", True))
        fb_mrc = bool(getattr(rec, "fallback_mrc", True))
        fb_port = bool(getattr(rec, "fallback_port", True))
        mrc = float(getattr(rec, "MRC2", np.nan))
        med_s2 = float(getattr(rec, "median_sigma2_250", np.nan))
        s2 = float(getattr(rec, "sigma2_20", np.nan))
        med_mrc = float(getattr(rec, "median_MRC2_250", np.nan))
        med_sp = float(getattr(rec, "median_sigma_portfolio_250", np.nan))
        sp = float(getattr(rec, "sigma_portfolio", np.nan))
        c01_m = float(rec.C01_MRC2)
        med_c01 = float(rec.median_C01_MRC2_250)

        w1_a01, w2_a01 = w1_v4, w2_s
        w1_b01, w2_b01 = w1_v4, (w2_v4 if fb_s2 else _cap_down(w2_v4, med_s2, s2))
        if (not np.isfinite(mrc)) or (not np.isfinite(med_mrc)):
            w2_b02 = w2_v4
        elif mrc <= 0:
            w2_b02 = w2_v4
        else:
            w2_b02 = _cap_down(w2_v4, med_mrc, mrc)
        w1_b02 = w1_v4
        if fb_port:
            w1_b03, w2_b03 = w1_v4, w2_v4
            port_f = 1.0
        else:
            port_f = float(min(1.0, med_sp / sp)) if (np.isfinite(med_sp) and np.isfinite(sp) and sp > 0) else 1.0
            w1_b03, w2_b03 = w1_v4 * port_f, w2_v4 * port_f
        if (not np.isfinite(c01_m)) or (not np.isfinite(med_c01)):
            w2_c01 = w2_s
        elif c01_m <= 0:
            w2_c01 = w2_s
        else:
            w2_c01 = _cap_down(w2_s, med_c01, c01_m)
        w1_c01 = w1_v4

        pack = {
            A01: (w1_a01, w2_a01),
            B01: (w1_b01, w2_b01),
            B02: (w1_b02, w2_b02),
            B03: (w1_b03, w2_b03),
            C01: (w1_c01, w2_c01),
        }
        has_top2 = id2 not in {"nan", "None", "", "NaT"} and id2 != id1
        for cid, (ww1, ww2) in pack.items():
            rows.append(_row(date, id1, 1, ww1, cid))
            if has_top2:
                rows.append(_row(date, id2, 2, ww2, cid))
        e_v4 = w1_v4 + w2_v4
        for cid, (ww1, ww2) in pack.items():
            e_c = ww1 + ww2
            scale = (e_c / e_v4) if e_v4 > WANT_TOP2_EPS else 1.0
            rows.append(_row(date, id1, 1, w1_v4 * scale, ctrl_code(cid)))
            if has_top2:
                rows.append(_row(date, id2, 2, w2_v4 * scale, ctrl_code(cid)))
        flags.append({
            "signal_date": str(pd.Timestamp(date).date()),
            "fallback_sigma2": fb_s2,
            "fallback_mrc": fb_mrc,
            "fallback_port": fb_port,
            "port_factor": port_f if not fb_port else 1.0,
        })
    return pd.DataFrame(rows), flags


def run_weights(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    ensure_round14_tree()
    if resume and WEIGHTS_PARQUET.is_file() and WEIGHTS_JSON.is_file():
        payload = load_json(WEIGHTS_JSON)
        if payload.get("status") == "DONE":
            return payload
    ov4, _, _ = build_version_overlayed("v4")
    v4_top = extract_top_weights(ov4, "v4")
    panel = load_risk_panel()
    if panel.empty:
        raise RuntimeError("RISK_PANEL_MISSING STOP")
    geo = build_daily_frame()
    cand, flags = build_candidate_weights(v4_top, panel, geo)
    all_w = pd.concat([v4_top, cand], ignore_index=True)
    assert_project_result_path(WEIGHTS_PARQUET)
    all_w.to_parquet(WEIGHTS_PARQUET, index=False)

    overlayed_map: dict[str, pd.DataFrame] = {}
    dest = overlayed_path("v4")
    assert_project_result_path(dest)
    ov4.to_parquet(dest, index=False)
    overlayed_map["v4"] = ov4
    for src in [*CANDIDATE_IDS, *[ctrl_code(c) for c in CANDIDATE_IDS]]:
        sub = cand.loc[cand["source"].eq(src)].drop_duplicates(["signal_date", "instrument_id"], keep="first")
        inj = inject_weights(ov4, sub)
        dest = overlayed_path(src)
        assert_project_result_path(dest)
        inj.to_parquet(dest, index=False)
        overlayed_map[src] = inj

    tests = run_unit_tests(ov4, {c: overlayed_map[c] for c in CANDIDATE_IDS})
    failed_ids = [c for c, row in (tests.get("identity") or {}).items() if not row.get("passed")]
    n_days = int(v4_top["signal_date"].nunique())
    payload = {
        "status": "DONE",
        "path": str(WEIGHTS_PARQUET),
        "n_rows": int(len(all_w)),
        "n_days": n_days,
        "sources": sorted(all_w["source"].astype(str).unique().tolist()),
        "tests": tests,
        "failed_identity": failed_ids,
        "fallback_share": {
            "sigma2": float(np.mean([f["fallback_sigma2"] for f in flags])) if flags else None,
            "mrc": float(np.mean([f["fallback_mrc"] for f in flags])) if flags else None,
            "port": float(np.mean([f["fallback_port"] for f in flags])) if flags else None,
        },
        "never_average_two_navs": True,
        "cartesian": False,
        "min_keep_zero_top2": True,
    }
    dump_json(WEIGHTS_JSON, payload)
    write_md(
        WEIGHTS_MD,
        "\n".join([
            "# Round 14 · 日度目标权重",
            "",
            "只改资金，不改身份。零 Top2 写 MIN_KEEP。",
            f"- 交易日数 = **{n_days}**",
            f"- 身份锁失败 = {failed_ids or '无'}",
            f"- 零 Top2 单测 = **{(tests.get('zero_top2') or {}).get('status')}**",
            f"- fallback sigma2/mrc/port = {payload['fallback_share']}",
            "",
        ]),
    )
    if failed_ids:
        payload["identity_status"] = FAIL_ALPHA_MUTATION
    return payload


def load_weights_table() -> pd.DataFrame:
    if not WEIGHTS_PARQUET.is_file():
        return pd.DataFrame()
    return pd.read_parquet(WEIGHTS_PARQUET)


def load_overlayed(code: str) -> pd.DataFrame:
    path = overlayed_path(code)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_weights_audit() -> dict[str, Any]:
    return load_json(WEIGHTS_JSON)


__all__ = [
    "build_candidate_weights",
    "load_overlayed",
    "load_weights_audit",
    "load_weights_table",
    "run_weights",
]
