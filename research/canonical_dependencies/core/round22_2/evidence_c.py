"""Track A 单因子：C1/C4 双标签 IC / Rolling / Shape / 相关 / ConditionalIC。只诊断，不改公式。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..round7.contract import FIVE_FEATURES, TEN_FEATURES
from ..round21.conditional import _daily_rankic, _fit_residual
from ..round21.evidence import _shape_one
from ..round21.panel import load_candidate_keys
from ..round22.evidence import _daily_ic
from .common import dump_json, finite_or_none, load_json, write_csv, write_md
from .contract import (
    C1_ID,
    C4_ID,
    C_PANEL_22_1,
    DIAG_IC_YEARS,
    FIVE_FEATURES as FIVE,
    FORMAL_IC_YEARS,
    KEYS,
    LABELS,
    MD,
    REDUNDANCY_RHO,
    RIDGE_ALPHA,
    ROUND22_2_ROOT,
    TEN_FEATURES as TEN,
)
from .freeze import require_frozen_gate

EV_JSON = ROUND22_2_ROOT / "evidence" / "c14.json"


def _year_mean(by_year: dict[str, Any], years: tuple[int, ...]) -> float | None:
    vals = [finite_or_none(by_year.get(str(y))) for y in years]
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _build_panel(pit: dict[str, Any]) -> pd.DataFrame:
    keys = load_candidate_keys()
    keep = [c for c in [*KEYS, *LABELS, "actual_exit_date", "next2_date"] if c in keys.columns]
    base = keys[keep].copy()
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    base["instrument_id"] = base["instrument_id"].astype(str)
    if not C_PANEL_22_1.is_file():
        raise RuntimeError("22.1 c_factor_panel missing; refuse to invent C values")
    c = pd.read_parquet(C_PANEL_22_1)
    c["trade_date"] = pd.to_datetime(c["trade_date"])
    c["instrument_id"] = c["instrument_id"].astype(str)
    lag = int((pit.get("c1") or {}).get("lag_days") or 0)
    model_id = (pit.get("c1") or {}).get("model_id") or C1_ID
    if lag > 0:
        c = c.sort_values(["instrument_id", "trade_date"])
        c[model_id] = c.groupby("instrument_id")[C1_ID].shift(lag)
    else:
        c[model_id] = c[C1_ID]
    cols = [c_ for c_ in (C1_ID, C4_ID, model_id) if c_ in c.columns]
    panel = base.merge(c[KEYS + cols].drop_duplicates(KEYS), on=KEYS, how="left")
    dest = ROUND22_2_ROOT / "data" / "c_factor_panel.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(dest, index=False)
    return panel


def _year_folds(dates: pd.Series) -> list[tuple[pd.Series, pd.Series]]:
    years = sorted({int(y) for y in pd.to_datetime(dates).dt.year.dropna()})
    out = []
    for y in years:
        train_years = [t for t in years if y - 3 <= t <= y - 1]
        if len(train_years) < 3:
            continue
        tr = dates[pd.to_datetime(dates).dt.year.isin(train_years)]
        te = dates[pd.to_datetime(dates).dt.year.eq(y)]
        if len(tr) and len(te):
            out.append((tr, te))
    return out


def _cs_corr(panel: pd.DataFrame, a: str, b: str) -> float | None:
    if a not in panel.columns or b not in panel.columns:
        return None
    rhos = []
    def _series(frame: pd.DataFrame, col: str) -> pd.Series:
        raw = frame[col]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        return pd.to_numeric(raw, errors="coerce")

    for _, block in panel.groupby("trade_date", sort=False):
        x = _series(block, a)
        y = _series(block, b)
        m = x.notna() & y.notna()
        if int(m.sum()) < 5:
            continue
        r = x.loc[m].corr(y.loc[m], method="spearman")
        if r is not None and np.isfinite(r):
            rhos.append(float(r))
    return float(np.mean(rhos)) if rhos else None


def _fold_rho(panel: pd.DataFrame, fid: str, locked: tuple[str, ...]) -> dict[str, Any]:
    dates = pd.to_datetime(panel["trade_date"])
    folds = _year_folds(dates)
    fx_raw = panel[fid]
    if isinstance(fx_raw, pd.DataFrame):
        fx_raw = fx_raw.iloc[:, 0]
    fx = pd.to_numeric(fx_raw, errors="coerce")
    best = {"nearest": None, "abs_rho": None}
    for col in locked:
        if col not in panel.columns:
            continue
        series = panel[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        gx = pd.to_numeric(series, errors="coerce")
        rhos = []
        for tr, _te in folds:
            tr_set = set(pd.to_datetime(tr))
            mask = dates.isin(tr_set) & fx.notna() & gx.notna()
            if int(mask.sum()) < 30:
                continue
            r = fx.loc[mask].corr(gx.loc[mask], method="spearman")
            if r is not None and np.isfinite(r):
                rhos.append(abs(float(r)))
        if rhos:
            m = float(np.mean(rhos))
            if best["abs_rho"] is None or m > best["abs_rho"]:
                best = {"nearest": col, "abs_rho": m}
    action = "REMOVE_REPLACE_ALLOWED" if (best["abs_rho"] or 0) >= REDUNDANCY_RHO else "ADD_ONLY"
    return {**best, "action": action, "threshold": REDUNDANCY_RHO}


def _conditional(panel: pd.DataFrame, fid: str, controls: list[str], label: str) -> dict[str, Any]:
    use = [c for c in controls if c in panel.columns]
    dates = pd.to_datetime(panel["trade_date"])
    folds = _year_folds(dates)
    ics = []
    n_skip = 0
    for tr, te in folds:
        tr_set = set(pd.to_datetime(tr))
        te_set = set(pd.to_datetime(te))
        train = panel.loc[dates.isin(tr_set), [*use, fid, label]].replace([np.inf, -np.inf], np.nan).dropna()
        test = panel.loc[dates.isin(te_set), [*use, fid, label, "trade_date"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(train) < 40 or len(test) < 20 or not use:
            n_skip += 1
            continue
        resid = _fit_residual(train[use].to_numpy(dtype=float), train[fid].to_numpy(dtype=float), test[use].to_numpy(dtype=float), test[fid].to_numpy(dtype=float))
        if resid is None:
            n_skip += 1
            continue
        ics.extend(_daily_rankic(test["trade_date"], resid, test[label].to_numpy(dtype=float)))
    arr = np.array(ics, dtype=float)
    if len(arr) < 60:
        return {"status": "NOT_IDENTIFIABLE", "n_oos_days": int(len(arr)), "n_folds_skipped": n_skip, "full_history_fit": False}
    return {
        "status": "OK",
        "n_oos_days": int(len(arr)),
        "n_folds_skipped": n_skip,
        "cond_rankic_mean": float(arr.mean()),
        "cond_rankicir": float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else None,
        "full_history_fit": False,
        "ridge_alpha": RIDGE_ALPHA,
        "n_controls": len(use),
        "controls": ",".join(use),
    }


def _shape(panel: pd.DataFrame, fid: str, label: str) -> dict[str, Any]:
    work = panel[["trade_date", fid, label]].copy()
    work[fid] = pd.to_numeric(work[fid], errors="coerce")
    work[label] = pd.to_numeric(work[label], errors="coerce")
    day_ids, _ = pd.factorize(work["trade_date"], sort=True)
    order = np.argsort(day_ids, kind="mergesort")
    sorted_ids = day_ids[order]
    bounds = np.append(np.flatnonzero(np.diff(sorted_ids, prepend=-1)), len(order))
    groups = {int(sorted_ids[a]): order[a:b] for a, b in zip(bounds[:-1], bounds[1:])}
    return _shape_one(groups, work[fid].to_numpy(dtype=float), work[label].to_numpy(dtype=float))


def _write_factory_sidecars(panel: pd.DataFrame, fids: list[str]) -> None:
    """Rolling / TopTail 从已算 IC/shape 落地，供 Factory 强制步读取。不改公式。"""
    from .contract import ROLLING_WINDOWS

    for fid in fids:
        if fid not in panel.columns:
            continue
        dest = ROUND22_2_ROOT / "factors" / fid
        dest.mkdir(parents=True, exist_ok=True)
        rolling: dict[str, Any] = {"status": "OK", "windows": {}, "recomputed_suite": False}
        for label in LABELS:
            if label not in panel.columns:
                continue
            daily = []
            for day, block in panel.groupby("trade_date", sort=True):
                x = pd.to_numeric(block[fid], errors="coerce")
                y = pd.to_numeric(block[label], errors="coerce")
                m = x.notna() & y.notna()
                if int(m.sum()) < 5:
                    continue
                r = x.loc[m].corr(y.loc[m], method="spearman")
                if r is not None and np.isfinite(r):
                    daily.append((pd.Timestamp(day), float(r)))
            if not daily:
                continue
            series = pd.Series({d: v for d, v in daily}).sort_index()
            block = {}
            for w in ROLLING_WINDOWS:
                path = series.rolling(int(w), min_periods=max(5, int(w) // 5)).mean().dropna()
                arr = path.to_numpy(dtype=float)
                block[str(w)] = {
                    "window": int(w),
                    "n_windows": int(len(arr)),
                    "terminal_mean": float(arr[-1]) if len(arr) else None,
                    "path_mean": float(arr.mean()) if len(arr) else None,
                    "do_not_pick_window_after_seeing_nav": True,
                }
            rolling[label] = block
        dump_json(dest / "rolling.json", rolling)
        shape = load_json(dest / "shape.json")
        tail = {}
        for label in LABELS:
            rec = shape.get(label) or {}
            tail[label] = {
                "top_tail_mean_spread": rec.get("top_tail_mean_spread"),
                "top2_minus_rest": rec.get("top2_minus_rest"),
                "q5_minus_q1": rec.get("q5_minus_q1"),
                "status": "OK" if rec else "MISSING",
            }
        dump_json(dest / "top_tail.json", tail)
        cond_src = ROUND22_2_ROOT / "evidence" / "c14_conditional.json"
        if cond_src.is_file() and not (dest / "conditional.json").is_file():
            dump_json(dest / "conditional.json", {"status": "OK", "diagnostic_only": True, "source": str(cond_src)})


def _write_hn2_sidecars(panel: pd.DataFrame) -> None:
    hn_path = ROUND22_2_ROOT / "data_recovery" / "hn2_panel.parquet"
    fid = "r22_2_hn2_avg_holding_chg"
    if not hn_path.is_file():
        return
    hn = pd.read_parquet(hn_path)
    if fid not in hn.columns:
        return
    hn["trade_date"] = pd.to_datetime(hn["trade_date"])
    hn["instrument_id"] = hn["instrument_id"].astype(str)
    work = panel[KEYS + [c for c in LABELS if c in panel.columns]].drop_duplicates(KEYS)
    work = work.merge(hn[KEYS + [fid]].drop_duplicates(KEYS), on=KEYS, how="left")
    dest = ROUND22_2_ROOT / "factors" / fid
    rec = {"factor_id": fid}
    for label in LABELS:
        if label not in work.columns:
            continue
        ic = _daily_ic(work, fid, label)
        rec[label] = ic
        dump_json(dest / f"ic_{label}.json", ic)
        sh = _shape(work, fid, label)
        dump_json(dest / f"shape_{label}.json", sh)
    dump_json(dest / "ic.json", rec)
    dump_json(dest / "shape.json", {lb: load_json(dest / f"shape_{lb}.json") for lb in LABELS})
    _write_factory_sidecars(work, [fid])


def run_evidence_c(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and EV_JSON.is_file() and MD["04"].is_file():
        existing = load_json(EV_JSON)
        if existing.get("status") == "DONE":
            try:
                panel = pd.read_parquet(ROUND22_2_ROOT / "data" / "c_factor_panel.parquet")
                fids = [existing.get("c1_model_id") or C1_ID, C4_ID]
                _write_factory_sidecars(panel, fids)
                _write_hn2_sidecars(panel)
            except Exception:
                pass
            return existing
    from .hypotheses import write_hypotheses

    write_hypotheses()
    pit = load_json(ROUND22_2_ROOT / "pit" / "c_pit.json")
    if not pit:
        raise RuntimeError("pit_c must run before evidence_c")
    panel = _build_panel(pit)
    # 尝试并上 14/13
    from ..round7.contract import FIVE_HEAD
    import pyarrow.parquet as pq

    cand = FIVE_HEAD.panel_dir / "candidate_panel.parquet"
    if cand.is_file():
        names = set(pq.read_schema(cand).names)
        extra = list(dict.fromkeys(c for c in (*FIVE_FEATURES, *TEN_FEATURES) if c in names and c not in panel.columns))
        if extra:
            feats = pd.read_parquet(cand, columns=[*KEYS, *extra])
            feats["trade_date"] = pd.to_datetime(feats["trade_date"])
            feats["instrument_id"] = feats["instrument_id"].astype(str)
            panel = panel.merge(feats.drop_duplicates(KEYS), on=KEYS, how="left")
    if panel.columns.duplicated().any():
        panel = panel.loc[:, ~panel.columns.duplicated()].copy()
    model_id = (pit.get("c1") or {}).get("model_id") or C1_ID
    fids = [model_id, C4_ID]
    ic_rows = []
    for fid in fids:
        rec = {"factor_id": fid}
        dest = ROUND22_2_ROOT / "factors" / fid
        for label in LABELS:
            ic = _daily_ic(panel, fid, label) if fid in panel.columns and label in panel.columns else {"status": "MISSING"}
            rec[label] = ic
            dump_json(dest / f"ic_{label}.json", ic)
        dump_json(dest / "ic.json", rec)
        ic_rows.append(rec)
        if fid in panel.columns:
            for label in LABELS:
                if label not in panel.columns:
                    continue
                sh = _shape(panel, fid, label)
                dump_json(dest / f"shape_{label}.json", sh)
            dump_json(dest / "shape.json", {lb: load_json(dest / f"shape_{lb}.json") for lb in LABELS})
    corr_c1_c4 = _cs_corr(panel, model_id, C4_ID)
    five_c1 = _fold_rho(panel, model_id, FIVE) if model_id in panel.columns else {}
    ten_c1 = _fold_rho(panel, model_id, TEN) if model_id in panel.columns else {}
    five_c4 = _fold_rho(panel, C4_ID, FIVE) if C4_ID in panel.columns else {}
    ten_c4 = _fold_rho(panel, C4_ID, TEN) if C4_ID in panel.columns else {}
    cond_rows = []
    if model_id in panel.columns and C4_ID in panel.columns:
        for label, locked in ((LABELS[0], list(FIVE)), (LABELS[1], list(TEN))):
            cond_rows.append({"factor": C4_ID, "label": label, "kind": "legacy_plus_c1", **_conditional(panel, C4_ID, locked + [model_id], label)})
            cond_rows.append({"factor": model_id, "label": label, "kind": "legacy", **_conditional(panel, model_id, locked, label)})
            cond_rows.append({"factor": C4_ID, "label": label, "kind": "legacy", **_conditional(panel, C4_ID, locked, label)})
    dump_json(ROUND22_2_ROOT / "correlation" / "c14_corr.json", {"corr_c1_c4": corr_c1_c4, "c1_five": five_c1, "c1_ten": ten_c1, "c4_five": five_c4, "c4_ten": ten_c4})
    dump_json(ROUND22_2_ROOT / "evidence" / "c14_conditional.json", {"rows": cond_rows, "full_history_fit": False, "diagnostic_only": True})
    write_csv(ROUND22_2_ROOT / "ic" / "c14_ic.csv", ic_rows)
    def _pack_ic(rec: dict[str, Any], label: str) -> dict[str, Any]:
        block = rec.get(label) or {}
        by = block.get("by_year") or {}
        return {
            "mean_all": block.get("mean"),
            "icir_all": block.get("icir"),
            "n_days": block.get("n_days"),
            "mean_2018_2026": _year_mean(by, FORMAL_IC_YEARS),
            "mean_2013_2017": _year_mean(by, DIAG_IC_YEARS),
            "status": block.get("status"),
        }

    payload = {
        "status": "DONE",
        "panel": str(ROUND22_2_ROOT / "data" / "c_factor_panel.parquet"),
        "n_rows": int(len(panel)),
        "c1_model_id": model_id,
        "c1_lag_days": int((pit.get("c1") or {}).get("lag_days") or 0),
        "formula_changed": False,
        "corr_c1_c4": corr_c1_c4,
        "c1_five": five_c1,
        "c1_ten": ten_c1,
        "c4_five": five_c4,
        "c4_ten": ten_c4,
        "ic": {r["factor_id"]: {lb: _pack_ic(r, lb) for lb in LABELS} for r in ic_rows},
        "conditional_c4_given_legacy_c1": [r for r in cond_rows if r.get("kind") == "legacy_plus_c1"],
        "conditional_is_diagnostic_only": True,
    }
    dump_json(EV_JSON, payload)
    _write_factory_sidecars(panel, fids)
    _write_hn2_sidecars(panel)
    write_md(
        MD["04"],
        "\n".join(
            [
                "# Round22.2 · 04 C1/C4 单因子证据",
                "",
                f"C1 模型 id `{model_id}` lag=`{payload['c1_lag_days']}`。公式未改。",
                f"Corr(C1,C4)=`{corr_c1_c4}`。",
                f"C1 vs Five 近邻 `{five_c1}`。C4 vs Five `{five_c4}`。",
                f"ConditionalIC(C4 | Legacy,C1) 仅诊断。full_history_fit=False。",
                f"C1 e2 2018–2026 RankIC=`{(payload['ic'].get(model_id) or {}).get('label_e2_net', {}).get('mean_2018_2026')}`。",
                f"C4 e2 2018–2026 RankIC=`{(payload['ic'].get(C4_ID) or {}).get('label_e2_net', {}).get('mean_2018_2026')}`。",
                "",
            ]
        ),
    )
    return payload
