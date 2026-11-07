"""14/13 + R17/R20 KEEP + ZyQuant 350 Novelty 库。只做合法键对齐。缺失不填 0。350 不进模型。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND20_ROOT, ROUND21_ROOT
from ..round7.contract import FIVE_FEATURES, TEN_FEATURES
from ..round18.contract import F2_NAME
from ..round20.panel import PANEL_PATH as R20_DIAG
from .common import dump_json, load_json
from .construct import PANEL_PATH as R21_PANEL
from .contract import CLUSTER_SPEARMAN_ABS, ZYQUANT_350_ROOT
from .panel import KEYS, ic_window, load_candidate_keys

KEEP_CSV = ROUND20_ROOT / "redundancy" / "keep_drop.csv"
LIB_JSON = ROUND21_ROOT / "data" / "library.json"
RESEARCH_PANEL = ROUND21_ROOT / "data" / "research_panel.parquet"
ZY_ALIGN = ROUND21_ROOT / "data" / "zyquant350_aligned.parquet"
ZY_JSON = ROUND21_ROOT / "data" / "zyquant350_align.json"


def load_r20_keep_names() -> dict[str, Any]:
    if not KEEP_CSV.is_file():
        raise RuntimeError(f"MISSING_ARTIFACT R20 keep_drop: {KEEP_CSV}")
    frame = pd.read_csv(KEEP_CSV)
    frame["name"] = frame["name"].astype(str)
    frame["action"] = frame["action"].astype(str)
    keep = frame.loc[frame["action"].str.startswith("KEEP")].copy()
    locked = set(FIVE_FEATURES) | set(TEN_FEATURES)
    names = [str(x) for x in keep["name"].tolist()]
    return {
        "all_keep": names,
        "locked_14_13": [n for n in names if n in locked],
        "keep_unlocked": [n for n in names if n not in locked],
        "f2_excluded": F2_NAME,
        "frame": keep,
    }


def _read_keys(path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    cols = [c for c in columns if c in names]
    frame = pd.read_parquet(path, columns=cols)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    for col in frame.columns:
        if col in KEYS:
            continue
        if col in {"next2_date", "actual_exit_date"}:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
            continue
        if col in {"is_first_board_hard"}:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def build_research_panel(*, resume: bool = True) -> pd.DataFrame:
    """R21 新因子 + R20 KEEP/14/13。只 left-join 候选键。不填 0。"""
    if resume and RESEARCH_PANEL.is_file() and LIB_JSON.is_file():
        frame = pd.read_parquet(RESEARCH_PANEL)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        for col in ("next2_date", "actual_exit_date"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col], errors="coerce")
        return frame
    if not R21_PANEL.is_file():
        raise RuntimeError(f"MISSING_ARTIFACT R21 factor_panel: {R21_PANEL}")
    if not R20_DIAG.is_file():
        raise RuntimeError(f"MISSING_ARTIFACT R20 diag_panel: {R20_DIAG}")
    keep = load_r20_keep_names()
    old_names = [n for n in keep["all_keep"] if n != F2_NAME]
    r21 = pd.read_parquet(R21_PANEL)
    r21["trade_date"] = pd.to_datetime(r21["trade_date"])
    r21["instrument_id"] = r21["instrument_id"].astype(str)
    for col in ("next2_date", "actual_exit_date"):
        if col in r21.columns:
            r21[col] = pd.to_datetime(r21[col], errors="coerce")
    old = _read_keys(R20_DIAG, [*KEYS, *old_names])
    extra = [c for c in old.columns if c not in r21.columns]
    if extra:
        r21 = r21.merge(old[KEYS + extra], on=KEYS, how="left")
    r21 = r21.replace([np.inf, -np.inf], np.nan)
    if r21.duplicated(KEYS).any():
        raise RuntimeError("research panel duplicate keys")
    RESEARCH_PANEL.parent.mkdir(parents=True, exist_ok=True)
    r21.to_parquet(RESEARCH_PANEL, index=False)
    payload = {
        "status": "DONE",
        "n_rows": int(len(r21)),
        "n_dates": int(r21["trade_date"].nunique()),
        "r20_keep_n": len(keep["all_keep"]),
        "r20_keep_unlocked_n": len(keep["keep_unlocked"]),
        "locked_14_13": keep["locked_14_13"],
        "old_merged": extra,
        "filled_with_zero": False,
        "f2_in_panel": F2_NAME in r21.columns,
        "path": str(RESEARCH_PANEL),
    }
    dump_json(LIB_JSON, payload)
    return r21


def _zyquant_hash_dir():
    if not ZYQUANT_350_ROOT.is_dir():
        return None
    hashes = [p for p in ZYQUANT_350_ROOT.iterdir() if p.is_dir()]
    return hashes[0] if hashes else None


def list_zyquant_350() -> list[dict[str, Any]]:
    root = _zyquant_hash_dir()
    if root is None:
        return []
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        pq = next(d.glob("*.parquet"), None)
        rows.append({"name": d.name, "path": str(pq) if pq else None, "has_parquet": pq is not None})
    return rows


def _key_tuples(keys: pd.DataFrame) -> list[tuple[str, str]]:
    d = pd.to_datetime(keys["trade_date"]).dt.strftime("%Y-%m-%d")
    i = keys["instrument_id"].astype(str)
    return list(zip(d.tolist(), i.tolist()))


def _zy_values_on_keys(path: str, keys: pd.DataFrame) -> np.ndarray:
    """DuckDB 等值键连接。禁止 asof。返回与 keys 行序对齐的值。"""
    import duckdb

    work = keys.loc[:, KEYS].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rid"] = np.arange(len(work), dtype=np.int64)
    con = duckdb.connect()
    con.register("keys", work)
    path_sql = str(path).replace("'", "''")
    out = con.execute(
        f"""
        SELECT k._rid, z.value
        FROM keys k
        LEFT JOIN read_parquet('{path_sql}') z
          ON k.instrument_id = CAST(z.instrument_id AS VARCHAR)
         AND CAST(k.trade_date AS DATE) = CAST(z.trade_date AS DATE)
        ORDER BY k._rid
        """
    ).df()
    return pd.to_numeric(out["value"], errors="coerce").to_numpy(dtype=float)


def _spearman_1d(a: np.ndarray, b: np.ndarray) -> tuple[float | None, int]:
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 80:
        return None, n
    x = pd.Series(a[mask]).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(b[mask]).rank(method="average").to_numpy(dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if den < 1e-18:
        return None, n
    rho = float(np.dot(x, y) / den)
    return (rho if np.isfinite(rho) else None), n


def align_zyquant_350(*, resume: bool = True) -> pd.DataFrame | None:
    """兼容旧接口：Novelty 不落 350 列宽表，只保证 JSON 元数据。"""
    if resume and ZY_JSON.is_file():
        meta = load_json(ZY_JSON)
        if meta.get("status") == "DONE":
            return None
    return None


def zyquant_novelty(panel: pd.DataFrame, new_names: list[str], *, resume: bool = True) -> list[dict[str, Any]]:
    """每个新因子对 350 库的最大 |Spearman|。合法键对齐。库因子不进模型。"""
    summary_path = ROUND21_ROOT / "data" / "zyquant350_novelty.json"
    if resume and summary_path.is_file() and ZY_JSON.is_file():
        prev = load_json(summary_path)
        if prev.get("status") == "DONE" and prev.get("rows"):
            return list(prev["rows"])
    work = ic_window(panel)[KEYS + [c for c in new_names if c in panel.columns]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    work = work.drop_duplicates(KEYS).sort_values(KEYS).reset_index(drop=True)
    new_mat = {n: pd.to_numeric(work[n], errors="coerce").to_numpy(dtype=float) for n in new_names if n in work.columns}
    catalog = list_zyquant_350()
    best: dict[str, dict[str, Any]] = {
        n: {"max_abs_rho": None, "nearest": None, "nearest_rho": None, "n_overlap": 0} for n in new_names
    }
    missing, skipped, ok = [], [], []
    jobs = []
    for rec in catalog:
        if rec.get("path"):
            jobs.append(rec)
        else:
            missing.append(rec["name"])
    print(f"[round21] zyquant350 novelty n={len(jobs)} legal key join; no model ingest", flush=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(rec: dict[str, Any]) -> tuple[str, np.ndarray | None, str | None]:
        try:
            arr = _zy_values_on_keys(rec["path"], work[KEYS])
        except Exception as exc:
            return rec["name"], None, str(exc)
        if int(np.isfinite(arr).sum()) == 0:
            return rec["name"], None, "no_overlap_after_legal_key_join"
        return rec["name"], arr, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_one, rec) for rec in jobs]
        done = 0
        for fut in as_completed(futs):
            done += 1
            name, arr, err = fut.result()
            if done % 25 == 0 or done == 1:
                print(f"[round21] zyquant350 {done}/{len(jobs)} {name}", flush=True)
            if err or arr is None:
                skipped.append({"name": name, "reason": err or "align_fail"})
                continue
            ok.append(name)
            for feat, vec in new_mat.items():
                rho, n = _spearman_1d(vec, arr)
                if rho is None:
                    continue
                cur = best[feat]
                if cur["max_abs_rho"] is None or abs(rho) > float(cur["max_abs_rho"]):
                    cur["max_abs_rho"] = abs(rho)
                    cur["nearest_rho"] = float(rho)
                    cur["nearest"] = name
                    cur["n_overlap"] = n
    out = []
    for feat in new_names:
        rec = best.get(feat) or {}
        out.append(
            {
                "factor": feat,
                "status": "DONE" if rec.get("nearest") else ("MISSING_NEW" if feat not in new_mat else "NO_OVERLAP"),
                "max_abs_rho": rec.get("max_abs_rho"),
                "nearest": rec.get("nearest"),
                "nearest_rho": rec.get("nearest_rho"),
                "n_overlap": rec.get("n_overlap") or 0,
                "cluster_ge_085": bool(rec.get("max_abs_rho") is not None and rec["max_abs_rho"] >= CLUSTER_SPEARMAN_ABS),
                "enters_model": False,
            }
        )
    dump_json(
        ZY_JSON,
        {
            "status": "DONE",
            "role": "Novelty 对照库，不进 PIT_ELIGIBLE_MODEL",
            "legal_align": "key join on (trade_date, instrument_id); no asof; no fillna 0; no wide panel",
            "n_catalog": len(catalog),
            "n_aligned": len(ok),
            "n_missing_parquet": len(missing),
            "n_skipped": len(skipped),
            "aligned_names": ok[:20],
            "missing": missing,
            "skipped": skipped[:30],
            "threshold": CLUSTER_SPEARMAN_ABS,
        },
    )
    dump_json(summary_path, {"status": "DONE", "rows": out, "n_library_used": len(ok)})
    return out


__all__ = [
    "KEEP_CSV",
    "LIB_JSON",
    "RESEARCH_PANEL",
    "ZY_ALIGN",
    "align_zyquant_350",
    "build_research_panel",
    "load_r20_keep_names",
    "zyquant_novelty",
]
