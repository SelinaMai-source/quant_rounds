"""从 locked alpha + exact overlay 提取日度目标权重，并构造 A01–A05。禁止双 NAV 相加。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND13_ROOT, assert_project_result_path, ensure_round13_tree
from ..round8.overlays import MIN_KEEP
from .common import dump_json, write_md
from .contract import (
    A01,
    A02,
    A03,
    A04,
    A05,
    CANDIDATE_IDS,
    RANK_NAV,
    TOP_K,
    WANT_TOP2_EPS,
    overlayed_path,
)
from .freeze import require_frozen
from .runner import build_version_overlayed

WEIGHTS_PARQUET = ROUND13_ROOT / "daily_target_weights.parquet"
WEIGHTS_JSON = ROUND13_ROOT / "weights" / "weights_audit.json"
WEIGHTS_MD = ROUND13_ROOT / "weights_audit.md"
FRAC = {1: float(RANK_NAV[0]), 2: float(RANK_NAV[1])}


def _prep(overlayed: pd.DataFrame, source: str) -> pd.DataFrame:
    work = overlayed.copy().reset_index(drop=True)
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["engine_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    top = work.loc[work["engine_rank"].le(TOP_K)].copy()
    top["engine_rank"] = top["engine_rank"].astype(int)
    top["allocation_multiplier"] = pd.to_numeric(top["allocation_multiplier"], errors="coerce").fillna(MIN_KEEP).clip(MIN_KEEP, 1.0)
    top["target_nav_fraction"] = top["engine_rank"].map(FRAC) * top["allocation_multiplier"]
    top["source"] = source
    return top[["signal_date", "instrument_id", "engine_rank", "target_nav_fraction", "allocation_multiplier", "source", "score"]]


def extract_top_weights(overlayed: pd.DataFrame, source: str) -> pd.DataFrame:
    return _prep(overlayed, source)


def _wide(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for src, df in parts.items():
        sub = df.copy()
        sub = sub.rename(columns={
            "instrument_id": f"id_{src}",
            "target_nav_fraction": f"w_{src}",
            "allocation_multiplier": f"m_{src}",
            "score": f"score_{src}",
        })
        frames.append(sub[["signal_date", "engine_rank", f"id_{src}", f"w_{src}", f"m_{src}"]])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["signal_date", "engine_rank"], how="outer")
    return out.sort_values(["signal_date", "engine_rank"]).reset_index(drop=True)


def _audit_union(wide: pd.DataFrame) -> dict[str, Any]:
    n_third = 0
    third_days: list[str] = []
    for day, g in wide.groupby("signal_date", sort=False):
        names = set()
        for src in ("v2", "v3", "v4"):
            col = f"id_{src}"
            names.update(str(x) for x in g[col].dropna().astype(str) if str(x) not in {"nan", "None"})
        if len(names) > TOP_K:
            n_third += 1
            third_days.append(str(pd.Timestamp(day).date()))
    if n_third:
        raise RuntimeError(
            f"UNION_THIRD_NAME STOP: {n_third} days have >{TOP_K} unique names; "
            f"examples={third_days[:5]}. 不得改 top_k。"
        )
    return {"n_days_with_third_name": 0, "stopped": False}


def _want(w: float) -> bool:
    return bool(np.isfinite(w) and float(w) > WANT_TOP2_EPS)


def _row_from(date, inst, rank, w, source: str) -> dict[str, Any]:
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
        "target_nav_fraction": float(w) if w > 0 else float(frac * MIN_KEEP),
        "allocation_multiplier": float(mult),
        "source": source,
        "clipped_to_unit": bool(frac > 0 and w / frac > 1.0 + 1e-12),
    }


def build_candidates(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_clip = {A03: 0}
    n_a04_drop = n_a04_min = n_a05_mean = n_a05_half = n_a05_drop = 0
    for rec in wide.itertuples(index=False):
        date = rec.signal_date
        rank = int(rec.engine_rank)
        ids = {s: getattr(rec, f"id_{s}") for s in ("v2", "v3", "v4")}
        ws = {s: float(getattr(rec, f"w_{s}")) if pd.notna(getattr(rec, f"w_{s}")) else 0.0 for s in ("v2", "v3", "v4")}
        id_v4 = ids["v4"] if pd.notna(ids["v4"]) else (ids["v3"] if pd.notna(ids["v3"]) else ids["v2"])
        if pd.isna(id_v4):
            continue
        w01 = 0.50 * ws["v3"] + 0.50 * ws["v4"]
        w02 = (ws["v2"] + ws["v3"] + ws["v4"]) / 3.0
        w03 = float(np.median([ws["v2"], ws["v3"], ws["v4"]]))
        rows.append(_row_from(date, id_v4 if pd.notna(ids["v3"]) else id_v4, rank, w01, A01))
        # A01 uses v3/v4 names; they should match. Prefer v4 identity.
        rows[-1]["instrument_id"] = str(id_v4)
        rows.append(_row_from(date, id_v4, rank, w02, A02))
        r03 = _row_from(date, id_v4, rank, w03, A03)
        if r03["clipped_to_unit"]:
            n_clip[A03] += 1
        rows.append(r03)

        if rank == 1:
            rows.append(_row_from(date, id_v4, 1, ws["v4"], A04))
            rows.append(_row_from(date, id_v4, 1, ws["v4"], A05))
            continue

        id3 = ids["v3"] if pd.notna(ids["v3"]) else None
        id4 = ids["v4"] if pd.notna(ids["v4"]) else None
        want3 = _want(ws["v3"])
        want4 = _want(ws["v4"])
        same = (id3 is not None and id4 is not None and str(id3) == str(id4))
        if want3 and want4 and same:
            rows.append(_row_from(date, id4, 2, min(ws["v3"], ws["v4"]), A04))
            n_a04_min += 1
        else:
            rows.append(_row_from(date, id_v4, 2, 0.0, A04))
            n_a04_drop += 1
        if same and (want3 or want4):
            if want3 and want4:
                rows.append(_row_from(date, id4, 2, 0.5 * ws["v3"] + 0.5 * ws["v4"], A05))
                n_a05_mean += 1
            elif want3:
                rows.append(_row_from(date, str(id3), 2, 0.5 * ws["v3"], A05))
                n_a05_half += 1
            else:
                rows.append(_row_from(date, str(id4), 2, 0.5 * ws["v4"], A05))
                n_a05_half += 1
        else:
            rows.append(_row_from(date, id_v4, 2, 0.0, A05))
            n_a05_drop += 1
    cand = pd.DataFrame(rows)
    meta = {
        "a03_n_clip_to_unit": int(n_clip[A03]),
        "a03_renormalize": False,
        "a03_engine_clip_only": True,
        "a04_n_min": n_a04_min,
        "a04_n_drop_top2": n_a04_drop,
        "a05_n_mean": n_a05_mean,
        "a05_n_half": n_a05_half,
        "a05_n_drop_top2": n_a05_drop,
    }
    return cand, meta


def inject_weights(base: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    work = base.copy().reset_index(drop=True)
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    w = weights.copy()
    w["signal_date"] = pd.to_datetime(w["signal_date"]).dt.normalize()
    w["instrument_id"] = w["instrument_id"].astype(str)
    merged = work.merge(
        w[["signal_date", "instrument_id", "allocation_multiplier", "engine_rank"]].rename(
            columns={"allocation_multiplier": "mult_w", "engine_rank": "rank_w"}
        ),
        on=["signal_date", "instrument_id"],
        how="left",
    )
    new_mult = pd.to_numeric(merged["allocation_multiplier"], errors="coerce").fillna(MIN_KEEP)
    inj = pd.to_numeric(merged["mult_w"], errors="coerce")
    mask_top = merged["_rank"].le(TOP_K)
    new_mult.loc[mask_top] = inj.loc[mask_top].fillna(MIN_KEEP)
    work["allocation_multiplier"] = new_mult.clip(MIN_KEEP, 1.0).to_numpy()
    mismatch = merged.loc[mask_top & inj.notna() & merged["rank_w"].notna() & (merged["_rank"] != merged["rank_w"])]
    if len(mismatch):
        raise RuntimeError(f"RANK_MISMATCH STOP: {len(mismatch)} rows where injected name rank ≠ engine rank")
    return work.drop(columns=["_rank"], errors="ignore")


def run_weights(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    ensure_round13_tree()
    if resume and WEIGHTS_PARQUET.is_file() and WEIGHTS_JSON.is_file():
        from .common import load_json as _load

        payload = _load(WEIGHTS_JSON)
        if payload.get("status") == "DONE":
            return payload
    parts = {}
    overlay_meta = {}
    for ver in ("v2", "v3", "v4"):
        ov, meta, _ = build_version_overlayed(ver)
        parts[ver] = extract_top_weights(ov, ver)
        overlay_meta[ver] = {k: meta.get(k) for k in ("kind", "round9_kind", "allocator", "top1_lock") if k in meta}
        dest = overlayed_path(ver)
        assert_project_result_path(dest)
        ov.to_parquet(dest, index=False)
    wide = _wide(parts)
    union = _audit_union(wide)
    v3t2 = wide.loc[wide["engine_rank"].eq(2)].copy()
    same_id = v3t2["id_v3"].astype(str) == v3t2["id_v4"].astype(str)
    w3 = pd.to_numeric(v3t2["w_v3"], errors="coerce").fillna(0.0)
    w4 = pd.to_numeric(v3t2["w_v4"], errors="coerce").fillna(0.0)
    pos_diff = same_id & ((w3 - w4).abs() > WANT_TOP2_EPS)
    id_diff = ~same_id
    want_both = same_id & (w3 > WANT_TOP2_EPS) & (w4 > WANT_TOP2_EPS)
    want_v3_only = same_id & (w3 > WANT_TOP2_EPS) & (w4 <= WANT_TOP2_EPS)
    want_v4_only = same_id & (w4 > WANT_TOP2_EPS) & (w3 <= WANT_TOP2_EPS)
    audit = {
        "n_days": int(wide["signal_date"].nunique()),
        "n_top2_days": int(len(v3t2)),
        "top2_identity_agree_share": float(same_id.mean()) if len(v3t2) else None,
        "top2_identity_disagree_n": int(id_diff.sum()),
        "top2_position_disagree_n": int(pos_diff.sum()),
        "top2_position_disagree_share": float(pos_diff.mean()) if len(v3t2) else None,
        "top2_both_want_n": int(want_both.sum()),
        "top2_v3_only_n": int(want_v3_only.sum()),
        "top2_v4_only_n": int(want_v4_only.sum()),
        "mean_abs_w_gap_same_id": float((w3 - w4).abs()[same_id].mean()) if same_id.any() else None,
        "union": union,
        "note": "分歧主要应是 Top2 乘数，不是标的身份。",
    }
    cand, cand_meta = build_candidates(wide)
    all_w = pd.concat([*parts.values(), cand], ignore_index=True)
    assert_project_result_path(WEIGHTS_PARQUET)
    all_w.to_parquet(WEIGHTS_PARQUET, index=False)
    base_v4, _, _ = build_version_overlayed("v4")
    for cid in CANDIDATE_IDS:
        inj = inject_weights(base_v4, cand.loc[cand["source"].eq(cid)])
        dest = overlayed_path(cid)
        assert_project_result_path(dest)
        inj.to_parquet(dest, index=False)
    payload = {
        "status": "DONE",
        "path": str(WEIGHTS_PARQUET),
        "overlay_meta": overlay_meta,
        "audit": audit,
        "candidates": cand_meta,
        "n_rows": int(len(all_w)),
        "sources": sorted(all_w["source"].astype(str).unique().tolist()),
        "never_average_two_navs": True,
        "engine_clip_documented": (
            "A03 不 renormalize。仅当 w/RANK_NAV[rank] > 1 时 clip 到 1.0（Stock 要求 multiplier∈(0,1]）。"
            "w=0 时写入 MIN_KEEP 以免删行导致 Top3 补位。"
        ),
    }
    dump_json(WEIGHTS_JSON, payload)
    write_md(
        WEIGHTS_MD,
        "\n".join([
            "# Round 13 · 日度目标权重",
            "",
            "来源 = locked alpha + exact overlay。**禁止**两条 NAV 相加。",
            f"落盘 `{WEIGHTS_PARQUET}`。",
            "",
            f"- 交易日数 = **{audit['n_days']}**",
            f"- Top2 身份一致比例 = **{audit['top2_identity_agree_share']}**（分歧 n={audit['top2_identity_disagree_n']}）",
            f"- Top2 仓位分歧 n = **{audit['top2_position_disagree_n']}** share={audit['top2_position_disagree_share']}",
            f"- v3 与 v4 都要 Top2 = {audit['top2_both_want_n']}；仅 v3={audit['top2_v3_only_n']}；仅 v4={audit['top2_v4_only_n']}",
            f"- union 第 3 名天数 = **{union['n_days_with_third_name']}**（出现则 STOP）",
            f"- A03 clip-to-unit n = {cand_meta['a03_n_clip_to_unit']}（不 renormalize）",
            f"- A04 min / drop-Top2 = {cand_meta['a04_n_min']} / {cand_meta['a04_n_drop_top2']}",
            f"- A05 mean / half / drop = {cand_meta['a05_n_mean']} / {cand_meta['a05_n_half']} / {cand_meta['a05_n_drop_top2']}",
            "",
        ]),
    )
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
    from .common import load_json

    return load_json(WEIGHTS_JSON)


__all__ = [
    "build_candidates",
    "extract_top_weights",
    "inject_weights",
    "load_overlayed",
    "load_weights_audit",
    "load_weights_table",
    "run_weights",
]
