"""身份锁 + 零 Top2 不拉 Top3。失败则取消资格 / STOP。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND14_ROOT
from ..round8.overlays import MIN_KEEP
from ..round13.weights import inject_weights
from .common import dump_json
from .contract import FAIL_ALPHA_MUTATION, RANK_NAV, TOP_K, WANT_TOP2_EPS

TEST_JSON = ROUND14_ROOT / "tests" / "unit_tests.json"


def _top_ids(overlayed: pd.DataFrame) -> pd.DataFrame:
    work = overlayed.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    recs = []
    for day, g in work.loc[work["_rank"].le(TOP_K)].groupby("signal_date", sort=False):
        r1 = g.loc[g["_rank"].eq(1), "instrument_id"]
        r2 = g.loc[g["_rank"].eq(2), "instrument_id"]
        recs.append({
            "signal_date": day,
            "id1": str(r1.iloc[0]) if len(r1) else "",
            "id2": str(r2.iloc[0]) if len(r2) else "",
        })
    return pd.DataFrame(recs).set_index("signal_date")


def identity_lock(v4: pd.DataFrame, cand: pd.DataFrame, code: str) -> dict[str, Any]:
    a = _top_ids(v4)
    b = _top_ids(cand)
    joined = a.join(b, lsuffix="_v4", rsuffix="_cand", how="outer")
    c1_v4 = joined["id1_v4"].astype(str)
    c1_c = joined["id1_cand"].astype(str)
    c2_v4 = joined["id2_v4"].astype(str)
    c2_c = joined["id2_cand"].astype(str)
    same1 = c1_v4 == c1_c
    same2 = c2_v4 == c2_c
    n_bad = int((~same1 | ~same2).sum())
    passed = n_bad == 0
    return {
        "code": code,
        "passed": passed,
        "status": "PASS" if passed else FAIL_ALPHA_MUTATION,
        "n_days": int(len(joined)),
        "n_mismatch": n_bad,
        "top1_mismatch_n": int((~same1).sum()),
        "top2_mismatch_n": int((~same2).sum()),
    }


def test_zero_top2_does_not_pull_top3(base: pd.DataFrame) -> dict[str, Any]:
    work = base.copy().reset_index(drop=True)
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    counts = work.groupby("signal_date")["_rank"].max()
    eligible = counts[counts >= 3]
    if eligible.empty:
        return {"passed": False, "status": "NO_DAY_WITH_TOP3", "note": "无法检验"}
    day = eligible.index[0]
    g = work.loc[work["signal_date"].eq(day)].sort_values("_rank")
    top1 = g.loc[g["_rank"].eq(1)].iloc[0]
    top2 = g.loc[g["_rank"].eq(2)].iloc[0]
    top3 = g.loc[g["_rank"].eq(3)].iloc[0]
    weights = pd.DataFrame([
        {
            "signal_date": day,
            "instrument_id": str(top1["instrument_id"]),
            "engine_rank": 1,
            "allocation_multiplier": float(top1["allocation_multiplier"]),
        },
        {
            "signal_date": day,
            "instrument_id": str(top2["instrument_id"]),
            "engine_rank": 2,
            "allocation_multiplier": MIN_KEEP,
        },
    ])
    inj = inject_weights(work.drop(columns=["_rank"], errors="ignore"), weights)
    inj["_rank"] = inj.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    day_inj = inj.loc[inj["signal_date"].eq(day)].sort_values("_rank")
    ids = day_inj.loc[day_inj["_rank"].le(TOP_K), "instrument_id"].astype(str).tolist()
    m2 = float(day_inj.loc[day_inj["_rank"].eq(2), "allocation_multiplier"].iloc[0])
    third_in_top = str(top3["instrument_id"]) in ids
    n_rows_before = int((work["signal_date"].eq(day)).sum())
    n_rows_after = int((inj["signal_date"].eq(day)).sum())
    passed = (
        (not third_in_top)
        and str(top1["instrument_id"]) in ids
        and str(top2["instrument_id"]) in ids
        and m2 <= MIN_KEEP * 10
        and n_rows_after == n_rows_before
        and abs(m2 - MIN_KEEP) <= 1e-18 or m2 == MIN_KEEP or m2 <= 1e-10
    )
    # tighten: multiplier is MIN_KEEP and Top3 not in top2
    passed = (not third_in_top) and (m2 <= 1e-10) and n_rows_after == n_rows_before
    return {
        "name": "test_zero_top2_does_not_pull_top3",
        "passed": bool(passed),
        "status": "PASS" if passed else "FAIL",
        "day": str(pd.Timestamp(day).date()),
        "top1": str(top1["instrument_id"]),
        "top2": str(top2["instrument_id"]),
        "top3": str(top3["instrument_id"]),
        "top2_multiplier": m2,
        "min_keep": MIN_KEEP,
        "third_pulled": third_in_top,
        "n_rows_before": n_rows_before,
        "n_rows_after": n_rows_after,
        "rank_nav": list(RANK_NAV),
        "want_top2_eps": WANT_TOP2_EPS,
    }


def run_unit_tests(v4: pd.DataFrame, overlayed_map: dict[str, pd.DataFrame]) -> dict[str, Any]:
    zero = test_zero_top2_does_not_pull_top3(v4)
    locks = {code: identity_lock(v4, ov, code) for code, ov in overlayed_map.items()}
    payload = {
        "status": "DONE",
        "zero_top2": zero,
        "identity": locks,
        "all_identity_pass": all(v.get("passed") for v in locks.values()),
        "zero_top2_pass": bool(zero.get("passed")),
    }
    dump_json(TEST_JSON, payload)
    if not zero.get("passed"):
        raise RuntimeError("ZERO_TOP2_PULLS_TOP3 STOP")
    return payload


__all__ = ["identity_lock", "run_unit_tests", "test_zero_top2_does_not_pull_top3"]
