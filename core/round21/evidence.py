"""每因子独立 Evidence Package。双标签 IC；薄日跳过不记 IC=0；内部 rolling 60/120/252 maturity-aware；shape；top-tail。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_ROOT
from ..round20.metrics import hac_mean, icir_of, pearson_ic, rank_ic, same_sign_years, seed_for
from ..round20.shape import _monotonic
from .common import dump_json, finite_or_none, load_json, write_md
from .construct import PANEL_PATH, run_construct
from .contract import (
    BLOCK_DAYS,
    BOOT_N,
    COMPLETE_IC_YEARS,
    IC_MIN_VALID_NAMES_PER_DAY,
    INTERNAL_ROLLING_WINDOWS,
    INTERNAL_WINDOW_NAMES,
    LABELS,
    MATURITY_AWARE_LABEL,
    MATURITY_BY,
    ROUND21_INTERNAL,
)
from .freeze import require_frozen
from .github_exact import run_github_exact
from .hypotheses import HYPOTHESES
from .panel import KEYS, LABEL_END, ic_window, load_candidate_keys
from .pit import run_pit

EVIDENCE_JSON = ROUND21_ROOT / "evidence" / "evidence.json"
EVIDENCE_MD = ROUND21_ROOT / "04_evidence.md"
FACTORS_DIR = ROUND21_ROOT / "factors"


def _daily_ic_one(
    dates: pd.DatetimeIndex,
    groups: dict[int, np.ndarray],
    fx: np.ndarray,
    y: np.ndarray,
    ready: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    skip = {"thin": 0, "degenerate": 0, "recorded": 0, "ic_recorded_as_zero": 0}
    for di, idx in groups.items():
        xv = fx[idx]
        yv = y[idx]
        rv = ready[idx]
        mask = np.isfinite(xv) & np.isfinite(yv)
        n_valid = int(mask.sum())
        if n_valid < IC_MIN_VALID_NAMES_PER_DAY:
            skip["thin"] += 1
            continue
        x1, y1 = xv[mask], yv[mask]
        if int(np.unique(x1).size) < 2 or int(np.unique(y1).size) < 2:
            skip["degenerate"] += 1
            continue
        xs = pd.Series(x1)
        ys = pd.Series(y1)
        ic = pearson_ic(xs, ys)
        ric = rank_ic(xs, ys)
        if ic is None and ric is None:
            skip["degenerate"] += 1
            continue
        ready_at = pd.NaT
        r1 = rv[mask]
        r1 = r1[~pd.isna(r1)]
        if len(r1):
            ready_at = pd.Timestamp(np.max(r1))
        rows.append(
            {
                "trade_date": dates[di],
                "n_valid": n_valid,
                "ic": ic,
                "rankic": ric,
                "ready_at": ready_at,
            }
        )
        skip["recorded"] += 1
        if ic == 0.0:
            skip["ic_recorded_as_zero"] += 1
    daily = pd.DataFrame(rows)
    if daily.empty:
        daily = pd.DataFrame(columns=["trade_date", "n_valid", "ic", "rankic", "ready_at"])
    return daily, skip


def _summarize(daily: pd.DataFrame, metric: str) -> dict[str, Any]:
    if daily.empty or metric not in daily.columns:
        return {
            "n_days_used": 0,
            "mean": None,
            "std": None,
            "icir": None,
            "yearly": {str(y): None for y in COMPLETE_IC_YEARS},
            "same_sign_years": 0,
            "hac": {"status": "TOO_SHORT"},
        }
    series = pd.to_numeric(daily[metric], errors="coerce")
    used = series[np.isfinite(series.to_numpy(dtype=float))]
    years = pd.to_datetime(daily.loc[used.index, "trade_date"]).dt.year
    yearly: dict[str, float | None] = {}
    for y in COMPLETE_IC_YEARS:
        block = used.loc[years.eq(y)]
        yearly[str(int(y))] = float(block.mean()) if len(block) else None
    arr = used.to_numpy(dtype=float)
    return {
        "n_days_used": int(len(arr)),
        "n_days_listed": int(daily["trade_date"].nunique()) if len(daily) else 0,
        "mean": float(np.mean(arr)) if len(arr) else None,
        "std": float(np.std(arr, ddof=1)) if len(arr) >= 2 else None,
        "icir": icir_of(arr),
        "yearly": yearly,
        "same_sign_years": same_sign_years(yearly, COMPLETE_IC_YEARS),
        "hac": hac_mean(arr, lags=BLOCK_DAYS),
        "namespace": "RAW_IC_FULLSAMPLE",
    }


def _rolling(daily: pd.DataFrame, metric: str, calendar: pd.DatetimeIndex) -> dict[str, Any]:
    out: dict[str, Any] = {"namespace": ROUND21_INTERNAL, "windows": {}}
    if daily.empty or metric not in daily.columns:
        for name, w in zip(INTERNAL_WINDOW_NAMES, INTERNAL_ROLLING_WINDOWS):
            out["windows"][name] = {"window": w, "n": 0, "terminal_mean": None, "path_mean": None}
        return out
    work = daily.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["ready_at"] = pd.to_datetime(work["ready_at"], errors="coerce")
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    cal = pd.DatetimeIndex(calendar).sort_values()
    work = work.loc[work["trade_date"].isin(cal) & np.isfinite(work[metric].to_numpy(dtype=float))].copy()
    if work.empty:
        for name, w in zip(INTERNAL_WINDOW_NAMES, INTERNAL_ROLLING_WINDOWS):
            out["windows"][name] = {"window": int(w), "n_windows": 0, "terminal_mean": None, "path_mean": None}
        return out
    # 成熟日 = ready_at（缺省用 trade_date）；同日多条取均值后按日历对齐。
    ready = work["ready_at"].fillna(work["trade_date"])
    aligned = pd.DataFrame({"asof": pd.to_datetime(ready), "v": work[metric]}).dropna()
    aligned = aligned.groupby("asof", sort=True)["v"].mean()
    series = aligned.reindex(cal)
    for name, w in zip(INTERNAL_WINDOW_NAMES, INTERNAL_ROLLING_WINDOWS):
        roll = series.rolling(int(w), min_periods=max(5, int(w) // 5))
        path = roll.mean().dropna()
        arr = path.to_numpy(dtype=float)
        out["windows"][name] = {
            "window": int(w),
            "name": name,
            "n_windows": int(len(arr)),
            "terminal_mean": float(arr[-1]) if len(arr) else None,
            "path_mean": float(arr.mean()) if len(arr) else None,
            "path_std": float(arr.std(ddof=1)) if len(arr) >= 2 else None,
            "path_icir": icir_of(arr) if len(arr) else None,
            "share_positive": float(np.mean(arr > 0)) if len(arr) else None,
            "maturity_aware": True,
            "maturity_by": MATURITY_BY,
            "do_not_pick_window_after_seeing_nav": True,
        }
    return out


def _shape_one(groups: dict[int, np.ndarray], fx: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    q_means = {i: [] for i in range(5)}
    n_days_q5 = 0
    n_days_q10 = 0
    n_days_skip_thin = 0
    n_days_skip_q10 = 0
    top_spreads: list[float] = []
    top2_spreads: list[float] = []
    n_bins_seen: list[int] = []
    for idx in groups.values():
        xv = fx[idx]
        yv = y[idx]
        mask = np.isfinite(xv) & np.isfinite(yv)
        n_valid = int(mask.sum())
        if n_valid < IC_MIN_VALID_NAMES_PER_DAY:
            n_days_skip_thin += 1
            continue
        x1, y1 = xv[mask], yv[mask]
        if int(np.unique(x1).size) < 2:
            continue
        order_x = np.argsort(x1, kind="mergesort")
        ranks = np.empty(n_valid, dtype=float)
        ranks[order_x] = (np.arange(n_valid, dtype=float) + 1.0) / float(n_valid)
        q5 = np.minimum((ranks * 5.0).astype(int), 4)
        bins = int(np.unique(q5).size)
        n_bins_seen.append(bins)
        if bins >= 2:
            n_days_q5 += 1
            for b in range(5):
                sel = y1[q5 == b]
                if len(sel):
                    q_means[b].append(float(np.mean(sel)))
        if n_valid >= 10:
            n_days_q10 += 1
            top = y1[ranks >= 0.90]
            bot = y1[ranks <= 0.10]
            if len(top) and len(bot):
                top_spreads.append(float(np.mean(top) - np.mean(bot)))
        else:
            n_days_skip_q10 += 1
        top2 = y1[order_x[-2:]]
        rest = y1[order_x[:-2]] if n_valid > 2 else np.array([])
        if len(top2) and len(rest):
            top2_spreads.append(float(np.mean(top2) - np.mean(rest)))
    q_avg = [float(np.mean(q_means[i])) if q_means[i] else None for i in range(5)]
    q5q1 = None
    if q_avg[0] is not None and q_avg[4] is not None:
        q5q1 = q_avg[4] - q_avg[0]
    return {
        "q1": q_avg[0],
        "q2": q_avg[1],
        "q3": q_avg[2],
        "q4": q_avg[3],
        "q5": q_avg[4],
        "q5_minus_q1": q5q1,
        "monotonicity": _monotonic(q_avg),
        "n_days_q5": n_days_q5,
        "n_days_q10": n_days_q10,
        "n_days_skip_thin": n_days_skip_thin,
        "n_days_q10_not_forced": n_days_skip_q10,
        "top_tail_mean_spread": float(np.mean(top_spreads)) if top_spreads else None,
        "top2_minus_rest": float(np.mean(top2_spreads)) if top2_spreads else None,
        "median_bins_q5": float(np.median(n_bins_seen)) if n_bins_seen else None,
        "forced_q10": False,
    }


def _write_factor_md(fid: str, pkg: dict[str, Any]) -> None:
    dest = FACTORS_DIR / fid / "evidence.md"
    ic = pkg.get("ic") or {}
    lines = [
        f"# {fid} Evidence Package",
        "",
        f"hypothesis=`{pkg.get('hypothesis_path')}` pit=`{pkg.get('pit_status')}`",
        f"GITHUB_EXACT=`{pkg.get('github_exact_status')}` {ROUND21_INTERNAL}=`{list(INTERNAL_WINDOW_NAMES)}`",
        "",
        "## IC（薄日跳过，不把 IC 记 0）",
        "",
    ]
    for lab, rec in ic.items():
        ric = (rec.get("rankic") or {}).get("mean")
        raw = (rec.get("ic") or {}).get("mean")
        lines.append(f"- `{lab}` raw IC=`{raw}` RankIC=`{ric}` days=`{(rec.get('rankic') or {}).get('n_days_used')}`")
    lines += ["", "## Rolling / shape / top-tail 见同目录 json。禁止用全局结果冒充。", ""]
    write_md(dest, "\n".join(lines))


def run_evidence(*, resume: bool = True, jobs: int = 8, **kwargs: Any) -> dict[str, Any]:
    require_frozen()
    if resume and EVIDENCE_JSON.is_file():
        existing = load_json(EVIDENCE_JSON)
        if existing.get("status") == "DONE":
            return existing
    construct = run_construct(resume=True, jobs=jobs, **kwargs)
    pit = run_pit(resume=True, jobs=jobs, **kwargs)
    pit_rows = {r["factor_id"]: r for r in pit.get("rows") or []}
    if not PANEL_PATH.is_file():
        raise RuntimeError("construct panel missing")
    panel = ic_window(pd.read_parquet(PANEL_PATH))
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    labels = load_candidate_keys()
    extra = [c for c in [*LABELS, "next2_date", "actual_exit_date"] if c in labels.columns and c not in panel.columns]
    if extra:
        panel = panel.merge(labels[KEYS + extra], on=KEYS, how="left")
    day_ids, unique_dates = pd.factorize(panel["trade_date"], sort=True)
    unique_dates = pd.DatetimeIndex(unique_dates)
    order = np.argsort(day_ids, kind="mergesort")
    sorted_ids = day_ids[order]
    bounds = np.append(np.flatnonzero(np.diff(sorted_ids, prepend=-1)), len(order))
    groups = {int(sorted_ids[a]): order[a:b] for a, b in zip(bounds[:-1], bounds[1:])}
    label_arrs = {
        lab: pd.to_numeric(panel[lab], errors="coerce").to_numpy(dtype=float)
        for lab in LABELS
        if lab in panel.columns
    }
    ready_arrs = {}
    for lab in LABELS:
        col = LABEL_END.get(lab)
        if col and col in panel.columns:
            ready_arrs[lab] = pd.to_datetime(panel[col], errors="coerce").to_numpy()
        else:
            ready_arrs[lab] = np.empty(len(panel), dtype="datetime64[ns]")
            ready_arrs[lab][:] = np.datetime64("NaT")

    github = run_github_exact(panel, resume=resume)
    gh_by = github.get("by_factor") or {}

    packages: list[dict[str, Any]] = []
    best_raw: list[dict[str, Any]] = []
    best_roll: list[dict[str, Any]] = []
    best_tail: list[dict[str, Any]] = []

    for card in HYPOTHESES:
        fid = card["factor_id"]
        dest = FACTORS_DIR / fid
        dest.mkdir(parents=True, exist_ok=True)
        rec_pit = pit_rows.get(fid) or {}
        if fid not in panel.columns:
            pkg = {
                "factor_id": fid,
                "status": "SKIPPED",
                "reason": "factor column missing; no copied global package",
                "pit_status": rec_pit.get("pit_status"),
            }
            dump_json(dest / "package.json", pkg)
            packages.append(pkg)
            continue
        fx = pd.to_numeric(panel[fid], errors="coerce").to_numpy(dtype=float)
        ic_block: dict[str, Any] = {}
        roll_block: dict[str, Any] = {}
        shape_block: dict[str, Any] = {}
        daily_all = []
        for lab, y in label_arrs.items():
            daily, skip = _daily_ic_one(unique_dates, groups, fx, y, ready_arrs[lab])
            daily["factor"] = fid
            daily["label"] = lab
            daily_all.append(daily)
            ic_block[lab] = {
                "skip": skip,
                "ic": _summarize(daily, "ic"),
                "rankic": _summarize(daily, "rankic"),
            }
            roll_block[lab] = {
                "ic": _rolling(daily, "ic", unique_dates),
                "rankic": _rolling(daily, "rankic", unique_dates),
                "maturity_aware": lab == MATURITY_AWARE_LABEL,
                "maturity_by": LABEL_END.get(lab),
            }
            shape_block[lab] = _shape_one(groups, fx, y)
        daily_cat = pd.concat(daily_all, ignore_index=True) if daily_all else pd.DataFrame()
        if len(daily_cat):
            daily_cat.to_parquet(dest / "daily_ic.parquet", index=False)
        dump_json(dest / "ic.json", ic_block)
        dump_json(dest / "rolling.json", roll_block)
        dump_json(dest / "shape.json", shape_block)
        dump_json(
            dest / "top_tail.json",
            {lab: {"top_tail_mean_spread": rec.get("top_tail_mean_spread"), "top2_minus_rest": rec.get("top2_minus_rest"), "q5_minus_q1": rec.get("q5_minus_q1")} for lab, rec in shape_block.items()},
        )
        gh = gh_by.get(fid) or {
            "status": "NOT_IN_GITHUB_EXACT",
            "namespace": "GITHUB_EXACT",
        }
        dump_json(dest / "github_exact.json", gh)
        ric_e2 = finite_or_none(((ic_block.get("label_e2_net") or {}).get("rankic") or {}).get("mean"))
        roll_e2 = finite_or_none(
            ((((roll_block.get("label_e2_net") or {}).get("rankic") or {}).get("windows") or {}).get("ROUND21_INTERNAL_W60") or {}).get("terminal_mean")
        )
        tail_e2 = finite_or_none((shape_block.get("label_e2_net") or {}).get("top_tail_mean_spread"))
        pkg = {
            "factor_id": fid,
            "family": card["family"],
            "status": "DONE",
            "copied_global_result": False,
            "hypothesis_path": str(dest / "hypothesis.yaml"),
            "pit_status": rec_pit.get("pit_status"),
            "ic": ic_block,
            "rolling_namespace": ROUND21_INTERNAL,
            "github_exact_status": gh.get("status"),
            "github_exact_namespace": "GITHUB_EXACT",
            "raw_rankic_e2": ric_e2,
            "rolling_w60_rankic_e2": roll_e2,
            "top_tail_e2": tail_e2,
        }
        dump_json(dest / "package.json", pkg)
        _write_factor_md(fid, pkg)
        packages.append(pkg)
        if ric_e2 is not None:
            best_raw.append({"factor_id": fid, "raw_rankic_e2": ric_e2, "pit_status": rec_pit.get("pit_status")})
        if roll_e2 is not None:
            best_roll.append({"factor_id": fid, "rolling_w60_rankic_e2": roll_e2, "pit_status": rec_pit.get("pit_status")})
        if tail_e2 is not None:
            best_tail.append({"factor_id": fid, "top_tail_e2": tail_e2, "pit_status": rec_pit.get("pit_status")})

    def _top(items: list[dict[str, Any]], key: str, n: int = 5) -> list[dict[str, Any]]:
        return sorted(items, key=lambda r: abs(float(r[key])), reverse=True)[:n]

    payload = {
        "status": "DONE",
        "round": 21,
        "n_packages": len(packages),
        "copied_global_to_all_factors": False,
        "thin_day_rule": f"n_valid < {IC_MIN_VALID_NAMES_PER_DAY} → skip, do not record IC=0",
        "internal_windows": list(INTERNAL_WINDOW_NAMES),
        "github": {
            "namespace": "GITHUB_EXACT",
            "status": github.get("status"),
            "exact": github.get("exact"),
            "scope": github.get("scope"),
            "reason": github.get("reason"),
        },
        "best_raw_rankic_e2": _top(best_raw, "raw_rankic_e2"),
        "best_rolling_w60_rankic_e2": _top(best_roll, "rolling_w60_rankic_e2"),
        "best_top_tail_e2": _top(best_tail, "top_tail_e2"),
        "packages": [
            {
                "factor_id": p["factor_id"],
                "status": p.get("status"),
                "pit_status": p.get("pit_status"),
                "raw_rankic_e2": p.get("raw_rankic_e2"),
                "rolling_w60_rankic_e2": p.get("rolling_w60_rankic_e2"),
                "top_tail_e2": p.get("top_tail_e2"),
                "github_exact_status": p.get("github_exact_status"),
            }
            for p in packages
        ],
        "construct_computed": construct.get("computed"),
    }
    dump_json(EVIDENCE_JSON, payload)
    lines = [
        "# Round21 · 逐因子 Evidence Package",
        "",
        f"packages=`{payload['n_packages']}` copied_global=`False`",
        f"GITHUB_EXACT status=`{github.get('status')}` exact=`{github.get('exact')}` scope=`{github.get('scope')}`",
        f"{ROUND21_INTERNAL} windows=`{list(INTERNAL_WINDOW_NAMES)}`",
        "",
        "## Best raw RankIC (label_e2_net)",
        "",
    ]
    for rec in payload["best_raw_rankic_e2"]:
        lines.append(f"- `{rec['factor_id']}` RankIC=`{rec['raw_rankic_e2']}` pit=`{rec['pit_status']}`")
    lines += ["", "## Best ROUND21_INTERNAL_W60 RankIC", ""]
    for rec in payload["best_rolling_w60_rankic_e2"]:
        lines.append(f"- `{rec['factor_id']}` `{rec['rolling_w60_rankic_e2']}`")
    lines += ["", "## Best top-tail", ""]
    for rec in payload["best_top_tail_e2"]:
        lines.append(f"- `{rec['factor_id']}` `{rec['top_tail_e2']}`")
    lines += ["", "每因子独立目录 `result/round21/factors/<id>/`。未跑 Add/Replace/combo/回测/seed/主报告。", ""]
    write_md(EVIDENCE_MD, "\n".join(lines))
    return payload
