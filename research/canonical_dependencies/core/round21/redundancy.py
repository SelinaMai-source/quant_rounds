"""与 14/13 + R17/R20 KEEP + ZyQuant 350 去冗余。簇阈 |Spearman|≥0.85。簇内必须 Original/New/Add/Replace。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..paths import ROUND21_ROOT
from ..round7.contract import FIVE_FEATURES, TEN_FEATURES
from ..round18.contract import F1_NAME, F2_NAME, F3_NAME
from ..round20.metrics import icir_of, rank_ic
from ..round20.panel import fold_masks
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import (
    BLOCKED_PIT,
    CLUSTER_SPEARMAN_ABS,
    COMPLETE_IC_YEARS,
    IC_YEARS,
    LABELS,
    PIT_ELIGIBLE_MODEL,
    REDUNDANCY_MD,
    RIDGE_ALPHA,
)
from .freeze import require_frozen
from .hypotheses import HYPOTHESES
from .library import build_research_panel, load_r20_keep_names, zyquant_novelty
from .panel import ic_window
from .pit import PIT_JSON, run_pit

RED_JSON = ROUND21_ROOT / "redundancy" / "redundancy.json"
KEEP_CSV = ROUND21_ROOT / "redundancy" / "keep_drop.csv"
EDGE_CSV = ROUND21_ROOT / "redundancy" / "cluster_edges.csv"
CLUSTER_CSV = ROUND21_ROOT / "redundancy" / "cluster_onr.csv"
NOVELTY_CSV = ROUND21_ROOT / "redundancy" / "zyquant350_novelty.csv"
FACTORS_DIR = ROUND21_ROOT / "factors"

HEADS = (
    ("label_e2_net", tuple(FIVE_FEATURES)),
    ("label_exit_rule_net", tuple(TEN_FEATURES)),
)


def _union_find(n: int):
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def _quality_from_package(fid: str) -> dict[str, Any]:
    pkg = load_json(FACTORS_DIR / fid / "package.json")
    ic = (pkg.get("ic") or {}) if pkg else {}

    def _lab(lab: str) -> dict[str, Any]:
        block = ((ic.get(lab) or {}).get("rankic") or {})
        return {
            "mean": finite_or_none(block.get("mean")),
            "icir": finite_or_none(block.get("icir")),
            "n_days_used": int(block.get("n_days_used") or 0),
        }

    e2, ex = _lab("label_e2_net"), _lab("label_exit_rule_net")
    icirs = [abs(v) for v in (e2.get("icir"), ex.get("icir")) if v is not None]
    means = [abs(v) for v in (e2.get("mean"), ex.get("mean")) if v is not None]
    return {
        "e2": e2,
        "exit": ex,
        "best_abs_icir": max(icirs) if icirs else 0.0,
        "best_abs_mean": max(means) if means else 0.0,
        "n_days_used": max(int(e2.get("n_days_used") or 0), int(ex.get("n_days_used") or 0)),
        "raw_rankic_e2": finite_or_none(pkg.get("raw_rankic_e2")),
    }


def _spearman_matrix(panel: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    day_ids, _ = pd.factorize(panel["trade_date"], sort=True)
    order = np.argsort(day_ids, kind="mergesort")
    sorted_ids = day_ids[order]
    bounds = np.append(np.flatnonzero(np.diff(sorted_ids, prepend=-1)), len(order))
    groups = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    mat = {n: pd.to_numeric(panel[n], errors="coerce").to_numpy(dtype=float) if n in panel.columns else np.full(len(panel), np.nan) for n in names}
    acc = np.zeros((len(names), len(names)))
    cnt = np.zeros((len(names), len(names)))
    for idx in groups:
        if len(idx) < 5:
            continue
        cols, data = [], []
        for n in names:
            x = mat[n][idx]
            if int(np.isfinite(x).sum()) < 5 or int(np.unique(x[np.isfinite(x)]).size) < 2:
                continue
            cols.append(n)
            data.append(pd.Series(x).rank(method="average").to_numpy())
        if len(cols) < 2:
            continue
        arr = np.vstack(data)
        arr = arr - np.nanmean(arr, axis=1, keepdims=True)
        finite = np.isfinite(arr)
        arr = np.where(finite, arr, 0.0)
        denom = finite.astype(float) @ finite.T
        cov = arr @ arr.T
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.sqrt(np.clip(np.diag(cov)[:, None] * np.diag(cov)[None, :], 1e-18, None))
        corr = np.where(denom >= 5, corr, np.nan)
        ix = [names.index(c) for c in cols]
        for i, a in enumerate(ix):
            for j, b in enumerate(ix):
                if i >= j:
                    continue
                v = corr[i, j]
                if np.isfinite(v):
                    acc[a, b] += float(v)
                    acc[b, a] += float(v)
                    cnt[a, b] += 1
                    cnt[b, a] += 1
    with np.errstate(invalid="ignore"):
        rho = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
    np.fill_diagonal(rho, 1.0)
    return rho, cnt


def _complete(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    use = [c for c in cols if c in frame.columns]
    work = frame[use].replace([np.inf, -np.inf], np.nan)
    return frame.loc[work.notna().all(axis=1)].copy()


def _ridge_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray | None:
    if len(x_train) < 30 or x_train.shape[1] == 0 or len(x_test) == 0:
        return None
    scaler = StandardScaler()
    try:
        xt = scaler.fit_transform(x_train)
        xe = scaler.transform(x_test)
    except ValueError:
        return None
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(xt, y_train)
    return model.predict(xe)


def _daily_rankic(dates: pd.Series, score: np.ndarray, label: np.ndarray) -> list[float]:
    work = pd.DataFrame({"d": pd.to_datetime(dates).to_numpy(), "s": score, "y": label})
    out: list[float] = []
    for _, day in work.groupby("d", sort=False):
        ic = rank_ic(day["s"], day["y"])
        if ic is not None:
            out.append(ic)
    return out


def _oos_rankic(panel: pd.DataFrame, cols: list[str], lab: str) -> dict[str, Any]:
    years = [y for y in IC_YEARS if int((panel["trade_date"].dt.year == y).sum()) > 0]
    oos: list[float] = []
    n_folds = 0
    for y in years:
        train_m, pred_m = fold_masks(panel, predict_year=y, label=lab)
        need = [*cols, lab]
        tr = _complete(panel.loc[train_m], need)
        te = _complete(panel.loc[pred_m], need)
        if len(tr) < 80 or len(te) < 20:
            continue
        ytr = pd.to_numeric(tr[lab], errors="coerce").to_numpy(dtype=float)
        yte = pd.to_numeric(te[lab], errors="coerce").to_numpy(dtype=float)
        if len(cols) == 1:
            score = pd.to_numeric(te[cols[0]], errors="coerce").to_numpy(dtype=float)
        else:
            score = _ridge_score(tr[cols].to_numpy(dtype=float), ytr, te[cols].to_numpy(dtype=float))
            if score is None:
                continue
        n_folds += 1
        oos.extend(_daily_rankic(te["trade_date"], score, yte))
    arr = np.array(oos, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "mean": float(np.mean(arr)) if len(arr) else None,
        "icir": icir_of(arr) if len(arr) else None,
        "n_oos_days": int(len(arr)),
        "n_folds": n_folds,
        "full_history_fit": False,
    }


def _cluster_onr(panel: pd.DataFrame, new_name: str, old_name: str) -> dict[str, Any]:
    rows = []
    for lab, _locked in HEADS:
        original = _oos_rankic(panel, [old_name], lab)
        new = _oos_rankic(panel, [new_name], lab)
        added = _oos_rankic(panel, [old_name, new_name], lab)
        replace = new
        rows.append(
            {
                "new": new_name,
                "original": old_name,
                "label": lab,
                "Original": original,
                "New": new,
                "Add": added,
                "Replace": replace,
                "add_minus_original": (
                    None
                    if original["mean"] is None or added["mean"] is None
                    else float(added["mean"] - original["mean"])
                ),
                "replace_minus_original": (
                    None
                    if original["mean"] is None or replace["mean"] is None
                    else float(replace["mean"] - original["mean"])
                ),
                "original_locked_14_13": old_name in FIVE_FEATURES or old_name in TEN_FEATURES,
                "cannot_delete_locked_from_v5": old_name in FIVE_FEATURES or old_name in TEN_FEATURES,
            }
        )
    return {"new": new_name, "original": old_name, "by_label": rows}


def run_redundancy(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and RED_JSON.is_file() and KEEP_CSV.is_file() and REDUNDANCY_MD.is_file():
        existing = load_json(RED_JSON)
        if existing.get("status") == "DONE":
            return existing
    pit = run_pit(resume=True)
    pit_map = {r["factor_id"]: r for r in (pit.get("rows") or [])}
    cards = {c["factor_id"]: c for c in HYPOTHESES}
    panel = ic_window(build_research_panel(resume=True))
    keep_old = load_r20_keep_names()
    new_ids = [c["factor_id"] for c in HYPOTHESES]
    pit_pass = [n for n in new_ids if (pit_map.get(n) or {}).get("pit_status") == "PIT_PASS"]
    blocked_pit = [n for n in new_ids if (pit_map.get(n) or {}).get("pit_status") == BLOCKED_PIT]
    # 簇空间：新可算因子 + 14/13 + R20 KEEP（F2 排除）
    names = []
    for n in [*keep_old["all_keep"], *new_ids]:
        if n == F2_NAME:
            continue
        if n not in panel.columns:
            continue
        if n not in names:
            names.append(n)
    rho, cnt = _spearman_matrix(panel, names)
    edges: list[dict[str, Any]] = []
    find, union = _union_find(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            v = rho[i, j]
            if v is None or not np.isfinite(v):
                continue
            if abs(float(v)) >= CLUSTER_SPEARMAN_ABS:
                union(i, j)
                edges.append({"a": a, "b": b, "rho": float(v), "n_days": int(cnt[i, j])})
    clusters: dict[int, list[str]] = {}
    for i, n in enumerate(names):
        clusters.setdefault(find(i), []).append(n)
    cluster_rows = []
    cluster_ids = {}
    for k, members in enumerate(sorted(clusters.values(), key=lambda xs: (-len(xs), xs[0])), start=1):
        cid = f"C{k:02d}"
        for m in members:
            cluster_ids[m] = cid
        cluster_rows.append({"cluster": cid, "n": len(members), "members": members})

    novelty = zyquant_novelty(panel, [n for n in new_ids if n in panel.columns], resume=True)
    novelty_map = {r["factor"]: r for r in novelty}

    decisions: list[dict[str, Any]] = []
    for feat in new_ids:
        rec_pit = pit_map.get(feat) or {}
        q = _quality_from_package(feat)
        reasons: list[str] = []
        action = "KEEP"
        pit_status = rec_pit.get("pit_status")
        if pit_status == BLOCKED_PIT:
            action = "DROP_BLOCKED_PIT"
            reasons.append("E/大宗无公告时点，BLOCKED_PIT。禁 PIT_ELIGIBLE_MODEL。诊断 only。")
        elif pit_status != "PIT_PASS":
            action = "DROP_PIT"
            reasons.append(f"PIT={pit_status}；{rec_pit.get('skip_reason')}")
        elif q["n_days_used"] == 0:
            action = "DROP_NO_CS_IC"
            reasons.append("无可用截面 IC 日。不把 IC 记 0。")
        elif q["best_abs_icir"] < 0.20 and q["best_abs_mean"] < 0.02:
            action = "DROP_WEAK"
            reasons.append(
                f"RankICIR 最大绝对值={q['best_abs_icir']:.3f} 且 |RankIC|={q['best_abs_mean']:.4f}，信息弱。"
                "簇相关不是唯一 DROP 依据。"
            )
        nov = novelty_map.get(feat) or {}
        if nov.get("cluster_ge_085"):
            reasons.append(
                f"ZyQuant350 近邻 `{nov.get('nearest')}` |ρ|={nov.get('max_abs_rho'):.3f}。"
                "Novelty 库不进模型；相关不是唯一 DROP。"
            )
        decisions.append(
            {
                "name": feat,
                "family": (cards.get(feat) or {}).get("family"),
                "cluster": cluster_ids.get(feat),
                "action": action,
                "reasons": reasons,
                "best_abs_icir": q["best_abs_icir"],
                "best_abs_mean": q["best_abs_mean"],
                "n_days_used": q["n_days_used"],
                "pit_status": pit_status,
                "promotion_eligible": pit_status == "PIT_PASS" and action.startswith("KEEP"),
                "model_eligible": PIT_ELIGIBLE_MODEL if (pit_status == "PIT_PASS" and action.startswith("KEEP")) else "BLOCKED",
                "zyquant_nearest": nov.get("nearest"),
                "zyquant_abs_rho": nov.get("max_abs_rho"),
                "near_neighbors": (cards.get(feat) or {}).get("near_neighbors"),
            }
        )
    by_name = {d["name"]: d for d in decisions}
    locked = set(FIVE_FEATURES) | set(TEN_FEATURES)
    for cl in cluster_rows:
        members = cl["members"]
        if len(members) < 2:
            continue
        new_keep = [m for m in members if m in by_name and by_name[m]["action"] == "KEEP"]
        if len(new_keep) < 2:
            continue
        new_keep.sort(key=lambda m: (by_name[m]["best_abs_icir"], by_name[m]["best_abs_mean"], m), reverse=True)
        winner = new_keep[0]
        for m in new_keep[1:]:
            by_name[m]["action"] = "DROP_REDUNDANT"
            by_name[m]["promotion_eligible"] = False
            by_name[m]["reasons"].append(
                f"与 `{winner}` 同簇 {cl['cluster']}，|Spearman|≥{CLUSTER_SPEARMAN_ABS}。"
                f"保留 ICIR 更高的 `{winner}`。"
                "相关不是唯一依据：同时比较了 ICIR。簇内仍跑 Original/New/Add/Replace。"
            )
        locked_members = [m for m in members if m in locked]
        if locked_members:
            by_name[winner]["reasons"].append(
                f"同簇含锁定 14/13 {locked_members}。不从 v5 删锁定列；候选仅作增量审计。"
            )
    decisions = list(by_name.values())

    onr_rows = []
    for cl in cluster_rows:
        if cl["n"] < 2:
            continue
        news = [m for m in cl["members"] if m in by_name]
        olds = [m for m in cl["members"] if m not in by_name]
        if not news or not olds:
            # 纯新簇：彼此也要 ONAR
            if len(news) >= 2:
                olds = news
            else:
                continue
        for new in news:
            for old in olds:
                if new == old:
                    continue
                rec = _cluster_onr(panel, new, old)
                onr_rows.append(rec)
                dest = FACTORS_DIR / new / "redundancy.json"
                dump_json(dest, rec)

    # 无簇的 PIT_PASS 也写单因子 redundancy 钩子
    clustered_new = {r["new"] for r in onr_rows}
    for feat in new_ids:
        if feat in clustered_new:
            continue
        dump_json(
            FACTORS_DIR / feat / "redundancy.json",
            {
                "factor_id": feat,
                "cluster": cluster_ids.get(feat),
                "cluster_members": next((c["members"] for c in cluster_rows if cluster_ids.get(feat) == c["cluster"]), [feat]),
                "Original_New_Add_Replace": "NO_CROSS_FAMILY_CLUSTER",
                "zyquant": novelty_map.get(feat),
                "decision": by_name.get(feat),
            },
        )

    keep_frame = pd.DataFrame(decisions)
    keep_frame["reason_text"] = keep_frame["reasons"].apply(lambda xs: " ".join(xs))
    KEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    keep_frame.drop(columns=["reasons"]).to_csv(KEEP_CSV, index=False)
    pd.DataFrame(edges).to_csv(EDGE_CSV, index=False)
    pd.DataFrame(novelty).to_csv(NOVELTY_CSV, index=False)
    flat_onr = []
    for rec in onr_rows:
        for row in rec.get("by_label") or []:
            flat_onr.append(
                {
                    "new": rec["new"],
                    "original": rec["original"],
                    "label": row["label"],
                    "original_mean": (row["Original"] or {}).get("mean"),
                    "new_mean": (row["New"] or {}).get("mean"),
                    "add_mean": (row["Add"] or {}).get("mean"),
                    "replace_mean": (row["Replace"] or {}).get("mean"),
                    "add_minus_original": row.get("add_minus_original"),
                    "replace_minus_original": row.get("replace_minus_original"),
                    "original_locked": row.get("original_locked_14_13"),
                }
            )
    pd.DataFrame(flat_onr).to_csv(CLUSTER_CSV, index=False)

    n_keep = int(keep_frame["action"].str.startswith("KEEP").sum())
    n_drop = int((~keep_frame["action"].str.startswith("KEEP")).sum())
    n_clustered = int(sum(1 for c in cluster_rows if c["n"] >= 2))
    replacements = [
        r
        for r in flat_onr
        if r.get("replace_minus_original") is not None
        and float(r["replace_minus_original"]) > 0
        and not r.get("original_locked")
    ]
    payload = {
        "status": "DONE",
        "threshold": CLUSTER_SPEARMAN_ABS,
        "cluster_not_sole_drop_reason": True,
        "cluster_must_run": ["Original", "New", "Add", "Replace"],
        "full_history_g": False,
        "n_keep": n_keep,
        "n_drop": n_drop,
        "n_clusters_ge2": n_clustered,
        "n_edges": len(edges),
        "pit_pass": pit_pass,
        "blocked_pit": blocked_pit,
        "blocked_pit_formal_combo": False,
        "f1_f3_rebranded": False,
        "zyquant_enters_model": False,
        "clusters": cluster_rows,
        "decisions": decisions,
        "onr_n": len(onr_rows),
        "replacements_ridge": replacements,
        "novelty_n": len(novelty),
        "csv": str(KEEP_CSV),
        "edges": str(EDGE_CSV),
        "onr_csv": str(CLUSTER_CSV),
        "novelty_csv": str(NOVELTY_CSV),
        "note": "簇内 Original/New/Add/Replace 为 fold Ridge，禁止全历史 g()。锁定 14/13 不删。E BLOCKED_PIT 不进正式组合。",
    }
    dump_json(RED_JSON, payload)
    lines = [
        "# 05 冗余簇与 KEEP/DROP",
        "",
        f"簇阈 `abs(Spearman)≥{CLUSTER_SPEARMAN_ABS}`，**不是唯一 DROP 依据**。",
        f"新因子 KEEP=`{n_keep}` DROP=`{n_drop}` 多成员簇=`{n_clustered}` 边=`{len(edges)}`。",
        "对照库：锁定 14/13 + R17/R20 KEEP + ZyQuant 350（Novelty only，合法键对齐）。",
        "簇内必须 Original / New / Add / Replace（fold Ridge，禁止全历史拟合）。",
        "E `BLOCKED_PIT` 不得进 `PIT_ELIGIBLE_MODEL`。F1/F3 不换名。",
        "",
        "## 多成员簇",
        "",
    ]
    for cl in cluster_rows:
        if cl["n"] < 2:
            continue
        lines.append(f"- `{cl['cluster']}` n={cl['n']}: " + ", ".join(f"`{m}`" for m in cl["members"]))
    lines += ["", "## PIT_PASS 决策", ""]
    for rec in keep_frame.loc[keep_frame["pit_status"].eq("PIT_PASS")].to_dict(orient="records"):
        lines.append(
            f"- `{rec['name']}` `{rec['action']}` ICIR=`{rec['best_abs_icir']:.3f}` "
            f"簇=`{rec['cluster']}` z350=`{rec.get('zyquant_nearest')}` ρ=`{rec.get('zyquant_abs_rho')}`"
        )
    lines += ["", f"csv：`{KEEP_CSV}`", f"ONAR：`{CLUSTER_CSV}`", f"novelty：`{NOVELTY_CSV}`", ""]
    write_md(REDUNDANCY_MD, "\n".join(lines))
    return payload


__all__ = ["CLUSTER_CSV", "EDGE_CSV", "KEEP_CSV", "RED_JSON", "run_redundancy"]
