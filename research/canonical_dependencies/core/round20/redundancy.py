"""Spearman |ρ|≥0.85 簇与 KEEP/DROP 书面理由。簇不是唯一 DROP 依据。F1/F3 不换名。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND20_ROOT
from .common import dump_json, load_json, write_md
from .contract import CLUSTER_SPEARMAN_ABS, F1_NAME, F2_NAME, F3_NAME, FIVE_FEATURES, REDUNDANCY_MD, TEN_FEATURES
from .freeze import require_frozen
from .ic import IC_CSV, run_ic
from .panel import build_diag_panel, ic_window, load_registry
from .pit import run_pit
from .shape import run_shape

RED_JSON = ROUND20_ROOT / "redundancy" / "redundancy.json"
KEEP_CSV = ROUND20_ROOT / "redundancy" / "keep_drop.csv"
EDGE_CSV = ROUND20_ROOT / "redundancy" / "cluster_edges.csv"


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


def _quality(ic_flat: pd.DataFrame, name: str) -> dict[str, Any]:
    sub = ic_flat.loc[ic_flat["factor"].eq(name) & ic_flat["metric"].eq("rankic")]
    e2 = sub.loc[sub["label"].eq("label_e2_net")]
    ex = sub.loc[sub["label"].eq("label_exit_rule_net")]
    def _row(block: pd.DataFrame) -> dict[str, Any]:
        if block.empty:
            return {"mean": None, "icir": None, "n_days_used": 0, "hac_p": None}
        rec = block.iloc[0].to_dict()
        return {
            "mean": rec.get("mean"),
            "icir": rec.get("icir"),
            "n_days_used": rec.get("n_days_used"),
            "hac_p": rec.get("hac_p"),
            "same_sign_years": rec.get("same_sign_years"),
        }
    a, b = _row(e2), _row(ex)
    icirs = [abs(v) for v in (a.get("icir"), b.get("icir")) if v is not None and np.isfinite(v)]
    best_icir = max(icirs) if icirs else 0.0
    means = [abs(v) for v in (a.get("mean"), b.get("mean")) if v is not None and np.isfinite(v)]
    best_mean = max(means) if means else 0.0
    n_used = max(int(a.get("n_days_used") or 0), int(b.get("n_days_used") or 0))
    return {"e2": a, "exit": b, "best_abs_icir": best_icir, "best_abs_mean": best_mean, "n_days_used": n_used}


def run_redundancy(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and RED_JSON.is_file() and KEEP_CSV.is_file() and REDUNDANCY_MD.is_file():
        existing = load_json(RED_JSON)
        if existing.get("status") == "DONE":
            return existing
    pit = {r["name"]: r for r in (run_pit(resume=True).get("rows") or [])}
    run_ic(resume=True)
    run_shape(resume=True)
    ic_flat = pd.read_csv(IC_CSV)
    panel = ic_window(build_diag_panel(resume=True))
    names = [str(x) for x in load_registry()["name"].tolist() if str(x) in panel.columns]
    # 日频截面 Spearman 平均；市场层退化为日均值时序 Spearman
    day_ids, _unique_dates = pd.factorize(panel["trade_date"], sort=True)
    order = np.argsort(day_ids, kind="mergesort")
    sorted_ids = day_ids[order]
    bounds = np.append(np.flatnonzero(np.diff(sorted_ids, prepend=-1)), len(order))
    groups = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    mat = {n: pd.to_numeric(panel[n], errors="coerce").to_numpy(dtype=float) for n in names}
    market = {n: bool((pit.get(n) or {}).get("market_level")) for n in names}
    acc = np.zeros((len(names), len(names)))
    cnt = np.zeros((len(names), len(names)))
    for idx in groups:
        if len(idx) < 5:
            continue
        cols = []
        data = []
        for n in names:
            x = mat[n][idx]
            if market[n]:
                continue
            if int(np.isfinite(x).sum()) < 5 or int(np.unique(x[np.isfinite(x)]).size) < 2:
                continue
            cols.append(n)
            data.append(pd.Series(x).rank(method="average").to_numpy())
        if len(cols) < 2:
            continue
        arr = np.vstack(data)
        # pairwise pearson of ranks = spearman
        arr = arr - np.nanmean(arr, axis=1, keepdims=True)
        # nan-safe: replace nan with 0 after centering finite; skip pairs with few overlap
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
    # 市场层：日均值时序 Spearman
    daily_mean = {}
    for n in names:
        s = pd.Series(mat[n], index=panel.index)
        daily_mean[n] = pd.Series(s.to_numpy(), index=panel["trade_date"]).groupby(level=0).mean()
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            if cnt[i, j] > 0 and not (market[a] or market[b]):
                continue
            sa = daily_mean[a]
            sb = daily_mean[b]
            aligned = pd.concat([sa, sb], axis=1).dropna()
            if len(aligned) < 40:
                continue
            rho = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))
            if not np.isfinite(rho):
                continue
            acc[i, j] = rho * max(cnt[i, j], 1)
            acc[j, i] = acc[i, j]
            cnt[i, j] = max(cnt[i, j], 1)
            cnt[j, i] = cnt[i, j]
    with np.errstate(invalid="ignore"):
        rho = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
    np.fill_diagonal(rho, 1.0)

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
    cluster_ids = {}
    cluster_rows = []
    for k, members in enumerate(sorted(clusters.values(), key=lambda xs: (-len(xs), xs[0])), start=1):
        cid = f"C{k:02d}"
        for m in members:
            cluster_ids[m] = cid
        cluster_rows.append({"cluster": cid, "n": len(members), "members": members})

    decisions: list[dict[str, Any]] = []
    for feat in names:
        rec_pit = pit.get(feat) or {}
        q = _quality(ic_flat, feat)
        reasons: list[str] = []
        action = "KEEP"
        locked = feat in FIVE_FEATURES or feat in TEN_FEATURES
        if feat == F2_NAME:
            action = "DROP_PROMOTION"
            reasons.append("F2 PIT_TIMESTAMP_UNVERIFIED，不得进可晋级模型。仍保留审计 IC，不删库。")
        elif locked:
            action = "KEEP_LOCKED"
            reasons.append("R7 锁定 14/13，KEEP。LOFO 诊断不得从 v5 删列。不是新假设。")
        elif rec_pit.get("pit_status") not in {"PIT_PASS", "PIT_INHERITED_LOCKED"}:
            action = "DROP_PIT"
            reasons.append(f"PIT={rec_pit.get('pit_status')}；{rec_pit.get('skip_reason')}")
        elif q["n_days_used"] == 0:
            action = "DROP_NO_CS_IC"
            reasons.append("无可用截面 IC 日（薄日跳过或日内无变异）。不把 IC 记 0。")
        elif q["best_abs_icir"] < 0.20 and q["best_abs_mean"] < 0.02:
            action = "DROP_WEAK"
            reasons.append(
                f"RankICIR 最大绝对值={q['best_abs_icir']:.3f} 且 |RankIC|={q['best_abs_mean']:.4f}，"
                "信息弱。簇相关不是唯一 DROP 依据。"
            )
        if feat in {F1_NAME, F3_NAME}:
            reasons.append("R18 公式已冻结，不得换名重开假设。")
            if action == "KEEP":
                action = "KEEP_AUDIT"
        decisions.append(
            {
                "name": feat,
                "family": rec_pit.get("family"),
                "cluster": cluster_ids.get(feat),
                "action": action,
                "reasons": reasons,
                "best_abs_icir": q["best_abs_icir"],
                "best_abs_mean": q["best_abs_mean"],
                "n_days_used": q["n_days_used"],
                "promotion_eligible": bool(rec_pit.get("promotion_eligible")) and action.startswith("KEEP") and feat != F2_NAME,
                "f1_f3_not_rebranded": feat in {F1_NAME, F3_NAME},
                "locked": locked,
            }
        )
    by_name = {d["name"]: d for d in decisions}
    # 簇内再裁：同一簇多个 KEEP 候选，留 RankICIR 最高者，其余 DROP_REDUNDANT（并写清还有弱 IC/锁定原因）
    for cl in cluster_rows:
        members = cl["members"]
        if len(members) < 2:
            continue
        keepable = [
            m
            for m in members
            if by_name[m]["action"] in {"KEEP", "KEEP_AUDIT"} and not by_name[m]["locked"]
        ]
        locked_members = [m for m in members if by_name[m]["locked"]]
        if keepable:
            keepable.sort(key=lambda m: (by_name[m]["best_abs_icir"], by_name[m]["best_abs_mean"], m), reverse=True)
            winner = keepable[0]
            for m in keepable[1:]:
                by_name[m]["action"] = "DROP_REDUNDANT"
                by_name[m]["promotion_eligible"] = False
                by_name[m]["reasons"].append(
                    f"与 `{winner}` 同簇 {cl['cluster']}，|Spearman|≥{CLUSTER_SPEARMAN_ABS}。"
                    f"保留 ICIR 更高的 `{winner}`（{by_name[winner]['best_abs_icir']:.3f} vs {by_name[m]['best_abs_icir']:.3f}）。"
                    "相关不是唯一依据：同时比较了 ICIR。"
                )
            if locked_members:
                by_name[winner]["reasons"].append(
                    f"同簇含锁定列 {locked_members}。不从 v5 删锁定列；候选仅作增量审计。"
                )
    decisions = list(by_name.values())
    keep_frame = pd.DataFrame(decisions)
    keep_frame["reason_text"] = keep_frame["reasons"].apply(lambda xs: " ".join(xs))
    KEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    keep_frame.drop(columns=["reasons"]).to_csv(KEEP_CSV, index=False)
    pd.DataFrame(edges).to_csv(EDGE_CSV, index=False)
    n_keep = int(keep_frame["action"].str.startswith("KEEP").sum())
    n_drop = int((~keep_frame["action"].str.startswith("KEEP")).sum())
    n_clustered = int(sum(1 for c in cluster_rows if c["n"] >= 2))
    payload = {
        "status": "DONE",
        "threshold": CLUSTER_SPEARMAN_ABS,
        "cluster_not_sole_drop_reason": True,
        "n_keep": n_keep,
        "n_drop": n_drop,
        "n_clusters_ge2": n_clustered,
        "n_edges": len(edges),
        "f1_f3_rebranded": False,
        "f2_promotion_eligible": False,
        "clusters": cluster_rows,
        "decisions": decisions,
        "csv": str(KEEP_CSV),
        "edges": str(EDGE_CSV),
        "note": "KEEP/DROP 书面理由已写。F1/F3 不换名。F2 不晋级。锁定列不删。",
    }
    dump_json(RED_JSON, payload)
    lines = [
        "# 05 冗余簇与 KEEP/DROP",
        "",
        f"初始簇阈 `abs(Spearman)≥{CLUSTER_SPEARMAN_ABS}`，**不是唯一 DROP 依据**。",
        f"KEEP=`{n_keep}` DROP=`{n_drop}` 多成员簇=`{n_clustered}` 边=`{len(edges)}`。",
        "F1/F3 不换名。F2 不得晋级。锁定 14/13 一律 KEEP_LOCKED。",
        "",
        "## 多成员簇",
        "",
    ]
    for cl in cluster_rows:
        if cl["n"] < 2:
            continue
        lines.append(f"- `{cl['cluster']}` n={cl['n']}: " + ", ".join(f"`{m}`" for m in cl["members"]))
    lines += ["", "## KEEP 非锁定（可进后续组合研究的候选）", ""]
    cand = keep_frame.loc[keep_frame["action"].isin(["KEEP", "KEEP_AUDIT"])]
    if cand.empty:
        lines.append("无。")
    for rec in cand.sort_values("best_abs_icir", ascending=False).to_dict(orient="records"):
        lines.append(
            f"- `{rec['name']}` `{rec['action']}` ICIR=`{rec['best_abs_icir']:.3f}` "
            f"簇=`{rec['cluster']}`：{rec['reason_text']}"
        )
    lines += ["", f"csv：`{KEEP_CSV}`", ""]
    write_md(REDUNDANCY_MD, "\n".join(lines))
    return payload


__all__ = ["EDGE_CSV", "KEEP_CSV", "RED_JSON", "run_redundancy"]
