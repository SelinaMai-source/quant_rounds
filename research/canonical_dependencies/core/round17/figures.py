"""25 张图。缺数据不造假。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND17_ROOT, assert_project_result_path, ensure_round17_tree
from .common import load_json

FIG_DIR = ROUND17_ROOT / "figures"
FIG_JSON = ROUND17_ROOT / "figures" / "figures.json"


def _save(fig, name: str) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    path = FIG_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return str(path)


def make_figures() -> dict[str, Any]:
    ensure_round17_tree()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {}
    skipped = []

    def skip(name, reason):
        skipped.append({"name": name, "reason": reason})

    # 1 A00 vs A01 nav
    try:
        a00 = pd.read_parquet(ROUND17_ROOT / "parity" / "R17_A00" / "nav.parquet")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pd.to_datetime(a00["trade_date"]), a00["total_asset"], label="A00")
        a01p = ROUND17_ROOT / "fourway" / "R17_A01" / "nav.parquet"
        if a01p.is_file():
            a01 = pd.read_parquet(a01p)
            ax.plot(pd.to_datetime(a01["trade_date"]), a01["total_asset"], label="A01")
        ax.legend()
        ax.set_title("A00 vs A01 NAV")
        paths["nav_a00_a01"] = _save(fig, "01_nav_a00_a01.png")
        plt.close(fig)
    except Exception as exc:
        skip("01_nav_a00_a01.png", str(exc))

    four = load_json(ROUND17_ROOT / "fourway" / "fourway.json")
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels, sharpes = [], []
        if four.get("A00"):
            labels.append("A00"); sharpes.append(four["A00"].get("sharpe"))
        if four.get("A01"):
            labels.append("A01"); sharpes.append(four["A01"].get("sharpe"))
        for item in (four.get("finalists") or [])[:12]:
            labels.append(item.get("B02_id"))
            sharpes.append((item.get("B02") or {}).get("sharpe"))
        x = np.arange(len(labels))
        ax.bar(x, [float(s or 0) if s is not None else np.nan for s in sharpes])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title("Sharpe four-way")
        paths["sharpe_fourway"] = _save(fig, "02_sharpe_fourway.png")
        plt.close(fig)
    except Exception as exc:
        skip("02_sharpe_fourway.png", str(exc))

    top = load_json(ROUND17_ROOT / "select" / "top100.json")
    try:
        recs = top.get("top100") or []
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([r.get("selection_score") for r in recs])
        ax.set_title("Top100 selection_score")
        paths["top100_score"] = _save(fig, "03_top100_score.png")
        plt.close(fig)
    except Exception as exc:
        skip("03_top100_score.png", str(exc))

    search = []
    root = ROUND17_ROOT / "search" / "trials"
    if root.is_dir():
        rows = []
        for p in list(root.glob("R17_T*/metrics.json"))[:5000]:
            rec = load_json(p)
            if rec.get("dev_sharpe") is not None:
                rows.append(float(rec["dev_sharpe"]))
        if rows:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(rows, bins=40)
            ax.set_title("Discovery Sharpe histogram (SUCCESS)")
            paths["disc_sharpe_hist"] = _save(fig, "04_disc_sharpe_hist.png")
            plt.close(fig)
        else:
            skip("04_disc_sharpe_hist.png", "no success sharpes")
    else:
        skip("04_disc_sharpe_hist.png", "no search trials")

    attr = load_json(ROUND17_ROOT / "attribution" / "increments.json")
    try:
        rows = attr.get("rows") or []
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(np.arange(len(rows)), [r.get("factor_sharpe") or np.nan for r in rows], label="factor")
        ax.bar(np.arange(len(rows)), [r.get("ensemble_sharpe") or np.nan for r in rows], alpha=0.5, label="ensemble")
        ax.legend()
        ax.set_title("Factor vs ensemble Sharpe increments")
        paths["increments"] = _save(fig, "05_increments.png")
        plt.close(fig)
    except Exception as exc:
        skip("05_increments.png", str(exc))

    names = [
        "06_dd_fourway.png", "07_cagr_fourway.png", "08_yearly_a00.png", "09_yearly_best_b02.png",
        "10_seed_corr_preflight.png", "11_top12_val_sharpe.png", "12_maxdd_scatter.png",
        "13_ndcg_hist.png", "14_family_coverage.png", "15_sleeve_mix.png",
        "16_bootstrap_diff.png", "17_cpcv_box.png", "18_stress_bps.png", "19_capacity.png",
        "20_spa.png", "21_pbo.png", "22_nav_best_b02.png", "23_concentration.png",
        "24_top1_identity.png", "25_selection_score_components.png",
    ]
    # additional real plots from available artifacts
    try:
        pre = load_json(ROUND17_ROOT / "preflight" / "preflight.json")
        five = ((pre.get("five") or {}).get("seed_diversity") or {})
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["unique_hashes", "mean_corr", "top1_agree"], [
            five.get("unique_prediction_hashes") or 0,
            five.get("mean_pairwise_corr") or 0,
            five.get("top1_mean_agreement") or 0,
        ])
        ax.set_title("Preflight five-head diversity")
        paths["preflight_div"] = _save(fig, "10_seed_corr_preflight.png")
        plt.close(fig)
        names.remove("10_seed_corr_preflight.png")
    except Exception as exc:
        skip("10_seed_corr_preflight.png", str(exc))

    try:
        a00 = four.get("A00") or {}
        y = a00.get("yearly_returns") or {}
        if y:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(list(y.keys()), [float(v) for v in y.values()])
            ax.set_title("A00 yearly returns")
            paths["yearly_a00"] = _save(fig, "08_yearly_a00.png")
            plt.close(fig)
            names.remove("08_yearly_a00.png")
        else:
            skip("08_yearly_a00.png", "no yearly")
    except Exception as exc:
        skip("08_yearly_a00.png", str(exc))

    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels, dds = [], []
        for lab, rec in (("A00", four.get("A00")), ("A01", four.get("A01"))):
            if rec:
                labels.append(lab); dds.append(rec.get("max_drawdown"))
        for item in (four.get("finalists") or [])[:12]:
            labels.append(item.get("B02_id"))
            dds.append((item.get("B02") or {}).get("max_drawdown"))
        ax.bar(np.arange(len(labels)), [float(v) if v is not None else np.nan for v in dds])
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title("MaxDD four-way")
        paths["dd_fourway"] = _save(fig, "06_dd_fourway.png")
        plt.close(fig)
        if "06_dd_fourway.png" in names:
            names.remove("06_dd_fourway.png")
    except Exception as exc:
        skip("06_dd_fourway.png", str(exc))

    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels, cagrs = [], []
        for lab, rec in (("A00", four.get("A00")), ("A01", four.get("A01"))):
            if rec:
                labels.append(lab); cagrs.append(rec.get("cagr"))
        for item in (four.get("finalists") or [])[:12]:
            labels.append(item.get("B02_id"))
            cagrs.append((item.get("B02") or {}).get("cagr"))
        ax.bar(np.arange(len(labels)), [float(v) if v is not None else np.nan for v in cagrs])
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title("CAGR four-way")
        paths["cagr_fourway"] = _save(fig, "07_cagr_fourway.png")
        plt.close(fig)
        if "07_cagr_fourway.png" in names:
            names.remove("07_cagr_fourway.png")
    except Exception as exc:
        skip("07_cagr_fourway.png", str(exc))

    try:
        best = None
        best_sh = None
        for item in four.get("finalists") or []:
            sh = (item.get("B02") or {}).get("sharpe")
            if sh is None:
                continue
            if best_sh is None or float(sh) > best_sh:
                best_sh = float(sh)
                best = item
        y = ((best or {}).get("B02") or {}).get("yearly_returns") or {}
        if y:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(list(y.keys()), [float(v) for v in y.values()])
            ax.set_title(f"{(best or {}).get('B02_id')} yearly returns")
            paths["yearly_best_b02"] = _save(fig, "09_yearly_best_b02.png")
            plt.close(fig)
            if "09_yearly_best_b02.png" in names:
                names.remove("09_yearly_best_b02.png")
        else:
            skip("09_yearly_best_b02.png", "no yearly")
    except Exception as exc:
        skip("09_yearly_best_b02.png", str(exc))

    try:
        t12 = load_json(ROUND17_ROOT / "select" / "top12.json")
        recs = t12.get("rows") or []
        if recs:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar([r.get("trial_id") for r in recs], [float(r.get("val_sharpe") or np.nan) for r in recs])
            ax.set_xticklabels([r.get("trial_id") for r in recs], rotation=90, fontsize=7)
            ax.set_title("Top12 2022-2023 val Sharpe")
            paths["top12_val"] = _save(fig, "11_top12_val_sharpe.png")
            plt.close(fig)
            if "11_top12_val_sharpe.png" in names:
                names.remove("11_top12_val_sharpe.png")
        else:
            skip("11_top12_val_sharpe.png", "no top12")
    except Exception as exc:
        skip("11_top12_val_sharpe.png", str(exc))

    try:
        recs = (top.get("top100") or [])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            [r.get("dev_sharpe") for r in recs],
            [r.get("dev_maxdd") for r in recs],
            s=12,
        )
        ax.set_xlabel("dev Sharpe")
        ax.set_ylabel("dev MaxDD")
        ax.set_title("Top100 discovery Sharpe vs MaxDD")
        paths["maxdd_scatter"] = _save(fig, "12_maxdd_scatter.png")
        plt.close(fig)
        if "12_maxdd_scatter.png" in names:
            names.remove("12_maxdd_scatter.png")
    except Exception as exc:
        skip("12_maxdd_scatter.png", str(exc))

    try:
        recs = (top.get("top100") or [])
        sleeves = {}
        for r in recs:
            sleeves[str(r.get("sleeve"))] = sleeves.get(str(r.get("sleeve")), 0) + 1
        if sleeves:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(list(sleeves), list(sleeves.values()))
            ax.set_title("Top100 sleeve mix")
            paths["sleeve_mix"] = _save(fig, "15_sleeve_mix.png")
            plt.close(fig)
            if "15_sleeve_mix.png" in names:
                names.remove("15_sleeve_mix.png")
        else:
            skip("15_sleeve_mix.png", "no sleeves")
    except Exception as exc:
        skip("15_sleeve_mix.png", str(exc))

    try:
        boot = load_json(ROUND17_ROOT / "bootstrap" / "bootstrap.json")
        pairs = boot.get("pairs") or {}
        if pairs:
            fig, ax = plt.subplots(figsize=(7, 3))
            labs = list(pairs)
            ax.bar(labs, [float((pairs[k] or {}).get("mean_diff_sharpe") or 0) for k in labs])
            ax.set_xticklabels(labs, rotation=30, fontsize=8)
            ax.set_title("Block-bootstrap mean Sharpe diff vs A00")
            paths["bootstrap_diff"] = _save(fig, "16_bootstrap_diff.png")
            plt.close(fig)
            if "16_bootstrap_diff.png" in names:
                names.remove("16_bootstrap_diff.png")
        else:
            skip("16_bootstrap_diff.png", "no pairs")
    except Exception as exc:
        skip("16_bootstrap_diff.png", str(exc))

    try:
        stress = load_json(ROUND17_ROOT / "stress" / "stress.json")
        fig, ax = plt.subplots(figsize=(7, 3))
        xs, ys, labs = [], [], []
        for i, (key, rec) in enumerate((stress.get("results") or {}).items()):
            xs.append(i)
            ys.append(float(rec.get("base_sharpe") or np.nan))
            labs.append(str(key))
            ax.plot([i, i], [rec.get("plus20", {}).get("sharpe"), rec.get("plus30", {}).get("sharpe")], marker="o")
        ax.scatter(xs, ys, label="base")
        ax.set_xticks(xs)
        ax.set_xticklabels(labs)
        ax.set_title("+20/+30bp Sharpe")
        paths["stress"] = _save(fig, "18_stress_bps.png")
        plt.close(fig)
        if "18_stress_bps.png" in names:
            names.remove("18_stress_bps.png")
    except Exception as exc:
        skip("18_stress_bps.png", str(exc))

    try:
        cap = load_json(ROUND17_ROOT / "capacity" / "capacity.json")
        rows = cap.get("rows") or []
        if rows:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar([str(int(r["nav"])) for r in rows], [r.get("sharpe") for r in rows])
            ax.set_title("Capacity Sharpe")
            paths["capacity"] = _save(fig, "19_capacity.png")
            plt.close(fig)
            if "19_capacity.png" in names:
                names.remove("19_capacity.png")
        else:
            skip("19_capacity.png", "no capacity")
    except Exception as exc:
        skip("19_capacity.png", str(exc))

    try:
        spa = load_json(ROUND17_ROOT / "spa" / "spa.json")
        inner = spa.get("spa") or {}
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(
            ["median", "p90", "best", "A00_dev"],
            [
                inner.get("median_dev_sharpe") or np.nan,
                inner.get("p90_dev_sharpe") or np.nan,
                inner.get("best_dev_sharpe") or np.nan,
                inner.get("a00_dev_sharpe") or np.nan,
            ],
        )
        ax.set_title("Discovery Sharpe max-stat summary")
        paths["spa"] = _save(fig, "20_spa.png")
        plt.close(fig)
        if "20_spa.png" in names:
            names.remove("20_spa.png")
    except Exception as exc:
        skip("20_spa.png", str(exc))

    try:
        b02p = None
        for item in four.get("finalists") or []:
            sh = (item.get("B02") or {}).get("sharpe")
            if sh is None:
                continue
            path = ROUND17_ROOT / "fourway" / str(item.get("B02_id")) / "nav.parquet"
            if path.is_file() and (b02p is None or float(sh) > float(b02p[0])):
                b02p = (float(sh), path, item.get("B02_id"))
        a00p = ROUND17_ROOT / "parity" / "R17_A00" / "nav.parquet"
        if b02p and a00p.is_file():
            fig, ax = plt.subplots(figsize=(8, 4))
            a00n = pd.read_parquet(a00p)
            b02n = pd.read_parquet(b02p[1])
            ax.plot(pd.to_datetime(a00n["trade_date"]), a00n["total_asset"], label="A00")
            ax.plot(pd.to_datetime(b02n["trade_date"]), b02n["total_asset"], label=str(b02p[2]))
            ax.legend()
            ax.set_title("A00 vs best B02 NAV")
            paths["nav_best_b02"] = _save(fig, "22_nav_best_b02.png")
            plt.close(fig)
            if "22_nav_best_b02.png" in names:
                names.remove("22_nav_best_b02.png")
        else:
            skip("22_nav_best_b02.png", "nav missing")
    except Exception as exc:
        skip("22_nav_best_b02.png", str(exc))

    try:
        recs = top.get("top100") or []
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([r.get("rp_sharpe") or r.get("dev_sharpe") for r in recs], label="sharpe")
        ax.plot([r.get("selection_score") for r in recs], label="score")
        ax.legend()
        ax.set_title("Top100 selection score")
        paths["score_components"] = _save(fig, "25_selection_score_components.png")
        plt.close(fig)
        if "25_selection_score_components.png" in names:
            names.remove("25_selection_score_components.png")
    except Exception as exc:
        skip("25_selection_score_components.png", str(exc))

    remaining = [n for n in names if n not in {Path(p).name for p in paths.values() if p}]
    for name in remaining:
        skip(name, "insufficient real data; not faked")

    payload = {"n_written": len(paths), "n_skipped": len(skipped), "paths": paths, "skipped": skipped, "faked": False}
    from .common import dump_json
    dump_json(FIG_JSON, payload)
    return payload
