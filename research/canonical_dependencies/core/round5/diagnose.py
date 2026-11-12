"""A：锁分稳定性诊断。复用 Round 4 Random Top1，禁止重抽。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND4_ROOT, ROUND5_ROOT, assert_project_result_path, ensure_round5_tree
from ..stock_readonly import load_fs00263_ensemble_predictions
from .factors import build_factor_panel
from .registry import LOCKED_FEATURES, V3_CAGR, V3_EX2024

COMPLETE_YEARS = tuple(range(2018, 2026))


def _write_json(rel: str, payload: Any) -> None:
    path = ensure_round5_tree() / rel
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _pct(v: Any, d: int = 2) -> str:
    if v is None:
        return "—"
    return f"{float(v) * 100:+.{d}f}%"


def _num(v: Any, d: int = 3) -> str:
    if v is None:
        return "—"
    return f"{float(v):.{d}f}"


def _load_r4_random() -> dict[str, Any]:
    path = ROUND4_ROOT / "01_attr" / "attribution.json"
    if not path.is_file():
        raise FileNotFoundError("缺少 Round 4 Random Top1，禁止重抽。先确认 result/round4/01_attr/attribution.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _v3_yearly() -> dict[str, Any]:
    report = json.loads((ROUND4_ROOT / "R4_B05" / "report.json").read_text(encoding="utf-8"))
    return report.get("metrics") or {}


def _join_scores_labels(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    pred = predictions.copy()
    pred["signal_date"] = pd.to_datetime(pred["signal_date"])
    pred["instrument_id"] = pred["instrument_id"].astype(str)
    lab = panel.copy()
    lab["trade_date"] = pd.to_datetime(lab["trade_date"])
    lab["instrument_id"] = lab["instrument_id"].astype(str)
    keep = ["trade_date", "instrument_id", "label_e2_net", "label_exit_rule_net", "label_d1_touch",
            "actual_exit_date", *list(LOCKED_FEATURES)]
    keep = [c for c in keep if c in lab.columns]
    merged = pred.merge(
        lab[keep].rename(columns={"trade_date": "signal_date"}),
        on=["signal_date", "instrument_id"],
        how="left",
    )
    return merged


def _rank_ic(block: pd.DataFrame, score="score", y="label_e2_net") -> float | None:
    s = pd.to_numeric(block[score], errors="coerce")
    t = pd.to_numeric(block[y], errors="coerce")
    mask = s.notna() & t.notna()
    if mask.sum() < 5:
        return None
    if s[mask].nunique() < 2 or t[mask].nunique() < 2:
        return None
    val = s[mask].corr(t[mask], method="spearman")
    if val is None or not np.isfinite(val):
        return None
    return float(val)


def _ndcg_at(block: pd.DataFrame, k: int, score="score", y="label_e2_net") -> float | None:
    from sklearn.metrics import ndcg_score

    s = pd.to_numeric(block[score], errors="coerce")
    t = pd.to_numeric(block[y], errors="coerce")
    mask = s.notna() & t.notna()
    if mask.sum() < 2:
        return None
    rel = t[mask].rank(method="average").to_numpy(dtype=float)
    return float(ndcg_score([rel], [s[mask].to_numpy()], k=k, ignore_ties=False))


def _top1_metrics(merged: pd.DataFrame) -> dict[str, Any]:
    ranked = merged.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True])
    top1 = ranked.groupby("signal_date", sort=True).head(1).copy()
    y = pd.to_numeric(top1.get("label_e2_net"), errors="coerce")
    day_mean = merged.groupby("signal_date")["label_e2_net"].mean()
    top1 = top1.merge(day_mean.rename("day_mean"), on="signal_date", how="left")
    lift = None
    if y.notna().any() and pd.to_numeric(top1["day_mean"], errors="coerce").replace(0, np.nan).notna().any():
        base = pd.to_numeric(top1["day_mean"], errors="coerce").replace(0, np.nan)
        lift = float((y / base).median())
    touch = pd.to_numeric(top1.get("label_d1_touch"), errors="coerce")
    big = (y <= -0.05) if y.notna().any() else pd.Series(dtype=float)
    return {
        "n": int(len(top1)),
        "win_rate": float((y > 0).mean()) if y.notna().any() else None,
        "mean_ret": float(y.mean()) if y.notna().any() else None,
        "top1_lift": lift,
        "continue_rate": float(touch.mean()) if touch.notna().any() else None,
        "bigloss_rate": float(big.mean()) if len(big) else None,
    }


def yearly_alpha_table(merged: pd.DataFrame) -> dict[str, Any]:
    rows = {}
    merged = merged.copy()
    merged["year"] = pd.to_datetime(merged["signal_date"]).dt.year
    for year, block in merged.groupby("year", sort=True):
        ics, n1, n3 = [], [], []
        for _, day in block.groupby("signal_date"):
            ic = _rank_ic(day)
            if ic is not None:
                ics.append(ic)
            v = _ndcg_at(day, 1)
            if v is not None:
                n1.append(v)
            v3 = _ndcg_at(day, 3)
            if v3 is not None:
                n3.append(v3)
        top = _top1_metrics(block)
        rows[str(int(year))] = {
            "rank_ic": float(np.mean(ics)) if ics else None,
            "ndcg@1": float(np.mean(n1)) if n1 else None,
            "ndcg@3": float(np.mean(n3)) if n3 else None,
            **top,
            "ytd": int(year) == 2026,
        }
    return rows


def factor_year_ic(merged: pd.DataFrame) -> dict[str, Any]:
    out = {}
    merged = merged.copy()
    merged["year"] = pd.to_datetime(merged["signal_date"]).dt.year
    for feat in LOCKED_FEATURES:
        if feat not in merged.columns:
            continue
        yearly = {}
        ics = []
        for year, block in merged.groupby("year"):
            day_ics = []
            for _, day in block.groupby("signal_date"):
                ic = _rank_ic(day, score=feat)
                if ic is not None:
                    day_ics.append(ic)
            mu = float(np.mean(day_ics)) if day_ics else None
            yearly[str(int(year))] = mu
            if mu is not None and int(year) in COMPLETE_YEARS:
                ics.append(mu)
        mean = float(np.mean(ics)) if ics else None
        std = float(np.std(ics, ddof=1)) if len(ics) >= 2 else None
        icir = (mean / std) if mean is not None and std not in (None, 0) else None
        signs = [float(np.sign(x)) for x in ics if x is not None and np.isfinite(x)]
        same = int(abs(sum(signs))) if signs else 0
        role = "弱"
        if icir is not None and abs(icir) >= 0.50 and same >= 5:
            role = "稳定核"
        elif same >= 6:
            role = "年份稳定"
        elif mean is not None and abs(mean) < 0.01:
            role = "弱"
        out[feat] = {"yearly": yearly, "mean": mean, "icir": icir, "same_sign_years": int(same), "role": role}
    # 冗余：|ρ|>0.90
    feats = [f for f in LOCKED_FEATURES if f in merged.columns]
    corr = merged[feats].corr(method="spearman")
    red = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            rho = corr.loc[a, b]
            if pd.notna(rho) and abs(float(rho)) > 0.90:
                red.append({"a": a, "b": b, "rho": float(rho)})
    return {"factors": out, "redundant_pairs": red}


def psi_ks(merged: pd.DataFrame) -> dict[str, Any]:
    from scipy.stats import ks_2samp

    base = merged.loc[pd.to_datetime(merged["signal_date"]).dt.year.between(2018, 2023)]
    y24 = merged.loc[pd.to_datetime(merged["signal_date"]).dt.year == 2024]
    y25 = merged.loc[pd.to_datetime(merged["signal_date"]).dt.year == 2025]
    rows = {}
    for col in ["score", *LOCKED_FEATURES]:
        if col not in merged.columns:
            continue

        def _psi(a, b, bins=10):
            aa = pd.to_numeric(a, errors="coerce").dropna()
            bb = pd.to_numeric(b, errors="coerce").dropna()
            if len(aa) < 50 or len(bb) < 50:
                return None
            qs = np.linspace(0, 1, bins + 1)
            edges = np.unique(np.quantile(aa, qs))
            if len(edges) < 3:
                return None
            pa = np.clip(np.histogram(aa, bins=edges)[0] / len(aa), 1e-6, 1)
            pb = np.clip(np.histogram(bb, bins=edges)[0] / len(bb), 1e-6, 1)
            return float(np.sum((pb - pa) * np.log(pb / pa)))

        rows[col] = {
            "psi_2024_vs_2018_2023": _psi(base[col], y24[col]),
            "psi_2025_vs_2018_2023": _psi(base[col], y25[col]),
            "ks_2024": None,
            "ks_2025": None,
        }
        aa = pd.to_numeric(base[col], errors="coerce").dropna()
        for lab, blk in (("ks_2024", y24), ("ks_2025", y25)):
            bb = pd.to_numeric(blk[col], errors="coerce").dropna()
            if len(aa) > 50 and len(bb) > 50:
                rows[col][lab] = float(ks_2samp(aa, bb).statistic)
    return rows


def selection_alpha_windows(v3_nav: pd.DataFrame, random_navs: list[pd.DataFrame]) -> dict[str, Any]:
    def _daily(nav):
        ordered = nav.sort_values("trade_date").copy()
        ordered["trade_date"] = pd.to_datetime(ordered["trade_date"])
        s = ordered.set_index("trade_date")["total_asset"].astype(float).pct_change().dropna()
        return s

    strat = _daily(v3_nav)
    rnd = [_daily(n) for n in random_navs if n is not None and not n.empty]
    if not rnd:
        return {}
    aligned = pd.concat(rnd, axis=1)
    med = aligned.median(axis=1)
    out = {}
    for win in (60, 120, 250):
        s = (1 + strat).rolling(win).apply(lambda x: float(np.prod(x) - 1), raw=True)
        r = (1 + med.reindex(strat.index)).rolling(win).apply(lambda x: float(np.prod(x) - 1), raw=True)
        diff = s - r
        out[str(win)] = {
            "median": float(diff.median()) if diff.notna().any() else None,
            "mean": float(diff.mean()) if diff.notna().any() else None,
            "p10": float(diff.quantile(0.10)) if diff.notna().any() else None,
        }
    return out


def _plot(yearly, factor_ic, psi, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest.mkdir(parents=True, exist_ok=True)
    years = [str(y) for y in range(2018, 2027) if str(y) in yearly]
    # 1 IC / NDCG
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(years, [yearly[y].get("rank_ic") for y in years], marker="o", label="Rank IC")
    ax.plot(years, [yearly[y].get("ndcg@1") for y in years], marker="o", label="NDCG@1")
    ax.plot(years, [yearly[y].get("ndcg@3") for y in years], marker="o", label="NDCG@3")
    ax.legend()
    ax.set_title("Fig1 yearly Rank IC / NDCG")
    fig.tight_layout()
    fig.savefig(dest / "fig1_ic_ndcg.png", dpi=120)
    plt.close(fig)
    # 2 lift / win
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(years, [yearly[y].get("win_rate") for y in years], marker="o", label="win")
    ax.plot(years, [yearly[y].get("continue_rate") for y in years], marker="o", label="continue")
    ax.plot(years, [yearly[y].get("bigloss_rate") for y in years], marker="o", label="bigloss")
    ax.legend()
    ax.set_title("Fig2 Top1 win / continue / bigloss")
    fig.tight_layout()
    fig.savefig(dest / "fig2_top1_rates.png", dpi=120)
    plt.close(fig)
    # 3 lift
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(years, [yearly[y].get("top1_lift") or 0 for y in years])
    ax.set_title("Fig3 Top1 lift")
    fig.tight_layout()
    fig.savefig(dest / "fig3_top1_lift.png", dpi=120)
    plt.close(fig)
    # 4 factor IC heatmap
    feats = list((factor_ic.get("factors") or {}))
    mat = []
    for f in feats:
        mat.append([(factor_ic["factors"][f]["yearly"] or {}).get(y) for y in years])
    if mat:
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(np.array(mat, dtype=float), aspect="auto", cmap="RdBu")
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, rotation=45)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=7)
        fig.colorbar(im, ax=ax)
        ax.set_title("Fig4 six-factor yearly IC")
        fig.tight_layout()
        fig.savefig(dest / "fig4_factor_ic.png", dpi=120)
        plt.close(fig)
    # 5 PSI
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(psi)
    ax.barh(names, [psi[n].get("psi_2024_vs_2018_2023") or 0 for n in names])
    ax.set_title("Fig5 PSI 2024 vs 2018-2023")
    fig.tight_layout()
    fig.savefig(dest / "fig5_psi_2024.png", dpi=120)
    plt.close(fig)
    # 8 score hist by era
    # 9 placeholder written by caller if needed


def write_diagnose(
    *,
    predictions=None,
    resume: bool = True,
    run_now: bool = True,
) -> dict[str, Any]:
    ensure_round5_tree()
    js = ROUND5_ROOT / "01_diagnose" / "diagnose.json"
    md = ROUND5_ROOT / "01_alpha_stability.md"
    if resume and js.is_file() and md.is_file():
        return json.loads(js.read_text(encoding="utf-8"))
    if not run_now:
        return {"status": "pending"}

    preds = predictions if predictions is not None else load_fs00263_ensemble_predictions()
    panel = build_factor_panel(resume=True)
    merged = _join_scores_labels(preds, panel)
    yearly = yearly_alpha_table(merged)
    factor_ic = factor_year_ic(merged)
    drift = psi_ks(merged)
    r4 = _load_r4_random()
    v3m = _v3_yearly()
    v3_2024 = (v3m.get("yearly_returns") or {}).get("2024")
    rand = r4.get("random_top1") or {}
    rand_2024 = rand.get("ret_2024_median")
    sel_alpha = None if v3_2024 is None or rand_2024 is None else float(v3_2024) - float(rand_2024)

    random_navs = []
    for seed in rand.get("seeds") or []:
        p = ROUND4_ROOT / f"R4_RAND_{seed}" / "nav.parquet"
        if p.is_file():
            random_navs.append(pd.read_parquet(p))
    v3_nav = pd.read_parquet(ROUND4_ROOT / "R4_B05" / "nav.parquet")
    windows = selection_alpha_windows(v3_nav, random_navs)

    # score / margin / ensemble
    from .overlays import margin_frame_from_scores

    margin = margin_frame_from_scores(preds)
    score_stats = {
        "score_std_by_year": merged.groupby(pd.to_datetime(merged["signal_date"]).dt.year)["score"]
        .std().to_dict(),
        "margin_p250_mean_2024": float(margin.loc[margin.index.year == 2024, "margin_p250"].mean())
        if (margin.index.year == 2024).any() else None,
        "margin_p250_mean_2018_2023": float(
            margin.loc[margin.index.year.isin(range(2018, 2024)), "margin_p250"].mean()
        ) if (margin.index.year.isin(range(2018, 2024))).any() else None,
    }

    eras = {}
    for name, years in (("2018-2023", range(2018, 2024)), ("2024", [2024]), ("2025", [2025])):
        sl = merged.loc[pd.to_datetime(merged["signal_date"]).dt.year.isin(years)]
        eras[name] = {
            "rank_ic": yearly_alpha_table(sl).get(str(list(years)[0] if len(years) == 1 else "agg")),
            "top1": _top1_metrics(sl),
        }
        ics = []
        for _, day in sl.groupby("signal_date"):
            ic = _rank_ic(day)
            if ic is not None:
                ics.append(ic)
        eras[name]["rank_ic"] = float(np.mean(ics)) if ics else None

    fig_dir = ensure_round5_tree() / "figures"
    _plot(yearly, factor_ic, drift, fig_dir)
    # fig8 margin
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(margin.index, margin["top1_top2_margin"], lw=0.6)
    ax.set_title("Fig8 Top1-Top2 margin")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig8_margin.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name, years in (("2018-2023", range(2018, 2024)), ("2024", [2024]), ("2025", [2025])):
        sl = merged.loc[pd.to_datetime(merged["signal_date"]).dt.year.isin(years), "score"]
        ax.hist(pd.to_numeric(sl, errors="coerce").dropna(), bins=40, alpha=0.4, label=name, density=True)
    ax.legend()
    ax.set_title("Fig9 score distribution by era")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig9_score_era.png", dpi=120)
    plt.close(fig)

    payload = {
        "written_at": datetime.now().astimezone().isoformat(),
        "random_top1_reused": True,
        "random_seeds": rand.get("seeds"),
        "yearly": yearly,
        "factor_ic": factor_ic,
        "drift": drift,
        "eras": eras,
        "score_stats": score_stats,
        "selection_alpha": {
            "v3_2024": v3_2024,
            "random_top1_2024_median": rand_2024,
            "selection_alpha_2024": sel_alpha,
            "windows": windows,
            "note": "Random Top1 复用 Round 4，未重抽。",
        },
        "v3": {
            "cagr": v3m.get("annualized_return"),
            "sharpe": v3m.get("sharpe_zero_rate"),
            "max_drawdown": v3m.get("max_drawdown"),
            "cagr_ex2024": v3m.get("cagr_ex2024"),
            "completed_trades": v3m.get("completed_trades"),
        },
        "anchor": {"cagr": V3_CAGR, "ex2024": V3_EX2024},
    }
    _write_json("01_diagnose/diagnose.json", payload)
    _write_md(payload)
    return payload


def _write_md(payload: dict[str, Any]) -> None:
    yearly = payload.get("yearly") or {}
    ev = payload.get("selection_alpha") or {}
    fic = payload.get("factor_ic") or {}
    lines = [
        "# 01 Alpha 稳定性（锁分 fs00263）",
        "",
        f"- 生成时间：{payload.get('written_at')}",
        "- 对照：`baseline_v3` = `R4_B05`",
        "- Random Top1：**复用 Round 4**，未重抽。",
        "- 性质：诊断，不调参。",
        "",
        "## 逐年 Rank IC / NDCG / Top1",
        "",
        "| 年 | Rank IC | NDCG@1 | NDCG@3 | Top1 lift | 胜率 | 连板 | 大亏 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for year in [str(y) for y in range(2018, 2027)]:
        row = yearly.get(year) or {}
        if not row:
            continue
        tag = " YTD" if row.get("ytd") else ""
        lines.append(
            f"| {year}{tag} | {_num(row.get('rank_ic'))} | {_num(row.get('ndcg@1'))} | {_num(row.get('ndcg@3'))} "
            f"| {_num(row.get('top1_lift'))} | {_pct(row.get('win_rate'))} | {_pct(row.get('continue_rate'))} "
            f"| {_pct(row.get('bigloss_rate'))} |"
        )
    lines += [
        "",
        "## Selection Alpha（复用 R4 Random Top1）",
        "",
        f"| 项 | 值 |",
        f"|---|---:|",
        f"| v3 2024 | {_pct(ev.get('v3_2024'))} |",
        f"| Event Beta（随机中位） | {_pct(ev.get('random_top1_2024_median'))} |",
        f"| Selection Alpha | {_pct(ev.get('selection_alpha_2024'))} |",
        "",
        "60/120/250 日窗口（策略滚动收益 − 随机中位滚动收益）：",
        "",
        "| 窗 | 中位 | 均值 | p10 |",
        "|---:|---:|---:|---:|",
    ]
    for win, row in (ev.get("windows") or {}).items():
        lines.append(f"| {win} | {_pct(row.get('median'))} | {_pct(row.get('mean'))} | {_pct(row.get('p10'))} |")
    lines += [
        "",
        "## 六因子年度 IC",
        "",
        "| 因子 | 角色 | 均值 IC | ICIR | 同号年数 |",
        "|---|---|---:|---:|---:|",
    ]
    for feat, row in (fic.get("factors") or {}).items():
        lines.append(
            f"| `{feat}` | {row.get('role')} | {_num(row.get('mean'))} | {_num(row.get('icir'))} "
            f"| {row.get('same_sign_years')} |"
        )
    red = fic.get("redundant_pairs") or []
    lines += ["", "## 冗余 |ρ|>0.90", "", ("无" if not red else json.dumps(red, ensure_ascii=False)), ""]
    eras = payload.get("eras") or {}
    lines += ["## 2018–2023 vs 2024 vs 2025", "", "| 段 | Rank IC | Top1 胜率 | Top1 均收益 |", "|---|---:|---:|---:|"]
    for name, row in eras.items():
        top = row.get("top1") or {}
        lines.append(f"| {name} | {_num(row.get('rank_ic'))} | {_pct(top.get('win_rate'))} | {_pct(top.get('mean_ret'))} |")
    lines += [
        "",
        "图：`result/round5/figures/fig1_ic_ndcg.png` … `fig5_psi_2024.png`、`fig8_margin.png`、`fig9_score_era.png`。",
        "",
    ]
    path = ROUND5_ROOT / "01_alpha_stability.md"
    assert_project_result_path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_diagnose_status() -> dict[str, Any]:
    path = ROUND5_ROOT / "01_diagnose" / "diagnose.json"
    if not path.is_file() or not (ROUND5_ROOT / "01_alpha_stability.md").is_file():
        return {"passed": False, "reason": "diagnose 未写"}
    return {"passed": True, **json.loads(path.read_text(encoding="utf-8"))}
