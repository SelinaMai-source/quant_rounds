"""PIT 风险面板 + 诊断。诊断不得改公式。本 stage 在 freeze 之前。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND14_ROOT, assert_project_result_path, ensure_round14_tree
from ..round10.contract import HERMES_DAILY_ADJ
from ..round13.weights import extract_top_weights
from .common import dump_json, load_json, write_md
from .contract import (
    INCLUDE_D,
    REF_MIN_OBS,
    REF_SIGNAL_DAYS,
    RHO_MIN_OBS,
    RHO_WINDOW,
    SIGMA_MIN_OBS,
    SIGMA_WINDOW,
)
from .parity import load_parity_status
from .runner import build_version_overlayed

RISK_PARQUET = ROUND14_ROOT / "risk" / "risk_panel.parquet"
RISK_JSON = ROUND14_ROOT / "risk" / "risk_panel_meta.json"
DIAG_JSON = ROUND14_ROOT / "risk" / "diagnostics.json"
DIAG_MD = ROUND14_ROOT / "01_risk_feature_diagnostics.md"


def _require_parity() -> dict[str, Any]:
    parity = load_parity_status()
    if not parity.get("passed"):
        raise RuntimeError("REPRODUCIBILITY_FAIL_OR_MISSING STOP")
    return parity


def _v4_daily() -> pd.DataFrame:
    ov, _, _ = build_version_overlayed("v4")
    top = extract_top_weights(ov, "v4")
    rows = []
    for day, g in top.groupby("signal_date", sort=False):
        g = g.sort_values("engine_rank")
        rec: dict[str, Any] = {"signal_date": pd.Timestamp(day).normalize()}
        for _, r in g.iterrows():
            rk = int(r["engine_rank"])
            rec[f"id{rk}"] = str(r["instrument_id"])
            rec[f"w{rk}"] = float(r["target_nav_fraction"])
            rec[f"m{rk}"] = float(r["allocation_multiplier"])
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    if "id2" not in out.columns:
        out["id2"] = pd.NA
        out["w2"] = 0.0
        out["m2"] = np.nan
    out["w1"] = pd.to_numeric(out.get("w1"), errors="coerce").fillna(0.0)
    out["w2"] = pd.to_numeric(out.get("w2"), errors="coerce").fillna(0.0)
    return out


def _load_returns(ids: list[str]) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    if not HERMES_DAILY_ADJ.exists():
        raise RuntimeError(f"HERMES daily_post_adjusted missing: {HERMES_DAILY_ADJ}")
    import pyarrow.dataset as ds

    uniq = sorted({str(x) for x in ids if str(x) not in {"nan", "None", ""}})
    table = ds.dataset(str(HERMES_DAILY_ADJ), format="parquet").to_table(
        columns=["trade_date", "instrument_id", "close_post"],
        filter=ds.field("instrument_id").isin(uniq),
    )
    adj = table.to_pandas()
    adj["trade_date"] = pd.to_datetime(adj["trade_date"]).dt.normalize()
    adj["instrument_id"] = adj["instrument_id"].astype(str)
    adj = adj.sort_values(["instrument_id", "trade_date"])
    adj["ret"] = adj.groupby("instrument_id", sort=False)["close_post"].pct_change()
    wide = adj.pivot(index="trade_date", columns="instrument_id", values="ret").sort_index()
    return wide, wide.index


def _window_end(calendar: pd.DatetimeIndex, day: pd.Timestamp) -> int:
    # include D: last position where calendar <= day
    pos = int(calendar.searchsorted(pd.Timestamp(day), side="right")) - 1
    return pos


def _slice_rets(wide: pd.DataFrame, calendar: pd.DatetimeIndex, end_pos: int, n: int, col: str) -> pd.Series:
    if end_pos < 0 or col not in wide.columns:
        return pd.Series(dtype=float)
    start = max(0, end_pos - n + 1)
    return wide.iloc[start : end_pos + 1][col]


def _ann_vol(rets: pd.Series, min_obs: int) -> float:
    x = pd.to_numeric(rets, errors="coerce").dropna()
    if len(x) < min_obs:
        return float("nan")
    s = float(x.std(ddof=1))
    if not np.isfinite(s):
        return float("nan")
    return s * np.sqrt(252.0)


def _rho(a: pd.Series, b: pd.Series, min_obs: int) -> float:
    pair = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(pair) < min_obs:
        return float("nan")
    return float(pair["a"].corr(pair["b"]))


def _sigma_mrc(w1: float, w2: float, s1: float, s2: float, rho: float, rho_ok: bool) -> dict[str, float]:
    out = {
        "sigma_portfolio": float("nan"),
        "MRC2": float("nan"),
        "RC_SHARE2": float("nan"),
        "corr_adjust": float(rho_ok),
    }
    if not (np.isfinite(s1) and np.isfinite(s2) and s1 >= 0 and s2 >= 0):
        return out
    use_rho = float(rho) if rho_ok and np.isfinite(rho) else 0.0
    cov = s1 * s2 * use_rho
    sigma = np.array([[s1 * s1, cov], [cov, s2 * s2]], dtype=float)
    w = np.array([float(w1), float(w2)], dtype=float)
    sw = sigma @ w
    var_p = float(w @ sw)
    if var_p <= 0 or not np.isfinite(var_p):
        return out
    sig_p = float(np.sqrt(var_p))
    mrc2 = float(w[1] * sw[1] / sig_p) if sig_p > 0 else float("nan")
    rc2 = float(w[1] * sw[1] / var_p) if var_p > 0 else float("nan")
    out["sigma_portfolio"] = sig_p
    out["MRC2"] = mrc2
    out["RC_SHARE2"] = rc2
    return out


def _rolling_median(values: np.ndarray, window: int, min_obs: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        hist = values[max(0, i - window) : i]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= min_obs:
            out[i] = float(np.median(hist))
    return out


def build_risk_panel(daily: pd.DataFrame) -> pd.DataFrame:
    ids = []
    for col in ("id1", "id2"):
        if col in daily.columns:
            ids.extend(daily[col].dropna().astype(str).tolist())
    wide, calendar = _load_returns(ids)
    rows = []
    for rec in daily.itertuples(index=False):
        day = pd.Timestamp(rec.signal_date).normalize()
        end_pos = _window_end(calendar, day)
        a = str(getattr(rec, "id1", ""))
        b = str(getattr(rec, "id2", ""))
        r1 = _slice_rets(wide, calendar, end_pos, SIGMA_WINDOW, a)
        r2 = _slice_rets(wide, calendar, end_pos, SIGMA_WINDOW, b)
        s1 = _ann_vol(r1, SIGMA_MIN_OBS)
        s2 = _ann_vol(r2, SIGMA_MIN_OBS)
        rr1 = _slice_rets(wide, calendar, end_pos, RHO_WINDOW, a)
        rr2 = _slice_rets(wide, calendar, end_pos, RHO_WINDOW, b)
        rho = _rho(rr1, rr2, RHO_MIN_OBS)
        rho_ok = bool(np.isfinite(rho))
        nxt1 = nxt2 = float("nan")
        if end_pos + 1 < len(calendar):
            nxt_day = calendar[end_pos + 1]
            if a in wide.columns and nxt_day in wide.index:
                nxt1 = float(wide.loc[nxt_day, a]) if pd.notna(wide.loc[nxt_day, a]) else float("nan")
            if b in wide.columns and nxt_day in wide.index:
                nxt2 = float(wide.loc[nxt_day, b]) if pd.notna(wide.loc[nxt_day, b]) else float("nan")
        w1 = float(rec.w1)
        w2 = float(rec.w2)
        pack = _sigma_mrc(w1, w2, s1, s2, rho, rho_ok)
        rows.append({
            "signal_date": day,
            "id1": a,
            "id2": b,
            "w1_v4": w1,
            "w2_v4": w2,
            "sigma1_20": s1,
            "sigma2_20": s2,
            "rho12_60": rho,
            "rho_available": rho_ok,
            "sigma_portfolio": pack["sigma_portfolio"],
            "MRC2": pack["MRC2"],
            "RC_SHARE2": pack["RC_SHARE2"],
            "sigma1_missing": not np.isfinite(s1),
            "sigma2_missing": not np.isfinite(s2),
            "next_ret1": nxt1,
            "next_ret2": nxt2,
            "next_port": (w1 * nxt1 + w2 * nxt2) if np.isfinite(nxt1) and np.isfinite(nxt2) else float("nan"),
            "include_d": INCLUDE_D,
        })
    panel = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    panel["median_sigma2_250"] = _rolling_median(panel["sigma2_20"].to_numpy(dtype=float), REF_SIGNAL_DAYS, REF_MIN_OBS)
    panel["median_sigma_portfolio_250"] = _rolling_median(
        panel["sigma_portfolio"].to_numpy(dtype=float), REF_SIGNAL_DAYS, REF_MIN_OBS
    )
    panel["median_MRC2_250"] = _rolling_median(panel["MRC2"].to_numpy(dtype=float), REF_SIGNAL_DAYS, REF_MIN_OBS)
    panel["fallback_sigma2"] = ~np.isfinite(panel["sigma2_20"]) | ~np.isfinite(panel["median_sigma2_250"])
    panel["fallback_mrc"] = (
        ~np.isfinite(panel["MRC2"]) | ~np.isfinite(panel["median_MRC2_250"]) | (panel["MRC2"] <= 0)
    )
    panel["fallback_port"] = ~np.isfinite(panel["sigma_portfolio"]) | ~np.isfinite(panel["median_sigma_portfolio_250"])
    panel["mrc_factor_is_one"] = panel["MRC2"] <= 0
    return panel


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(x) < 30:
        return None
    return float(x["a"].corr(x["b"], method="spearman"))


def _qtable(panel: pd.DataFrame, feat: str, target: str, nq: int = 5) -> list[dict[str, Any]]:
    x = pd.to_numeric(panel[feat], errors="coerce")
    y = pd.to_numeric(panel[target], errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 50:
        return []
    q = pd.qcut(x.loc[ok], nq, duplicates="drop")
    out = []
    for label, idx in y.loc[ok].groupby(q, observed=False).groups.items():
        yy = y.loc[idx]
        out.append({
            "bucket": str(label),
            "n": int(len(yy)),
            "mean_target": float(yy.mean()),
            "mean_loss": float((-yy.clip(upper=0)).mean()),
            "tail_share": float((yy <= np.nanpercentile(y.loc[ok], 10)).mean()),
        })
    return out


def run_diagnostics(panel: pd.DataFrame) -> dict[str, Any]:
    future_loss = (-pd.to_numeric(panel["next_port"], errors="coerce")).clip(lower=0)
    future_top2_loss = (-pd.to_numeric(panel["next_ret2"], errors="coerce")).clip(lower=0)
    feats = ["sigma1_20", "sigma2_20", "rho12_60", "sigma_portfolio", "MRC2", "RC_SHARE2"]
    spearman = {}
    for f in feats:
        spearman[f] = {
            "vs_next_port": _spearman(panel[f], panel["next_port"]),
            "vs_next_ret2": _spearman(panel[f], panel["next_ret2"]),
            "vs_port_loss": _spearman(panel[f], future_loss),
            "vs_top2_loss": _spearman(panel[f], future_top2_loss),
        }
    by_year = {}
    years = pd.to_datetime(panel["signal_date"]).dt.year
    for year, g in panel.groupby(years):
        by_year[str(int(year))] = {
            "n": int(len(g)),
            "sigma2_median": float(g["sigma2_20"].median()) if g["sigma2_20"].notna().any() else None,
            "mrc2_median": float(g["MRC2"].median()) if g["MRC2"].notna().any() else None,
            "rho_available_share": float(g["rho_available"].mean()) if len(g) else None,
            "fallback_sigma2_share": float(g["fallback_sigma2"].mean()) if len(g) else None,
            "spearman_sigma2_vs_top2_loss": _spearman(g["sigma2_20"], future_top2_loss.reindex(g.index)),
        }
    quantiles = {
        "sigma2_20_vs_next_ret2": _qtable(panel, "sigma2_20", "next_ret2"),
        "MRC2_vs_next_port": _qtable(panel, "MRC2", "next_port"),
        "sigma_portfolio_vs_next_port": _qtable(panel, "sigma_portfolio", "next_port"),
    }
    desc = {}
    for f in feats:
        s = pd.to_numeric(panel[f], errors="coerce")
        desc[f] = {
            "n": int(s.notna().sum()),
            "missing_share": float(s.isna().mean()),
            "p10": float(s.quantile(0.10)) if s.notna().any() else None,
            "p50": float(s.quantile(0.50)) if s.notna().any() else None,
            "p90": float(s.quantile(0.90)) if s.notna().any() else None,
        }
    return {
        "status": "DONE",
        "formula_changed": False,
        "note": "诊断不得改公式。分位/Spearman/分年仅描述。",
        "n_days": int(len(panel)),
        "include_d": INCLUDE_D,
        "missing": {
            "sigma1": float(panel["sigma1_missing"].mean()),
            "sigma2": float(panel["sigma2_missing"].mean()),
            "rho": float((~panel["rho_available"]).mean()),
            "fallback_sigma2": float(panel["fallback_sigma2"].mean()),
            "fallback_mrc": float(panel["fallback_mrc"].mean()),
            "fallback_port": float(panel["fallback_port"].mean()),
        },
        "describe": desc,
        "spearman": spearman,
        "quantiles": quantiles,
        "by_year": by_year,
    }


def run_risk(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    _require_parity()
    ensure_round14_tree()
    if resume and RISK_PARQUET.is_file() and RISK_JSON.is_file() and DIAG_MD.is_file():
        existing = load_json(RISK_JSON)
        if existing.get("status") == "DONE":
            return existing
    daily = _v4_daily()
    panel = build_risk_panel(daily)
    assert_project_result_path(RISK_PARQUET)
    panel.to_parquet(RISK_PARQUET, index=False)
    diag = run_diagnostics(panel)
    dump_json(DIAG_JSON, diag)
    meta = {
        "status": "DONE",
        "path": str(RISK_PARQUET),
        "n_days": int(len(panel)),
        "include_d": INCLUDE_D,
        "sigma_window": SIGMA_WINDOW,
        "sigma_min_obs": SIGMA_MIN_OBS,
        "rho_window": RHO_WINDOW,
        "rho_min_obs": RHO_MIN_OBS,
        "ref_signal_days": REF_SIGNAL_DAYS,
        "ref_min_obs": REF_MIN_OBS,
        "missing_fill_zero": False,
        "fallback_to_v4": True,
        "hermes": str(HERMES_DAILY_ADJ),
        "diagnostics_changed_formulas": False,
        "missing": diag.get("missing"),
    }
    dump_json(RISK_JSON, meta)
    lines = [
        "# Round 14 · 风险特征诊断",
        "",
        "PIT：含 D 收盘。缺失不填 0，回退 v4 原仓。**诊断不得改公式。**",
        "",
        f"- 交易日数 = **{len(panel)}**",
        f"- sigma2 缺失比例 = {diag['missing']['sigma2']}",
        f"- rho 不可用比例 = {diag['missing']['rho']}（显式标记，不默默当 0）",
        f"- fallback_sigma2 / mrc / port = {diag['missing']['fallback_sigma2']} / {diag['missing']['fallback_mrc']} / {diag['missing']['fallback_port']}",
        "",
        "## 分布",
        "",
        "| 特征 | n | missing | p10 | p50 | p90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for f, d in (diag.get("describe") or {}).items():
        lines.append(
            f"| `{f}` | {d.get('n')} | {d.get('missing_share')} | {d.get('p10')} | {d.get('p50')} | {d.get('p90')} |"
        )
    lines += ["", "## Spearman（特征 vs 未来收益/亏损）", ""]
    for f, row in (diag.get("spearman") or {}).items():
        lines.append(
            f"- `{f}`: next_port={row.get('vs_next_port')} next_ret2={row.get('vs_next_ret2')} "
            f"port_loss={row.get('vs_port_loss')} top2_loss={row.get('vs_top2_loss')}"
        )
    lines += ["", "## 分年", "", "| 年 | n | sigma2 p50 | MRC2 p50 | rho可用 | fallback_sigma2 |", "|---|---:|---:|---:|---:|---:|"]
    for year, row in (diag.get("by_year") or {}).items():
        lines.append(
            f"| {year} | {row.get('n')} | {row.get('sigma2_median')} | {row.get('mrc2_median')} | "
            f"{row.get('rho_available_share')} | {row.get('fallback_sigma2_share')} |"
        )
    lines += [
        "",
        "分位表见 `risk/diagnostics.json`。本文件只描述，不回写公式。",
        "",
    ]
    write_md(DIAG_MD, "\n".join(lines))
    return meta


def load_risk_panel() -> pd.DataFrame:
    if not RISK_PARQUET.is_file():
        return pd.DataFrame()
    return pd.read_parquet(RISK_PARQUET)


def load_risk_meta() -> dict[str, Any]:
    return load_json(RISK_JSON)


def require_risk() -> dict[str, Any]:
    meta = load_risk_meta()
    if meta.get("status") != "DONE" or not RISK_PARQUET.is_file() or not DIAG_MD.is_file():
        raise RuntimeError("RISK_PANEL_OR_DIAGNOSTICS_MISSING STOP — freeze 前必须先写诊断")
    return meta


__all__ = [
    "build_risk_panel",
    "load_risk_meta",
    "load_risk_panel",
    "require_risk",
    "run_risk",
]
