"""所有者修补：C5–C11 折内物化 + C8 泄漏重训。

属主审计发现：
1. 兄弟实现的 C8 用「全样本」shrunk_ic 拟合权重后物化进面板 → 监督权重前视。
   本模块按折重拟合：年 y 的行只用 [y-3, y-1] 窗口估计（与正典 3y/1y WF 对齐）。
2. C5–C7/C9–C11 原为 OOF_DIAGNOSTIC（未物化）。本模块把生成列按同一折内口径
   物化进 4 个头面板（`_fi` 后缀 = fold-internal），真实训练 + 全袖套。
3. 物化数学严格镜像 research/factor_combination 各模块（fit 在窗口、apply 在目标年）；
   历史不足的 fold 输出 NaN，不填 0。XGB 原生处理 NaN。
4. 泄漏列 r21_c8_shrunk_ic 从面板移除（备份在 data/backup_pre_materialize/）。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..paths import PROJECT_ROOT, ROUND21_4_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import COMBO_MATRIX, MD, PREREGISTERED_INTERACTIONS, RAW_BASELINE, exp_dir
from .train import DATA, train_one

sys.path.insert(0, str(PROJECT_ROOT))
from research.factor_combination.factor_family_graph import graph  # noqa: E402

MAT_JSON = ROUND21_4_ROOT / "combo" / "materialize.json"
BACKUP_DIR = DATA / "backup_pre_materialize"
PANEL_KEYS = ("five_training", "five_candidate", "ten_training", "ten_candidate")
KEYS = ["trade_date", "instrument_id"]
LEAKED_C8_COL = "r21_c8_shrunk_ic"

# k 固定：用首个预测年（2018）的训练窗 2015–2017 按 var=0.95 选定后冻结（嵌套纪律）
K_ANCHOR_YEAR = 2018


# ---------------------------------------------------------------------------
# 折内拟合数学（镜像 research/factor_combination；fit/apply 分离）
# ---------------------------------------------------------------------------

def _num(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return frame[list(cols)].apply(pd.to_numeric, errors="coerce")


def _z_fit(win: pd.DataFrame, feats: Sequence[str]) -> tuple[pd.Series, pd.Series]:
    x = _num(win, feats)
    mu = x.mean()
    sd = x.std(ddof=1).replace(0, np.nan)
    return mu, sd


def _z_apply(frame: pd.DataFrame, feats: Sequence[str], mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    return (_num(frame, feats) - mu) / sd


def _pca_fit(win: pd.DataFrame, feats: Sequence[str], *, var: float | None, k: int | None):
    """返回 (mu, sd, vt, k)。镜像 dimension_reduction._pca_cols。"""
    mu, sd = _z_fit(win, feats)
    z = _z_apply(win, feats, mu, sd).fillna(0.0)
    if z.shape[0] < 10 or not feats:
        return mu, sd, None, 0
    arr = z.to_numpy(dtype=float)
    arr = arr - arr.mean(axis=0)
    _, s, vt = np.linalg.svd(arr, full_matrices=False)
    expl = (s ** 2) / max((s ** 2).sum(), 1e-12)
    if k is None:
        k = int(np.searchsorted(np.cumsum(expl), float(var or 0.95)) + 1)
    k = max(1, min(int(k), arr.shape[1]))
    vt = vt[:k].copy()
    # 符号对齐：每个成分 |loading| 最大的特征取正号，保证折间符号约定确定
    for j in range(k):
        idx = int(np.argmax(np.abs(vt[j])))
        if vt[j, idx] < 0:
            vt[j] = -vt[j]
    return mu, sd, vt, k


def _pca_apply(frame: pd.DataFrame, feats: Sequence[str], mu, sd, vt, k: int, names: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    if vt is None or k <= 0:
        for n in names:
            out[n] = np.nan
        return out
    z = _z_apply(frame, feats, mu, sd).fillna(0.0).to_numpy(dtype=float)
    z = z - np.nan_to_num(mu.to_numpy(dtype=float), nan=0.0) * 0.0  # 与 fit 同口径：z 已减均值
    scores = z @ vt.T
    for i, n in enumerate(names):
        out[n] = scores[:, i] if i < scores.shape[1] else np.nan
    return out


def _ridge_beta(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x + float(alpha) * np.eye(x.shape[1])
    try:
        return np.linalg.solve(xtx, x.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(xtx, x.T @ y, rcond=None)[0]


def _fit_predict_linear(model: str, fit: pd.DataFrame, target: pd.DataFrame, feats: Sequence[str], label: str) -> pd.Series:
    """镜像 supervised_composites 的 ridge/elasticnet/pls（fit/apply 分离）。"""
    x = _num(fit, feats)
    y = pd.to_numeric(fit[label], errors="coerce")
    mask = y.notna() & x.notna().any(axis=1)
    if int(mask.sum()) < 30:
        return pd.Series(np.nan, index=target.index)
    xm = x.loc[mask].fillna(0.0).to_numpy(dtype=float)
    ym = y.loc[mask].to_numpy(dtype=float)
    full = _num(target, feats).fillna(0.0).to_numpy(dtype=float)
    if model == "ridge":
        beta = _ridge_beta(xm, ym, 1.0)
        pred = full @ beta
    elif model == "elasticnet":
        alpha, l1 = 1.0, 0.5
        yc = ym - ym.mean()
        xc = xm - xm.mean(axis=0)
        beta = np.zeros(xm.shape[1])
        l1_pen, l2_pen = alpha * l1, alpha * (1.0 - l1)
        for _ in range(80):
            for j in range(xm.shape[1]):
                r = yc - xc @ beta + beta[j] * xc[:, j]
                rho = float(xc[:, j] @ r)
                z = float(xc[:, j] @ xc[:, j]) + l2_pen
                if z <= 0:
                    beta[j] = 0.0
                elif rho > l1_pen:
                    beta[j] = (rho - l1_pen) / z
                elif rho < -l1_pen:
                    beta[j] = (rho + l1_pen) / z
                else:
                    beta[j] = 0.0
        pred = (full - xm.mean(axis=0)) @ beta
    else:  # pls（镜像模块：首权重 w0 打分）
        yc = ym - ym.mean()
        xc = xm - xm.mean(axis=0)
        w0 = xc.T @ yc
        w0 = w0 / (np.linalg.norm(w0) or 1.0)
        pred = (full - xm.mean(axis=0)) @ w0
    return pd.Series(pred, index=target.index)


def _shrunk_ic_weights(win: pd.DataFrame, feats: Sequence[str], label: str, shrink: float) -> dict[str, float]:
    w: dict[str, float] = {}
    y = pd.to_numeric(win[label], errors="coerce")
    for f in feats:
        x = pd.to_numeric(win[f], errors="coerce")
        mask = x.notna() & y.notna()
        ic = float(x.loc[mask].corr(y.loc[mask], method="spearman") or 0.0) if int(mask.sum()) >= 30 else 0.0
        w[f] = (1.0 - float(shrink)) * ic
    s = sum(abs(v) for v in w.values()) or 1.0
    return {k: v / s for k, v in w.items()}


# ---------------------------------------------------------------------------
# 物化
# ---------------------------------------------------------------------------

def _interaction_pairs(frame_cols: Sequence[str]) -> list[tuple[str, str]]:
    pairs = []
    for a, b in PREREGISTERED_INTERACTIONS:
        ca = next((c for c in frame_cols if c.startswith(f"r21_{a.lower()}")), None)
        cb = next((c for c in frame_cols if c.startswith(f"r21_{b.lower()}")), None)
        if ca and cb:
            pairs.append((ca, cb))
    return pairs[:6]


def _family_map() -> dict[str, list[str]]:
    return graph(list(FIVE_FEATURES))


def fi_column_names(k_gpc: int) -> dict[str, list[str]]:
    fam = _family_map()
    multi = [k for k, v in fam.items() if len(v) >= 2]
    pairs_placeholder = ["a", "b"]  # 名称在物化时解析
    del pairs_placeholder
    return {
        "C5": [f"r21_c5_gpc_pc{i+1}" for i in range(k_gpc)],
        "C6": [f"r21_c6_fpc_{k}_pc1" for k in multi],
        "C7": ["r21_c7_g0_ridge_resid_fi"],
        "C8": ["r21_c8_shrunk_ic_fi"],
        "C9": ["r21_c9_linear_combo_fi"],
        "C10": [],  # 物化时按解析对命名
        "C11": [f"r21_c11_fam_{k}" for k in fam],
    }


def materialize(*, resume: bool = True) -> dict[str, Any]:
    if resume and MAT_JSON.is_file():
        existing = load_json(MAT_JSON)
        if existing.get("status") == "DONE":
            return existing
    base = pd.read_parquet(DATA / "five_candidate.parquet")
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    base["instrument_id"] = base["instrument_id"].astype(str)
    feats = [c for c in FIVE_FEATURES if c in base.columns]
    label = FIVE_HEAD.target
    fam = _family_map()
    multi = {k: v for k, v in fam.items() if len(v) >= 2}
    pairs = _interaction_pairs(base.columns)

    # k 冻结：2018 折训练窗（2015–2017）
    yrs = base["trade_date"].dt.year
    anchor_win = base.loc[yrs.between(K_ANCHOR_YEAR - 3, K_ANCHOR_YEAR - 1)]
    _, _, _, k_gpc = _pca_fit(anchor_win, feats, var=0.95, k=None)
    k_gpc = max(int(k_gpc), 1)

    names = fi_column_names(k_gpc)
    names["C10"] = [f"r21_c10_int_{a}__{b}" for a, b in pairs]
    all_new = [c for v in names.values() for c in v]

    out = base[KEYS].copy()
    for c in all_new:
        out[c] = np.nan
    model_choices: dict[str, str] = {}
    years = sorted(yrs.unique())
    for y in years:
        win = base.loc[yrs.between(int(y) - 3, int(y) - 1)]
        tgt_mask = yrs.eq(int(y))
        tgt = base.loc[tgt_mask]
        if len(win) < 100 or tgt.empty:
            continue
        idx = out.index[tgt_mask]
        # C5 global PCA（k 冻结）
        mu, sd, vt, _ = _pca_fit(win, feats, var=None, k=k_gpc)
        out.loc[idx, names["C5"]] = _pca_apply(tgt, feats, mu, sd, vt, k_gpc, names["C5"]).to_numpy()
        # C6 family PC1
        for fam_name, members in multi.items():
            col = f"r21_c6_fpc_{fam_name}_pc1"
            use = [c for c in members if c in feats]
            mu, sd, vt, k = _pca_fit(win, use, var=None, k=1)
            # 符号锚定：与家族等权 z 均值非负相关
            if vt is not None and k:
                zw = _z_apply(win, use, mu, sd).fillna(0.0).to_numpy(dtype=float)
                if float(np.corrcoef(zw @ vt[0], zw.mean(axis=1))[0, 1]) < 0:
                    vt = -vt
            out.loc[idx, col] = _pca_apply(tgt, use, mu, sd, vt, 1, [col]).to_numpy().ravel()
        # C7 ridge residual of g0 vs others
        others = [c for c in feats if c != "g0_amount_log"]
        xw = _num(win, others).fillna(0.0).to_numpy(dtype=float)
        yw = pd.to_numeric(win["g0_amount_log"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(yw)
        if m.sum() > 20:
            beta = _ridge_beta(xw[m], yw[m], 1.0)
            xt = _num(tgt, others).fillna(0.0).to_numpy(dtype=float)
            yt = pd.to_numeric(tgt["g0_amount_log"], errors="coerce").to_numpy(dtype=float)
            out.loc[idx, names["C7"][0]] = yt - xt @ beta
        # C8 shrunk IC composite（折内）
        w = _shrunk_ic_weights(win, feats, label, 0.5)
        combo = sum(pd.to_numeric(tgt[f], errors="coerce").fillna(0.0) * w[f] for f in feats)
        out.loc[idx, names["C8"][0]] = combo.to_numpy()
        # C9 inner-WF 选 ridge/elasticnet/pls：内层 fit [y-3,y-2] 验 [y-1]
        inner_fit = base.loc[yrs.between(int(y) - 3, int(y) - 2)]
        inner_val = base.loc[yrs.eq(int(y) - 1)]
        best_model, best_ic = "ridge", -np.inf
        if len(inner_fit) >= 100 and len(inner_val) >= 50:
            vy = pd.to_numeric(inner_val[label], errors="coerce")
            for model in ("ridge", "elasticnet", "pls"):
                pv = _fit_predict_linear(model, inner_fit, inner_val, feats, label)
                mm = pv.notna() & vy.notna()
                ic = float(pv.loc[mm].corr(vy.loc[mm], method="spearman")) if int(mm.sum()) >= 30 else -np.inf
                if np.isfinite(ic) and ic > best_ic:
                    best_model, best_ic = model, ic
        model_choices[str(int(y))] = best_model
        out.loc[idx, names["C9"][0]] = _fit_predict_linear(best_model, win, tgt, feats, label).to_numpy()
        # C10 预注册交互（窗口 z）
        for (a, b), col in zip(pairs, names["C10"]):
            mu, sd = _z_fit(win, [a, b])
            zt = _z_apply(tgt, [a, b], mu, sd)
            out.loc[idx, col] = (zt[a] * zt[b]).to_numpy()
        # C11 家族复合 mean_z（窗口 z）
        for fam_name, members in fam.items():
            use = [c for c in members if c in feats]
            mu, sd = _z_fit(win, use)
            zt = _z_apply(tgt, use, mu, sd)
            out.loc[idx, f"r21_c11_fam_{fam_name}"] = zt.mean(axis=1).to_numpy()

    # 备份 + 写面板 + 移除泄漏列
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    coverage: dict[str, Any] = {}
    for key in PANEL_KEYS:
        p = DATA / f"{key}.parquet"
        if not p.is_file():
            continue
        bak = BACKUP_DIR / p.name
        if not bak.is_file():
            shutil.copy2(p, bak)
        frame = pd.read_parquet(p)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        if LEAKED_C8_COL in frame.columns:
            frame = frame.drop(columns=[LEAKED_C8_COL])
        add = out[KEYS + all_new].drop_duplicates(KEYS)
        before = len(frame)
        merged = frame.merge(add, on=KEYS, how="left")
        cov = float(merged[all_new[0]].notna().mean()) if all_new else 0.0
        n_match = int(merged[names["C8"][0]].notna().sum())
        coverage[key] = {"rows": before, "c8_fi_nonnull": n_match, "first_col_nonnull_share": cov}
        merged.to_parquet(p, index=False)
    # 删除旧 C8 候选（泄漏训练产物），由重训重建
    old_c8 = exp_dir("R21_4_C8")
    if old_c8.is_dir():
        shutil.rmtree(old_c8)
    payload = {
        "status": "DONE",
        "k_gpc_frozen": k_gpc,
        "k_anchor_year": K_ANCHOR_YEAR,
        "columns": names,
        "all_new": all_new,
        "leaked_col_dropped": LEAKED_C8_COL,
        "backup": str(BACKUP_DIR),
        "coverage": coverage,
        "c9_model_by_year": model_choices,
        "interaction_pairs": pairs,
        "filled_with_zero": False,
        "window": "year y rows fitted on years [y-3, y-1]; <100 rows -> NaN",
    }
    dump_json(MAT_JSON, payload)
    return payload


# ---------------------------------------------------------------------------
# 规格 + 训练 + 全袖套
# ---------------------------------------------------------------------------

def combo_specs() -> list[dict[str, Any]]:
    mat = load_json(MAT_JSON)
    names = mat["columns"]
    fam = _family_map()
    multi_members = [m for k, v in fam.items() if len(v) >= 2 for m in v]

    def spec(code: str, extras: Sequence[str], drops: Sequence[str], role: str = "COMBO") -> dict[str, Any]:
        ten_drops = tuple(d for d in drops if d in TEN_FEATURES)
        return {
            "code": f"R21_4_{code}",
            "role": role,
            "factor_id": "combo_materialized_fi",
            "five_extras": tuple(extras),
            "ten_extras": tuple(extras),
            "five_drops": tuple(drops),
            "ten_drops": ten_drops,
            "formal": True,
            "head": "both_90_10",
            "alias_from": None,
            "force_retrain": True,
        }

    return [
        spec("C5", names["C5"], FIVE_FEATURES),                     # PCA 替代原特征
        spec("C6", names["C6"], multi_members),                     # 家族 PC1 替代多成员家族
        spec("C7", names["C7"], ()),                                # keep_old=True
        spec("C8", names["C8"], (), role="COMBO_C8"),               # 折内重训替代泄漏版
        spec("C9", names["C9"], ()),
        spec("C10", names["C10"], ()),
        spec("C11", names["C11"], FIVE_FEATURES),                   # 家族复合替代原特征
    ]


def train_and_backtest(code: str, *, n_jobs: int = 4) -> dict[str, Any]:
    from .backtest import run_full_backtest

    spec = next(s for s in combo_specs() if s["code"] == f"R21_4_{code}" or s["code"] == code)
    print(f"[materialize] TRAIN {spec['code']} extras={spec['five_extras']} drops={len(spec['five_drops'])}", flush=True)
    train_one(spec, resume=True, n_jobs=n_jobs)
    rec = run_full_backtest(spec["code"], spec, resume=True)
    rec["combo"] = code
    rec["method"] = next((c["method"] for c in COMBO_MATRIX if c["code"] == code), code)
    return rec


# ---------------------------------------------------------------------------
# 汇总：更新 combo_run.json + MD13–17/19
# ---------------------------------------------------------------------------

def finalize() -> dict[str, Any]:
    from .combo import RUN_JSON

    payload = load_json(RUN_JSON)
    rows = payload.get("rows") or []
    mat = load_json(MAT_JSON)
    superseded = list(payload.get("superseded") or [])
    kept = []
    for r in rows:
        if r.get("combo") == "C8" and finite_or_none(r.get("sharpe")) is not None:
            old = dict(r)
            old["superseded_reason"] = "full-sample shrunk_ic fit leakage; refit fold-internal (r21_c8_shrunk_ic_fi)"
            superseded.append(old)
            continue
        if r.get("status") == "OOF_DIAGNOSTIC":
            continue  # 被真实行替代
        kept.append(r)
    new_rows = []
    for spec in combo_specs():
        code = spec["code"]
        summ = load_json(exp_dir(code) / "summary.json")
        if finite_or_none(summ.get("sharpe")) is None:
            reason = "no summary"
            if code == "R21_4_C10":
                reason = (
                    "BLOCKED_DATA_SPARSE_LEGS: 6 个预注册交互同时非空占比每年=0%"
                    "（v1 公告腿全历史≤2.5%，t1 早期≤2.8%）；正典 complete-cases 政策下训练折为空；"
                    "缺失不填 0，禁换对（事后选择）。证据见 materialize.json interaction_pairs 与覆盖率审计"
                )
            new_rows.append({"code": code, "combo": code.replace("R21_4_", ""), "status": "BLOCKED", "reason": reason, "method": next((c["method"] for c in COMBO_MATRIX if c["code"] == code.replace("R21_4_", "")), code)})
            continue
        oof = load_json(exp_dir(code) / "oof.json")
        rec = dict(summ)
        rec.pop("resumed", None)
        rec["combo"] = code.replace("R21_4_", "")
        rec["method"] = next((c["method"] for c in COMBO_MATRIX if c["code"] == rec["combo"]), code)
        rec["five_oof"] = oof.get("five_oof")
        rec["ten_oof"] = oof.get("ten_oof")
        rec["five_extras"] = oof.get("five_extras")
        rec["fold_internal"] = True
        new_rows.append(rec)
    rows = kept + new_rows
    a0 = next((r for r in rows if r.get("combo") == "C0"), {})
    a0_sh = finite_or_none(a0.get("sharpe"))
    full = [r for r in rows if finite_or_none(r.get("sharpe")) is not None]
    best = max(full, key=lambda r: float(r.get("sharpe")), default={})
    n_better = sum(1 for r in full if a0_sh is not None and float(r.get("sharpe")) > a0_sh and r.get("combo") != "C0")
    verdict = RAW_BASELINE if n_better == 0 else "COMBO_BETTER_THAN_RAW"
    payload.update(
        {
            "status": "DONE",
            "n": len(rows),
            "n_full": len(full),
            "rows": rows,
            "superseded": superseded,
            "verdict": verdict,
            "portfolio": {"baseline": "R21_4_A0", "best": best.get("code"), "n_better_than_raw": n_better, "verdict": verdict},
            "materialize": {"k_gpc_frozen": mat.get("k_gpc_frozen"), "columns": mat.get("columns"), "c9_model_by_year": mat.get("c9_model_by_year")},
            "correction": "C8 初版全样本拟合存在前视，已按折内口径重训；C5–C11 折内物化后真实训练+全袖套。",
        }
    )
    dump_json(RUN_JSON, payload)
    _write_md(payload, mat)
    return payload


def _fmt(r: dict[str, Any]) -> str:
    sh = finite_or_none(r.get("sharpe"))
    cg = finite_or_none(r.get("cagr"))
    nav = finite_or_none(r.get("nav_end"))
    return (
        f"| {r.get('combo')} | {r.get('method')} | `{r.get('code')}` | "
        f"{'' if sh is None else f'{sh:.4f}'} | {'' if cg is None else f'{cg:.4f}'} | "
        f"{'' if nav is None else f'{nav:,.2f}'} | {r.get('completed_trades')} |"
    )


def _write_md(payload: dict[str, Any], mat: dict[str, Any]) -> None:
    rows = payload.get("rows") or []
    by = {r.get("combo"): r for r in rows}
    table = "\n".join(_fmt(r) for r in rows if finite_or_none(r.get("sharpe")) is not None)
    head = "| C | 方法 | 代号 | Sharpe | CAGR | NAV | 笔数 |\n| --- | --- | --- | ---: | ---: | ---: | ---: |"
    write_md(MD["19"], f"# Round21.4 · 19 C0–C11 结果\n\n{head}\n{table}\n\n"
                       f"verdict=`{payload.get('verdict')}` n_full=`{payload.get('n_full')}`。\n\n"
                       f"更正：C8 初版全样本 shrunk_ic 拟合存在前视 → 折内重训（superseded 保留旧行）。"
                       f"C5–C11 生成列折内物化（`_fi`）后真实训练+全袖套；窗口=年 y 用 [y-3, y-1]，历史不足→NaN 不填 0。\n")
    write_md(MD["13"], f"# Round21.4 · 13 降维 C5–C6\n\nC5 global_pca：k=`{mat.get('k_gpc_frozen')}`（2018 折训练窗 var=0.95 选定后冻结）。"
                       f"C6 family_pca：多成员家族 PC1 替代，单例保留。折内 fit/apply 分离，符号锚定。\n"
                       f"C5 Sharpe=`{(by.get('C5') or {}).get('sharpe')}` C6 Sharpe=`{(by.get('C6') or {}).get('sharpe')}`\n")
    write_md(MD["14"], f"# Round21.4 · 14 正交化 C7\n\ng0 对其余 13 特征 ridge(α=1) 残差，折内估计 β；keep_old=True 作为 extras。\n"
                       f"C7 Sharpe=`{(by.get('C7') or {}).get('sharpe')}`\n")
    write_md(MD["15"], f"# Round21.4 · 15 监督 C8–C9\n\nC8 shrunk_ic(shrink=0.5) 折内权重重训（修正全样本前视）。"
                       f"C9 inner-WF 选 ridge/elasticnet/pls：内层 fit [y-3,y-2] 验 [y-1]，逐折选择=`{mat.get('c9_model_by_year')}`。\n"
                       f"C8 Sharpe=`{(by.get('C8') or {}).get('sharpe')}` C9 Sharpe=`{(by.get('C9') or {}).get('sharpe')}`\n")
    write_md(MD["16"], f"# Round21.4 · 16 非线性 C10\n\n预注册交互（禁无限乘积）对=`{mat.get('interaction_pairs')}`，窗口 z 后乘积。\n"
                       f"C10 BLOCKED_DATA_SPARSE_LEGS：6 交互同时非空占比每年=0%（v1 公告腿≤2.5%）；"
                       f"正典 complete-cases 下训练折为空；缺失不填 0，禁事后换对。列已物化备查，未开袖套。\n")
    write_md(MD["17"], f"# Round21.4 · 17 分层 C11\n\n经济家族 mean_z 复合替代 14 原特征；不改 O01。\n"
                       f"C11 Sharpe=`{(by.get('C11') or {}).get('sharpe')}`\n")


def main(argv: list[str]) -> Any:
    cmd = argv[1] if len(argv) > 1 else "materialize"
    if cmd == "materialize":
        return materialize(resume="--force" not in argv)
    if cmd == "train":
        return train_and_backtest(argv[2], n_jobs=int(argv[3]) if len(argv) > 3 else 4)
    if cmd == "finalize":
        return finalize()
    raise SystemExit(f"unknown cmd {cmd}")


if __name__ == "__main__":
    main(sys.argv)
