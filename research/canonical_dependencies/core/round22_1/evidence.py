"""Track B：11 个 PIT_PASS 因子 Rolling / Conditional / Shape / Top-tail。禁止 SKIP_IF_MISSING。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND22_1_ROOT, ROUND22_ROOT
from ..round7.contract import FIVE_FEATURES, TEN_FEATURES
from ..round20.metrics import icir_of, rank_ic
from ..round20.panel import fold_masks
from ..round21.conditional import _daily_rankic, _fit_residual
from ..round21.contract import INTERNAL_ROLLING_WINDOWS, INTERNAL_WINDOW_NAMES, RIDGE_ALPHA
from ..round21.evidence import _rolling, _shape_one
from ..round21.panel import load_candidate_keys
from .common import dump_json, load_json, write_md
from .contract import (
    COMPLETE,
    FAILED,
    FULL_EVIDENCE_COMPLETE,
    KEYS,
    LABELS,
    MD,
    NOT_IDENTIFIABLE,
    PIT_PASS,
    PIT_PASS_IDS,
    ROLLING_WINDOWS,
)
from .evaluate import evaluate_factor
from .freeze import require_frozen_gate
from .ic_audit import run_ic_audit

EVIDENCE_JSON = ROUND22_1_ROOT / "evidence" / "evidence.json"


def _panel() -> pd.DataFrame:
    import pyarrow.parquet as pq

    from ..round7.contract import FIVE_HEAD

    panel = pd.read_parquet(ROUND22_ROOT / "data" / "factor_panel.parquet")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    labels = load_candidate_keys()
    extra = [c for c in [*LABELS, "next2_date", "actual_exit_date"] if c in labels.columns]
    keep = list(dict.fromkeys([*KEYS, *extra]))
    base = panel.merge(labels[keep], on=KEYS, how="inner")
    cand_path = FIVE_HEAD.panel_dir / "candidate_panel.parquet"
    if cand_path.is_file():
        names = set(pq.read_schema(cand_path).names)
        feat_cols = [c for c in (*FIVE_FEATURES, *TEN_FEATURES, "fb_close_seal_streak_norm") if c in names and c not in base.columns]
        if feat_cols:
            feats = pd.read_parquet(cand_path, columns=[*KEYS, *feat_cols])
            feats["trade_date"] = pd.to_datetime(feats["trade_date"])
            feats["instrument_id"] = feats["instrument_id"].astype(str)
            base = base.merge(feats.drop_duplicates(KEYS), on=KEYS, how="left")
    if base.columns.duplicated().any():
        base = base.loc[:, ~base.columns.duplicated()].copy()
    return base


def _groups(panel: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict[int, np.ndarray]]:
    day_ids, unique_dates = pd.factorize(panel["trade_date"], sort=True)
    unique_dates = pd.DatetimeIndex(unique_dates)
    order = np.argsort(day_ids, kind="mergesort")
    sorted_ids = day_ids[order]
    bounds = np.append(np.flatnonzero(np.diff(sorted_ids, prepend=-1)), len(order))
    groups = {int(sorted_ids[a]): order[a:b] for a, b in zip(bounds[:-1], bounds[1:])}
    return unique_dates, groups


def _neighbor_control(panel: pd.DataFrame, fid: str, locked: tuple[str, ...]) -> str | None:
    fx = pd.to_numeric(panel[fid], errors="coerce")
    best = None
    best_abs = -1.0
    for col in locked:
        if col not in panel.columns or col == fid:
            continue
        other = panel[col]
        if isinstance(other, pd.DataFrame):
            other = other.iloc[:, 0]
        rho = fx.corr(pd.to_numeric(other, errors="coerce"), method="spearman")
        if rho is None or not np.isfinite(rho):
            continue
        if abs(float(rho)) > best_abs:
            best_abs = abs(float(rho))
            best = col
    return best


def _conditional_one(panel: pd.DataFrame, fid: str) -> dict[str, Any]:
    years = sorted(int(y) for y in panel["trade_date"].dt.year.dropna().unique())
    heads = (
        ("label_e2_net", tuple(FIVE_FEATURES), "five_14"),
        ("label_exit_rule_net", tuple(TEN_FEATURES), "ten_13"),
    )
    rows: list[dict[str, Any]] = []
    for lab, locked, tag in heads:
        lock = [c for c in locked if c in panel.columns and c != fid]
        if lab not in panel.columns:
            rows.append({"factor": fid, "label": lab, "status": NOT_IDENTIFIABLE, "reason": "label missing", "full_history_fit": False})
            continue
        neighbor = _neighbor_control(panel, fid, locked)
        for controls, kind in ((lock, tag), ([neighbor] if neighbor else [], "nearest_neighbor")):
            if not controls:
                rows.append({"factor": fid, "label": lab, "kind": kind, "status": NOT_IDENTIFIABLE, "reason": "no controls", "full_history_fit": False})
                continue
            oos: list[float] = []
            n_folds = 0
            skipped = 0
            for y in years:
                train_m, pred_m = fold_masks(panel, predict_year=y, label=lab)
                cols = [*controls, fid, lab]
                use = panel.loc[:, [c for c in cols if c in panel.columns] + (["trade_date"] if "trade_date" not in cols else [])]
                tr = use.loc[train_m].replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in cols if c in use.columns])
                te = use.loc[pred_m].replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in cols if c in use.columns])
                if len(tr) < 80 or len(te) < 20:
                    skipped += 1
                    continue
                resid = _fit_residual(
                    tr[controls].to_numpy(dtype=float),
                    pd.to_numeric(tr[fid], errors="coerce").to_numpy(dtype=float),
                    te[controls].to_numpy(dtype=float),
                    pd.to_numeric(te[fid], errors="coerce").to_numpy(dtype=float),
                )
                if resid is None:
                    skipped += 1
                    continue
                n_folds += 1
                oos.extend(_daily_rankic(te["trade_date"], resid, pd.to_numeric(te[lab], errors="coerce").to_numpy(dtype=float)))
            arr = np.array(oos, dtype=float)
            status = "OK" if len(arr) >= 60 else NOT_IDENTIFIABLE
            rows.append(
                {
                    "factor": fid,
                    "label": lab,
                    "kind": kind,
                    "status": status,
                    "n_oos_days": int(len(arr)),
                    "n_folds": n_folds,
                    "n_folds_skipped": skipped,
                    "cond_rankic_mean": float(np.mean(arr)) if len(arr) else None,
                    "cond_rankicir": icir_of(arr) if len(arr) else None,
                    "full_history_fit": False,
                    "ridge_alpha": RIDGE_ALPHA,
                    "controls": ",".join(str(c) for c in controls if c),
                    "n_controls": int(len(controls)),
                }
            )
        if fid in {"r22_d1_minute_high_touch_rank", "r22_d2_minute_high_lead_min", "r22_d3_lead_resid_past_only"} and "fb_close_seal_streak_norm" in panel.columns:
            seal = ["fb_close_seal_streak_norm"]
            oos = []
            n_folds = 0
            skipped = 0
            for y in years:
                train_m, pred_m = fold_masks(panel, predict_year=y, label=lab)
                cols = [*seal, fid, lab]
                tr = panel.loc[train_m, [c for c in cols if c in panel.columns] + ["trade_date"]].replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in cols if c in panel.columns])
                te = panel.loc[pred_m, [c for c in cols if c in panel.columns] + ["trade_date"]].replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in cols if c in panel.columns])
                if len(tr) < 80 or len(te) < 20:
                    skipped += 1
                    continue
                resid = _fit_residual(
                    tr[seal].to_numpy(dtype=float),
                    pd.to_numeric(tr[fid], errors="coerce").to_numpy(dtype=float),
                    te[seal].to_numpy(dtype=float),
                    pd.to_numeric(te[fid], errors="coerce").to_numpy(dtype=float),
                )
                if resid is None:
                    skipped += 1
                    continue
                n_folds += 1
                oos.extend(_daily_rankic(te["trade_date"], resid, pd.to_numeric(te[lab], errors="coerce").to_numpy(dtype=float)))
            arr = np.array(oos, dtype=float)
            rows.append(
                {
                    "factor": fid,
                    "label": lab,
                    "kind": "vs_fb_close_seal_streak_norm",
                    "status": "OK" if len(arr) >= 60 else NOT_IDENTIFIABLE,
                    "n_oos_days": int(len(arr)),
                    "n_folds": n_folds,
                    "n_folds_skipped": skipped,
                    "cond_rankic_mean": float(np.mean(arr)) if len(arr) else None,
                    "cond_rankicir": icir_of(arr) if len(arr) else None,
                    "full_history_fit": False,
                    "ridge_alpha": RIDGE_ALPHA,
                    "controls": "fb_close_seal_streak_norm",
                    "n_controls": 1,
                }
            )
    return {"factor_id": fid, "full_history_fit": False, "ridge_alpha": RIDGE_ALPHA, "rows": rows, "status": COMPLETE}


def _copy_pit(fid: str) -> dict[str, Any]:
    dest = ROUND22_1_ROOT / "factors" / fid
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "pit.json").is_file():
        return load_json(dest / "pit.json")
    src = load_json(ROUND22_ROOT / "factors" / fid / "pit.json")
    if not src:
        pit = load_json(ROUND22_ROOT / "pit" / "pit.json")
        src = next((r for r in (pit.get("rows") or []) if r.get("factor_id") == fid), {"factor_id": fid, "pit_status": PIT_PASS})
    dump_json(dest / "pit.json", src)
    red = ROUND22_ROOT / "factors" / fid / "redundancy.json"
    if red.is_file() and not (dest / "redundancy.json").is_file():
        dump_json(dest / "redundancy.json", load_json(red))
    return src


def run_evidence(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and EVIDENCE_JSON.is_file() and MD["02"].is_file():
        existing = load_json(EVIDENCE_JSON)
        if existing.get("status") in {"DONE", COMPLETE, FULL_EVIDENCE_COMPLETE}:
            return existing
    run_ic_audit(resume=True)
    panel = _panel()
    c_panel_path = ROUND22_1_ROOT / "data_recovery" / "c_factor_panel.parquet"
    extra_ids: tuple[str, ...] = ()
    if c_panel_path.is_file():
        cpan = pd.read_parquet(c_panel_path)
        cpan["trade_date"] = pd.to_datetime(cpan["trade_date"])
        cpan["instrument_id"] = cpan["instrument_id"].astype(str)
        add_cols = [c for c in ("r22_c1_xl_net_imbalance", "r22_c4_flow_persist_5") if c in cpan.columns]
        if add_cols:
            panel = panel.merge(cpan[KEYS + add_cols].drop_duplicates(KEYS), on=KEYS, how="left")
            extra_ids = tuple(add_cols)
    if panel.columns.duplicated().any():
        panel = panel.loc[:, ~panel.columns.duplicated()].copy()
    unique_dates, groups = _groups(panel)
    packages: list[dict[str, Any]] = []
    missing_any = False
    for fid in (*PIT_PASS_IDS, *extra_ids):
        dest = ROUND22_1_ROOT / "factors" / fid
        dest.mkdir(parents=True, exist_ok=True)
        pit = _copy_pit(fid)
        if fid not in panel.columns:
            pkg = {"factor_id": fid, "status": FAILED, "reason": "factor column missing"}
            dump_json(dest / "package.json", pkg)
            packages.append(pkg)
            missing_any = True
            continue
        fx = pd.to_numeric(panel[fid], errors="coerce").to_numpy(dtype=float)
        roll_block: dict[str, Any] = {}
        shape_block: dict[str, Any] = {}
        tail_block: dict[str, Any] = {}
        for lab in LABELS:
            if lab not in panel.columns:
                continue
            y = pd.to_numeric(panel[lab], errors="coerce").to_numpy(dtype=float)
            daily_path = dest / f"daily_ic_{lab}.parquet"
            ic_src = load_json(dest / f"ic_{lab}.json")
            # 用 Track A 日表重建 rolling 输入
            audit = ROUND22_1_ROOT / "ic" / "daily_ic_input.parquet"
            if audit.is_file():
                d = pd.read_parquet(audit)
                d = d.loc[d["factor_id"].eq(fid) & d["label"].eq(lab) & d["eligibility"].eq("ELIGIBLE")].copy()
                daily = pd.DataFrame(
                    {
                        "trade_date": pd.to_datetime(d["trade_date"]),
                        "n_valid": pd.to_numeric(d["n_pairs"], errors="coerce"),
                        "rankic": pd.to_numeric(d["rho"], errors="coerce"),
                        "ic": pd.to_numeric(d["rho"], errors="coerce"),
                        "ready_at": pd.to_datetime(d["trade_date"]),
                    }
                )
            else:
                daily = pd.DataFrame(columns=["trade_date", "n_valid", "rankic", "ic", "ready_at"])
            if len(daily):
                daily.to_parquet(daily_path, index=False)
            roll = _rolling(daily, "rankic", unique_dates)
            # 有效交易日窗 + 分年 2018–2026YTD
            yearly = {}
            if len(daily):
                daily = daily.copy()
                daily["year"] = pd.to_datetime(daily["trade_date"]).dt.year
                for yv, block in daily.groupby("year"):
                    if int(yv) < 2018:
                        continue
                    arr = pd.to_numeric(block["rankic"], errors="coerce").dropna().to_numpy(dtype=float)
                    yearly[str(int(yv))] = {
                        "n": int(len(arr)),
                        "mean": float(arr.mean()) if len(arr) else None,
                        "status": "OK" if len(arr) >= 20 else NOT_IDENTIFIABLE,
                    }
            roll["windows_trading_days"] = list(ROLLING_WINDOWS)
            roll["by_year_2018_2026"] = yearly
            roll["status"] = COMPLETE if any((roll.get("windows") or {}).get(n, {}).get("n_windows") for n in INTERNAL_WINDOW_NAMES) else NOT_IDENTIFIABLE
            roll_block[lab] = roll
            shape = _shape_one(groups, fx, y)
            shape["label"] = lab
            shape["forced_quintile"] = False
            shape["top_tail_definition"] = "因子值截面排名上/下 10% 的标签均值差；不足 10 名不强制十分位"
            shape["rank_vs_value"] = "rank_of_factor_value"
            shape_block[lab] = shape
            tail_block[lab] = {
                "top_tail_mean_spread": shape.get("top_tail_mean_spread"),
                "top2_minus_rest": shape.get("top2_minus_rest"),
                "q5_minus_q1": shape.get("q5_minus_q1"),
                "definition": "因子值截面排名分位，不是成交账本 Top1/Top2",
                "status": COMPLETE if shape.get("n_days_q5") else NOT_IDENTIFIABLE,
            }
        dump_json(dest / "rolling.json", roll_block)
        dump_json(dest / "shape.json", shape_block)
        dump_json(dest / "top_tail.json", tail_block)
        cond = _conditional_one(panel, fid)
        dump_json(dest / "conditional.json", cond)
        needed = [dest / n for n in ("rolling.json", "conditional.json", "shape.json", "top_tail.json", "ic.json")]
        have = [p for p in needed if p.is_file() and p.stat().st_size > 8]
        status = FULL_EVIDENCE_COMPLETE if len(have) == 5 else FAILED
        if len(have) < 5:
            missing_any = True
        pkg = {
            "factor_id": fid,
            "status": status,
            "pit_status": pit.get("pit_status"),
            "copied_global_result": False,
            "n_evidence_files": len(have),
            "ic_status": (load_json(dest / "ic.json") or {}).get("status"),
        }
        dump_json(dest / "package.json", pkg)
        factory = evaluate_factor(fid, through="PromotionGate", persist=True)
        pkg["factory_status"] = factory.get("status")
        packages.append(pkg)

    n_full = int(sum(p.get("status") == FULL_EVIDENCE_COMPLETE and p.get("factor_id") in PIT_PASS_IDS for p in packages))
    n_extra_full = int(sum(p.get("status") == FULL_EVIDENCE_COMPLETE and p.get("factor_id") not in PIT_PASS_IDS for p in packages))
    payload = {
        "status": FULL_EVIDENCE_COMPLETE if n_full == len(PIT_PASS_IDS) else ("PARTIAL" if n_full else FAILED),
        "n": len(packages),
        "n_full": n_full,
        "n_extra_full": n_extra_full,
        "skip_if_missing_removed": True,
        "mandatory": ["Rolling", "Shape", "TopTail", "ConditionalIC"],
        "rolling_windows": list(ROLLING_WINDOWS),
        "internal_window_names": list(INTERNAL_WINDOW_NAMES),
        "conditional": "fold-in Ridge residual IC; Five 14 / Ten 13 / nearest neighbor; D1-D3 vs fb_close_seal_streak_norm",
        "full_history_fit": False,
        "packages": packages,
    }
    dump_json(EVIDENCE_JSON, payload)
    lines = [
        "# Round22.1 · 02 Rolling / Conditional / Shape",
        "",
        f"11 个 PIT_PASS 因子。完整证据包 `{n_full}/{len(PIT_PASS_IDS)}`。",
        "Rolling 60/120/252 为有效交易日窗，保留 daily_ic + n_pairs + validity。分年 2018–2026YTD。",
        "Conditional：训练折 fit Ridge，测试折残差 IC。Five 控 14 列，Ten 控 13 列，另做最近邻单控。D1/D2/D3 相对 `fb_close_seal_streak_norm` 必做。禁止全历史一次拟合。",
        "Shape/Top-tail：按当日可识别分组，不足不强行五分位。Five/Ten 标签分开。Top-tail 是因子值截面排名分位，不是成交账本 Top1/Top2。",
        "mandatory 阶段已去掉 SKIP_IF_MISSING。缺任一 JSON 不得标 FULL_EVIDENCE_COMPLETE。",
        "",
    ]
    write_md(MD["02"], "\n".join(lines))
    write_md(
        MD["14"],
        "# Round22.1 · 14 PIT_PASS 证据矩阵\n\n"
        + "\n".join(f"- `{p.get('factor_id')}` status=`{p.get('status')}` files=`{p.get('n_evidence_files')}`" for p in packages)
        + "\n",
    )
    return payload
