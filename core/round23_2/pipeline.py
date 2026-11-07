"""Round 23.2 主流程。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.r225241_sleeve import apply_b03_o01
from core.research_integrity_v2.io_v2 import dump_json, load_json, load_stage, save_stage, write_df, write_parquet, write_text
from core.research_integrity_v2.merge_rank import explicit_merge, rank_unique
from core.research_integrity_v2.stats_v2 import paired_block_bootstrap_v2, spa_hansen_or_blocked
from core.research_integrity_v2.time_contract import LEGACY_OPEN_INFORMATION_LANE, shanghai_now
from core.round11.common import daily_from_nav
from core.round22_3.contract import FIVE_FEATURES, TEN_FEATURES
from core.round22_6.sleeve import run_formal_sleeve
from core.round23_2.contract import (
    DEFAULT_W_LGBM,
    DEFAULT_W_TEN,
    E0,
    GAP7,
    HYPOTHESES,
    INNER_BLOCK,
    INNER_BOOT_N,
    LGBM_WEIGHTS,
    MIN_INNER_DAYS,
    MIN_INNER_TRADES,
    PANEL,
    ROOT,
    R23_1,
    TEN_PANEL,
    TEN_TRAIN,
    TEN_WEIGHTS,
    TRAIN,
    anchor_copy,
)
from core.round23_2.lgbm_train import train_lgbm_ranker_trunc
from core.round7.contract import FIVE_HEAD, TEN_HEAD


def _skip(name: str, resume: bool) -> dict[str, Any] | None:
    return load_stage(ROOT, name) if resume else None


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (ROOT / "frozen_manifest.json").is_file():
        return load_json(ROOT / "frozen_manifest.json")
    body = {
        "round": "23.2",
        "frozen_before_new_nav": True,
        "freeze_timestamp": shanghai_now().isoformat(),
        "anchor": anchor_copy(),
        "lgbm_weights": list(LGBM_WEIGHTS),
        "ten_weights": list(TEN_WEIGHTS),
        "default_w_lgbm": DEFAULT_W_LGBM,
        "default_w_ten": DEFAULT_W_TEN,
        "no_joint_m2_m3": True,
        "diversity_is_diagnostic_only": True,
        "gap7_renamed": "GAP7_ALLOCATION_HALF_DIAGNOSTIC",
        "information_lane": LEGACY_OPEN_INFORMATION_LANE,
        "hypotheses": list(HYPOTHESES),
        "current_baseline_changed": False,
        "auto_promote": False,
    }
    dump_json(ROOT / "frozen_manifest.json", body)
    return save_stage(ROOT, "freeze", {**body, "status": "FROZEN"})


def _v5_fused() -> pd.DataFrame:
    from core.round17.backtest import fuse_heads

    five = pd.read_parquet(E0 / "five_pred.parquet")
    ten = pd.read_parquet(E0 / "ten_pred.parquet")
    return fuse_heads(five, ten)


def _blend_ten(five: pd.DataFrame, ten: pd.DataFrame, w_ten: float) -> pd.DataFrame:
    a = five.rename(columns={"score": "five"})
    b = ten.rename(columns={"score": "ten"})
    for frame in (a, b):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    work = explicit_merge(a, b, on=["signal_date", "instrument_id"], how="inner", validate="one_to_one")
    # gap eligibility via fuse path for default; for non-default still apply same gap filter
    from core.round17.backtest import load_gaps
    from core.round7.contract import GAP_FLOOR

    work = work.merge(load_gaps(), on=["signal_date", "instrument_id"], how="left")
    work["d1_open_gap"] = pd.to_numeric(work["d1_open_gap"], errors="coerce")
    work = work.loc[np.isfinite(work["d1_open_gap"]) & work["d1_open_gap"].ge(GAP_FLOOR)].copy()
    work["pct_five"] = work.groupby("signal_date")["five"].rank(method="average", pct=True)
    work["pct_ten"] = work.groupby("signal_date")["ten"].rank(method="average", pct=True)
    work["score"] = float(w_ten) * work["pct_ten"] + (1.0 - float(w_ten)) * work["pct_five"]
    return work[["signal_date", "instrument_id", "score"]]


def blend_xgb_lgbm(xgb: pd.DataFrame, lgbm: pd.DataFrame, w_lgbm: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """w_lgbm=0 直接返回冻结 v5，不走变换流水线。"""
    if float(w_lgbm) == 0.0:
        return xgb.copy(), pd.DataFrame({"signal_date": [], "reason": [], "n": []})
    a = xgb.rename(columns={"score": "xgb"})
    b = lgbm.rename(columns={"score": "lgbm"})
    for frame in (a, b):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    # left join from xgb universe; missing lgbm => fallback that day to v5
    fallback_rows = []
    out_parts = []
    for day, block in a.groupby("signal_date", sort=True):
        day_l = b.loc[b["signal_date"].eq(day)]
        merged = block.merge(day_l, on=["signal_date", "instrument_id"], how="left")
        missing = int(merged["lgbm"].isna().sum())
        if missing:
            fallback_rows.append({"signal_date": str(pd.Timestamp(day).date()), "reason": "MISSING_LGBM_PRED", "n": missing})
            out_parts.append(block.rename(columns={"xgb": "score"})[["signal_date", "instrument_id", "score"]])
            continue
        merged["px"] = merged["xgb"].rank(method="average", pct=True)
        merged["pl"] = merged["lgbm"].rank(method="average", pct=True)
        merged["score"] = float(w_lgbm) * merged["pl"] + (1.0 - float(w_lgbm)) * merged["px"]
        out_parts.append(merged[["signal_date", "instrument_id", "score"]])
    out = pd.concat(out_parts, ignore_index=True) if out_parts else xgb.copy()
    return out, pd.DataFrame(fallback_rows)


def run_endpoint_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("endpoint", resume)
    if cached:
        return cached
    v5 = _v5_fused()
    five = pd.read_parquet(E0 / "five_pred.parquet")
    ten = pd.read_parquet(E0 / "ten_pred.parquet")
    ten90 = _blend_ten(five, ten, 0.90)
    # compare scores
    a = v5.sort_values(["signal_date", "instrument_id"]).reset_index(drop=True)
    b = ten90.sort_values(["signal_date", "instrument_id"]).reset_index(drop=True)
    same_keys = a[["signal_date", "instrument_id"]].equals(b[["signal_date", "instrument_id"]])
    score_max_diff = float(np.nanmax(np.abs(a["score"].to_numpy() - b["score"].to_numpy()))) if same_keys and len(a) == len(b) else None
    zero_w, fb = blend_xgb_lgbm(v5, v5, 0.0)
    identity_zero = zero_w.equals(v5) or (
        len(zero_w) == len(v5)
        and float(np.nanmax(np.abs(zero_w.sort_values(["signal_date", "instrument_id"])["score"].to_numpy() - v5.sort_values(["signal_date", "instrument_id"])["score"].to_numpy()))) <= 1e-15
    )
    # NAV parity via formal sleeve
    s_v5 = run_formal_sleeve(ROOT / "endpoint" / "V5", v5, {"code": "R232_EP_V5", "round": "23.2", "scope": "FULL_HISTORY_FIRST_TO_SECOND"}, resume=resume, counts_as_alpha=False, top_changed=False, fee_same_direction=True, not_only_year=True)
    s_90 = run_formal_sleeve(ROOT / "endpoint" / "TEN90", ten90, {"code": "R232_EP_TEN90", "round": "23.2", "scope": "FULL_HISTORY_FIRST_TO_SECOND"}, resume=resume, counts_as_alpha=False, top_changed=False, fee_same_direction=True, not_only_year=True)
    s_z = run_formal_sleeve(ROOT / "endpoint" / "WLGBM0", zero_w, {"code": "R232_EP_WLGBM0", "round": "23.2", "scope": "FULL_HISTORY_FIRST_TO_SECOND"}, resume=resume, counts_as_alpha=False, top_changed=False, fee_same_direction=True, not_only_year=True)
    nav_ok_90 = abs(float(s_v5["nav_end"]) - float(s_90["nav_end"])) <= max(0.01, 1e-8 * float(s_v5["nav_end"]))
    nav_ok_z = abs(float(s_v5["nav_end"]) - float(s_z["nav_end"])) <= max(0.01, 1e-8 * float(s_v5["nav_end"]))
    rows = [
        {"check": "ten_weight_0.90_score_keys", "pass": same_keys},
        {"check": "ten_weight_0.90_score_max_diff", "pass": score_max_diff is not None and score_max_diff <= 1e-12, "value": score_max_diff},
        {"check": "ten_weight_0.90_nav", "pass": nav_ok_90, "v5_nav": s_v5.get("nav_end"), "cand_nav": s_90.get("nav_end")},
        {"check": "w_lgbm_0_returns_frozen_path", "pass": identity_zero},
        {"check": "w_lgbm_0_nav", "pass": nav_ok_z, "v5_nav": s_v5.get("nav_end"), "cand_nav": s_z.get("nav_end")},
    ]
    write_df(ROOT / "endpoint_parity.csv", pd.DataFrame(rows))
    payload = {"status": "PASS" if all(r["pass"] for r in rows) else "FAIL", "rows": rows, "fallback_schema": list(fb.columns)}
    dump_json(ROOT / "endpoint_parity.json", payload)
    return save_stage(ROOT, "endpoint", payload)


def run_lgbm(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    cached = _skip("lgbm", resume)
    if cached:
        return cached
    # control: old R23.1 if present
    old_five = R23_1 / "lgbm" / "five" / "train_meta.json"
    control = load_json(old_five) if old_five.is_file() else {"status": "MISSING_ARTIFACT"}
    five_meta = train_lgbm_ranker_trunc(
        training=pd.read_parquet(TRAIN),
        candidates=pd.read_parquet(PANEL),
        features=list(FIVE_FEATURES),
        target=FIVE_HEAD.target,
        label_end_column=FIVE_HEAD.label_end_column,
        n_bins=FIVE_HEAD.rank_n_bins,
        eval_at=[1],
        truncation_level=4,
        output_dir=ROOT / "lgbm" / "five_trunc4",
        resume=resume,
    )
    ten_meta = train_lgbm_ranker_trunc(
        training=pd.read_parquet(TEN_TRAIN),
        candidates=pd.read_parquet(TEN_PANEL),
        features=list(TEN_FEATURES),
        target=TEN_HEAD.target,
        label_end_column=TEN_HEAD.label_end_column,
        n_bins=TEN_HEAD.rank_n_bins,
        eval_at=[2],
        truncation_level=5,
        output_dir=ROOT / "lgbm" / "ten_trunc5",
        resume=resume,
    )
    rows = [
        {"head": "five", "truncation_level": 4, "eval_at": [1], "rows": five_meta.get("rows"), "control_old_rows": control.get("rows")},
        {"head": "ten", "truncation_level": 5, "eval_at": [2], "rows": ten_meta.get("rows")},
    ]
    write_df(ROOT / "lgbm_training_objective_audit.csv", pd.DataFrame(rows))
    payload = {"status": "DONE", "five": five_meta, "ten": ten_meta, "eval_at_is_not_truncation": True}
    dump_json(ROOT / "lgbm.json", payload)
    return save_stage(ROOT, "lgbm", payload)


def _maxdd_from_nav(nav: pd.DataFrame) -> float:
    from core.research_sleeve import metrics_from_nav_file
    # nav already metrics in summary; compute from series
    s = daily_from_nav(nav)
    wealth = (1.0 + s).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min())


def _inner_utility(nav: pd.DataFrame) -> float:
    r = daily_from_nav(nav)
    return float(np.log1p(r).mean())


def _select_weight_by_inner(
    configs: list[tuple[str, pd.DataFrame]],
    *,
    year: int,
    default_key: str,
    resume: bool,
) -> dict[str, Any]:
    """两个半年块完整袖套重放。"""
    blocks = [
        (pd.Timestamp(year - 1, 1, 1), pd.Timestamp(year - 1, 6, 30), "H1"),
        (pd.Timestamp(year - 1, 7, 1), pd.Timestamp(year - 1, 12, 31), "H2"),
    ]
    records = []
    default_scores = dict(configs)[default_key]
    for key, scores in configs:
        utils = []
        dds = []
        n_days = 0
        n_trades = 0
        daily_parts = []
        for start, end, tag in blocks:
            sub = scores.loc[pd.to_datetime(scores["signal_date"]).between(start, end)]
            dest = ROOT / "inner" / f"{year}" / key / tag
            if sub.empty:
                utils.append(None)
                continue
            sleeve = run_formal_sleeve(
                dest,
                sub,
                {
                    "code": f"INNER_{year}_{key}_{tag}",
                    "round": "23.2",
                    "scope": "DIAGNOSTIC_ONLY",
                    "window_start": str(start.date()),
                    "window_end": str(end.date()),
                },
                resume=resume,
                counts_as_alpha=False,
                top_changed=None,
                fee_same_direction=True,
                not_only_year=True,
            )
            nav = pd.read_parquet(dest / "nav.parquet")
            utils.append(_inner_utility(nav))
            dds.append(_maxdd_from_nav(nav))
            n_days += int(nav.shape[0])
            n_trades += int(sleeve.get("completed_trades") or 0)
            daily_parts.append(daily_from_nav(nav).rename(key))
        default_dds = []
        for start, end, tag in blocks:
            dest = ROOT / "inner" / f"{year}" / default_key / tag
            if (dest / "nav.parquet").is_file():
                default_dds.append(_maxdd_from_nav(pd.read_parquet(dest / "nav.parquet")))
        mean_u = float(np.mean([u for u in utils if u is not None])) if any(u is not None for u in utils) else None
        maxdd = float(np.min(dds)) if dds else None
        def_dd = float(np.min(default_dds)) if default_dds else None
        dd_ok = True if maxdd is None or def_dd is None else (maxdd >= def_dd - 0.02)
        # paired bootstrap vs default on concatenated overlapping days
        boot_ok = False
        mean_delta = None
        if daily_parts and key != default_key:
            # rebuild default daily for same blocks
            def_parts = []
            for start, end, tag in blocks:
                p = ROOT / "inner" / f"{year}" / default_key / tag / "nav.parquet"
                if p.is_file():
                    def_parts.append(daily_from_nav(pd.read_parquet(p)))
            if def_parts and daily_parts:
                cand = pd.concat(daily_parts)
                base = pd.concat(def_parts)
                cand = cand[~cand.index.duplicated(keep="last")].sort_index()
                base = base[~base.index.duplicated(keep="last")].sort_index()
                both = pd.concat([cand.rename("c"), base.rename("b")], axis=1, join="inner")
                if both.isna().any().any():
                    # 禁止静默 dropna 缩短评价窗；缺日则该配置不记 boot_ok
                    boot_ok = False
                    mean_delta = None
                elif len(both) >= 3 * INNER_BLOCK:
                    mean_delta = float((both["c"] - both["b"]).mean())
                    rng = np.random.default_rng(17)
                    n = len(both)
                    width = INNER_BLOCK
                    means = []
                    diff = (both["c"] - both["b"]).to_numpy()
                    for _ in range(INNER_BOOT_N):
                        starts = rng.integers(0, n, size=int(np.ceil(n / width)))
                        idx = np.concatenate([np.arange(int(s), int(s) + width) % n for s in starts])[:n]
                        means.append(float(diff[idx].mean()))
                    lb = float(np.quantile(means, 0.10))
                    boot_ok = mean_delta > 0 and lb > 0
        eligible = key == default_key or (
            n_days >= MIN_INNER_DAYS and n_trades >= MIN_INNER_TRADES and dd_ok and boot_ok and mean_delta is not None and mean_delta > 0
        )
        records.append(
            {
                "year": year,
                "key": key,
                "utility": mean_u,
                "n_days": n_days,
                "n_trades": n_trades,
                "maxdd": maxdd,
                "dd_ok": dd_ok,
                "mean_delta": mean_delta,
                "boot_ok": boot_ok if key != default_key else True,
                "eligible": eligible,
                "distance_to_default": 0.0 if key == default_key else abs(float(key.split("_")[-1]) - float(default_key.split("_")[-1])) if "_" in key else 1.0,
            }
        )
    eligible = [r for r in records if r["eligible"]]
    if not eligible:
        chosen = default_key
        reason = "FALLBACK_DEFAULT"
    else:
        eligible.sort(key=lambda r: (r["distance_to_default"], -(r["utility"] if r["utility"] is not None else -np.inf)))
        chosen = eligible[0]["key"]
        reason = "INNER_REPLAY"
    return {"year": year, "chosen": chosen, "reason": reason, "records": records}


def run_models(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("models", resume)
    if cached:
        return cached
    v5 = _v5_fused()
    five = pd.read_parquet(E0 / "five_pred.parquet")
    ten = pd.read_parquet(E0 / "ten_pred.parquet")
    lgbm_five = pd.read_parquet(ROOT / "lgbm" / "five_trunc4" / "predictions.parquet")
    lgbm_ten = pd.read_parquet(ROOT / "lgbm" / "ten_trunc5" / "predictions.parquet")
    from core.round17.backtest import fuse_heads

    lgbm = fuse_heads(lgbm_five, lgbm_ten)
    # M0
    m0 = run_formal_sleeve(ROOT / "models" / "M0_v5", v5, {"code": "R232_M0", "round": "23.2", "scope": "FULL_HISTORY_FIRST_TO_SECOND"}, resume=resume, counts_as_alpha=False, top_changed=False, fee_same_direction=True, not_only_year=True)
    # M1 LGBM alone 90/10
    m1 = run_formal_sleeve(ROOT / "models" / "M1_lgbm", lgbm, {"code": "R232_M1", "round": "23.2", "scope": "FULL_HISTORY_FIRST_TO_SECOND"}, resume=resume, top_changed=None, fee_same_direction=True, not_only_year=True)
    # M2 weights
    m2_scores = {}
    fallback_all = []
    for w in LGBM_WEIGHTS:
        scores, fb = blend_xgb_lgbm(v5, lgbm, w)
        m2_scores[f"w_{w:.2f}"] = scores
        if len(fb):
            fb = fb.copy()
            fb["w_lgbm"] = w
            fallback_all.append(fb)
        if abs(w - 0.0) < 1e-12:
            # identity path already endpoint-tested; still store
            continue
    if fallback_all:
        write_df(ROOT / "lgbm_fallback_days.csv", pd.concat(fallback_all, ignore_index=True))
    # inner select M2 per year
    weight_rows = []
    m2_parts = []
    for year in range(2018, 2027):
        configs = [(f"w_{w:.2f}", m2_scores[f"w_{w:.2f}"]) for w in LGBM_WEIGHTS]
        sel = _select_weight_by_inner(configs, year=year, default_key="w_0.00", resume=resume)
        weight_rows.append({"year": year, "family": "M2", "chosen": sel["chosen"], "reason": sel["reason"]})
        key = sel["chosen"]
        part = m2_scores[key]
        part = part.loc[pd.to_datetime(part["signal_date"]).dt.year.eq(year)]
        m2_parts.append(part)
        write_df(ROOT / "inner" / f"m2_year_{year}.csv", pd.DataFrame(sel["records"]))
    m2_scores_final = pd.concat(m2_parts, ignore_index=True)
    m2 = run_formal_sleeve(ROOT / "models" / "M2_selected", m2_scores_final, {"code": "R232_M2", "round": "23.2", "scope": "FULL_HISTORY_FIRST_TO_SECOND"}, resume=resume, top_changed=None, fee_same_direction=True, not_only_year=True)
    # M3 ten weights only
    m3_scores = {f"t_{w:.2f}": _blend_ten(five, ten, w) for w in TEN_WEIGHTS}
    m3_parts = []
    for year in range(2018, 2027):
        configs = [(f"t_{w:.2f}", m3_scores[f"t_{w:.2f}"]) for w in TEN_WEIGHTS]
        sel = _select_weight_by_inner(configs, year=year, default_key="t_0.90", resume=resume)
        weight_rows.append({"year": year, "family": "M3", "chosen": sel["chosen"], "reason": sel["reason"]})
        part = m3_scores[sel["chosen"]]
        m3_parts.append(part.loc[pd.to_datetime(part["signal_date"]).dt.year.eq(year)])
        write_df(ROOT / "inner" / f"m3_year_{year}.csv", pd.DataFrame(sel["records"]))
    m3 = run_formal_sleeve(ROOT / "models" / "M3_selected", pd.concat(m3_parts, ignore_index=True), {"code": "R232_M3", "round": "23.2", "scope": "FULL_HISTORY_FIRST_TO_SECOND"}, resume=resume, top_changed=None, fee_same_direction=True, not_only_year=True)
    write_df(ROOT / "weights_by_fold.csv", pd.DataFrame(weight_rows))
    write_df(ROOT / "inner_policy_replay.csv", pd.DataFrame(weight_rows))
    rows = []
    for code, summary in (("M0", m0), ("M1", m1), ("M2", m2), ("M3", m3)):
        rows.append({"code": code, "sharpe": summary.get("sharpe"), "cagr": summary.get("cagr"), "nav_end": summary.get("nav_end"), "nav_improvement": summary.get("nav_improvement"), "economically_material": summary.get("economically_material"), "trades": summary.get("completed_trades")})
    write_df(ROOT / "model_comparison.csv", pd.DataFrame(rows))
    # diversity diagnostic only
    both = v5.rename(columns={"score": "xgb"}).merge(lgbm.rename(columns={"score": "lgbm"}), on=["signal_date", "instrument_id"], how="inner")
    from core.r225241_stats import daily_spearman

    diversity = {"daily_spearman": daily_spearman(both, "xgb", "lgbm"), "diagnostic_only": True, "not_an_admission_gate": True}
    dump_json(ROOT / "diversity_diagnostic.json", diversity)
    payload = {"status": "DONE", "m0": m0, "m1": m1, "m2": m2, "m3": m3, "weights": weight_rows, "diversity": diversity}
    dump_json(ROOT / "models.json", payload)
    return save_stage(ROOT, "models", {"status": "DONE"})


def run_risk(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("risk", resume)
    if cached:
        return cached
    v5 = _v5_fused()
    overlay, _ = apply_b03_o01(v5)
    # planned orders
    ranked, _ = rank_unique(overlay)
    planned = ranked.loc[ranked["raw_rank"].le(2)].copy()
    planned["intended_slot"] = planned["raw_rank"]
    write_parquet(ROOT / "planned_orders.parquet", planned)
    # R0
    r0 = run_formal_sleeve(ROOT / "risk" / "R0", overlay, {"code": "R232_R0", "round": "23.2", "scope": "FULL_HISTORY_FIRST_TO_SECOND"}, resume=resume, already_overlayed=True, counts_as_alpha=False, top_changed=False, fee_same_direction=True, not_only_year=True)
    # R1 GAP7 allocation half — LEGACY OPEN INFO
    labels = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", "d1_open_gap"])
    labels["signal_date"] = pd.to_datetime(labels["trade_date"])
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    r1_scores = overlay.merge(labels[["signal_date", "instrument_id", "d1_open_gap"]], on=["signal_date", "instrument_id"], how="left")
    gap = pd.to_numeric(r1_scores["d1_open_gap"], errors="coerce")
    mult = np.where(gap.ge(GAP7), 0.5, 1.0)
    r1_scores["allocation_multiplier"] = pd.to_numeric(r1_scores.get("allocation_multiplier", 1.0), errors="coerce").fillna(1.0) * mult
    r1 = run_formal_sleeve(
        ROOT / "risk" / "R1_GAP7",
        r1_scores[["signal_date", "instrument_id", "score", "allocation_multiplier"]],
        {"code": "R232_R1_GAP7_ALLOCATION_HALF_DIAGNOSTIC", "round": "23.2", "scope": "EXECUTION_SCENARIO", "lane": LEGACY_OPEN_INFORMATION_LANE},
        resume=resume,
        already_overlayed=True,
        scenario=True,
        counts_as_alpha=False,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    # R2 matched exposure: train-period ratio per outer year
    parts = []
    ratios = []
    r1_alloc = r1_scores[["signal_date", "instrument_id", "allocation_multiplier"]].rename(
        columns={"allocation_multiplier": "r1_allocation_multiplier"}
    )
    p1 = planned.merge(r1_alloc, on=["signal_date", "instrument_id"], how="left")
    for year in range(2018, 2027):
        p1_train = p1.loc[pd.to_datetime(p1["signal_date"]).dt.year.between(year - 3, year - 1)]
        exp1 = (
            float(pd.to_numeric(p1_train["r1_allocation_multiplier"], errors="coerce").fillna(1.0).mean())
            if len(p1_train)
            else 1.0
        )
        exp0 = 1.0
        ratio = float(np.clip(exp1 / exp0 if exp0 else 1.0, 0.0, 1.0))
        ratios.append({"year": year, "ratio": ratio})
        year_scores = overlay.loc[pd.to_datetime(overlay["signal_date"]).dt.year.eq(year)].copy()
        year_scores["allocation_multiplier"] = ratio
        parts.append(year_scores)
    write_df(ROOT / "risk_r2_ratios.csv", pd.DataFrame(ratios))
    r2_scores = pd.concat(parts, ignore_index=True)
    r2 = run_formal_sleeve(
        ROOT / "risk" / "R2_MATCHED",
        r2_scores,
        {"code": "R232_R2_MATCHED_EXPOSURE", "round": "23.2", "scope": "DIAGNOSTIC_ONLY"},
        resume=resume,
        already_overlayed=True,
        counts_as_alpha=False,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    rows = []
    for code, summary, scope in (("R0", r0, "FULL"), ("R1", r1, "LEGACY_OPEN"), ("R2", r2, "MATCHED")):
        rows.append({"code": code, "scope": scope, "sharpe": summary.get("sharpe"), "cagr": summary.get("cagr"), "max_drawdown": summary.get("max_drawdown"), "nav_end": summary.get("nav_end"), "trades": summary.get("completed_trades")})
    write_df(ROOT / "risk_vs_alpha_comparison.csv", pd.DataFrame(rows))
    payload = {"status": "DONE", "r0": r0, "r1": r1, "r2": r2, "r1_not_deployable": True, "rename": "GAP7_ALLOCATION_HALF_DIAGNOSTIC"}
    dump_json(ROOT / "risk.json", payload)
    return save_stage(ROOT, "risk", payload)


def run_execution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("execution", resume)
    if cached:
        return cached
    v5 = _v5_fused()
    rows = []
    # S0
    s0 = run_formal_sleeve(ROOT / "exec" / "S0", v5, {"code": "R232_S0", "round": "23.2", "scope": "FULL_HISTORY_FIRST_TO_SECOND"}, resume=resume, counts_as_alpha=False, top_changed=False, fee_same_direction=True, not_only_year=True)
    rows.append({"scenario": "S0", **{k: s0.get(k) for k in ("sharpe", "cagr", "nav_end", "completed_trades")}})
    # S1/S2 buy extra slippage via validation hooks
    for name, bps in (("S1", 10), ("S2", 20)):
        from core.r225241_sleeve import run_round7

        dest = ROOT / "exec" / name
        obs = run_round7(
            dest,
            v5,
            {"code": f"R232_{name}", "round": "23.2", "scope": "EXECUTION_SCENARIO"},
            resume=resume,
            scenario=True,
            counts_as_alpha=False,
            apply_v5_gate=False,
            validation_hooks={"buy_extra_slippage_bps": bps},
        )
        rows.append({"scenario": name, "extra_buy_slippage_bps": bps, **{k: obs.get(k) for k in ("sharpe", "cagr", "nav_end", "completed_trades")}})
    # S3: 先固定 Top1/Top2 计划槽位，再在成交层去掉开盘封板；不重排、不递补 Top3
    overlay, _ = apply_b03_o01(v5)
    from core.r225241_sleeve import market

    prepared = market()["prepared"]
    calendar = pd.DatetimeIndex(prepared.calendar)
    pos = {pd.Timestamp(d): i for i, d in enumerate(calendar)}
    work = overlay.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort").copy()
    work["rk"] = work.groupby("signal_date").cumcount() + 1
    top = work.loc[work["rk"].le(2)].copy()
    top["entry_date"] = top["signal_date"].map(
        lambda day: calendar[pos[pd.Timestamp(day)] + 1]
        if pd.Timestamp(day) in pos and pos[pd.Timestamp(day)] + 1 < len(calendar)
        else pd.NaT
    )
    lookup = prepared.market_lookup
    sealed_flags = []
    for rec in top.itertuples(index=False):
        sealed = False
        if pd.notna(rec.entry_date):
            try:
                bar = lookup.loc[(pd.Timestamp(rec.entry_date), rec.instrument_id)]
                limit_up = float(bar["limit_up"]) if np.isfinite(bar["limit_up"]) and 0 < float(bar["limit_up"]) < 99_999 else np.nan
                sealed = bool(np.isfinite(limit_up) and np.isfinite(bar["open"]) and float(bar["open"]) + 1e-6 >= limit_up - 0.005)
            except KeyError:
                sealed = False
        sealed_flags.append(sealed)
    top["open_limit_up"] = sealed_flags
    n_cancel = int(top["open_limit_up"].sum())
    # 仅保留计划槽位中未封板者；预算不重分给未入选名
    s3 = top.loc[~top["open_limit_up"], [c for c in ("signal_date", "instrument_id", "score", "allocation_multiplier") if c in top.columns]].copy()
    if "allocation_multiplier" not in s3.columns:
        s3["allocation_multiplier"] = 1.0
    s3_out = run_formal_sleeve(
        ROOT / "exec" / "S3",
        s3,
        {
            "code": "R232_S3_OPEN_SEAL_CANCEL",
            "round": "23.2",
            "scope": "EXECUTION_SCENARIO",
            "n_cancelled": n_cancel,
            "no_rerank_no_replacement": True,
        },
        resume=resume,
        already_overlayed=True,
        scenario=True,
        counts_as_alpha=False,
        top_changed=None,
        fee_same_direction=True,
        not_only_year=True,
    )
    rows.append({"scenario": "S3", "n_cancelled": n_cancel, **{k: s3_out.get(k) for k in ("sharpe", "cagr", "nav_end", "completed_trades")}})
    write_df(ROOT / "execution_sensitivity.csv", pd.DataFrame(rows))
    return save_stage(ROOT, "execution", {"status": "DONE", "rows": rows})


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    series = {}
    for code, path in (
        ("M0", ROOT / "models" / "M0_v5" / "nav.parquet"),
        ("M1", ROOT / "models" / "M1_lgbm" / "nav.parquet"),
        ("M2", ROOT / "models" / "M2_selected" / "nav.parquet"),
        ("M3", ROOT / "models" / "M3_selected" / "nav.parquet"),
    ):
        if path.is_file():
            series[code] = daily_from_nav(pd.read_parquet(path))
    boots = {}
    for name in ("M1", "M2", "M3"):
        if name in series and "M0" in series:
            boots[name] = paired_block_bootstrap_v2(series[name], series["M0"])
    panel = pd.DataFrame(series)
    spa = spa_hansen_or_blocked(panel.dropna(how="any"), baseline="M0") if not panel.empty else {"status": "BLOCKED_STAT_TEST"}
    payload = {"bootstrap": boots, "spa": spa}
    write_df(ROOT / "statistics.csv", pd.DataFrame([{"name": k, "status": v.get("status")} for k, v in boots.items()] + [{"name": "SPA", "status": spa.get("status")}]))
    dump_json(ROOT / "statistics.json", payload)
    return save_stage(ROOT, "stats", payload)


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    ep = load_json(ROOT / "endpoint_parity.json")
    models = load_json(ROOT / "models.json")
    risk = load_json(ROOT / "risk.json")
    weights = pd.read_csv(ROOT / "weights_by_fold.csv") if (ROOT / "weights_by_fold.csv").is_file() else pd.DataFrame()
    material = any(bool((models.get(k) or {}).get("economically_material")) for k in ("m1", "m2", "m3"))
    decision = "KEEP_BASELINE_V5" if not material else "V6_RESEARCH_CANDIDATE_NOT_PROMOTED"
    payload = {
        "decision": decision,
        "endpoint_parity": ep.get("status"),
        "zero_weight_identity": ep.get("status") == "PASS",
        "ensemble_selected_weights": weights.to_dict(orient="records") if len(weights) else [],
        "risk_configuration_verdict": "DIAGNOSTIC_ONLY_NOT_ALPHA_REPLACEMENT",
        "current_baseline_changed": False,
        "auto_promote": False,
        "production_config_changed": False,
    }
    dump_json(ROOT / "decision.json", payload)
    write_text(
        ROOT / "FINAL_DECISION.md",
        "\n".join(
            [
                "# Round23.2 最终裁决",
                "",
                f"- 端点一致性：`{ep.get('status')}`",
                f"- 决策：`{decision}`",
                "- 多样性仅诊断，不作准入。",
                "- R1=GAP7_ALLOCATION_HALF_DIAGNOSTIC，不部署。",
                "- CURRENT_BASELINE_CHANGED=false",
                "- PRODUCTION_CONFIG_CHANGED=false",
                "",
            ]
        ),
    )
    # all trials
    parts = []
    for name in ("model_comparison.csv", "risk_vs_alpha_comparison.csv", "execution_sensitivity.csv"):
        path = ROOT / name
        if path.is_file():
            df = pd.read_csv(path)
            df["source_table"] = name
            parts.append(df)
    if parts:
        write_df(ROOT / "all_trials.csv", pd.concat(parts, ignore_index=True, sort=False))
    return save_stage(ROOT, "decision", payload)


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dec = load_json(ROOT / "decision.json")
    lines = [
        "# Round23.2 研究报告",
        "",
        "## 1. 修复",
        "- w_lgbm=0 直接返回冻结 v5 路径。",
        "- 内层选择改为完整袖套重放 + Mean(log(1+r))，不再用标签均值代替。",
        "- 多样性降为诊断；允许全年 w_lgbm=0。",
        "- GAP7 半仓改名 GAP7_ALLOCATION_HALF_DIAGNOSTIC。",
        "",
        "## 2. 端点与模型",
        f"- endpoint=`{dec.get('endpoint_parity')}` decision=`{dec.get('decision')}`",
        "- 详见 model_comparison.csv / weights_by_fold.csv / risk_vs_alpha_comparison.csv / execution_sensitivity.csv",
        "",
        "## 3. 是否替换 baseline",
        f"- `{dec.get('decision')}`；CURRENT 未改。",
        "",
    ]
    write_text(ROOT / "ROUND23_2_RESEARCH_REPORT.md", "\n".join(lines))
    return save_stage(ROOT, "report", {"status": "DONE"})


STAGES = {
    "freeze": run_freeze,
    "endpoint": run_endpoint_parity,
    "lgbm": run_lgbm,
    "models": run_models,
    "risk": run_risk,
    "execution": run_execution,
    "stats": run_stats,
    "decision": run_decision,
    "report": run_report,
}


def run_all(*, resume: bool = True, jobs: int = 2, **kwargs: Any) -> dict[str, Any]:
    out = {}
    for name, fn in STAGES.items():
        print(f"[round23_2] stage={name}", flush=True)
        out[name] = fn(resume=resume, jobs=jobs, **kwargs)
    return out
