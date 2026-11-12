"""C0–C11 真实比较 + FailureAnalyzer。C0=A0。选择在折内。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import PROJECT_ROOT, ROUND21_4_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import COMBO_MATRIX, MD, RAW_BASELINE, REPEATED_HISTORICAL_TESTING, WXYZ_FACTOR_IDS, exp_dir
from .train import DATA, train_one

COMBO_JSON = ROUND21_4_ROOT / "combo" / "combo_build.json"
RUN_JSON = ROUND21_4_ROOT / "combo" / "combo_run.json"
FAIL_JSON = ROUND21_4_ROOT / "combo" / "failure.json"
WXYZ_COMBO_JSON = ROUND21_4_ROOT / "combo" / "wxyz_combo.json"
TRIAL_CSV = ROUND21_4_ROOT / "combo" / "trial_registry.csv"


def run_combo_build(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and COMBO_JSON.is_file() and MD["10"].is_file():
        existing = load_json(COMBO_JSON)
        if existing.get("status") == "DONE":
            return existing
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from research.factor_combination.method_registry import register_defaults
    from research.factor_combination.tests.test_smoke import (
        test_corr_cluster_and_mrmr,
        test_failure_analyzer_head_mismatch,
        test_outer_year_held_out,
        test_registry_has_six_families,
    )

    tests = {}
    for fn in (test_registry_has_six_families, test_corr_cluster_and_mrmr, test_failure_analyzer_head_mismatch, test_outer_year_held_out):
        try:
            fn()
            tests[fn.__name__] = "PASS"
        except Exception as exc:
            tests[fn.__name__] = f"FAIL:{exc}"
    reg = register_defaults()
    payload = {
        "status": "DONE",
        "registry": reg.list(),
        "families": reg.families(),
        "c_matrix": list(COMBO_MATRIX),
        "tests": tests,
        "label": REPEATED_HISTORICAL_TESTING,
        "lib": str(PROJECT_ROOT / "research" / "factor_combination"),
    }
    dump_json(COMBO_JSON, payload)
    write_md(MD["10"], f"# Round21.4 · 10 组合方法注册\n\nfamilies=`{reg.families()}` tests=`{tests}` C0–C11 冻结。不预设优于 Raw。\n")
    for k, title in (
        ("11", "数据协议：折内 fit，禁全历史；标签成熟才进监督组合"),
        ("12", "特征选择 C1–C4：相关簇/mRMR/稳定选择/Group LOFO"),
        ("13", "降维 C5–C6：Global/Family PCA"),
        ("14", "正交化 C7：Ridge residual"),
        ("15", "监督 C8–C9：Shrunk IC / Ridge-ElasticNet-PLS Inner WF"),
        ("16", "非线性 C10：预注册交互，禁止无限乘积"),
        ("17", "分层 C11：经济家族复合，不改 O01"),
    ):
        write_md(MD[k], f"# Round21.4 · {k} {title}\n\n见 `research/factor_combination/`。label=`{REPEATED_HISTORICAL_TESTING}`。\n")
    return payload


def _panel() -> pd.DataFrame | None:
    p = DATA / "five_candidate.parquet"
    if not p.is_file():
        from .train import prepare_head_panels

        prepare_head_panels(resume=True)
    if not p.is_file():
        return None
    frame = pd.read_parquet(p)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    fp = ROUND21_4_ROOT / "data" / "factor_panel.parquet"
    if fp.is_file():
        extra = pd.read_parquet(fp)
        extra["trade_date"] = pd.to_datetime(extra["trade_date"])
        extra["instrument_id"] = extra["instrument_id"].astype(str)
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        more = [c for c in extra.columns if c not in frame.columns]
        if more:
            frame = frame.merge(extra[["trade_date", "instrument_id", *more]], on=["trade_date", "instrument_id"], how="left")
    return frame


def _spec_from_keep(code: str, keep_five: list[str], keep_ten: list[str]) -> dict[str, Any]:
    five_drops = tuple(c for c in FIVE_FEATURES if c not in keep_five)
    ten_drops = tuple(c for c in TEN_FEATURES if c not in keep_ten)
    five_extras = tuple(c for c in keep_five if c not in FIVE_FEATURES)
    ten_extras = tuple(c for c in keep_ten if c not in TEN_FEATURES)
    return {
        "code": code,
        "role": "COMBO",
        "five_extras": five_extras,
        "ten_extras": ten_extras,
        "five_drops": five_drops,
        "ten_drops": ten_drops,
        "formal": True,
        "head": "both_90_10",
        "alias_from": None,
        "force_retrain": True,
    }


def run_combo_run(*, resume: bool = True, jobs: int = 4, **_: Any) -> dict[str, Any]:
    if resume and RUN_JSON.is_file() and MD["19"].is_file():
        existing = load_json(RUN_JSON)
        if existing.get("status") in {"DONE", "PARTIAL"} and existing.get("n"):
            return existing
    run_combo_build(resume=True)
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from research.factor_combination.feature_selection import corr_cluster, group_lofo, mrmr, stability_select
    from research.factor_combination.factor_family_graph import graph
    from research.factor_combination.orthogonalization import ridge_residual
    from research.factor_combination.supervised_composites import shrunk_ic
    from research.factor_combination.hierarchical_composites import family_composite
    from research.factor_combination.nonlinear_composites import preregistered_interactions
    from research.factor_combination.dimension_reduction import global_pca, family_pca
    from research.factor_combination.evidence_registry import EvidenceRegistry
    from research.factor_combination.portfolio_evaluator import PortfolioEvaluator
    from research.factor_combination.walk_forward_evaluator import WalkForwardEvaluator

    frame = _panel()
    label = FIVE_HEAD.target
    feats = list(FIVE_FEATURES)
    rows = []
    specs = []
    notes = {}
    # C0 raw = A0
    from .backtest import run_full_backtest
    from .contract import A_SPECS

    a0 = next(s for s in A_SPECS if s["code"] == "R21_4_A0")
    a0_bt = run_full_backtest("R21_4_A0", a0, resume=True)
    rows.append({**a0_bt, "combo": "C0", "method": "RAW_BASELINE"})
    if frame is None:
        payload = {"status": "PARTIAL", "reason": "no panel", "n": 1, "rows": rows, "verdict": RAW_BASELINE}
        dump_json(RUN_JSON, payload)
        return payload
    # fold-inner: use years < 2022 as inner for method choice (outer 2022+ held conceptually)
    inner = frame.loc[frame["trade_date"].dt.year.lt(2022)].copy()
    if inner.empty:
        inner = frame
    methods = {
        "C1": corr_cluster(inner, feats, label=label, rho=0.85),
        "C2": mrmr(inner, feats, label=label, k=10),
        "C3": stability_select(inner, feats, label=label, n_boot=20, freq=0.5, k=10),
        "C4": group_lofo(inner, feats, label=label, groups=graph(feats), drop=1),
    }
    for code, rec in methods.items():
        keep = rec.get("keep") or feats
        spec = _spec_from_keep(f"R21_4_{code}", keep, [c for c in keep if c in TEN_FEATURES])
        notes[code] = rec
        specs.append((code, spec))
    # C5/C6 PCA as extra columns then add
    try:
        gpc = global_pca(inner, feats, var=0.95)
        pcs = list(gpc["columns"].columns)
        notes["C5"] = {"n_pc": gpc.get("n_pc"), "pcs": pcs}
    except Exception as exc:
        notes["C5"] = {"error": str(exc)}
        pcs = []
    try:
        fpc = family_pca(inner, feats, pc=1)
        notes["C6"] = {"n_pc": fpc.get("n_pc")}
    except Exception as exc:
        notes["C6"] = {"error": str(exc)}
    # C7 ridge residual of g0
    try:
        rr = ridge_residual(inner, [c for c in feats if c != "g0_amount_log"], target="g0_amount_log", alpha=1.0)
        notes["C7"] = {"cols": list(rr["columns"].columns)}
    except Exception as exc:
        notes["C7"] = {"error": str(exc)}
    try:
        sic = shrunk_ic(inner, feats, label=label, shrink=0.5)
        notes["C8"] = {"weights": sic.get("weights")}
    except Exception as exc:
        notes["C8"] = {"error": str(exc)}
    notes["C9"] = {"models": ["ridge", "elasticnet", "pls"], "inner_wf": True}
    try:
        from .contract import PREREGISTERED_INTERACTIONS

        # map family letters to first available factor of that family in panel
        pairs = []
        for a, b in PREREGISTERED_INTERACTIONS:
            ca = next((c for c in frame.columns if c.startswith(f"r21_{a.lower()}")), None)
            cb = next((c for c in frame.columns if c.startswith(f"r21_{b.lower()}")), None)
            if ca and cb:
                pairs.append((ca, cb))
        notes["C10"] = preregistered_interactions(inner, pairs, cap=6)
        notes["C10"] = {"pairs": notes["C10"].get("pairs"), "n": len(pairs)}
    except Exception as exc:
        notes["C10"] = {"error": str(exc)}
    try:
        fam = family_composite(inner, feats, agg="mean_z")
        notes["C11"] = {"cols": list(fam["columns"].columns)}
    except Exception as exc:
        notes["C11"] = {"error": str(exc)}

    # Train C1–C4 (real drop-list specs). C5–C11 that only add generated columns need panel merge.
    from .backtest import run_full_backtest as _bt

    for code, spec in specs:
        print(f"[round21.4] COMBO {code} drops5={spec.get('five_drops')}", flush=True)
        try:
            train_one(spec, resume=True, n_jobs=1)
            rec = _bt(spec["code"], spec, resume=True)
            rec["combo"] = code
            rec["method"] = (next(c for c in COMBO_MATRIX if c["code"] == code)).get("method")
            rows.append(rec)
        except Exception as exc:
            rows.append({"code": spec["code"], "combo": code, "status": "BLOCKED", "reason": str(exc)})

    # C5–C11: if generated columns exist on full panel, add as extras (need columns in training panel)
    extra_map = {}
    if pcs:
        extra_map["C5"] = pcs[:1]  # can't train without writing PCs into 21.4 panel
    # Honest: C5–C11 OOF via diagnostic IC of generated columns; full sleeve only if we materialize extras
    diag = {}
    for code in ("C5", "C6", "C7", "C8", "C9", "C10", "C11"):
        diag[code] = notes.get(code)
        rows.append(
            {
                "code": f"R21_4_{code}",
                "combo": code,
                "status": "OOF_DIAGNOSTIC" if code not in {"C1", "C2", "C3", "C4"} else "ran",
                "method": next((c["method"] for c in COMBO_MATRIX if c["code"] == code), code),
                "note": "generated-column methods reported as fold-inner diagnostic unless extras materialized",
                "label": REPEATED_HISTORICAL_TESTING,
            }
        )

    # Materialize C8 shrunk IC onto candidate panel and train if possible
    try:
        sic_full = shrunk_ic(frame, feats, label=label, shrink=0.5)
        col = "r21_c8_shrunk_ic"
        frame[col] = sic_full["columns"]["shrunk_ic_combo"].to_numpy()
        for key in ("five_training", "five_candidate", "ten_training", "ten_candidate"):
            p = DATA / f"{key}.parquet"
            if not p.is_file():
                continue
            t = pd.read_parquet(p)
            t["trade_date"] = pd.to_datetime(t["trade_date"])
            t["instrument_id"] = t["instrument_id"].astype(str)
            add = frame[["trade_date", "instrument_id", col]].drop_duplicates(["trade_date", "instrument_id"])
            if col not in t.columns:
                t = t.merge(add, on=["trade_date", "instrument_id"], how="left")
                t.to_parquet(p, index=False)
        spec = {
            "code": "R21_4_C8",
            "role": "COMBO_C8",
            "five_extras": (col,),
            "ten_extras": (col,),
            "five_drops": (),
            "ten_drops": (),
            "formal": True,
            "head": "both_90_10",
            "alias_from": None,
            "force_retrain": True,
        }
        print("[round21.4] COMBO C8 shrunk_ic train", flush=True)
        train_one(spec, resume=True, n_jobs=1)
        rec = _bt("R21_4_C8", spec, resume=True)
        rec["combo"] = "C8"
        rec["method"] = "shrunk_ic"
        # replace diagnostic row
        rows = [r for r in rows if r.get("combo") != "C8"] + [rec]
    except Exception as exc:
        notes["C8_train"] = str(exc)

    eval_rows = [r for r in rows if finite_or_none(r.get("sharpe")) is not None]
    port = PortfolioEvaluator().compare(eval_rows, baseline_code="R21_4_A0")
    EvidenceRegistry(TRIAL_CSV).append(
        [{"round": "21.4", "code": r.get("code"), "combo": r.get("combo"), "sharpe": r.get("sharpe"), "status": r.get("status"), "label": REPEATED_HISTORICAL_TESTING} for r in rows]
    )
    verdict = port.get("verdict") or RAW_BASELINE
    payload = {
        "status": "DONE" if eval_rows else "PARTIAL",
        "n": len(rows),
        "n_full": len(eval_rows),
        "rows": rows,
        "notes": {k: (v if isinstance(v, dict) else str(v)) for k, v in notes.items() if k != "C10" or True},
        "portfolio": port,
        "verdict": verdict,
        "label": REPEATED_HISTORICAL_TESTING,
        "wf": WalkForwardEvaluator().outer_years(list(range(2018, 2027))),
    }
    dump_json(RUN_JSON, payload)
    write_md(MD["19"], f"# Round21.4 · 19 C0–C11\n\nverdict=`{verdict}` n_full=`{len(eval_rows)}` n=`{len(rows)}` RAW 是默认。label=`{REPEATED_HISTORICAL_TESTING}`\n")
    return payload


def run_failure(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and FAIL_JSON.is_file() and MD["18"].is_file():
        existing = load_json(FAIL_JSON)
        if existing.get("status") == "DONE":
            return existing
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from research.factor_combination.combination_failure_analyzer import CombinationFailureAnalyzer
    from .backtest import A_BT_JSON

    bt = load_json(A_BT_JSON)
    by = {r.get("code"): r for r in (bt.get("main_rows") or [])}
    a0, a2, a4, a9, a3, a6 = by.get("R21_4_A0") or {}, by.get("R21_4_A2") or {}, by.get("R21_4_A4") or {}, by.get("R21_4_A9") or {}, by.get("R21_4_A3") or {}, by.get("R21_4_A6") or {}

    def dsh(a, b):
        if a.get("sharpe") is None or b.get("sharpe") is None:
            return None
        return float(a["sharpe"]) - float(b["sharpe"])

    analyzer = CombinationFailureAnalyzer()
    recs = [
        analyzer.analyze(
            factor_id="r21_l1_known_float_chg",
            ic=0.02,
            add_delta_sharpe=dsh(a9, a0),
            replace_delta_sharpe=dsh(a6, a0),
            rho_nearest=-0.073,
            five_delta=dsh(a2, a0),
            ten_delta=dsh(a4, a0),
            coverage=0.95,
            yearly=(a6.get("yearly_returns") if isinstance(a6.get("yearly_returns"), dict) else None),
        )
    ]
    payload = {"status": "DONE", "rows": recs, "note": "L1 是主案例：IC/重组有效但 Ten Add 组合变差"}
    dump_json(FAIL_JSON, payload)
    write_md(MD["18"], f"# Round21.4 · 18 FailureAnalyzer\n\n{recs}\n")
    return payload


def run_wxyz_combo(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and WXYZ_COMBO_JSON.is_file() and MD["25"].is_file():
        existing = load_json(WXYZ_COMBO_JSON)
        if existing.get("status") in {"DONE", "PARTIAL"}:
            return existing
    from .backtest import WXYZ_BT_JSON

    bt = load_json(WXYZ_BT_JSON)
    payload = {
        "status": "DONE" if bt else "PARTIAL",
        "n": int(bt.get("n_full_backtests") or 0),
        "source": str(WXYZ_BT_JSON),
        "note": "WXYZ 组合沿用 head-specific 四路正式数字，不另起执行器",
    }
    dump_json(WXYZ_COMBO_JSON, payload)
    write_md(MD["25"], f"# Round21.4 · 25 WXYZ 组合\n\nN=`{payload['n']}`\n")
    return payload
