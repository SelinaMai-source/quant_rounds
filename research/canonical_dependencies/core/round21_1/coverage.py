"""10 个 PIT_FAIL_COVERAGE 法医。禁止填 0。E 族仍 BLOCKED_PIT。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_1_ROOT, ROUND21_ROOT
from ..round21.construct import F_CACHE, PANEL_PATH
from ..round21.min1 import AC_CACHE
from .common import dump_json, load_json, write_md
from .companion import build_companions
from .contract import (
    BLOCKED_PIT,
    BLOCKED_PIT_R21,
    COVERAGE_CLASSES,
    COVERAGE_MD,
    DIAGNOSTIC_ONLY,
    KEYS,
    PIT_ELIGIBLE_MODEL,
    PIT_FAIL_COVERAGE_R21,
    PIT_MIN_FINITE,
    PIT_PASS_R21,
)

COV_JSON = ROUND21_1_ROOT / "coverage" / "coverage.json"
COV_CSV = ROUND21_1_ROOT / "coverage" / "coverage_failure_reasons.csv"
PIT_CSV = ROUND21_1_ROOT / "coverage" / "companion_pit.csv"


def _finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _share(mask: pd.Series) -> float:
    n = int(len(mask))
    return float(mask.sum() / n) if n else 0.0


def _classify_ac(panel: pd.DataFrame, ac: pd.DataFrame) -> list[dict[str, Any]]:
    work = panel[KEYS].copy()
    work["a1"] = _finite(panel["r21_a1_vw_limit_gap"]) if "r21_a1_vw_limit_gap" in panel.columns else np.nan
    work["a2"] = _finite(panel["r21_a2_minutes_to_reseal"]) if "r21_a2_minutes_to_reseal" in panel.columns else np.nan
    work["a3"] = _finite(panel["r21_a3_retouch_count"]) if "r21_a3_retouch_count" in panel.columns else np.nan
    work["b4"] = _finite(panel["r21_b4_activity_x_resilience"]) if "r21_b4_activity_x_resilience" in panel.columns else np.nan
    work["c1"] = _finite(panel["r21_c1_min1_amihud"]) if "r21_c1_min1_amihud" in panel.columns else np.nan
    work["c2"] = _finite(panel["r21_c2_post_touch_amt_vs_impact"]) if "r21_c2_post_touch_amt_vs_impact" in panel.columns else np.nan
    work["c3"] = _finite(panel["r21_c3_post_touch_path_eff"]) if "r21_c3_post_touch_path_eff" in panel.columns else np.nan
    work["c4"] = _finite(panel["r21_c4_downside_quad_var"]) if "r21_c4_downside_quad_var" in panel.columns else np.nan
    if not ac.empty:
        ac_use = ac.copy()
        ac_use["trade_date"] = pd.to_datetime(ac_use["trade_date"])
        ac_use["instrument_id"] = ac_use["instrument_id"].astype(str)
        keep = [c for c in ("touch_flag", "touch_seconds") if c in ac_use.columns]
        work = work.merge(ac_use[KEYS + keep], on=KEYS, how="left")
    else:
        work["touch_flag"] = None
        work["touch_seconds"] = np.nan
    flag = work["touch_flag"].astype(str) if "touch_flag" in work.columns else pd.Series("", index=work.index)
    min1_missing = flag.eq("SKIP_DATA") | (work["a3"].isna() & work["c1"].isna())
    min1_ok = ~min1_missing
    a3v = work["a3"].fillna(0)

    rows = []

    def _pack(fid: str, cls: str, finite: pd.Series, applicable: pd.Series, note: str, **extra: Any) -> dict[str, Any]:
        n = int(len(work))
        n_fin = int(np.isfinite(finite.to_numpy(dtype=float)).sum())
        n_app = int(applicable.sum())
        n_app_fin = int((applicable & np.isfinite(finite.to_numpy(dtype=float))).sum())
        n_true_miss = int((~applicable & finite.isna()).sum()) if cls == "TRUE_DATA_MISSING" else int((min1_missing & finite.isna()).sum())
        cond = (n_app_fin / n_app) if n_app else None
        return {
            "factor_id": fid,
            "classification": cls,
            "universe_n": n,
            "universe_finite": n_fin,
            "universe_coverage": n_fin / n if n else 0.0,
            "applicable_n": n_app,
            "applicable_finite": n_app_fin,
            "conditional_coverage": cond,
            "true_missing_n": n_true_miss,
            "true_missing_rate": n_true_miss / n if n else 0.0,
            "filled_with_zero": False,
            "note": note,
            **extra,
        }

    # A1: 触板后无「低于涨停且有成交」→ 公式不可定义
    a1_fin = work["a1"]
    a1_event = min1_ok & a1_fin.isna()
    a1_data = min1_missing & a1_fin.isna()
    a1_cls = "EVENT_CONDITIONAL" if int(a1_event.sum()) >= int(a1_data.sum()) else "TRUE_DATA_MISSING"
    if int(a1_event.sum()) > 0 and a1_cls == "EVENT_CONDITIONAL":
        a1_cls = "FORMULA_UNDEFINED" if int((min1_ok & a1_fin.isna()).sum()) > 0 else a1_cls
        # 有分钟但无 below-limit-with-trade：公式分母不存在
        a1_cls = "EVENT_CONDITIONAL"
    rows.append(
        _pack(
            "r21_a1_vw_limit_gap",
            a1_cls,
            a1_fin,
            min1_ok & ~a1_event if False else min1_ok,
            "对照 A3/C1 ~99.8% 排除全局 min1 缺失。A1 有限当且仅当触板后存在 close<limit 且 amount>0 的分钟。无此事件则分母不存在。",
            n_event_na=int(a1_event.sum()),
            n_min1_missing=int(a1_data.sum()),
            a3_universe_coverage=float(work["a3"].notna().mean()),
            c1_universe_coverage=float(work["c1"].notna().mean()),
        )
    )

    # A2: 无开板或无回封
    a2_fin = work["a2"]
    never_open = min1_ok & a3v.eq(0) & a2_fin.isna()
    opened_no_reseal = min1_ok & a3v.gt(0) & a2_fin.isna()
    a2_data = min1_missing & a2_fin.isna()
    if int(never_open.sum()) >= int(opened_no_reseal.sum()) and int(never_open.sum()) >= int(a2_data.sum()):
        a2_cls = "FORMULA_UNDEFINED"
        a2_app = min1_ok & a3v.gt(0)
        a2_note = "从未开板：first_leave_sec 不存在，公式不可定义。"
    elif int(opened_no_reseal.sum()) >= int(a2_data.sum()):
        a2_cls = "EVENT_CONDITIONAL"
        a2_app = min1_ok & a3v.gt(0)
        a2_note = "已开板但未回封：事件未发生，保持 NA。"
    else:
        a2_cls = "TRUE_DATA_MISSING"
        a2_app = min1_ok
        a2_note = "分钟/触板时点缺失。"
    rows.append(
        _pack(
            "r21_a2_minutes_to_reseal",
            a2_cls,
            a2_fin,
            a2_app,
            a2_note,
            n_never_open=int(never_open.sum()),
            n_opened_no_reseal=int(opened_no_reseal.sum()),
            n_min1_missing=int(a2_data.sum()),
        )
    )

    # B4 继承 A2
    b4_fin = work["b4"]
    rows.append(
        _pack(
            "r21_b4_activity_x_resilience",
            a2_cls,
            b4_fin,
            a2_app,
            "B4 = B1 * 1/(1+A2)。A2 为 NA 则交互传播 NA。分类继承 A2。",
            inherited_from="r21_a2_minutes_to_reseal",
        )
    )

    # C2: 触板后振幅=0
    c2_fin = work["c2"]
    c2_amp0 = min1_ok & work["c1"].notna() & c2_fin.isna()
    c2_no_path = min1_ok & work["c1"].isna() & c2_fin.isna()
    c2_data = min1_missing & c2_fin.isna()
    if int(c2_amp0.sum()) >= int(c2_no_path.sum()) and int(c2_amp0.sum()) >= int(c2_data.sum()):
        c2_cls = "EVENT_CONDITIONAL"
        c2_app = min1_ok & work["c1"].notna()
        c2_note = "触板后 max(high)-min(low)=0（未打开），分母不可用。"
    elif int(c2_no_path.sum()) >= int(c2_data.sum()):
        c2_cls = "FORMULA_UNDEFINED"
        c2_app = min1_ok & work["c1"].notna()
        c2_note = "触板后无可用分钟路径。"
    else:
        c2_cls = "TRUE_DATA_MISSING"
        c2_app = min1_ok
        c2_note = "分钟缺失。"
    rows.append(
        _pack(
            "r21_c2_post_touch_amt_vs_impact",
            c2_cls,
            c2_fin,
            c2_app,
            c2_note,
            n_amp0=int(c2_amp0.sum()),
            n_no_path=int(c2_no_path.sum()),
            n_min1_missing=int(c2_data.sum()),
        )
    )

    # C3 / C4: 路径长度不足
    for fid, col in (("r21_c3_post_touch_path_eff", "c3"), ("r21_c4_downside_quad_var", "c4")):
        fin = work[col]
        short = min1_ok & fin.isna() & (work["c1"].notna() | work["a3"].notna())
        data = min1_missing & fin.isna()
        if int(short.sum()) >= int(data.sum()):
            cls = "EVENT_CONDITIONAL"
            app = min1_ok & (work["c1"].notna() | work["a3"].notna())
            note = "触板后有效 close 点数 < 2 或路径长度/二次变差为 0，公式分母不存在。"
        else:
            cls = "TRUE_DATA_MISSING"
            app = min1_ok
            note = "分钟缺失。"
        rows.append(
            _pack(fid, cls, fin, app, note, n_short_path=int(short.sum()), n_min1_missing=int(data.sum()))
        )
    return rows


def _classify_f(panel: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    f_known = None
    if F_CACHE.is_file():
        fcache = pd.read_parquet(F_CACHE)
        fcache["trade_date"] = pd.to_datetime(fcache["trade_date"])
        fcache["instrument_id"] = fcache["instrument_id"].astype(str)
        col = "f_known_at" if "f_known_at" in fcache.columns else ("available_at" if "available_at" in fcache.columns else None)
        if col:
            f_known = fcache[KEYS + [col]].copy()
            f_known["f_known_at"] = pd.to_datetime(f_known[col], errors="coerce")
    work = panel[KEYS].copy()
    work["year"] = pd.to_datetime(work["trade_date"]).dt.year
    if f_known is not None:
        work = work.merge(f_known[KEYS + ["f_known_at"]], on=KEYS, how="left")
    else:
        work["f_known_at"] = pd.NaT
    has_asof = work["f_known_at"].notna()
    early = work["year"].isin({2013, 2014})
    for fid in ("r21_f1_fin_bal_chg", "r21_f2_fin_repay_pressure", "r21_f3_fin_buy_intensity", "r21_f4_fin_bal_accel"):
        fin = _finite(panel[fid]) if fid in panel.columns else pd.Series(np.nan, index=panel.index)
        structural = ~has_asof
        formula_na = has_asof & fin.isna()
        early_miss = early & fin.isna()
        if int(structural.sum()) >= int(formula_na.sum()):
            cls = "STRUCTURAL_NOT_APPLICABLE"
            app = has_asof
            note = "非两融标的不在 daily_margin 表；join 后保持 NULL。禁止把结构性不适用当数据质量失败，禁止填 0。"
        elif int(early_miss.sum()) > 0 and float(early_miss.mean()) > 0.2:
            cls = "SOURCE_ALIGNMENT_FAILURE"
            app = has_asof
            note = "2013–14 available_at 未对齐或 asof 失败。禁止用 trade_date 当天当已发布。"
        else:
            cls = "FORMULA_UNDEFINED"
            app = has_asof
            note = "两融行存在但上期余额/还款/买入缺失，公式分母不可用。"
        n = int(len(work))
        n_fin = int(np.isfinite(fin.to_numpy(dtype=float)).sum())
        n_app = int(app.sum())
        n_app_fin = int((app.to_numpy() & np.isfinite(fin.to_numpy(dtype=float))).sum())
        rows.append(
            {
                "factor_id": fid,
                "classification": cls,
                "universe_n": n,
                "universe_finite": n_fin,
                "universe_coverage": n_fin / n if n else 0.0,
                "applicable_n": n_app,
                "applicable_finite": n_app_fin,
                "conditional_coverage": (n_app_fin / n_app) if n_app else None,
                "true_missing_n": int(formula_na.sum()),
                "true_missing_rate": float(formula_na.mean()) if n else 0.0,
                "structural_n": int(structural.sum()),
                "structural_rate": float(structural.mean()) if n else 0.0,
                "n_2013_2014_missing": int(early_miss.sum()),
                "filled_with_zero": False,
                "note": note,
            }
        )
    return rows


def _pit_one(fid: str, series: pd.Series, *, family: str, formula: str) -> dict[str, Any]:
    n = int(len(series))
    s = pd.to_numeric(series, errors="coerce")
    v = int(np.isfinite(s.to_numpy(dtype=float)).sum())
    cov = v / n if n else 0.0
    finite = s[np.isfinite(s.to_numpy(dtype=float))]
    nunique = int(finite.nunique(dropna=True)) if len(finite) else 0
    if family == "E":
        status, eligible = BLOCKED_PIT, DIAGNOSTIC_ONLY
        reason = "E 族无公告时点，仍禁模型"
    elif v == 0:
        status, eligible = "BLOCKED_DATA", DIAGNOSTIC_ONLY
        reason = "all missing; no fill-0"
    elif nunique <= 1:
        status, eligible = "PIT_FAIL_DEGENERATE", DIAGNOSTIC_ONLY
        reason = "degenerate"
    elif cov < PIT_MIN_FINITE:
        status, eligible = "PIT_FAIL_COVERAGE", DIAGNOSTIC_ONLY
        reason = f"finite_share={cov:.4f} < {PIT_MIN_FINITE}"
    else:
        status, eligible = "PIT_PASS", PIT_ELIGIBLE_MODEL
        reason = None
    return {
        "factor_id": fid,
        "family": family,
        "formula": formula,
        "pit_status": status,
        "model_eligible": eligible,
        "n": n,
        "n_valid": v,
        "coverage": cov,
        "n_unique": nunique,
        "skip_reason": reason,
        "filled_with_zero": False,
    }


def run_coverage(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    if resume and COV_JSON.is_file() and COV_CSV.is_file() and COVERAGE_MD.is_file():
        existing = load_json(COV_JSON)
        if existing.get("status") == "DONE":
            return existing
    if not PANEL_PATH.is_file():
        raise RuntimeError(f"MISSING_ARTIFACT R21 factor_panel (read-only): {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    ac = pd.read_parquet(AC_CACHE) if AC_CACHE.is_file() else pd.DataFrame()
    if not ac.empty:
        ac["trade_date"] = pd.to_datetime(ac["trade_date"])
        ac["instrument_id"] = ac["instrument_id"].astype(str)

    rows = _classify_ac(panel, ac) + _classify_f(panel)
    for fid in BLOCKED_PIT_R21:
        s = _finite(panel[fid]) if fid in panel.columns else pd.Series(np.nan, index=panel.index)
        n = int(len(panel))
        v = int(np.isfinite(s.to_numpy(dtype=float)).sum())
        rows.append(
            {
                "factor_id": fid,
                "classification": "TIMESTAMP_UNVERIFIED",
                "universe_n": n,
                "universe_finite": v,
                "universe_coverage": v / n if n else 0.0,
                "applicable_n": 0,
                "applicable_finite": 0,
                "conditional_coverage": None,
                "true_missing_n": n - v,
                "true_missing_rate": (n - v) / n if n else 0.0,
                "filled_with_zero": False,
                "note": "E 族 block_trading 仅 TRADE_DATE，无 ANN/available_at。仍 BLOCKED_PIT，禁入正式模型。",
                "blocked_pit": True,
            }
        )

    bad_cls = [r for r in rows if r.get("classification") not in COVERAGE_CLASSES]
    if bad_cls:
        raise RuntimeError(f"invalid coverage class: {[r['factor_id'] for r in bad_cls]}")

    frame = pd.DataFrame(rows)
    COV_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(COV_CSV, index=False)

    companions, companion_meta = build_companions(panel, ac, resume=resume)
    pit_rows = []
    for rec in companion_meta.get("features") or []:
        fid = rec["factor_id"]
        pit = _pit_one(fid, companions[fid], family=rec["family"], formula=rec["formula"])
        dest = ROUND21_1_ROOT / "factors" / fid
        dest.mkdir(parents=True, exist_ok=True)
        dump_json(dest / "pit.json", pit)
        dump_json(dest / "coverage.json", {k: pit[k] for k in ("n", "n_valid", "coverage")})
        pit_rows.append(pit)
    pit_df = pd.DataFrame(pit_rows)
    pit_df.to_csv(PIT_CSV, index=False)
    rehabilitated = [r["factor_id"] for r in pit_rows if r["pit_status"] == "PIT_PASS"]

    payload = {
        "status": "DONE",
        "round": "21.1",
        "n_universe": int(len(panel)),
        "min_finite": PIT_MIN_FINITE,
        "filled_with_zero": False,
        "e_family_blocked": list(BLOCKED_PIT_R21),
        "fail_original": list(PIT_FAIL_COVERAGE_R21),
        "pass_original": list(PIT_PASS_R21),
        "rows": rows,
        "companion_pit_pass": rehabilitated,
        "companion_n": len(pit_rows),
        "csv": str(COV_CSV),
        "pit_csv": str(PIT_CSV),
        "r21_panel_readonly": str(PANEL_PATH),
    }
    dump_json(COV_JSON, payload)
    lines = [
        "# Round21.1 · 01 覆盖法医",
        "",
        f"候选池 n=`{len(panel)}`。门槛 `finite_share<{PIT_MIN_FINITE}`。**禁止填 0**。",
        "",
        "分类仅允许：`STRUCTURAL_NOT_APPLICABLE` / `EVENT_CONDITIONAL` / `TRUE_DATA_MISSING` / "
        "`SOURCE_ALIGNMENT_FAILURE` / `FORMULA_UNDEFINED` / `TIMESTAMP_UNVERIFIED`。",
        "",
        "| factor | class | universe cov | cond cov | true miss | note |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['factor_id']}` | {r['classification']} | {r.get('universe_coverage'):.4f} | "
            f"{r.get('conditional_coverage')} | {r.get('true_missing_rate')} | {r.get('note')} |"
        )
    lines += [
        "",
        "## 同伴新表达 PIT",
        "",
        f"PIT_PASS companions=`{rehabilitated}`。原 FAIL 公式不得冒充 PASS。E 族仍禁。",
        "",
    ]
    for r in pit_rows:
        lines.append(
            f"- `{r['factor_id']}` {r['pit_status']} cov=`{r['coverage']:.4f}` eligible=`{r['model_eligible']}`"
        )
    write_md(COVERAGE_MD, "\n".join(lines))
    return payload
