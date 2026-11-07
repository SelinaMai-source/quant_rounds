"""A0–A9 全袖套 Round7 5bp + HISTORICAL_FEE_CORRECTED。必须有真实 Sharpe。"""

from __future__ import annotations

import shutil
from typing import Any

import pandas as pd

from ..paths import ROUND21_4_ROOT
from ..round11.common import daily_from_nav, metrics_from_nav, observed_from_report
from .common import dump_json, finite_or_none, load_json, slim_obs, write_md
from .contract import (
    A_CONTRASTS,
    A_SPECS,
    ECONOMICALLY_IMMATERIAL,
    FEE_SCHEDULE,
    HISTORICAL_FEE_CORRECTED,
    MD,
    REPLICATION_ANCHORS,
    SAME_OPEN_EXECUTION_OPTIMISM,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    WXYZ_FACTOR_IDS,
    exp_dir,
    lineage_root,
    material_vs_v5,
)
from .train import train_one

BT_JSON = ROUND21_4_ROOT / "backtest" / "decision.json"
A_BT_JSON = ROUND21_4_ROOT / "backtest" / "amatrix_decision.json"
BT_CSV = ROUND21_4_ROOT / "portfolio_backtest_results.csv"
YEAR_CSV = ROUND21_4_ROOT / "yearly_results.csv"
CMP_CSV = ROUND21_4_ROOT / "final_comparison.csv"
FEE_CSV = ROUND21_4_ROOT / "backtest" / "fee_both_calibers.csv"
SENS_ROOT = ROUND21_4_ROOT / "backtest" / "fee_sensitivity"
WXYZ_BT_JSON = ROUND21_4_ROOT / "backtest" / "wxyz_fourway.json"


def _metrics_from_dest(dest) -> dict[str, Any]:
    nav_p = dest / "nav.parquet"
    report = load_json(dest / "report.json")
    obs = observed_from_report(report) if report else {}
    if nav_p.is_file():
        nav = pd.read_parquet(nav_p)
        if not nav.empty and "total_asset" in nav.columns:
            m = metrics_from_nav(nav)
            daily = daily_from_nav(nav)
            vol = float(daily.std(ddof=1)) if len(daily) > 2 else None
            sharpe = float(daily.mean() / vol * (252 ** 0.5)) if vol and vol > 0 else None
            obs = {
                **obs,
                "cagr": finite_or_none(m.get("annualized_return") or obs.get("cagr")),
                "sharpe": finite_or_none(sharpe if sharpe is not None else m.get("sharpe_zero_rate") or obs.get("sharpe")),
                "max_drawdown": finite_or_none(m.get("max_drawdown") or obs.get("max_drawdown")),
                "nav_end": finite_or_none(float(nav["total_asset"].iloc[-1])),
                "total_return": finite_or_none(m.get("total_return") or obs.get("total_return")),
                "n_obs": int(len(daily)),
            }
    return obs


def _fee_breakdown(dest) -> dict[str, Any]:
    trades_p = dest / "trades.parquet"
    out = {"commission": None, "stamp_tax": None, "transfer": None, "turnover": None, "n_trades": None}
    if not trades_p.is_file():
        return out
    t = pd.read_parquet(trades_p)
    out["n_trades"] = int(len(t))
    for col, key in (
        ("commission", "commission"),
        ("commission_fee", "commission"),
        ("stamp_tax", "stamp_tax"),
        ("stamp", "stamp_tax"),
        ("transfer_fee", "transfer"),
        ("transfer", "transfer"),
        ("turnover", "turnover"),
        ("amount", "turnover"),
    ):
        if col in t.columns and out.get(key) is None:
            out[key] = finite_or_none(pd.to_numeric(t[col], errors="coerce").sum())
    report = load_json(dest / "report.json") or {}
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    aliases = {
        "commission": ("commission", "commission_fee"),
        "stamp_tax": ("stamp_tax", "stamp"),
        "transfer": ("transfer_fee", "transfer"),
        "turnover": ("turnover", "turnover_value", "amount"),
    }
    for key, names in aliases.items():
        if out.get(key) is not None:
            continue
        for name in names:
            if finite_or_none(metrics.get(name)) is not None:
                out[key] = finite_or_none(metrics.get(name))
                break
            if finite_or_none(report.get(name)) is not None:
                out[key] = finite_or_none(report.get(name))
                break
            fees = report.get("fees") if isinstance(report.get("fees"), dict) else {}
            if finite_or_none(fees.get(name)) is not None:
                out[key] = finite_or_none(fees.get(name))
                break
    costs_p = dest / "costs.parquet"
    if costs_p.is_file():
        c = pd.read_parquet(costs_p)
        for col, key in (("commission", "commission"), ("transfer_fee", "transfer"), ("stamp_tax", "stamp_tax"), ("turnover", "turnover")):
            if out.get(key) is None and col in c.columns:
                out[key] = finite_or_none(pd.to_numeric(c[col], errors="coerce").sum())
    return out


_SLEEVE_NAMES = (
    "nav.parquet",
    "report.json",
    "trades.parquet",
    "cashflows.parquet",
    "costs.parquet",
    "positions.parquet",
    "rejections.parquet",
    "sleeve_states.parquet",
    "overlayed.parquet",
    "overlay_meta.json",
    "predictions.parquet",
)


def _import_sleeve(dest, sources: list) -> bool:
    new_hash = load_json(dest / "prediction_hash.json")
    if not new_hash:
        return False
    for src in sources:
        if not src or not (src / "nav.parquet").is_file() or not (src / "report.json").is_file():
            continue
        old_hash = load_json(src / "prediction_hash.json")
        if old_hash and old_hash != new_hash:
            # 导入预测后重新算的 hash 可能因 parquet 元数据不同；若预测文件字节相同则仍可导入
            pass
        five_ok = (src / "five_pred.parquet").is_file() and file_equalish(src / "five_pred.parquet", dest / "five_pred.parquet")
        ten_ok = (src / "ten_pred.parquet").is_file() and file_equalish(src / "ten_pred.parquet", dest / "ten_pred.parquet")
        if not (five_ok and ten_ok) and old_hash != new_hash:
            continue
        for name in _SLEEVE_NAMES:
            sp = src / name
            if sp.is_file():
                shutil.copy2(sp, dest / name)
        ov = dest / "overlayed.parquet"
        if ov.is_file():
            from .contract import overlayed_path

            op = overlayed_path(dest.name)
            op.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ov, op)
        dump_json(dest / "imported_sleeve.json", {"source": str(src), "reason": "identical predictions; replay sleeve not alias ADD"})
        return True
    return False


def file_equalish(a, b) -> bool:
    from .common import file_sha256

    ha, hb = file_sha256(a), file_sha256(b)
    return bool(ha and hb and ha == hb)


def _sleeve_sources(spec: dict[str, Any], dest):
    out = []
    lin = spec.get("lineage_import")
    if lin:
        out.append(lineage_root(lin[0]) / "candidates" / lin[1])
    ident = spec.get("identity_with")
    if ident:
        out.append(exp_dir(ident))
    imported = load_json(dest / "imported_by_hash.json")
    if imported.get("source"):
        out.append(__import__("pathlib").Path(imported["source"]))
    return out


def run_full_backtest(code: str, spec: dict[str, Any] | None = None, *, resume: bool = True) -> dict[str, Any]:
    dest = exp_dir(code)
    if resume and (dest / "nav.parquet").is_file() and (dest / "summary.json").is_file():
        existing = load_json(dest / "summary.json")
        if finite_or_none(existing.get("sharpe")) is not None:
            existing["resumed"] = True
            return existing
    if spec is None:
        spec = next((s for s in A_SPECS if s["code"] == code), {"code": code})
    if resume and not (dest / "nav.parquet").is_file():
        _import_sleeve(dest, _sleeve_sources(spec, dest))
    five_p, ten_p = dest / "five_pred.parquet", dest / "ten_pred.parquet"
    if not five_p.is_file() or not ten_p.is_file():
        train_one(spec, resume=True)
    if not five_p.is_file() or not ten_p.is_file():
        return {"code": code, "status": "SKIP_NO_OOF"}
    if resume and not (dest / "nav.parquet").is_file():
        _import_sleeve(dest, _sleeve_sources(spec, dest))
    if not ((dest / "nav.parquet").is_file() and (dest / "report.json").is_file()):
        from .engine import run_v5_fused_backtest

        five = pd.read_parquet(five_p)
        ten = pd.read_parquet(ten_p)
        extra = {"code": code, "role": (spec or {}).get("role"), "fee_schedule": FEE_SCHEDULE, "validation_hooks": None, "round": "21.4"}
        run_v5_fused_backtest(dest, five, ten, extra=extra, resume=resume)
    recomputed = _metrics_from_dest(dest)
    fees = _fee_breakdown(dest)
    oof = load_json(dest / "oof.json")
    obs = recomputed
    payload = {
        **obs,
        **{k: recomputed.get(k) if recomputed.get(k) is not None else obs.get(k) for k in ("cagr", "sharpe", "max_drawdown", "nav_end", "total_return")},
        "code": code,
        "role": (spec or {}).get("role"),
        "factor_id": (spec or {}).get("factor_id") or oof.get("factor_id"),
        "five_oof": oof.get("five_oof"),
        "ten_oof": oof.get("ten_oof"),
        "five_drops": oof.get("five_drops"),
        "ten_drops": oof.get("ten_drops"),
        "fee_schedule": FEE_SCHEDULE,
        "fees": fees,
        "looked_at_combo_nav": True,
    }
    if payload.get("sharpe") is None:
        payload["status"] = "BLOCKED"
        payload["reason"] = "sharpe_none_after_recompute"
    dump_json(dest / "summary.json", payload)
    return payload


def _fee_sens(code: str, *, resume: bool = True) -> dict[str, Any]:
    ov = exp_dir(code) / "overlayed.parquet"
    dest = SENS_ROOT / code
    if not ov.is_file():
        return {"code": code, "status": "BLOCKED", "reason": "no_overlayed", "fee": HISTORICAL_FEE_CORRECTED, "sharpe": None}
    leftover = load_json(dest / "report.json") if (dest / "report.json").is_file() else {}
    hooks = leftover.get("validation_hooks")
    broken_hooks = isinstance(hooks, str)
    if (dest / "nav.parquet").is_file():
        recomputed = _metrics_from_dest(dest)
        if finite_or_none(recomputed.get("sharpe")) is not None:
            fees = _fee_breakdown(dest)
            report = leftover or load_json(dest / "report.json")
            obs = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
            payload = {
                "code": code,
                "status": "ran",
                "fee": HISTORICAL_FEE_CORRECTED,
                "cagr": recomputed.get("cagr") or obs.get("annualized_return"),
                "sharpe": recomputed.get("sharpe"),
                "max_drawdown": recomputed.get("max_drawdown") or obs.get("max_drawdown"),
                "completed_trades": obs.get("completed_trades") or leftover.get("completed_trades"),
                "nav_end": recomputed.get("nav_end"),
                "total_return": recomputed.get("total_return") or obs.get("total_return"),
                "fees": fees,
                "same_open_execution": SAME_OPEN_EXECUTION_OPTIMISM,
            }
            dump_json(dest / "fee_summary.json", payload)
            return payload
    if broken_hooks or ((dest / "report.json").is_file() and not (dest / "nav.parquet").is_file()):
        resume = False
    from ..round17.backtest import run_canonical

    extra = {
        "code": code,
        "fee_schedule": HISTORICAL_FEE_CORRECTED,
        "historical_stamp_tax": True,
        "round": "21.4",
        "same_open_execution": SAME_OPEN_EXECUTION_OPTIMISM,
    }
    ran = run_canonical(dest, pd.read_parquet(ov), extra=extra, resume=resume)
    obs = ran.get("observed") or {}
    if not obs:
        obs = _metrics_from_dest(dest)
    recomputed = _metrics_from_dest(dest)
    fees = _fee_breakdown(dest)
    sharpe = recomputed.get("sharpe") if recomputed.get("sharpe") is not None else finite_or_none(obs.get("sharpe"))
    payload = {
        "code": code,
        "status": "ran" if sharpe is not None else "BLOCKED",
        "reason": None if sharpe is not None else "sharpe_none",
        "fee": HISTORICAL_FEE_CORRECTED,
        "cagr": recomputed.get("cagr") or obs.get("cagr"),
        "sharpe": sharpe,
        "max_drawdown": recomputed.get("max_drawdown") or obs.get("max_drawdown"),
        "completed_trades": obs.get("completed_trades") or recomputed.get("completed_trades") or (load_json(dest / "report.json").get("metrics") or {}).get("completed_trades"),
        "nav_end": recomputed.get("nav_end") or obs.get("nav_end"),
        "total_return": recomputed.get("total_return") or obs.get("total_return"),
        "fees": fees,
        "same_open_execution": SAME_OPEN_EXECUTION_OPTIMISM,
    }
    dump_json(dest / "fee_summary.json", payload)
    dump_json(dest / "summary.json", payload)
    return payload


def _trade_keys(code: str) -> set:
    p = exp_dir(code) / "trades.parquet"
    if not p.is_file():
        return set()
    t = pd.read_parquet(p, columns=["entry_date", "instrument_id"])
    return set(zip(t["entry_date"].astype(str), t["instrument_id"].astype(str)))


def _pack_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    v5_sh = float(V5_ANCHOR["sharpe"])
    v5_cg = float(V5_ANCHOR["cagr"])
    v5_dd = float(V5_ANCHOR["max_drawdown"])
    # 冻结门 Top1/Top2 变化测融合口径：与 A0(=v5) 实际成交 (entry_date, instrument) 集合差异
    a0_keys = _trade_keys("R21_4_A0")
    main = []
    for r in rows:
        keys = _trade_keys(str(r.get("code")))
        top_changed = bool(a0_keys and keys and (1.0 - len(keys & a0_keys) / len(keys)) > 0)
        mat = material_vs_v5(r, top_changed=top_changed)
        ds = None if finite_or_none(r.get("sharpe")) is None else float(r["sharpe"]) - v5_sh
        if ds is not None and abs(ds) < 0.01 and abs(float(r.get("nav_end") or 0) - V5_NAV_END_EXACT) < 100:
            mat = {**mat, "economically_material": False, "predeclared": ECONOMICALLY_IMMATERIAL}
        main.append(
            {
                "code": r.get("code"),
                "role": r.get("role"),
                "factor_id": r.get("factor_id"),
                "cagr": r.get("cagr"),
                "sharpe": r.get("sharpe"),
                "max_drawdown": r.get("max_drawdown"),
                "completed_trades": r.get("completed_trades"),
                "nav_end": r.get("nav_end"),
                "total_return": r.get("total_return"),
                "delta_sharpe": ds,
                "delta_cagr": None if finite_or_none(r.get("cagr")) is None else float(r["cagr"]) - v5_cg,
                "delta_maxdd": None if finite_or_none(r.get("max_drawdown")) is None else float(r["max_drawdown"]) - v5_dd,
                "fee_schedule": FEE_SCHEDULE,
                "material": mat,
            }
        )
    return main


def _replication_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r.get("code"): r for r in rows}
    out = {}
    for code, anchor in REPLICATION_ANCHORS.items():
        r = by.get(code) or {}
        nav_ok = finite_or_none(r.get("nav_end")) is not None and abs(float(r["nav_end"]) - float(anchor["nav_end"])) < 1.0
        sh_ok = finite_or_none(r.get("sharpe")) is not None and abs(float(r["sharpe"]) - float(anchor["sharpe"])) < 1e-4
        out[code] = {
            "source": anchor["source"],
            "nav_ok": bool(nav_ok),
            "sharpe_ok": bool(sh_ok),
            "passed": bool(nav_ok and sh_ok),
            "observed_nav": r.get("nav_end"),
            "anchor_nav": anchor["nav_end"],
            "observed_sharpe": r.get("sharpe"),
            "anchor_sharpe": anchor["sharpe"],
            "delta_nav_vs_v5_anchor": anchor.get("delta_nav_vs_v5"),
        }
    return out


def run_amatrix_backtest(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and A_BT_JSON.is_file() and MD["07"].is_file():
        existing = load_json(A_BT_JSON)
        if existing.get("status") == "DONE" and int(existing.get("n_full_backtests") or 0) >= len(A_SPECS):
            return existing
    specs = {s["code"]: s for s in A_SPECS}
    rows = []
    for spec in A_SPECS:
        print(f"[round21.4] A BACKTEST {spec['code']}", flush=True)
        rows.append(run_full_backtest(spec["code"], spec, resume=resume))
    fee_rows = [_fee_sens(spec["code"], resume=resume) for spec in A_SPECS]
    main_rows = _pack_rows(rows)
    year_rows = []
    for r in rows:
        for y, val in (r.get("yearly_returns") or {}).items():
            year_rows.append({"code": r.get("code"), "year": y, "return": val, "track": "A"})
    pd.DataFrame(main_rows).to_csv(ROUND21_4_ROOT / "backtest" / "a_matrix.csv", index=False)
    pd.DataFrame(fee_rows).to_csv(ROUND21_4_ROOT / "backtest" / "a_fee_corrected.csv", index=False)
    if year_rows:
        pd.DataFrame(year_rows).to_csv(YEAR_CSV, index=False)
    contrasts = {}
    by = {r["code"]: r for r in rows}
    for a, b, name in A_CONTRASTS:
        ra, rb = by.get(a) or {}, by.get(b) or {}
        contrasts[name] = {
            "a": a,
            "b": b,
            "delta_sharpe": None if ra.get("sharpe") is None or rb.get("sharpe") is None else float(ra["sharpe"]) - float(rb["sharpe"]),
            "delta_cagr": None if ra.get("cagr") is None or rb.get("cagr") is None else float(ra["cagr"]) - float(rb["cagr"]),
            "delta_nav": None if ra.get("nav_end") is None or rb.get("nav_end") is None else float(ra["nav_end"]) - float(rb["nav_end"]),
            "delta_maxdd": None if ra.get("max_drawdown") is None or rb.get("max_drawdown") is None else float(ra["max_drawdown"]) - float(rb["max_drawdown"]),
        }
    source = "UNIDENTIFIABLE"
    d_drop = abs((contrasts.get("A1-A0_drop_g0_five") or {}).get("delta_nav") or 0)
    d_l1 = abs((contrasts.get("A3-A1_l1_after_drop") or {}).get("delta_nav") or 0)
    d_ten = abs((contrasts.get("A4-A0_add_l1_ten") or {}).get("delta_nav") or 0)
    # 43 元 = A3 − A0。A3 与 A0 共享同一 Ten 头（feature_list hash 已证），
    # 故 +43 只能来自 Five 头（10% 融合权重）：
    # A3−A0 = (A1−A0) + (A3−A1) = drop_g0(−145万) + 加回g0且加L1(+145万) 的对消残差。
    # d_ten（Ten+L1 −1024万）解释的是 21.3 E3 −940万，与 43 元无关，不得混入来源标签。
    if d_drop > 1e6 and d_l1 > 1e6 and abs(d_drop - d_l1) <= 0.05 * max(d_drop, d_l1):
        source = "FIVE_10PCT_OFFSET_RECOMPOSITION_RESIDUAL"
    elif d_drop > d_l1 * 2 and d_drop > 100:
        source = "MOSTLY_DROP_G0_AMOUNT_LOG"
    elif d_l1 > d_drop * 2 and d_l1 > 100:
        source = "MOSTLY_ADD_L1_FIVE"
    else:
        source = "FIVE_10PCT_NOISE"
    a3_nav = (by.get("R21_4_A3") or {}).get("nav_end")
    a0_nav = (by.get("R21_4_A0") or {}).get("nav_end")
    forty_three = None if a3_nav is None or a0_nav is None else float(a3_nav) - float(a0_nav)
    repl = _replication_check(rows)
    fee_dir = True
    a0f = next((x for x in fee_rows if x.get("code") == "R21_4_A0"), {})
    a6f = next((x for x in fee_rows if x.get("code") == "R21_4_A6"), {})
    if a0f.get("sharpe") is not None and a6f.get("sharpe") is not None:
        main_sign = 0
        if (by.get("R21_4_A6") or {}).get("sharpe") is not None:
            main_sign = 1 if float(by["R21_4_A6"]["sharpe"]) > float((by.get("R21_4_A0") or {}).get("sharpe") or 0) else -1
        fee_sign = 1 if float(a6f["sharpe"]) > float(a0f["sharpe"]) else -1
        fee_dir = main_sign == 0 or fee_sign == main_sign
    best = max((r for r in main_rows if finite_or_none(r.get("sharpe")) is not None), key=lambda r: float(r["sharpe"]), default={})
    payload = {
        "status": "DONE",
        "n_full_backtests": len(rows),
        "main_rows": main_rows,
        "fee_rows": fee_rows,
        "contrasts": contrasts,
        "forty_three_rmb": forty_three,
        "forty_three_source": source,
        "ten_l1_effect_nav": (contrasts.get("A4-A0_add_l1_ten") or {}).get("delta_nav"),
        "ten_l1_effect_note": "A4−A0 的 Ten+L1 效应解释 21.3 E3 −940万，与 43 元（A3−A0，Five 头）无关",
        "replication": repl,
        "fee_direction_holds": fee_dir,
        "mixed_into_main": False,
        "same_open_execution": SAME_OPEN_EXECUTION_OPTIMISM,
        "best_vs_v5": best,
        "economically_material": bool((best.get("material") or {}).get("economically_material")),
        "a3_ok": (repl.get("R21_4_A3") or {}).get("passed"),
        "a6_ok": (repl.get("R21_4_A6") or {}).get("passed"),
    }
    dump_json(A_BT_JSON, payload)
    dump_json(BT_JSON, payload)
    pd.DataFrame(main_rows).to_csv(BT_CSV, index=False)
    pd.DataFrame(main_rows).to_csv(CMP_CSV, index=False)
    pd.DataFrame(fee_rows).to_csv(FEE_CSV, index=False)
    write_md(
        MD["07"],
        "\n".join(
            [
                "# Round21.4 · 07 L1 A 矩阵全引擎",
                "",
                f"A0–A9 全袖套。口径=`{FEE_SCHEDULE}`。43元=`{forty_three}` 来源=`{source}`。",
                f"A3 复现=`{(repl.get('R21_4_A3') or {}).get('passed')}` A6 复现=`{(repl.get('R21_4_A6') or {}).get('passed')}`",
                f"对照：{contrasts}",
                f"费用校正 N=`{len(fee_rows)}` direction_holds=`{fee_dir}`。",
                "",
            ]
        ),
    )
    return payload


def _wxyz_specs(factor_id: str) -> list[dict[str, Any]]:
    """Head-specific 四路：E0 复用 A0；Five-only add；Ten-only add；Both add。"""
    return [
        {
            "code": f"R21_4_W_{factor_id}_five",
            "role": "HEAD_FIVE_ADD",
            "factor_id": factor_id,
            "five_extras": (factor_id,),
            "ten_extras": (),
            "five_drops": (),
            "ten_drops": (),
            "formal": True,
            "head": "both_90_10",
            "alias_from": None,
            "force_retrain": True,
        },
        {
            "code": f"R21_4_W_{factor_id}_ten",
            "role": "HEAD_TEN_ADD",
            "factor_id": factor_id,
            "five_extras": (),
            "ten_extras": (factor_id,),
            "five_drops": (),
            "ten_drops": (),
            "formal": True,
            "head": "both_90_10",
            "alias_from": None,
            "force_retrain": True,
        },
        {
            "code": f"R21_4_W_{factor_id}_both",
            "role": "HEAD_BOTH_ADD",
            "factor_id": factor_id,
            "five_extras": (factor_id,),
            "ten_extras": (factor_id,),
            "five_drops": (),
            "ten_drops": (),
            "formal": True,
            "head": "both_90_10",
            "alias_from": None,
            "force_retrain": True,
        },
        {
            "code": f"R21_4_W_{factor_id}_replace",
            "role": "HEAD_BOTH_REPLACE_NEAREST",
            "factor_id": factor_id,
            "five_extras": (factor_id,),
            "ten_extras": (factor_id,),
            "five_drops": (),
            "ten_drops": (),
            "formal": True,
            "head": "both_90_10",
            "alias_from": None,
            "force_retrain": True,
            "note": "nearest drop filled after correlate",
        },
    ]


def run_wxyz_fourway(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    if resume and WXYZ_BT_JSON.is_file() and MD["24"].is_file():
        existing = load_json(WXYZ_BT_JSON)
        if existing.get("status") in {"DONE", "PARTIAL"}:
            return existing
    from .pit import PIT_JSON

    pit = load_json(PIT_JSON)
    pass_ids = list(pit.get("pit_pass") or [])
    # 资源：每族取 |RankIC| 最高 1 个，最多 4 个因子 × 四路
    ic_csv = ROUND21_4_ROOT / "new_factor_IC.csv"
    if ic_csv.is_file() and pass_ids:
        ic = pd.read_csv(ic_csv)
        ic = ic.loc[ic["factor_id"].astype(str).isin(pass_ids)]
        ic["abs_ic"] = pd.to_numeric(ic.get("rankic_mean"), errors="coerce").abs()
        picked = []
        for fam in ("w", "x", "y", "z"):
            sub = ic.loc[ic["factor_id"].astype(str).str.startswith(f"r21_{fam}")]
            if sub.empty:
                continue
            best = sub.sort_values("abs_ic", ascending=False).iloc[0]["factor_id"]
            picked.append(str(best))
        if picked:
            pass_ids = list(dict.fromkeys(picked))
    if not pass_ids:
        payload = {"status": "PARTIAL", "reason": "no PIT_PASS factors", "n_full_backtests": 0, "rows": []}
        dump_json(WXYZ_BT_JSON, payload)
        write_md(MD["24"], "# Round21.4 · 24 WXYZ head-specific 四路\n\n无 PIT_PASS，未开全袖套。\n")
        return payload
    from .correlate import CORR_JSON

    corr = load_json(CORR_JSON)
    nearest = corr.get("nearest") or {}
    from ..round7.contract import FIVE_FEATURES, TEN_FEATURES

    specs = []
    for fid in pass_ids:
        for spec in _wxyz_specs(fid):
            if spec["role"] == "HEAD_BOTH_REPLACE_NEAREST":
                nb = nearest.get(fid) or {}
                majority = nb.get("majority_legacy")
                if majority in FIVE_FEATURES:
                    spec["five_drops"] = (majority,)
                if majority in TEN_FEATURES:
                    spec["ten_drops"] = (majority,)
            specs.append(spec)
    rows = []
    for spec in specs:
        print(f"[round21.4] WXYZ TRAIN+BT {spec['code']}", flush=True)
        try:
            train_one(spec, resume=resume, n_jobs=1 if jobs > 1 else 2)
            rows.append(run_full_backtest(spec["code"], spec, resume=resume))
        except Exception as exc:
            rows.append({"code": spec["code"], "status": "BLOCKED", "reason": str(exc), "factor_id": spec.get("factor_id")})
    main = _pack_rows([r for r in rows if finite_or_none(r.get("sharpe")) is not None])
    payload = {
        "status": "DONE" if main else "PARTIAL",
        "n_full_backtests": len(main),
        "n_attempted": len(rows),
        "pit_pass": pass_ids,
        "main_rows": main,
        "raw_rows": rows,
    }
    dump_json(WXYZ_BT_JSON, payload)
    write_md(MD["24"], f"# Round21.4 · 24 WXYZ head-specific 四路\n\nPIT_PASS=`{pass_ids}` 全引擎 N=`{len(main)}` attempted=`{len(rows)}`\n")
    return payload
