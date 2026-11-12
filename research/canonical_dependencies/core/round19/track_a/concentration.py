"""a-concentration：39 次 leave-one-event-out 全引擎 + leave-top3 POST_HOC。"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from ...paths import ROUND19_ROOT, assert_project_result_path
from ...round12.runner import hist_hooks, run_r12_backtest
from ...round15.runner import apply_top2_veto
from ...round16_5.common import nav_end
from ...round17.backtest import cached_market, cached_overlay_v4
from ..common import daily_from_nav, dump_json, load_json, write_md
from ..contract import POST_HOC_STRESS_DIAGNOSTIC
from .engine import o01_signal_dates
from .support import (
    CONC_CKPT,
    CONC_JSON,
    EVENTS_JSON,
    load_nav_trades,
    pack_daily,
    require_c0,
    require_c1,
    resume_ok,
    skip_result,
    ta_md,
)

_G: dict[str, Any] = {}


def _metrics(nav: pd.DataFrame) -> dict[str, Any]:
    daily = daily_from_nav(nav)
    pack = pack_daily(daily)
    pack["nav_end"] = nav_end(nav)
    return pack


def _one_veto(dropped: list[str]) -> dict[str, Any]:
    base = pd.DatetimeIndex(pd.to_datetime(_G["o01_dates"])).normalize()
    drop = pd.DatetimeIndex(pd.to_datetime(dropped)).normalize()
    keep = base.difference(drop)
    ov = apply_top2_veto(_G["overlayed"], keep)
    ran = run_r12_backtest(
        ROUND19_ROOT / "track_a" / "concentration" / "_mem",
        prepared_market=_G["market"],
        market_meta=_G["meta"],
        overlayed=ov,
        validation_hooks=hist_hooks(),
        extra_report={"role": "loo_full_engine", "dropped": dropped},
        resume=False,
        write=False,
    )
    pack = _metrics(ran["nav"])
    pack["dropped"] = dropped
    pack["n_dropped"] = int(len(drop))
    pack["status"] = "ok"
    return pack


def _load_ckpt() -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not CONC_CKPT.is_file():
        return done
    for line in CONC_CKPT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = ",".join(rec.get("dropped") or [])
        if rec.get("status") == "ok":
            done[key] = rec
    return done


def run_concentration(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    existing = load_json(CONC_JSON) if resume else {}
    if (
        existing.get("status") == "DONE"
        and int(existing.get("n_loo_ok") or 0) >= 39
        and (existing.get("leave_top3") or {}).get("dates")
    ):
        return existing
    events = load_json(EVENTS_JSON)
    fill_days = events.get("fill_change_dates") or []
    if len(fill_days) != 39:
        if not fill_days:
            payload = skip_result("a-concentration", "FILL_CHANGE_DATES_MISSING", status="BLOCKED")
            dump_json(CONC_JSON, payload)
            return payload
    try:
        c0 = require_c0()
        c1 = require_c1()
        _, n0 = load_nav_trades(c0)
        _, n1 = load_nav_trades(c1)
        o01_dates = o01_signal_dates()
    except FileNotFoundError as exc:
        payload = skip_result("a-concentration", str(exc), status="BLOCKED")
        dump_json(CONC_JSON, payload)
        return payload
    pack0 = pack_daily(daily_from_nav(n0))
    pack1 = pack_daily(daily_from_nav(n1))
    end0 = nav_end(n0)
    end1 = nav_end(n1)
    inc_nav = None if (end0 is None or end1 is None) else float(end1 - end0)
    inc_cagr = (pack1.get("cagr") or 0) - (pack0.get("cagr") or 0)
    inc_sh = (pack1.get("sharpe") or 0) - (pack0.get("sharpe") or 0)

    _G["overlayed"] = cached_overlay_v4().reset_index(drop=True)
    loaded = cached_market()
    _G["market"] = loaded["prepared"]
    _G["meta"] = loaded
    _G["o01_dates"] = list(pd.DatetimeIndex(o01_dates).normalize())

    keys = [[d] for d in fill_days]
    done = _load_ckpt() if resume else {}
    todo = [k for k in keys if ",".join(k) not in done]
    CONC_CKPT.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(CONC_CKPT)
    if todo:
        n_jobs = max(int(jobs), 1)
        if n_jobs == 1:
            with CONC_CKPT.open("a", encoding="utf-8") as fh:
                for dropped in todo:
                    rec = _one_veto(dropped)
                    done[",".join(dropped)] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    fh.flush()
        else:
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
                futs = {pool.submit(_one_veto, dropped): dropped for dropped in todo}
                with CONC_CKPT.open("a", encoding="utf-8") as fh:
                    for fut in as_completed(futs):
                        dropped = futs[fut]
                        try:
                            rec = fut.result()
                        except Exception as exc:
                            rec = {
                                "dropped": dropped,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        done[",".join(dropped)] = rec
                        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        fh.flush()

    loo_rows = []
    for d in fill_days:
        rec = done.get(d) or {}
        if rec.get("status") != "ok":
            continue
        nav_i = rec.get("nav_end")
        contrib = None if (end1 is None or nav_i is None) else float(end1) - float(nav_i)
        loo_rows.append(
            {
                "signal_date": d,
                "cagr": rec.get("cagr"),
                "sharpe": rec.get("sharpe"),
                "nav_end": nav_i,
                "cagr_increment_vs_c0": (rec.get("cagr") or 0) - (pack0.get("cagr") or 0),
                "sharpe_increment_vs_c0": (rec.get("sharpe") or 0) - (pack0.get("sharpe") or 0),
                "nav_contrib_vs_c1": contrib,
            }
        )
    loo_df = pd.DataFrame(loo_rows)
    n_flip = 0
    if not loo_df.empty and inc_cagr > 0:
        n_flip = int((loo_df["cagr_increment_vs_c0"] <= 0).sum())
    ranked = []
    if not loo_df.empty:
        ranked = loo_df.sort_values("nav_contrib_vs_c1", ascending=False, na_position="last")
        top3 = [str(x) for x in ranked.head(3)["signal_date"].tolist()]
    else:
        top3 = []

    top3_rec = None
    if top3:
        key = ",".join(top3)
        if resume and key in done and done[key].get("status") == "ok":
            top3_rec = done[key]
        else:
            top3_rec = _one_veto(top3)
            with CONC_CKPT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(top3_rec, ensure_ascii=False, default=str) + "\n")
    remaining_after_top3 = None
    if top3_rec and top3_rec.get("status") == "ok":
        remaining_after_top3 = {
            "cagr_increment_vs_c0": (top3_rec.get("cagr") or 0) - (pack0.get("cagr") or 0),
            "sharpe_increment_vs_c0": (top3_rec.get("sharpe") or 0) - (pack0.get("sharpe") or 0),
            "nav_end": top3_rec.get("nav_end"),
            "nav_increment_vs_c0": None
            if (end0 is None or top3_rec.get("nav_end") is None)
            else float(top3_rec["nav_end"]) - float(end0),
        }
    depends_few = bool(n_flip > 0 or (remaining_after_top3 is not None and (remaining_after_top3.get("cagr_increment_vs_c0") or 0) <= 0))
    if not loo_df.empty and inc_nav:
        top1_share = None
        top1 = float(ranked.iloc[0]["nav_contrib_vs_c1"] or 0) if len(ranked) else None
        if top1 is not None and abs(inc_nav) > 1e-9:
            top1_share = float(top1 / inc_nav)
            if abs(top1_share) > 0.5:
                depends_few = True
    else:
        top1_share = None

    payload = {
        "status": "DONE",
        "n_events": int(len(fill_days)),
        "n_loo_ok": int(len(loo_rows)),
        "c0": pack0,
        "c1": pack1,
        "nav_end_difference": inc_nav,
        "cagr_increment": inc_cagr,
        "sharpe_increment": inc_sh,
        "n_loo_sign_flip": int(n_flip),
        "top1_nav_share_of_increment": top1_share,
        "depends_on_few_extremes": depends_few,
        "loo": loo_rows,
        "leave_top3": {
            "label": POST_HOC_STRESS_DIAGNOSTIC,
            "dates": top3,
            "result": remaining_after_top3,
        },
        "note": "LOO 从 O01 全信号 veto 集去掉该成交变化日，全引擎重放。leave-top3 按 LOO NAV 贡献事后挑选，标 POST_HOC。",
    }
    dump_json(CONC_JSON, payload)
    write_md(
        ta_md("concentration.md"),
        "\n".join(
            [
                "# Round19 Track A · a-concentration",
                "",
                f"N_events=`{len(fill_days)}` LOO_ok=`{len(loo_rows)}` sign_flip=`{n_flip}` depends_few=`{depends_few}`。",
                f"leave-top3 `{POST_HOC_STRESS_DIAGNOSTIC}` dates=`{top3}` remaining CAGR inc=`{(remaining_after_top3 or {}).get('cagr_increment_vs_c0')}`。",
                "",
            ]
        ),
    )
    return payload
