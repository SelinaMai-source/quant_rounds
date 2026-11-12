"""冻结 registry 的训练 + v5 回测。禁止事后加组。checkpoint/resume。"""

from __future__ import annotations

import traceback
from typing import Any

import pandas as pd

from ..paths import ROUND20_ROOT, ROUND7_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from .common import dump_json, load_json, slim_obs, write_md
from .contract import (
    COLSAMPLE,
    COMBO_TRAIN_MD,
    F2_NAME,
    FEE_SCHEDULE,
    SKIP_DEPENDENCY,
    SUBSAMPLE,
    XGB_N_JOBS,
    exp_dir,
)
from .engine import copy_a00_from_parity, run_v5_fused_backtest
from .freeze import require_frozen
from .train import new_columns, train_both_heads_family, unique_keep

DATA = ROUND20_ROOT / "data"
KEYS = ["trade_date", "instrument_id"]
COMBO_TRAIN_JSON = ROUND20_ROOT / "combo" / "combo_train.json"
COMBO_RES_CSV = ROUND20_ROOT / "combo" / "combo_results.csv"
_G: dict[str, Any] = {}


def prepare_panels(*, resume: bool = True) -> dict[str, Any]:
    paths = {
        "five_training": DATA / "five_training.parquet",
        "five_candidate": DATA / "five_candidate.parquet",
        "ten_training": DATA / "ten_training.parquet",
        "ten_candidate": DATA / "ten_candidate.parquet",
        "five_frozen": DATA / "five_frozen.parquet",
        "ten_frozen": DATA / "ten_frozen.parquet",
    }
    if resume and all(p.is_file() for p in paths.values()):
        return {"status": "resumed", "paths": {k: str(v) for k, v in paths.items()}}
    from ..round17.factors import load_new_factors
    from ..round18.factors import PANEL as R18_PANEL

    r17 = load_new_factors()
    r18 = pd.read_parquet(R18_PANEL) if R18_PANEL.is_file() else None
    if r18 is not None:
        r18["trade_date"] = pd.to_datetime(r18["trade_date"])
        r18["instrument_id"] = r18["instrument_id"].astype(str)
    DATA.mkdir(parents=True, exist_ok=True)
    for head, tkey, ckey in (
        (FIVE_HEAD, "five_training", "five_candidate"),
        (TEN_HEAD, "ten_training", "ten_candidate"),
    ):
        training = pd.read_parquet(head.panel_dir / "training_panel.parquet")
        candidates = pd.read_parquet(head.panel_dir / "candidate_panel.parquet")
        training["trade_date"] = pd.to_datetime(training["trade_date"])
        candidates["trade_date"] = pd.to_datetime(candidates["trade_date"])
        training["instrument_id"] = training["instrument_id"].astype(str)
        candidates["instrument_id"] = candidates["instrument_id"].astype(str)
        extra_r17 = [c for c in r17.columns if c not in KEYS and c not in training.columns]
        if extra_r17:
            training = training.merge(r17[KEYS + extra_r17], on=KEYS, how="left")
            extra_c = [c for c in extra_r17 if c not in candidates.columns]
            if extra_c:
                candidates = candidates.merge(r17[KEYS + extra_c], on=KEYS, how="left")
        if r18 is not None:
            extra_r18 = [c for c in r18.columns if c not in KEYS and c not in training.columns]
            if extra_r18:
                training = training.merge(r18[KEYS + extra_r18], on=KEYS, how="left")
                extra_c = [c for c in extra_r18 if c not in candidates.columns]
                if extra_c:
                    candidates = candidates.merge(r18[KEYS + extra_c], on=KEYS, how="left")
        training.to_parquet(paths[tkey], index=False)
        candidates.to_parquet(paths[ckey], index=False)
    pd.read_parquet(ROUND7_ROOT / "models" / "five" / "predictions.parquet").to_parquet(paths["five_frozen"], index=False)
    pd.read_parquet(ROUND7_ROOT / "models" / "ten" / "predictions.parquet").to_parquet(paths["ten_frozen"], index=False)
    meta = {
        "status": "prepared",
        "paths": {k: str(v) for k, v in paths.items()},
        "filled_with_zero": False,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
    }
    dump_json(DATA / "panels.json", meta)
    return meta


def init_panels() -> None:
    if _G:
        return
    prepare_panels(resume=True)
    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate", "five_frozen", "ten_frozen"):
        path = DATA / f"{key}.parquet"
        frame = pd.read_parquet(path)
        if "trade_date" in frame.columns:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        if "instrument_id" in frame.columns:
            frame["instrument_id"] = frame["instrument_id"].astype(str)
        if "signal_date" in frame.columns:
            frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        _G[key] = frame


def parse_extras(raw: Any) -> tuple[str, ...]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(unique_keep([str(x) for x in raw if str(x) and str(x) != "nan"]))
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return ()
    return tuple(unique_keep([p.strip() for p in text.replace(",", "|").split("|") if p.strip()]))


def _done(dest) -> dict[str, Any] | None:
    path = dest / "summary.json"
    if not path.is_file():
        return None
    payload = load_json(path)
    status = str(payload.get("status") or "")
    if status.startswith("SKIP") or status in {SKIP_DEPENDENCY, "PASS", "ran", "resumed", "copied", "aliased"}:
        return payload
    if (dest / "nav.parquet").is_file() and payload.get("sharpe") is not None:
        return payload
    return None


def skip_payload(code: str, spec: dict[str, Any], reason: str, status: str) -> dict[str, Any]:
    dest = exp_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    obs = {
        "code": code,
        "role": spec.get("role"),
        "status": status,
        "reason": reason,
        "formal": bool(spec.get("formal", True)),
        "model": spec.get("model") or "M0",
        "five_extras": list(spec.get("five_extras") or spec.get("extras") or ()),
        "ten_extras": list(spec.get("ten_extras") or spec.get("extras") or ()),
        "cagr": None,
        "sharpe": None,
        "max_drawdown": None,
        "completed_trades": None,
        "nav_end": None,
        "fee_schedule": FEE_SCHEDULE,
        "out_dir": str(dest),
    }
    dump_json(dest / "summary.json", obs)
    return obs


def run_one(spec: dict[str, Any], *, resume: bool = True, n_jobs: int = XGB_N_JOBS) -> dict[str, Any]:
    code = str(spec["code"])
    dest = exp_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    if resume:
        existing = _done(dest)
        if existing is not None:
            existing["resumed"] = True
            return existing
    if spec.get("role") == "exact_v5_replay" or code == "R20_A00":
        return copy_a00_from_parity(dest)
    alias = spec.get("alias_of")
    if alias:
        src = exp_dir(str(alias))
        src_sum = load_json(src / "summary.json") if (src / "summary.json").is_file() else {}
        if src_sum and (src_sum.get("sharpe") is not None or str(src_sum.get("status") or "").startswith("SKIP")):
            payload = dict(src_sum)
            payload["code"] = code
            payload["role"] = spec.get("role")
            payload["alias_of"] = alias
            payload["status"] = src_sum.get("status") if str(src_sum.get("status") or "").startswith("SKIP") else "aliased"
            payload["formal"] = bool(spec.get("formal", True))
            payload["note"] = f"selection-rule representative aliased from frozen registry {alias}; not independent OOS"
            dump_json(dest / "summary.json", payload)
            return payload
    extras = parse_extras(spec.get("extras"))
    five_extras = parse_extras(spec.get("five_extras") if spec.get("five_extras") is not None else extras)
    ten_extras = parse_extras(spec.get("ten_extras") if spec.get("ten_extras") is not None else extras)
    if F2_NAME in five_extras or F2_NAME in ten_extras:
        return skip_payload(code, spec, "F2 is PIT_TIMESTAMP_UNVERIFIED; not promotion-eligible", "SKIP_F2")
    force_retrain = bool(spec.get("force_retrain"))
    five_new = new_columns(FIVE_FEATURES, five_extras)
    ten_new = new_columns(TEN_FEATURES, ten_extras)
    if not five_new and not ten_new and not force_retrain:
        return skip_payload(
            code,
            spec,
            "extras already inside locked 14/13; no new columns for either head",
            "SKIP_NO_NEW_FEATURES",
        )
    init_panels()
    missing = []
    for name in unique_keep([*five_new, *ten_new]):
        in_five = name in _G["five_training"].columns and name in _G["five_candidate"].columns
        in_ten = name in _G["ten_training"].columns and name in _G["ten_candidate"].columns
        if name in five_new and not in_five:
            missing.append(f"five:{name}")
        if name in ten_new and not in_ten:
            missing.append(f"ten:{name}")
    if missing:
        return skip_payload(code, spec, "missing columns: " + ",".join(missing), "SKIP_MISSING_COLUMN")
    family = str(spec.get("model") or "M0")
    try:
        trained = train_both_heads_family(
            family=family,
            five_training=_G["five_training"],
            five_candidates=_G["five_candidate"],
            ten_training=_G["ten_training"],
            ten_candidates=_G["ten_candidate"],
            five_extras=five_extras,
            ten_extras=ten_extras,
            output_dir=dest,
            resume=resume,
            five_frozen=_G["five_frozen"],
            ten_frozen=_G["ten_frozen"],
            force_retrain=force_retrain,
            n_jobs=n_jobs,
        )
    except Exception as exc:
        return skip_payload(code, spec, f"{type(exc).__name__}: {exc}", "SKIP_TRAIN_ERROR")
    if trained.get("status") == SKIP_DEPENDENCY:
        payload = skip_payload(code, spec, trained.get("reason") or SKIP_DEPENDENCY, SKIP_DEPENDENCY)
        payload["probe"] = trained.get("probe")
        payload["model_family"] = trained.get("model_family")
        dump_json(dest / "summary.json", payload)
        return payload
    five_pred = trained["five_pred"]
    ten_pred = trained["ten_pred"]
    extra = {
        "code": code,
        "role": spec.get("role"),
        "five_extras": list(five_extras),
        "ten_extras": list(ten_extras),
        "model": family,
        "formal": bool(spec.get("formal", True)),
        "force_retrain": force_retrain,
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
    }
    obs = run_v5_fused_backtest(dest, five_pred, ten_pred, extra=extra, resume=resume)
    slim = slim_obs(obs)
    payload = {
        **slim,
        "code": code,
        "role": spec.get("role"),
        "formal": bool(spec.get("formal", True)),
        "model": family,
        "model_family": trained.get("model_family") or family,
        "force_retrain": force_retrain,
        "five_extras": list(five_extras),
        "ten_extras": list(ten_extras),
        "five_new": trained.get("five_new"),
        "ten_new": trained.get("ten_new"),
        "five": trained.get("five"),
        "ten": trained.get("ten"),
        "yearly_returns": obs.get("yearly_returns") or {},
        "nav_end": obs.get("nav_end"),
        "out_dir": str(dest),
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "filled_with_zero": False,
        "status": obs.get("status") or "ran",
    }
    if family.upper() in {"M1", "RIDGE", "ELASTICNET"}:
        payload["loss_ne_ndcg"] = True
        payload["loss"] = "mse_on_within_day_label_rank_pct"
        payload["note"] = "M1 Ridge/ElasticNet rank proxy; loss ≠ NDCG"
    dump_json(dest / "summary.json", payload)
    return payload


def registry_specs() -> list[dict[str, Any]]:
    from .combo import REG_CSV

    frame = pd.read_csv(REG_CSV)
    specs = []
    for rec in frame.to_dict(orient="records"):
        extras = parse_extras(rec.get("extras"))
        specs.append({
            "code": rec["code"],
            "role": rec.get("role"),
            "extras": extras,
            "five_extras": extras,
            "ten_extras": extras,
            "formal": True,
            "model": "M0",
            "force_retrain": False,
            "includes_f2": bool(rec.get("includes_f2")),
            "note": rec.get("note"),
        })
    return specs


def _job(spec: dict[str, Any], n_jobs: int) -> dict[str, Any]:
    print(f"[round20] start {spec['code']} role={spec.get('role')} model={spec.get('model')}", flush=True)
    try:
        obs = run_one(spec, resume=True, n_jobs=n_jobs)
    except Exception as exc:
        obs = skip_payload(spec["code"], spec, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}", "SKIP_ERROR")
    print(
        f"[round20] done {obs.get('code')} status={obs.get('status')} sharpe={obs.get('sharpe')} reason={obs.get('reason')}",
        flush=True,
    )
    return obs


def _write_combo_md(rows: list[dict[str, Any]]) -> None:
    trained = [r for r in rows if r.get("sharpe") is not None]
    skipped = [r for r in rows if str(r.get("status") or "").startswith("SKIP") or r.get("status") == SKIP_DEPENDENCY]
    best = None
    if trained:
        best = max(trained, key=lambda r: (float(r.get("sharpe") or -999), float(r.get("cagr") or -999)))
    lines = [
        "# 08 冻结 combo registry 训练（walk-forward）",
        "",
        "registry 在看 NAV 前已冻结。本文件只记录训练/回测，**不追加** Best×Best 新组。",
        "引擎：fuse 90/10 + B03 + `apply_top2_veto` O01 + Round7 5bp，**不传** `validation_hooks`。",
        f"subsample=`{SUBSAMPLE}` colsample=`{COLSAMPLE}`。",
        f"tested=`{len(rows)}` trained_with_metrics=`{len(trained)}` skip=`{len(skipped)}`。",
        "",
    ]
    if best:
        lines += [
            f"best (by Sharpe, not independent OOS) = `{best.get('code')}` Sharpe=`{best.get('sharpe')}` "
            f"CAGR=`{best.get('cagr')}` MaxDD=`{best.get('max_drawdown')}` trades=`{best.get('completed_trades')}` "
            f"NAV_end=`{best.get('nav_end')}` extras=`{best.get('five_extras') or best.get('extras')}`",
            "",
        ]
    if skipped:
        lines += ["## SKIP", ""]
        for r in skipped:
            lines.append(f"- `{r.get('code')}` `{r.get('status')}` {r.get('reason')}")
        lines.append("")
    lines += [
        "| code | role | status | CAGR | Sharpe | MaxDD | trades | extras |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        extras = r.get("five_extras") or r.get("extras") or []
        if isinstance(extras, (list, tuple)):
            extra_s = "|".join(str(x) for x in extras)
        else:
            extra_s = str(extras)
        lines.append(
            f"| `{r.get('code')}` | {r.get('role')} | {r.get('status')} | {r.get('cagr')} | "
            f"{r.get('sharpe')} | {r.get('max_drawdown')} | {r.get('completed_trades')} | `{extra_s}` |"
        )
    write_md(COMBO_TRAIN_MD, "\n".join(lines))


def run_combo_train(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    require_frozen()
    from .combo import REG_CSV, REG_JSON

    freeze = load_json(REG_JSON)
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("combo registry not FROZEN; refuse to train")
    prepare_panels(resume=True)
    specs = registry_specs()
    if len(specs) > 100:
        raise RuntimeError(f"registry has {len(specs)} > 100; refuse")
    xgb_jobs = 1 if int(jobs) > 1 else XGB_N_JOBS
    rows: list[dict[str, Any]] = []
    if int(jobs) <= 1:
        init_panels()
        for spec in specs:
            rows.append(_job(dict(spec), xgb_jobs))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        init_panels()
        with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
            futs = {pool.submit(_job, dict(spec), xgb_jobs): spec["code"] for spec in specs}
            order = {spec["code"]: i for i, spec in enumerate(specs)}
            done = []
            for fut in as_completed(futs):
                done.append(fut.result())
            done.sort(key=lambda r: order.get(r.get("code"), 99))
            rows = done
    trained = [r for r in rows if r.get("sharpe") is not None]
    skipped = [r for r in rows if str(r.get("status") or "").startswith("SKIP") or r.get("status") == SKIP_DEPENDENCY]
    best = None
    if trained:
        best = max(trained, key=lambda r: (float(r.get("sharpe") or -999), float(r.get("cagr") or -999)))
        best = {k: best.get(k) for k in (
            "code", "role", "status", "cagr", "sharpe", "max_drawdown", "completed_trades",
            "nav_end", "five_extras", "ten_extras", "out_dir",
        )}
    slim_rows = []
    for r in rows:
        slim_rows.append({
            "code": r.get("code"),
            "role": r.get("role"),
            "status": r.get("status"),
            "reason": r.get("reason"),
            "cagr": r.get("cagr"),
            "sharpe": r.get("sharpe"),
            "max_drawdown": r.get("max_drawdown"),
            "completed_trades": r.get("completed_trades"),
            "n_obs": r.get("n_trading_observations") or r.get("n_obs"),
            "nav_end": r.get("nav_end"),
            "extras": "|".join(r.get("five_extras") or r.get("extras") or []) if isinstance(r.get("five_extras") or r.get("extras"), (list, tuple)) else r.get("five_extras") or r.get("extras"),
            "formal": r.get("formal"),
            "out_dir": r.get("out_dir"),
        })
    pd.DataFrame(slim_rows).to_csv(COMBO_RES_CSV, index=False)
    payload = {
        "status": "DONE",
        "trained": True,
        "looked_at_combo_nav": True,
        "post_hoc_best_x_best": False,
        "n_registry": int(len(specs)),
        "n_tested": int(len(rows)),
        "n_with_metrics": int(len(trained)),
        "n_skip": int(len(skipped)),
        "skips": [{"code": r.get("code"), "status": r.get("status"), "reason": r.get("reason")} for r in skipped],
        "best": best,
        "csv": str(COMBO_RES_CSV),
        "registry": str(REG_CSV),
        "note": "did not mutate frozen experiment_registry.csv; no Best×Best added after NAV",
    }
    dump_json(COMBO_TRAIN_JSON, payload)
    _write_combo_md(rows)
    return payload


__all__ = [
    "COMBO_RES_CSV",
    "COMBO_TRAIN_JSON",
    "init_panels",
    "parse_extras",
    "prepare_panels",
    "registry_specs",
    "run_combo_train",
    "run_one",
    "skip_payload",
]
