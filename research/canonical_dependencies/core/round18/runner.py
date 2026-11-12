"""训练 + 正式回测矩阵。F2 口径失败则含 F2 组 SKIP。D 系列 DIAGNOSTIC_ONLY。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND18_ROOT
from ..round7.contract import FIVE_HEAD, TEN_HEAD
from .backtest import run_fused_backtest
from .common import dump_json, load_json, write_md
from .contract import (
    A00,
    EXPERIMENTS,
    F2_NAME,
    PIT_TIMESTAMP_UNVERIFIED,
    SKIP_FACTOR_FOR_PROMOTION,
    exp_dir,
)
from .factors import load_factor_panel
from .freeze import require_frozen
from .parity import load_parity_status
from .train import train_both_heads

DATA = ROUND18_ROOT / "data"
RUN_JSON = ROUND18_ROOT / "metrics" / "matrix.json"
RUN_MD = ROUND18_ROOT / "08_experiment_matrix.md"
KEYS = ["trade_date", "instrument_id"]
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
    factors = load_factor_panel()
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
        training = training.merge(factors, on=KEYS, how="left")
        candidates = candidates.merge(factors, on=KEYS, how="left")
        paths[tkey].parent.mkdir(parents=True, exist_ok=True)
        training.to_parquet(paths[tkey], index=False)
        candidates.to_parquet(paths[ckey], index=False)
    from ..paths import ROUND7_ROOT

    pd.read_parquet(ROUND7_ROOT / "models" / "five" / "predictions.parquet").to_parquet(paths["five_frozen"], index=False)
    pd.read_parquet(ROUND7_ROOT / "models" / "ten" / "predictions.parquet").to_parquet(paths["ten_frozen"], index=False)
    meta = {"status": "prepared", "paths": {k: str(v) for k, v in paths.items()}, "filled_with_zero": False}
    dump_json(DATA / "panels.json", meta)
    return meta


def _init() -> None:
    if _G:
        return
    _G["five_training"] = pd.read_parquet(DATA / "five_training.parquet")
    _G["five_candidate"] = pd.read_parquet(DATA / "five_candidate.parquet")
    _G["ten_training"] = pd.read_parquet(DATA / "ten_training.parquet")
    _G["ten_candidate"] = pd.read_parquet(DATA / "ten_candidate.parquet")
    _G["five_frozen"] = pd.read_parquet(DATA / "five_frozen.parquet")
    _G["ten_frozen"] = pd.read_parquet(DATA / "ten_frozen.parquet")
    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate"):
        _G[key]["trade_date"] = pd.to_datetime(_G[key]["trade_date"])
        _G[key]["instrument_id"] = _G[key]["instrument_id"].astype(str)


def _f2_skip_reason() -> str | None:
    pit = load_json(ROUND18_ROOT / "pit" / "pit.json")
    f2 = (pit.get("factors") or {}).get(F2_NAME) or {}
    meta = load_json(ROUND18_ROOT / "factors" / "f2_meta.json")
    status = meta.get("status")
    if status in {"SKIP_SCHEMA", "SKIP_DATA"}:
        return status
    if not meta.get("schema_units_consistent", True) and status == "SKIP_SCHEMA":
        return "SKIP_SCHEMA"
    return None


def run_one(spec: dict[str, Any], *, resume: bool = True) -> dict[str, Any]:
    code = spec["code"]
    dest = exp_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    if code == A00:
        par = load_parity_status()
        obs = dict(par.get("observed") or {})
        obs.update({"code": A00, "status": "PASS" if par.get("passed") else "FAIL", "role": spec["role"], "formal": True})
        dump_json(dest / "summary.json", obs)
        return obs
    extras = set(spec.get("five") or ()) | set(spec.get("ten") or ())
    skip = None
    if F2_NAME in extras:
        skip = _f2_skip_reason()
        if skip is None:
            # 单位一致则正式可跑；PIT 未验证则不得晋升
            pit = load_json(ROUND18_ROOT / "pit" / "pit.json")
            f2 = (pit.get("factors") or {}).get(F2_NAME) or {}
            if f2.get("pit_status") == PIT_TIMESTAMP_UNVERIFIED and spec.get("formal"):
                skip_promo = SKIP_FACTOR_FOR_PROMOTION
            else:
                skip_promo = None
        else:
            skip_promo = skip
    else:
        skip_promo = None
    if skip:
        obs = {"code": code, "status": skip, "reason": f"F2 {skip}", "formal": spec.get("formal"), "cagr": None, "sharpe": None}
        dump_json(dest / "summary.json", obs)
        return obs
    _init()
    trained = train_both_heads(
        five_training=_G["five_training"],
        five_candidates=_G["five_candidate"],
        ten_training=_G["ten_training"],
        ten_candidates=_G["ten_candidate"],
        five_extras=list(spec.get("five") or ()),
        ten_extras=list(spec.get("ten") or ()),
        output_dir=dest,
        resume=resume,
        five_frozen=_G["five_frozen"],
        ten_frozen=_G["ten_frozen"],
    )
    obs = run_fused_backtest(
        dest,
        trained["five_pred"],
        trained["ten_pred"],
        extra={
            "code": code,
            "role": spec["role"],
            "five_extras": list(spec.get("five") or ()),
            "ten_extras": list(spec.get("ten") or ()),
            "fee_schedule": "HISTORICAL_FEE_CORRECTED",
            "formal": spec.get("formal"),
        },
        resume=resume,
    )
    obs["code"] = code
    obs["role"] = spec["role"]
    obs["formal"] = spec.get("formal")
    obs["five_extras"] = list(spec.get("five") or ())
    obs["ten_extras"] = list(spec.get("ten") or ())
    obs["promotion_eligible"] = bool(spec.get("formal")) and skip_promo is None
    if skip_promo:
        obs["promotion_block"] = skip_promo
    dump_json(dest / "summary.json", obs)
    return obs


def _job(spec: dict[str, Any]) -> dict[str, Any]:
    print(f"[round18] start {spec['code']} role={spec['role']}", flush=True)
    obs = run_one(spec, resume=True)
    print(f"[round18] done {spec['code']} status={obs.get('status')} sharpe={obs.get('sharpe')}", flush=True)
    return obs


def run_matrix(*, resume: bool = True, jobs: int = 4, **_: Any) -> dict[str, Any]:
    require_frozen()
    prepare_panels(resume=resume)
    if int(jobs) <= 1:
        rows = []
        for spec in EXPERIMENTS:
            rows.append(_job(dict(spec)))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        rows = []
        with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
            futs = {pool.submit(_job, dict(spec)): spec["code"] for spec in EXPERIMENTS}
            for fut in as_completed(futs):
                rows.append(fut.result())
        order = {spec["code"]: i for i, spec in enumerate(EXPERIMENTS)}
        rows.sort(key=lambda r: order.get(r.get("code"), 99))
    payload = {"status": "done", "rows": rows}
    dump_json(RUN_JSON, payload)
    lines = ["# 08 实验矩阵", "", "| code | role | formal | status | CAGR | Sharpe | MaxDD | trades |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r.get('code')} | {r.get('role')} | {r.get('formal')} | {r.get('status')} | "
            f"{r.get('cagr')} | {r.get('sharpe')} | {r.get('max_drawdown')} | {r.get('completed_trades')} |"
        )
    write_md(RUN_MD, "\n".join(lines))
    return payload
