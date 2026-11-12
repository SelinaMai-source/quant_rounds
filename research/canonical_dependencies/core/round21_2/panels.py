"""R21.2 研究面板：R20 头面板 + 本轮新因子。不写 R21/R21.1。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND20_ROOT, ROUND21_2_ROOT
from .common import dump_json, load_json
from .construct import PANEL_PATH
from .contract import KEYS

DATA = ROUND21_2_ROOT / "data"
RESEARCH = DATA / "research_panel.parquet"
SAFE_JSON = DATA / "pit_safe_factors.json"
_G: dict[str, Any] = {}


def prepare_head_panels(*, resume: bool = True) -> dict[str, Any]:
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
    if not PANEL_PATH.is_file():
        from .construct import run_construct

        run_construct(resume=True)
    new = pd.read_parquet(PANEL_PATH)
    new["trade_date"] = pd.to_datetime(new["trade_date"])
    new["instrument_id"] = new["instrument_id"].astype(str)
    extra_new = [c for c in new.columns if c not in KEYS]
    DATA.mkdir(parents=True, exist_ok=True)
    r20 = ROUND20_ROOT / "data"
    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate"):
        src = r20 / f"{key}.parquet"
        if not src.is_file():
            raise RuntimeError(f"MISSING_ARTIFACT R20 panel {src}")
        frame = pd.read_parquet(src)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        extra = [c for c in extra_new if c not in frame.columns]
        if extra:
            frame = frame.merge(new[KEYS + extra], on=KEYS, how="left")
        frame.to_parquet(paths[key], index=False)
    pd.read_parquet(r20 / "five_frozen.parquet").to_parquet(paths["five_frozen"], index=False)
    pd.read_parquet(r20 / "ten_frozen.parquet").to_parquet(paths["ten_frozen"], index=False)
    meta = {"status": "prepared", "paths": {k: str(v) for k, v in paths.items()}, "filled_with_zero": False}
    dump_json(DATA / "panels.json", meta)
    return meta


def load_research_panel(*, resume: bool = True) -> pd.DataFrame:
    if resume and RESEARCH.is_file():
        frame = pd.read_parquet(RESEARCH)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        return frame
    prepare_head_panels(resume=True)
    frame = pd.read_parquet(DATA / "five_candidate.parquet")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    RESEARCH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(RESEARCH, index=False)
    return frame


def pit_safe_factors(*, resume: bool = True) -> list[str]:
    if resume and SAFE_JSON.is_file():
        existing = load_json(SAFE_JSON)
        if existing.get("factors"):
            return list(existing["factors"])
    from .pit import PIT_JSON

    pit = load_json(PIT_JSON)
    factors = [r["factor_id"] for r in (pit.get("rows") or []) if r.get("pit_status") == "PIT_PASS"]
    dump_json(SAFE_JSON, {"factors": factors})
    return factors


def init_heads() -> None:
    if _G:
        return
    prepare_head_panels(resume=True)
    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate", "five_frozen", "ten_frozen"):
        frame = pd.read_parquet(DATA / f"{key}.parquet")
        if "trade_date" in frame.columns:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        if "instrument_id" in frame.columns:
            frame["instrument_id"] = frame["instrument_id"].astype(str)
        if "signal_date" in frame.columns:
            frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        _G[key] = frame
