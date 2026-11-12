"""R21.1 研究面板：只读合并 R21/R20 面板 + 同伴列。不写 round21。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND20_ROOT, ROUND21_1_ROOT, ROUND21_ROOT
from ..round21.construct import PANEL_PATH as R21_PANEL
from .common import dump_json, load_json
from .contract import KEYS, PIT_PASS_R21
from .companion import COMP_PATH, build_companions

DATA = ROUND21_1_ROOT / "data"
RESEARCH = DATA / "research_panel.parquet"
RESEARCH_JSON = DATA / "research_panel.json"
SAFE_JSON = DATA / "pit_safe_factors.json"


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
    r21 = pd.read_parquet(R21_PANEL)
    r21["trade_date"] = pd.to_datetime(r21["trade_date"])
    r21["instrument_id"] = r21["instrument_id"].astype(str)
    ac_path = ROUND21_ROOT / "data" / "ac_factors.parquet"
    ac = pd.read_parquet(ac_path) if ac_path.is_file() else pd.DataFrame()
    if not ac.empty:
        ac["trade_date"] = pd.to_datetime(ac["trade_date"])
        ac["instrument_id"] = ac["instrument_id"].astype(str)
    companions, _ = build_companions(r21, ac, resume=resume)
    extra_new = [c for c in companions.columns if c not in KEYS]
    DATA.mkdir(parents=True, exist_ok=True)
    r20 = ROUND20_ROOT / "data"
    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate"):
        src = r20 / f"{key}.parquet"
        if not src.is_file():
            raise RuntimeError(f"MISSING_ARTIFACT R20 panel {src}")
        frame = pd.read_parquet(src)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        extra = [c for c in r21.columns if c not in frame.columns and c not in {"is_first_board_hard"}]
        if extra:
            frame = frame.merge(r21[KEYS + extra], on=KEYS, how="left")
        extra_c = [c for c in extra_new if c not in frame.columns]
        if extra_c:
            frame = frame.merge(companions[KEYS + extra_c], on=KEYS, how="left")
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
        for col in ("next2_date", "actual_exit_date"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col], errors="coerce")
        return frame
    prepare_head_panels(resume=True)
    # 用 five_candidate 作候选截面（一进二首板）
    frame = pd.read_parquet(DATA / "five_candidate.parquet")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    RESEARCH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(RESEARCH, index=False)
    dump_json(
        RESEARCH_JSON,
        {
            "status": "DONE",
            "n": int(len(frame)),
            "n_dates": int(frame["trade_date"].nunique()),
            "path": str(RESEARCH),
            "universe": "first_board_candidates",
        },
    )
    return frame


def pit_safe_factors(*, resume: bool = True) -> list[str]:
    if resume and SAFE_JSON.is_file():
        existing = load_json(SAFE_JSON)
        if existing.get("factors"):
            return list(existing["factors"])
    from .coverage import COV_JSON

    cov = load_json(COV_JSON)
    extras = list(cov.get("companion_pit_pass") or [])
    factors = list(dict.fromkeys([*PIT_PASS_R21, *extras]))
    dump_json(SAFE_JSON, {"factors": factors, "pass_original": list(PIT_PASS_R21), "companions": extras})
    return factors


_G: dict[str, Any] = {}


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
