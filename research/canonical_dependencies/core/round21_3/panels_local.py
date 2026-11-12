"""21.3 研究面板：R20 头面板 + 21.2 已有因子 + 本轮 T/U/V。不写 21.2。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND20_ROOT, ROUND21_2_ROOT, ROUND21_3_ROOT
from .common import dump_json
from .construct import PANEL_PATH
from .contract import KEYS

DATA = ROUND21_3_ROOT / "data"
RESEARCH = DATA / "research_panel.parquet"


def prepare_discovery_panels(*, resume: bool = True) -> dict[str, Any]:
    paths = {
        "five_training": DATA / "five_training.parquet",
        "five_candidate": DATA / "five_candidate.parquet",
        "ten_training": DATA / "ten_training.parquet",
        "ten_candidate": DATA / "ten_candidate.parquet",
    }
    if resume and all(p.is_file() for p in paths.values()):
        return {"status": "resumed", "paths": {k: str(v) for k, v in paths.items()}}
    # 优先从 21.2 已合并面板拷列，再 merge 本轮新因子
    src_root = ROUND21_2_ROOT / "data"
    new = pd.read_parquet(PANEL_PATH) if PANEL_PATH.is_file() else pd.DataFrame()
    extra_new = [c for c in new.columns if c not in KEYS] if not new.empty else []
    DATA.mkdir(parents=True, exist_ok=True)
    r20 = ROUND20_ROOT / "data"
    for key in paths:
        src = src_root / f"{key}.parquet"
        if not src.is_file():
            src = r20 / f"{key}.parquet"
        if not src.is_file():
            raise RuntimeError(f"MISSING_ARTIFACT panel {src}")
        frame = pd.read_parquet(src)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        if extra_new:
            add = [c for c in extra_new if c not in frame.columns]
            if add:
                nxt = new.copy()
                nxt["trade_date"] = pd.to_datetime(nxt["trade_date"])
                nxt["instrument_id"] = nxt["instrument_id"].astype(str)
                frame = frame.merge(nxt[KEYS + add], on=KEYS, how="left")
        frame.to_parquet(paths[key], index=False)
    meta = {"status": "prepared", "paths": {k: str(v) for k, v in paths.items()}, "filled_with_zero": False}
    dump_json(DATA / "panels.json", meta)
    return meta


def load_research_panel(*, resume: bool = True) -> pd.DataFrame:
    if resume and RESEARCH.is_file():
        frame = pd.read_parquet(RESEARCH)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        return frame
    prepare_discovery_panels(resume=True)
    frame = pd.read_parquet(DATA / "five_candidate.parquet")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    RESEARCH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(RESEARCH, index=False)
    return frame
