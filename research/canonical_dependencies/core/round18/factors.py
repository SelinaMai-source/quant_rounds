"""编排 F1/F2/F3 构建并合并到首板键。缺失不填 0。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND18_ROOT, assert_project_result_path
from ..round7.contract import FIVE_HEAD, TEN_HEAD
from .common import dump_json, write_md
from .contract import F1_NAME, F2_NAME, F3_NAME
from .f1 import build_f1
from .f2 import build_f2
from .f3 import build_f3

KEYS = ["trade_date", "instrument_id"]
PANEL = ROUND18_ROOT / "data" / "processed" / "factor_panel.parquet"
BUILD_JSON = ROUND18_ROOT / "factors" / "factor_build.json"
BUILD_MD = ROUND18_ROOT / "05_factor_build.md"


def candidate_keys() -> pd.DataFrame:
    frames = []
    for head in (FIVE_HEAD, TEN_HEAD):
        for name in ("training_panel.parquet", "candidate_panel.parquet"):
            path = head.panel_dir / name
            if path.is_file():
                part = pd.read_parquet(path, columns=KEYS)
                frames.append(part)
    keys = pd.concat(frames, ignore_index=True)
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    keys["instrument_id"] = keys["instrument_id"].astype(str)
    return keys.drop_duplicates(KEYS).sort_values(KEYS).reset_index(drop=True)


def build_factors(*, resume: bool = True) -> dict[str, Any]:
    keys = candidate_keys()
    f1, m1 = build_f1(keys, resume=resume)
    f2, m2 = build_f2(keys, resume=resume)
    v_seal = f1[KEYS + ["V_seal"]] if "V_seal" in f1.columns else None
    f3, m3 = build_f3(keys, v_seal=v_seal, resume=resume)
    panel = keys.copy()
    for frame, col in ((f1, F1_NAME), (f2, F2_NAME), (f3, F3_NAME)):
        add = frame[KEYS + [col]].copy()
        add["trade_date"] = pd.to_datetime(add["trade_date"])
        add["instrument_id"] = add["instrument_id"].astype(str)
        panel = panel.merge(add, on=KEYS, how="left")
    assert_project_result_path(PANEL)
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL, index=False)
    payload = {
        "status": "built",
        "n_keys": int(len(panel)),
        "f1": m1,
        "f2": m2,
        "f3": m3,
        "n_valid": {
            F1_NAME: int(panel[F1_NAME].notna().sum()),
            F2_NAME: int(panel[F2_NAME].notna().sum()),
            F3_NAME: int(panel[F3_NAME].notna().sum()),
        },
        "filled_with_zero": False,
        "panel": str(PANEL),
    }
    dump_json(BUILD_JSON, payload)
    write_md(
        BUILD_MD,
        "# 05 因子构建\n\n"
        f"- keys=`{payload['n_keys']}`\n"
        f"- F1 valid=`{payload['n_valid'][F1_NAME]}` status=`{m1.get('status', 'built')}`\n"
        f"- F2 valid=`{payload['n_valid'][F2_NAME]}` status=`{m2.get('status')}` pit=`{m2.get('pit_status')}`\n"
        f"- F3 valid=`{payload['n_valid'][F3_NAME]}` used_total_mv=`False`\n"
        "- 缺失保持 NaN，未 fillna(0)。\n",
    )
    return payload


def load_factor_panel() -> pd.DataFrame:
    if not PANEL.is_file():
        raise FileNotFoundError(PANEL)
    frame = pd.read_parquet(PANEL)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    return frame
