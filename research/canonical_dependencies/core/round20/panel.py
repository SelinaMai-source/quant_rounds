"""合并盘点 78 因子 + 双标签。缺失不填 0。只写 result/round20。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND20_ROOT, SIGNAL_END
from ..round7.contract import FIVE_HEAD, FIVE_FEATURES, TEN_FEATURES
from ..round17.factors import FACTOR_PARQUET as R17_FACTORS
from ..round18.factors import PANEL as R18_PANEL
from .common import dump_json
from .contract import LABELS
from .inventory import REG_CSV

KEYS = ["trade_date", "instrument_id"]
LABEL_END = {
    "label_e2_net": "next2_date",
    "label_exit_rule_net": "actual_exit_date",
}
PANEL_PATH = ROUND20_ROOT / "factors" / "diag_panel.parquet"
PANEL_JSON = ROUND20_ROOT / "factors" / "diag_panel.json"

EXTRA_COLS = [
    *LABELS,
    "next2_date",
    "actual_exit_date",
    "is_first_board_hard",
]


def load_registry() -> pd.DataFrame:
    if not REG_CSV.is_file():
        raise RuntimeError(f"MISSING_ARTIFACT inventory registry: {REG_CSV}")
    frame = pd.read_csv(REG_CSV)
    frame["name"] = frame["name"].astype(str)
    return frame


def factor_names(registry: pd.DataFrame | None = None) -> list[str]:
    reg = registry if registry is not None else load_registry()
    return [str(x) for x in reg["name"].tolist()]


def _read_keys(path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    cols = [c for c in columns if c in names]
    frame = pd.read_parquet(path, columns=cols)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    for col in frame.columns:
        if col in KEYS:
            continue
        if col in {"next2_date", "actual_exit_date"}:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
            continue
        if col in {"is_first_board_hard"}:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def build_diag_panel(*, resume: bool = True) -> pd.DataFrame:
    if resume and PANEL_PATH.is_file() and PANEL_JSON.is_file():
        frame = pd.read_parquet(PANEL_PATH)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        for col in ("next2_date", "actual_exit_date"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col], errors="coerce")
        return frame
    reg = load_registry()
    names = factor_names(reg)
    cand_path = FIVE_HEAD.panel_dir / "candidate_panel.parquet"
    if not cand_path.is_file():
        raise RuntimeError(f"MISSING_ARTIFACT candidate_panel: {cand_path}")
    import pyarrow.parquet as pq

    from .contract import F1_NAME, F2_NAME, F3_NAME

    cand_cols = set(pq.read_schema(cand_path).names)
    from_cand = [n for n in names if n in cand_cols]
    missing_cand = [n for n in names if n not in cand_cols]
    cols = list(dict.fromkeys([*KEYS, *EXTRA_COLS, *from_cand]))
    panel = _read_keys(cand_path, cols)
    r18_names: list[str] = []
    if R17_FACTORS.is_file():
        r17_keep = [n for n in names if n.startswith("r17_")]
        r17 = _read_keys(R17_FACTORS, [*KEYS, *r17_keep])
        panel = panel.merge(r17, on=KEYS, how="left")
    else:
        raise RuntimeError(f"MISSING_ARTIFACT R17 new_factors: {R17_FACTORS}")
    if R18_PANEL.is_file():
        r18_keep = [n for n in names if n in {F1_NAME, F2_NAME, F3_NAME}]
        r18 = _read_keys(R18_PANEL, [*KEYS, *r18_keep])
        r18_names = [c for c in r18.columns if c not in KEYS]
        panel = panel.merge(r18, on=KEYS, how="left")
    else:
        raise RuntimeError(f"MISSING_ARTIFACT R18 factor_panel: {R18_PANEL}")
    still_missing = [n for n in names if n not in panel.columns]
    for col in still_missing:
        panel[col] = np.nan
    panel = panel.replace([np.inf, -np.inf], np.nan)
    if panel.duplicated(KEYS).any():
        raise RuntimeError("diag panel duplicate keys")
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    payload = {
        "status": "DONE",
        "n_rows": int(len(panel)),
        "n_dates": int(panel["trade_date"].nunique()),
        "date_min": str(panel["trade_date"].min().date()),
        "date_max": str(panel["trade_date"].max().date()),
        "n_factors": len(names),
        "from_candidate": from_cand,
        "from_r17": [n for n in names if n.startswith("r17_")],
        "from_r18": r18_names,
        "missing_from_candidate_resolved_elsewhere": missing_cand,
        "still_missing": still_missing,
        "filled_with_zero": False,
        "path": str(PANEL_PATH),
        "signal_end": str(SIGNAL_END),
        "five_features": list(FIVE_FEATURES),
        "ten_features": list(TEN_FEATURES),
    }
    dump_json(PANEL_JSON, payload)
    return panel


def ic_window(panel: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp("2018-01-01")
    end = pd.Timestamp(str(SIGNAL_END))
    work = panel.loc[panel["trade_date"].between(start, end)].copy()
    return work


def fold_masks(
    panel: pd.DataFrame,
    *,
    predict_year: int,
    label: str,
) -> tuple[pd.Series, pd.Series]:
    train_start, train_end = predict_year - 3, predict_year - 1
    years = panel["trade_date"].dt.year
    pred = years.eq(predict_year)
    train = years.between(train_start, train_end)
    end_col = LABEL_END.get(label)
    if end_col and end_col in panel.columns:
        end = pd.to_datetime(panel[end_col], errors="coerce")
        train = train & end.dt.year.le(train_end)
    return train, pred


__all__ = [
    "EXTRA_COLS",
    "KEYS",
    "LABEL_END",
    "PANEL_JSON",
    "PANEL_PATH",
    "build_diag_panel",
    "factor_names",
    "fold_masks",
    "ic_window",
    "load_registry",
]
