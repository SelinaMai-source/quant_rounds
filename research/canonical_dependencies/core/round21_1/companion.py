"""同伴 NA / flag / 事件条件因子。各自新 factor_id，不得冒充原公式 PASS。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_1_ROOT
from .common import dump_json
from .contract import KEYS

COMP_PATH = ROUND21_1_ROOT / "data" / "companion_panel.parquet"
COMP_JSON = ROUND21_1_ROOT / "data" / "companion_panel.json"

COMPANION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "factor_id": "r21_a1_event_na_flag",
        "family": "A",
        "short": "A1 事件不适用",
        "formula": "I(min1_present & r21_a1_vw_limit_gap is NA)",
        "kind": "NotApplicableIndicator",
        "source": "r21_a1_vw_limit_gap",
    },
    {
        "factor_id": "r21_a1_missing_flag",
        "family": "A",
        "short": "A1 分钟缺失",
        "formula": "I(min1_missing)",
        "kind": "MissingnessFlag",
        "source": "r21_a1_vw_limit_gap",
    },
    {
        "factor_id": "r21_a2_event_na_flag",
        "family": "A",
        "short": "A2 无回封/无开板",
        "formula": "I(min1_present & r21_a2_minutes_to_reseal is NA)",
        "kind": "NotApplicableIndicator",
        "source": "r21_a2_minutes_to_reseal",
    },
    {
        "factor_id": "r21_a2_opened_flag",
        "family": "A",
        "short": "触板后曾开板",
        "formula": "I(r21_a3_retouch_count > 0)",
        "kind": "EventConditionalFeature",
        "source": "r21_a3_retouch_count",
    },
    {
        "factor_id": "r21_c2_zero_amp_flag",
        "family": "C",
        "short": "触板后振幅为 0",
        "formula": "I(min1_present & r21_c1 finite & r21_c2 is NA)",
        "kind": "NotApplicableIndicator",
        "source": "r21_c2_post_touch_amt_vs_impact",
    },
    {
        "factor_id": "r21_c3_short_path_flag",
        "family": "C",
        "short": "触板后路径过短",
        "formula": "I(min1_present & r21_c3_post_touch_path_eff is NA)",
        "kind": "NotApplicableIndicator",
        "source": "r21_c3_post_touch_path_eff",
    },
    {
        "factor_id": "r21_f_margin_eligible",
        "family": "F",
        "short": "两融资格",
        "formula": "I(daily_margin asof available_at exists)",
        "kind": "NotApplicableIndicator",
        "source": "r21_f1_fin_bal_chg",
    },
    {
        "factor_id": "r21_f_alignment_fail",
        "family": "F",
        "short": "两融 asof 失败",
        "formula": "I(margin_row_exists_but_value_NA)",
        "kind": "MissingnessFlag",
        "source": "r21_f1_fin_bal_chg",
    },
)


def build_companions(
    panel: pd.DataFrame,
    ac: pd.DataFrame,
    *,
    resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if resume and COMP_PATH.is_file() and COMP_JSON.is_file():
        frame = pd.read_parquet(COMP_PATH)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        from .common import load_json

        return frame, load_json(COMP_JSON)

    work = panel[KEYS].copy()
    a1 = pd.to_numeric(panel.get("r21_a1_vw_limit_gap"), errors="coerce")
    a2 = pd.to_numeric(panel.get("r21_a2_minutes_to_reseal"), errors="coerce")
    a3 = pd.to_numeric(panel.get("r21_a3_retouch_count"), errors="coerce")
    c1 = pd.to_numeric(panel.get("r21_c1_min1_amihud"), errors="coerce")
    c2 = pd.to_numeric(panel.get("r21_c2_post_touch_amt_vs_impact"), errors="coerce")
    c3 = pd.to_numeric(panel.get("r21_c3_post_touch_path_eff"), errors="coerce")
    f1 = pd.to_numeric(panel.get("r21_f1_fin_bal_chg"), errors="coerce")
    if not ac.empty and "touch_flag" in ac.columns:
        ac_use = ac[KEYS + ["touch_flag"]].copy()
        ac_use["trade_date"] = pd.to_datetime(ac_use["trade_date"])
        ac_use["instrument_id"] = ac_use["instrument_id"].astype(str)
        work = work.merge(ac_use, on=KEYS, how="left")
        min1_missing = work["touch_flag"].astype(str).eq("SKIP_DATA") | (a3.isna() & c1.isna())
    else:
        min1_missing = a3.isna() & c1.isna()
    min1_ok = ~min1_missing.to_numpy()

    out = work[KEYS].copy()
    out["r21_a1_event_na_flag"] = (min1_ok & a1.isna().to_numpy()).astype(np.float64)
    out["r21_a1_missing_flag"] = min1_missing.astype(np.float64)
    out["r21_a2_event_na_flag"] = (min1_ok & a2.isna().to_numpy()).astype(np.float64)
    opened = a3.gt(0)
    out["r21_a2_opened_flag"] = np.where(a3.notna(), opened.astype(np.float64), np.nan)
    out["r21_c2_zero_amp_flag"] = (min1_ok & c1.notna().to_numpy() & c2.isna().to_numpy()).astype(np.float64)
    out["r21_c3_short_path_flag"] = (min1_ok & c3.isna().to_numpy()).astype(np.float64)

    from ..round21.construct import F_CACHE

    if F_CACHE.is_file():
        fcache = pd.read_parquet(F_CACHE)
        fcache["trade_date"] = pd.to_datetime(fcache["trade_date"])
        fcache["instrument_id"] = fcache["instrument_id"].astype(str)
        col = "f_known_at" if "f_known_at" in fcache.columns else ("available_at" if "available_at" in fcache.columns else None)
        if col:
            tmp = fcache[KEYS + [col]].copy()
            tmp["f_known_at"] = pd.to_datetime(tmp[col], errors="coerce")
            out = out.merge(tmp[KEYS + ["f_known_at"]], on=KEYS, how="left")
            has = out["f_known_at"].notna()
            out["r21_f_margin_eligible"] = has.astype(np.float64)
            out["r21_f_alignment_fail"] = (has & f1.isna()).astype(np.float64)
            out = out.drop(columns=["f_known_at"])
        else:
            out["r21_f_margin_eligible"] = np.nan
            out["r21_f_alignment_fail"] = np.nan
    else:
        out["r21_f_margin_eligible"] = np.nan
        out["r21_f_alignment_fail"] = np.nan

    COMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(COMP_PATH, index=False)
    dest_root = ROUND21_1_ROOT / "factors"
    for spec in COMPANION_SPECS:
        fid = spec["factor_id"]
        d = dest_root / fid
        d.mkdir(parents=True, exist_ok=True)
        vals = out[KEYS + [fid]].copy()
        vals.to_parquet(d / "values.parquet", index=False)
        dump_json(d / "definition.json", spec)
    meta = {
        "status": "DONE",
        "n": int(len(out)),
        "filled_with_zero": False,
        "features": list(COMPANION_SPECS),
        "path": str(COMP_PATH),
        "note": "同伴是新表达。原 A1/A2/C2/C3/F 公式仍按原 PIT 状态，不得冒充 PASS。",
    }
    dump_json(COMP_JSON, meta)
    return out, meta
