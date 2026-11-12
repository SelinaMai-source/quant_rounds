"""单因子 IC / Rolling / Shape / Top-tail。标签成熟后才评价。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_2_ROOT
from ..round20.metrics import icir_of, pearson_ic, rank_ic
from .common import dump_json, finite_or_none, write_md
from .construct import PANEL_PATH
from .contract import KEYS, LABELS, MD
from .panels import load_research_panel, pit_safe_factors
from .pit import run_pit

EV_JSON = ROUND21_2_ROOT / "evidence" / "evidence.json"
IC_CSV = ROUND21_2_ROOT / "new_factor_IC.csv"
ROLL_CSV = ROUND21_2_ROOT / "rolling_factor_evidence.csv"


def _daily_ic(frame: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(frame[label], errors="coerce")
    x = pd.to_numeric(frame[col], errors="coerce")
    ready = pd.to_datetime(frame.get("actual_exit_date" if label == "label_exit_rule_net" else "next2_date"), errors="coerce")
    for day, idx in frame.groupby("trade_date", sort=True).groups.items():
        xv = x.loc[idx].to_numpy(dtype=float)
        yv = y.loc[idx].to_numpy(dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        if int(mask.sum()) < 5:
            continue
        if int(np.unique(xv[mask]).size) < 2 or int(np.unique(yv[mask]).size) < 2:
            continue
        xs, ys = pd.Series(xv[mask]), pd.Series(yv[mask])
        rows.append(
            {
                "trade_date": pd.Timestamp(day),
                "n_valid": int(mask.sum()),
                "ic": pearson_ic(xs, ys),
                "rankic": rank_ic(xs, ys),
                "ready_at": pd.Timestamp(ready.loc[idx].max()) if ready.loc[idx].notna().any() else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def _roll(daily: pd.DataFrame, window: int) -> dict[str, Any]:
    if daily.empty or "rankic" not in daily.columns:
        return {"window": window, "rankic_mean": None, "icir": None, "pos_ratio": None}
    s = pd.to_numeric(daily["rankic"], errors="coerce").dropna()
    if len(s) < max(20, window // 4):
        return {"window": window, "rankic_mean": float(s.mean()) if len(s) else None, "icir": icir_of(s), "pos_ratio": float((s > 0).mean()) if len(s) else None, "note": "short"}
    roll = s.rolling(window, min_periods=max(10, window // 5)).mean()
    return {
        "window": window,
        "rankic_mean": float(s.mean()),
        "roll_last": float(roll.dropna().iloc[-1]) if roll.notna().any() else None,
        "icir": icir_of(s),
        "pos_ratio": float((s > 0).mean()),
        "neg_streak_max": int((s < 0).astype(int).groupby((s >= 0).astype(int).cumsum()).sum().max()) if len(s) else 0,
    }


def run_evidence(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and EV_JSON.is_file() and IC_CSV.is_file() and MD["06"].is_file():
        from .common import load_json

        existing = load_json(EV_JSON)
        if existing.get("status") == "DONE":
            return existing
    run_pit(resume=True)
    panel = load_research_panel(resume=True)
    factors = [c for c in pit_safe_factors(resume=False) if c in panel.columns]
    # 也对未过 PIT 的已计算列做诊断 IC
    from .contract import NEW_FACTOR_IDS

    diag = [c for c in NEW_FACTOR_IDS if c in panel.columns]
    ic_rows = []
    roll_rows = []
    shape_rows = []
    packages = {}
    for fid in diag:
        pack = {"factor_id": fid, "formal": fid in factors}
        for lab in LABELS:
            if lab not in panel.columns:
                continue
            daily = _daily_ic(panel, fid, lab)
            ric = pd.to_numeric(daily["rankic"], errors="coerce") if not daily.empty else pd.Series(dtype=float)
            rec = {
                "factor_id": fid,
                "label": lab,
                "raw_ic": float(pd.to_numeric(daily["ic"], errors="coerce").mean()) if not daily.empty else None,
                "raw_rankic": float(ric.mean()) if ric.notna().any() else None,
                "icir": icir_of(ric) if ric.notna().any() else None,
                "n_days": int(ric.notna().sum()) if not daily.empty else 0,
            }
            ic_rows.append(rec)
            pack[lab] = rec
            dest = ROUND21_2_ROOT / "factors" / fid
            dest.mkdir(parents=True, exist_ok=True)
            if not daily.empty:
                daily.to_parquet(dest / f"daily_ic_{lab}.parquet", index=False)
            for w, name in ((60, "ROUND21_INTERNAL_W60"), (120, "ROUND21_INTERNAL_W120"), (252, "ROUND21_INTERNAL_W252")):
                rr = _roll(daily, w)
                rr.update({"factor_id": fid, "label": lab, "window_name": name, "namespace": "ROUND21_INTERNAL"})
                roll_rows.append(rr)
            # shape / top-tail
            x = pd.to_numeric(panel[fid], errors="coerce")
            y = pd.to_numeric(panel[lab], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            q = None
            if int(mask.sum()) > 200:
                xx, yy = x[mask], y[mask]
                bins = pd.qcut(xx.rank(method="first"), 5, labels=False, duplicates="drop")
                means = yy.groupby(bins).mean()
                q = {int(i): float(means.loc[i]) for i in means.index}
                top = float(yy.loc[bins == bins.max()].mean()) if len(means) else None
                bot = float(yy.loc[bins == bins.min()].mean()) if len(means) else None
                pack[f"toptail_{lab}"] = {"top": top, "bottom": bot, "spread": None if top is None or bot is None else top - bot, "quintile": q}
            shape_rows.append({"factor_id": fid, "label": lab, "quintile": q})
        dump_json(ROUND21_2_ROOT / "factors" / fid / "evidence.json", pack)
        packages[fid] = pack
    pd.DataFrame(ic_rows).to_csv(IC_CSV, index=False)
    pd.DataFrame(roll_rows).to_csv(ROLL_CSV, index=False)
    best = sorted((r for r in ic_rows if r.get("label") == "label_e2_net" and r.get("raw_rankic") is not None), key=lambda r: abs(r["raw_rankic"]), reverse=True)[:5]
    payload = {
        "status": "DONE",
        "n_ic": len(ic_rows),
        "best_abs_rankic_e2": best,
        "n_formal": len(factors),
        "filled_ic_as_zero": False,
    }
    dump_json(EV_JSON, payload)
    lines = [
        "# Round21.2 · 06 单因子证据",
        "",
        f"formal PIT_PASS N=`{len(factors)}`。薄日跳过不记 IC=0。内部窗 60/120/252。",
        f"best |RankIC| e2=`{best}`",
        "",
    ]
    write_md(MD["06"], "\n".join(lines))
    # 08 shape
    write_md(
        MD["08"],
        "\n".join(
            [
                "# Round21.2 · 08 非线性与 Top-tail",
                "",
                "五分位均值。不要求完美单调。Top-tail only 单独标记。",
                "",
                f"N shape rows=`{len(shape_rows)}`",
                "",
            ]
        ),
    )
    dump_json(ROUND21_2_ROOT / "shape" / "shape.json", {"rows": shape_rows})
    return payload
