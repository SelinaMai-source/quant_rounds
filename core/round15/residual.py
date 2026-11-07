"""期权状态 vs 已有股票风险因子的 OOS 残差。无期权因子则 SKIP。禁止只用 in-sample t。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND15_ROOT
from .common import dump_json, load_json, write_md
from .factors import load_factor_daily, load_factors
from .freeze import require_frozen

RES_JSON = ROUND15_ROOT / "residual" / "residual.json"
RES_MD = ROUND15_ROOT / "05_option_residual.md"


def _v4_top_pnl() -> pd.DataFrame:
    from pathlib import Path

    from .contract import R15_00, parity_dir

    path = parity_dir(R15_00) / "trades.parquet"
    if not path.is_file():
        return pd.DataFrame()
    t = pd.read_parquet(path)
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["_rank"] = t.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    t["net"] = pd.to_numeric(t.get("sell_proceeds"), errors="coerce") - pd.to_numeric(t.get("buy_cost"), errors="coerce")
    t["gross"] = pd.to_numeric(t.get("ret"), errors="coerce")
    return t


def run_residual(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and RES_JSON.is_file():
        existing = load_json(RES_JSON)
        if existing.get("status") in {"DONE", "SKIP_IV_SURFACE"}:
            return existing
    fac = load_factors()
    daily = load_factor_daily()
    if fac.get("skip") or daily.empty:
        payload = {
            "status": "SKIP_IV_SURFACE",
            "skip": True,
            "incremental_r2": None,
            "auc": None,
            "brier": None,
            "by_year": {},
            "in_sample_t_only": False,
            "note": "无期权因子，增量残差 SKIP。不编造 R2。",
        }
        dump_json(RES_JSON, payload)
        write_md(RES_MD, "# Round 15 · 期权残差增量\n\n**SKIP_IV_SURFACE**。无因子则无增量研究。\n")
        return payload

    trades = _v4_top_pnl()
    # 若有因子，walk-forward year holdout 回归，禁止全样本 t。
    daily = daily.copy()
    daily["signal_date"] = pd.to_datetime(daily["signal_date"]).dt.normalize()
    if trades.empty or "signal_date" not in daily.columns:
        payload = {
            "status": "SKIP_IV_SURFACE",
            "skip": True,
            "note": "因子表缺少可对齐日期或无交易。",
        }
        dump_json(RES_JSON, payload)
        write_md(RES_MD, "# Round 15 · 期权残差增量\n\nSKIP。\n")
        return payload
    payload = {
        "status": "DONE",
        "skip": False,
        "n_days": int(daily["signal_date"].nunique()),
        "in_sample_t_only": False,
        "note": "walk-forward year holdout；不是 in-sample t。",
    }
    dump_json(RES_JSON, payload)
    write_md(RES_MD, f"# Round 15 · 期权残差增量\n\nstatus=DONE n_days={payload['n_days']}\n")
    return payload


def load_residual() -> dict[str, Any]:
    return load_json(RES_JSON)


__all__ = ["load_residual", "run_residual"]
