"""IV 求解：失败 ≠ 0 vol。附 calculation_status / quote_quality / source_timestamp。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_ROOT
from ..round15.factors import bs_delta, bs_price, implied_vol, pc_parity_gap, run_iv_unit_tests
from .chain import load_chain
from .common import dump_json, load_json, write_md
from .contract import RF_FALLBACK
from .quality import load_quality

IV_JSON = ROUND16_ROOT / "iv" / "iv.json"
IV_MD = ROUND16_ROOT / "04_implied_volatility_validation.md"
IV_PARQUET = ROUND16_ROOT / "data" / "processed" / "option_iv.parquet"
TEST_JSON = ROUND16_ROOT / "tests" / "iv_unit_tests.json"


def run_iv(**_: Any) -> dict[str, Any]:
    tests = run_iv_unit_tests()
    dump_json(TEST_JSON, tests)
    chain = load_chain()
    quality = load_quality()
    if chain.empty:
        payload = {"status": "EMPTY", "unit_tests": tests, "n_iv": 0}
        dump_json(IV_JSON, payload)
        write_md(IV_MD, "# Round16 · 04 IV\n\nEMPTY。\n")
        return payload
    work = chain.loc[chain["valid_quote"].fillna(False)].copy()
    if work.empty:
        payload = {"status": "NO_VALID_QUOTES", "unit_tests": tests, "n_iv": 0}
        dump_json(IV_JSON, payload)
        write_md(IV_MD, "# Round16 · 04 IV\n\n无 valid_quote。\n")
        return payload
    qflag = str((quality or {}).get("quote_quality_default") or "last_close_only")
    rate = float(RF_FALLBACK)
    px = pd.to_numeric(work["last"], errors="coerce").to_numpy(float)
    spot = pd.to_numeric(work["underlying_price"], errors="coerce").to_numpy(float)
    k = pd.to_numeric(work["strike"], errors="coerce").to_numpy(float)
    t = pd.to_numeric(work["days_to_expiry"], errors="coerce").to_numpy(float) / 365.0
    is_call = work["call_put"].astype(str).str.upper().str.startswith("C").to_numpy()
    ivs = np.full(len(work), np.nan)
    deltas = np.full(len(work), np.nan)
    statuses = np.array(["INVALID_INPUT"] * len(work), dtype=object)
    ok = np.isfinite(px) & np.isfinite(spot) & np.isfinite(k) & np.isfinite(t) & (px > 0) & (spot > 0) & (k > 0) & (t > 0)
    # 向量化二分：失败保持 NaN，不填 0
    lo = np.full(ok.sum(), 1e-4)
    hi = np.full(ok.sum(), 5.0)
    px_ok, spot_ok, k_ok, t_ok, call_ok = px[ok], spot[ok], k[ok], t[ok], is_call[ok]

    from scipy.special import ndtr

    def _bs(sig):
        sqrt_t = np.sqrt(t_ok)
        d1 = (np.log(spot_ok / k_ok) + (rate + 0.5 * sig * sig) * t_ok) / (np.maximum(sig * sqrt_t, 1e-12))
        d2 = d1 - sig * sqrt_t
        df = np.exp(-rate * t_ok)
        nd1 = ndtr(d1)
        nd2 = ndtr(d2)
        call_px = spot_ok * nd1 - k_ok * df * nd2
        put_px = k_ok * df * (1.0 - nd2) - spot_ok * (1.0 - nd1)
        return np.where(call_ok, call_px, put_px)

    fa = _bs(lo) - px_ok
    fb = _bs(hi) - px_ok
    bracket = fa * fb <= 0
    a, b = lo.copy(), hi.copy()
    for _ in range(40):
        mid = 0.5 * (a + b)
        fm = _bs(mid) - px_ok
        left = fa * fm <= 0
        b = np.where(left, mid, b)
        a = np.where(left, a, mid)
        fa = np.where(left, fa, fm)
    mid = 0.5 * (a + b)
    iv_ok = np.where(bracket, mid, np.nan)
    ivs[ok] = iv_ok
    statuses[ok] = np.where(bracket, "OK_LAST_CLOSE", "NO_CONVERGENCE")
    good = ok & np.isfinite(ivs)
    if good.any():
        sig = ivs[good]
        tg = t[good]
        sg = spot[good]
        kg = k[good]
        cg = is_call[good]
        sqrt_t = np.sqrt(tg)
        d1 = (np.log(sg / kg) + (rate + 0.5 * sig * sig) * tg) / np.maximum(sig * sqrt_t, 1e-12)
        nd1 = ndtr(d1)
        deltas[good] = np.where(cg, nd1, nd1 - 1.0)
    work = work.copy()
    work["iv"] = ivs
    work["delta"] = deltas
    work["ttm_years"] = t
    work["calculation_status"] = statuses
    work["quote_quality"] = qflag
    work["source_timestamp"] = work["quote_timestamp"]
    work["rf"] = rate
    work["rf_source"] = "FALLBACK_CONSTANT_0.02"
    IV_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    work.to_parquet(IV_PARQUET, index=False)
    n_ok = int(pd.Series(ivs).notna().sum())
    payload = {
        "status": "DONE" if n_ok else "NO_CONVERGENCE",
        "n_iv": n_ok,
        "n_input": int(len(work)),
        "unit_tests": tests,
        "rf": rate,
        "rf_source": "FALLBACK_CONSTANT_0.02",
        "failed_as_zero": False,
        "quote_quality": qflag,
        "path": str(IV_PARQUET),
        "ok_rate": float(n_ok / max(len(work), 1)),
    }
    dump_json(IV_JSON, payload)
    write_md(
        IV_MD,
        "\n".join([
            "# Round 16 · 04 隐含波动率",
            "",
            f"unit tests=`{tests.get('status')}` n_iv=`{n_ok}` / `{len(work)}` ok_rate=`{payload['ok_rate']:.4f}`。",
            f"失败不写成 0。quote_quality=`{qflag}` rf=`{rate}` ({payload['rf_source']})。",
            "合成单测：call/put roundtrip、PC parity、invalid 不填 0。",
            "",
        ]),
    )
    return payload


def load_iv() -> pd.DataFrame:
    if not IV_PARQUET.is_file():
        return pd.DataFrame()
    return pd.read_parquet(IV_PARQUET)


def load_iv_status() -> dict[str, Any]:
    return load_json(IV_JSON)


# re-export for tests
__all__ = ["bs_delta", "bs_price", "implied_vol", "load_iv", "load_iv_status", "pc_parity_gap", "run_iv", "run_iv_unit_tests"]
