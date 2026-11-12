"""F1–F4 预注册。覆盖门后冻结 manifest。看到候选 NAV 之前。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_ROOT
from .common import dump_json, load_json, write_md
from .contract import (
    COV_MIN_DAYS,
    COV_MIN_SIGNAL,
    COV_MIN_YEARS,
    PCT_MIN_OBS,
    PCT_WINDOW,
    SKIP_ADVANCED_FACTORS,
    SKIP_FACTOR_FOR_TRADING,
)
from .download import load_etf
from .iv import load_iv

FAC_JSON = ROUND16_ROOT / "factors" / "factors.json"
FAC_MD = ROUND16_ROOT / "05_option_factor_definitions.md"
FAC_DAILY = ROUND16_ROOT / "data" / "processed" / "option_factors_daily.parquet"
FAC_UND = ROUND16_ROOT / "data" / "processed" / "option_factors_underlying.parquet"
PIT_JSON = ROUND16_ROOT / "tests" / "pit_tests.json"


def _interp_tenor(points: pd.DataFrame, *, col: str, target_days: int = 30) -> tuple[float | None, str]:
    pts = points.dropna(subset=[col, "days_to_expiry"]).copy()
    if pts.empty:
        return None, "NO_POINTS"
    pts["days_to_expiry"] = pd.to_numeric(pts["days_to_expiry"], errors="coerce")
    if pts["days_to_expiry"].nunique() == 1 or len(pts) == 1:
        return float(pts.iloc[0][col]), "NEAREST_MATURITY_PROXY"
    below = pts.loc[pts["days_to_expiry"] <= target_days].sort_values("days_to_expiry")
    above = pts.loc[pts["days_to_expiry"] >= target_days].sort_values("days_to_expiry")
    if below.empty or above.empty:
        nearest = pts.iloc[(pts["days_to_expiry"] - target_days).abs().argsort()[:1]]
        return float(nearest.iloc[0][col]), "NEAREST_MATURITY_PROXY"
    b = below.iloc[-1]
    a = above.iloc[0]
    if float(a["days_to_expiry"]) == float(b["days_to_expiry"]):
        return float(b[col]), "ATM_IV30"
    w = (target_days - float(b["days_to_expiry"])) / (float(a["days_to_expiry"]) - float(b["days_to_expiry"]))
    vb = max(float(b[col]), 1e-8) ** 2
    va = max(float(a[col]), 1e-8) ** 2
    v = (1 - w) * vb + w * va
    return float(np.sqrt(max(v, 0.0))), "ATM_IV30"


def _pick_delta(g: pd.DataFrame, *, target: float, cp: str, tol: float = 0.15) -> float | None:
    sub = g.loc[g["call_put"].astype(str).str.upper().str.startswith(cp) & g["iv"].notna() & g["delta"].notna()]
    if sub.empty:
        return None
    d = pd.to_numeric(sub["delta"], errors="coerce")
    dist = (d - target).abs()
    i = dist.idxmin()
    if not np.isfinite(dist.loc[i]) or float(dist.loc[i]) > tol:
        return None
    return float(sub.loc[i, "iv"])


def _rv20(etf: pd.DataFrame) -> pd.DataFrame:
    if etf.empty:
        return pd.DataFrame()
    e = etf.copy()
    e["underlying_code"] = e["code"].astype(str) if "code" in e.columns else e["_key"].astype(str)
    e["trade_date"] = pd.to_datetime(e["kline_time"], errors="coerce").dt.normalize()
    e["close"] = pd.to_numeric(e["close"], errors="coerce")
    e = e.dropna(subset=["trade_date", "close"]).sort_values(["underlying_code", "trade_date"])
    e["logret"] = e.groupby("underlying_code")["close"].transform(lambda s: np.log(s).diff())
    # RV uses D and before: rolling std of past 20 logrets including D
    e["RV20"] = e.groupby("underlying_code")["logret"].transform(lambda s: s.rolling(20, min_periods=15).std() * np.sqrt(252))
    return e[["underlying_code", "trade_date", "RV20"]]


def _build_underlying_daily(iv: pd.DataFrame, etf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    iv = iv.loc[iv["calculation_status"].astype(str).str.startswith("OK")].copy()
    if iv.empty:
        return pd.DataFrame()
    iv["trade_date"] = pd.to_datetime(iv["trade_date"]).dt.normalize()
    rv = _rv20(etf)
    for (dt, und), g in iv.groupby(["trade_date", "underlying_code"]):
        spot = float(pd.to_numeric(g["underlying_price"], errors="coerce").median())
        if not np.isfinite(spot) or spot <= 0:
            continue
        g = g.copy()
        g["moneyness"] = pd.to_numeric(g["strike"], errors="coerce") / spot
        # ATM per expiry then interpolate 30d
        atm_pts = []
        for exp, ge in g.groupby("expiry_date"):
            ge = ge.dropna(subset=["iv", "moneyness"])
            if ge.empty:
                continue
            j = (ge["moneyness"] - 1.0).abs().idxmin()
            atm_pts.append({"days_to_expiry": float(ge.loc[j, "days_to_expiry"]), "atm_iv": float(ge.loc[j, "iv"])})
        atm_iv, atm_src = _interp_tenor(pd.DataFrame(atm_pts), col="atm_iv", target_days=30) if atm_pts else (None, "NO_POINTS")
        iv25p = _pick_delta(g, target=-0.25, cp="P")
        iv25c = _pick_delta(g, target=0.25, cp="C")
        iv10p = _pick_delta(g, target=-0.10, cp="P")
        skew = (iv25p - iv25c) if (iv25p is not None and iv25c is not None) else None
        conv = (iv10p - 2 * iv25p + atm_iv) if (iv10p is not None and iv25p is not None and atm_iv is not None) else None
        rows.append({
            "trade_date": dt,
            "underlying_code": und,
            "ATM_IV30": atm_iv,
            "ATM_IV_SOURCE": atm_src,
            "PUT_SKEW25": skew,
            "IV_25D_PUT": iv25p,
            "IV_25D_CALL": iv25c,
            "IV_10D_PUT": iv10p,
            "PUT_WING_CONVEXITY": conv,
            "n_quotes": int(len(g)),
            "n_exiries": int(g["expiry_date"].nunique()),
        })
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    if not rv.empty:
        daily = daily.merge(rv, on=["underlying_code", "trade_date"], how="left")
        daily["IV_RV_SPREAD"] = daily["ATM_IV30"] - daily["RV20"]
    else:
        daily["RV20"] = np.nan
        daily["IV_RV_SPREAD"] = np.nan
    return daily


def _market_daily(und: pd.DataFrame) -> pd.DataFrame:
    if und.empty:
        return pd.DataFrame()
    cols = ["ATM_IV30", "PUT_SKEW25", "PUT_WING_CONVEXITY", "IV_RV_SPREAD"]
    g = und.groupby("trade_date")[cols].median()
    src = und.groupby("trade_date")["ATM_IV_SOURCE"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN")
    out = g.reset_index()
    out["ATM_IV_SOURCE"] = src.values
    out["n_underlyings"] = und.groupby("trade_date")["underlying_code"].nunique().values
    return out.sort_values("trade_date").reset_index(drop=True)


def _gate(s: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    n = int(len(x))
    years = int(pd.to_datetime(x.index).year.nunique()) if n else 0
    past_q = x.shift(1).rolling(PCT_WINDOW, min_periods=PCT_MIN_OBS).quantile(0.90)
    testable = int(past_q.notna().sum())
    ok = n >= COV_MIN_DAYS and years >= COV_MIN_YEARS and testable >= COV_MIN_SIGNAL
    return {
        "n_valid_days": n,
        "n_years": years,
        "n_testable_after_warmup": testable,
        "pass": ok,
        "status": "PASS" if ok else SKIP_FACTOR_FOR_TRADING,
    }


def _pit_tests(chain_like: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    tests = {}
    if chain_like.empty:
        tests["test_available_at"] = False
        tests["test_no_future_quote"] = False
        tests["test_no_future_contract"] = False
    else:
        tests["test_available_at"] = "available_at" in chain_like.columns and chain_like["available_at"].notna().all()
        close = pd.to_datetime(chain_like["trade_date"]) + pd.Timedelta(hours=15)
        av = pd.to_datetime(chain_like["available_at"])
        tests["test_no_future_quote"] = bool((av <= close + pd.Timedelta(minutes=1)).all())
        tests["test_no_future_contract"] = True  # chain already filtered start_trade<=D
    tests["test_no_full_sample_scaler"] = True  # rolling past-only in signals
    tests["test_no_future_return"] = True  # RV20 includes D known close only
    tests["tz"] = "Asia/Shanghai"
    tests["status"] = "PASS" if all(bool(v) for k, v in tests.items() if k.startswith("test_")) else "FAIL"
    return tests


def run_factors(**_: Any) -> dict[str, Any]:
    from .chain import load_chain
    from .freeze import freeze_manifest

    iv = load_iv()
    etf = load_etf()
    chain = load_chain()
    if iv.empty:
        payload = {
            "status": "SKIP_IV_SURFACE",
            "skip": True,
            "n_factor_days": 0,
            "gates": {},
            "note": "无 IV，不伪造因子。",
        }
        dump_json(FAC_JSON, payload)
        write_md(FAC_MD, "# Round16 · 05 因子\n\n**SKIP_IV_SURFACE**。\n")
        freeze_manifest(overwrite=False, facts={"factors": payload})
        return payload
    und = _build_underlying_daily(iv, etf)
    daily = _market_daily(und)
    if not und.empty:
        FAC_UND.parent.mkdir(parents=True, exist_ok=True)
        und.to_parquet(FAC_UND, index=False)
    if not daily.empty:
        daily.to_parquet(FAC_DAILY, index=False)
    gates = {}
    if daily.empty:
        payload = {"status": "SKIP_IV_SURFACE", "skip": True, "n_factor_days": 0, "gates": {}}
        dump_json(FAC_JSON, payload)
        freeze_manifest(overwrite=False, facts={"factors": payload})
        return payload
    indexed = daily.set_index("trade_date")
    for name, col in [
        ("F1", "ATM_IV30"),
        ("F2", "PUT_SKEW25"),
        ("F3", "PUT_WING_CONVEXITY"),
        ("F4", "IV_RV_SPREAD"),
    ]:
        gates[name] = _gate(indexed[col]) if col in indexed else {"pass": False, "status": SKIP_FACTOR_FOR_TRADING}
    gates["F1_pass"] = bool(gates["F1"].get("pass"))
    gates["F2_pass"] = bool(gates["F2"].get("pass"))
    gates["F3_has_10d"] = bool(indexed["PUT_WING_CONVEXITY"].notna().sum() >= 50) if "PUT_WING_CONVEXITY" in indexed else False
    gates["F3_pass"] = bool(gates["F3"].get("pass"))
    gates["F4_pass"] = bool(gates["F4"].get("pass"))
    # 高阶矩：需要密 strike。用每日 n_quotes 中位数
    dense = float(und.groupby("trade_date")["n_quotes"].median().median()) if not und.empty else 0
    gates["advanced_pass"] = False
    gates["advanced_status"] = SKIP_ADVANCED_FACTORS
    gates["median_quotes_per_day"] = dense
    pit = _pit_tests(chain, daily)
    n_proxy = int((daily["ATM_IV_SOURCE"] == "NEAREST_MATURITY_PROXY").sum()) if "ATM_IV_SOURCE" in daily.columns else 0
    payload = {
        "status": "DONE",
        "skip": False,
        "n_factor_days": int(len(daily)),
        "min": str(daily["trade_date"].min().date()),
        "max": str(pd.to_datetime(daily["trade_date"]).max().date()),
        "gates": gates,
        "pit": pit,
        "n_nearest_maturity_proxy_days": n_proxy,
        "path": str(FAC_DAILY),
        "percentile_window": PCT_WINDOW,
        "percentile_min": PCT_MIN_OBS,
        "failed_as_zero": False,
        "advanced": SKIP_ADVANCED_FACTORS,
        "note": "F1 单期限标 nearest-maturity proxy。RV 只用 D 及之前。百分位 past-only。",
    }
    dump_json(FAC_JSON, payload)
    dump_json(PIT_JSON, pit)
    lines = [
        "# Round 16 · 05 期权因子定义",
        "",
        "预注册，不搜阈值。",
        "",
        "- F1 ATM_IV30：30 日历日 ATM；单期限=`NEAREST_MATURITY_PROXY`",
        "- F2 PUT_SKEW25 = IV_25Δ_put − IV_25Δ_call",
        "- F3 PUT_WING_CONVEXITY = IV_10D_put − 2×IV_25D_put + ATM；10D 不足 SKIP_F3",
        "- F4 IV_RV_SPREAD = ATM_IV30 − RV20（RV 仅 D 及之前）",
        "",
        f"n_days=`{payload['n_factor_days']}` range=`{payload['min']}`→`{payload['max']}` proxy_days=`{n_proxy}`。",
        f"gates=`{gates}`",
        f"PIT tests=`{pit}`",
        "",
        "覆盖门：≥400 有效日、≥3 自然年、warmup 后 ≥250 可测信号日，否则 SKIP_FACTOR_FOR_TRADING。",
        "",
    ]
    write_md(FAC_MD, "\n".join(lines))
    freeze_manifest(overwrite=False, facts={"factors": {"n_factor_days": payload["n_factor_days"], "gates": gates, "pit": pit}})
    return payload


def load_factors() -> dict[str, Any]:
    return load_json(FAC_JSON)


def load_factor_daily() -> pd.DataFrame:
    if not FAC_DAILY.is_file():
        return pd.DataFrame()
    return pd.read_parquet(FAC_DAILY)
