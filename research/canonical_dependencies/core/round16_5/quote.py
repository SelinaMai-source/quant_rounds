"""报价 / PIT 诊断。不编造 bid/ask。不写 secrets。"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_5_ROOT, ROUND16_ROOT
from ..round16.chain import load_chain
from ..round16.common import redact
from ..round16.factors import load_factors
from ..round16.quality import load_quality
from .common import dump_json, load_json, write_md
from .contract import BID_ASK_NOT_AVAILABLE, PIT_TIMESTAMP_UNVERIFIED, RF_FALLBACK, RF_GRID
from .freeze import require_frozen

Q_JSON = ROUND16_5_ROOT / "quote" / "quote.json"
Q_MD = ROUND16_5_ROOT / "08_quote_quality.md"
PIT_JSON = ROUND16_5_ROOT / "pit" / "pit.json"
PIT_MD = ROUND16_5_ROOT / "09_pit.md"

LOCAL_PROBE = (
    Path("/home/mxy/AmazingData"),
    ROUND16_ROOT / "data",
    Path("/data1/stock-mxy/课题/一进二策略/result/round16/data"),
    Path("/data1/stock-mxy/data"),
)


def _scan_bid_ask_files(root: Path, *, limit: int = 40, depth: int = 3) -> dict[str, Any]:
    if not root.exists():
        return {"path": str(root), "exists": False, "bid_ask_files": []}
    hits = []
    keys = ("bid", "ask", "snapshot", "quote")
    try:
        for dirpath, dirnames, filenames in __import__("os").walk(root):
            rel_depth = len(Path(dirpath).relative_to(root).parts)
            if rel_depth >= depth:
                dirnames[:] = []
            for name in filenames:
                low = name.lower()
                if any(k in low for k in keys) and Path(name).suffix.lower() in {".parquet", ".csv", ".feather", ".pkl"}:
                    hits.append(str(Path(dirpath) / name))
                    if len(hits) >= limit:
                        return {"path": str(root), "exists": True, "n_hits": len(hits), "bid_ask_files_sample": hits[:10]}
    except Exception as exc:
        return {"path": str(root), "exists": True, "error": type(exc).__name__, "bid_ask_files": []}
    return {"path": str(root), "exists": True, "n_hits": len(hits), "bid_ask_files_sample": hits[:10]}


def _probe_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {"api": "query_snapshot", "called": False}
    try:
        import AmazingData as ad

        fn = getattr(ad.MarketData, "query_snapshot", None)
        out["present"] = fn is not None
        if fn is not None:
            try:
                out["signature"] = str(inspect.signature(fn))
            except (TypeError, ValueError):
                out["signature"] = "UNAVAILABLE"
            out["doc_head"] = ((inspect.getdoc(fn) or "").splitlines() or [""])[0][:300]
        try:
            from ..round16.sdk import ensure_login, to_frame

            _, meta = ensure_login()
            out["login_ok"] = bool((meta or {}).get("ok") or (meta or {}).get("status") == "OK" or meta)
            out["login_redacted"] = redact({k: (meta or {}).get(k) for k in ("status", "ok", "error", "classify")})
            md = ad.MarketData()
            out["called"] = True
            raw = None
            try:
                raw = md.query_snapshot(code=["510300C"], period=86400000)
            except TypeError:
                try:
                    raw = md.query_snapshot(["510300C"])
                except Exception as exc:
                    out["call_error"] = type(exc).__name__
            except Exception as exc:
                out["call_error"] = type(exc).__name__
            frame = to_frame(raw) if raw is not None else pd.DataFrame()
            out["n_rows"] = int(len(frame))
            out["columns"] = [str(c) for c in frame.columns][:40]
            cols_l = {str(c).lower() for c in frame.columns}
            out["has_bid"] = any("bid" in c for c in cols_l)
            out["has_ask"] = any("ask" in c for c in cols_l)
            if frame.empty:
                out["status"] = BID_ASK_NOT_AVAILABLE
        except Exception as exc:
            out["probe_error"] = type(exc).__name__
            out["status"] = BID_ASK_NOT_AVAILABLE
    except Exception as exc:
        out["present"] = False
        out["import_error"] = type(exc).__name__
        out["status"] = BID_ASK_NOT_AVAILABLE
    if "has_bid" in out and not (out.get("has_bid") and out.get("has_ask")):
        out["status"] = BID_ASK_NOT_AVAILABLE
    return redact(out)


def _chain_diagnostics(chain: pd.DataFrame) -> dict[str, Any]:
    if chain.empty:
        return {"status": "EMPTY"}
    last = pd.to_numeric(chain.get("last"), errors="coerce") if "last" in chain.columns else pd.Series(dtype=float)
    bid = pd.to_numeric(chain["bid"], errors="coerce") if "bid" in chain.columns else pd.Series(np.nan, index=chain.index)
    ask = pd.to_numeric(chain["ask"], errors="coerce") if "ask" in chain.columns else pd.Series(np.nan, index=chain.index)
    vol = pd.to_numeric(chain["volume"], errors="coerce") if "volume" in chain.columns else pd.Series(np.nan, index=chain.index)
    iv = pd.to_numeric(chain["iv"], errors="coerce") if "iv" in chain.columns else pd.Series(dtype=float)
    has_ba = bid.notna() & ask.notna() if len(bid) else pd.Series(False, index=chain.index)
    stale = pd.Series(False, index=chain.index)
    if set(["open", "high", "low"]).issubset(chain.columns):
        stale = last.gt(0) & chain["open"].eq(chain["high"]) & chain["high"].eq(chain["low"])
    extreme_iv = (iv > 3.0) | (iv < 0.01) if len(iv) else pd.Series(False, index=chain.index)
    return {
        "n": int(len(chain)),
        "bilateral_rate": float(has_ba.mean()) if len(chain) else 0.0,
        "zero_volume_rate": float((vol.fillna(0) <= 0).mean()) if len(chain) else None,
        "stale_rate": float(stale.mean()) if len(chain) else None,
        "extreme_iv_rate": float(extreme_iv.mean()) if len(iv) else None,
        "bid_ask_status": BID_ASK_NOT_AVAILABLE if float(has_ba.mean() or 0) < 0.01 else "PRESENT",
    }


def run_quote(**_: Any) -> dict[str, Any]:
    require_frozen()
    chain = load_chain()
    quality = load_quality()
    fac = load_factors()
    local = [_scan_bid_ask_files(p) for p in LOCAL_PROBE]
    snap = _probe_snapshot()
    diag = _chain_diagnostics(chain)
    ba = diag.get("bid_ask_status") == BID_ASK_NOT_AVAILABLE and not snap.get("has_bid")
    if ba:
        verdict = "LAST_PRICE_PROXY_ONLY"
    elif diag.get("bilateral_rate", 0) >= 0.8:
        verdict = "QUOTE_QUALITY_SUPPORTED"
    elif diag.get("bilateral_rate", 0) >= 0.05:
        verdict = "QUOTE_QUALITY_PARTIALLY_SUPPORTED"
    else:
        verdict = "LAST_PRICE_PROXY_ONLY"
    if quality.get("valid_quote_rate") is not None and float(quality.get("valid_quote_rate") or 0) < 0.5:
        verdict = "QUOTE_QUALITY_FAIL"
    rf = {
        "source": (fac.get("iv") or {}).get("rf_source") if isinstance(fac.get("iv"), dict) else "FALLBACK_CONSTANT_0.02",
        "fallback": RF_FALLBACK,
        "grid_if_no_history": list(RF_GRID),
        "historical_rate_available": False,
        "signal_agreement_note": "无历史利率序列则只报 1%/2%/3% 信号一致率（若 IV 可重算）；本轮不重算全样本 IV。",
    }
    payload = {
        "status": "DONE",
        "verdict": verdict,
        "bid_ask": BID_ASK_NOT_AVAILABLE if ba else "PRESENT",
        "iv_limitation": "last_close_only",
        "no_fabricated_quotes": True,
        "not_full_iv_surface": ba,
        "snapshot_probe": snap,
        "local_inventory": local,
        "chain": diag,
        "round16_quality": {k: quality.get(k) for k in ("status", "valid_quote_rate", "bilateral_rate", "quote_quality_default", "notes")},
        "rf": rf,
        "no_pnl_selected_filters": True,
        "secrets_redacted": True,
    }
    dump_json(Q_JSON, payload)
    write_md(
        Q_MD,
        "\n".join([
            "# Round 16.5 · 08 报价质量",
            "",
            f"裁决=`{verdict}` bid/ask=`{payload['bid_ask']}` IV=`last_close_only`。",
            "不编造买卖价。无双边不得写成完整 IV 面。报告不含 password/token/host。",
            f"AmazingData query_snapshot present=`{snap.get('present')}` called=`{snap.get('called')}` has_bid=`{snap.get('has_bid')}`。",
            "",
        ]),
    )
    return payload


def run_pit(**_: Any) -> dict[str, Any]:
    require_frozen()
    fac = load_factors()
    pit16 = fac.get("pit") or {}
    chain = load_chain()
    unverified = False
    detail: dict[str, Any] = {"round16_pit": pit16}
    if chain.empty or "available_at" not in chain.columns:
        unverified = True
        detail["missing_available_at"] = True
    else:
        close = pd.to_datetime(chain["trade_date"]) + pd.Timedelta(hours=15)
        av = pd.to_datetime(chain["available_at"])
        detail["available_at_le_close"] = bool((av <= close + pd.Timedelta(minutes=1)).all())
        # 决策 = D 收盘后；下单 = D+1 开盘。无法从日 K 核实盘中 snapshot 时点。
        detail["decision_vs_order"] = "decision=D_close; order=D+1_open; snapshot_intraday_unverified"
        unverified = True
    status = PIT_TIMESTAMP_UNVERIFIED if unverified else ("PASS" if pit16.get("status") == "PASS" else "FAIL")
    payload = {
        "status": status,
        "rule": "available_at <= decision < order",
        "tz": "Asia/Shanghai",
        "entry_unchanged": True,
        "same_open_execution_optimism": True,
        "deployment_feasibility_separate": True,
        "detail": detail,
        "note": "相对优势与 DEPLOYMENT_FEASIBILITY / SAME_OPEN_EXECUTION_OPTIMISM 分开写。不改入场时点。",
    }
    dump_json(PIT_JSON, payload)
    write_md(
        PIT_MD,
        "\n".join([
            "# Round 16.5 · 09 PIT",
            "",
            f"status=`{status}`。规则 `available_at <= decision < order`。",
            "日 K last_close 无法核实盘中 snapshot。不改入场时点。",
            "",
        ]),
    )
    return payload


def load_quote() -> dict[str, Any]:
    return load_json(Q_JSON)


def load_pit() -> dict[str, Any]:
    return load_json(PIT_JSON)
