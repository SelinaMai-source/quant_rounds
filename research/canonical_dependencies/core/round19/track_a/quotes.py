"""a-quotes：报价质量。不编造 bid/ask。无双边 → LAST_PRICE_PROXY_ONLY。"""

from __future__ import annotations

from typing import Any

from ...round16.chain import load_chain
from ...round16.factors import load_factors
from ...round16.quality import load_quality
from ...round16_5.contract import BID_ASK_NOT_AVAILABLE, RF_FALLBACK, RF_GRID
from ...round16_5.quote import LOCAL_PROBE, _chain_diagnostics, _probe_snapshot, _scan_bid_ask_files
from ..common import dump_json, write_md
from ..contract import LAST_PRICE_PROXY_ONLY
from .support import QUOTE_JSON, resume_ok, skip_result, ta_md


def run_quotes(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    existing = resume_ok(QUOTE_JSON) if resume else None
    if existing is not None:
        return existing
    try:
        chain = load_chain()
        quality = load_quality()
        fac = load_factors()
    except Exception as exc:
        payload = skip_result("a-quotes", f"{type(exc).__name__}:{exc}")
        dump_json(QUOTE_JSON, payload)
        return payload
    local = [_scan_bid_ask_files(p) for p in LOCAL_PROBE]
    snap = _probe_snapshot()
    diag = _chain_diagnostics(chain)
    ba = diag.get("bid_ask_status") == BID_ASK_NOT_AVAILABLE and not snap.get("has_bid")
    if ba:
        verdict = LAST_PRICE_PROXY_ONLY
    elif diag.get("bilateral_rate", 0) >= 0.8:
        verdict = "QUOTE_QUALITY_SUPPORTED"
    elif diag.get("bilateral_rate", 0) >= 0.05:
        verdict = "QUOTE_QUALITY_PARTIALLY_SUPPORTED"
    else:
        verdict = LAST_PRICE_PROXY_ONLY
    if quality.get("valid_quote_rate") is not None and float(quality.get("valid_quote_rate") or 0) < 0.5:
        verdict = "QUOTE_QUALITY_FAIL"
    payload = {
        "status": "DONE",
        "verdict": verdict,
        "bid_ask": BID_ASK_NOT_AVAILABLE if ba else "PRESENT",
        "iv_limitation": "last_close_only",
        "no_fabricated_quotes": True,
        "not_full_iv_surface": True if ba else bool(diag.get("bilateral_rate", 0) < 0.8),
        "snapshot_probe": snap,
        "local_inventory": local,
        "chain": diag,
        "round16_quality": {
            k: quality.get(k) for k in ("status", "valid_quote_rate", "bilateral_rate", "quote_quality_default", "notes")
        },
        "rf": {
            "fallback": RF_FALLBACK,
            "grid_if_no_history": list(RF_GRID),
            "source": (fac.get("iv") or {}).get("rf_source") if isinstance(fac.get("iv"), dict) else "FALLBACK_CONSTANT_0.02",
        },
        "historical_publish_vs_redownload": "2018 报价历史发布时间 ≠ 2026 重新下载时间。优先本地缓存，禁止虚构 bid/ask。",
    }
    dump_json(QUOTE_JSON, payload)
    write_md(
        ta_md("quotes.md"),
        "\n".join(
            [
                "# Round19 Track A · a-quotes",
                "",
                f"裁决=`{verdict}` bid/ask=`{payload['bid_ask']}`。不编造买卖价。无双边不得写成完整 IV 面。",
                "",
            ]
        ),
    )
    return payload
