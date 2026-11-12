"""报价质量门：不把缺失填 0；无双边不把 last 当可靠 IV 输入。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_ROOT
from .chain import load_chain
from .common import dump_json, load_json, write_md

Q_JSON = ROUND16_ROOT / "quality" / "quality.json"
Q_MD = ROUND16_ROOT / "03_option_quote_quality.md"


def run_quality(**_: Any) -> dict[str, Any]:
    chain = load_chain()
    if chain.empty:
        payload = {"status": "EMPTY", "valid_quote_rate": 0.0}
        dump_json(Q_JSON, payload)
        write_md(Q_MD, "# Round16 · 03 报价质量\n\nEMPTY。\n")
        return payload
    n = int(len(chain))
    last = pd.to_numeric(chain["last"], errors="coerce")
    strike = pd.to_numeric(chain["strike"], errors="coerce")
    ttm = pd.to_numeric(chain["days_to_expiry"], errors="coerce")
    spot = pd.to_numeric(chain["underlying_price"], errors="coerce")
    bid = pd.to_numeric(chain["bid"], errors="coerce")
    ask = pd.to_numeric(chain["ask"], errors="coerce")
    vol = pd.to_numeric(chain["volume"], errors="coerce")
    has_ba = bid.notna() & ask.notna()
    ba_ok = has_ba & bid.ge(0) & ask.ge(bid)
    last_ok = last.gt(0) & np.isfinite(last)
    gates = (
        last_ok
        & strike.gt(0)
        & ttm.gt(0)
        & spot.gt(0)
        & np.isfinite(strike)
        & np.isfinite(spot)
    )
    zero_vol = vol.fillna(0).le(0)
    stale = last_ok & (chain.get("open").eq(chain.get("high")) & chain.get("high").eq(chain.get("low"))) if "open" in chain.columns else pd.Series(False, index=chain.index)
    payload = {
        "status": "DONE",
        "n": n,
        "valid_quote_rate": float(gates.mean()) if n else 0.0,
        "invalid_quote_rate": float((~gates).mean()) if n else 1.0,
        "stale_quote_rate": float(stale.mean()) if n else None,
        "zero_volume_rate": float(zero_vol.mean()) if n else None,
        "bilateral_rate": float(has_ba.mean()) if n else 0.0,
        "bilateral_ok_rate": float(ba_ok.mean()) if n else 0.0,
        "missing_filled_with_zero": False,
        "last_used_as_reliable_iv": False,
        "last_allowed_descriptive_with_flag": True,
        "quote_quality_default": "last_close_only" if float(has_ba.mean() or 0) < 0.05 else "bid_ask",
        "notes": "日 K 无 bid/ask。last 仅描述性 IV，标记 last_close_only。缺失不填 0。",
    }
    dump_json(Q_JSON, payload)
    write_md(
        Q_MD,
        "\n".join([
            "# Round 16 · 03 期权报价质量",
            "",
            f"n=`{n}` valid_quote_rate=`{payload['valid_quote_rate']:.4f}` invalid=`{payload['invalid_quote_rate']:.4f}`。",
            f"bilateral_rate=`{payload['bilateral_rate']}` zero_volume=`{payload['zero_volume_rate']}` stale=`{payload['stale_quote_rate']}`。",
            "missing→0：**否**。无可靠双边：**不**把 last 当可靠 IV 输入；描述性计算标记 `last_close_only`。",
            "",
        ]),
    )
    return payload


def load_quality() -> dict[str, Any]:
    return load_json(Q_JSON)
