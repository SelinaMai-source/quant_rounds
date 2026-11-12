"""Stage 0 终端块。主报告由 Track C 后续代理写。不写「写入时间」。不改 CURRENT。"""

from __future__ import annotations

from typing import Any

from .common import load_json
from .contract import CURRENT_BASELINE, MANIFEST_MD, OPTION_CANDIDATE, PARITY_MD
from .freeze import FREEZE_JSON
from .parity import PARITY_JSON


def print_stage0_block(parity: dict[str, Any] | None = None, freeze: dict[str, Any] | None = None) -> None:
    parity = parity or load_json(PARITY_JSON)
    freeze = freeze or load_json(FREEZE_JSON)
    obs = parity.get("observed") or {}
    views = parity.get("cagr_views") or {}
    cal = (views.get("calendar_365_25") or {}).get("cagr")
    t252 = (views.get("trading_252_cagr_of") or {}).get("cagr")
    ta = freeze.get("track_a") or {}
    tb = freeze.get("track_b") or {}
    print("", flush=True)
    print("===== ROUND19 STAGE 0 =====", flush=True)
    print(f"CURRENT={CURRENT_BASELINE} CANDIDATE={OPTION_CANDIDATE} AUTO_PROMOTE=False", flush=True)
    print(f"parity={parity.get('status')} formal_promotion_blocked={parity.get('formal_promotion_blocked')}", flush=True)
    print(
        f"observed CAGR={obs.get('cagr')} Sharpe={obs.get('sharpe')} "
        f"MaxDD={obs.get('max_drawdown')} trades={obs.get('completed_trades')}",
        flush=True,
    )
    print(
        f"n_trading_observations={parity.get('n_trading_observations')} "
        f"completed_trades={parity.get('n_completed_trades')}",
        flush=True,
    )
    print(f"CAGR calendar365.25={cal} vs cagr_of_252={t252}", flush=True)
    print(f"manifest_a sha256={ta.get('sha256')} path={ta.get('path')}", flush=True)
    print(f"manifest_b sha256={tb.get('sha256')} path={tb.get('path')}", flush=True)
    print(f"parity_md={PARITY_MD}", flush=True)
    print(f"manifest_md={MANIFEST_MD}", flush=True)
    print("STOP after Stage 0. Track A/B/C not run.", flush=True)
    print("===== END STAGE 0 =====", flush=True)


def run_report(**kwargs: Any) -> dict[str, Any]:
    from .track_c.report import run_report as _run

    return _run(**kwargs)


def print_complete_block(payload: dict[str, Any] | None = None) -> None:
    from .track_c.report import print_part10

    print_part10(payload)
