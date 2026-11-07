"""有符号 NAV 增量。禁止对相对变化取绝对值。"""

from __future__ import annotations

from typing import Any


def nav_improvement(candidate_nav: float | None, baseline_nav: float | None) -> float | None:
    if candidate_nav is None or baseline_nav is None:
        return None
    base = float(baseline_nav)
    if base == 0.0:
        return None
    return float(candidate_nav) / base - 1.0


def signed_nav_metrics(candidate_nav: float | None, baseline_nav: float | None, *, min_rel: float) -> dict[str, Any]:
    improvement = nav_improvement(candidate_nav, baseline_nav)
    if candidate_nav is None or baseline_nav is None or improvement is None:
        return {
            "nav_improvement": None,
            "delta_terminal_nav_rmb": None,
            "relative_terminal_nav_change": None,
            "nav_improvement_ok": None,
            "status": "NOT_EVALUATED",
        }
    delta_rmb = float(candidate_nav) - float(baseline_nav)
    ok = bool(improvement >= float(min_rel))
    return {
        "nav_improvement": float(improvement),
        "delta_terminal_nav_rmb": float(delta_rmb),
        "relative_terminal_nav_change": float(improvement),
        "nav_improvement_ok": ok,
        "status": "PASS" if ok else "FAIL",
    }
