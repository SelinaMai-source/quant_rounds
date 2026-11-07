"""Canonical DSR：日收益 → 日 Sharpe → 日样本数。年化只用于展示。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND12_ROOT
from ..round11.stats import dsr, moments, psr
from .common import daily_from_nav, daily_sr, dump_json, load_json, sharpe_of, write_md
from .contract import R12_01_TARGETS
from .trials import load_trials

DSR_MD = ROUND12_ROOT / "07_canonical_dsr.md"
DSR_JSON = ROUND12_ROOT / "dsr" / "dsr.json"


def canonical_dsr(daily: pd.Series, n_trials: int) -> dict[str, Any]:
    x = daily.dropna().astype(float)
    mom = moments(x)
    sr = daily_sr(x)
    n = int(len(x))
    packed = dsr(float(sr or 0.0), n, int(n_trials), mom["skew"], mom["kurt"])
    packed.update({
        "convention": "daily_sr + daily_N",
        "daily_sr": sr,
        "ann_sharpe_display_only": sharpe_of(x),
        "n_days": n,
        "n_trials": int(n_trials),
        "moments": mom,
        "psr_zero": psr(float(sr or 0.0), n, mom["skew"], mom["kurt"], 0.0),
    })
    return packed


def run_dsr_unit_tests(*, seed: int = 17) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = 2000
    zero = pd.Series(rng.normal(0.0, 0.01, size=n))
    pos = pd.Series(rng.normal(0.0008, 0.01, size=n))
    short = pos.iloc[:400]
    d_zero = canonical_dsr(zero, 50)
    d_pos = canonical_dsr(pos, 50)
    d_more = canonical_dsr(pos, 500)
    d_short = canonical_dsr(short, 50)
    tests = {
        "zero_alpha_dsr_not_high": {
            "ok": (d_zero.get("dsr") is None) or (float(d_zero["dsr"]) < 0.90),
            "dsr": d_zero.get("dsr"),
        },
        "positive_alpha_stronger_than_zero": {
            "ok": (d_pos.get("dsr") is not None)
            and (d_zero.get("dsr") is not None)
            and float(d_pos["dsr"]) > float(d_zero["dsr"]),
            "pos": d_pos.get("dsr"),
            "zero": d_zero.get("dsr"),
        },
        "more_trials_does_not_raise_dsr": {
            "ok": (d_more.get("dsr") is not None)
            and (d_pos.get("dsr") is not None)
            and float(d_more["dsr"]) <= float(d_pos["dsr"]) + 1e-9,
            "n50": d_pos.get("dsr"),
            "n500": d_more.get("dsr"),
        },
        "shorter_sample_not_stronger": {
            "ok": (d_short.get("dsr") is not None)
            and (d_pos.get("dsr") is not None)
            and float(d_short["dsr"]) <= float(d_pos["dsr"]) + 1e-6,
            "short": d_short.get("dsr"),
            "full": d_pos.get("dsr"),
        },
        "uses_daily_sr_not_ann": {
            "ok": abs(float(d_pos.get("sr") or 0) - float(d_pos.get("daily_sr") or 0)) < 1e-12,
            "sr": d_pos.get("sr"),
            "daily_sr": d_pos.get("daily_sr"),
            "ann": d_pos.get("ann_sharpe_display_only"),
        },
    }
    passed = all(v.get("ok") for v in tests.values())
    return {"passed": passed, "status": "PASS" if passed else "DSR_IMPLEMENTATION_FAIL", "tests": tests}


def _r12_01_daily() -> pd.Series:
    path = ROUND12_ROOT / "parity" / "R12_01_CORRECTED_V4" / "nav.parquet"
    if path.is_file():
        return daily_from_nav(pd.read_parquet(path))
    from ..paths import ROUND11_ROOT

    alt = ROUND11_ROOT / "cost" / "R11_V4_CORRECTED" / "nav.parquet"
    return daily_from_nav(pd.read_parquet(alt))


def run_dsr(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = ROUND12_ROOT / "dsr"
    dest.mkdir(parents=True, exist_ok=True)
    if resume and DSR_JSON.is_file():
        existing = load_json(DSR_JSON)
        if existing.get("status") in {"DONE", "DSR_IMPLEMENTATION_FAIL"}:
            return existing
    tests = run_dsr_unit_tests()
    trials = load_trials()
    n_set = {
        "documented": int(trials.get("n_documented") or 1315),
        "discovered": int(trials.get("n_full_discovered") or 1339),
        "upper": int(trials.get("n_conservative_upper") or 2678),
        "comparable_recoverable": int(trials.get("n_matrix") or 1311),
    }
    if min(n_set.values()) <= 4:
        raise RuntimeError("n_trials<=4 forbidden")
    daily = _r12_01_daily()
    sensitivity = {name: canonical_dsr(daily, nt) for name, nt in n_set.items()}
    canonical_name = "discovered"
    canonical = sensitivity[canonical_name]
    payload = {
        "status": "DONE" if tests["passed"] else "DSR_IMPLEMENTATION_FAIL",
        "unit_tests": tests,
        "n_trials_set": n_set,
        "canonical_n_choice": canonical_name,
        "canonical_n_reason": "已落盘研究宇宙 = discovered；upper 只作压力，不提名。",
        "CANONICAL_DSR": canonical.get("dsr"),
        "canonical": canonical,
        "sensitivity": {k: {"dsr": v.get("dsr"), "sr_star": v.get("sr_star"), "n_trials": v.get("n_trials")} for k, v in sensitivity.items()},
        "r12_01_targets_ref": dict(R12_01_TARGETS),
        "mixed_ann_sr_with_daily_n": False,
    }
    dump_json(DSR_JSON, payload)
    write_md(
        DSR_MD,
        "\n".join([
            "# Round 12 · Canonical DSR",
            "",
            "锁死：日收益 → 日均/日标准差 → **日 Sharpe** → **日样本数**。年化只展示。",
            f"单测 = **{tests['status']}**。",
            "",
            f"- CANONICAL_DSR (N=discovered={n_set['discovered']}) = **{canonical.get('dsr')}**",
            f"- daily SR = `{canonical.get('daily_sr')}`；n_days = `{canonical.get('n_days')}`",
            f"- SR* = `{canonical.get('sr_star')}`",
            f"- 年化 Sharpe（仅展示）= `{canonical.get('ann_sharpe_display_only')}`",
            "",
            "敏感度：documented / discovered / upper / comparable-recoverable。只提名 discovered。",
            "",
        ]),
    )
    return payload


def load_dsr() -> dict[str, Any]:
    return load_json(DSR_JSON)


__all__ = ["canonical_dsr", "load_dsr", "run_dsr", "run_dsr_unit_tests"]
