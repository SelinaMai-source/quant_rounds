"""TRUE PBO 2.0（取消 max_cols=80）+ staircase / SPA。禁止把 loss proxy 叫 PBO。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND11_ROOT, ROUND12_ROOT
from ..round11.common import daily_from_nav, load_legacy_dailies
from ..round11.stats import hac_mean_pvalue, selection_pbo, spa_test, staircase
from .common import dump_json, load_json, pct, write_md
from .deploy import load_deploy
from .dsr import load_dsr
from .placebo import load_placebo
from .timing_placebo import load_timing
from .trials import load_matrix, load_trials, load_universe_keys

PBO_MD = ROUND12_ROOT / "08_true_pbo.md"
PBO_JSON = ROUND12_ROOT / "pbo" / "pbo.json"
STAIR_MD = ROUND12_ROOT / "09_staircase.md"
STAIR_JSON = ROUND12_ROOT / "staircase" / "staircase.json"
STATS_MD = ROUND12_ROOT / "10_stats_summary.md"
STATS_JSON = ROUND12_ROOT / "stats" / "stats.json"


def _pbo_for(matrix: pd.DataFrame, keys: list[str], label: str) -> dict[str, Any]:
    cols = [k for k in keys if k in matrix.columns]
    if len(cols) < 8:
        return {
            "status": "TRUE_PBO_NOT_FULLY_IDENTIFIABLE",
            "label": "TRUE_PBO_NOT_FULLY_IDENTIFIABLE",
            "universe": label,
            "n_strats": len(cols),
            "reason": f"comparable universe too small: {len(cols)}",
        }
    out = selection_pbo(matrix[cols])
    out["universe"] = label
    out["n_requested"] = len(keys)
    return out


def run_pbo(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = ROUND12_ROOT / "pbo"
    dest.mkdir(parents=True, exist_ok=True)
    if resume and PBO_JSON.is_file():
        existing = load_json(PBO_JSON)
        if existing.get("status") == "DONE":
            return existing
    matrix = load_matrix()
    keys = load_universe_keys()
    trials = load_trials()
    u_window = keys.get("U_WINDOW") or []
    u_promo = keys.get("U_PROMO") or []
    u_round = keys.get("U_ROUND") or {}
    u_family = keys.get("U_FAMILY") or {}
    by_universe = {
        "U_WINDOW": _pbo_for(matrix, list(u_window), "U_WINDOW"),
        "U_PROMO": _pbo_for(matrix, list(u_promo), "U_PROMO"),
    }
    within_round = {src: _pbo_for(matrix, list(v or []), f"U_ROUND::{src}") for src, v in u_round.items() if isinstance(v, list) and len(v) >= 8}
    within_family = {fam: _pbo_for(matrix, list(v or []), f"U_FAMILY::{fam}") for fam, v in u_family.items() if isinstance(v, list) and len(v) >= 8}
    global_identifiable = False
    payload = {
        "status": "DONE",
        "method": "TRUE_SELECTION_BASED_PBO",
        "definition": "每 split 在 IS 选胜者 → 看其 OOS 相对秩 → PBO = 劣于中位比例。禁止把 loss proxy 叫 PBO。",
        "n_matrix": int(matrix.shape[1]) if not matrix.empty else 0,
        "max_cols_80_removed": True,
        "by_universe": by_universe,
        "within_round": {k: {kk: vv for kk, vv in v.items() if kk != "note"} for k, v in within_round.items()},
        "within_family": {k: {kk: vv for kk, vv in v.items() if kk != "note"} for k, v in within_family.items()},
        "TRUE_GLOBAL_PBO": None,
        "TRUE_GLOBAL_PBO_STATUS": "TRUE_GLOBAL_PBO_NOT_IDENTIFIABLE",
        "do_not_extrapolate_to_1339_or_2678": True,
        "n_trials": {
            "documented": trials.get("n_documented"),
            "discovered": trials.get("n_full_discovered"),
            "upper": trials.get("n_conservative_upper"),
            "comparable_recoverable": trials.get("n_matrix"),
        },
        "r11_truncated_n": 80,
        "note": "全宇宙仍含暗试/ABORT/无账本，不得把可恢复子集外推到 1339/2678。",
    }
    dump_json(PBO_JSON, payload)
    uw = by_universe["U_WINDOW"]
    up = by_universe["U_PROMO"]
    write_md(
        PBO_MD,
        "\n".join([
            "# Round 12 · TRUE PBO 2.0",
            "",
            "取消 `max_cols=80`。矩阵列数 = "
            f"**{payload['n_matrix']}**（Round11 截断 n=80）。",
            "",
            f"- U_WINDOW PBO = **{uw.get('pbo')}** (n={uw.get('n_strats')}, `{uw.get('label')}`)",
            f"- U_PROMO PBO = **{up.get('pbo')}** (n={up.get('n_strats')}, `{up.get('label')}`)",
            f"- 全宇宙 = **TRUE_GLOBAL_PBO_NOT_IDENTIFIABLE**（不得外推到 1339/2678）",
            "",
            "这是 selection-based PBO，不是 probability-of-loss proxy。",
            "",
        ]),
    )
    _ = global_identifiable
    return payload


def _classify_step(hac: dict[str, Any], pbo_sensitive: bool) -> str:
    p = hac.get("p_two_sided")
    mu = hac.get("mean")
    if p is None or mu is None:
        return "NOT_IDENTIFIABLE"
    if pbo_sensitive:
        return "SELECTION_SENSITIVE"
    if float(mu) > 0 and float(p) < 0.10:
        return "STRUCTURAL_INCREMENT"
    if float(mu) > 0:
        return "WEAK"
    return "WEAK"


def run_staircase(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = ROUND12_ROOT / "staircase"
    dest.mkdir(parents=True, exist_ok=True)
    if resume and STAIR_JSON.is_file():
        existing = load_json(STAIR_JSON)
        if existing.get("status") == "DONE":
            return existing
    legacy = load_legacy_dailies()
    corrected = {}
    for key in ("v1", "v2", "v3", "v4"):
        path = ROUND11_ROOT / "cost" / f"R11_{key.upper()}_CORRECTED" / "nav.parquet"
        if path.is_file():
            corrected[key] = daily_from_nav(pd.read_parquet(path))
    pbo = load_json(PBO_JSON)
    promo_pbo = ((pbo.get("by_universe") or {}).get("U_PROMO") or {}).get("pbo")
    sensitive = promo_pbo is not None and float(promo_pbo) >= 0.50
    stair_l = staircase(legacy)
    stair_c = staircase(corrected) if len(corrected) == 4 else {"status": "NEED_CORRECTED"}
    classes = {}
    for name, block in stair_l.items():
        classes[f"legacy::{name}"] = _classify_step(block, sensitive)
    if isinstance(stair_c, dict) and "v2_minus_v1" in stair_c:
        for name, block in stair_c.items():
            classes[f"corrected::{name}"] = _classify_step(block, sensitive)
    spa_l = spa_test(legacy["v4"], pd.DataFrame({k: v for k, v in legacy.items() if k != "v4"}))
    spa_c = None
    if len(corrected) >= 2 and "v4" in corrected:
        spa_c = spa_test(corrected["v4"], pd.DataFrame({k: v for k, v in corrected.items() if k != "v4"}))
    payload = {
        "status": "DONE",
        "legacy": stair_l,
        "corrected": stair_c,
        "classification": classes,
        "spa_legacy": spa_l,
        "spa_corrected": spa_c,
        "promo_pbo": promo_pbo,
        "selection_sensitive_flag": sensitive,
        "pit_corrected_run": False,
        "note": "R12_02 未达到质量门槛时，主要统计只在 LEGACY 与 COST_CORRECTED 上各跑一遍。",
    }
    dump_json(STAIR_JSON, payload)
    write_md(
        STAIR_MD,
        "\n".join([
            "# Round 12 · staircase / SPA",
            "",
            "v1→v2→v3→v4：raw delta + HAC + family max-stat / SPA。不是推广许可。",
            "",
            "| 步 | 分类 |",
            "|---|---|",
            *[f"| `{k}` | **{v}** |" for k, v in classes.items()],
            "",
            f"U_PROMO PBO = {promo_pbo} → selection_sensitive_flag={sensitive}。",
            "",
        ]),
    )
    return payload


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = ROUND12_ROOT / "stats"
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "DONE",
        "pbo": load_json(PBO_JSON),
        "dsr": load_dsr(),
        "staircase": load_json(STAIR_JSON),
        "deploy": {k: {kk: vv for kk, vv in row.items() if kk in {"cagr", "sharpe", "max_drawdown", "status"}}
                   for k, row in (load_deploy().get("results") or {}).items()},
        "placebo": load_placebo(),
        "timing": load_timing(),
        "not_forward_oos": True,
    }
    dump_json(STATS_JSON, payload)
    write_md(STATS_MD, "# Round 12 · 统计汇总\n\n见 PBO / DSR / staircase / placebo 分报告。CPCV 不是 forward OOS。\n")
    return payload


def load_pbo() -> dict[str, Any]:
    return load_json(PBO_JSON)


def load_staircase() -> dict[str, Any]:
    return load_json(STAIR_JSON)


def load_stats() -> dict[str, Any]:
    return load_json(STATS_JSON)


__all__ = ["load_pbo", "load_staircase", "load_stats", "run_pbo", "run_staircase", "run_stats"]
