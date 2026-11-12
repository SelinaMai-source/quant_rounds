"""registry v2 + 可比宇宙。已有 1311 条日收益直接入矩阵，取消 max_cols=80。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import RESULT_ROOT, ROUND11_ROOT, ROUND12_ROOT
from .common import daily_from_nav, dump_json, load_json, write_md
from .contract import PRE_REGISTERED_IDS
from .deploy import load_deploy

TRIALS_MD = ROUND12_ROOT / "06_trial_registry_v2.md"
REG_CSV = ROUND12_ROOT / "selection" / "research_trial_registry_v2.csv"
MATRIX_PATH = ROUND12_ROOT / "selection" / "daily_matrix.parquet"
TRIALS_JSON = ROUND12_ROOT / "selection" / "trials.json"


def _family_of(trial_id: str, source_round: str) -> str:
    tid = str(trial_id or "")
    src = str(source_round or "")
    if tid.startswith("R12_"):
        return "round12_preregistered"
    if "B03" in tid or tid in {"R9_B03", "R10_00_baseline_v4", "R11_V4"}:
        return "promo_v4"
    if "E02" in tid or tid in {"R8_E02", "R11_V3"}:
        return "promo_v3"
    if "top1_minus_median" in tid or tid in {"R11_V2", "R4_E_top1_minus_median"}:
        return "promo_v2"
    if "DUAL90" in tid or "FS01122" in tid or tid in {"R11_V1"}:
        return "promo_v1"
    if src.startswith("round"):
        return f"round_{src}"
    if src.startswith("replay"):
        return f"replay_{src}"
    prefix = tid.split("_")[0] if tid else src
    return prefix or "unknown"


def _window_of(n_days: Any) -> str:
    try:
        n = int(n_days)
    except (TypeError, ValueError):
        return "unknown"
    if n >= 2000:
        return "full_2018_2026"
    if n >= 1500:
        return "long"
    if n >= 500:
        return "medium"
    return "short"


def _cost_of(path: str) -> str:
    p = str(path).lower()
    if "corrected" in p or "corr_buy" in p or "r12_01" in p:
        return "historical_stamp_tax"
    if "buy" in p and "bp" in p:
        return "buy_stress"
    return "legacy_flat_5bp"


def _load_r11() -> pd.DataFrame:
    csv = ROUND11_ROOT / "trials" / "research_trial_registry.csv"
    if csv.is_file():
        return pd.read_csv(csv)
    return pd.DataFrame()


def _load_r11_index() -> list[dict[str, Any]]:
    path = ROUND11_ROOT / "trials" / "daily_matrix_index.json"
    payload = load_json(path)
    return list(payload.get("rows") or [])


def _add_round12_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deploy = load_deploy()
    extra = []
    for code, row in (deploy.get("results") or {}).items():
        if row.get("status") == "SKIPPED_QUALITY":
            extra.append({
                "trial_id": code,
                "source_round": "round12",
                "path": "",
                "cagr": None,
                "sharpe": None,
                "max_drawdown": None,
                "has_nav": False,
                "has_daily": False,
                "family": "round12_preregistered",
                "parent": row.get("parent") or "R12_01_CORRECTED_V4",
                "window": "full_2018_2026",
                "cost": "historical_stamp_tax",
                "recoverable": False,
                "unrecoverable_reason": row.get("reason") or "SKIPPED",
            })
            continue
        dest = row.get("out_dir") or ""
        nav = Path(dest) / "nav.parquet" if dest else None
        extra.append({
            "trial_id": code,
            "source_round": "round12",
            "path": dest,
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "has_nav": bool(nav and nav.is_file()),
            "has_daily": bool(nav and nav.is_file()),
            "family": row.get("family") or "round12_preregistered",
            "parent": row.get("parent") or "R12_01_CORRECTED_V4",
            "window": "full_2018_2026",
            "cost": _cost_of(dest),
            "recoverable": bool(nav and nav.is_file()),
            "nav_path": str(nav) if nav else "",
        })
    # also register anchors
    for code in ("R12_00_LEGACY_V4", "R12_01_CORRECTED_V4"):
        dest = ROUND12_ROOT / "parity" / code
        nav = dest / "nav.parquet"
        extra.append({
            "trial_id": code,
            "source_round": "round12",
            "path": str(dest),
            "cagr": None,
            "sharpe": None,
            "has_nav": nav.is_file(),
            "has_daily": nav.is_file(),
            "family": "round12_preregistered",
            "parent": "R9_B03",
            "window": "full_2018_2026",
            "cost": _cost_of(code),
            "recoverable": nav.is_file(),
            "nav_path": str(nav) if nav.is_file() else "",
        })
    return rows + extra


def _build_matrix(index_rows: list[dict[str, Any]], extra_navs: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cols: dict[str, pd.Series] = {}
    meta = []
    for row in index_rows:
        path = Path(row.get("nav_path") or "")
        if not path.is_file():
            continue
        try:
            s = daily_from_nav(pd.read_parquet(path))
        except Exception:
            continue
        key = f"{row.get('source_round')}::{row.get('trial_id')}"
        cols[key] = s
        meta.append({**row, "key": key, "n_loaded": int(len(s)), "recoverable": True})
    for row in extra_navs:
        path = Path(row.get("nav_path") or "")
        if not path.is_file():
            continue
        key = f"{row.get('source_round')}::{row.get('trial_id')}"
        if key in cols:
            continue
        try:
            s = daily_from_nav(pd.read_parquet(path))
        except Exception:
            continue
        cols[key] = s
        meta.append({**row, "key": key, "n_loaded": int(len(s)), "recoverable": True})
    if not cols:
        return pd.DataFrame(), meta
    matrix = pd.concat(cols, axis=1).sort_index()
    return matrix, meta


def _universes(reg: pd.DataFrame, matrix: pd.DataFrame) -> dict[str, Any]:
    keys = list(matrix.columns)
    def cols_for(mask: pd.Series) -> list[str]:
        ids = set(mask.astype(str))
        return [k for k in keys if k.split("::", 1)[-1] in ids or k in ids]

    u_window = [k for k in keys if matrix[k].notna().sum() >= 1800]
    promo_ids = set(reg.loc[reg["family"].astype(str).str.startswith("promo_"), "trial_id"].astype(str))
    promo_ids.update({"R9_B03", "R8_E02", "R4_E_top1_minus_median", "R7_FS01122_DUAL90_10_CLEAN", "R11_V1", "R11_V2", "R11_V3", "R11_V4", "R12_00_LEGACY_V4", "R12_01_CORRECTED_V4"})
    u_promo = [k for k in keys if k.split("::", 1)[-1] in promo_ids]
    by_round: dict[str, list[str]] = {}
    for k in keys:
        src = k.split("::", 1)[0]
        by_round.setdefault(src, []).append(k)
    by_family: dict[str, list[str]] = {}
    fam_map = dict(zip(reg["trial_id"].astype(str), reg["family"].astype(str))) if "family" in reg.columns else {}
    for k in keys:
        tid = k.split("::", 1)[-1]
        fam = fam_map.get(tid) or k.split("::", 1)[0]
        by_family.setdefault(fam, []).append(k)
    return {
        "U_WINDOW": {"n": len(u_window), "keys": u_window, "rule": "同一预测窗：日收益天数≥1800"},
        "U_PROMO": {"n": len(u_promo), "keys": u_promo, "rule": "v1→v4 晋升家族及当时认真 challenger"},
        "U_ROUND": {src: {"n": len(v), "keys": v} for src, v in by_round.items()},
        "U_FAMILY": {fam: {"n": len(v), "keys": v} for fam, v in by_family.items() if len(v) >= 4},
    }


def run_trials(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = ROUND12_ROOT / "selection"
    dest.mkdir(parents=True, exist_ok=True)
    if resume and TRIALS_JSON.is_file() and MATRIX_PATH.is_file() and REG_CSV.is_file():
        existing = load_json(TRIALS_JSON)
        if existing.get("status") == "DONE" and existing.get("n_matrix", 0) >= 1000:
            return existing
    r11 = _load_r11()
    index_rows = _load_r11_index()
    if r11.empty:
        r11 = pd.DataFrame(index_rows)
    r11["family"] = [ _family_of(r.get("trial_id"), r.get("source_round")) for r in r11.to_dict("records") ]
    r11["parent"] = r11.get("source_round")
    r11["window"] = [ _window_of(r.get("n") if "n" in r11.columns else None) for r in r11.to_dict("records") ]
    if "n" not in r11.columns:
        nmap = { (row.get("source_round"), row.get("trial_id")): row.get("n") for row in index_rows }
        r11["window"] = [
            _window_of(nmap.get((r.get("source_round"), r.get("trial_id"))))
            for r in r11.to_dict("records")
        ]
    r11["cost"] = r11["path"].astype(str).map(_cost_of) if "path" in r11.columns else "legacy_flat_5bp"
    nav_set = { (row.get("source_round"), str(row.get("trial_id"))) for row in index_rows }
    r11["recoverable"] = [
        (r.get("source_round"), str(r.get("trial_id"))) in nav_set or bool(r.get("has_nav"))
        for r in r11.to_dict("records")
    ]
    extra = _add_round12_rows([])
    extra_df = pd.DataFrame(extra)
    if not extra_df.empty:
        r11 = pd.concat([r11, extra_df], ignore_index=True, sort=False)
    r11.to_csv(REG_CSV, index=False)
    matrix, meta = _build_matrix(index_rows, extra)
    matrix.to_parquet(MATRIX_PATH)
    universes = _universes(r11, matrix)
    n_doc = int(r11["cagr"].notna().sum()) if "cagr" in r11.columns else int(len(r11))
    n_disc = int(len(r11.drop_duplicates(["source_round", "trial_id"]))) if "trial_id" in r11.columns else int(len(r11))
    n_upper = int(max(n_disc * 2, 2678))
    payload = {
        "status": "DONE",
        "n_documented": n_doc,
        "n_full_discovered": n_disc,
        "n_conservative_upper": n_upper,
        "n_matrix": int(matrix.shape[1]),
        "n_days": int(matrix.shape[0]),
        "csv": str(REG_CSV),
        "matrix": str(MATRIX_PATH),
        "universes": {k: ({"n": v.get("n"), "rule": v.get("rule")} if "n" in v else {kk: vv.get("n") for kk, vv in v.items()}) for k, v in universes.items()},
        "universe_keys_path": str(dest / "universe_keys.json"),
        "max_cols_80_removed": True,
        "do_not_extrapolate_to_1339": True,
        "round12_ids_registered": list(PRE_REGISTERED_IDS),
        "unrecoverable_note": "ABORT / 无账本 / 暗试标 unrecoverable；不把 1339 塞进一个矩阵。",
    }
    dump_json(dest / "universe_keys.json", {k: (v.get("keys") if "keys" in v else {kk: vv.get("keys") for kk, vv in v.items()}) for k, v in universes.items()})
    dump_json(TRIALS_JSON, payload)
    write_md(
        TRIALS_MD,
        "\n".join([
            "# Round 12 · research trial registry v2",
            "",
            f"- documented = **{n_doc}**",
            f"- discovered = **{n_disc}**",
            f"- conservative upper = **{n_upper}**",
            f"- 日收益矩阵列数 = **{matrix.shape[1]}**（已取消 max_cols=80）",
            f"- U_WINDOW n = {(universes.get('U_WINDOW') or {}).get('n')}",
            f"- U_PROMO n = {(universes.get('U_PROMO') or {}).get('n')}",
            "",
            "禁止把可恢复子集外推到 1339/2678。暗试仍 unrecoverable。",
            "Round12 新策略已登记进 v2。",
            "",
        ]),
    )
    return payload


def load_trials() -> dict[str, Any]:
    return load_json(TRIALS_JSON)


def load_matrix() -> pd.DataFrame:
    if MATRIX_PATH.is_file():
        return pd.read_parquet(MATRIX_PATH)
    return pd.DataFrame()


def load_universe_keys() -> dict[str, Any]:
    return load_json(ROUND12_ROOT / "selection" / "universe_keys.json")


__all__ = ["load_matrix", "load_trials", "load_universe_keys", "run_trials"]
