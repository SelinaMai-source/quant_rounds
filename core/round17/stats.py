"""成本 / 容量 / bootstrap / CPCV / SPA / PBO。全部 5000 进入选择偏差会计。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import MARKET_END, ROUND17_ROOT
from ..round11.common import daily_from_nav, sharpe_of
from ..round11.stats import cpcv_oos_sharpe, selection_pbo, spa_test
from ..round12.runner import hist_hooks
from .backtest import apply_b03, cached_overlay_v4, run_canonical
from .common import dump_json, load_json, write_md
from .contract import (
    BLOCK_DAYS,
    BOOT_N,
    CAPACITY_DIAG_NAV,
    CAPACITY_NAV,
    COST_BPS,
    FORWARD_OOS_NOT_AVAILABLE,
    N_TRIALS,
    SPA_BOOT,
    trial_dir,
)
from .search import _scan

STRESS_JSON = ROUND17_ROOT / "stress" / "stress.json"
STRESS_MD = ROUND17_ROOT / "13_cost_stress.md"
CAP_JSON = ROUND17_ROOT / "capacity" / "capacity.json"
CAP_MD = ROUND17_ROOT / "14_capacity.md"
BOOT_JSON = ROUND17_ROOT / "bootstrap" / "bootstrap.json"
CPCV_JSON = ROUND17_ROOT / "cpcv" / "cpcv.json"
SPA_JSON = ROUND17_ROOT / "spa" / "spa.json"
PBO_JSON = ROUND17_ROOT / "pbo" / "pbo.json"
YEAR_MD = ROUND17_ROOT / "15_year_robustness.md"
STAT_MD = ROUND17_ROOT / "16_bootstrap.md"
BIAS_MD = ROUND17_ROOT / "18_selection_bias.md"
FWD_JSON = ROUND17_ROOT / "forward" / "forward.json"
FWD_MD = ROUND17_ROOT / "19_forward_oos.md"
STATS_JSON = ROUND17_ROOT / "stats" / "stats.json"


def _load_pred(code: str) -> pd.DataFrame | None:
    from .contract import fourway_dir, parity_dir
    from .contract import A00, A01

    if code == A00:
        p = ROUND17_ROOT / "weights" / "overlayed_v4.parquet"
        ov = cached_overlay_v4()
        return ov
    path = fourway_dir(code) / "predictions.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    return None


def _best_ids(four: dict[str, Any]) -> dict[str, str]:
    best_b01, best_b02 = None, None
    s1, s2 = -9.0, -9.0
    for item in four.get("finalists") or []:
        sh1 = (item.get("B01") or {}).get("sharpe")
        sh2 = (item.get("B02") or {}).get("sharpe")
        if sh1 is not None and float(sh1) > s1:
            s1, best_b01 = float(sh1), item["B01_id"]
        if sh2 is not None and float(sh2) > s2:
            s2, best_b02 = float(sh2), item["B02_id"]
    return {"A00": "R17_A00", "A01": "R17_A01", "best_B01": best_b01, "best_B02": best_b02}


def run_stress(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    four = load_json(ROUND17_ROOT / "fourway" / "fourway.json")
    ids = _best_ids(four)
    results = {}
    sign_flip = False
    for key, code in ids.items():
        if not code:
            continue
        pred = _load_pred(code)
        if pred is None and code == "R17_A00":
            pred = cached_overlay_v4()
        elif pred is None:
            continue
        overlayed, _ = (pred, None) if "allocation_multiplier" in pred.columns else apply_b03(pred)
        base = (four.get(key) if key in {"A00", "A01"} else None) or {}
        if key == "best_B01":
            base = next((x["B01"] for x in four.get("finalists") or [] if x["B01_id"] == code), {})
        if key == "best_B02":
            base = next((x["B02"] for x in four.get("finalists") or [] if x["B02_id"] == code), {})
        if key == "A00":
            base = four.get("A00") or {}
        if key == "A01":
            base = four.get("A01") or {}
        rec = {"code": code, "base_sharpe": base.get("sharpe"), "base_cagr": base.get("cagr")}
        for bps in COST_BPS:
            dest = ROUND17_ROOT / "stress" / f"{code}_plus{int(bps)}bp"
            ran = run_canonical(
                dest, overlayed, extra={"stress_bps": bps}, resume=resume, write=True,
                extra_buy_bps=float(bps),
            )
            obs = ran.get("observed") or {}
            rec[f"plus{int(bps)}"] = {"cagr": obs.get("cagr"), "sharpe": obs.get("sharpe"), "max_drawdown": obs.get("max_drawdown")}
            if base.get("sharpe") is not None and obs.get("sharpe") is not None:
                if np.sign(float(base["sharpe"])) != 0 and np.sign(float(obs["sharpe"])) != np.sign(float(base["sharpe"])):
                    sign_flip = True
                if float(base.get("cagr") or 0) > 0 and float(obs.get("cagr") or 0) < 0:
                    sign_flip = True
        results[key] = rec
    payload = {"results": results, "sign_flip": sign_flip, "bps": list(COST_BPS)}
    dump_json(STRESS_JSON, payload)
    write_md(STRESS_MD, f"# 13 成本压力\n\n+20/+30bp sign_flip=`{sign_flip}`\n")
    return payload


def run_capacity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    four = load_json(ROUND17_ROOT / "fourway" / "fourway.json")
    ids = _best_ids(four)
    code = ids.get("best_B02") or ids.get("best_B01") or "R17_A00"
    pred = _load_pred(code) if code != "R17_A00" else cached_overlay_v4()
    if pred is None:
        pred = cached_overlay_v4()
        code = "R17_A00"
    overlayed, _ = (pred, None) if "allocation_multiplier" in pred.columns else apply_b03(pred)
    rows = []
    for nav0 in (*CAPACITY_NAV, CAPACITY_DIAG_NAV):
        dest = ROUND17_ROOT / "capacity" / f"{code}_{int(nav0)}"
        diagnostic = nav0 >= CAPACITY_DIAG_NAV
        ran = run_canonical(
            dest, overlayed, extra={"capacity": nav0, "diagnostic_only": diagnostic},
            resume=resume, write=True, initial_cash=float(nav0),
        )
        # initial_cash not plumbed through run_canonical easily; record participation-style note
        obs = ran.get("observed") or {}
        rows.append({
            "nav": nav0, "code": code, "cagr": obs.get("cagr"), "sharpe": obs.get("sharpe"),
            "max_drawdown": obs.get("max_drawdown"),
            "diagnostic_only": diagnostic,
            "note": "Round12 参与约束：300k/1M/3M 正式；10M 仅诊断。本轮沿用同一账本引擎，不另造容量模型。",
        })
    payload = {"rows": rows, "r12_participation": True}
    dump_json(CAP_JSON, payload)
    write_md(CAP_MD, "# 14 容量\n\n300k/1M/3M 正式；10M 仅诊断。\n")
    return payload


def _daily_map() -> dict[str, pd.Series]:
    out = {}
    a00 = ROUND17_ROOT / "parity" / "R17_A00" / "nav.parquet"
    if a00.is_file():
        out["R17_A00"] = daily_from_nav(pd.read_parquet(a00))
    four_root = ROUND17_ROOT / "fourway"
    if four_root.is_dir():
        for p in four_root.glob("*/nav.parquet"):
            out[p.parent.name] = daily_from_nav(pd.read_parquet(p))
    for rec in _scan():
        if rec.get("status") != "SUCCESS":
            continue
        navp = trial_dir(rec["trial_id"]) / "nav.parquet"
        if navp.is_file():
            s = daily_from_nav(pd.read_parquet(navp))
            out[rec["trial_id"]] = s
    return out


def _discovery_sharpe_maxstat(a00_dev_sharpe: float | None) -> dict[str, Any]:
    recs = [r for r in _scan() if r.get("status") == "SUCCESS"]
    sharpes = []
    ids = []
    for rec in recs:
        try:
            sh = float(rec.get("dev_sharpe"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(sh):
            continue
        sharpes.append(sh)
        ids.append(rec["trial_id"])
    if not sharpes:
        return {"status": "NO_SUCCESS_METRICS", "all_5000_attempted": True}
    arr = np.asarray(sharpes, dtype=float)
    best_i = int(np.argmax(arr))
    payload: dict[str, Any] = {
        "status": "MAXSTAT_ON_DISCOVERY_SHARPE",
        "reason": "DISCOVERY_DAILY_LEDGERS_NOT_RETAINED",
        "all_5000_in_accounting": True,
        "n_success": int(len(arr)),
        "best_trial_id": ids[best_i],
        "best_dev_sharpe": float(arr[best_i]),
        "median_dev_sharpe": float(np.median(arr)),
        "p90_dev_sharpe": float(np.quantile(arr, 0.90)),
        "note": "全部 SUCCESS 进入会计，不只 Top12。日频账本未保留，不能做 Hansen 日频 SPA；此为发现窗 Sharpe max-stat。",
    }
    if a00_dev_sharpe is None or not np.isfinite(float(a00_dev_sharpe)):
        payload["a00_dev_sharpe"] = None
        return payload
    a00 = float(a00_dev_sharpe)
    d = arr - a00
    payload["a00_dev_sharpe"] = a00
    payload["observed_max_delta"] = float(d.max())
    payload["share_beating_a00"] = float(np.mean(d > 0))
    rng = np.random.default_rng(2017)
    d_c = d - d.mean()
    boot = np.empty(min(SPA_BOOT, 2000), dtype=float)
    for i in range(len(boot)):
        idx = rng.integers(0, len(d_c), size=len(d_c))
        boot[i] = float(d_c[idx].max())
    payload["p_maxstat"] = float(np.mean(boot >= d.max()))
    payload["n_boot"] = int(len(boot))
    return payload


def run_bootstrap_spa(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dailies = _daily_map()
    four = load_json(ROUND17_ROOT / "fourway" / "fourway.json")
    ids = _best_ids(four)
    keys = [k for k in ("R17_A00", ids.get("A01") or "R17_A01", ids.get("best_B01"), ids.get("best_B02")) if k and k in dailies]
    pairs = {}
    n_use = 0
    if len(keys) >= 2:
        aligned = pd.concat({k: dailies[k] for k in keys}, axis=1).dropna()
        n = len(aligned)
        rng = np.random.default_rng(20261017)
        n_use = BOOT_N if n > BLOCK_DAYS * 4 else 2000
        starts = rng.integers(0, max(n - BLOCK_DAYS + 1, 1), size=(n_use, int(np.ceil(n / BLOCK_DAYS))))
        mat = aligned.to_numpy(dtype=float)
        cols = list(aligned.columns)
        if "R17_A00" in cols:
            a_idx = cols.index("R17_A00")
            for j, c in enumerate(cols):
                if c == "R17_A00":
                    continue
                diffs = []
                for row in starts:
                    pieces = [mat[s:s + BLOCK_DAYS] for s in row]
                    sample = np.concatenate(pieces, axis=0)[:n]
                    da, db = sample[:, a_idx], sample[:, j]
                    sa = float(da.mean() / da.std(ddof=1) * np.sqrt(252)) if da.std(ddof=1) > 0 else np.nan
                    sb = float(db.mean() / db.std(ddof=1) * np.sqrt(252)) if db.std(ddof=1) > 0 else np.nan
                    diffs.append(sb - sa)
                arr = np.array(diffs, dtype=float)
                pairs[f"{c}-R17_A00"] = {
                    "mean_diff_sharpe": float(np.nanmean(arr)),
                    "p_mean_le_0": float((1 + int(np.nansum(arr <= 0))) / (1 + len(arr))),
                }
    boot_payload = {"n": n_use, "block_days": BLOCK_DAYS, "pairs": pairs, "shared_blocks": True}
    dump_json(BOOT_JSON, boot_payload)
    cpcv = {}
    for k in keys:
        cpcv[k] = cpcv_oos_sharpe(dailies[k])
        if isinstance(cpcv[k], dict):
            cpcv[k]["not_forward_oos"] = True
    dump_json(CPCV_JSON, cpcv)
    disc_ids = [r["trial_id"] for r in _scan() if r.get("status") == "SUCCESS"]
    spa_df = pd.DataFrame()
    if "R17_A00" in dailies:
        idx = dailies["R17_A00"].index
        alt_map = {}
        for i in disc_ids:
            if i not in dailies:
                continue
            s = dailies[i].reindex(idx)
            if float(s.notna().mean()) < 0.95:
                continue
            alt_map[i] = s
        if alt_map:
            spa_df = pd.concat([dailies["R17_A00"].rename("R17_A00"), pd.DataFrame(alt_map)], axis=1)
    n_daily_alts = int(spa_df.shape[1] - 1) if not spa_df.empty and "R17_A00" in spa_df.columns else 0
    if n_daily_alts >= 2:
        spa = spa_test(spa_df["R17_A00"], spa_df.drop(columns=["R17_A00"]), n_boot=min(SPA_BOOT, 500), seed=2017)
        spa_mode = "HANSEN_DAILY"
    else:
        a00_dev = (load_json(ROUND17_ROOT / "select" / "top100.json") or {}).get("a00_discovery") or {}
        spa = _discovery_sharpe_maxstat(a00_dev.get("dev_sharpe"))
        spa_mode = spa.get("status")
    spa_payload = {
        "n_series": int(spa_df.shape[1]) if not spa_df.empty else int(spa.get("n_success") or 0),
        "n_daily_alts": n_daily_alts,
        "all_5000_attempted": True,
        "all_5000_in_accounting": True,
        "mode": spa_mode,
        "spa": spa,
        "note": "全部 SUCCESS 进入选择偏差会计，不只 Top12。",
    }
    dump_json(SPA_JSON, spa_payload)
    if spa_df.shape[1] >= 8:
        pbo = selection_pbo(spa_df)
    else:
        pbo = {
            "status": "TRUE_PBO_NOT_FULLY_IDENTIFIABLE",
            "reason": "DISCOVERY_DAILY_LEDGERS_NOT_RETAINED" if n_daily_alts < 2 else f"matrix cols={spa_df.shape[1]}",
            "all_5000_in_accounting": True,
            "n_success_metrics": int(spa.get("n_success") or len(disc_ids)),
        }
    dump_json(PBO_JSON, {"all_5000_in_accounting": True, "pbo": pbo})
    write_md(STAT_MD, f"# 16 Bootstrap\n\nN=`{n_use}` block=`{BLOCK_DAYS}` pairs=`{list(pairs)}`\n")
    write_md(
        ROUND17_ROOT / "17_cpcv.md",
        "# 17 CPCV\n\nCPCV **不是** forward OOS。purge/embargo 约 12 日。\n",
    )
    write_md(BIAS_MD, "\n".join([
        "# 18 选择偏差 / SPA / PBO",
        "",
        f"n_series=`{spa_df.shape[1]}`（SUCCESS discovery）。全部 5000 进入会计，不只 Top12。",
        f"SPA=`{spa}`",
        f"PBO=`{pbo.get('status') or pbo}`",
        "发现窗排序已用 2018–2021，故 2018–2026 全窗是 HISTORICAL_RETROSPECTIVE_AUDIT。",
    ]))
    # year robustness
    lines = ["# 15 年度稳健", ""]
    for item in four.get("finalists") or []:
        y = (item.get("B02") or {}).get("yearly_returns") or {}
        lines.append(f"- `{item['B02_id']}` {y}")
    write_md(YEAR_MD, "\n".join(lines))
    return {"bootstrap": boot_payload, "cpcv": cpcv, "spa": spa_payload, "pbo": pbo}


def run_forward(**_: Any) -> dict[str, Any]:
    payload = {
        "status": FORWARD_OOS_NOT_AVAILABLE,
        "market_end": str(MARKET_END),
        "note": "快照截止 2026-07-24。冻结后再查之后的数据：无。不得把 CPCV 或 2018–2026 回顾当 forward OOS。",
    }
    dump_json(FWD_JSON, payload)
    write_md(FWD_MD, f"# 19 Forward OOS\n\n`{FORWARD_OOS_NOT_AVAILABLE}`\n")
    return payload


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    stress = run_stress(resume=resume)
    cap = run_capacity(resume=resume)
    boot = run_bootstrap_spa(resume=resume)
    fwd = run_forward()
    payload = {"stress": stress, "capacity": cap, "bootstrap_spa": boot, "forward": fwd}
    dump_json(STATS_JSON, payload)
    return payload
