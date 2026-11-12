"""分年 / HAC / block bootstrap 5000 / SPA 数值 p。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_3_ROOT
from ..round8.robustness import block_bootstrap_delta
from ..round11.common import daily_from_nav
from .backtest import BT_JSON, L1_BT_JSON
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import BLOCK_DAYS, BOOT_N, E0_CODE, FORWARD_OOS_NOT_AVAILABLE, L1_E0, L1_E3, MD, REPEATED_HISTORICAL_TESTING, SPA_BOOT, exp_dir

STATS_JSON = ROUND21_3_ROOT / "stats" / "stats.json"


def _nav(code: str) -> pd.DataFrame | None:
    path = exp_dir(code) / "nav.parquet"
    if not path.is_file():
        return None
    nav = pd.read_parquet(path)
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    return nav


def _window(nav: pd.DataFrame, *, years: tuple[int, ...] | None = None, drop: tuple[int, ...] = ()) -> dict[str, Any]:
    daily = daily_from_nav(nav)
    idx_year = pd.to_datetime(daily.index).year
    if years:
        daily = daily.loc[idx_year.isin(years)].copy()
    if drop:
        daily = daily.loc[~pd.to_datetime(daily.index).year.isin(drop)].copy()
    if len(daily) < 10:
        return {"status": "TOO_SHORT", "n": int(len(daily))}
    wealth = (1.0 + daily.astype(float)).cumprod()
    span = max((pd.Timestamp(daily.index.max()) - pd.Timestamp(daily.index.min())).days, 1)
    cagr = float(wealth.iloc[-1] ** (365.25 / span) - 1.0)
    vol = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / vol * np.sqrt(252.0)) if vol > 0 else None
    peak = wealth.cummax()
    dd = float((wealth / peak - 1.0).min()) if len(wealth) else None
    return {"cagr": finite_or_none(cagr), "sharpe": finite_or_none(sharpe), "max_drawdown": finite_or_none(dd), "n": int(len(daily))}


def _hac_paired(a: pd.Series, b: pd.Series, *, lags: int = 20) -> dict[str, Any]:
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = (aligned["a"] - aligned["b"]).astype(float)
    n = int(len(d))
    if n < lags * 2:
        return {"status": "TOO_SHORT", "n": n}
    mu = float(d.mean())
    x = d.to_numpy()
    x = x - mu
    gamma0 = float(np.dot(x, x) / n)
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        gk = float(np.dot(x[k:], x[:-k]) / n)
        var += 2.0 * w * gk
    se = float(np.sqrt(max(var, 0.0) / n))
    t = mu / se if se > 0 else None
    return {"mean_delta": mu, "hac_se": se, "t": finite_or_none(t), "n": n, "lags": lags, "status": "DONE"}


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and STATS_JSON.is_file() and MD["17"].is_file() and MD["19"].is_file():
        existing = load_json(STATS_JSON)
        have = {r.get("code") for r in (existing.get("rows") or [])}
        bt_codes = {r.get("code") for r in (load_json(BT_JSON).get("main_rows") or [])}
        l1_codes = {r.get("code") for r in (load_json(L1_BT_JSON).get("main_rows") or [])}
        newer = (bt_codes | l1_codes) - have
        if existing.get("status") == "DONE" and (existing.get("spa") or {}).get("p_spa") is not None and not newer:
            return existing
    bt = load_json(BT_JSON)
    l1 = load_json(L1_BT_JSON)
    codes = [r["code"] for r in (bt.get("main_rows") or [])]
    for r in l1.get("main_rows") or []:
        if r.get("code") not in codes:
            codes.append(r["code"])
    if not codes:
        codes = [L1_E0, L1_E3]
    e0_code = E0_CODE if E0_CODE in codes or _nav(E0_CODE) is not None else L1_E0
    e0_nav = _nav(e0_code)
    dailies: dict[str, pd.Series] = {}
    rows = []
    for code in codes:
        nav = _nav(code)
        if nav is None:
            rows.append({"code": code, "status": "NO_NAV"})
            continue
        daily = daily_from_nav(nav)
        dailies[code] = daily
        rec = {
            "code": code,
            "y2018_2020": _window(nav, years=(2018, 2019, 2020)),
            "y2021_2023": _window(nav, years=(2021, 2022, 2023)),
            "y2022": _window(nav, years=(2022,)),
            "y2024_2025": _window(nav, years=(2024, 2025)),
            "y2026_ytd": _window(nav, years=(2026,)),
            "ex2024": _window(nav, drop=(2024,)),
            "ex2025": _window(nav, drop=(2025,)),
            "ex2024_2025": _window(nav, drop=(2024, 2025)),
            "tag": REPEATED_HISTORICAL_TESTING,
        }
        if e0_nav is not None and code != e0_code:
            e0d = daily_from_nav(e0_nav)
            rec["hac_vs_e0"] = _hac_paired(daily, e0d, lags=BLOCK_DAYS)
            try:
                rec["block_bootstrap_vs_e0"] = block_bootstrap_delta(daily, e0d, block=BLOCK_DAYS, n=BOOT_N, seed=21)
            except Exception as exc:
                rec["block_bootstrap_vs_e0"] = {"status": "NOT_IDENTIFIABLE", "reason": str(exc)}
            rec["block_bootstrap_tag"] = "STATISTICAL RESAMPLING NOT OOS"
        rows.append(rec)

    spa_payload: dict[str, Any] = {"status": "NOT_IDENTIFIABLE"}
    if e0_code in dailies and len(dailies) >= 2:
        try:
            from ..round11.stats import spa_test

            alts = {k: v for k, v in dailies.items() if k != e0_code}
            if alts:
                mat = pd.concat(alts, axis=1)
                spa = spa_test(dailies[e0_code], mat, n_boot=SPA_BOOT, seed=17)
                spa_payload = {
                    "status": spa.get("status") or "DONE",
                    "observed_statistic": spa.get("observed_max_mean_delta"),
                    "observed_max_mean_delta": spa.get("observed_max_mean_delta"),
                    "n_alternatives": spa.get("n_alts"),
                    "n_alts": spa.get("n_alts"),
                    "p_spa": spa.get("p_spa"),
                    "n_boot": SPA_BOOT,
                    "block": BLOCK_DAYS,
                    "note": "NOT independent OOS; REPEATED_HISTORICAL_TESTING",
                }
        except Exception as exc:
            spa_payload = {"status": "NOT_IDENTIFIABLE", "reason": str(exc)}

    payload = {
        "status": "DONE" if spa_payload.get("p_spa") is not None or spa_payload.get("status") == "TOO_SHORT" else "PARTIAL",
        "n": len(rows),
        "block_days": BLOCK_DAYS,
        "boot_n": BOOT_N,
        "spa": spa_payload,
        "forward": FORWARD_OOS_NOT_AVAILABLE,
        "tag": REPEATED_HISTORICAL_TESTING,
        "rows": rows,
    }
    dump_json(STATS_JSON, payload)
    trial_rows = []
    for r in rows:
        trial_rows.append(
            {
                "trial_id": r.get("code"),
                "round": "21.3",
                "tag": r.get("tag") or REPEATED_HISTORICAL_TESTING,
                "forward_oos": FORWARD_OOS_NOT_AVAILABLE,
                "block_bootstrap": r.get("block_bootstrap_vs_e0"),
                "hac": r.get("hac_vs_e0"),
                "y2018_2020_sharpe": (r.get("y2018_2020") or {}).get("sharpe"),
                "y2021_2023_sharpe": (r.get("y2021_2023") or {}).get("sharpe"),
                "y2022_sharpe": (r.get("y2022") or {}).get("sharpe"),
                "y2024_2025_sharpe": (r.get("y2024_2025") or {}).get("sharpe"),
                "y2026_ytd_sharpe": (r.get("y2026_ytd") or {}).get("sharpe"),
                "ex2024_sharpe": (r.get("ex2024") or {}).get("sharpe"),
                "ex2025_sharpe": (r.get("ex2025") or {}).get("sharpe"),
                "ex2024_2025_sharpe": (r.get("ex2024_2025") or {}).get("sharpe"),
            }
        )
    trial_path = ROUND21_3_ROOT / "stats" / "trial_registry.csv"
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
    payload["trial_registry"] = str(trial_path)
    dump_json(STATS_JSON, payload)
    lines = [
        "# Round21.3 · 17 分年稳健性",
        "",
        f"窗：2018–2020 / 2021–2023 / 2022 / 2024–2025 / 2026 YTD / Ex2024 / Ex2025 / Ex2024+2025。Bootstrap N=`{BOOT_N}` block=`{BLOCK_DAYS}`。",
        f"标签=`{REPEATED_HISTORICAL_TESTING}`。Forward=`{FORWARD_OOS_NOT_AVAILABLE}`。",
        "",
        "| code | 18-20 Sh | 21-23 Sh | 2022 Sh | 24-25 Sh | 26 YTD | Ex24 | Ex25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.get('code')}` | {(r.get('y2018_2020') or {}).get('sharpe')} | "
            f"{(r.get('y2021_2023') or {}).get('sharpe')} | {(r.get('y2022') or {}).get('sharpe')} | "
            f"{(r.get('y2024_2025') or {}).get('sharpe')} | {(r.get('y2026_ytd') or {}).get('sharpe')} | "
            f"{(r.get('ex2024') or {}).get('sharpe')} | {(r.get('ex2025') or {}).get('sharpe')} |"
        )
    write_md(MD["17"], "\n".join(lines))
    write_md(
        MD["19"],
        "\n".join(
            [
                "# Round21.3 · 19 统计检验",
                "",
                f"HAC lags=`{BLOCK_DAYS}`。block bootstrap N=`{BOOT_N}` block=`{BLOCK_DAYS}`。",
                f"SPA observed=`{spa_payload.get('observed_statistic')}` n_alts=`{spa_payload.get('n_alternatives')}` p=`{spa_payload.get('p_spa')}` n_boot=`{SPA_BOOT}` status=`{spa_payload.get('status')}`。",
                f"标签=`{REPEATED_HISTORICAL_TESTING}`。重抽样不是独立 OOS。",
                "",
            ]
        ),
    )
    return payload
