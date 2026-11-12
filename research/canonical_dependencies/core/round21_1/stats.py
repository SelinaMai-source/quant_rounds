"""分年 / Ex / block bootstrap 5000 / SPA。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..round8.robustness import block_bootstrap_delta
from ..round11.common import daily_from_nav
from ..round16.common import cagr_of
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import BLOCK_DAYS, BOOT_N, E0_CODE, FORWARD_OOS_NOT_AVAILABLE, SPA_BOOT, STATS_MD, exp_dir
from .backtest import BT_JSON

from ..paths import ROUND21_1_ROOT

STATS_JSON = ROUND21_1_ROOT / "stats" / "stats.json"


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


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and STATS_JSON.is_file() and STATS_MD.is_file():
        existing = load_json(STATS_JSON)
        if existing.get("status") == "DONE":
            return existing
    bt = load_json(BT_JSON)
    codes = [r["code"] for r in (bt.get("main_rows") or [])]
    e0_nav = _nav(E0_CODE)
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
            "y2024_2025": _window(nav, years=(2024, 2025)),
            "y2026_ytd": _window(nav, years=(2026,)),
            "ex2024": _window(nav, drop=(2024,)),
            "ex2025": _window(nav, drop=(2025,)),
            "ex2024_2025": _window(nav, drop=(2024, 2025)),
            "tag": "REPEATED_HISTORICAL_TESTING",
        }
        if e0_nav is not None and code != E0_CODE:
            try:
                rec["block_bootstrap_vs_e0"] = block_bootstrap_delta(
                    daily, daily_from_nav(e0_nav), block=BLOCK_DAYS, n=BOOT_N, seed=21
                )
            except Exception as exc:
                rec["block_bootstrap_vs_e0"] = {"status": "NOT_IDENTIFIABLE", "reason": str(exc)}
            rec["block_bootstrap_tag"] = "STATISTICAL RESAMPLING NOT OOS"
        rows.append(rec)

    spa_payload: dict[str, Any] = {"status": "NOT_IDENTIFIABLE"}
    if E0_CODE in dailies and len(dailies) >= 2:
        try:
            from ..round11.stats import spa_test

            alts = {k: v for k, v in dailies.items() if k != E0_CODE}
            if alts:
                mat = pd.concat(alts, axis=1)
                spa_payload = {"status": "DONE", **spa_test(dailies[E0_CODE], mat, n_boot=SPA_BOOT, seed=17), "note": "NOT independent OOS"}
        except Exception as exc:
            spa_payload = {"status": "NOT_IDENTIFIABLE", "reason": str(exc)}

    payload = {
        "status": "DONE",
        "n": len(rows),
        "block_days": BLOCK_DAYS,
        "boot_n": BOOT_N,
        "spa": spa_payload,
        "forward": FORWARD_OOS_NOT_AVAILABLE,
        "rows": rows,
    }
    dump_json(STATS_JSON, payload)
    trial_rows = []
    for r in rows:
        trial_rows.append(
            {
                "trial_id": r.get("code"),
                "round": "21.1",
                "tag": r.get("tag") or "REPEATED_HISTORICAL_TESTING",
                "forward_oos": FORWARD_OOS_NOT_AVAILABLE,
                "block_bootstrap": r.get("block_bootstrap_vs_e0"),
                "y2018_2020_sharpe": (r.get("y2018_2020") or {}).get("sharpe"),
                "y2021_2023_sharpe": (r.get("y2021_2023") or {}).get("sharpe"),
                "y2024_2025_sharpe": (r.get("y2024_2025") or {}).get("sharpe"),
                "y2026_ytd_sharpe": (r.get("y2026_ytd") or {}).get("sharpe"),
                "ex2024_sharpe": (r.get("ex2024") or {}).get("sharpe"),
                "ex2025_sharpe": (r.get("ex2025") or {}).get("sharpe"),
                "ex2024_2025_sharpe": (r.get("ex2024_2025") or {}).get("sharpe"),
            }
        )
    trial_path = ROUND21_1_ROOT / "stats" / "trial_registry.csv"
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
    payload["trial_registry"] = str(trial_path)
    dump_json(STATS_JSON, payload)
    lines = [
        "# Round21.1 · 12 稳健性 / SPA",
        "",
        f"窗：2018–2020 / 2021–2023 / 2024–2025 / 2026 YTD / Ex2024 / Ex2025 / Ex2024+2025。Bootstrap N=`{BOOT_N}` block=`{BLOCK_DAYS}`。",
        f"SPA=`{spa_payload.get('status')}`。Forward=`{FORWARD_OOS_NOT_AVAILABLE}`。2018–2026 标 `REPEATED_HISTORICAL_TESTING`。",
        "",
        "| code | 18-20 Sh | 21-23 Sh | 24-25 Sh | 26 YTD | Ex24 | Ex25 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.get('code')}` | {(r.get('y2018_2020') or {}).get('sharpe')} | "
            f"{(r.get('y2021_2023') or {}).get('sharpe')} | {(r.get('y2024_2025') or {}).get('sharpe')} | "
            f"{(r.get('y2026_ytd') or {}).get('sharpe')} | {(r.get('ex2024') or {}).get('sharpe')} | "
            f"{(r.get('ex2025') or {}).get('sharpe')} |"
        )
    write_md(STATS_MD, "\n".join(lines))
    return payload
