"""GitHub-exact：在子集上跑 clone 的 build/run 脚本。窗口名 GITHUB_EXACT_*，与 ROUND21_INTERNAL 分名。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_ROOT, assert_project_result_path
from .common import dump_json, write_md
from .contract import (
    BLOCKED_PIT,
    EXACT_GITHUB_REPRODUCTION_PASS,
    GITHUB_CLONE_DEST,
    GITHUB_EXACT,
    GITHUB_EXACT_SUBSET,
    GITHUB_SPARSE_DIR,
    LABELS,
)
from .hypotheses import HYPOTHESES

GH_ROOT = ROUND21_ROOT / "github" / "exact_subset"
FACTORS_DIR = ROUND21_ROOT / "factors"
ADAPTER_MD = ROUND21_ROOT / "github" / "GITHUB_EXACT_SUBSET.md"


def _iso_close(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts).tz_localize("Asia/Shanghai") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).tz_convert("Asia/Shanghai")
    t = t.normalize() + pd.Timedelta(hours=15)
    return t.isoformat()


def _label_meta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label_id": "label_e2_net",
                "decision_schedule": "every_trading_day",
                "decision_interval_observations": 1,
                "entry_rule": "next_open",
                "exit_rule": "next2_close",
                "holding_horizon_observations": 2,
                "execution_delay_observations": 1,
                "cost_model": "ROUND7_FLAT_5BP",
                "label_is_net_of_costs": True,
                "overlapping_positions": True,
            },
            {
                "label_id": "label_exit_rule_net",
                "decision_schedule": "every_trading_day",
                "decision_interval_observations": 1,
                "entry_rule": "next_open",
                "exit_rule": "stock_exit_rule_actual",
                "holding_horizon_observations": 10,
                "execution_delay_observations": 1,
                "cost_model": "ROUND7_FLAT_5BP",
                "label_is_net_of_costs": True,
                "overlapping_positions": True,
            },
        ]
    )


def _recipe(label_id: str, holding: int, window_id: str, sens: list[int]) -> dict[str, Any]:
    return {
        "id": f"GITHUB_EXACT_{label_id}",
        "consensus_id": f"GITHUB_EXACT_{label_id}",
        "label_id": label_id,
        "window_id": window_id,
        "primary_metric": "rank_ic",
        "strategy_contract": {
            "decision_schedule": "every_trading_day",
            "decision_interval_observations": 1,
            "entry_rule": "next_open",
            "exit_rule": "next2_close" if label_id == "label_e2_net" else "stock_exit_rule_actual",
            "holding_horizon_observations": holding,
            "execution_delay_observations": 1,
            "cost_model": "ROUND7_FLAT_5BP",
            "require_net_of_costs": True,
            "overlapping_positions": holding > 1,
        },
        "preferred_group_count": 3,
        "fallback_group_count": 2,
        "minimum_pool_size": 5,
        "minimum_group_size": 2,
        "require_all_groups": False,
        "factor_non_null_ratio": 0.40,
        "valid_decision_coverage": 0.40,
        "minimum_valid_decisions": 20,
        "valid_decision_cap": 120,
        "fdr_alpha": 0.05,
        "confidence_level": 0.95,
        "hac_primary_lag": "auto",
        "hac_sensitivity_lags": sens,
        "minimum_effect": 0.001,
        "positive_period_share": 0.6,
        "monotonicity_threshold": 0.6,
        "minimum_consensus_windows": 1,
        "require_consistent_direction": True,
    }


def _scripts() -> dict[str, Path]:
    root = GITHUB_CLONE_DEST / GITHUB_SPARSE_DIR / "scripts"
    return {
        "build": root / "build_factor_plan.py",
        "run": root / "run_factor_evidence.py",
        "root": root,
    }


def _slice_csv(src: Path, dest: Path, factor_id: str) -> None:
    if not src.is_file():
        return
    frame = pd.read_csv(src)
    if "factor_id" not in frame.columns:
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return
    sub = frame.loc[frame["factor_id"].astype(str).eq(factor_id)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(dest, index=False)


def run_github_exact(panel: pd.DataFrame, *, resume: bool = True) -> dict[str, Any]:
    dest = GH_ROOT
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest)
    status_path = dest / "STATUS.json"
    if resume and status_path.is_file() and (dest / "evidence" / "SUCCESS.json").is_file():
        return json.loads(status_path.read_text(encoding="utf-8"))

    scripts = _scripts()
    cards = {c["factor_id"]: c for c in HYPOTHESES}
    by_factor: dict[str, Any] = {}
    if not scripts["build"].is_file() or not scripts["run"].is_file():
        payload = {
            "status": "NOT_EXACT",
            "exact": False,
            "namespace": GITHUB_EXACT,
            "reason": f"GitHub scripts missing under {GITHUB_CLONE_DEST}",
            "scope": None,
            "by_factor": {
                fid: {
                    "status": "NOT_EXACT",
                    "namespace": GITHUB_EXACT,
                    "reason": "scripts missing",
                }
                for fid in cards
            },
        }
        dump_json(status_path, payload)
        return payload

    start = pd.Timestamp(GITHUB_EXACT_SUBSET["start"])
    end = pd.Timestamp(GITHUB_EXACT_SUBSET["end"])
    window_id = GITHUB_EXACT_SUBSET["window_id"]
    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work = work.loc[work["trade_date"].between(start, end)].copy()
    if work.empty:
        payload = {
            "status": "NOT_EXACT",
            "exact": False,
            "namespace": GITHUB_EXACT,
            "reason": "subset panel empty",
            "scope": window_id,
            "by_factor": {},
        }
        dump_json(status_path, payload)
        return payload

    obs_rows = []
    factor_rows = []
    include = []
    for card in HYPOTHESES:
        fid = card["factor_id"]
        if card.get("blocked") == BLOCKED_PIT or card["family"] == "E":
            by_factor[fid] = {
                "status": "SKIPPED_BLOCKED_PIT",
                "namespace": GITHUB_EXACT,
                "reason": "no announcement timestamp; not submitted to GitHub-exact",
            }
            continue
        if fid not in work.columns:
            by_factor[fid] = {
                "status": "SKIPPED_BLOCKED_DATA",
                "namespace": GITHUB_EXACT,
                "reason": "column missing",
            }
            continue
        if int(pd.to_numeric(work[fid], errors="coerce").notna().sum()) == 0:
            by_factor[fid] = {
                "status": "SKIPPED_BLOCKED_DATA",
                "namespace": GITHUB_EXACT,
                "reason": "no finite values in subset; not filled 0",
            }
            continue
        include.append(fid)

    known_cache: dict[str, pd.DataFrame] = {}
    for fid in include:
        vp = FACTORS_DIR / fid / "values.parquet"
        if vp.is_file():
            kv = pd.read_parquet(vp, columns=["trade_date", "instrument_id", "known_at"])
            kv["trade_date"] = pd.to_datetime(kv["trade_date"])
            kv["instrument_id"] = kv["instrument_id"].astype(str)
            known_cache[fid] = kv

    exit_col = {"label_e2_net": "next2_date", "label_exit_rule_net": "actual_exit_date"}
    for _, row in work.iterrows():
        day = pd.Timestamp(row["trade_date"])
        inst = str(row["instrument_id"])
        oid = f"fb|{day.date()}|{inst}"
        decision = _iso_close(day)
        added = False
        for lab in LABELS:
            y = row.get(lab)
            exit_d = row.get(exit_col[lab])
            if pd.isna(exit_d):
                continue
            exit_ts = pd.Timestamp(exit_d)
            if exit_ts < day:
                continue
            try:
                yv = float(y)
                if not np.isfinite(yv):
                    yv = np.nan
            except (TypeError, ValueError):
                yv = np.nan
            obs_rows.append(
                {
                    "observation_id": oid,
                    "pool_id": "first_board_five_head",
                    "decision_time": decision,
                    "instrument_id": inst,
                    "label_id": lab,
                    "label_value": yv,
                    "label_exit_time": _iso_close(exit_ts),
                }
            )
            added = True
        if not added:
            continue
        for fid in include:
            val = row.get(fid)
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fv):
                continue
            known = decision
            kv = known_cache.get(fid)
            if kv is not None:
                hit = kv.loc[kv["trade_date"].eq(day) & kv["instrument_id"].eq(inst), "known_at"]
                if len(hit) and pd.notna(hit.iloc[0]):
                    known = _iso_close(pd.Timestamp(hit.iloc[0]))
            if pd.Timestamp(known) > pd.Timestamp(decision):
                continue
            factor_rows.append(
                {
                    "observation_id": oid,
                    "factor_id": fid,
                    "factor_value": fv,
                    "factor_known_at": known,
                }
            )

    obs = pd.DataFrame(obs_rows)
    fac = pd.DataFrame(factor_rows)
    meta = pd.DataFrame([{"factor_id": fid, "factor_type": "continuous"} for fid in include])
    labels = _label_meta()
    cfg = {
        "study_id": GITHUB_EXACT_SUBSET["study_id"],
        "timezone": "Asia/Shanghai",
        "thresholds_confirmed": True,
        "deployments": [
            {
                "id": GITHUB_EXACT_SUBSET["deployment_id"],
                "information_cutoff": f"{end.date()}T15:00:00+08:00",
                "windows": {window_id: f"{start.date()}T00:00:00+08:00"},
            }
        ],
        "recipes": [
            _recipe("label_e2_net", 2, window_id, [8]),
            _recipe("label_exit_rule_net", 10, window_id, [20]),
        ],
        "notes": {
            "namespace": GITHUB_EXACT,
            "not_internal_60_120_252": True,
            "first_board_pool_thresholds_adapted": True,
            "no_github_score_invented": True,
        },
    }
    obs_p = dest / "observations.csv"
    fac_p = dest / "factor_values.csv"
    meta_p = dest / "factor_metadata.csv"
    lab_p = dest / "label_metadata.csv"
    cfg_p = dest / "study_config.json"
    obs.to_csv(obs_p, index=False)
    fac.to_csv(fac_p, index=False)
    meta.to_csv(meta_p, index=False)
    labels.to_csv(lab_p, index=False)
    cfg_p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plan_dir = dest / "plan"
    ev_dir = dest / "evidence"
    if plan_dir.exists():
        import shutil

        shutil.rmtree(plan_dir)
    if ev_dir.exists():
        shutil.rmtree(ev_dir)

    env_python = sys.executable
    build = subprocess.run(
        [
            env_python,
            str(scripts["build"]),
            "--config",
            str(cfg_p),
            "--observations",
            str(obs_p),
            "--factors",
            str(fac_p),
            "--factor-metadata",
            str(meta_p),
            "--labels",
            str(lab_p),
            "--output-dir",
            str(plan_dir),
        ],
        cwd=str(scripts["root"]),
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        payload = {
            "status": "NOT_EXACT",
            "exact": False,
            "namespace": GITHUB_EXACT,
            "reason": f"build_factor_plan failed: {build.stderr[-2000:]}",
            "stdout": build.stdout[-1000:],
            "scope": window_id,
            "by_factor": {
                **by_factor,
                **{
                    fid: {"status": "NOT_EXACT", "namespace": GITHUB_EXACT, "reason": "plan build failed"}
                    for fid in include
                },
            },
        }
        dump_json(status_path, payload)
        write_md(ADAPTER_MD, f"# GITHUB_EXACT subset\n\nNOT_EXACT\n\n```\n{build.stderr[-2000:]}\n```\n")
        return payload

    run = subprocess.run(
        [
            env_python,
            str(scripts["run"]),
            "--plan-dir",
            str(plan_dir),
            "--observations",
            str(obs_p),
            "--factors",
            str(fac_p),
            "--factor-metadata",
            str(meta_p),
            "--labels",
            str(lab_p),
            "--output-dir",
            str(ev_dir),
        ],
        cwd=str(scripts["root"]),
        capture_output=True,
        text=True,
        check=False,
    )
    exact = run.returncode == 0 and (ev_dir / "SUCCESS.json").is_file()
    status = EXACT_GITHUB_REPRODUCTION_PASS if exact else "NOT_EXACT"
    for fid in include:
        fdir = FACTORS_DIR / fid / "github_exact"
        fdir.mkdir(parents=True, exist_ok=True)
        if exact:
            for name in (
                "factor_health.csv",
                "daily_factor_metrics.csv",
                "factor_summary.csv",
                "period_stability.csv",
                "period_factor_lineage.csv",
                "failure_records.csv",
            ):
                _slice_csv(ev_dir / name, fdir / name, fid)
        by_factor[fid] = {
            "status": status if exact else "NOT_EXACT",
            "namespace": GITHUB_EXACT,
            "window_id": window_id,
            "scope": "SUBSET_2018H1",
            "not_round21_internal": True,
            "dir": str(fdir),
            "reason": None if exact else (run.stderr[-1500:] or "run_factor_evidence failed"),
        }
    payload = {
        "status": status,
        "exact": exact,
        "namespace": GITHUB_EXACT,
        "scope": window_id,
        "n_obs": int(len(obs)),
        "n_factor_rows": int(len(fac)),
        "included": include,
        "reason": None if exact else (run.stderr[-2000:] or run.stdout[-2000:]),
        "stdout": (run.stdout or "")[-1500:],
        "scripts": {k: str(v) for k, v in scripts.items()},
        "commit_clone": str(GITHUB_CLONE_DEST),
        "thresholds_note": "first-board pool adapted (min_pool=5); GitHub windows are not 60/120/252",
        "by_factor": by_factor,
    }
    dump_json(status_path, payload)
    write_md(
        ADAPTER_MD,
        "\n".join(
            [
                "# GITHUB_EXACT subset",
                "",
                f"status=`{status}` scope=`{window_id}` exact=`{exact}`",
                "命名空间 `GITHUB_EXACT` ≠ `ROUND21_INTERNAL_W60/120/252`。源码无 Score，不伪造。",
                f"included=`{include}`",
                f"reason=`{payload['reason']}`",
                "",
            ]
        ),
    )
    return payload
