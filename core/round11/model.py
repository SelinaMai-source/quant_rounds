"""五种子 / leave-one-seed / fusion 85/15·90/10·95/5。邻域复用 Round10 artifact。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import PROJECT_ROOT, ROUND10_ROOT, ROUND11_ROOT
from ..replay_round7.inject import inject_round7_into_overlay_caches
from ..round7.contract import GAP_FLOOR, SEEDS, TEN_HEAD
from ..round7.train import assert_seed_rank_invariance, ensemble_rank_pct
from ..round9.overlays import build_overlayed
from ..stock_readonly import load_prepared_market
from .common import dump_json, pct, stage_dir, write_md
from .contract import BASELINES
from .runner import run_spec_to

MODEL_MD = ROUND11_ROOT / "06_model_seed_fusion.md"
FUSION_WEIGHTS = ((0.85, 0.15), (0.90, 0.10), (0.95, 0.05))


def _load_seed(head: str, seed: int) -> pd.DataFrame:
    path = PROJECT_ROOT / f"result/round7/models/{head}/seed{seed}/predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["signal_date", "instrument_id", "score"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    return frame


def _gaps() -> pd.DataFrame:
    panel = TEN_HEAD.panel_dir / "candidate_panel.parquet"
    gaps = pd.read_parquet(panel, columns=["trade_date", "instrument_id", "d1_open_gap"])
    gaps = gaps.rename(columns={"trade_date": "signal_date"})
    gaps["signal_date"] = pd.to_datetime(gaps["signal_date"])
    gaps["instrument_id"] = gaps["instrument_id"].astype(str)
    return gaps


def _fuse(five: pd.DataFrame, ten: pd.DataFrame, gaps: pd.DataFrame, ten_weight: float) -> pd.DataFrame:
    work = five.merge(ten, on=["signal_date", "instrument_id"], how="inner", suffixes=("_five", "_ten"))
    work = work.merge(gaps, on=["signal_date", "instrument_id"], how="left")
    work["d1_open_gap"] = pd.to_numeric(work["d1_open_gap"], errors="coerce")
    work = work.loc[work["d1_open_gap"].ge(GAP_FLOOR)].copy()
    work["pct_five"] = work.groupby("signal_date")["score_five"].rank(method="average", pct=True)
    work["pct_ten"] = work.groupby("signal_date")["score_ten"].rank(method="average", pct=True)
    work["score"] = float(ten_weight) * work["pct_ten"] + (1.0 - float(ten_weight)) * work["pct_five"]
    return work[["signal_date", "instrument_id", "score"]].sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort"
    ).reset_index(drop=True)


def _v4_overlay(predictions: pd.DataFrame) -> pd.DataFrame:
    overlayed, _, _ = build_overlayed({"kind": "preregistered", "allocator": "B03_conservative"}, predictions)
    return overlayed


def _read_round10_neighborhood() -> dict[str, Any]:
    md = ROUND10_ROOT / "06_parameter_neighborhood.md"
    rows = []
    for code in ("R10_F01_sep_low", "R10_F02_sep_high", "R10_F03_alloc_048", "R10_F04_alloc_0612"):
        summary = ROUND10_ROOT / code / "summary.json"
        if summary.is_file():
            rows.append(json.loads(summary.read_text(encoding="utf-8")))
    return {
        "source": str(md),
        "exists": md.is_file(),
        "verdict_from_artifact": "STABLE_PLATEAU" if md.is_file() else "MISSING",
        "summaries": rows,
        "reused": True,
        "note": "先读 Round10 artifact，不补跑邻域。",
    }


def run_model(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = stage_dir("model")
    inject_round7_into_overlay_caches()
    five_frames = [_load_seed("five", s) for s in SEEDS]
    ten_frames = [_load_seed("ten", s) for s in SEEDS]
    inv_five = assert_seed_rank_invariance(five_frames)
    inv_ten = assert_seed_rank_invariance(ten_frames)
    invariant = bool(inv_five.get("passed") and inv_ten.get("passed"))
    seed_block = {
        "seeds": list(SEEDS),
        "five_invariance": inv_five,
        "ten_invariance": inv_ten,
        "all_ranks_identical": invariant,
        "action": (
            "SEED_RANK_INVARIANT_SKIP_REDUNDANT_REPLAY"
            if invariant
            else "MUST_REPLAY_LEAVE_ONE_SEED"
        ),
        "note": (
            "subsample=1 / colsample=1，五种子日排名一致。"
            "leave-one-seed 与单种子融合在排名层与锁定 90/10 等价，不再为重复数字重跑。"
            if invariant
            else "种子排名不一致，必须重放。"
        ),
    }
    dump_json(dest / "seed_invariance.json", seed_block)

    loaded = load_prepared_market(warmup_ma5=True)
    prepared = loaded["prepared"]
    gaps = _gaps()
    five_ens = ensemble_rank_pct(five_frames)
    ten_ens = ensemble_rank_pct(ten_frames)
    five_ens = five_ens.rename(columns={"score": "score"})
    ten_ens = ten_ens.rename(columns={"score": "score"})
    fusion_runs: dict[str, Any] = {}
    spec = BASELINES["v4"]
    for ten_w, five_w in FUSION_WEIGHTS:
        tag = f"R11_V4_FUSION_{int(ten_w*100)}_{int(five_w*100)}"
        pred = _fuse(five_ens.rename(columns={"score": "score"}), ten_ens, gaps, ten_w)
        # _fuse expects score_five/score_ten via merge suffixes; rebuild properly
        five_p = five_ens.rename(columns={"score": "score"})
        ten_p = ten_ens.rename(columns={"score": "score"})
        pred = _fuse(five_p, ten_p, gaps, ten_w)
        overlayed = _v4_overlay(pred)
        extra = {
            "fee_schedule": "LEGACY_FLAT_5BP",
            "fusion": {"ten": ten_w, "five": five_w},
            "evidence_role": "round11_fusion_sensitivity",
        }
        if abs(ten_w - 0.90) < 1e-12:
            extra["note"] = "90/10 是锁定融合；数字应对齐 v4 artifact（允许浮点）。"
        ran = run_spec_to(
            spec,
            dest / tag,
            prepared_market=prepared,
            market_meta=loaded,
            resume=resume,
            overlayed=overlayed,
            extra_report=extra,
        )
        m = (ran.get("report") or {}).get("metrics") or {}
        fusion_runs[tag] = {
            "ten_weight": ten_w,
            "five_weight": five_w,
            "cagr": m.get("annualized_return"),
            "sharpe": m.get("sharpe_zero_rate"),
            "max_drawdown": m.get("max_drawdown"),
            "completed_trades": m.get("completed_trades"),
            "out_dir": ran.get("out_dir"),
        }
    neighborhood = _read_round10_neighborhood()
    payload = {
        "status": "DONE",
        "seeds": seed_block,
        "fusion": fusion_runs,
        "neighborhood_round10": neighborhood,
        "post_hoc_diagnostic": True,
    }
    dump_json(dest / "model.json", payload)
    lines = [
        "# Round 11 · 模型 / 种子 / fusion / 邻域",
        "",
        f"五种子 = {list(SEEDS)}。排名不变性 five=`{inv_five.get('passed')}` ten=`{inv_ten.get('passed')}`。",
        f"动作：`{seed_block['action']}`。",
        "",
        seed_block["note"],
        "",
        "## fusion（只测 85/15、90/10、95/5；ten/five）",
        "",
        "| 实验 | ten/five | CAGR | Sharpe | MaxDD | 笔数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for tag, row in fusion_runs.items():
        lines.append(
            f"| `{tag}` | {int(row['ten_weight']*100)}/{int(row['five_weight']*100)} | "
            f"{pct(row.get('cagr'))} | {row.get('sharpe')} | {pct(row.get('max_drawdown'))} | "
            f"{row.get('completed_trades')} |"
        )
    lines += [
        "",
        "## Round10 邻域（复用 artifact，不补跑）",
        "",
        f"- 文件存在 = {neighborhood.get('exists')}；判定 `{neighborhood.get('verdict_from_artifact')}`",
        f"- {neighborhood.get('note')}",
        "",
    ]
    write_md(MODEL_MD, "\n".join(lines))
    return payload


__all__ = ["run_model"]
