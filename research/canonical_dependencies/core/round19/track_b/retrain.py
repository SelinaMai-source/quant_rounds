"""B00 locked replay vs B01 force retrain original 14/13. No F1/F2/F3."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...round17.backtest import apply_b03, cached_overlay_v4, fuse_heads, run_canonical
from ...round18.backtest import run_fused_backtest, summarize
from ..common import slim_obs
from ..contract import HISTORICAL_FEE_CORRECTED, V4_ANCHOR
from .support import (
    B00,
    B01,
    cand_dir,
    dump_json,
    file_sha256,
    load_json,
    load_r18_panels,
    load_r7_preds,
    overlay_path,
    tb_dir,
    write_md,
)
from .train import train_both_heads


def _write_overlay(code: str, overlayed: pd.DataFrame) -> None:
    dest = overlay_path(code)
    dest.parent.mkdir(parents=True, exist_ok=True)
    overlayed.to_parquet(dest, index=False)
    cand_dir(code)
    overlayed.to_parquet(cand_dir(code) / "overlayed.parquet", index=False)


def run_b00(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = cand_dir(B00)
    extra = {
        "code": B00,
        "role": "locked_round7_prediction_replay",
        "fee_schedule": HISTORICAL_FEE_CORRECTED,
        "retrain": False,
        "force_retrain": False,
        "extras": [],
    }
    if resume and (dest / "report.json").is_file() and (dest / "nav.parquet").is_file():
        report = load_json(dest / "report.json")
        obs = slim_obs(summarize({"status": "resumed", "report": report, "out_dir": str(dest)}))
        r7 = load_r7_preds()
        payload = {
            "status": "resumed",
            "code": B00,
            "observed": obs,
            "five_rows": int(len(r7["five"])),
            "ten_rows": int(len(r7["ten"])),
            "out_dir": str(dest),
        }
        dump_json(dest / "summary.json", payload)
        return payload
    ov4 = cached_overlay_v4()
    _write_overlay(B00, ov4)
    r7 = load_r7_preds()
    r7["five"].to_parquet(dest / "five_frozen.parquet", index=False)
    r7["ten"].to_parquet(dest / "ten_frozen.parquet", index=False)
    fused = fuse_heads(r7["five"], r7["ten"])
    fused.to_parquet(dest / "fused_from_r7.parquet", index=False)
    ov_from_r7, meta = apply_b03(fused)
    ov_from_r7.to_parquet(dest / "overlayed_from_r7_fuse.parquet", index=False)
    dump_json(dest / "overlay_from_r7_meta.json", meta)
    ran = run_canonical(dest, ov4, extra=extra, resume=resume, write=True)
    obs = slim_obs(ran.get("observed") or {})
    payload = {
        "status": "PASS" if (
            obs.get("completed_trades") == V4_ANCHOR["completed_trades"]
            and obs.get("cagr") is not None
        ) else "RAN",
        "code": B00,
        "role": "locked_round7_prediction_replay",
        "retrain": False,
        "force_retrain": False,
        "observed": obs,
        "anchor": dict(V4_ANCHOR),
        "five_rows": int(len(r7["five"])),
        "ten_rows": int(len(r7["ten"])),
        "overlay_sha256": file_sha256(overlay_path(B00)),
        "out_dir": str(dest),
        "note": "A00-class locked overlay replay. Not a fresh original-feature retrain.",
    }
    dump_json(dest / "summary.json", payload)
    dump_json(tb_dir("retrain") / "R19_B00.json", payload)
    return payload


def run_b01(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = cand_dir(B01)
    extra = {
        "code": B01,
        "role": "force_retrain_original_14_13",
        "fee_schedule": HISTORICAL_FEE_CORRECTED,
        "retrain": True,
        "force_retrain": True,
        "five_extras": [],
        "ten_extras": [],
    }
    if resume and (dest / "report.json").is_file() and (dest / "nav.parquet").is_file() and (
        dest / "five" / "predictions.parquet"
    ).is_file():
        report = load_json(dest / "report.json")
        obs = slim_obs(summarize({"status": "resumed", "report": report, "out_dir": str(dest)}))
        five = pd.read_parquet(dest / "five" / "predictions.parquet")
        ten = pd.read_parquet(dest / "ten" / "predictions.parquet")
        payload = {
            "status": "resumed",
            "code": B01,
            "observed": obs,
            "five_rows": int(len(five)),
            "ten_rows": int(len(ten)),
            "force_retrain": True,
            "extras": [],
            "out_dir": str(dest),
        }
        dump_json(dest / "summary.json", payload)
        return payload
    print("[round19/track_b] R19_B01 force_retrain original 14/13 via train_head_open", flush=True)
    panels = load_r18_panels()
    trained = train_both_heads(
        five_training=panels["five_training"],
        five_candidates=panels["five_candidate"],
        ten_training=panels["ten_training"],
        ten_candidates=panels["ten_candidate"],
        five_extras=(),
        ten_extras=(),
        output_dir=dest,
        resume=resume,
        five_frozen=panels["five_frozen"],
        ten_frozen=panels["ten_frozen"],
        force_retrain=True,
    )
    five_pred = trained["five_pred"]
    ten_pred = trained["ten_pred"]
    fused = fuse_heads(five_pred, ten_pred)
    fused.to_parquet(dest / "predictions.parquet", index=False)
    overlayed, meta = apply_b03(fused)
    dump_json(dest / "overlay_meta.json", meta)
    _write_overlay(B01, overlayed)
    obs = run_fused_backtest(dest, five_pred, ten_pred, extra=extra, resume=resume)
    slim = slim_obs(obs)
    five_rep = {k: trained["five"].get(k) for k in (
        "status", "features", "rows", "days", "mean_fold_ndcg", "force_retrain", "extras", "filled_with_zero"
    )}
    ten_rep = {k: trained["ten"].get(k) for k in (
        "status", "features", "rows", "days", "mean_fold_ndcg", "force_retrain", "extras", "filled_with_zero"
    )}
    payload = {
        "status": obs.get("status") or "ran",
        "code": B01,
        "role": "force_retrain_original_14_13",
        "retrain": True,
        "force_retrain": True,
        "extras": [],
        "observed": slim,
        "five": five_rep,
        "ten": ten_rep,
        "five_rows": int(len(five_pred)),
        "ten_rows": int(len(ten_pred)),
        "filled_with_zero": False,
        "out_dir": str(dest),
        "note": "Round18 open path, original 14/13 only, no F1/F2/F3.",
    }
    dump_json(dest / "summary.json", payload)
    dump_json(tb_dir("retrain") / "R19_B01.json", payload)
    return payload


def run_fresh_retrain(*, resume: bool = True, **kwargs: Any) -> dict[str, Any]:
    b00 = run_b00(resume=resume, **kwargs)
    b01 = run_b01(resume=resume, **kwargs)
    payload = {
        "status": "done",
        "B00": b00,
        "B01": b01,
        "force_retrain_used": True,
        "b01_extras": [],
        "round18_originals_untouched": True,
    }
    dump_json(tb_dir("retrain") / "fresh_retrain.json", payload)
    lines = [
        "# Track B · B00 locked vs B01 force retrain",
        "",
        "B00 = locked Round7 overlay replay（同类 A00）。B01 = `force_retrain` 原 14/13，无 F1/F2/F3。",
        "未改 `result/round18/` 原件。未搜 epsilon / 公式 / head 权重。",
        "",
        f"- B00 CAGR=`{(b00.get('observed') or {}).get('cagr')}` Sharpe=`{(b00.get('observed') or {}).get('sharpe')}` "
        f"MaxDD=`{(b00.get('observed') or {}).get('max_drawdown')}` trades=`{(b00.get('observed') or {}).get('completed_trades')}`",
        f"- B01 CAGR=`{(b01.get('observed') or {}).get('cagr')}` Sharpe=`{(b01.get('observed') or {}).get('sharpe')}` "
        f"MaxDD=`{(b01.get('observed') or {}).get('max_drawdown')}` trades=`{(b01.get('observed') or {}).get('completed_trades')}`",
        f"- B00 five/ten rows=`{b00.get('five_rows')}`/`{b00.get('ten_rows')}`",
        f"- B01 five/ten rows=`{b01.get('five_rows')}`/`{b01.get('ten_rows')}`",
        f"- B01 force_retrain=`True` extras=`[]`",
        "",
    ]
    write_md(tb_dir("retrain") / "02_fresh_retrain.md", "\n".join(lines))
    return payload


__all__ = ["run_b00", "run_b01", "run_fresh_retrain"]
