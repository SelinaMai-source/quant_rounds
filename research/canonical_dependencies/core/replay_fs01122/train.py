"""R5 D/E/F/G 与 R6 B–K：父 Alpha = fs01122 14 因子。产物只写 replay_fs01122。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..local_executor import run_local_fixed_sleeve_backtest, write_backtest_artifacts
from ..paths import (
    FS01122_FEATURES,
    MARKET_END,
    PREDICTION_START,
    SIGNAL_END,
    assert_project_result_path,
    ensure_replay_fs01122_tree,
    replay_fs01122_dir,
)
from ..round2.data import SEEDS
from ..round4.runner import attach_round4_fields
from ..round5.factors import FAMILY_COLUMNS
from ..round5.overlays import build_date_multiplier as r5_build
from ..round5.registry import R5Spec, V3_OVERLAY
from ..round5.train import ensemble_rank_pct, ranker_config_r5
from ..round6.registry import R6Spec, SPECS as R6_SPECS
from ..round6.train import (
    apply_label_transform,
    apply_representation,
    TRAIN_SEEDS,
    _local_rolling,
)
from ..round2.overlays import apply_date_multiplier
from ..stock_readonly import load_fs01122_training_panel, load_prepared_market
from .catalog import ReplayJob, train_r5_jobs, train_r6_jobs
from .inject import inject_fs01122_into_overlay_caches
from .runner import _config_for, run_job


def _panel(features: Sequence[str], target: str) -> pd.DataFrame:
    frame = load_fs01122_training_panel()
    if "label_exit_rule_net" in frame.columns and "label_bigloss_5" not in frame.columns:
        frame["label_bigloss_5"] = (
            pd.to_numeric(frame["label_exit_rule_net"], errors="coerce") <= -0.05
        ).astype(float)
        frame.loc[frame["label_exit_rule_net"].isna(), "label_bigloss_5"] = pd.NA
    extra = [
        "trade_date", "instrument_id", target, "actual_exit_date", "next2_date",
        "label_e2_net", "label_exit_rule_net", "label_d1_touch", "label_bigloss_5",
    ]
    cols = []
    for c in list(features) + extra:
        if c in frame.columns and c not in cols:
            cols.append(c)
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise RuntimeError(f"fs01122 panel missing features {missing}")
    return frame[cols].copy()


def _pred_dir(round_name: str, key: str) -> Path:
    path = replay_fs01122_dir(round_name) / "predictions" / str(key)
    path.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    return path


def _model_dir(round_name: str, key: str) -> Path:
    path = replay_fs01122_dir(round_name) / "models" / str(key)
    path.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    return path


def load_local_predictions(round_name: str, key: str) -> pd.DataFrame:
    path = _pred_dir(round_name, key) / "predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    frame = pd.read_parquet(path)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    return frame.loc[frame["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)].copy()


def _parent_features(orig: Sequence[str], family: str | None = None, available: set[str] | None = None) -> tuple[str, ...]:
    feats = list(FS01122_FEATURES)
    locked6 = {
        "fb_limit_close_minute_ratio", "h5_close_limit_retention", "h1_last15m_volume_ratio",
        "g0_amt_vs_5", "ind_first_board_count_log", "ind_stock_seal_quality_rank",
    }
    extras = [c for c in orig if c not in locked6 and c not in feats]
    if family and family in FAMILY_COLUMNS:
        for col in FAMILY_COLUMNS[family]:
            if col not in feats and col not in extras:
                extras.append(col)
            if len([e for e in extras if e not in FS01122_FEATURES]) >= 3:
                break
    out = tuple(dict.fromkeys([*feats, *extras]))
    if available:
        out = tuple(c for c in out if c in available)
        if not out:
            out = tuple(c for c in FS01122_FEATURES if c in available)
    return out


def train_key(
    key: str,
    features: Sequence[str],
    *,
    round_name: str = "round5",
    target: str = "label_e2_net",
    objective: str = "rank:ndcg",
    eval_metric: str = "ndcg@3",
    seeds: Sequence[int] = SEEDS,
    resume: bool = True,
    invert_score: bool = False,
    representation: str = "raw",
    label_transform: str = "none",
    window_kind: str = "3y",
    fold_rule: str = "r5_local",
    half_life_months: int | None = None,
) -> dict[str, Any]:
    dest = _pred_dir(round_name, key)
    ens_path = dest / "predictions.parquet"
    meta_path = dest / "ensemble.json"
    if resume and ens_path.is_file() and meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    feats = list(features)
    panel = _panel(feats, target)
    panel = apply_representation(panel, feats, representation)
    panel, target_use = apply_label_transform(panel, kind=label_transform, target=target)
    years = sorted(pd.to_datetime(panel["trade_date"]).dt.year.unique())
    first_year = max(2018, int(min(years)) + 3) if years else 2018
    mode = "ranker"
    if str(objective).startswith("reg:"):
        mode = "regressor"
    elif str(objective).startswith("binary"):
        mode = "classifier"

    seed_frames = []
    seed_meta = []
    for seed in seeds:
        cfg = ranker_config_r5(
            objective=objective if mode == "ranker" else ("rank:ndcg" if mode == "classifier" else objective),
            eval_metric=eval_metric if str(eval_metric).startswith("ndcg@") else "ndcg@3",
            random_state=int(seed),
        )
        out = _model_dir(round_name, key) / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        seed_pred = out / "predictions.parquet"
        if resume and seed_pred.is_file():
            seed_frames.append(pd.read_parquet(seed_pred))
            seed_meta.append({"seed": int(seed), "status": "resumed"})
            continue
        preds, meta = _local_rolling(
            panel, panel, out,
            features=feats, target=target_use, config=cfg,
            first_prediction_year=first_year, last_prediction_year=2026,
            mode=mode, invert_score=invert_score,
            window_kind=window_kind, fold_rule=fold_rule,
            half_life_months=half_life_months,
        )
        seed_frames.append(preds)
        seed_meta.append({"seed": int(seed), "status": meta.get("status")})
    ensemble = ensemble_rank_pct(seed_frames)
    assert_project_result_path(ens_path)
    ensemble.to_parquet(ens_path, index=False)
    payload = {
        "status": "trained_oos",
        "key": key,
        "features": feats,
        "target": target_use,
        "objective": objective,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "parent": "fs01122_14factor",
        "representation": representation,
        "label_transform": label_transform,
        "window_kind": window_kind,
        "fold_rule": fold_rule,
        "half_life_months": half_life_months,
        "n_rows": int(len(ensemble)),
        "seed_meta": seed_meta,
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def _backtest_preds(code: str, round_name: str, preds: pd.DataFrame, *, prepared, meta, title: str) -> dict[str, Any]:
    job = ReplayJob(
        code=code,
        round_name=round_name,
        title_zh=title,
        overlay=dict(V3_OVERLAY),
        alpha_source="local_retrain",
        stage="train",
    )
    return run_job(job, predictions=preds, prepared_market=prepared, market_meta=meta, resume=True)


_WINDOW_BY_CODE = {
    "R6_B01": "2y",
    "R6_B02": "3y",
    "R6_B03": "5y",
    "R6_B04": "expanding",
}
_REPR_BY_CODE = {
    "R6_C01": "raw",
    "R6_C02": "pct",
    "R6_C03": "robust_z",
    "R6_C04": "hybrid",
}


def _shortlist_window_repr(r6: dict[str, Any]) -> tuple[str | None, str | None]:
    def score(code: str):
        payload = r6.get(code) or {}
        bt = payload.get("backtest") if isinstance(payload, dict) else {}
        if not isinstance(bt, dict):
            return None
        s = bt.get("summary") or {}
        sharpe = s.get("sharpe")
        cagr = s.get("cagr")
        if sharpe is None:
            return None
        return (float(sharpe), float(cagr or 0))

    best_win, best_win_s = None, None
    for code, kind in _WINDOW_BY_CODE.items():
        sc = score(code)
        if sc is not None and (best_win_s is None or sc > best_win_s):
            best_win, best_win_s = kind, sc
    best_repr, best_repr_s = None, None
    for code, kind in _REPR_BY_CODE.items():
        sc = score(code)
        if sc is not None and (best_repr_s is None or sc > best_repr_s):
            best_repr, best_repr_s = kind, sc
    return best_win, best_repr


def run_train(*, resume: bool = True) -> dict[str, Any]:
    ensure_replay_fs01122_tree()
    inject_fs01122_into_overlay_caches()
    loaded = load_prepared_market(warmup_ma5=True)
    prepared, meta = loaded["prepared"], loaded
    out: dict[str, Any] = {"r5": {}, "r6": {}}

    for item in train_r5_jobs():
        code = item["code"]
        key = item.get("prediction_key") or code
        family = item.get("family")
        orig = item.get("features") or []
        if code.startswith("R5_E") and orig:
            # 旧 E 是从六因子删一；本线从 14 因子删对应列，其余用满 14
            drop = [c for c in FS01122_FEATURES if c not in orig]
            feats = tuple(c for c in FS01122_FEATURES if c not in set(drop[:1])) if drop else tuple(orig)
            if set(orig) < set(FS01122_FEATURES) or True:
                # map old 6-factor drop onto 14-factor: drop the same named col if present else skip extras
                dropped = [c for c in {
                    "fb_limit_close_minute_ratio", "h5_close_limit_retention", "h1_last15m_volume_ratio",
                    "g0_amt_vs_5", "ind_first_board_count_log", "ind_stock_seal_quality_rank",
                } if c not in orig]
                feats = tuple(f for f in FS01122_FEATURES if f not in dropped)
        else:
            avail = None
            try:
                avail = set(load_fs01122_training_panel().columns)
            except Exception:
                avail = None
            feats = _parent_features(orig, family, available=avail)
        invert = item.get("target") == "label_bigloss_5"
        print(f"[train r5] {code} key={key} nfeat={len(feats)} obj={item.get('objective')}", flush=True)
        trained = train_key(
            key, feats, round_name="round5",
            target=item.get("target") or "label_e2_net",
            objective=item.get("objective") or "rank:ndcg",
            eval_metric=item.get("eval_metric") or "ndcg@3",
            resume=resume, invert_score=invert,
            window_kind="3y", fold_rule="r5_local",
        )
        if trained.get("status") == "SKIPPED_UNSUPPORTED":
            out["r5"][code] = trained
            continue
        preds = load_local_predictions("round5", key)
        out["r5"][code] = {
            "train": trained,
            "backtest": _backtest_preds(code, "round5", preds, prepared=prepared, meta=meta, title=item["title_zh"]),
        }

    for item in train_r6_jobs():
        code = item["code"]
        key = item.get("prediction_key") or code
        reuse = item.get("reuse_round5_key")
        feats = tuple(FS01122_FEATURES)
        r6_kw = dict(
            target=item.get("target") or "label_exit_rule_net",
            objective=item.get("objective") or "rank:ndcg",
            eval_metric=item.get("eval_metric") or "ndcg@3",
            resume=resume,
            representation=item.get("representation") or "raw",
            label_transform=item.get("label_transform") or "none",
            window_kind=item.get("window_kind") or "3y",
            fold_rule=item.get("fold_rule") or "stock",
            half_life_months=item.get("half_life_months"),
        )
        if code in {"R6_D02", "R6_D03", "R6_D04"}:
            best_win, best_repr = _shortlist_window_repr(out.get("r6") or {})
            if code == "R6_D02" and best_win:
                r6_kw["window_kind"] = best_win
            if code == "R6_D03" and best_repr:
                r6_kw["representation"] = best_repr
            if code == "R6_D04":
                if best_win:
                    r6_kw["window_kind"] = best_win
                if best_repr:
                    r6_kw["representation"] = best_repr
        print(
            f"[train r6] {code} key={key} win={r6_kw['window_kind']} "
            f"repr={r6_kw['representation']} obj={r6_kw['objective']}",
            flush=True,
        )
        if reuse:
            r5_path = _pred_dir("round5", reuse) / "predictions.parquet"
            can_reuse = (
                r5_path.is_file()
                and (item.get("window_kind") or "3y") == "3y"
                and (item.get("representation") or "raw") == "raw"
                and (item.get("label_transform") or "none") == "none"
                and not item.get("half_life_months")
            )
            if can_reuse:
                preds = pd.read_parquet(r5_path)
                dest = _pred_dir("round6", key)
                dest.mkdir(parents=True, exist_ok=True)
                preds.to_parquet(dest / "predictions.parquet", index=False)
                trained = {"status": "reused_r5", "key": key, "from": reuse}
            else:
                trained = train_key(key, feats, round_name="round6", **r6_kw)
        else:
            trained = train_key(key, feats, round_name="round6", **r6_kw)
        if trained.get("status") == "SKIPPED_UNSUPPORTED":
            out["r6"][code] = trained
            continue
        try:
            preds = load_local_predictions("round6", key)
        except FileNotFoundError:
            preds = load_local_predictions("round5", reuse) if reuse else None
        if preds is None:
            out["r6"][code] = {"train": trained, "backtest": {"status": "SKIPPED_NO_PREDICTIONS"}}
            continue
        out["r6"][code] = {
            "train": trained,
            "backtest": _backtest_preds(code, "round6", preds, prepared=prepared, meta=meta, title=item["title_zh"]),
        }

    # R6 F/G/H/K：对齐旧设计，父分改为 14 因子 canonical/pairwise
    try:
        from core.round6.meta import context_multiplier, context_series, top1_agreement_rate
        from core.round6.overlays import attach_stage1
        from core.round6.train import ensemble_rank_pct as r6_ens

        can = None
        pair = None
        for cand in ("CANONICAL", "F00_stock_objective", "D00_6factor_n50d2"):
            for rnd in ("round6", "round5"):
                p = _pred_dir(rnd, cand) / "predictions.parquet"
                if p.is_file() and can is None and cand != "D01_pair_canonical":
                    can = pd.read_parquet(p)
                if p.is_file() and "pair" in cand.lower():
                    pair = pd.read_parquet(p)
        if pair is None:
            p = _pred_dir("round6", "D01_pair_canonical") / "predictions.parquet"
            if p.is_file():
                pair = pd.read_parquet(p)
            else:
                p5 = _pred_dir("round5", "F03_pairwise") / "predictions.parquet"
                if p5.is_file():
                    pair = pd.read_parquet(p5)
        if can is not None:
            can["signal_date"] = pd.to_datetime(can["signal_date"])
            out["r6"]["R6_CANONICAL_RETRAIN"] = out["r6"].get("R6_CANONICAL_RETRAIN") or {
                "backtest": _backtest_preds(
                    "R6_CANONICAL_RETRAIN", "round6", can, prepared=prepared, meta=meta, title="canonical 14 因子",
                )
            }
        if can is not None and pair is not None:
            pair["signal_date"] = pd.to_datetime(pair["signal_date"])
            agr = top1_agreement_rate(can, pair)
            if agr <= 0.95:
                mix = r6_ens([can, pair])
                dest = _pred_dir("round6", "G01_blend")
                mix.to_parquet(dest / "predictions.parquet", index=False)
                out["r6"]["R6_G01"] = _backtest_preds(
                    "R6_G01", "round6", mix, prepared=prepared, meta=meta, title="NDCG+Pairwise 等权",
                )
            else:
                out["r6"]["R6_G01"] = {"status": "ABORT", "reason": f"Top1 一致率 {agr:.2%} > 95%"}
            out["r6"]["R6_D01"] = out["r6"].get("R6_D01") or _backtest_preds(
                "R6_D01", "round6", pair, prepared=prepared, meta=meta, title="pairwise × canonical",
            )
        ctx = context_series()
        if ctx and can is not None:
            parts = [context_multiplier(s) for s in ctx.values()]
            idx = parts[0].index
            for p in parts[1:]:
                idx = idx.union(p.index)
            comb = parts[0].reindex(idx).fillna(1.0)
            for p in parts[1:]:
                comb = comb * p.reindex(idx).fillna(1.0)
            job = ReplayJob(
                code="R6_H01",
                round_name="round6",
                title_zh="context overlay",
                overlay=attach_stage1(dict(V3_OVERLAY), comb.clip(0, 1)),
                stage="train",
            )
            out["r6"]["R6_H01"] = run_job(job, predictions=can, prepared_market=prepared, market_meta=meta, resume=resume)
        # K：把已跑的 canonical / pairwise / G / H 最多 6 个做成 finalist 别名
        k_src = [c for c in ("R6_CANONICAL_RETRAIN", "R6_D01", "R6_D04", "R6_G01", "R6_H01", "R6_B07") if c in out["r6"]]
        for i, src in enumerate(k_src[:6], start=1):
            src_dir = replay_fs01122_dir("round6", src)
            if (src_dir / "report.json").is_file():
                dest = replay_fs01122_dir("round6", f"R6_K{i:02d}")
                dest.mkdir(parents=True, exist_ok=True)
                for name in ("report.json", "summary.json", "nav.parquet", "trades.parquet"):
                    s = src_dir / name
                    if s.is_file():
                        dest.joinpath(name).write_bytes(s.read_bytes())
                note = dest / "alias.json"
                note.write_text(json.dumps({"alias_of": src, "parent": "fs01122_14factor"}, indent=2) + "\n")
                out["r6"][f"R6_K{i:02d}"] = {"status": "alias", "of": src}
    except Exception as exc:
        out["r6_structure_error"] = repr(exc)

    _ = R5Spec, R6Spec, R6_SPECS, TRAIN_SEEDS, r5_build, apply_date_multiplier
    _ = run_local_fixed_sleeve_backtest, write_backtest_artifacts, attach_round4_fields, _config_for
    return out
