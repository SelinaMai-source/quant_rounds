"""收集旧表出现过的 overlay 代号。F 组标 ABORT_SEMANTIC_SHIFT。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..experiment_registry import EXPERIMENTS as R1_EXPERIMENTS
from ..paths import PROJECT_ROOT, RESULT_ROOT
from ..round2.registry import EXPERIMENTS as R2_EXPERIMENTS
from ..round3.registry import SHORTLIST as R3_SHORTLIST
from ..round4.registry import SPECS as R4_SPECS
from ..round5.registry import SPECS as R5_SPECS
from ..round6.registry import SPECS as R6_SPECS
from .contract import F_ABORT_PREFIXES


@dataclass
class ReplayJob:
    code: str
    round_name: str  # round1 … round6
    title_zh: str
    overlay: dict[str, Any] = field(default_factory=dict)
    exit_mode: str = "close_from_d2"
    name_budget_nav_fraction: float = 0.3
    market_up_ratio_min: float = 0.45
    top_k: int | None = None
    rank_name_budget_nav_fractions: tuple[float, ...] | None = None
    skip_years: tuple[int, ...] = ()
    seeds: tuple[int, ...] | None = None
    seed_leave_out: int | None = None
    cost_overrides: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    diagnostic_only: bool = False
    skip: bool = False
    skip_reason: str = ""
    alpha_source: str = "locked_fs01122"
    abort: str | None = None
    needs_overlay_loop: bool = False
    pit_correct_g02: bool = False
    stage: str = "overlay"


def _cfg_dir(round_name: str) -> Path:
    return RESULT_ROOT / round_name / "configs"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _f_abort(code: str, top_k: int | None, fracs) -> str | None:
    if code == "R4_F_TOP3_diag":
        return None
    if any(code.startswith(p) or code == p for p in F_ABORT_PREFIXES):
        return "ABORT_SEMANTIC_SHIFT"
    if top_k is not None and int(top_k) >= 2 and code.startswith("R4_") and "F" in code:
        return "ABORT_SEMANTIC_SHIFT"
    if fracs is not None and tuple(fracs) not in {(0.25, 0.15), (0.10, 0.10, 0.10)} and int(top_k or 1) >= 2:
        if code.startswith("R4_F") or code.startswith("R4_P_11") or code.startswith("R4_P_12") or code == "R4_T06":
            return "ABORT_SEMANTIC_SHIFT"
    return None


def _from_r2(spec) -> ReplayJob:
    return ReplayJob(
        code=spec.code,
        round_name="round2",
        title_zh=spec.title_zh,
        overlay=dict(spec.overlay or {}),
        exit_mode=spec.exit_mode,
        name_budget_nav_fraction=float(spec.name_budget_nav_fraction),
        market_up_ratio_min=float(spec.market_up_ratio_min),
        diagnostic_only=bool(spec.diagnostic_only),
        skip=bool(spec.skip),
        skip_reason=spec.skip_reason,
        needs_overlay_loop=bool(spec.needs_overlay_loop),
        stage="overlay",
    )


def overlay_jobs() -> list[ReplayJob]:
    jobs: list[ReplayJob] = []
    seen: set[str] = set()

    def add(job: ReplayJob) -> None:
        if job.code in seen:
            return
        seen.add(job.code)
        abort = _f_abort(job.code, job.top_k, job.rank_name_budget_nav_fractions)
        if abort:
            job.abort = abort
        jobs.append(job)

    for spec in R1_EXPERIMENTS.values():
        add(ReplayJob(
            code=spec.code,
            round_name="round1",
            title_zh=spec.title_zh,
            overlay={},
            exit_mode=spec.exit_mode,
            stage="exit",
        ))

    for spec in R2_EXPERIMENTS.values():
        add(_from_r2(spec))
    for path in sorted(_cfg_dir("round2").glob("R2_*.json")):
        raw = _load_json(path) or {}
        code = str(raw.get("code") or path.stem)
        add(ReplayJob(
            code=code,
            round_name="round2",
            title_zh=str(raw.get("title_zh") or code),
            overlay=dict(raw.get("overlay") or {}),
            exit_mode=str(raw.get("exit_mode") or "close_from_d2"),
            name_budget_nav_fraction=float(raw.get("name_budget_nav_fraction") or 0.3),
            market_up_ratio_min=float(raw.get("market_up_ratio_min") or 0.45),
        ))

    for spec in R3_SHORTLIST.values():
        add(ReplayJob(
            code=spec.code,
            round_name="round3",
            title_zh=spec.title_zh,
            overlay=dict(spec.overlay or {}),
            exit_mode=spec.exit_mode,
            name_budget_nav_fraction=float(spec.name_budget_nav_fraction),
            market_up_ratio_min=float(spec.market_up_ratio_min),
            skip_years=tuple(spec.skip_years or ()),
            seeds=spec.seeds,
            seed_leave_out=spec.seed_leave_out,
            cost_overrides=dict(spec.cost_overrides or {}),
            pit_correct_g02=bool(spec.pit_correct_g02),
        ))
    for path in sorted(_cfg_dir("round3").glob("R3_*.json")):
        raw = _load_json(path) or {}
        code = str(raw.get("code") or path.stem)
        seeds = raw.get("seeds")
        add(ReplayJob(
            code=code,
            round_name="round3",
            title_zh=str(raw.get("title_zh") or code),
            overlay=dict(raw.get("overlay") or {}),
            exit_mode=str(raw.get("exit_mode") or "close_from_d2"),
            name_budget_nav_fraction=float(raw.get("name_budget_nav_fraction") or 0.3),
            market_up_ratio_min=float(raw.get("market_up_ratio_min") or 0.45),
            skip_years=tuple(raw.get("skip_years") or ()),
            seeds=tuple(seeds) if seeds else None,
            seed_leave_out=raw.get("seed_leave_out"),
            cost_overrides=dict(raw.get("cost_overrides") or {}),
            pit_correct_g02=bool(raw.get("pit_correct_g02")),
        ))

    for spec in R4_SPECS.values():
        add(ReplayJob(
            code=spec.code,
            round_name="round4",
            title_zh=spec.title_zh,
            overlay=dict(spec.overlay or {}),
            exit_mode=spec.exit_mode,
            name_budget_nav_fraction=float(spec.name_budget_nav_fraction),
            market_up_ratio_min=float(spec.market_up_ratio_min),
            top_k=int(spec.top_k),
            rank_name_budget_nav_fractions=(
                tuple(spec.rank_name_budget_nav_fractions)
                if spec.rank_name_budget_nav_fractions is not None else None
            ),
            skip_years=tuple(spec.skip_years or ()),
            seeds=spec.seeds,
            seed_leave_out=spec.seed_leave_out,
            cost_overrides=dict(spec.cost_overrides or {}),
            random_seed=spec.random_seed,
            diagnostic_only=bool(spec.diagnostic),
        ))
    for path in sorted(_cfg_dir("round4").glob("R4_*.json")):
        raw = _load_json(path) or {}
        code = str(raw.get("code") or path.stem)
        fracs = raw.get("rank_name_budget_nav_fractions")
        add(ReplayJob(
            code=code,
            round_name="round4",
            title_zh=str(raw.get("title_zh") or code),
            overlay=dict(raw.get("overlay") or {}),
            exit_mode=str(raw.get("exit_mode") or "close_from_d2"),
            name_budget_nav_fraction=float(raw.get("name_budget_nav_fraction") or 0.3),
            market_up_ratio_min=float(raw.get("market_up_ratio_min") or 0.45),
            top_k=int(raw["top_k"]) if raw.get("top_k") is not None else None,
            rank_name_budget_nav_fractions=tuple(fracs) if fracs else None,
            skip_years=tuple(raw.get("skip_years") or ()),
            cost_overrides=dict(raw.get("cost_overrides") or {}),
            random_seed=raw.get("random_seed"),
            diagnostic_only=bool(raw.get("diagnostic")),
        ))

    for spec in R5_SPECS.values():
        if spec.alpha_source != "locked_fs00263":
            continue
        add(ReplayJob(
            code=spec.code,
            round_name="round5",
            title_zh=spec.title_zh,
            overlay=dict(spec.overlay or {}),
            exit_mode=spec.exit_mode,
            name_budget_nav_fraction=float(spec.name_budget_nav_fraction),
            market_up_ratio_min=float(spec.market_up_ratio_min),
            alpha_source="locked_fs01122",
            stage="locked_overlay",
        ))
    for spec in R6_SPECS.values():
        if spec.alpha_source != "locked_fs00263":
            continue
        if spec.forensic_only:
            continue
        add(ReplayJob(
            code=spec.code,
            round_name="round6",
            title_zh=spec.title_zh,
            overlay=dict(spec.overlay or {}),
            exit_mode=spec.exit_mode,
            name_budget_nav_fraction=float(spec.name_budget_nav_fraction),
            market_up_ratio_min=float(spec.market_up_ratio_min),
            alpha_source="locked_fs01122",
            diagnostic_only=bool(spec.diagnostic),
            stage="locked_overlay",
        ))

    # Top3 诊断：旧 F 在已是 Top2 的底上改为 Top3，避免「Top2 上再 Top2」
    add(ReplayJob(
        code="R4_F_TOP3_diag",
        round_name="round4",
        title_zh="诊断 Top3 10/10/10（替换旧 F Top2 语义）",
        overlay={"kind": "stack", "market": {"kind": "identity"}, "confidence": {"kind": "margin", "low_mult": 0.5, "cutoff": 0.30}},
        top_k=3,
        rank_name_budget_nav_fractions=(0.10, 0.10, 0.10),
        diagnostic_only=True,
        stage="f_diag",
    ))
    _ = PROJECT_ROOT
    return jobs


def train_r5_jobs() -> list[dict[str, Any]]:
    out = []
    for spec in R5_SPECS.values():
        if spec.alpha_source != "local_retrain":
            continue
        if not str(spec.code).startswith("R5_"):
            continue
        family = spec.code[3]
        if family not in "DEFG":
            continue
        out.append({
            "code": spec.code,
            "title_zh": spec.title_zh,
            "prediction_key": spec.prediction_key,
            "features": list(spec.features or ()),
            "objective": spec.objective,
            "eval_metric": spec.eval_metric,
            "target": spec.target,
            "family": spec.family,
            "stage": spec.stage,
        })
    return out


def train_r6_jobs() -> list[dict[str, Any]]:
    out = []
    for spec in R6_SPECS.values():
        tags = set(spec.tags or ())
        if spec.forensic_only:
            continue
        if spec.alpha_source == "locked_fs00263":
            continue
        if spec.stage in {"parity", "forensics"}:
            continue
        letter = spec.code.split("_")[1][:1] if "_" in spec.code else ""
        if spec.code == "R6_CANONICAL_RETRAIN" or letter in "BCDE":
            out.append({
                "code": spec.code,
                "title_zh": spec.title_zh,
                "prediction_key": spec.prediction_key,
                "reuse_round5_key": spec.reuse_round5_key,
                "features": list(spec.features or ()),
                "objective": spec.objective,
                "eval_metric": spec.eval_metric,
                "target": spec.target,
                "window_kind": spec.window_kind,
                "representation": spec.representation,
                "label_transform": spec.label_transform,
                "half_life_months": spec.half_life_months,
                "fold_rule": spec.fold_rule,
                "use_stock_rolling": spec.use_stock_rolling,
                "stage": spec.stage,
            })
    return out
