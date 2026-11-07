"""Round 11 冻结契约。真值只认四份 legacy artifact，禁止用 prompt 重写数字。

v1 = R7_FS01122_DUAL90_10_CLEAN
v2 = replay_round7 的 R4_E_top1_minus_median（不是旧 fs00263 的 result/round4/R4_E）
v3 = R8_E02
v4 = R9_B03 / R10_00（CURRENT；historical，非 final OOS）
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..paths import (
    PROJECT_ROOT,
    RESULT_ROOT,
    REPLAY_ROUND7_ROOT,
    ROUND4_ROOT,
    ROUND7_ROOT,
    ROUND8_ROOT,
    ROUND9_ROOT,
    ROUND11_ROOT,
)
from ..stock_readonly import sha256_file

BASELINE_V1 = "R7_FS01122_DUAL90_10_CLEAN"
BASELINE_V2 = "R4_E_top1_minus_median"
BASELINE_V3 = "R8_E02"
BASELINE_V4 = "R9_B03"

V1_DIR = ROUND7_ROOT / BASELINE_V1
V2_DIR = REPLAY_ROUND7_ROOT / "round4" / BASELINE_V2
V3_DIR = ROUND8_ROOT / BASELINE_V3
V4_DIR = ROUND9_ROOT / BASELINE_V4
FORBIDDEN_OLD_V2_DIR = ROUND4_ROOT / BASELINE_V2

SNAPSHOT_ID = "hermes-cn-a-2010-20260724-v7-all-features"
SNAPSHOT_FINGERPRINT = "1306e83ba88ba79340d2a8d824b914938fe9808bfcb1560d6abcf3c3c188304c"
PARITY_ABS_NAV = 1e-4
PARITY_METRIC_ABS = 1e-8
RANK_NAV = (0.3, 0.1)
TOP_K = 2

# 精确值来自各 artifact report.json / summary.json，禁止四舍五入后再当目标。
V1_TARGETS = {
    "cagr": 0.46980280179436673,
    "sharpe": 1.6684780129023091,
    "max_drawdown": -0.25415996098636673,
    "completed_trades": 2158,
    "total_return": 25.979409141148523,
    "loyo_median": 0.40250567256021896,
    "loyo_worst": 0.2888546366399338,
}
V2_TARGETS = {
    "cagr": 0.5075432102926214,
    "sharpe": 1.716603623385515,
    "max_drawdown": -0.24367908587033738,
    "completed_trades": 1787,
    "total_return": 32.51487361549896,
    "loyo_median": 0.4387436528489338,
    "loyo_worst": 0.3158490870739581,
}
V3_TARGETS = {
    "cagr": 0.5299285369905697,
    "sharpe": 1.7964712763996256,
    "max_drawdown": -0.24504544808039253,
    "completed_trades": 1783,
    "total_return": 37.01949791961103,
    "loyo_median": 0.4617879099240748,
    "loyo_worst": 0.3304383573245333,
}
V4_TARGETS = {
    "cagr": 0.5373748002477328,
    "sharpe": 1.8433564243143739,
    "max_drawdown": -0.24751010811952434,
    "completed_trades": 1149,
    "total_return": 38.63211154586613,
    "loyo_median": 0.47336040019298853,
    "loyo_worst": 0.3354717292490337,
}

BLEND_PRED = ROUND7_ROOT / "predictions" / "dual90_10_gap_floor" / "predictions.parquet"
RISK_PRED = ROUND7_ROOT / "predictions" / "clean_risk" / "predictions.parquet"

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
MANIFEST_NAME = "round11_validation_manifest_v1.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round11_validation_audit_a5113ff6.plan.md")
PLAN_COPY = ROUND11_ROOT / "ROUND11_PLAN.md"
CONTRACT_MD = ROUND11_ROOT / "00_baseline_contracts.md"
PARITY_MD = ROUND11_ROOT / "01_reproducibility_parity.md"

STRATEGY_SOURCE_FILES = (
    "core/local_executor.py",
    "core/replay_round7/inject.py",
    "core/replay_round7/contract.py",
    "core/round2/overlays.py",
    "core/round4/overlays.py",
    "core/round7/contract.py",
    "core/round8/overlays.py",
    "core/round8/geometry.py",
    "core/round8/contract.py",
    "core/round9/overlays.py",
    "core/round9/allocator.py",
    "core/round9/geometry.py",
    "core/round9/contract.py",
)

PROTOCOL_SOURCE_FILES = (
    "core/round11/contract.py",
    "core/round11/freeze.py",
    "core/round11/runner.py",
    "core/round11/parity.py",
    "core/round11/pipeline.py",
    "core/round11/common.py",
    "core/round11/pit.py",
    "core/round11/engine.py",
    "core/round11/cost.py",
    "core/round11/time_stage.py",
    "core/round11/model.py",
    "core/round11/placebo.py",
    "core/round11/trials.py",
    "core/round11/alphaforge.py",
    "core/round11/stats.py",
    "core/round11/capacity.py",
    "core/round11/dossier.py",
    "tools/run_round11_validation.py",
    "scripts/run_round11_validation.py",
)


@dataclass(frozen=True)
class BaselineSpec:
    version: str
    code: str
    title_zh: str
    anchor_dir: Path
    overlay_spec: dict[str, Any]
    predictions: str
    targets: dict[str, Any]
    parent: str
    replay_code: str
    notes: tuple[str, ...] = field(default_factory=tuple)


BASELINES: dict[str, BaselineSpec] = {
    "v1": BaselineSpec(
        version="v1",
        code=BASELINE_V1,
        title_zh="Round7 双头 90/10 + clean_risk + Top2 30/10",
        anchor_dir=V1_DIR,
        overlay_spec={"kind": "v1_clean"},
        predictions="locked_risk",
        targets=dict(V1_TARGETS),
        parent="(none; locked candidate)",
        replay_code="R11_V1",
        notes=("clean_risk 日乘子已写入预测", "不是再叠 v2 市场层"),
    ),
    "v2": BaselineSpec(
        version="v2",
        code=BASELINE_V2,
        title_zh="replay 契约下 B01a × top1−median（不是旧 fs00263 R4_E）",
        anchor_dir=V2_DIR,
        overlay_spec={"kind": "exact_v2"},
        predictions="locked_alpha",
        targets=dict(V2_TARGETS),
        parent=BASELINE_V1,
        replay_code="R11_V2",
        notes=(
            "真值目录 = result/replay_round7/round4/R4_E_top1_minus_median",
            "禁止使用 result/round4/R4_E_top1_minus_median（旧 fs00263）",
            "used_locked_round7_risk=false",
        ),
    ),
    "v3": BaselineSpec(
        version="v3",
        code=BASELINE_V3,
        title_zh="E02：双头分歧只乘 Top2（A/B/C = 1.00/0.75/0.50）",
        anchor_dir=V3_DIR,
        overlay_spec={"kind": "exact_e02"},
        predictions="locked_alpha",
        targets=dict(V3_TARGETS),
        parent=BASELINE_V2,
        replay_code="R11_V3",
        notes=("Top1 LOCK = v2", "e02 只改 Top2 乘子"),
    ),
    "v4": BaselineSpec(
        version="v4",
        code=BASELINE_V4,
        title_zh="B03 conservative replacement 九格；CURRENT historical 基线",
        anchor_dir=V4_DIR,
        overlay_spec={"kind": "preregistered", "allocator": "B03_conservative"},
        predictions="locked_alpha",
        targets=dict(V4_TARGETS),
        parent=BASELINE_V3,
        replay_code="R11_V4",
        notes=(
            "HIGH=A / MEDIUM=B / LOW=C",
            "replacement：A→0/5/10；B→0/0/5；C→0/0/0",
            "不再乘 E02 的 1.00/0.75/0.50",
            "historical，非 final OOS / production",
        ),
    ),
}


def replay_dir(spec: BaselineSpec) -> Path:
    return ROUND11_ROOT / "parity" / spec.replay_code


def source_tree_sha256(files: tuple[str, ...] | None = None) -> dict[str, Any]:
    """策略代码树哈希。无 git 不假装 commit。"""
    rels = files or STRATEGY_SOURCE_FILES
    per_file: dict[str, str] = {}
    tree = hashlib.sha256()
    missing: list[str] = []
    for rel in rels:
        path = PROJECT_ROOT / rel
        if not path.is_file():
            per_file[rel] = "MISSING"
            missing.append(rel)
            continue
        digest = sha256_file(path)
        per_file[rel] = digest
        tree.update(rel.encode("utf-8"))
        tree.update(b"\0")
        tree.update(digest.encode("ascii"))
        tree.update(b"\n")
    return {
        "algorithm": "sha256",
        "method": "sorted_relpath_plus_file_digest",
        "git_commit": None,
        "git_commit_note": "本课题无可用 git commit；不伪造",
        "tree": tree.hexdigest(),
        "files": per_file,
        "missing": missing,
        "n_files": len(rels),
        "n_missing": len(missing),
    }


def artifact_file_hashes(directory: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("nav.parquet", "trades.parquet", "report.json", "summary.json", "config.json"):
        path = directory / name
        if path.is_file():
            out[name] = sha256_file(path)
    return out


def _metric_drift(observed: dict[str, Any], targets: dict[str, Any]) -> dict[str, float]:
    drift: dict[str, float] = {}
    for key, target in targets.items():
        got = observed.get(key)
        if got is None:
            drift[key] = float("inf")
            continue
        if key == "completed_trades":
            drift[key] = float(abs(int(got) - int(target)))
        else:
            drift[key] = abs(float(got) - float(target))
    return drift


def inspect_baseline(spec: BaselineSpec) -> dict[str, Any]:
    """从 artifact 读真值，不靠 prompt 回忆。"""
    import pandas as pd

    report_path = spec.anchor_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing anchor report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = {}
    config_path = spec.anchor_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = {}
    summary_path = spec.anchor_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trades = pd.read_parquet(spec.anchor_dir / "trades.parquet")
    nav = pd.read_parquet(spec.anchor_dir / "nav.parquet")
    trades["signal_date"] = pd.to_datetime(trades["signal_date"]).dt.normalize()
    ranks = trades.groupby("signal_date")["score"].rank(method="first", ascending=False)
    metrics = report.get("metrics") or {}
    observed = {
        "cagr": metrics.get("annualized_return", summary.get("cagr")),
        "sharpe": metrics.get("sharpe_zero_rate", summary.get("sharpe")),
        "max_drawdown": metrics.get("max_drawdown", summary.get("max_drawdown")),
        "completed_trades": metrics.get("completed_trades", summary.get("completed_trades")),
        "total_return": metrics.get("total_return", summary.get("total_return")),
        "loyo_median": metrics.get("loyo_median", summary.get("loyo_median")),
        "loyo_worst": metrics.get("loyo_worst", summary.get("loyo_worst")),
    }
    costs = (report.get("config") or {}).get("costs") or {}
    unique_mult = []
    if "allocation_multiplier" in trades.columns:
        unique_mult = sorted(float(x) for x in trades["allocation_multiplier"].dropna().unique())
    day_nunique = 0
    n_signal_days = 0
    if "allocation_multiplier" in trades.columns:
        day_nunique = int(trades.groupby("signal_date")["allocation_multiplier"].nunique().gt(1).sum())
        n_signal_days = int(trades["signal_date"].nunique())
    drift = _metric_drift(observed, spec.targets)
    return {
        "version": spec.version,
        "code": spec.code,
        "anchor_dir": str(spec.anchor_dir),
        "exists": True,
        "n_trades": int(len(trades)),
        "n_nav": int(len(nav)),
        "n_rank1": int((ranks == 1).sum()),
        "n_rank2": int((ranks == 2).sum()),
        "unique_multipliers": unique_mult,
        "days_with_split_mult": day_nunique,
        "n_signal_days_in_trades": n_signal_days,
        "observed": observed,
        "targets": dict(spec.targets),
        "anchor_drift": drift,
        "anchor_matches_hardcoded": all(
            (v < 1e-12) if k != "completed_trades" else (v == 0.0) for k, v in drift.items()
        ),
        "snapshot_id": report.get("snapshot_id"),
        "snapshot_fingerprint": report.get("snapshot_fingerprint"),
        "top_k": report.get("top_k") or (report.get("config") or {}).get("top_k"),
        "fracs": report.get("rank_name_budget_nav_fractions")
        or (report.get("config") or {}).get("rank_name_budget_nav_fractions"),
        "exit_mode": (report.get("config") or {}).get("exit_mode"),
        "costs": costs,
        "sell_tax_bps": costs.get("sell_tax_bps"),
        "used_locked_round7_risk": report.get("used_locked_round7_risk"),
        "overlay_spec": config.get("overlay_spec") or config.get("overlay") or report.get("overlay"),
        "yearly_returns": metrics.get("yearly_returns") or summary.get("yearly_returns") or {},
        "artifact_sha256": artifact_file_hashes(spec.anchor_dir),
        "forbidden_old_v2_exists": FORBIDDEN_OLD_V2_DIR.is_dir(),
        "forbidden_old_v2_dir": str(FORBIDDEN_OLD_V2_DIR),
    }


def inspect_all() -> dict[str, Any]:
    facts = {key: inspect_baseline(spec) for key, spec in BASELINES.items()}
    all_ok = all(item.get("anchor_matches_hardcoded") for item in facts.values())
    snaps = {item.get("snapshot_id") for item in facts.values()}
    fps = {item.get("snapshot_fingerprint") for item in facts.values()}
    return {
        "baselines": facts,
        "all_anchors_match_hardcoded": all_ok,
        "snapshot_ids": sorted(str(x) for x in snaps),
        "snapshot_fingerprints": sorted(str(x) for x in fps),
        "shared_snapshot": snaps == {SNAPSHOT_ID} and fps == {SNAPSHOT_FINGERPRINT},
        "old_fs00263_v2_present": FORBIDDEN_OLD_V2_DIR.is_dir(),
        "v2_canonical_dir": str(V2_DIR),
        "v2_forbidden_dir": str(FORBIDDEN_OLD_V2_DIR),
    }


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round11_tree

    ensure_round11_tree()
    assert_project_result_path(PLAN_COPY)
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    if not text.startswith("<!-- 本文件是开工计划"):
        PLAN_COPY.write_text(header + text, encoding="utf-8")
    else:
        PLAN_COPY.write_text(text, encoding="utf-8")
    return PLAN_COPY


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def write_contract_markdown(facts: dict[str, Any] | None = None) -> Path:
    from ..paths import assert_project_result_path, ensure_round11_tree

    ensure_round11_tree()
    payload = facts or inspect_all()
    baselines = payload["baselines"]
    tree = source_tree_sha256()
    pred_hashes = {
        "blend": sha256_file(BLEND_PRED) if BLEND_PRED.is_file() else "MISSING",
        "risk": sha256_file(RISK_PRED) if RISK_PRED.is_file() else "MISSING",
    }

    def _row(key: str) -> dict[str, Any]:
        return baselines[key]

    def _status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    v1, v2, v3, v4 = _row("v1"), _row("v2"), _row("v3"), _row("v4")
    lines = [
        "# Round 11 · baseline 契约（v1–v4，artifact 真值）",
        "",
        "本文只认 **已有账本**，不认 prompt 回忆。不覆盖 legacy `result/roundN/`。",
        "本轮是核验 / 过拟合审计，**不产生 `baseline_v5`**，不为过关改策略。",
        "",
        "## 冻结 lineage",
        "",
        "| 版本 | 代码 | 真值目录 | 禁止当成真值 |",
        "|---|---|---|---|",
        f"| v1 | `{BASELINE_V1}` | `{V1_DIR}` | — |",
        f"| v2 | `{BASELINE_V2}` | `{V2_DIR}` | `{FORBIDDEN_OLD_V2_DIR}`（旧 fs00263 Top1） |",
        f"| v3 | `{BASELINE_V3}` | `{V3_DIR}` | — |",
        f"| v4 CURRENT | `{BASELINE_V4}`（= R10_00） | `{V4_DIR}` | 把 v4 写成 final OOS |",
        "",
        f"- 旧 fs00263 `result/round4/R4_E_top1_minus_median` 是否存在：**{'是' if payload.get('old_fs00263_v2_present') else '否'}**。存在也不准当 v2。",
        f"- 共享 snapshot：`{SNAPSHOT_ID}` / `{SNAPSHOT_FINGERPRINT}`。本轮观察一致 = **{_status(payload.get('shared_snapshot'))}**。",
        f"- 硬编码目标 vs artifact：`all_anchors_match_hardcoded` = **{_status(payload.get('all_anchors_match_hardcoded'))}**。",
        "",
        "## 执行器与费用（四条账本相同）",
        "",
        "- 正式账本：`local_executor` → Stock `fixed_sleeve_backtest` / `run_fixed_dual_sleeve_top3_backtest`",
        "- 双袖套，初始现金 30 万（每袖 15 万）",
        "- `top_k=2`，`rank_name_budget_nav_fractions=[0.3, 0.1]`",
        "- `exit_mode=close_from_d2`，`max_holding_sessions=10`",
        "- 当前契约费用：滑点 12.3bp、佣金 1bp、过户 0.1bp、**卖印花税 5bp 全样本**",
        "- 引擎**不支持按日印花税**。历史 10bp→5bp 修正只属于后续 `cost` stage，本 PHASE 不跑。",
        "",
        "## 锁定数字（report.metrics / summary）",
        "",
        "| 版本 | CAGR | Sharpe | MaxDD | 笔数 | 累计 | LOYO med | LOYO worst | artifact 对齐 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in ("v1", "v2", "v3", "v4"):
        item = _row(key)
        obs = item["observed"]
        lines.append(
            "| {ver} `{code}` | {cagr} | {sharpe:.4f} | {mdd} | {n} | {tr:.6f} | {lm} | {lw} | {ok} |".format(
                ver=key,
                code=item["code"],
                cagr=_pct(obs.get("cagr")),
                sharpe=float(obs.get("sharpe") or 0),
                mdd=_pct(obs.get("max_drawdown")),
                n=obs.get("completed_trades"),
                tr=float(obs.get("total_return") or 0),
                lm=_pct(obs.get("loyo_median")),
                lw=_pct(obs.get("loyo_worst")),
                ok=_status(item.get("anchor_matches_hardcoded")),
            )
        )
    lines += [
        "",
        "精确值（禁止再四舍五入当目标）：",
        "",
    ]
    for key in ("v1", "v2", "v3", "v4"):
        item = _row(key)
        obs = item["observed"]
        lines.append(
            f"- `{key}` CAGR=`{obs.get('cagr')}` Sharpe=`{obs.get('sharpe')}` "
            f"MaxDD=`{obs.get('max_drawdown')}` trades=`{obs.get('completed_trades')}` "
            f"total=`{obs.get('total_return')}`"
        )
    lines += [
        "",
        "## 分版本 checklist（来自 artifact）",
        "",
        "### v1 · `R7_FS01122_DUAL90_10_CLEAN`",
        "",
        "| 项 | 契约 | artifact | 状态 |",
        "|---|---|---|---|",
        f"| 预测 | Round7 `clean_risk` | `{RISK_PRED}` | {_status(RISK_PRED.is_file())} |",
        f"| overlay | 风险乘子已写入预测；不再叠 B01a | overlay=`{v1.get('overlay_spec')}` | 记录 |",
        f"| 容量 | Top2 30/10 | top_k=`{v1.get('top_k')}` fracs=`{v1.get('fracs')}` | {_status(int(v1.get('top_k') or 0)==2)} |",
        f"| 退出 | close_from_d2 | `{v1.get('exit_mode')}` | {_status(v1.get('exit_mode')=='close_from_d2')} |",
        f"| 卖印花税 | 5bp | `{v1.get('sell_tax_bps')}` | {_status(float(v1.get('sell_tax_bps') or 0)==5.0)} |",
        f"| 笔数 | 2158 | {v1.get('n_trades')}（Top1 {v1.get('n_rank1')} / Top2 {v1.get('n_rank2')}） | {_status(v1.get('n_trades')==2158)} |",
        "",
        "### v2 · replay `R4_E_top1_minus_median`",
        "",
        "| 项 | 契约 | artifact | 状态 |",
        "|---|---|---|---|",
        f"| 真值目录 | replay_round7 | `{v2.get('anchor_dir')}` | {_status(Path(str(v2.get('anchor_dir'))).is_dir())} |",
        f"| 不是旧 fs00263 | 禁止 `{FORBIDDEN_OLD_V2_DIR}` | 旧目录存在={payload.get('old_fs00263_v2_present')} | 已隔离 |",
        f"| clean_risk | **否** | used_locked_round7_risk=`{v2.get('used_locked_round7_risk')}` | {_status(v2.get('used_locked_round7_risk') is False)} |",
        f"| 市场×截面 | B01a 45/50/60 × top1−median 底20%×0.5 | overlay=`{v2.get('overlay_spec')}` | 记录 |",
        f"| 乘子粒度 | 整日同乘 | 同日不同乘子天数={v2.get('days_with_split_mult')} / {v2.get('n_signal_days_in_trades')} | {_status(v2.get('days_with_split_mult')==0)} |",
        f"| 笔数 | 1787 | {v2.get('n_trades')}（Top1 {v2.get('n_rank1')} / Top2 {v2.get('n_rank2')}） | {_status(v2.get('n_trades')==1787)} |",
        "",
        "观察到的成交乘子：" + ", ".join(f"{x:g}" for x in (v2.get("unique_multipliers") or [])) + "。",
        "",
        "### v3 · `R8_E02`",
        "",
        "| 项 | 契约 | artifact | 状态 |",
        "|---|---|---|---|",
        f"| overlay | custom / e02=true；只乘 Top2 | `{v3.get('overlay_spec')}` | 记录 |",
        f"| Top1 | LOCK = v2 | — | 代码锁 |",
        f"| 笔数 | 1783 | {v3.get('n_trades')}（Top1 {v3.get('n_rank1')} / Top2 {v3.get('n_rank2')}） | {_status(v3.get('n_trades')==1783)} |",
        f"| 卖印花税 | 5bp | `{v3.get('sell_tax_bps')}` | {_status(float(v3.get('sell_tax_bps') or 0)==5.0)} |",
        "",
        "### v4 · `R9_B03`（CURRENT）",
        "",
        "| 项 | 契约 | artifact | 状态 |",
        "|---|---|---|---|",
        f"| overlay | preregistered / B03_conservative | `{v4.get('overlay_spec')}` | 记录 |",
        f"| 九格 | replacement 0/5/10，不再乘 E02 | A→0/5/10；B→0/0/5；C→0/0/0 | 代码锁 |",
        f"| 笔数 | 1149 | {v4.get('n_trades')}（Top1 {v4.get('n_rank1')} / Top2 {v4.get('n_rank2')}） | {_status(v4.get('n_trades')==1149)} |",
        f"| 身份 | historical / NOT_final_OOS | — | 锁定 |",
        "",
        "## 分年收益（artifact yearly_returns）",
        "",
    ]
    years = sorted({
        str(y)
        for item in baselines.values()
        for y in (item.get("yearly_returns") or {})
    })
    header = "| 年 | " + " | ".join(f"{k}" for k in ("v1", "v2", "v3", "v4")) + " |"
    sep = "|---|" + "---:|" * 4
    lines += [header, sep]
    for year in years:
        cells = []
        for key in ("v1", "v2", "v3", "v4"):
            yr = (baselines[key].get("yearly_returns") or {}).get(year)
            if yr is None:
                yr = (baselines[key].get("yearly_returns") or {}).get(int(year) if year.isdigit() else year)
            cells.append(_pct(yr) if yr is not None else "—")
        lines.append(f"| {year} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## artifact sha256（freeze 前已计算，不改文件）",
        "",
    ]
    for key in ("v1", "v2", "v3", "v4"):
        item = _row(key)
        lines.append(f"### {key} `{item['code']}`")
        lines.append("")
        for name, digest in (item.get("artifact_sha256") or {}).items():
            lines.append(f"- `{name}` = `{digest}`")
        lines.append("")
    lines += [
        "## 锁定预测 sha256",
        "",
        f"- `dual90_10_gap_floor/predictions.parquet` = `{pred_hashes['blend']}`",
        f"- `clean_risk/predictions.parquet` = `{pred_hashes['risk']}`",
        "",
        "## source_tree_sha256（策略代码，无 git commit）",
        "",
        f"- tree = `{tree['tree']}`",
        f"- method = `{tree['method']}`",
        f"- git_commit = `{tree['git_commit']}`（{tree['git_commit_note']}）",
        "",
    ]
    for rel, digest in tree["files"].items():
        lines.append(f"- `{rel}` = `{digest}`")
    lines += [
        "",
        "## 本 PHASE 明确不做",
        "",
        "- 不改策略、不升 v5、不为过关调参",
        "- 不跑 cost-corrected / placebo / AlphaForge / dossier",
        "- 不写追踪表「写入时间」",
        "- 不覆盖 legacy 账本；复现产物只写 `result/round11/parity/`",
        "",
        "## 复现入口",
        "",
        "```bash",
        "python tools/run_round11_validation.py --stage parity",
        "```",
        "",
        "manifest 必须在测试前冻结。parity 失败 = `REPRODUCIBILITY_FAIL`，停止后续 stage。",
        "",
    ]
    body = "\n".join(lines)
    assert_project_result_path(CONTRACT_MD)
    CONTRACT_MD.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")
    return CONTRACT_MD


__all__ = [
    "BASELINES",
    "BASELINE_V1",
    "BASELINE_V2",
    "BASELINE_V3",
    "BASELINE_V4",
    "CONTRACT_MD",
    "CURSOR_PLAN",
    "FORBIDDEN_OLD_V2_DIR",
    "MANIFEST_NAME",
    "PARITY_ABS_NAV",
    "PARITY_MD",
    "PARITY_METRIC_ABS",
    "PLAN_COPY",
    "PROTOCOL_SOURCE_FILES",
    "RANK_NAV",
    "SNAPSHOT_FINGERPRINT",
    "SNAPSHOT_ID",
    "STRATEGY_SOURCE_FILES",
    "TOP_K",
    "TRACKER_PATH",
    "V1_DIR",
    "V2_DIR",
    "V3_DIR",
    "V4_DIR",
    "artifact_file_hashes",
    "copy_plan",
    "inspect_all",
    "inspect_baseline",
    "replay_dir",
    "source_tree_sha256",
    "write_contract_markdown",
]
