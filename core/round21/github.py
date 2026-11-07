"""sparse clone xiejinglover/alpha-forge rolling-factor-evidence。以源码为准，不凭目录名补算法。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from ..paths import ROUND21_ROOT
from .common import dump_json, write_md
from .contract import (
    EXACT_GITHUB_REPRODUCTION_PASS,
    GITHUB_CLONE_DEST,
    GITHUB_EXACT,
    GITHUB_MD,
    GITHUB_METHOD_STATUS_CLONED,
    GITHUB_METHOD_STATUS_UNVERIFIED,
    GITHUB_REPO,
    GITHUB_SPARSE_DIR,
    ROUND21_INTERNAL,
)

GH_JSON = ROUND21_ROOT / "github" / "github.json"
MAP_MD = ROUND21_ROOT / "github" / "GITHUB_METHOD_MAPPING.md"
REP_MD = ROUND21_ROOT / "github" / "GITHUB_METHOD_REPRODUCTION.md"
DIFF_MD = ROUND21_ROOT / "github" / "GITHUB_METHOD_DIFFERENCES.md"


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, capture_output=True)


def _no_proxy_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    return env


def _existing_sha(dest: Path) -> str | None:
    git_dir = dest / ".git"
    if not git_dir.exists():
        return None
    got = _run(["git", "rev-parse", "HEAD"], cwd=dest)
    if got.returncode != 0:
        return None
    return got.stdout.strip() or None


def _sparse_clone(dest: Path) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / GITHUB_SPARSE_DIR / "SKILL.md").is_file():
        sha = _existing_sha(dest)
        return {
            "action": "reuse",
            "ok": True,
            "sha": sha,
            "dest": str(dest),
            "stderr": "",
        }
    if dest.exists() and any(dest.iterdir()):
        # incomplete leftover
        pass
    env = _no_proxy_env()
    clone = _run(
        [
            "git",
            "-c",
            "http.proxy=",
            "-c",
            "https.proxy=",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "--depth",
            "1",
            GITHUB_REPO,
            str(dest),
        ],
        env=env,
    )
    if clone.returncode != 0:
        return {
            "action": "clone",
            "ok": False,
            "sha": None,
            "dest": str(dest),
            "stderr": (clone.stderr or clone.stdout or "")[-2000:],
        }
    sparse = _run(
        ["git", "-c", "http.proxy=", "-c", "https.proxy=", "sparse-checkout", "set", GITHUB_SPARSE_DIR],
        cwd=dest,
        env=env,
    )
    sha = _existing_sha(dest)
    return {
        "action": "clone",
        "ok": sparse.returncode == 0 and bool(sha) and (dest / GITHUB_SPARSE_DIR / "SKILL.md").is_file(),
        "sha": sha,
        "dest": str(dest),
        "stderr": (sparse.stderr or "")[-2000:],
    }


def _list_source(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            out.append(str(path.relative_to(root)))
    return out


def _read_head(path: Path, n: int = 40) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[:n])


def run_github(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and GH_JSON.is_file():
        existing = __import__("json").loads(GH_JSON.read_text(encoding="utf-8"))
        if existing.get("status") in {GITHUB_METHOD_STATUS_CLONED, GITHUB_METHOD_STATUS_UNVERIFIED}:
            return existing

    clone = _sparse_clone(GITHUB_CLONE_DEST)
    skill_root = GITHUB_CLONE_DEST / GITHUB_SPARSE_DIR
    files = _list_source(skill_root)
    cloned = bool(clone.get("ok")) and bool(clone.get("sha")) and bool(files)
    # 读到源码 ≠ 在一进二标签上跑通他们的执行器。
    status = GITHUB_METHOD_STATUS_CLONED if cloned else GITHUB_METHOD_STATUS_UNVERIFIED
    exact_pass = False
    sha = clone.get("sha")
    entries = {
        "skill": str(skill_root / "SKILL.md"),
        "build_plan": str(skill_root / "scripts" / "build_factor_plan.py"),
        "run_evidence": str(skill_root / "scripts" / "run_factor_evidence.py"),
        "core": str(skill_root / "scripts" / "factor_evidence_core.py"),
        "study_contract": str(skill_root / "references" / "study-contract.md"),
        "statistical_method": str(skill_root / "references" / "statistical-method.md"),
        "adapter_guide": str(skill_root / "references" / "adapter-guide.md"),
    }
    present = {k: Path(v).is_file() for k, v in entries.items()}

    payload = {
        "status": status,
        "GITHUB_METHOD_STATUS": status,
        "exact_github_reproduction_pass": exact_pass,
        "claimed_exact_pass": EXACT_GITHUB_REPRODUCTION_PASS if exact_pass else None,
        "repo": GITHUB_REPO,
        "sparse_dir": GITHUB_SPARSE_DIR,
        "dest": str(GITHUB_CLONE_DEST),
        "commit_sha": sha,
        "clone": clone,
        "files": files,
        "n_files": len(files),
        "entries_present": present,
        "github_has_score": False,
        "github_window_model": "deployment.windows[window_id] calendar start → information_cutoff; NOT 60/120/252",
        "internal_namespace": ROUND21_INTERNAL,
        "github_namespace": GITHUB_EXACT,
        "scope": "边际预测证据 only：同周期 RankIC、分组 spread、HAC/BH、滚动健康。不做去冗余/条件增量/调参/组合/账户回测。",
        "note": "未在本 todo 跑 GitHub 执行器；禁止写 EXACT_GITHUB_REPRODUCTION_PASS。",
    }
    dump_json(GH_JSON, payload)

    map_lines = [
        "# GITHUB_METHOD_MAPPING",
        "",
        f"仓库 `{GITHUB_REPO}` 目录 `{GITHUB_SPARSE_DIR}` commit=`{sha}` status=`{status}`。",
        "",
        "以 clone 源码为准（SKILL.md / scripts / references），不凭目录名补算法。",
        "",
        "## 入口（源码实读）",
        "",
        "- `scripts/build_factor_plan.py`：只用候选池结构、因子覆盖、时间戳和元数据；**不读 label_value** 决定阈值/组数。",
        "- `scripts/run_factor_evidence.py`：校验冻结输入哈希、PIT、标签路径后算 health / daily RankIC+spread / HAC+BH / 经济幅度 / 稳定性 / 暴露。",
        "- 标准输入：`observations.csv` `factor_values.csv` `factor_metadata.csv` `label_metadata.csv` 可选 `exposures.csv`。",
        "",
        "## 一进二映射（预登记，本 todo 不执行）",
        "",
        f"- 命名空间 `{GITHUB_EXACT}` vs `{ROUND21_INTERNAL}` 必须分名。",
        "- 标签：`label_e2_net` 与 `label_exit_rule_net`（maturity-aware，按 `actual_exit_date`）各成 `label_id`。",
        "- GitHub 窗口 = `deployment.windows[window_id]` 日历起点到 `information_cutoff`，**不是** 60/120/252。",
        "- 内部滚动窗 = `ROUND21_INTERNAL_W60/W120/W252`，禁止事后挑窗。",
        "- GitHub 结论字段是 `economic_eligible` / `model_candidate` / `window_economic_eligible`。源码**没有 Score**，不伪造。",
        "- 内部 Evidence 是向量/分层硬过滤+Pareto，不优化人为加权总分。",
        "",
        "## 源文件",
        "",
    ]
    for rel in files:
        map_lines.append(f"- `{rel}`")
    map_lines.append("")
    write_md(MAP_MD, "\n".join(map_lines))

    rep_lines = [
        "# GITHUB_METHOD_REPRODUCTION",
        "",
        f"status=`{status}` commit=`{sha}`",
        "",
        "本 todo **没有**把一进二五/十标签跑进 `run_factor_evidence.py`。",
        "因此 **不得** 写 `EXACT_GITHUB_REPRODUCTION_PASS`。",
        "",
        "已核：",
        "",
        f"- clone dest=`{GITHUB_CLONE_DEST}` action=`{clone.get('action')}` ok=`{clone.get('ok')}`",
        f"- SKILL/scripts/references present=`{present}`",
        "- 适配器与小样本五/十标签映射留待 evidence stage。",
        "",
    ]
    if not cloned:
        rep_lines += [
            f"clone 失败。GITHUB_METHOD_STATUS=`{GITHUB_METHOD_STATUS_UNVERIFIED}`。",
            "继续内部流程。stderr:",
            "",
            "```",
            str(clone.get("stderr") or ""),
            "```",
            "",
        ]
    write_md(REP_MD, "\n".join(rep_lines))

    diff_lines = [
        "# GITHUB_METHOD_DIFFERENCES",
        "",
        "源码边界（SKILL.md）：只做候选池内 PIT 因子的边际预测证据。",
        "不做相关去冗余、模型条件增量、调参、路线组合、正式账户回测。",
        "",
        "| 项 | GitHub-exact | ROUND21_INTERNAL |",
        "|---|---|---|",
        "| 窗口 | deployment.windows 日历 | 60/120/252 交易日滚动 |",
        "| 主指标 | recipe.primary_metric（默认 rank_ic 或 spread） | 双标签 IC/RankIC/HAC/块 bootstrap |",
        "| 结论 | economic_eligible / model_candidate | 向量硬过滤 + Pareto，无总分 |",
        "| Score | **无，不伪造** | **无，不伪造** |",
        "| 组数 | 计划阶段按池结构冻 5→3 | 内部 shape Q1–Q5，不强制 Q10 |",
        "| FDR | BH family = deployment×recipe×pool×label×主指标 | 内部另计，不与 GitHub q 混名 |",
        "",
        "一进二正式 NAV 仍走 Stock `round7_fixed_sleeve_config`，GitHub 产物不得替换。",
        "",
    ]
    write_md(DIFF_MD, "\n".join(diff_lines))
    write_md(
        GITHUB_MD,
        "\n".join(
            [
                "# Round21 · GitHub rolling-factor-evidence",
                "",
                f"status=`{status}` SHA=`{sha}` exact_pass=`{exact_pass}`",
                "",
                f"详见 `{MAP_MD.name}` / `{REP_MD.name}` / `{DIFF_MD.name}`。",
                "",
            ]
        ),
    )
    payload["map_md"] = str(MAP_MD)
    payload["rep_md"] = str(REP_MD)
    payload["diff_md"] = str(DIFF_MD)
    dump_json(GH_JSON, payload)
    return payload
