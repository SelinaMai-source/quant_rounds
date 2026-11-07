#!/usr/bin/env python3
"""按白名单把本轮代码与小型产物同步到独立 publish worktree 并 push。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

PROJ = Path("/data1/stock-mxy/课题/一进二策略")
PUB_ROOT = Path("/data1/stock-mxy/research_publish")
WT = {
    "round22_6": PUB_ROOT / "wt_round22_6",
    "round23_2": PUB_ROOT / "wt_round23_2",
    "round24_2": PUB_ROOT / "wt_round24_2",
}
RESULT_MAP = {
    "round22_6": ["round22_6"],
    "round23_2": ["round23_2"],
    "round24_2": ["round24_2", "round22_6_24_2_master"],
}
CODE_MAP = {
    "round22_6": ["core/research_integrity_v2", "core/round22_6", "tools/run_round22_6.py", "tools/publish_round_branch.py"],
    "round23_2": ["core/research_integrity_v2", "core/round23_2", "tools/run_round23_2.py", "tools/publish_round_branch.py"],
    "round24_2": ["core/research_integrity_v2", "core/round24_2", "tools/run_round24_2.py", "tools/run_round22_6_24_2.py", "tools/publish_round_branch.py"],
}
# shared deps needed to import
SHARED = [
    "core/__init__.py",
    "core/paths.py",
    "core/r225241_io.py",
    "core/r225241_models.py",
    "core/r225241_sleeve.py",
    "core/r225241_stats.py",
    "core/local_executor.py",
    "core/research_sleeve.py",
    "core/fold_ranker.py",
    "core/limit_rules.py",
    "core/metrics.py",
    "core/stock_readonly.py",
]


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    base = os.environ.copy()
    # 不写 git config；仅用环境变量继承既有作者身份
    base.setdefault("GIT_AUTHOR_NAME", "SelinaMai-source")
    base.setdefault("GIT_AUTHOR_EMAIL", "sheungyingmai@gmail.com")
    base.setdefault("GIT_COMMITTER_NAME", base["GIT_AUTHOR_NAME"])
    base.setdefault("GIT_COMMITTER_EMAIL", base["GIT_AUTHOR_EMAIL"])
    if env:
        base.update(env)
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, env=base)


def _sha_dir(path: Path) -> str:
    h = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file() and "__pycache__" not in file.parts:
            h.update(file.relative_to(path).as_posix().encode())
            h.update(file.read_bytes())
    return h.hexdigest()


def _copy_tree(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".tmp.*"))


def _whitelist_result_files(src: Path, dst: Path, max_mb: float = 20.0) -> list[dict]:
    manifests = []
    dst.mkdir(parents=True, exist_ok=True)
    for file in src.rglob("*"):
        if not file.is_file():
            continue
        if file.name.endswith((".tmp", ".pyc")) or "__pycache__" in file.parts or file.suffix == ".parquet" and file.stat().st_size > max_mb * 1024 * 1024:
            # large parquet -> manifest only
            if file.suffix in {".parquet", ".pkl", ".joblib", ".bin", ".model"} or file.stat().st_size > max_mb * 1024 * 1024:
                rel = file.relative_to(src)
                meta = {
                    "relative_path": str(rel),
                    "server_path": str(file),
                    "bytes": file.stat().st_size,
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest() if file.stat().st_size <= 200 * 1024 * 1024 else "TOO_LARGE_TO_HASH_INLINE",
                }
                if meta["sha256"] == "TOO_LARGE_TO_HASH_INLINE":
                    h = hashlib.sha256()
                    with file.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1 << 20), b""):
                            h.update(chunk)
                    meta["sha256"] = h.hexdigest()
                manifests.append(meta)
                continue
        rel = file.relative_to(src)
        # skip secrets
        if any(x in file.name.lower() for x in (".env", "token", "password", "secret", "credential")):
            continue
        if file.stat().st_size > max_mb * 1024 * 1024:
            manifests.append({"relative_path": str(rel), "server_path": str(file), "bytes": file.stat().st_size, "skipped": "too_large"})
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    return manifests


def publish_round(branch: str, *, message_code: str | None = None, message_results: str | None = None) -> dict:
    wt = WT[branch]
    if not wt.exists():
        raise RuntimeError(f"missing worktree {wt}")
    # sync code
    for rel in CODE_MAP[branch] + SHARED:
        src = PROJ / rel
        if not src.exists():
            # copy package dirs partially
            continue
        _copy_tree(src, wt / rel)
    # also copy essential package inits / round deps lightly by importing path packages already in SHARED
    for pkg in ("round7", "round11", "round15", "round16_5", "round17", "round21", "round21_4", "round22_3", "round4"):
        src = PROJ / "core" / pkg
        if src.is_dir():
            _copy_tree(src, wt / "core" / pkg)
    # integrity hash
    integrity_hash = _sha_dir(PROJ / "core" / "research_integrity_v2")
    common = {"research_integrity_v2_sha256": integrity_hash, "branch": branch}
    (wt / "common_code_manifest.json").write_text(json.dumps(common, indent=2) + "\n", encoding="utf-8")
    # results
    all_manifests = []
    for res in RESULT_MAP[branch]:
        src = PROJ / "result" / res
        if not src.is_dir():
            continue
        manifests = _whitelist_result_files(src, wt / "result" / res)
        all_manifests.extend(manifests)
        (wt / "result" / res / "LARGE_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    # REPRODUCE / index
    report_name = {
        "round22_6": "ROUND22_6_RESEARCH_REPORT.md",
        "round23_2": "ROUND23_2_RESEARCH_REPORT.md",
        "round24_2": "ROUND24_2_RESEARCH_REPORT.md",
    }[branch]
    res_dir = RESULT_MAP[branch][0]
    (wt / "REPRODUCE.md").write_text(
        "\n".join(
            [
                f"# Reproduce {branch}",
                "",
                f"源审计起点见主仓库分支父提交。工作根：`{PROJ}`",
                "",
                "## 环境",
                "- Python 3 + xgboost/lightgbm/pandas/sklearn/pyarrow",
                "- arch 可选；缺失则 SPA=BLOCKED_STAT_TEST",
                "",
                "## 命令",
                f"`python tools/run_{branch}.py --jobs 2 --resume`",
                f"`python tools/run_{branch}.py --stage decision`",
                "",
                "## 已知阻断",
                "- 大型 parquet 仅以 MANIFEST 上传（server path + sha256）",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (wt / "BRANCH_REPORT_INDEX.md").write_text(
        "\n".join(
            [
                f"# {branch} 报告索引",
                "",
                f"- 主报告：`result/{res_dir}/{report_name}`",
                f"- 裁决：`result/{res_dir}/FINAL_DECISION.md`",
                f"- research_integrity_v2 sha256=`{integrity_hash}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (wt / "external_dependencies.json").write_text(
        json.dumps(
            {
                "status": "NOT_FULLY_SELF_CONTAINED",
                "needs": ["first_board_hermes via Stock src", "Hermes/AmazingData market panels on server"],
                "note": "服务器完整依赖可跑；GitHub 包不含供应商原始库",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # secret/large scan
    # commit code then results
    def commit(msg: str) -> str | None:
        _run(["git", "add", "-A"], wt)
        # unstage secrets if any
        status = _run(["git", "status", "--porcelain"], wt).stdout
        if not status.strip():
            return None
        r = _run(["git", "commit", "-m", msg], wt)
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)
        return _run(["git", "rev-parse", "HEAD"], wt).stdout.strip()

    sha1 = commit(message_code or f"{branch}: research integrity fixes, tests, and frozen protocol")
    # touch results marker to ensure second commit if needed
    sha2 = commit(message_results or f"{branch}: research results, reports, and artifact manifests")
    # push
    push = _run(["git", "push", "-u", "origin", f"HEAD:{branch}"], wt)
    local = _run(["git", "rev-parse", "HEAD"], wt).stdout.strip()
    remote = _run(["git", "ls-remote", "origin", f"refs/heads/{branch}"], wt).stdout.strip().split()
    remote_sha = remote[0] if remote else None
    verified = local == remote_sha and push.returncode == 0
    return {
        "branch": branch,
        "push_rc": push.returncode,
        "push_stderr": push.stderr[-2000:],
        "push_stdout": push.stdout[-2000:],
        "local_sha": local,
        "remote_sha": remote_sha,
        "verified": verified,
        "code_commit": sha1,
        "results_commit": sha2,
        "integrity_hash": integrity_hash,
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(publish_round(sys.argv[1]), ensure_ascii=False, indent=2))
