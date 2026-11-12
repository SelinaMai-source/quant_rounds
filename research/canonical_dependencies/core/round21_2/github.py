"""复查 GitHub rolling-factor-evidence SHA 7316e40。不伪造 Score。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from ..round21.contract import GITHUB_CLONE_DEST, GITHUB_REPO, GITHUB_SPARSE_DIR
from .common import dump_json, write_md
from .contract import GITHUB_PINNED_SHA, MD
from .evidence import ROLL_CSV

GH_JSON = Path(__file__).resolve().parents[2] / "result" / "round21_2" / "github" / "github.json"


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, capture_output=True)


def _ensure_clone() -> dict[str, Any]:
    dest = GITHUB_CLONE_DEST
    skill = dest / GITHUB_SPARSE_DIR / "SKILL.md"
    sha = None
    if (dest / ".git").exists():
        got = _run(["git", "rev-parse", "HEAD"], cwd=dest)
        sha = (got.stdout or "").strip() or None
    if skill.is_file() and sha == GITHUB_PINNED_SHA:
        return {"action": "reuse", "ok": True, "sha": sha, "dest": str(dest)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not (dest / ".git").exists():
        # leftover without git
        pass
    if not (dest / ".git").exists():
        clone = _run(
            ["git", "-c", "http.proxy=", "-c", "https.proxy=", "clone", "--filter=blob:none", "--sparse", "--depth", "1", GITHUB_REPO, str(dest)]
        )
        if clone.returncode != 0:
            return {"action": "clone_fail", "ok": False, "stderr": clone.stderr, "sha": None}
        _run(["git", "sparse-checkout", "set", GITHUB_SPARSE_DIR], cwd=dest)
    _run(["git", "fetch", "--depth", "1", "origin", GITHUB_PINNED_SHA], cwd=dest)
    chk = _run(["git", "checkout", GITHUB_PINNED_SHA], cwd=dest)
    got = _run(["git", "rev-parse", "HEAD"], cwd=dest)
    sha = (got.stdout or "").strip() or None
    return {"action": "checkout", "ok": sha == GITHUB_PINNED_SHA, "sha": sha, "dest": str(dest), "stderr": chk.stderr}


def run_github(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and GH_JSON.is_file() and MD["07"].is_file():
        from .common import load_json

        existing = load_json(GH_JSON)
        if existing.get("status") in {"SOURCE_CLONED_AND_READ", "REFERENCE_UNVERIFIED"}:
            return existing
    clone = _ensure_clone()
    skill = Path(clone.get("dest") or "") / GITHUB_SPARSE_DIR / "SKILL.md"
    scripts = Path(clone.get("dest") or "") / GITHUB_SPARSE_DIR / "scripts"
    payload = {
        "status": "SOURCE_CLONED_AND_READ" if clone.get("ok") else "REFERENCE_UNVERIFIED",
        "sha": clone.get("sha"),
        "pinned": GITHUB_PINNED_SHA,
        "sha_match": clone.get("sha") == GITHUB_PINNED_SHA,
        "skill_exists": skill.is_file(),
        "scripts": sorted(p.name for p in scripts.glob("*.py")) if scripts.is_dir() else [],
        "github_has_score": False,
        "exact_score_forged": False,
        "namespace_github": "GITHUB_EXACT",
        "namespace_internal": ["ROUND21_INTERNAL_W60", "ROUND21_INTERNAL_W120", "ROUND21_INTERNAL_W252"],
        "internal_rolling_csv": str(ROLL_CSV) if ROLL_CSV.is_file() else None,
        "clone": clone,
        "r21_2018h1": "EXACT_GITHUB_REPRODUCTION_PASS subset only; this round extends internal windows to new factors",
    }
    dump_json(GH_JSON, payload)
    write_md(
        MD["07"],
        "\n".join(
            [
                "# Round21.2 · 07 GitHub rolling-factor-evidence",
                "",
                f"SHA pinned=`{GITHUB_PINNED_SHA}` observed=`{clone.get('sha')}` match=`{payload['sha_match']}` status=`{payload['status']}`",
                "不伪造 GitHub Evidence Score。内部窗与 GITHUB_EXACT 分名。",
                f"scripts=`{payload['scripts']}`",
                "",
            ]
        ),
    )
    return payload
