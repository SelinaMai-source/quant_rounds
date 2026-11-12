"""GitHub rolling-factor-evidence 固定 SHA 7316e40。不编造 Score。"""

from __future__ import annotations

import subprocess
from typing import Any

from ..paths import ROUND21_3_ROOT
from ..round21.contract import GITHUB_CLONE_DEST, GITHUB_REPO, GITHUB_SPARSE_DIR
from .common import dump_json, write_md
from .contract import GITHUB_PINNED_SHA, MD

GH_JSON = ROUND21_3_ROOT / "github" / "github.json"


def run_github(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and GH_JSON.is_file() and MD["07"].is_file():
        from .common import load_json

        existing = load_json(GH_JSON)
        if existing.get("status") in {"DONE", "SOURCE_CLONED_AND_READ"}:
            return existing
    dest = GITHUB_CLONE_DEST
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha = None
    status = "REFERENCE_UNVERIFIED"
    try:
        if not (dest / ".git").exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", GITHUB_REPO, str(dest)], check=True, timeout=180)
            subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", GITHUB_SPARSE_DIR], check=True, timeout=60)
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", GITHUB_PINNED_SHA], check=False, timeout=180)
        subprocess.run(["git", "-C", str(dest), "checkout", GITHUB_PINNED_SHA], check=True, timeout=60)
        sha = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        status = "SOURCE_CLONED_AND_READ" if sha.startswith("7316e40") else "REFERENCE_UNVERIFIED"
    except Exception as exc:
        payload = {"status": "BLOCKED", "reason": str(exc), "pinned": GITHUB_PINNED_SHA, "invented_score": False}
        dump_json(GH_JSON, payload)
        write_md(MD["07"], f"# Round21.3 · 07 GitHub\n\nBLOCKED `{exc}` pinned=`{GITHUB_PINNED_SHA}` 不编造 Score。\n")
        return payload
    payload = {
        "status": "DONE",
        "clone_status": status,
        "sha": sha,
        "pinned": GITHUB_PINNED_SHA,
        "dest": str(dest),
        "invented_score": False,
        "note": "rolling-factor-evidence 只作边际参考，不发明 Score",
    }
    dump_json(GH_JSON, payload)
    write_md(MD["07"], f"# Round21.3 · 07 GitHub\n\nSHA=`{sha}` pinned=`{GITHUB_PINNED_SHA}` status=`{status}` 不编造 Score。\n")
    return payload
