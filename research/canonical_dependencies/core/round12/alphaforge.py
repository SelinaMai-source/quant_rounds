"""AlphaForge：只对指定仓库隔离重试一次。失败不阻断。"""

from __future__ import annotations

from typing import Any

from ..paths import PROJECT_ROOT, ROUND12_ROOT
from ..round11.alphaforge import _ls_remote, _read_pbo_definition, _run, REPO, VENDOR, VENV
from .common import dump_json, write_md

AF_MD = ROUND12_ROOT / "13_alphaforge.md"
AF_JSON = ROUND12_ROOT / "alphaforge" / "alphaforge.json"


def run_alphaforge(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = ROUND12_ROOT / "alphaforge"
    dest.mkdir(parents=True, exist_ok=True)
    if resume and AF_JSON.is_file():
        from .common import load_json

        existing = load_json(AF_JSON)
        if existing.get("status"):
            return existing
    remote = _ls_remote()
    payload: dict[str, Any] = {
        "repo": REPO,
        "ls_remote": remote,
        "retry": "once",
        "blocks_pipeline": False,
        "post_hoc_diagnostic": True,
    }
    if not remote.get("ok"):
        payload.update({
            "status": "ALPHAFORGE_REPO_ACCESS_FAIL_ROUND12",
            "used": "native_stats_only",
            "math_label": "LOCAL_EQUIVALENT_NOT_ALPHA_FORGE",
        })
        dump_json(AF_JSON, payload)
        write_md(AF_MD, "# Round 12 · AlphaForge\n\n**ALPHAFORGE_REPO_ACCESS_FAIL_ROUND12**。只重试一次。native 统计继续。不阻断。\n")
        return payload
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    if not (VENDOR / ".git").is_dir():
        clone = _run(["git", "clone", "--depth", "1", REPO, str(VENDOR)], timeout=180)
        payload["clone"] = {"returncode": clone.returncode, "stderr": (clone.stderr or "")[:400]}
        if clone.returncode != 0:
            payload["status"] = "ALPHAFORGE_REPO_ACCESS_FAIL_ROUND12"
            payload["math_label"] = "LOCAL_EQUIVALENT_NOT_ALPHA_FORGE"
            dump_json(AF_JSON, payload)
            write_md(AF_MD, "# Round 12 · AlphaForge\n\nclone 失败。native 继续。不阻断。\n")
            return payload
    rev = _run(["git", "rev-parse", "HEAD"], cwd=VENDOR)
    payload["commit"] = (rev.stdout or "").strip()
    definition = _read_pbo_definition(VENDOR)
    payload["definition"] = {
        "classification": definition.get("classification"),
        "pbo_files": [
            {k: v for k, v in item.items() if k != "snippet"} | {"snippet_head": (item.get("snippet") or "")[:600]}
            for item in definition.get("pbo_files") or []
        ],
    }
    payload["status"] = "CLONED_AND_READ"
    payload["pbo_name"] = definition.get("classification")
    payload["stats_only"] = True
    payload["backtest_engine_used"] = False
    dump_json(AF_JSON, payload)
    write_md(
        AF_MD,
        "\n".join([
            "# Round 12 · AlphaForge",
            "",
            f"仓库 `{REPO}` commit=`{payload.get('commit')}`。只重试一次。",
            f"PBO 命名：**{payload.get('pbo_name')}**。只用统计模块对照 PSR/DSR/PBO/CPCV/SPA。",
            "",
        ]),
    )
    return payload


def load_alphaforge() -> dict[str, Any]:
    from .common import load_json

    return load_json(AF_JSON)


__all__ = ["load_alphaforge", "run_alphaforge"]
