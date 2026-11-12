"""AlphaForge：先 ls-remote 指定仓库，读 SKILL/源码再命名 PBO。隔离 venv。只用统计模块。"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, ROUND11_ROOT
from .common import dump_json, stage_dir, write_md

AF_MD = ROUND11_ROOT / "09_alphaforge.md"
REPO = "https://github.com/xiejinglover/alpha-forge"
VENDOR = PROJECT_ROOT / "tools" / "vendor" / "alpha-forge"
VENV = PROJECT_ROOT / "tools" / "vendor" / "alpha-forge-venv"


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}:
            env.pop(key, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_clean_env(),
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, (exc.stdout or "") if isinstance(exc.stdout, str) else "", f"TIMEOUT {timeout}s")


def _ls_remote() -> dict[str, Any]:
    proc = _run(["git", "ls-remote", REPO, "HEAD"], timeout=25)
    ok = proc.returncode == 0 and bool(proc.stdout.strip())
    sha = proc.stdout.split()[0] if ok else None
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[:500],
        "stderr": (proc.stderr or "")[:500],
        "head": sha,
    }


def _read_pbo_definition(root: Path) -> dict[str, Any]:
    hits = []
    for rel in ("SKILL.md", "README.md", "README.rst"):
        path = root / rel
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            hits.append({"file": rel, "n_chars": len(text), "head": text[:4000]})
    pbo_files = []
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "def pbo" in text or "def PBO" in text or "probability of backtest" in text.lower():
            pbo_files.append({
                "file": str(path.relative_to(root)),
                "snippet": "\n".join(
                    line for line in text.splitlines()
                    if "pbo" in line.lower() or "logit" in line.lower() or "cscv" in line.lower()
                )[:4000],
                "has_logit": "logit" in text.lower(),
                "has_cscv": "cscv" in text.lower() or "combinatorial" in text.lower(),
                "has_is_oos": "in_sample" in text.lower() or "is_sharpe" in text.lower() or "OOS" in text,
            })
    classification = "UNREAD"
    if pbo_files:
        sel = any(x.get("has_logit") and (x.get("has_cscv") or x.get("has_is_oos")) for x in pbo_files)
        if sel:
            classification = "TRUE_SELECTION_BASED_PBO"
        else:
            classification = "PROBABILITY_OF_LOSS_PROXY"
    return {"docs": hits, "pbo_files": pbo_files, "classification": classification}


def run_alphaforge(**_: Any) -> dict[str, Any]:
    dest = stage_dir("alphaforge")
    remote = _ls_remote()
    payload: dict[str, Any] = {
        "repo": REPO,
        "ls_remote": remote,
        "post_hoc_diagnostic": True,
    }
    if not remote.get("ok"):
        payload.update({
            "status": "ALPHAFORGE_REPO_ACCESS_FAIL",
            "used": "native_stats_only",
            "math_label": "LOCAL_EQUIVALENT_NOT_ALPHA_FORGE",
        })
        dump_json(dest / "alphaforge.json", payload)
        write_md(AF_MD, "# Round 11 · AlphaForge\n\n**ALPHAFORGE_REPO_ACCESS_FAIL**。native 统计继续，标 `LOCAL_EQUIVALENT_NOT_ALPHA_FORGE`。\n")
        return payload

    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    if not (VENDOR / ".git").is_dir():
        clone = _run(["git", "clone", "--depth", "1", REPO, str(VENDOR)], timeout=180)
        payload["clone"] = {"returncode": clone.returncode, "stderr": (clone.stderr or "")[:400]}
        if clone.returncode != 0:
            payload["status"] = "ALPHAFORGE_REPO_ACCESS_FAIL"
            payload["math_label"] = "LOCAL_EQUIVALENT_NOT_ALPHA_FORGE"
            dump_json(dest / "alphaforge.json", payload)
            write_md(AF_MD, "# Round 11 · AlphaForge\n\nclone 失败。native 继续。\n")
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
        "docs": [{"file": d["file"], "n_chars": d["n_chars"]} for d in definition.get("docs") or []],
    }
    # isolated venv — no global pip upgrade
    if not (VENV / "bin" / "python").is_file():
        created = _run(["python3", "-m", "venv", str(VENV)], timeout=60)
        payload["venv_create"] = created.returncode
    pip = VENV / "bin" / "pip"
    py = VENV / "bin" / "python"
    req = None
    for name in ("requirements.txt", "pyproject.toml"):
        if (VENDOR / name).is_file():
            req = name
            break
    payload["venv_python"] = str(py)
    payload["requirements"] = req
    payload["status"] = "CLONED_AND_READ"
    payload["pbo_name"] = definition.get("classification")
    payload["stats_only"] = True
    payload["backtest_engine_used"] = False
    dump_json(dest / "alphaforge.json", payload)
    dump_json(dest / "pbo_source_read.json", definition)
    write_md(
        AF_MD,
        "\n".join([
            "# Round 11 · AlphaForge",
            "",
            f"仓库 `{REPO}` commit=`{payload.get('commit')}`。",
            f"先读源码后的 PBO 命名：**{payload.get('pbo_name')}**。",
            "",
            "只用统计模块，不用它的回测引擎。隔离 venv，不 `pip upgrade` 全局包。",
            "",
            "若 `pbo()` 只是 test-fold 亏损概率 → 必须写 `PROBABILITY_OF_LOSS_PROXY`。",
            "多策略 IS 选胜再看 OOS rank logit → 才写 `TRUE_SELECTION_BASED_PBO`。",
            "",
        ]),
    )
    return payload


def load_alphaforge() -> dict[str, Any]:
    path = ROUND11_ROOT / "alphaforge" / "alphaforge.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


__all__ = ["load_alphaforge", "run_alphaforge"]
