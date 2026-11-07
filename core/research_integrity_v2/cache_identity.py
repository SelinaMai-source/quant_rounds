"""缓存指纹绑定完整实验身份。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_json(obj: Any) -> str:
    return _sha_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))


def file_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ExperimentIdentity:
    experiment_id: str
    code_hash: str
    input_data_hash: str
    label_definition_hash: str
    label_maturity_hash: str
    feature_order_hash: str
    train_test_fold_hash: str
    model_parameters_hash: str
    seed: int
    execution_contract_hash: str
    cost_contract_hash: str
    extras: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        body = asdict(self)
        return _sha_json(body)


def fingerprint(identity: ExperimentIdentity | dict[str, Any]) -> str:
    if isinstance(identity, ExperimentIdentity):
        return identity.fingerprint()
    required = [
        "code_hash",
        "input_data_hash",
        "label_definition_hash",
        "label_maturity_hash",
        "feature_order_hash",
        "train_test_fold_hash",
        "model_parameters_hash",
        "seed",
        "execution_contract_hash",
        "cost_contract_hash",
    ]
    missing = [k for k in required if k not in identity]
    if missing:
        raise ValueError(f"incomplete cache identity: missing {missing}")
    return _sha_json(identity)


def cache_valid(dest: Path, identity: ExperimentIdentity | dict[str, Any], *, require_files: tuple[str, ...] = ("summary.json",)) -> bool:
    dest = Path(dest)
    meta_path = dest / "cache_identity.json"
    if not meta_path.is_file():
        return False
    for name in require_files:
        if not (dest / name).is_file():
            return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("fingerprint") == fingerprint(identity)


def write_identity(dest: Path, identity: ExperimentIdentity | dict[str, Any]) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    payload = asdict(identity) if isinstance(identity, ExperimentIdentity) else dict(identity)
    payload["fingerprint"] = fingerprint(identity)
    path = dest / "cache_identity.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
