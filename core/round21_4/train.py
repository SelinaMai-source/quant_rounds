"""A0–A9 真实 XGB WF。复用 train_both_heads 的 five_drops/ten_drops。禁止 alias ADD。

特征列表与 21.2/21.3 候选完全一致时按 hash 导入预测（21.3 既有模式），不是 alias ADD。
A2 / A4 必须新训；A8 在 A4 完成后按 hash 导入。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..paths import ROUND21_2_ROOT, ROUND21_3_ROOT, ROUND21_4_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from ..round20.train import train_both_heads, unique_keep
from ..round21.runner import oof_rankic
from ..round21_2.panels import _G, init_heads
from .common import dump_json, feature_set_hash, file_sha256, load_json, sha256_json, write_md
from .contract import (
    A_IDENTITY_ASSERTS,
    A_SPECS,
    COLSAMPLE,
    IMPORTED_BY_HASH,
    MD,
    SUBSAMPLE,
    TRAINED_REAL,
    XGB_N_JOBS,
    exp_dir,
    lineage_root,
    model_parameters,
)

MATRIX_JSON = ROUND21_4_ROOT / "configs" / "a_matrix.json"
TRAIN_JSON = ROUND21_4_ROOT / "model" / "amatrix_train.json"
OOF_CSV = ROUND21_4_ROOT / "OOF_model_results.csv"
DATA = ROUND21_4_ROOT / "data"


def _identity(spec: dict[str, Any], five_feat: Sequence[str], ten_feat: Sequence[str]) -> dict[str, Any]:
    body = {
        "code": spec["code"],
        "role": spec.get("role"),
        "added_five": list(spec.get("five_extras") or ()),
        "added_ten": list(spec.get("ten_extras") or ()),
        "dropped_five": list(spec.get("five_drops") or ()),
        "dropped_ten": list(spec.get("ten_drops") or ()),
        "head": spec.get("head"),
        "five_features": list(five_feat),
        "ten_features": list(ten_feat),
        "five_feature_set_hash": feature_set_hash(list(five_feat)),
        "ten_feature_set_hash": feature_set_hash(list(ten_feat)),
    }
    body["cache_key"] = sha256_json(body)
    return body


def _write_ident(dest, spec, five_feat, ten_feat) -> dict[str, Any]:
    ident = _identity(spec, five_feat, ten_feat)
    dump_json(dest / "feature_list.json", {"five": list(five_feat), "ten": list(ten_feat)})
    dump_json(dest / "dropped_features.json", {"five": list(spec.get("five_drops") or ()), "ten": list(spec.get("ten_drops") or ())})
    dump_json(dest / "added_features.json", {"five": list(spec.get("five_extras") or ()), "ten": list(spec.get("ten_extras") or ())})
    dump_json(dest / "cache_key.json", ident)
    dump_json(dest / "model_config.json", model_parameters(17))
    return ident


def _top12(pred: pd.DataFrame) -> pd.DataFrame:
    work = pred.copy()
    if "signal_date" in work.columns and "trade_date" not in work.columns:
        work["trade_date"] = pd.to_datetime(work["signal_date"])
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["rank"] = work.groupby("trade_date")["score"].rank(ascending=False, method="first")
    return work.loc[work["rank"].le(2), ["trade_date", "instrument_id", "score", "rank"]].sort_values(["trade_date", "rank"])


def _done(dest, ident: dict[str, Any]) -> dict[str, Any] | None:
    path = dest / "oof.json"
    key_path = dest / "cache_key.json"
    if not path.is_file() or not (dest / "five_pred.parquet").is_file() or not (dest / "ten_pred.parquet").is_file():
        return None
    if key_path.is_file() and load_json(key_path).get("cache_key") != ident.get("cache_key"):
        return None
    payload = load_json(path)
    if payload.get("status") == "OOF" or payload.get("five_oof"):
        payload["resumed"] = True
        return payload
    return None


def _copy_preds(src: Path, dest: Path) -> bool:
    if not (src / "five_pred.parquet").is_file() or not (src / "ten_pred.parquet").is_file():
        return False
    for name in ("five_pred.parquet", "ten_pred.parquet"):
        shutil.copy2(src / name, dest / name)
    return True


def _feature_match(src: Path, five_feat, ten_feat) -> bool:
    src_feat = load_json(src / "feature_list.json")
    if not src_feat:
        return False
    return list(src_feat.get("five") or []) == list(five_feat) and list(src_feat.get("ten") or []) == list(ten_feat)


def _maybe_import(spec: dict[str, Any], dest, five_feat, ten_feat, ident) -> str | None:
    """按特征列表精确匹配导入。不是 alias ADD。"""
    candidates: list[Path] = []
    lin = spec.get("lineage_import")
    if lin:
        candidates.append(lineage_root(lin[0]) / "candidates" / lin[1])
    ident_with = spec.get("identity_with")
    if ident_with:
        candidates.append(exp_dir(ident_with))
    for other in A_SPECS:
        if other["code"] == spec["code"]:
            continue
        other_dir = exp_dir(other["code"])
        if (other_dir / "five_pred.parquet").is_file():
            candidates.append(other_dir)
    seen: set[str] = set()
    for src in candidates:
        key = str(src.resolve()) if src.exists() else str(src)
        if key in seen:
            continue
        seen.add(key)
        if not src.is_dir():
            continue
        if not _feature_match(src, five_feat, ten_feat):
            continue
        if not _copy_preds(src, dest):
            continue
        dump_json(
            dest / "imported_by_hash.json",
            {
                "source": str(src),
                "reason": "identical five/ten feature lists; hash import not alias ADD",
                "five_feature_set_hash": ident.get("five_feature_set_hash"),
                "ten_feature_set_hash": ident.get("ten_feature_set_hash"),
            },
        )
        return str(src)
    return None


def _finish(spec, dest, five, ten, five_feat, ten_feat, ident, *, origin: str) -> dict[str, Any]:
    five.to_parquet(dest / "five_pred.parquet", index=False)
    ten.to_parquet(dest / "ten_pred.parquet", index=False)
    daily = pd.concat([five.assign(head="five"), ten.assign(head="ten")], ignore_index=True)
    daily.to_parquet(dest / "daily_prediction.parquet", index=False)
    top = pd.concat([_top12(five).assign(head="five"), _top12(ten).assign(head="ten")], ignore_index=True)
    top.to_parquet(dest / "daily_top1_top2.parquet", index=False)
    dump_json(
        dest / "prediction_hash.json",
        {
            "five_pred": file_sha256(dest / "five_pred.parquet"),
            "ten_pred": file_sha256(dest / "ten_pred.parquet"),
        },
    )
    dump_json(
        dest / "model_hash.json",
        {
            "cache_key": ident.get("cache_key"),
            "five_feature_set_hash": ident.get("five_feature_set_hash"),
            "ten_feature_set_hash": ident.get("ten_feature_set_hash"),
        },
    )
    init_heads()
    five_oof = oof_rankic(five, _G["five_candidate"], target=FIVE_HEAD.target)
    ten_oof = oof_rankic(ten, _G["ten_candidate"], target=TEN_HEAD.target)
    payload = {
        "code": spec["code"],
        "role": spec.get("role"),
        "status": "OOF",
        "origin": origin,
        "factor_id": spec.get("factor_id") or "r21_l1_known_float_chg",
        "formal": bool(spec.get("formal", True)),
        "five_extras": list(spec.get("five_extras") or ()),
        "ten_extras": list(spec.get("ten_extras") or ()),
        "five_drops": list(spec.get("five_drops") or ()),
        "ten_drops": list(spec.get("ten_drops") or ()),
        "five_features": list(five_feat),
        "ten_features": list(ten_feat),
        "five_feature_set_hash": feature_set_hash(list(five_feat)),
        "ten_feature_set_hash": feature_set_hash(list(ten_feat)),
        "cache_key": ident.get("cache_key"),
        "alias_from": None,
        "five_oof": five_oof,
        "ten_oof": ten_oof,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "filled_with_zero": False,
        "out_dir": str(dest),
    }
    dump_json(dest / "oof.json", payload)
    return payload


def prepare_head_panels(*, resume: bool = True) -> dict[str, Any]:
    """把 21.3 已含 L1 的头面板拷到 21.4 data（只读源，本轮落盘）。"""
    DATA.mkdir(parents=True, exist_ok=True)
    names = (
        "five_training.parquet",
        "five_candidate.parquet",
        "ten_training.parquet",
        "ten_candidate.parquet",
        "five_frozen.parquet",
        "ten_frozen.parquet",
    )
    src_roots = [ROUND21_3_ROOT / "data", ROUND21_2_ROOT / "data"]
    copied = {}
    for name in names:
        dest = DATA / name
        if resume and dest.is_file():
            copied[name] = "resumed"
            continue
        found = None
        for root in src_roots:
            sp = root / name
            if sp.is_file():
                found = sp
                break
        if found is None and name in {"five_frozen.parquet", "ten_frozen.parquet"}:
            from ..paths import ROUND20_ROOT

            sp = ROUND20_ROOT / "data" / name
            if sp.is_file():
                found = sp
        if found is None:
            raise RuntimeError(f"MISSING_ARTIFACT head panel {name}")
        shutil.copy2(found, dest)
        copied[name] = str(found)
    if not (resume and (DATA / "panels.json").is_file()):
        dump_json(DATA / "panels.json", {"status": "prepared", "copied": copied, "filled_with_zero": False})
    merge_research_factors()
    return {"status": "prepared", "copied": copied}


def merge_research_factors() -> dict[str, Any]:
    """把本轮 WXYZ panel 左连接到头面板。缺失不填 0。"""
    fp = ROUND21_4_ROOT / "data" / "factor_panel.parquet"
    if not fp.is_file():
        return {"status": "SKIP", "reason": "no factor_panel"}
    extra = pd.read_parquet(fp)
    extra["trade_date"] = pd.to_datetime(extra["trade_date"])
    extra["instrument_id"] = extra["instrument_id"].astype(str)
    new_cols = [c for c in extra.columns if c not in {"trade_date", "instrument_id"}]
    if not new_cols:
        return {"status": "SKIP", "reason": "no extra cols"}
    add = extra[["trade_date", "instrument_id", *new_cols]].drop_duplicates(["trade_date", "instrument_id"])
    touched = []
    from ..round21_2.panels import _G as G

    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate"):
        p = DATA / f"{key}.parquet"
        if not p.is_file():
            continue
        frame = pd.read_parquet(p)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        missing = [c for c in new_cols if c not in frame.columns]
        if missing:
            frame = frame.merge(add[["trade_date", "instrument_id", *missing]], on=["trade_date", "instrument_id"], how="left")
            frame.to_parquet(p, index=False)
            touched.append(key)
        if key in G:
            G[key] = frame
    return {"status": "MERGED", "n_cols": len(new_cols), "touched": touched}


def _bind_heads() -> None:
    prepare_head_panels(resume=True)
    from ..round21_2.panels import _G as G

    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate", "five_frozen", "ten_frozen"):
        frame = pd.read_parquet(DATA / f"{key}.parquet")
        if "trade_date" in frame.columns:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        if "instrument_id" in frame.columns:
            frame["instrument_id"] = frame["instrument_id"].astype(str)
        if "signal_date" in frame.columns:
            frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        G[key] = frame


def train_one(spec: dict[str, Any], *, resume: bool = True, n_jobs: int = XGB_N_JOBS) -> dict[str, Any]:
    if spec.get("alias_from"):
        raise RuntimeError(f"{spec['code']} forbids alias_from; Replace must retrain")
    dest = exp_dir(spec["code"])
    dest.mkdir(parents=True, exist_ok=True)
    five_feat = [x for x in unique_keep(list(FIVE_FEATURES) + list(spec.get("five_extras") or ())) if x not in set(spec.get("five_drops") or ())]
    ten_feat = [x for x in unique_keep(list(TEN_FEATURES) + list(spec.get("ten_extras") or ())) if x not in set(spec.get("ten_drops") or ())]
    ident = _write_ident(dest, spec, five_feat, ten_feat)
    if resume:
        existing = _done(dest, ident)
        if existing is not None:
            return existing
        src = _maybe_import(spec, dest, five_feat, ten_feat, ident)
        if src:
            five = pd.read_parquet(dest / "five_pred.parquet")
            ten = pd.read_parquet(dest / "ten_pred.parquet")
            return _finish(spec, dest, five, ten, five_feat, ten_feat, ident, origin=IMPORTED_BY_HASH)
    _bind_heads()
    init_heads()
    missing_five = [c for c in five_feat if c not in _G["five_training"].columns]
    missing_ten = [c for c in ten_feat if c not in _G["ten_training"].columns]
    if missing_five or missing_ten:
        raise RuntimeError(f"{spec['code']} missing columns five={missing_five} ten={missing_ten}")
    trained = train_both_heads(
        five_training=_G["five_training"],
        five_candidates=_G["five_candidate"],
        ten_training=_G["ten_training"],
        ten_candidates=_G["ten_candidate"],
        five_extras=list(spec.get("five_extras") or ()),
        ten_extras=list(spec.get("ten_extras") or ()),
        five_drops=list(spec.get("five_drops") or ()),
        ten_drops=list(spec.get("ten_drops") or ()),
        output_dir=dest,
        resume=resume,
        five_frozen=_G["five_frozen"],
        ten_frozen=_G["ten_frozen"],
        force_retrain=True,
        n_jobs=n_jobs,
    )
    return _finish(spec, dest, trained["five_pred"], trained["ten_pred"], five_feat, ten_feat, ident, origin=TRAINED_REAL)


def _assert_identities(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["code"]: r for r in rows}
    out = []
    ok_all = True
    for a, b in A_IDENTITY_ASSERTS:
        ra, rb = by.get(a) or {}, by.get(b) or {}
        same = (
            ra.get("five_feature_set_hash")
            and ra.get("five_feature_set_hash") == rb.get("five_feature_set_hash")
            and ra.get("ten_feature_set_hash") == rb.get("ten_feature_set_hash")
        )
        ok_all = ok_all and bool(same)
        out.append({"a": a, "b": b, "passed": bool(same), "five_a": ra.get("five_feature_set_hash"), "five_b": rb.get("five_feature_set_hash"), "ten_a": ra.get("ten_feature_set_hash"), "ten_b": rb.get("ten_feature_set_hash")})
    return {"passed": bool(ok_all), "pairs": out}


def _assert_replicate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["code"]: r for r in rows}
    checks = []
    for spec in A_SPECS:
        if not spec.get("replicate_of"):
            continue
        row = by.get(spec["code"]) or {}
        src = lineage_root("round21_2" if "21_2" in spec["replicate_of"] else "round21_3") / "candidates" / spec["replicate_of"]
        src_feat = load_json(src / "feature_list.json")
        five_ok = list(row.get("five_features") or []) == list(src_feat.get("five") or [])
        ten_ok = list(row.get("ten_features") or []) == list(src_feat.get("ten") or [])
        checks.append(
            {
                "code": spec["code"],
                "source": spec["replicate_of"],
                "five_ok": five_ok,
                "ten_ok": ten_ok,
                "passed": bool(five_ok and ten_ok),
                "five_features": row.get("five_features"),
                "src_five": src_feat.get("five"),
                "ten_features": row.get("ten_features"),
                "src_ten": src_feat.get("ten"),
            }
        )
    return {"passed": all(c["passed"] for c in checks) if checks else False, "checks": checks}


def freeze_a_matrix() -> dict[str, Any]:
    payload = {
        "status": "FROZEN",
        "n_specs": len(A_SPECS),
        "specs": list(A_SPECS),
        "no_alias_add_as_replace": True,
        "fusion": "0.90 ten + 0.10 five",
        "five_only_drop": "g0_amount_log",
        "note": "A6 Ten g0 drop is NO-OP because g0_amount_log not in TEN_FEATURES",
    }
    dump_json(MATRIX_JSON, payload)
    return payload


def run_amatrix_train(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    if resume and TRAIN_JSON.is_file():
        existing = load_json(TRAIN_JSON)
        if existing.get("status") == "DONE" and int(existing.get("n_oof_models") or 0) >= len(A_SPECS):
            return existing
    _bind_heads()
    init_heads()
    freeze_a_matrix()
    xgb_jobs = 1 if int(jobs) > 1 else XGB_N_JOBS
    # 先跑可导入的，再跑必须新训的 A2/A4，最后 A8（≡A4）
    order = list(A_SPECS)
    need_train_last = {"R21_4_A2", "R21_4_A4", "R21_4_A8"}
    first = [s for s in order if s["code"] not in need_train_last]
    mid = [s for s in order if s["code"] in {"R21_4_A2", "R21_4_A4"}]
    last = [s for s in order if s["code"] == "R21_4_A8"]
    rows: list[dict[str, Any]] = []

    def _run(spec: dict[str, Any]) -> dict[str, Any]:
        print(f"[round21.4] TRAIN {spec['code']} role={spec.get('role')} 5+={spec.get('five_extras')} 10+={spec.get('ten_extras')} 5-={spec.get('five_drops')} 10-={spec.get('ten_drops')}", flush=True)
        return train_one(spec, resume=resume, n_jobs=xgb_jobs)

    for spec in first:
        rows.append(_run(spec))
    # A2/A4 共享 _G 面板，禁止线程并行训练
    for spec in mid:
        rows.append(_run(spec))
    for spec in last:
        rows.append(_run(spec))
    by_code = {r["code"]: r for r in rows}
    ordered = [by_code[s["code"]] for s in A_SPECS if s["code"] in by_code]
    identity = _assert_identities(ordered)
    replicate = _assert_replicate(ordered)
    if not identity.get("passed"):
        raise RuntimeError(f"A identity assertion failed: {identity}")
    if not replicate.get("passed"):
        raise RuntimeError(f"A3/A6 replication assertion failed: {replicate}")
    OOF_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "code": r.get("code"),
                "role": r.get("role"),
                "origin": r.get("origin"),
                "five_oof": (r.get("five_oof") or {}).get("oof_rankic_mean"),
                "ten_oof": (r.get("ten_oof") or {}).get("oof_rankic_mean"),
                "five_drops": r.get("five_drops"),
                "ten_drops": r.get("ten_drops"),
                "five_feature_set_hash": r.get("five_feature_set_hash"),
                "ten_feature_set_hash": r.get("ten_feature_set_hash"),
                "cache_key": r.get("cache_key"),
            }
            for r in ordered
        ]
    ).to_csv(OOF_CSV, index=False)
    payload = {
        "status": "DONE",
        "n_oof_models": len(ordered),
        "identity": identity,
        "replicate": replicate,
        "no_alias_add": True,
        "rows": ordered,
        "a3_ok": any(c.get("code") == "R21_4_A3" and c.get("passed") for c in replicate.get("checks") or []),
        "a6_ok": any(c.get("code") == "R21_4_A6" and c.get("passed") for c in replicate.get("checks") or []),
    }
    dump_json(TRAIN_JSON, payload)
    write_md(
        MD["06"],
        "# Round21.4 · 06 L1 A 矩阵训练\n\n"
        f"N=`{len(ordered)}` 恒等=`{identity.get('passed')}` 复现=`{replicate.get('passed')}` 禁止 alias ADD。\n"
        f"A3 21.2=`{payload['a3_ok']}` A6 21.3=`{payload['a6_ok']}`\n",
    )
    return payload
