"""Survivorship：AmazingData hist status 若可下则重建；否则 UNQUANTIFIED，不标 PASS。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND15_ROOT
from .common import dump_json, load_json, redact, write_md
from .contract import UNQUANTIFIED
from .freeze import require_frozen
from .sdk import load_sdk

SURV_JSON = ROUND15_ROOT / "survivorship" / "survivorship.json"
SURV_MD = ROUND15_ROOT / "09_survivorship.md"


def run_survivorship(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and SURV_JSON.is_file():
        existing = load_json(SURV_JSON)
        if existing.get("status") in {UNQUANTIFIED, "DONE"}:
            return existing
    sdk = load_sdk()
    status = UNQUANTIFIED
    detail: dict[str, Any] = {
        "sdk_status": sdk.get("status"),
        "login_status": sdk.get("login_status"),
        "option_available": sdk.get("option_available"),
    }
    note = "Round12 仍 UNQUANTIFIED。本轮仅在 SDK 可调用 get_hist_code_list / get_history_stock_status 时重建。"
    from .sdk import DATA_ROOT

    hist_opt = DATA_ROOT / "option_hist_code_list.parquet"
    hist_stock = DATA_ROOT / "hist_code_list.parquet"
    status_sample = DATA_ROOT / "history_stock_status_sample.parquet"
    if hist_opt.is_file():
        detail["option_hist_code_list"] = str(hist_opt)
        detail["option_hist_only"] = True
    if hist_stock.is_file():
        detail["stock_hist_code_list"] = str(hist_stock)
    if status_sample.is_file():
        detail["status_sample"] = str(status_sample)
    # 不再二次 login：连接数已满会卡住。无全市场 get_history_stock_status 则 UNQUANTIFIED。
    detail["no_second_login"] = True
    if not status_sample.is_file():
        note = (
            "SDK 可登录，但 freeze 阶段 hist 曾超时/连接数超限；"
            "未拿到全市场 get_history_stock_status。Round12 口径延续，标 UNQUANTIFIED，不标 PASS。"
        )
    if status == "PARTIAL_SAMPLE_ONLY":
        pass
    else:
        status = UNQUANTIFIED
    payload = {
        "status": status,
        "pass": False,
        "not_pass": True,
        "detail": detail,
        "note": note,
    }
    dump_json(SURV_JSON, payload)
    write_md(
        SURV_MD,
        "\n".join([
            "# Round 15 · Survivorship",
            "",
            f"status=**{status}**。不标 PASS。",
            "",
            note,
            "",
        ]),
    )
    return payload


def load_survivorship() -> dict[str, Any]:
    return load_json(SURV_JSON)


__all__ = ["load_survivorship", "run_survivorship"]
