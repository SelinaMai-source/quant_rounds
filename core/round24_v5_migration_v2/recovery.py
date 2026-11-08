"""数据恢复记录。下载次数为 0 时必须写出具体原因。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from core.research_integrity_v3.gates import sha256_file
from core.round24_v5_migration_v2.contract import HERMES_RAW, HERMES_V7


def _env_present() -> list[str]:
    keys = ("CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD", "CH_USER", "CH_PASSWORD", "HERMES_MYSQL_USER", "HERMES_MYSQL_PASSWORD", "JQ_USERNAME", "AMAZINGDATA_USER")
    return [key for key in keys if os.environ.get(key)]


def _sample_lag(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    frame = pd.read_parquet(path, columns=["trade_date", "available_at"])
    lag_hours = (pd.to_datetime(frame["available_at"]) - pd.to_datetime(frame["trade_date"])).dt.total_seconds() / 3600.0
    return {
        "exists": True,
        "rows": int(len(frame)),
        "available_at_min": str(pd.to_datetime(frame["available_at"]).min()),
        "available_at_max": str(pd.to_datetime(frame["available_at"]).max()),
        "lag_hours_median": float(lag_hours.median()),
        "same_day_available_share": float((lag_hours <= 22).mean()),
    }


def build_recovery_jobs() -> pd.DataFrame:
    creds = _env_present()
    money = Path(HERMES_V7) / "daily_money_flow/year=2018/month=01/part-2018-01.parquet"
    limit = Path(HERMES_V7) / "daily_limit_events/year=2018/month=01/part-2018-01.parquet"
    t1 = Path(HERMES_RAW) / "equ_h2l_factor_t1"
    t2 = Path(HERMES_RAW) / "equ_h2l_factor_t2"
    money_lag = _sample_lag(money)
    limit_lag = _sample_lag(limit)
    rows = [
        {
            "job_id": "money_flow_local",
            "数据源": "hermes_v7_normalized",
            "表": "daily_money_flow",
            "字段": "net_inflow_l,retail_net_inflow,turnover_value,available_at",
            "候选键文件hash": "",
            "日期区间": "2018-01 sample",
            "请求方式": "read_local_partition",
            "missing_before": 0,
            "rows_downloaded": 0,
            "source_files": str(money),
            "checksum": sha256_file(money) if money.is_file() else "",
            "coverage_after": money_lag.get("rows"),
            "PIT_status": "AVAILABILITY_UNVERIFIED",
            "错误原因": f"本地分区已存在，未再下载。available_at 中位数滞后 {money_lag.get('lag_hours_median')} 小时，同一日可得比例 {money_lag.get('same_day_available_share')}。这是回填时间，不能当作 D 日 22:00 的 known_at。",
        },
        {
            "job_id": "limit_events_local",
            "数据源": "hermes_v7_normalized",
            "表": "daily_limit_events",
            "字段": "first_limit_value,last_limit_value,available_at",
            "候选键文件hash": "",
            "日期区间": "2018-01 sample",
            "请求方式": "read_local_partition",
            "missing_before": 0,
            "rows_downloaded": 0,
            "source_files": str(limit),
            "checksum": sha256_file(limit) if limit.is_file() else "",
            "coverage_after": limit_lag.get("rows"),
            "PIT_status": "AVAILABILITY_UNVERIFIED",
            "错误原因": f"本地分区已存在，未再下载。available_at 中位数滞后 {limit_lag.get('lag_hours_median')} 小时。不能先用今天的版本再假装历史可得。",
        },
        {
            "job_id": "equ_h2l_factor_t1_pcv",
            "数据源": "hermes_raw_then_clickhouse",
            "表": "equ_h2l_factor_t1",
            "字段": "PCV,BCVP_15MIN,UPDATE_TIME",
            "候选键文件hash": "",
            "日期区间": "2010-01-01..2026-07-24",
            "请求方式": "local_dir_then_authorized_query",
            "missing_before": 1,
            "rows_downloaded": 0,
            "source_files": str(t1),
            "checksum": "",
            "coverage_after": 0,
            "PIT_status": "BLOCKED_ACCESS",
            "错误原因": "本地 raw 只有 equ_h2l_factor_t2，没有 equ_h2l_factor_t1。环境变量中没有 ClickHouse/Hermes/JQ 用户名或密码，下载次数为 0。不猜测口令，不用 t2 的别的列冒充 PCV 或 BCVP_15MIN。",
        },
        {
            "job_id": "g_prim_vendor_column",
            "数据源": "hermes_v7_daily_intraday_factors_and_t2",
            "表": "daily_intraday_factors / equ_h2l_factor_t2",
            "字段": "g_prim_px_vol_corr",
            "候选键文件hash": "",
            "日期区间": "schema_only",
            "请求方式": "schema_scan",
            "missing_before": 1,
            "rows_downloaded": 0,
            "source_files": str(Path(HERMES_V7) / "daily_intraday_factors"),
            "checksum": "",
            "coverage_after": 0,
            "PIT_status": "BLOCKED",
            "错误原因": "抽样分区列名含 price_volume_corr、PV_CORR，不含 g_prim_px_vol_corr。内部公式未公开，禁止改名后沿用。无新下载。t2 目录存在：" + str(t2.exists()) + "。凭证键：" + ",".join(creds),
        },
    ]
    return pd.DataFrame(rows)
