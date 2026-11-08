# Reproduce round24_v5_migration_v2

工作根：`/data1/stock-mxy/课题/一进二策略`。发布工作树不替代数据根。
审计锚点不改：baseline_v5=`feda6870a26f6efe5d722fd9c3b11130ca266d85`，round23_3=`0d20428fb85003a0a8f383a0e38cefaefe19e961`，round24_v5_migration=`68efe0a29c2b89c044ae916af3981e49ab5be345`。

## 环境
- Python 3.13.13
- xgboost、lightgbm、pandas、numpy、pyarrow、scikit-learn
- arch 未安装。SPA 保持 BLOCKED_STAT_TEST，不把自写检验改名。
- 训练并行上限 2。

## 命令
`python tools/run_round24_v5_migration_v2.py --stage all --jobs 2 --resume`

## 数据接口
- 正式袖套行情：Hermes v7 快照 `data/datasets/hermes-cn-a-2010-20260724-v7-all-features`，经只读 `load_prepared_market`。
- R23.4 内层 2017 窗口另读同一快照 2016-11-01 起的 daily_raw，不改 2018–2026 正式加载起点。
- 没有 ClickHouse / Hermes MySQL / JQ / AmazingData 凭证，所以没有新下载。

## 阻断
- 严格 lane 特征数为 0。标签 FAIL_PARITY（可比 153，未解释错配 15）。正式训练未运行。B0 只是工作日现金占位。

run_code_commit 在结果提交里填写。
