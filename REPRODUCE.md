# Reproduce round24_2

源审计起点见主仓库分支父提交。工作根：`/data1/stock-mxy/课题/一进二策略`

## 环境
- Python 3 + xgboost/lightgbm/pandas/sklearn/pyarrow
- arch 可选；缺失则 SPA=BLOCKED_STAT_TEST

## 命令
`python tools/run_round24_2.py --jobs 2 --resume`
`python tools/run_round24_2.py --stage decision`

## 已知阻断
- 大型 parquet 仅以 MANIFEST 上传（server path + sha256）

