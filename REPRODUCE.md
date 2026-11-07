# baseline_v5 复现说明

## 环境

- 工作根（服务器）：`/data1/stock-mxy/课题/一进二策略`（与 `/home/mxy/Stock/课题/一进二策略` 同 inode）
- **禁止**使用 `/home/mxy/课题/一进二策略`
- Python3 + pandas / numpy / pyarrow / xgboost
- 正式引擎：Stock `first_board_hermes`（只读 PYTHONPATH）

## 本分支可立即核验（无需重跑引擎）

```bash
cd /path/to/quant_rounds   # 本分支检出
PYTHONPATH=. python tools/run_baseline_v5.py --dry-run
PYTHONPATH=. python tools/run_baseline_v5.py --tests
PYTHONPATH=. python tools/run_baseline_v5.py --parity
```

`--parity` 对照本分支内 `result/replay_r7_caliber/v5_R16_O01/replay_summary.json` 与冻结锚点。

## 服务器全量重放（需外部数据）

```bash
cd /data1/stock-mxy/课题/一进二策略
python tools/replay_r7_caliber.py
# 或
PYTHONPATH=. python tools/run_baseline_v5.py --reproduce
```

## 已知阻断

- GitHub 包 **不含** Hermes 原始行情库、完整候选面板 parquet 全量 vendor dump
- 无 Stock `first_board_hermes` 时不能重算正式 NAV，但可核验已上传报告数字
- 状态：`external_dependencies.json` → `NOT_FULLY_SELF_CONTAINED`

## 源版本指针

- 契约：`core/round7/contract.py`、`core/round9/contract.py`（B03）、`core/round19/contract.py`（O01_FORMULA）
- 费用：`core/local_executor.round7_fixed_sleeve_config`
- 权威回放：`result/replay_r7_caliber/`（ROUND7_FLAT_5BP）
