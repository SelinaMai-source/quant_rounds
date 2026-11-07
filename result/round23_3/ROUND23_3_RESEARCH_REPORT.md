# Round23.3 研究报告

## 结论
决策：**KEEP_BASELINE_V5**。Parity=`PASS`。Inner 窗修复=`PASS`。
本轮只在冻结 baseline_v5 上检验 E1/E2/E3/E4 的条件增量；不自动升级生产配置。

## 修复
- Inner Replay 适配器传入 backtest_start/end 与 signal_start/end；半年块若出现全历史 NAV 日则 FAIL。
- 默认权重不再因 distance=0 永远获胜；仅非默认且过全部门槛者竞争。
- w_lgbm=0 直接返回冻结 v5 分数；LGBM 缺失按日回退并记账，不静默缩小候选池。

## E1 Five/Ten 权重
- 状态：`DONE`；详见 `e1_results.csv` / `selected_weights.csv`。

## E2 XGB/LGBM
- 状态：`DONE`；详见 `e2_results.csv`。

## E3 Stacking
- 状态：`NOT_TRIGGERED`；无条件增量则 NOT_TRIGGERED。

## E4 局部纠错
- 状态：`DONE`；CH1 要求 trained_pair_id2/3 与执行一致。

## 统计与执行压力
- Bootstrap 5000 × block20 见 `bootstrap_spa.csv`。
- S0–S3 见 `execution_sensitivity.csv`（S3 成交层取消，不重排不递补）。

## 冻结标志
- CURRENT_BASELINE_CHANGED=FALSE
- PRODUCTION_CONFIG_CHANGED=FALSE
- AUTO_PROMOTE=FALSE
