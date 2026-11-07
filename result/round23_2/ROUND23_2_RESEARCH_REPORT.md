# Round23.2 研究报告

## 1. 修复
- w_lgbm=0 直接返回冻结 v5 路径。
- 内层选择改为完整袖套重放 + Mean(log(1+r))，不再用标签均值代替。
- 多样性降为诊断；允许全年 w_lgbm=0。
- GAP7 半仓改名 GAP7_ALLOCATION_HALF_DIAGNOSTIC。

## 2. 端点与模型
- endpoint=`PASS` decision=`KEEP_BASELINE_V5`
- 详见 model_comparison.csv / weights_by_fold.csv / risk_vs_alpha_comparison.csv / execution_sensitivity.csv

## 3. 是否替换 baseline
- `KEEP_BASELINE_V5`；CURRENT 未改。
