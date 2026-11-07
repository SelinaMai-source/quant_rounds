# baseline_v5 模型注册表

## 身份

- 策略：`baseline_v5` = `R16_O01`
- 父结构：`R9_B03`（B03 资金格 + clean_risk 双头融合之上叠加 O01 Top2 veto）

## Five 头

- 特征：14 列，顺序见 `FACTOR_REGISTRY.md` / `core/round7/contract.py::FIVE_FEATURES`
- 标签：`label_e2_net`；成熟列 `next2_date`
- 分档：`rank_n_bins=5`；`eval_metric=ndcg@1`
- Seeds：[17, 42, 73, 101, 202]（无采样时 seed 排名应不变；集成仍取截面 rank 百分位均值）

## Ten 头

- 特征：13 列 = Five 去掉 `g0_amount_log`
- 标签：`label_exit_rule_net`；成熟列 `actual_exit_date`
- 分档：`rank_n_bins=10`；`eval_metric=ndcg@2`
- Seeds：同 Five

## XGBRanker 冻结参数（两头相同，仅 seed 变）

```json
{
  "objective": "rank:ndcg",
  "eval_metric": null,
  "ndcg_exp_gain": true,
  "n_estimators": 100,
  "learning_rate": 0.05,
  "max_depth": 2,
  "min_child_weight": 1.0,
  "subsample": 1.0,
  "colsample_bytree": 1.0,
  "reg_lambda": 1.0,
  "reg_alpha": 0.0,
  "tree_method": "hist",
  "n_jobs": 2
}
```

硬约束：`subsample=1.0`，`colsample_bytree=1.0`。

## 训练折

- Walk-forward：预测年 **Y** 使用 **Y-3..Y-1** 已成熟标签训练
- ranking group = `signal_date`
- 预测窗：2018–2026（正式评价窗 2018-01-02～2026-07-24）

## 融合

1. 各头多种子 → 日内 rank 百分位平均  
2. `d1_open_gap` 有限且 ≥ −0.01  
3. 再做当日截面百分位  
4. `score = 0.90 * pct_ten + 0.10 * pct_five`  
5. 并列：`instrument_id` 升序稳定打破

## 组合层

- clean_risk：竞价/风格健康度缩放（乘数 ∈ {0.25,0.375,0.5,0.75,1.0}）
- B03：agreement×separation 九格 → Top2 NAV 份额 {0,0.05,0.10}（见 `core/round9/contract.py::B03_CELL`）
- O01：`PUT_SKEW25.shift(1) > rolling(252, min_periods=126).quantile(0.90) → apply_top2_veto rank=2 MIN_KEEP`
- 双袖套 Top1/Top2：初始 30万，两袖套各 15万；rank 预算 30%/10% NAV；`close_from_d2`；涨停顺延

## 产物指针

- Five/Ten 训练报告与 predictions：`result/round7/models/`
- 融合与 clean_risk：`result/round7/predictions/`
- 权威 v5 回放：`result/replay_r7_caliber/v5_R16_O01/`
