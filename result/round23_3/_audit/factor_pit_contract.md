# baseline_v5 因子 PIT 契约

## 总则

- 决策时点默认 **D 收盘**；执行买入 **D+1 开盘**。
- 训练缺失：**保留 NaN，complete-case；禁止 fillna(0)**。
- **今日下载 ≠ PIT_SAFE**。资金流/微观/基本面必须以 `available_at` / `UPDATE_TIME` / 公告 `known_at` 为准。

## 分列

| feature_id | available_at | PIT 结论 | 备注 |
|---|---|---|---|
| g_prim_px_vol_corr | D_close | 条件安全 | 依赖当日分钟缓存完整 |
| fb_close_seal_streak_norm | D_close | PIT_SAFE | 分钟+涨停价 |
| fb_last30_amount_ratio | D_close | PIT_SAFE | 分钟成交额 |
| h5_last_first_limit_value_change_sym | D_close | PIT_SAFE | limit_events |
| h1_consistent_buy_ratio | D_close / UPDATE_TIME | 条件 | ClickHouse argMax(UPDATE_TIME)；无时点证明则不可称 PIT_SAFE |
| h1_last15m_volume_ratio | D_close / UPDATE_TIME | 条件 | 同上 |
| h4_large_net_to_turnover | D_close | 条件 | daily_money_flow asof |
| h4_retail_net_to_turnover | D_close | 条件 | 同上 |
| g0_amount_log | D_close | PIT_SAFE | daily_raw |
| g0_amt_vs_5 | D_close | PIT_SAFE | 仅用 ≤D 历史 |
| g1_ret_5 | D_close | PIT_SAFE | 后复权 ≤D |
| g3_days_since_limit | D_close | PIT_SAFE | 历史涨停 ≤D-? 不含当日 shift |
| ind_seal_rate_excess | D_close | 条件 | 行业 membership PIT |
| d1_open_gap | D+1_09:25 | **OPEN_INFORMATION / BLOCKED_PIT@D_close** | 源码 open/pre_close-1 |

## 标签成熟（防泄漏）

- Five：`label_e2_net`，成熟列 `next2_date`；训练折要求 `label_end.year <= train_end`。
- Ten：`label_exit_rule_net`，成熟列 `actual_exit_date`；同上。
- 风险：若忽略 label_end 成熟条件，退出规则标签会泄漏未来持有路径。**本审计不静默修补基线。**
