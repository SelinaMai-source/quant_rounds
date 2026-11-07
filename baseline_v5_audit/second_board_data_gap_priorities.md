# 二板数据缺口优先级

基于 **已有** Round24 `second_board.parquet` 宇宙（2015-01-01..2026-07-24），非一进二池。

## P0（阻断迁移训练特征）
1. **分钟路径**（`fb_close_seal_streak_norm`, `fb_last30_amount_ratio`, `g_prim_px_vol_corr`）：二板日分钟未建；首板 close-limit 分钟快照不可当二板全集。
2. **微观 PCV/BCVP**（`h1_*`）：ClickHouse 抽取绑首板 selection；无 UPDATE_TIME 证明不可称 PIT_SAFE。
3. **`d1_open_gap`**：BLOCKED_PIT（OPEN_INFORMATION）。

## P1（可重算但未落面板）
1. `h4_*` 资金流比：Hermes `daily_money_flow` 可按二板键重算；需年频覆盖审计后再标 PIT_SAFE。
2. `h5_last_first_limit_value_change_sym`：`daily_limit_events` 重算。
3. `ind_seal_rate_excess`：行业+涨停事件重算；注意首板构建过滤。

## P2（已有适配列 / 语义问题）
1. `g0_amount_log` / `g0_amt_vs_5` / `g1_ret_5`：面板已有 `r24_*` 适配，可公式迁移但建议新名。
2. `g3_days_since_limit`：SEMANTIC_REDEFINITION，二板截面退化。

## 原则
- 今日下载 ≠ PIT_SAFE
- 不按历史 PnL 挑选迁移因子
- 本审计不重训 2to3
