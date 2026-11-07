# Round24 V5→二进三迁移研究报告

## 研究问题
本轮不是单纯训练一个二进三模型，而是回答 v5 十四因子哪些可迁、哪些需重算/改语义/缺数据，以及双 Head 与 B03/O01 迁移后在真实成交约束下是否改善净收益。

## 决策：`SECOND_TO_THIRD_RESEARCH_CANDIDATE`
- 最佳候选：`V5T`
- Forward：`FORWARD_OOS_NOT_AVAILABLE`（无独立 Forward → RESEARCH_CANDIDATE，非 LIVE_READY）

## 因子与特征集
- CORE3=['g0_amount_log', 'g0_amt_vs_5', 'g1_ret_5']
- TARGET7=['second_board_amount_log', 'second_board_turnover_surprise', 'second_board_return_5', 'first_to_second_amount_ratio', 'second_board_market_breadth', 'second_board_first_touch_time', 'second_board_tail_seal_fraction']
- V5_MIGRATED=['g_prim_px_vol_corr', 'fb_close_seal_streak_norm', 'fb_last30_amount_ratio', 'h5_last_first_limit_value_change_sym', 'h1_consistent_buy_ratio', 'h1_last15m_volume_ratio', 'h4_large_net_to_turnover', 'h4_retail_net_to_turnover', 'g0_amount_log', 'g0_amt_vs_5', 'g1_ret_5', 'g3_days_since_prev_limit_before_streak', 'ind_seal_rate_excess']
- 排除 D-close：`d1_open_gap`；g3 新 ID=`g3_days_since_prev_limit_before_streak`
- 详见 `second_to_third_factor_mapping.csv` / `second_board_coverage_matrix.csv` / `BLOCKED_ITEMS.csv`

## 标签
- 来源：`simulate_single_opportunity`；parity=`FAIL_PARITY` n=`150` mismatch=`13`
- 禁止 next2_date 作为真实退出成熟时间；未成交区分 UNFILLED/RIGHT_CENSORED。

## 模型矩阵
```
       code                    status   sharpe      cagr  max_drawdown       nav_end  trades  promotion_blocked  compare_to_1to2_v5
       CASH                      CASH      NaN  0.000000      0.000000 300000.000000     0.0               True               False
         T0                       ran 0.409333  0.069498     -0.546538 533065.252041  1158.0               True               False
         K3                       ran 0.430826  0.081035     -0.620171 584314.106751  1243.0               True               False
         K7                       ran 0.127632 -0.003189     -0.571388 291913.041999  1260.0               True               False
        V5F                       ran 0.510188  0.099126     -0.407468 673462.874995  1228.0               True               False
        V5T                       ran 0.576659  0.124617     -0.498321 819416.363421  1250.0               True               False
     V5DUAL                       ran 0.550049  0.116093     -0.465627 767779.957191  1240.0               True               False
   V5DIRECT BLOCKED_FEATURE_SEMANTICS      NaN       NaN           NaN           NaN     NaN               True               False
V5DUAL_TOPK                       ran 0.038357 -0.021919     -0.548074 248181.928630  1248.0               True               False
```

## 消融与压力
- overlay_ablation.csv：CORE3/V5 × NoOverlay / TRANSFERRED_FIRST_TO_SECOND_RISK_RULES
- execution_sensitivity.csv：S0–S3（S3 不重排不递补）

## 冻结标志
- CURRENT_BASELINE_CHANGED=FALSE
- PRODUCTION_CONFIG_CHANGED=FALSE
- AUTO_PROMOTE=FALSE
