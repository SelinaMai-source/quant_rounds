# 一进二 → 二进三 因子迁移计划（仅计划，不重训）

## 范围
- 源：baseline_v5 Five14/Ten13（源码核实）
- 目标宇宙：Round24 已有二板候选 `result/round24/data/second_board.parquet`
- **不**按历史收益选因子；**不**启动 Round23.3 / 新 2to3 训练

## 迁移类汇总

| 类 | 因子 |
|---|---|
| RECOMPUTE_ON_SECOND_BOARD | fb_*, h5_*, h4_*, g0_amt_vs_5, g1_ret_5, ind_seal_rate_excess, g_prim_px_vol_corr（需供应商公式/分钟） |
| SEMANTIC_REDEFINITION | g0_amount_log→r24_amount_log; g3_days_since_limit |
| BLOCKED_DATA | h1_consistent_buy_ratio, h1_last15m_volume_ratio；分钟未建时 fb/g_prim 实操阻断 |
| BLOCKED_PIT | d1_open_gap |
| DIRECT_TRANSFER | 无（不可直接搬首板面板值） |

## 建议阶段（仍不执行训练）
1. 冻结本审计映射表
2. 为 P1 因子写二板重算脚本与 PIT 测试
3. 分钟/微观要么建二板抽取，要么从候选特征集删除
4. 显式排除 d1_open_gap 出 D 收盘模型
5. 再进入 Round24 因子迁移评审

## 详见
`second_to_third_factor_mapping.csv`、`second_board_coverage_matrix.csv`
