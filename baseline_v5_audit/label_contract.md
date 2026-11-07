# baseline_v5 标签契约

## Five：`label_e2_net`
- 买：D+1 开盘；卖：D+2 收盘（固定持有至 next2_date）
- 净收益含费用分项（与 ROUND7 费用日程对齐的标签口径）
- 成熟：`next2_date` 必须已知后才可入训

## Ten：`label_exit_rule_net`
- 买：D+1 开盘；卖：退出规则路径（`close_from_d2`：D+2 起首个非涨停收盘等）
- 成熟：`actual_exit_date`
- 与组合层退出规则对齐，但仍是**标签**，不等于实盘成交核验

## 泄漏风险（记录，不静默修）
1. 未按 label_end 成熟切折 → 用未来退出日样本训练  
2. `d1_open_gap` 作为特征把 D+1 开盘信息注入 D 日排序（历史车道允许，实盘决策不可用）  
3. 标签费用 vs 组合费用日程名同为 ROUND7_FLAT_5BP，但若改一边未改另一边会造成训练/回测不一致  

权威面板：
- Five: `.../aligned_original_candidate_full44_plus_d1_open_gap_v1`
- Ten: `.../aligned_original_candidate_fs01122_chain_label_v1`
