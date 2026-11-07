# Round22.6 研究报告

## 1. 相对旧代码修了什么
- 经济门改为 CORRECTED：有符号 nav_improvement；缺费用/Top 证据为 NOT_EVALUATED，不得默认 True。
- 旧 daily_spearman 误称 conditional_ic → 现名 raw_rank_ic；真条件诊断为 LINEAR_PARTIAL_INFORMATION_DIAGNOSTIC（折内 Ridge α=10 残差）。
- FS2 的 Five/Ten 完整袖套都跑完，无因 F1 提前 break。
- C4 feature-level placebo 只打乱 C4；旧分数/仓位打乱改名为 PORTFOLIO_SCORE_ALLOCATION_RANDOMIZATION。
- 缓存与正式摘要绑定完整实验身份；账本容差 max(0.01, 1e-8·NAV)。

## 2. 哪些旧结论仍成立 / 撤回
- C1 继续归档，不扩展。
- C4 公式未改。
- ROUND7_FLAT_5BP 名称保留，但真实费用分项已记录，不可理解为总成本仅 5bp。
- 费用分项：{'schedule_name': 'ROUND7_FLAT_5BP', 'slippage_bps': 12.3, 'commission_bps': 1.0, 'transfer_fee_bps': 0.1, 'sell_tax_bps': 5.0, 'note': '契约名含5BP，真实分项如上；买卖合计远不止5bp', 'source': 'core.local_executor.round7_fixed_sleeve_config'}

## 3. 训练与回测计数
- FS2 heads verdict=`FS2_DIRECT_ADD_NOT_SUPPORTED`
- Challenger verdict=`LOCAL_REORDER_EFFECT_ONLY`
- 单测 passed=19 failed=0

## 4. 实验数值
- F1/F2 明细见 `fs2_head_backtests.csv`；partial IC 见 `raw_vs_partial_ic.csv`。
- F1 ΔSharpe=`-0.23911686697244594` F2 ΔSharpe=`-0.6655601003729505`
- Challenger vs CH0=`False` vs CH1=`False`

## 5. 基线与信息契约
- 锁定基线 R16_O01；信息车道 `LEGACY_OPEN_INFORMATION_LANE`（含 d1_open_gap）。
- PIT=`True`

## 6. 是否可替换 baseline
- 决策：`KEEP_BASELINE_V5`；CURRENT 未改。

## 7. 下一步（仅基于本轮）
- 不把 Challenger 最佳结果自动带入 R23.2。
- R23.2 从冻结 v5 端点一致性开始。

## 8. 文件
- 根目录 `/data1/stock-mxy/课题/一进二策略/result/round22_6`
- frozen_manifest sha 见文件；decision=`e25b4cf9d9565deb565c326345e1d8e764d6b32c16fe72f640a1b5d96d2726b5`
