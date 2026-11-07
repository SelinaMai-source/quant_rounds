# Round24.2 研究报告

## 1. 修复
- 训练预热恢复到 2015+ 二板信号；绩效窗仍 2018-01-02～2026-07-24。
- 缺行业不再删分钟列；TARGET7 含 first_touch 与 tail_seal。
- simulate_single_opportunity：RIGHT_CENSORED 不作 0；未成交 UNFILLED_ACTION。

## 2. 决策
- `SECOND_TO_THIRD_RESEARCH_CANDIDATE` best=`K7` warmup=`True`

## 3. 文件
- `/data1/stock-mxy/课题/一进二策略/result/round24_2`
