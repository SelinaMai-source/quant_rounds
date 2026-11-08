# Round23.4 研究报告

## 问题
在保留收益的前提下，动态融合能否稳定降低回撤。本轮不把 2019=10%、2021=25% 写成新的择时规则。

## 契约
主比较仍是历史开盘信息契约下的研究。d1_open_gap 仍在冻结的 v5 融合里，因为 w=0 必须与原 v5 的分数和候选键一致。这不是无前视证明，也不声称实盘可执行。
内层 w=0 是该半年块上重新拟合的 XGB，不等于已经发布的全年 v5 OOF。外层部署才把冻结的全年 v5 XGB 与全年 LGBM OOF 按内层选出的权重混合。

遗留重放：`PASS`。通过引用 replay_parity.csv 与 R1 summary 的哈希，容差为 Sharpe/CAGR/MaxDD 1e-6、净值 1 元、成交笔数相等。
R0 引擎 CAGR/Sharpe/MaxDD/NAV/成交：0.5499812832273525 / 1.8756627913361097 / -0.240563277153944 / 12750089.355724689 / 1107
R1 遗留 E2 重放：见 replay_parity.csv，与公布参考值一致。
R2 增长：0.48975491754015854 / 1.7137614930533682 / -0.240563277153944 / 9083522.100338858 / 1109
R3 风险：0.5233768799549983 / 1.8072149629593852 / -0.2877362832410054 / 10994652.01202771 / 1106
R4 内层敞口缩放：0.5498413557369592 / 1.8808859957398565 / -0.23959027762313911 / 12740244.678041838 / 1107

## 内层
每个半年块在块开始前一日截止，使用此前 36 个月且 label_end 不晚于截止日的样本，分别训练 XGB 与 LightGBM。2017 年行情来自同一 Hermes v7 快照的更早分区，因为正式评估窗口的热身只到 2017-12。两块现金各自从 300000 重置。风险指标对两块等权，不把重置净值拼成一条价格路径。外层只做评价。
R2 在多数年份没有合格非默认权重。R3 只在内层门槛通过的年份离开 0。样本不足或门槛不通过时权重为 0，默认权重不进入“离默认最近”的排序。

## 回撤与费用
交易现金盈亏加总不等于复利净值差，见 risk_increment_bridge.csv。回撤事件同时覆盖基准最差三段和候选最差三段。S1/S2 是成交层额外不利滑点，S3 不删分数、不递补；开盘涨停由引擎原有拒绝处理，因此 S3 与 S0 的汇总可以相同。

## 统计
CAGR 年化与 core.metrics 相同，按 365.25 个日历日。Sharpe 为日收益 mean/std(ddof=1)*sqrt(252)。风险区间是 5000 次环形块抽样，主块长 20，并报告 10 和 40。arch 未能导入，SPA 状态是 BLOCKED_STAT_TEST，没有把自写检验改名为 SPA。

## 裁决
- Alpha：ALPHA_NOT_PROMOTED
- 风险：RISK_GATE_NOT_MET
- R3 相对 R0 的 CAGR 下降超过 1 个百分点，Ulcer 点估计变差，块抽样下界不支持改善，也不优于匹配敞口的 R4。
- 没有独立 forward，不能标 LIVE_READY。
- CURRENT_BASELINE_CHANGED=FALSE
- PRODUCTION_CONFIG_CHANGED=FALSE
- AUTO_PROMOTE=FALSE

## 通过项引用
- replay_parity.csv sha256=02b83a93e9fb4a5991a91aab5fb06dbbfe0674746c6cd6c9e6c1c1ff14bb5135
- R0_V5/summary.json sha256=6e0e61ab8df6f22e1ae4766f4384a75adc8b78fad938e874832c89c7a2d013dc
- R1_LEGACY_E2_REPLAY/summary.json sha256=57faef8638ee7f58d2485e14a856a05fc8b636afb89906059fa2e81e3141a929
- drawdown_episode_audit.csv sha256=642b3e0d6d2f755a09562d6fd224f57ae9c4285070a3b704b75c6f2461f06a93
- risk_metric_comparison.csv sha256=be55a031b0f183ee946770825eb42932286d13598c63f7bf4a3e82a49df980c5
- inner_selection_by_fold.csv sha256=20477cc3a3cfb464a19b6e2969204540f7967aa5f02bf368945d25ebb738eaba
- matched_exposure_control.csv sha256=f139b98148df751977d89c4be73f4e2597febad7b40d3e194cd64ee39d3c25ef
- execution_sensitivity.csv sha256=ab56bba6157969bc6aca2804a50c05e800a65cfed30b49b467b4cbaf5fca628b
- bootstrap_risk.csv sha256=475a29968428e3164d574c4345d31b65683032fe5069c0b93f75159c92acd9e8
