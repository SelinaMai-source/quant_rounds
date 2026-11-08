# Round24 Migration V2 研究报告

## 结论
本轮没有把迁移标成完成。本地已经有资金流、涨停事件和日线分区，但抽样显示 `available_at` 比交易日晚数年，这是回填时间，不是 D 日 22:00 的可知时间。

`equ_h2l_factor_t1` 不在本地 raw 目录。环境里没有 ClickHouse、Hermes、JQ 的凭证，所以 PCV 和 BCVP_15MIN 的下载次数是 0。没有用 t2 的其他列改名顶替。

`g_prim_px_vol_corr` 不在日内因子列名里。`price_volume_corr` 不能改叫这个名字。

因此没有运行名为 FULL 的模型，也没有把部分列送进正式训练。B2/B4 因为叠加规则的期权可得时间没有重算，标成 BLOCKED_OVERLAY_PIT。

标签状态：`FAIL_PARITY`。真实样本 200 条，可比 153 条，未解释错配 15 条，缺一侧收益 47 条。错配不能变成研究候选。对账只覆盖遗留退出，不是 CAUSAL_TARGET_V1。

g0_amount_log 在 2019-01 首板重叠样本上与 log1p(amount) 一致，但 daily_raw 没有历史 known_at，严格 lane 仍不收。Ten 理论列不含 g0_amount_log。

分钟快照目录名是 close-limit，单日行数远小于全市场一分钟，不能用来重建封板时长。行业表抽样里 known_at 不晚于 effective_from，这还不是每个决策日的成员回放证明。

未来信息测试通过 2 项、失败 0 项。T4 和 T10 通过。T9 是 BLOCKED_OVERLAY_PIT。其余因没有正式模型而未评估。
下载行数：0。B0 现金净值用工作日序列占位，不是交易所日历。

## 冻结
- CURRENT_BASELINE_CHANGED=FALSE
- PRODUCTION_CONFIG_CHANGED=FALSE
- AUTO_PROMOTE=FALSE
- LIVE_READY=FALSE

主 lane 不使用 d1_open_gap，也不使用 D+1 全天行情做 D 日晚选股。原 v5 开盘信息回放只作为 LEGACY_REFERENCE，本轮没有修改它。
