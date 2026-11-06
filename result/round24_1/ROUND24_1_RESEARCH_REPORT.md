# Round24.1 二进三研究报告

决策：**SECOND_TO_THIRD_RESEARCH_ONLY**。这是独立策略线。不替换 baseline_2to3_v0，也不和一进二的 CAGR 或 Sharpe 比高低。

板块审计：{'status': 'OK', 'checked_panel_rows': 394, 'checked_main_st_rows': 105, 'mismatches': 0, 'flat_1099_only_inclusions': 0, 'panel_keys_in_short_window': 137, 'board_ok': True, 'st_cutover': '2026-07-06', 'used_flat_1099_for_all_names': False, 'audit_rule': '只核对二板面板成员是否按法定涨跌幅封板。一板 ST 封板不要求进入二板面板。', 'examples': [], 'rebuild_required': False}

特征集=REDUCED，列=['r241_second_board_amount_log', 'r241_second_board_turnover_surprise', 'r241_second_board_return_5', 'r241_first_to_second_amount_ratio', 'r241_second_board_market_breadth']。缺失没有填 0，FULL 和 REDUCED 没有混在同一张结果里。

标签 `label_2to3_net` 仍是诊断：close(D+2)/open(D+1)-1-0.001。组合净值走 ROUND7_FLAT_5BP，执行器不读取标签，因此训练成本没有在净值里再扣一次。
模型训练用的是模型无关的 `label_2to3_realized_exit_net_v1`：可交易才有标签，不可交易或未成熟不是 0。卖出沿用涨停顺延，不是事后挑最赚钱的卖点。

T0 产物 Sharpe=-0.22484957184300788 CAGR=-0.14476030537635076 MaxDD=-0.8993177200034931。
T2 产物 Sharpe=0.026012651400586034 CAGR=-0.04959394046701038 MaxDD=-0.7019425219727122。
上面两个数字来自既有产物文件。若和口头记忆不一致，以文件为准。

第三板封板率=0.36858533002845867，可交易比例=0.772468557789406，开盘封死频率=0.2239511612962453。这三件事分开报。
RankIC：{'status': 'OK', 'rankic_mean': 0.08571033322283575, 'rankic_days': 1375, 'thin_days': 313}。不足 5 个名字的日子不做截面 RankIC。

A4 门：{'level': False, 'paired_increment': False, 'fee_robust': True, 'catastrophic': True, 'execution_dependent': False, 'passed': False}

分类器输出不是成交概率。没有订单簿，不能把 P 说成排队成交概率。没有 D+1 分钟队列时，开盘封死压力只是压力，不是核验过的排队。

```
                id result
        H241_BOARD   PASS
H241_NO_DOUBLE_FEE   PASS
     H241_FEATURES   PASS
        H241_STAGE   FAIL
```
