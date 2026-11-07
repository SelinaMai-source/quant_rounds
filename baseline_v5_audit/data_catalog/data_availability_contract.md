# 数据可得性契约

1. Hermes v7 as_of_date = 2026-07-24；增量 EOD 未并入快照的日期不得假装在快照内。
2. AmazingData 全市场分钟：2017–2021 较全；2022+ 全市场稀疏；收盘涨停分钟快照覆盖首板子集，**不等于二板分钟全集**。
3. 资金流 `daily_money_flow`：早期年份缺失 → 特征 NaN（complete-case 丢弃），不可填 0。
4. 微观 PCV/BCVP：依赖 ClickHouse `UPDATE_TIME`；今日重下历史行 ≠ 历史时点可得。
5. `d1_open_gap`：D+1 开盘信息；历史回测可用，D 收盘决策不可用。
6. 行业 membership：用 effective_from 做 PIT；不确定则标 PIT_UNCERTAIN。
7. ST：必须用 known_at / 公告时点，禁止用事后 ST 状态回刷。
