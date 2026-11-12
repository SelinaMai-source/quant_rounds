# 服务器量化数据清单

- 盘点日期：2026-10-04
- 盘点人：只读巡检（`df` / `du` / `ls` / parquet 元数据与列统计 / 小表全量读取 / 供应商 `source_schema.json` 注释）。未改代码、未改数据、未连上 ClickHouse 业务查询。
- 范围：当前用户可读的 A 股量化研究数据。主路径是 `/home/mxy/Stock`（软链到 `/data1/stock-mxy`）、`/data1/Stock`、`/home/mxy/AmazingData`。`/data1`、`/data2` 上其他用户目录只记录可见性和体积。操作系统目录、`node_modules`、`.cache`、虚拟环境不计入数据资产。
- 口径：下文数字来自本次命令。写明「2026-09-27 审计、本次未重算」的句子沿用同目录 `result/data_inventory/` 里已经落盘的审计，本次没有再跑一遍。写「未核实」的，是这次没打开或权限不够。
- 研究截止：规范化日频快照 `as_of_date = 2026-07-24`。增量 EOD 批次文件名最晚到 `2026-09-30`，尚未并入该快照。

机器可读的字段级目录仍在 `result/data_inventory/`：`data_catalog.csv`、`field_catalog.csv`、`data_quality_summary.csv`、`data_relationships.csv`（生成于 2026-09-27）。本文是给人挑策略用的清单，字段太多时按组列出，关键字段不省略。

---

## 1. 总览

| 数据域 | 存储位置 | 体积 | 时间覆盖 | 频率 | 用途 |
|---|---|---|---|---|---|
| Hermes v7 规范化快照 | `/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features`（`/data1/Stock/data/datasets/` 下有同名副本） | 8.6G | 日频主表 2010-01-04 .. 2026-07-24 | 日 / 事件 | 全 A 日线、后复权、估值、资金流、涨跌停事件、两融、财报、行业、ST、股本、交易日历 |
| Hermes 原始 29 表 | `/data1/Stock/data/acquisitions/hermes-cn-a-2010-20260724-v7-rebuild/raw` | 采集目录 `…-v7-rebuild` 2.1G（`du`）；同级旧采集 `hermes-cn-a-2010-20260724` 7.3G | 与 v7 同源，截止 2026-07-24 | 日 / 事件 | v7 没保留的列：成交笔数、各档买卖笔数、涨停触及次数 |
| Hermes 增量 EOD | `/data1/Stock/data_live/inbox/batches/hermes-eod-*` | `inbox` 合计 17G（含 `server_relay` 12G） | 批次日 2026-08-31 .. 2026-09-30，共 19 个交易日文件名 | 日终增量 | 把研究库往后接的原料；缺 3 张表，且未并入 v7 |
| AmazingData 全市场 1 分钟 | `/home/mxy/AmazingData/data/amazingdata_raw/kline_min1` | 20G | 2017–2021 全市场；2022–2024 每年约 10–11 只 | 1 分钟 | 样本内日内特征、执行、退出 |
| AmazingData 收盘涨停分钟 | `/data1/stock-mxy/data/source_snapshots/amazingdata-cn-a-close-limit-min1-*` | 两段合计约 1.0G（目录 `source_snapshots` 1017M） | 2013-01-04 .. 2026-07-24 | 1 分钟，仅 D 日收盘涨停股 | 一进二封板路径 |
| 股东户数 / 大宗交易 | `/home/mxy/AmazingData/data/amazingdata_raw/{holder_num,block_trading}/all.parquet` | 各约 5MB | 户数公告日 19930112 .. 20260919；大宗 20020319 .. 20260918 | 事件 | 筹码、折溢价事件 |
| 聚宽阶段缓存 | `/data1/Stock/.jq_stage` | 3.6G | `get_price` 2013-01-04 .. 2026-08-14 | 日 | 行情交叉校验、逐日行业、沪深300/中证500/中证1000 成分 |
| NShape 因子库 | `/data1/Stock/.zyquant/cache/factors-pit-ex-st-v2` | 25G | 抽样因子 `ret20`：2013-01-04 .. 2026-06-10，12,770,392 行 | 日 | 350 个已算好的全市场因子 |
| 沪深300 日行情 | `/data1/Stock/data/whitehorse/csi300.parquet` | 312KB | 2007-01-04 .. 2026-09-09，4,785 行 | 日 | 大盘价格 |
| 低换手派生日频面板 | `课题/低换手策略1/result/YB/derived/panel/daily_panel.parquet` | 745M | 2017-01-03 .. 2026-07-24，10,106,408 行，5,447 只 | 日 | 低换手回测直接输入；行业两列为空类型 |
| 一进二首板面板 | `data/first_board_hermes/aligned_original_candidate_full44_2014warmup_2018oos_v1/candidate_panel.parquet` | 面板目录属 `first_board_hermes` 1.2G | 2013-01-07 .. 2026-07-01，63,696 行 | 事件日 | 一进二特征+标签，标签列与特征同表 |
| ClickHouse `hermes` | `127.0.0.1:18123`，版本 25.3.14.1，`/ping` 返回 `Ok` | 行数未知 | 未知 | 源库 | 611 张表的 schema 在本地；默认用户认证失败 |
| 龙虎榜 / 题材 / 北向明细 / Tick / 新闻全文 / 一致预期 | 本地文件未找到 | — | — | — | 龙虎榜、题材、解禁等只在源库 schema 中 |

磁盘（`df -h`）：

| 文件系统 | 容量 | 已用 | 可用 | 挂载 |
|---|---|---|---|---|
| `/dev/mapper/ubuntu--vg-lv--0` | 437G | 161G | 254G | `/` |
| `/dev/nvme0n1` | 7.0T | 369G | 6.3T | `/data1` |
| `/dev/nvme1n1` | 7.0T | 23G | 6.6T | `/data2` |

`/data`、`/mnt`、`/opt` 没有研究数据。`df` 显示 `/data1` 已用 369G；对可读顶层目录 `du` 合计约 270G。差额未逐文件解释，无权限子目录（CNE6、`warehouse`、`stock-xzq`）的内容没有计入 `du`。

---

## 2. 存储布局

```
/data1                         df 已用 369G
├── Stock/                     181G   root 所有，mxy 可读（部分子目录 700）
│   ├── data/                  37G    数据集、采集、因子缓存、whitehorse、nshape 研究副本
│   ├── data_live/             22G    增量 EOD、inbox、nshape 实盘缓存
│   ├── .zyquant/cache/        25G    NShape 350 因子
│   ├── .ad_cache/             7.2G   AmazingData h5 缓存（财报、状态、成分）
│   ├── .ad_stage/             5.4G   AD 日 K 批次 + financials 5.1G
│   ├── .jq_stage/             3.6G   聚宽
│   ├── logs/                  49G    日志，不是行情
│   ├── runs/                  34G    回测/训练产物
│   └── warehouse/datasets/cn-a-jq-2013-2026-r3   目录存在，内部 Permission denied
├── stock-mxy/                 68G    /home/mxy/Stock 软链目标
│   ├── data/                  14G    v7、分钟快照、一进二因子缓存与面板
│   ├── runs/                  23G    一进二搜索与实验（含 factor_search 22G）
│   └── 课题/                  32G    低换手 31G + 一进二 1.5G
├── xj/ETF轮动/backtests/      21G    仅回测输出，未见 ETF 行情库
└── stock-xzq/                 当前用户不可读（du 只见到目录项 4K）

/data2
├── Ljf/                       19G    他人量化目录；min5 文件权限 600，打不开
└── Ljf2/                      4.0G   含 swing5m 缓存 3.2G，未当成本用户数据源

/home/mxy
├── AmazingData/               22G    分钟线 20G + 户数/大宗 parquet + 本地 h5
├── 课题/一进二策略/           1.5G   与 stock-mxy 下的一进二不是同一 inode，缺 round7、round9
└── D:/AmazingData_local_data/ 20K    空壳目录
```

低换手课题根：`/data1/stock-mxy/课题/低换手策略1`。README 写明旧路径 `/home/mxy/课题/低换手策略1` 已不存在。本次确认该旧路径不存在。

---

## 3. Hermes v7 规范化快照

路径：`/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features`

- 格式：按年/月分区的 parquet。`manifest.json`：`schema_version` 1.2，`as_of_date` 2026-07-24，`created_at` 2026-08-30T20:26:23Z，fingerprint `1306e83ba88ba79340d2a8d824b914938fe9808bfcb1560d6abcf3c3c188304c`。
- 体积：8.6G。19 张表。
- 股票池：`instruments` 5,807 行，全部 `asset_type=stock`。交易所：XSHE 3,047、XSHG 2,426、XBEI 334。`delist_date` 非空 277 行。`lot_size` 全部 100，`sell_delay_days` 全部 1（T+1）。科创板 200 股起买没有单独字段。
- 日频主键：`(trade_date, instrument_id)`。2026-09-27 审计记录日频表重复键为 0，本次未重算。
- 复权：manifest `adjustment.method = vendor_post_adjusted`，锚点 `Hermes ACCUM_ADJ_FACTOR_2`，`event_adjustments = 55001`。前复权文件未找到。后复权因子累积历史事件，用来算收益是时点安全的。
- 费用：`market_rules` 3 行（XSHG/XSHE/XBEI），`commission_bps`、`minimum_commission`、`sell_tax_bps`、`buy_tax_bps`、`transfer_fee_bps` 全为 NaN，`source=source_missing`，`scenario=False`。manifest `backtest_ready=false`，原因是没有完整历史费率序列。回测费率要在策略配置里单设。低换手 README 的约定是单边佣金 0.1% + 买卖滑点各 0.1% + 卖出印花税万五。
- 更新：这是 2026-08-30 建成的静态快照。更晚的行情在第 4 节增量批次里，没有写回本目录。
- 副本：`/data1/Stock/data/datasets/` 下有同名目录。抽查 `daily_raw/.../part-2022-03.parquet` 两边都是 3,509,922 字节、inode 不同。本次没有做全库校验。
- 加载：低换手 `core/data/hermes.py` 的 `read_hermes_table`；面板构建在 `core/data/hermes_panel.py`。

`daily_raw` 按年去重股票数（本次对全表 `trade_date, instrument_id` 计算）：2014 年 2,595 只 / 620,557 行；2018 年 3,574 / 857,069；2020 年 4,197 / 955,604；2025 年 5,499 / 1,317,313。停牌行 `paused=true` 共 424,336。

### 3.1 `daily_raw` 原始日线

- 199 个 parquet，14,522,056 行，日期统计 2010-01-04 .. 2026-07-24。
- 粒度：股票 × 交易日。含停牌行。
- 可支撑：日频价量、涨跌停价约束、低换手、动量、一进二的次日开盘成交。

| 字段 | 含义 | 单位 / 备注 | 缺失 |
|---|---|---|---|
| trade_date | 交易日 | date | 统计完整 |
| instrument_id | 证券代码，形如 `000001.XSHE` | — | — |
| open, high, low, close, pre_close | 不复权 OHLC、前收 | 价格；供应商 `mkt_equd` 注释为今开盘/最高/最低/今收盘/昨收盘 | 停牌行 open/high/low 为 0 的细节：2026-09-27 审计，本次未重算 |
| volume | 成交量 | `int64`；供应商注释「成交量」，未写股或手 | — |
| amount | 成交金额 | 供应商 `TURNOVER_VALUE` 注释「成交金额」，未写元 | — |
| paused | 停牌 | bool；true 共 424,336 行 | — |
| limit_up, limit_down | 涨停价、跌停价 | 价格。2026-09-27 审计：3,305 行 `limit_up=999999.999`（无涨跌停哨兵），另有约 0.16% 空值。本次未重算 | 见左 |
| source_record_id, source_updated_at | 溯源 | — | — |

原始表 `mkt_equd` 比 v7 多：`ACT_PRE_CLOSE_PRICE`（实际昨收）、`DEAL_AMOUNT`（成交笔数）、`CHG_PCT`、`TURNOVER_RATE`、`PE`/`PE1`/`PB`、`NEG_MARKET_VALUE`、`MARKET_VALUE`。这些在 v7 里被拆到估值表或直接丢掉。成交笔数要做「笔均额」时读 raw，不要在 v7 里找。

板块涨跌停幅度本次没有重新用分位数验证。2026-09-27 审计的规则是：主板 10%（ST 5%）、创业板 2020-08-24 前 10%、之后 20%、科创板 20%、北交所 30%。

### 3.2 `daily_post_adjusted` 后复权日线

- 14,522,056 行，与 `daily_raw` 同行数，同一日期区间。
- 字段：`open_post, high_post, low_post, close_post, pre_close_post, adjustment_factor, factor_source, adjustment_version`。
- `adjustment_factor` 对应供应商 `ACCUM_ADJ_FACTOR_2`（累计后复权因子）。2026-09-27 审计范围 1.0 .. 11,091.8，本次未重算极值。
- 可支撑：跨除权的收益率、动量、波动。做成交约束仍用 `daily_raw` 的涨跌停价和 `paused`。

### 3.3 `daily_valuation` 估值、市值、换手

- 14,512,461 行，2010-01-04 .. 2026-07-24。比 `daily_raw` 少 9,595 行。
- 主键 `(trade_date, instrument_id)`。

| 组 | 字段 | 供应商侧含义 |
|---|---|---|
| 估值 | pe_ttm, pe_lyr, pb, ps_ttm, pcf_ttm, pcf_operating_ttm, dividend_yield | 市盈率、市净率、市销率、市现率、股息率。v7 字段名即口径；极值与负值本次未重算（2026-09-27 记有 pe_ttm 到 ±3e6 量级） |
| 规模 | total_shares, market_cap, circulating_shares, circulating_market_cap, free_float_shares, free_float_market_cap, a_shares, a_market_cap | 总股本/总市值、流通股本/流通市值、自由流通股本/市值、A 股股本/市值。注释未标注「元」或「股」 |
| 流动性 | turnover_rate | 日换手率。低换手面板另造了 `turnover = amount / float_mv`，与本列并存 |

`available_at` 被设成与 `trade_date` 相同，无法从快照证明 PE/PB 的 TTM 在财报公告次日才切换。用估值做严格 PIT 时，应自己用 `financial_reports.available_at` 重算。

raw 还有 `mkt_equd_eval_new` 的 `FREE_TURNOVER_RATE` 等，v7 未单独保留该列名。

### 3.4 `daily_money_flow` 资金流

- 12,092,055 行，日期统计 2014-01-02 .. 2026-07-24（151 个 row group 有 min/max，48 个没有统计，空月分区会这样）。
- 源表 `mkt_equ_mf_new`。供应商注释把流入分成小单/中单/大单/超大单，按订单金额分档，注释里没有投资者身份。
- 单位：注释为「流入/流出/净流入/成交金额」，未写货币单位。

| 字段 | 含义 |
|---|---|
| inflow, outflow, net_inflow | 流入、流出、净流入 |
| inflow_{s,m,l,xl}, outflow_{s,m,l,xl}, net_inflow_{s,m,l,xl} | 小/中/大/超大单 |
| main_net_inflow, retail_net_inflow | 供应商 `MAIN_FLOW`「主力净流入」、`SMAIN_FLOW`「散户净流入」。2026-09-27 在 2024 年验证 main=L+XL、retail=S+M，本次未重算 |
| net_in_open, net_in_close | 开盘净流入、尾盘净流入 |
| turnover_value | 成交金额 |
| available_at | 2026-09-27：最早 2020-09-23，2021 年前大量回填，2022 起基本当日入库。本次未重算 |

v7 丢掉、raw 仍在的列：`DEAL_AMOUNT` 成交笔数，`TURNOVER_VOL`，`BUY_VOL_*` / `SELL_VOL_*` 各档量，`BUY_ORD_*` / `SELL_ORD_*` 各档笔数。笔数可以做单笔规模，只能读 raw。

可支撑：日频资金流、大单集中度、尾盘流入。不能支撑：真实机构/游资身份、龙虎榜席位。

### 3.5 `daily_intraday_factors` 日内因子日频摘要

- 13,865,908 行，2011-01-04 .. 2026-07-24。源表 `equ_h2l_factor_t2`，35 列里的值字段在 v7 中有 29 个。
- 全市场日频，由供应商用日内数据聚合成一天一行。本地没有用来复原这些因子的 tick。

| 组 | v7 字段 | 供应商注释 |
|---|---|---|
| 价量相关 | dvn_dvn_corr, dvn_dvp_corr, dvn_v_corr, dvp_dpn_corr, dvp_dpp_corr, dvp_dvn_corr, dvp_dvp_corr, dvp_v_corr, price_volume_corr, return_volume_corr, volume_leads_amplitude_corr | 差分相关、价量相关、涨跌幅与成交量相关、先量后振幅 |
| 成交额分布 | volume_weighted_entropy, weighted_close_ratio, weighted_close_skew | 单位成交额熵、加权收盘价比、加权收盘价偏度 |
| 非流动性 | buy_illiquidity, sell_illiquidity, shortest_path_illiquidity, price_resiliency | 买单/卖单非流动性、K 线最短路径非流动性、价格弹性 |
| 分时换手 | overnight / hour1..hour4 的 turnover_rate 与 pure_turnover_rate，reformed_pure_turnover_rate | 隔夜与第 1–4 小时流通换手率、纯净换手率、结构化纯净换手率 |
| 单笔 | average_trade_inflow_ratio, average_trade_net_inflow_ratio | 平均单笔流入金额占比、流入流出金额值比 |

2026-09-27 审计：约 23,094 行绝对值大于 1 的「相关」，`hour1_turnover_rate` 约 28% 为负，`price_resiliency` 有极端值。这些列的数值尺度和注释里的「相关系数/换手率」不一致，使用前要当供应商变换后的得分。`available_at` 最早 2024-02-05，更早历史是回填。本次未重算缺失率。

同族 `equ_h2l_factor_t1`（35 列）、`t3`（34 列）、`t4`（34 列）只在源库 schema。本地微结构快照只抽了涨停股上的少数几列，见第 6 节。

### 3.6 `daily_limit_events` 涨跌停事件

- 541,031 行，2010-01-04 .. 2026-07-24。主键 `(trade_date, instrument_id, limit_type)`。
- 源表 `mkt_limit_ind`。只含触及涨跌停的股票日，不是全市场每日一行。
- 2026-09-27：up 341,101 / down 199,930。本次未重算分类计数。

| 字段 | 含义 | 备注 |
|---|---|---|
| limit_type, source_limit_type_code | up / down | — |
| limit_price | 涨跌停价 | — |
| first/last/max_limit_volume, first/last/max_limit_value | 首次、最后一次、最高封单量与金额 | 供应商注释 |
| close_limit_volume, close_limit_value | 收盘封单量与金额 | 2026-09-27：收盘未封住时为 NULL（涨停约 30.9%、跌停约 41.1%），语义空值 |
| first_limit_time_seconds, last_limit_time_seconds | 首次/最后涨跌停时刻，午夜起秒数 | 含集合竞价早于 09:30 的行（2026-09-27：88,586 行） |
| first_limit_minutes_from_open | 距 09:30 的分钟，墙钟，含午休 | 竞价封板被截成 0 |
| available_at | 供应商更新日 | 2026-09-27：最早 2023-08-15，2010 年代历史是回填修订版 |

可支撑：封板时刻、封单强度、跌停压力、一进二。不能从本表直接得到「开板次数」。开板次数的近似在分钟快照和 raw `mkt_limit.UP_LIMIT_REACHED_TIMES`（供应商触及次数，语义不同于分钟段数）。

raw `mkt_limit` 另有全市场涨跌停价和 `UP_LIMIT_REACHED_TIMES` / `DOWN_LIMIT_REACHED_TIMES`，14,499,152 行（2026-09-27 行数，本次未重算；表目录存在）。

### 3.7 `daily_margin` 融资融券

- 6,214,181 行，2010-03-31 .. 2026-07-24。源表 `fst_detail`。只覆盖两融标的，不是全 A。
- 2026-09-27 标的数：2010 年 96、2014 年 916、2020 年 1,914、2025 年 3,821。本次按年 `count_distinct` 因空分区日期类型失败，未重算。

| 字段 | 供应商注释 |
|---|---|
| financing_balance | 本日融资余额 |
| financing_buy_value | 本日融资买入额 |
| financing_repayment_value | 本日融资偿还额 |
| securities_lending_balance_volume | 本日融券余量 |
| securities_lending_sell_volume | 本日融券卖出量 |
| securities_lending_repayment_volume | 本日融券偿还量 |
| securities_lending_balance_value | 本日融券余量金额 |
| margin_balance | 本日融资融券余额 |

2026-09-27：2016 年起 `available_at = trade_date + 1`，2025 年入库约在 D+1 15:00–16:00（上海），晚于 D+1 开盘。D 日收盘信号要用 D−1 的两融。本次未重算时点。

源库还有 `fst_target`（标的纳入剔除）、`fst_total` 等，本地 raw 未下载。

### 3.8 财务与指标

三张表都带 `available_at`。manifest `financials.pit_validated = true`。

| 表 | 行数 | 日期统计 | 粒度 |
|---|---|---|---|
| financial_reports | 1,685,981 | published_at 2009-04-07 .. 2026-07-24；available_at 2010-01-04 .. 2026-07-27 | 一份报表一行 |
| financial_facts | 59,189,850 | fiscal_period_end 2007-12-31 .. 2026-06-30；available_at 2010-01-04 .. 2026-07-27 | 报表 × 科目 |
| fundamental_metrics | 13,867,551 | available_at 2009-04-08 .. 2026-07-27 | 指标 × 报告期 × 口径 |

`financial_reports` 字段：`report_id, instrument_id, statement_type, fiscal_period_start/end, filing_period_end, record_kind, published_at, available_at, revision_sequence, currency, source_report_type`。2026-09-27：`available_at = published_at + 1 天` 的比例为 100%，`revision_sequence` 最大 16，`statement_type` 为 balance/income/cash_flow，`record_kind` 为 current/comparative。本次未重算这些比例。

`financial_facts` 字段：`report_id, item_code, statement_type, value, unit, value_basis, source_field`。科目个数本次未重算（2026-09-27：资产负债表 159、利润表 80、现金流量表 80）。

`fundamental_metrics` 本次扫完全表，31 个 `metric_code`：

- 每股与盈利：`basic_eps, diluted_eps, hermes_eps_ttm, roe, roa, hermes_roe_q_pct, hermes_inc_return_q_pct, net_margin, gross_margin, operating_margin`
- 利润与增速：`revenue, operating_cost, operating_profit, net_profit, net_profit_parent, revenue_yoy, net_profit_parent_yoy, hermes_total_revenue_yoy_q_pct, hermes_net_profit_yoy_q_pct`
- 现金流：`operating_cash_flow, operating_cash_flow_yoy, free_cash_flow, capex, ocf_to_net_profit, hermes_nocf_coverage_ttm`
- 资产负债：`total_assets, total_liability, total_equity, equity_parent, debt_to_assets, current_ratio`

另有 `basis`（instant/ytd 等）、`unit`、`source_report_ids`、`quality_status`。用这些列做因子时过滤 `available_at <= 信号日`。业绩预告（`fdmt_ef_new` 等）、业绩快报（`fdmt_ee`）、预约披露日在源库 schema，本地未下载。

### 3.9 参考表：行业、ST、股本、公司行为、日历、证券主数据

| 表 | 行数 | 范围 | 主键 | 要点 |
|---|---|---|---|---|
| industry_membership | 26,529 | effective_from 1996-03-12 .. 2026-07-24 | classification + instrument + effective_from | `SW_L1:SW` 11,701 段；`SW_L1:V2021` 14,828 段。只有申万一级。V2021 的生效日起点早于该分类发布年，2026-09-27 判定为分类回填，`known_at == effective_from`。2013–2021 用 V2021 做行业中性有事后分类风险 |
| special_treatment | 3,345 | effective_from 1998-04-28 .. 2026-07-23 | instrument + effective_from | state_code 1–7 的次数：1 为 1083，2 为 1486，3 为 105，4 为 152，5 为 274，6 为 93，7 为 152。代码含义本次未对照字典 |
| share_capital | 154,575 | effective_from 1990-12-10 .. 2026-07-24；available_at 最晚 2026-08-18 | capital_event_id | 总股本、非流通、限售、可交易、A/B/H 股。available_at 晚于快照 as_of，使用时要按信号日截断 |
| corporate_actions | 55,001 | ex_date 2010-01-15 .. 2026-07-24；announced_at 2009-03-21 .. 2026-07-20 | event_id | cash_dividend 46,299；bonus 8,546；rights_issue 153；merge 3。字段含每股现金、送转比例、配股价、登记日、除权日、派息日 |
| trade_calendar | 9,571 | 2010-01-04 .. 2026-07-24 | trade_date + exchange | XSHG 4,020、XSHE 4,020、XBEI 1,531 |
| instruments | 5,807 | list_date 1990-12-01 .. 2026-07-24 | instrument_id | 见本章开头。`delist_date` 是未来信息，不能用来在历史上过滤股票池 |
| universe_membership | 5,807 | — | universe_id + instrument_id | `effective_to` 与退市相关。2026-09-27 标为 PIT_UNSAFE。本次未重算 |
| market_rules | 3 | 2010-01-01 .. 2026-07-24 | rule_id | 费率全空，见本章开头 |

解禁、增减持、回购、质押、定增：本地 v7/raw 都没有对应表。源库 schema 有 `equ_shares_float`、`equ_sh_schg`、`equ_share_buy_back`、`equ_pledge_*`、`equ_spo`。

---

## 4. Hermes 原始采集与盘后增量

### 4.1 静态 raw（29 张）

目录：`/data1/Stock/data/acquisitions/hermes-cn-a-2010-20260724-v7-rebuild/raw`

`equ_allot, equ_div_pit, equ_free_shares, equ_h2l_factor_t2, equ_inst_sstate, equ_shares_change, equ_splits, fdmt_main_data_q_pit, fdmt_md_n_ttmp, fst_detail, md_inst_type, md_sec_chg, md_sec_halt, md_sec_symbol, md_security, md_trade_cal, md_type, mkt_adjf_af, mkt_div_yield, mkt_equd, mkt_equd_adj_af, mkt_equd_eval, mkt_equd_eval_new, mkt_equ_mf_new, mkt_limit, mkt_limit_ind, vw_fdmt_bs_new, vw_fdmt_cf_new, vw_fdmt_is_new`

行数以 2026-09-27 `data_catalog.csv` 为准（本次未逐表重数）。和策略直接相关、v7 没有的列：

- `mkt_equd.DEAL_AMOUNT`：成交笔数
- `mkt_limit.UP_LIMIT_REACHED_TIMES` / `DOWN_LIMIT_REACHED_TIMES`：涨跌停触及次数
- `mkt_equ_mf_new` 的 `BUY_ORD_*` / `SELL_ORD_*`：各档笔数，以及各档成交量
- `md_sec_halt`：停复牌事件（v7 只有 `paused` 布尔）
- `fdmt_main_data_q_pit`（125 列）、`fdmt_md_n_ttmp`（167 列）：季度 PIT 主指标与 TTM，宽表，比 `fundamental_metrics` 的 31 个指标更细
- `vw_fdmt_bs_new` / `is` / `cf`：三大表宽表

同级还有早期快照，表更少：

| 目录 | du | 相对 v7 少什么 |
|---|---|---|
| `datasets/hermes-cn-a-2010-2026` | 与 r2 同级，父目录 `datasets` 16G | 无资金流、涨跌停事件、两融、日内因子 |
| `datasets/hermes-cn-a-2010-2026-r2` | 同上 | 同上 |
| `datasets/hermes-cn-a-2010-2026-mf` | 7.3G | 有资金流；无涨跌停事件、两融、日内因子。NShape 因子库建在这一版上 |

采集状态库：`…-v7-rebuild/state.sqlite`（以及另外两份 acquisition 的 sqlite）。本次未打开。

### 4.2 增量 EOD（研究库截止日之后）

`/data1/Stock/data_live/inbox/batches/` 下 19 个目录，文件名日期：

`20260831, 20260901, 20260902, 20260903, 20260904, 20260907, 20260908, 20260909, 20260910, 20260914, 20260915, 20260916, 20260918, 20260921, 20260922, 20260923, 20260924, 20260929, 20260930`

抽查 `hermes-eod-20260930-…/raw/mkt_equd/part-20260930-20260930.parquet`：5,651 行，`TRADE_DATE` 为 2026-09-30。列包含不复权 OHLC、成交量、成交金额、`DEAL_AMOUNT`、PE/PB、市值、涨跌幅、换手、`UPDATE_TIME`。

该批 raw 子目录比静态 29 表少这三张：`equ_h2l_factor_t2`、`fst_detail`、`mkt_limit_ind`。因此 2026-07-24 之后，日线/复权/资金流/估值/财报宽表有增量文件，日内因子、两融明细、涨跌停事件没有出现在这 19 个批次里。

`data_live/releases/hermes-cn-market-eod-*` 在 2026-09-27 的 `du` 里每个目录 4K，像空目录；本次没有逐个打开。`server_relay/relay/batches` 有 17 个日期目录（2026-09-07 .. 2026-09-30，含 09-11、09-17、09-28，这三天不在 inbox 的 19 个批次里）。relay 内部文件本次未打开。

v7 在 2026-07-24 结束，inbox 批次从 2026-08-31 开始。2026-07-27 .. 2026-08-28 这段，两处都没有对应批次。是否还有别的目录补上，本次未找到。

---

## 5. 分钟线

### 5.1 全市场 1 分钟（AmazingData）

路径：`/home/mxy/AmazingData/data/amazingdata_raw/kline_min1/parts/year=YYYY/*.parquet`

| 年 | 文件数（每文件一只股票） | 字节 |
|---|---|---|
| 2017 | 3,464 | 3,616,391,033 |
| 2018 | 3,568 | 3,825,521,952 |
| 2019 | 3,765 | 4,158,810,892 |
| 2020 | 4,153 | 4,580,220,583 |
| 2021 | 4,618 | 5,116,307,986 |
| 2022 | 11 | 12,658,128 |
| 2023 | 11 | 12,517,361 |
| 2024 | 10 | 11,719,299 |

抽样 `year=2021/sh600645.parquet`：58,320 行（58,320 / 240 = 243 个交易日），字段 `股票代码, 交易日期, bar_time, open, high, low, close, volume, amount, source`。`bar_time` 从 09:30 到 14:59。抽样 `year=2022/sh600016.parquet`：58,080 行，同样是 09:30–14:59。没有单独的 09:25 集合竞价 bar。09:30 这根是否并入竞价量，未核实。

低换手 `result/YB/derived/kline_min1` 的 20G 是另一份拷贝：抽查 `sh600645.parquet` 与 raw 的 inode 不同、`nlink=1`。文件数与 raw 的 2017/2018/2021/2022 一致。

没有 5 分钟、15 分钟、30 分钟的独立全市场库（本用户可读范围内）。`/data2/Ljf/quant/data/min5` 有大量 `*.SZ.parquet` / `*.SH.parquet`，`000001.SZ.parquet` 权限 `-rw------- root`，打开失败。

可支撑：2017–2021 的日内因子、开盘成交仿真、D+1..D+3 分钟退出研究。2022 年以后做不到全市场分钟回测。

### 5.2 收盘涨停子集分钟

| 快照 | 日文件数 | 行数 | 字节 |
|---|---|---|---|
| `…/amazingdata-cn-a-close-limit-min1-20130104-20211231-v1/minute_bars` | 2,188 | 33,657,610 | 649,475,125 |
| `…/amazingdata-cn-a-close-limit-min1-20220104-20260724-v1/minute_bars` | 1,103 | 19,479,964 | 385,529,045 |

字段：`trade_date, instrument_id, bar_time, open, high, low, close, volume, amount, event_available_at, source_fetched_at, source_record_id, source_batch_id`。

股票范围是当天收盘涨停的股票，且只有信号日 D 的 bar。没有 D+1 到 D+3。一进二可以用它做 D 日封板路径，不能用它做持有期的分钟退出。2026-09-27：与 Hermes 日收盘不一致的天数，2013–2021 段 161 条、2022+ 段 1 条。本次未重算。股票只数本次未重算（当时为 3,675 与 4,582）。

Tick、逐笔、十档、委托队列：在上述数据根目录没有对应数据文件。Hermes `source_schema.json` 的 611 张表名里也没有这类表。

---

## 6. 涨停微结构快照与生态缓存

这些是从 ClickHouse / v7 抽出来的研究缓存，只覆盖收盘涨停或市场日度，不是新的原始行情。

| 数据 | 行数 | 字段 |
|---|---|---|
| 微结构 2013-01-04 .. 2021-12-31 `daily_microstructure` | 140,240 | `h1_auction_volume_ratio, h1_consistent_buy_ratio, h1_consistent_sell_ratio, h1_last15m_volume_ratio, h4_smart_money_price, h4_volume_order_imbalance, h4_informed_buy_close_ratio, fb_touch_count, source_updated_at` |
| 微结构 2022-01-04 .. 2026-07-24 | 81,158 | 同上 |
| 市场涨停统计（两段） | 2,188 + 1,103 天（2022 段已核实 1,103 行） | `UP_NUM, PRE_UP_NUM, SUC_RATIO, CON_RATIO, FAIL_RATIO, UPLIMIT_CHG` |
| `first_board_market_limit_down_opening_…_v2/market_limit_down_context.parquet` | 3,291 | 收盘跌停家数、触及跌停、开盘跌停、5/20 日开盘跌停分位 |
| `first_board_true_board_ecology_…/board_ecology.parquet` | 9,736 | 日 × 板块：封板率、昨日涨停红盘率、昨日涨停晋级率、跌停压力 |
| `first_board_single_model_combined_…/factors.parquet` | 221,398 | 21 个分钟/行业封板因子，见列名于该文件 |

`fb_touch_count` 在微结构快照里来自供应商 `UP_LIMIT_REACHED_TIMES`（2026-09-27：中位数 12，最大 12,009）。一进二面板里的同名列是分钟触板段数（中位数 1）。两套数字不能混用。

源表 `mkt_limit_stats` 在 schema 里有 20 列，本地只抽了上面 6 个值字段。`DOWN_NUM`、炸板侧、成交额侧未落地。`mkt_equd_ind.CON_LIMIT`（连板天数）未下载，连板高度需要用 `daily_raw` 自行累加收盘涨停。

---

## 7. AmazingData 事件与 h5 缓存

### 7.1 已打开的 parquet

| 文件 | 行数 | 日期 | 字段 |
|---|---|---|---|
| `holder_num/all.parquet`（5,019,897 字节） | 507,743 | `ANN_DT` 19930112 .. 20260919；`HOLDER_ENDDATE` 19921231 .. 20260918 | `MARKET_CODE, ANN_DT, HOLDER_ENDDATE, HOLDER_NUM, HOLDER_TOTAL_NUM` |
| `block_trading/all.parquet`（5,273,102 字节） | 314,397 | `TRADE_DATE` 20020319 .. 20260918 | `MARKET_CODE, SECURITY_NAME, B_SHARE_PRICE, B_SHARE_VOLUME, B_FREQUENCY, B_SHARE_AMOUNT, B_BUYER_NAME, B_SELLER_NAME, BLOCK_AVG_VOLUME` |

低换手加载函数 `load_holder_num_announcements` 用公告日 `ANN_DT`，不用 `HOLDER_ENDDATE`。大宗金额单位本次未在注释里核实；加载代码路径见 `core/data/amazingdata.py`。

同目录还有几 KB 的 smoke parquet，不要当成全量。

`load_equity_structure` / `load_stock_basic` 读取 `amazingdata_raw/equity_structure/all.parquet` 与 `stock_basic/all.parquet`。这两个 parquet 本次不存在。股本事件应使用 v7 `share_capital`，或 `.ad_cache/infodata/equity_structure/equity_structure.h5`（1 个文件，70M 目录，未打开）。

### 7.2 未打开的 h5（`.ad_cache` 7.2G）

| 路径 | 文件数 | 体积量级 |
|---|---|---|
| `infodata/balance_sheet` | 5,463 | 1.2G |
| `infodata/cash_flow` | 5,461 | 2.0G |
| `infodata/Income` | 5,461 | 2.0G |
| `basedata/hist_stock_status` | 未逐个计数 | 993M |
| `basedata/hist_code_list` | 未逐个计数 | 1.2G |
| `infodata/index_constituent` | 4：`000016.SH.h5, 000300.SH.h5, 000852.SH.h5, 000905.SH.h5` | 524K |
| `infodata/industry_constituent` | 31 | 920K |
| `infodata/dividend` | 1 | 33M |
| `infodata/equity_structure` | 1 | 70M |
| `infodata/right_issue` | 未逐个计数 | 560K |
| `basedata/backward_factor` | 未逐个计数 | 6.8M |

公告日是否可靠：未核实。财务研究以 v7 的 `available_at` 为准。`.ad_stage/financials` 5.1G、日 K 批次合计约 0.3G（12,379,955 行是 2026-09-27 的计数，本次未重算）。

`/home/mxy/AmazingData/data/AmazingData_local_data` 另有 1.3G basedata/infodata，含 `holder_num.h5`、`block_trading.h5`。与 parquet 全量的关系未逐行核对。

---

## 8. 聚宽阶段数据

路径：`/data1/Stock/.jq_stage`，3.6G。

| 子目录 | 文件数 | 体积 | 本次核实 |
|---|---|---|---|
| get_price | 276 个 parquet | 393M | 合计 18,087,197 行。字段 `time, code, open, high, low, close, pre_close, volume, money, paused, high_limit, low_limit, factor`。`time` 2013-01-04 .. 2026-08-14 |
| get_industry | 3,306 | 3.1G | 未打开。2026-09-27 描述为逐日 `{sw_l1, sw_l2, sw_l3, zjw}`。历史日期返回的是当时分类还是事后新分类，仍未核实 |
| get_index_stocks | 9,918 个 pickle | 100M | 未打开。2026-09-27：沪深300 / 中证500 / 中证1000 的逐日成分列表，另有空列表 |
| get_extras | 6 | 2.8M | 未打开。2026-09-27：含 `is_st` |
| get_corporate_actions | 20 | 4.0M | 未打开 |
| get_all_securities | 2 | 224K | 未打开 |
| get_trade_days | 1 | 48K | 未打开 |
| get_privilege | — | 8K | 未打开 |

`get_price` 比 v7 多覆盖到 2026-08-14，仍早于 inbox 的 2026-09-30，也盖不住 2026-07-25 .. 2026-08-14 之外的缺口全段。它有 `factor` 与 `high_limit`，可做行情交叉校验。指数权重表 `idx_weight` 只在源库 schema（6 列），本地没有权重时间序列。

---

## 9. 指数、行业、北向、龙虎榜、文本

本地有价格或成分的指数：

- 沪深300 日线 `csi300.parquet`：4,785 行，2007-01-04 .. 2026-09-09。字段 `trade_date, open, high, low, close, pre_close, amount, volume`。
- 成分成员：聚宽逐日 pickle（未打开内容）；AD h5：上证50 `000016`、沪深300 `000300`、中证1000 `000852`、中证500 `000905`。h5 内部日期与 PIT 未核实。
- 上证综指、深证成指、创业板指、科创50 的独立日线文件，本次未找到。可以用全 A `daily_raw` 自建等权或市值加权。
- 源库 `mkt_idxd` 15 列，未下载。

行业：v7 只有申万一级两套分类，见 3.9。申万二级/三级只出现在聚宽 `get_industry` 的 2026-09-27 描述里，本次未打开 pickle。

北向持股明细：本地未找到，源库 schema 也没有独立 HK 持股表。2026-09-27：只有未下载的 `equ_fancy_factors_table3/4` 里的 `HK_HOLDRATIO_*` 派生列。

龙虎榜：本地未找到。源库 schema 有 `mkt_rank_list_stocks`（14 列）、`mkt_rank_list_sales`（16 列）、`mkt_rank_list_invs`、`mkt_rank_list_spec`。行数与日期未知（ClickHouse 需要密码）。

题材：本地未找到。源库有 `mkt_themed`（21 列，含涨停家数与主力净流入）、`md_sec_type`（10 列，含 INTO_DATE/OUT_DATE）。成员表是否回填，未核实。

新闻、公告全文、研报原文、分析师一致预期：本地未找到。源库没有全文表。派生热度/情感在 `equ_fancy_factors_table5/9`，未下载。

CNE6：`/data1/Stock/data/reference_factors/hermes_cne6_sw21_v1/as_of=2026-08-31` 存在，内部 Permission denied。不能当可用风险模型。

---

## 10. 已计算因子库

### 10.1 NShape `factors-pit-ex-st-v2`

- 路径：`/data1/Stock/.zyquant/cache/factors-pit-ex-st-v2/9dca5afe…/`
- 25G，350 个因子目录。抽样 `ret20`：一行一因子值，列 `trade_date, instrument_id, value`，12,770,392 行，2013-01-04 .. 2026-06-10。
- 名称前缀粗计（按目录名第一段）：`fund` 49、`market` 28、`nshape` 20、`close` 17、`volume` 16、`ret1` 16、`amount` 14，其余为波动、回撤、均线、估值、涨停记忆等。
- 2026-09-27：基于 `hermes-cn-a-2010-2026-mf`，剔除 ST 的 PIT 股票池，5,779 只。本次未重算股票数，也未逐个因子核对截止日。与 v7 混用时，2026-06-11 .. 2026-07-24 以及 mf 快照与 v7 的差异都要单独对齐。
- 同级 `cache/factors` 只有 4K，没有第二套因子体。

### 10.2 一进二因子缓存（`/data1/stock-mxy/data/factor_cache`，2.6G）

主要是首板样本上的 `fb_*`、`g_prim_*`、`h1_*`、`h4_*`、`h5_*`、`h6_*`、`h8_*`、`ind_*`，加上全市场 Hermes expansion/native。`g_prim_*` 的生产代码 2026-09-27 未在仓库里找到，本次未再搜。面板 `candidate_panel.parquet` 63,696 行、2013-01-07 .. 2026-07-01，特征与 `label_e2_gross/net`、`next1_open`、`next2_close` 等未来列在同一张表。训练时必须用特征白名单。

### 10.3 低换手因子面板

`课题/低换手策略1/result/YB/factor_panels` 3.0G，24 个 parquet，是策略因子结果，不是独立行情源。日频面板 `daily_panel.parquet`：

- 10,106,408 行，manifest 记录 5,447 只，2017-01-03 .. 2026-07-24，来源 v7，`exclude_bj` 后的沪深股票。
- 字段含 OHLCV、`float_mv, total_mv, turnover（amount/流通市值）, turnover_vendor, is_st, is_suspended, limit_up/down, open_limit_up/down, listing_days`。
- `sw_l1_v2021` 与 `industry` 的 parquet 类型是 `null`，整列没有值。行业中性不能读这两列，要回连 v7 `industry_membership` 或聚宽行业。
- manifest 里的 `path` 仍写着已废弃的 `/home/mxy/课题/低换手策略1/...`，实际文件在 stock-mxy 课题目录。

---

## 11. 源库里有 schema、本地没有行的数据

`source_schema.json`（v7-rebuild，4.7MB）：20,267 条列记录，611 张表。本地 raw 只有 29 张。ClickHouse `127.0.0.1:18123` ping 成功，`SELECT 1` 用 default 用户返回 `REQUIRED_PASSWORD`（版本 25.3.14.1）。`/data1/Stock/.env` 属 root 且权限 600，本次未读取、未尝试绕过。因此这 582 张未下载表的行数和日期都未知。

和选策略最相关的未下载表：

| 类别 | 代表表 | 本地 |
|---|---|---|
| 龙虎榜 | `mkt_rank_list_stocks/sales/invs/spec` | 无 |
| 题材 | `mkt_themed`、`md_sec_type` | 无 |
| 高频因子全市场 | `equ_h2l_factor_t1/t3/t4` | 只在涨停子集上有 7 个字段 |
| 市场涨停统计全字段 | `mkt_limit_stats`（20 列） | 6 个值字段 |
| 连板 | `mkt_equd_ind` | 无 |
| 指数行情与权重 | `mkt_idxd`、`idx_weight`、`idx_cons*` | 无（仅有沪深300价格和聚宽/AD 成分） |
| 业绩预告/快报 | `fdmt_ef_new`（35 列）、`fdmt_ee` | 无 |
| 解禁、增减持、回购、质押、定增 | `equ_shares_float` 等 | 无 |
| 前十大股东、机构持仓 | `equ_sh_ten`、`equ_inst_hold` | 无 |
| 股东户数源表 | `equ_shareholder_num` | 无；本地用的是 AmazingData parquet |
| 另类因子（北向比例、新闻条数、研报情感） | `equ_fancy_factors_table1..10` | 无 |
| 风险模型 | `dy1d_*_cne6_sw21` 等 | 本地目录不可读 |

---

## 12. 课题与回测产出

这些目录是研究结果，依赖上面的行情，本身不提供新的原始字段。

| 课题 | 路径 | 体积 | 依赖的数据 | 本次看到的结果目录 |
|---|---|---|---|---|
| 低换手 | `/data1/stock-mxy/课题/低换手策略1` | 31G | v7 日线+估值+ST+行业；AD 分钟（2017–2021）；股东户数；大宗。面板行业列为空 | `result/YB` 24G（含分钟拷贝 20G）、round3–7、`hermes_full_rerun` 204M。README 当前基线代号 `YB_04.1b` |
| 一进二（完整树） | `/data1/stock-mxy/课题/一进二策略` | 1.5G | v7、收盘涨停分钟、微结构快照、首板面板、`runs/first_board_factor_search_5000_v1` | result 含 round1–9、replay |
| 一进二（家目录另一份） | `/home/mxy/课题/一进二策略` | 1.5G | 同上 | inode 不同。result 有 round1–6、round8、replay，没有 round7、round9 目录 |
| 一进二运行产物 | `/data1/stock-mxy/runs` | 23G | 同上 | 最大项 `first_board_factor_search_5000_v1` 22G |
| Whitehorse | `/data1/Stock/data/whitehorse` | 3.5G | 沪深300 + 自建面板 | `v47_ml` 1.2G、`v47_rules` 2.4G、`csi300.parquet`。面板列本次未展开 |
| NShape 研究副本 | `/data1/Stock/data/nshape` | 1.1G | mf 快照因子 | `pit_ex_st_v2` 462M、`pit_ex_st_v3_hermes` 463M、`research` 200M |
| nshape 实盘缓存 | `/data1/Stock/data_live/nshape` | 5.3G | 因子缓存 5.2G | `features` 51M、`paper_nav`、`signals`。未打开 |
| ETF 轮动回测 | `/data1/xj/ETF轮动/backtests` | 21G | 未知（目录内是回测名，如 XGBRanker） | 未见行情子目录 |
| `/data1/Stock/runs` | 34G | 历史实验 | 未逐个打开 |
| `/data1/Stock/logs` | 49G | — | 日志 |

`docs_sun` 下是周报 markdown，不是行情。

---

## 13. 数据缺口与策略可行性

图例：能做 = 现有字段足够搭一版日频或事件研究；受限 = 能做一段样本或必须接受时点/覆盖缺陷；做不到 = 缺关键字段。

| 策略类型 | 判定 | 依赖的现有数据 | 缺口 |
|---|---|---|---|
| 全 A 日频多因子（价值、质量、规模、动量、低波） | 能做，样本截止 2026-07-24 | `daily_raw` + `daily_post_adjusted` + `daily_valuation` + `fundamental_metrics` + 交易日历 + ST | 无一致预期、无业绩预告；PE/PB 的 TTM 切换时点不可从 `available_at` 证明；费率表为空；2026-07-25 之后未并入快照 |
| 低换手 / 换手率结构 | 能做日频；日内换手只到 2021 | 面板 `turnover` 与 `turnover_vendor`；v7 `turnover_rate`；t2 分时换手（2011+，数值像变换得分） | 面板行业列为空；2022+ 没有全市场分钟，做不了「加速换手/UTD」原文那种日内换手；`equity_structure/all.parquet` 不存在 |
| 行业中性、行业轮动（申万一级） | 受限 | v7 `SW_L1:V2021` 与旧版 `SW_L1:SW` | V2021 生效日回填到 1996；没有 Hermes 的申万二/三级；聚宽 L2/L3 未打开、历史分类时点未核实 |
| 指数增强（沪深300 / 中证500 / 中证1000） | 受限 | 聚宽逐日成分（未打开内容）、AD 四个 h5、沪深300 价格到 2026-09-09、个股日线到 2026-07-24 | 无指数权重；无其他宽基的完整本地行情；成分 h5 未核对 PIT |
| 资金流 / 大单 | 能做日频，2014 起 | v7 资金流 + raw 各档笔数 | 分档是订单金额，不是机构席位；2021 年前 `available_at` 回填；2026-07-24 之后增量批次里有 `mkt_equ_mf_new`，未并入研究表 |
| 融资余额、融资买入 | 受限 | `daily_margin` | 只覆盖两融标的；D 日信号要用 D−1（2026-09-27 时点）；增量 EOD 批次不含 `fst_detail` |
| 一进二 / 首板 / 打板 | 能做日频进出；分钟特征仅限 D 日涨停股 | 涨跌停事件、`daily_raw` 次日开盘、收盘涨停分钟、微结构 7 列、生态缓存、首板面板 | 无龙虎榜、无题材成分、无 tick；2022+ 不能用全市场分钟做卖出路径；连板天数要自己算；面板标签列在同一文件 |
| 跌停、情绪、连板高度 | 能做日频 | 涨跌停事件的 down 行、跌停生态缓存、板块生态 9,736 行 | 源表涨停统计的跌停家数、成交额侧未下载；`CON_LIMIT` 未下载 |
| 日内择时、T+0、分钟退出 | 受限在 2017–2021 | 全市场 1 分钟 | 2022–2024 每年约 10 只；无 2025–2026 全市场分钟；无 tick/L2；集合竞价独立 bar 未找到 |
| 事件：分红送转、ST、停牌、大宗、股东户数 | 能做 | 公司行为、ST、`paused`、`md_sec_halt`、大宗、户数 | 解禁、增减持、回购、质押、定增、业绩预告未落地 |
| 高频做市、盘口、逐笔 | 做不到 | — | 无 tick、无十档。仅有供应商日频高频因子和涨停子集上的 7 个代理字段 |
| 北向、龙虎榜席位 | 做不到（本地） | — | 需从源库补抽；当前账号读不到 ClickHouse |
| 新闻舆情、分析师预期 | 做不到 | — | 无全文、无一致预期表 |
| ETF 轮动 | 做不到（就现有行情库而言） | `/data1/xj` 只有 21G 回测 | 未找到 ETF 日线/分钟库 |
| 实盘当日信号 | 受限 | 增量批次最晚文件名 2026-09-30 | 未并入 v7；缺日内因子、两融、涨跌停事件；2026-07-27 .. 2026-08-28 无批次；`releases` 未确认有内容 |

---

## 14. 基于现有数据可开展的策略方向

下面只按已经落地的字段判断。每一类都写清用哪些表、缺什么就停。

1. **沪深日频低换手 / 反转（现有课题的数据条件是齐的）**  
   用 `daily_panel`（2017-01-03 .. 2026-07-24，已剔北交所）或直接用 v7 `daily_raw` + `daily_valuation`。换手有两个口径：`amount / circulating_market_cap` 与供应商 `turnover_rate`。ST 用 `special_treatment` 或面板 `is_st`，停牌用 `paused`，涨跌停用 `limit_up/down`。行业中性要另接 `industry_membership`，因为面板行业列是空类型。样本外若要越过 2026-07-24，得自己拼接 inbox 的 `mkt_equd`，并处理 7 月 27 日到 8 月 28 日的空窗。全市场日内换手因子只能做到 2021 年底。

2. **质量、价值、成长的日频选股**  
   用 `fundamental_metrics` 的 31 个指标，过滤 `available_at`。估值用 `daily_valuation` 的 PE/PB/PS/股息率，规模用自由流通市值。财报科目细项用 `financial_facts` 或 raw 宽表。做不到预期驱动（预告、快报、一致预期都没下载）。估值 TTM 与公告日的对齐如果要严格，需自建，不能相信 `daily_valuation.available_at`。

3. **日频动量、波动、回撤**  
   收益用 `daily_post_adjusted`，成交约束用 `daily_raw`。NShape 的 `ret*`、`volatility*`、`drawup*`、`pullback*` 已算到 2026-06-10（至少 `ret20` 如此），底库是 mf 快照，和 v7 混用前要核对。无分钟数据时，隔夜跳空只能用次日 `open/pre_close`。

4. **资金流与两融情绪**  
   2014 起可做大单/超大单净流入、开盘与尾盘净流入、流入占成交额。raw 笔数可做单笔金额。两融用 `financing_buy_value`、余额变化，信号日取前一交易日，股票池随标的扩容（2025 年约三千多只，该只数来自 2026-09-27）。不能把「主力」写成机构行为。

5. **一进二：D 日收盘打分，D+1 开盘买入，日线卖出**  
   数据够做这一条执行路径：涨跌停事件（封板时间、封单额）、收盘涨停分钟（只覆盖 D 日）、`daily_raw` 的次日开盘与涨跌停、板块生态与跌停家数。首板面板已经放到 2026-07-01。卖出若依赖 D+1..D+3 的分钟路径，只有 2017–2021 能做，2022 之后只能用日线规则（开盘价、收盘价、是否一字）。龙虎榜、题材联动、tick 级封单衰减做不到。

6. **市场状态门控（给 1 和 5 用）**  
   已有市场上涨比例类缓存、跌停家数、分板块封板率与昨日涨停晋级率、微结构快照里的封板率/连板率/炸板率，覆盖到 2026-07-24 附近（生态与跌停上下文 3,291 天）。涨停统计全字段里的跌停家数和两市成交额未下载，要更全的情绪表需要补抽或用 `daily_raw` 自己数。

7. **事件驱动：高送转、分红、ST 摘帽、复牌、大宗、户数变化**  
   分红送转和配股在 `corporate_actions`（除权日+公告日）。ST 区间在 `special_treatment`。复牌用 `paused` 或 raw `md_sec_halt`。大宗与户数 parquet 更新到 2026-09。解禁与股东增减持没有本地表，这类策略做不到。

8. **指数成分内的日频策略**  
   可以限制在沪深300、中证500、中证1000，前提是先把聚宽 pickle 或 AD h5 读开并核对历史成分。权重优化、行业内相对基准的精确超额，缺权重和缺多数指数行情，做不到标准的指数增强组合优化。沪深300 本身的价格可以当基准，到 2026-09-09。

明确停掉的方向：Tick/L2 策略、2022 年后的全市场分钟策略、ETF 轮动（没有 ETF 行情库）、北向持股变化、龙虎榜席位、新闻文本、分析师一致预期、直接读取 CNE6。这些要么文件不存在，要么目录不可读，要么只在需要密码的源库里。

---

## 15. 盘点方法与局限

本次打开并核对过的：

- `df`、顶层 `du`、目录列举。
- v7 全部 19 张表的 parquet schema、行数、日期列 min/max（来自 parquet statistics）。
- `daily_raw` 停牌行数，以及 2014/2018/2020/2025 的股票去重数。
- `instruments`、`industry_membership`、`corporate_actions`、`trade_calendar`、`special_treatment`、`market_rules` 的全表分布或全表内容。
- `fundamental_metrics.metric_code` 全表去重（31 个）。
- 分钟线按年文件数与两个股票文件的 schema、行数、时间范围。
- 股东户数、大宗、沪深300、两段涨停分钟、两段微结构、跌停上下文、板块生态、首板面板、低换手面板、NShape `ret20`、聚宽 `get_price` 全部 276 个文件的行数与日期统计。
- 增量批次目录名，以及 2026-09-30 的 `mkt_equd` 一个文件。
- `source_schema.json` 的表数、列数，以及 `mkt_equd`、`mkt_equ_mf_new`、`mkt_limit_ind`、`fst_detail`、`equ_h2l_factor_t2`、`mkt_equd_adj_af` 的中文注释。
- ClickHouse `/ping` 与一次无密码查询。

未打开或打开失败：

- ClickHouse 业务表（default 用户认证失败）。`.env` 未读。
- `.ad_cache` 与 AmazingData 本地 h5（财报、状态、指数成分、行业成分、股本）。
- 聚宽 `get_industry`、`get_index_stocks`、`get_extras` 等 pickle。
- CNE6 `as_of=2026-08-31`、`warehouse/datasets/cn-a-jq-2013-2026-r3`、`/data1/stock-xzq`。
- `/data2/Ljf/quant/data/min5` 等 600 权限文件；`Ljf2` 内容未抽样。
- `data_live/releases`、`server_relay` 内部文件、`state.sqlite`。
- NShape 其余 349 个因子的日期（只抽了 `ret20`）。
- Whitehorse `v47_*` 面板列、`/data1/xj` 回测内部、`/data1/Stock/logs` 与大部分 `runs`。
- 2026-09-27 质量项里未在本节「打开并核对」中重做的部分：资金流恒等式、日内因子超界比例、`limit_up` 哨兵行数、涨停事件与收盘价的 565 天差异、`available_at` 滞后分布、聚宽行业的历史时点。

和 2026-09-27 审计不一致、以本次为准的事实：股东户数 507,743 行（当时 507,745）；大宗 314,397 行（当时 314,399）；低换手分钟目录与 AmazingData raw 抽查文件不是同一 inode；`/data1/xj` 有 21G ETF 回测；增量 EOD 批次已存在至 2026-09-30；家目录与 `stock-mxy` 各有一份一进二课题树。
