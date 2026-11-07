# baseline_v5 数据源目录（审计摘录）

完整服务器清单见 `/data1/stock-mxy/数据清单.md`（2026-10-04）与 `result/data_inventory/`。  
**本目录不上传供应商原始 parquet 全量**；仅 schema / 路径 / 哈希 / 覆盖统计。

## 核心依赖

| 域 | 路径 | 用途 |
|---|---|---|
| Hermes v7 | `/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features` | 日线/资金流/涨停事件/行业/ST/日历 |
| Five 面板 | `/data1/stock-mxy/data/first_board_hermes/aligned_original_candidate_full44_plus_d1_open_gap_v1` | Five14 + d1_open_gap + 标签 |
| Ten 面板 | `/data1/stock-mxy/data/first_board_hermes/aligned_original_candidate_fs01122_chain_label_v1` | Ten13 + exit 标签 |
| 分钟 | AmazingData kline_min1 + close-limit 快照 | fb_* / g_prim_* |
| 微观 | ClickHouse equ_h2l_factor_t1 抽取 | h1_* PCV/BCVP |
| 二板宇宙 | `/home/mxy/Stock/课题/一进二策略/result/round24/data/second_board.parquet` | Round24 已有 2to3 候选（2015-01-05..2026-07-24，n=29701） |

## 机器可读
- `table_schema.csv` — 列级 schema
- `data_source_manifest.csv` — 源清单
- `data_availability_contract.md` — 可得性/PIT
- `data_coverage_by_year.csv` — 年频覆盖
- `data_catalog_server_full.csv` — 服务器 inventory 副本（非原始行情）
