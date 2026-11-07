# ROUND23.3 + ROUND24 V5 迁移总报告

## 冻结标志

- CURRENT_BASELINE_CHANGED=FALSE
- PRODUCTION_CONFIG_CHANGED=FALSE
- AUTO_PROMOTE=FALSE

## 源分支

| 分支 | SHA |
|---|---|
| baseline_v5 | `feda6870a26f6efe5d722fd9c3b11130ca266d85` |
| round22_6 | `89311afafab3c257c1daf41eaedac2d14bfdf958` |
| round23_2 | `37b65c56a2ca31dacc640cef7cbbe1b420ad2613` |
| round24_2 | `6581d3a245fcee5af8a6d7b6676a92950a8569a9` |

目标：`round23_3`（父=round23_2）、`round24_v5_migration`（父=baseline_v5，选择性引入 round24_2 模块）。

## Round23.3

- Parity：**PASS**（E1_T090 / E2_L000 与 v5 Sharpe=1.8756627913361097、CAGR=0.5499812832273525、NAV=12750089.355724689、笔数=1107 一致）
- Inner 窗修复：**PASS**（半年块约 119 NAV 日，不再出现全历史 2076 日）
- 默认权重逻辑已修；单元测试非默认可胜出
- E1：固定权重均未过经济门；NESTED 全折回退默认 t_0.90
- E2：LGBM 权重增大后变差；NESTED ΔSharpe≈-0.01，未过经济门
- E3：**NOT_TRIGGERED**（无条件增量）
- E4：SCORE_ONLY / DISAGREEMENT 均弱于 BASE
- 决策：**KEEP_BASELINE_V5**
- 详见 `result/round23_3/ROUND23_3_RESEARCH_REPORT.md`

## Round24 V5→二进三迁移

- 二板候选：29701（2015–2026-07-24）
- 特征集已冻结：CORE3 / TARGET7 / V5_MIGRATED（排除 d1_open_gap；g3 新 ID）
- V5DIRECT：**BLOCKED_FEATURE_SEMANTICS**
- 标签对账：**FAIL_PARITY**（可比样本 mismatch_rate≈0.116）→ **promotion_blocked=true**（不编造 PASS）
- 矩阵最佳匹配：**V5T** Sharpe=0.5767 CAGR=0.1246 MaxDD=-0.4983 NAV=819416（仅二进三线内比较，不与一进二 v5 排名）
- Forward：`FORWARD_OOS_NOT_AVAILABLE` → **SECOND_TO_THIRD_RESEARCH_CANDIDATE**，非 LIVE_READY
- 详见 `result/round24_v5_migration/ROUND24_MIGRATION_RESEARCH_REPORT.md`

## 路径

- `/data1/stock-mxy/课题/一进二策略/result/round23_3/`
- `/data1/stock-mxy/课题/一进二策略/result/round24_v5_migration/`
- `/data1/stock-mxy/课题/一进二策略/result/round23_3_round24_master/`
