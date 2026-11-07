# 源码完备性报告

## 总评
**SOURCE_WITH_EXTERNAL_DATA_DEPENDENCY**

分支内：训练/融合/B03/O01/local_executor/parity 完整。  
外部：Stock `first_board_hermes` 因子构建 + Hermes/AmazingData/CH 数据。`source_refs/` 含关键 builder 审计副本。

| 组件 | 状态 |
|---|---|
| round7 contract / fold_ranker / research_sleeve / local_executor | COMPLETE_SOURCE |
| fuse/B03/O01/parity | COMPLETE_SOURCE |
| 分钟/日频/资金流/行业公式实现 | SOURCE_WITH_EXTERNAL_DATA_DEPENDENCY |
| 供应商 PCV/BCVP 内部公式 | INCOMPLETE_SOURCE（仅列映射） |
| g_prim_px_vol_corr 本地公式 | INCOMPLETE_SOURCE（仅缓存列） |

关键训练/回测链 **不是** INCOMPLETE_SOURCE；完整因子重算依赖外部数据且部分供应商公式未展开。
