# 复现 baseline_v5 审计与 parity

## 环境
- 发布树：`/data1/stock-mxy/research_publish/wt_baseline_v5`
- 课题根：`/data1/stock-mxy/课题/一进二策略`
- `PYTHONPATH` 含 `/data1/stock-mxy/src`
- 外部数据见 `data_catalog/`

## 核验冻结数字（不重训）
```bash
python -c "import json; print(json.load(open('result/baseline_v5/frozen_identity.json')))"
# 对比 baseline_v5_audit/baseline_metrics.json
```

## 阅读顺序
1. feature_order / factor_registry / factor_formulas
2. model/label/fold contracts
3. execution / B03 / cost
4. data_catalog
5. SECOND_TO_THIRD_FACTOR_MIGRATION_PLAN
6. parity_report
