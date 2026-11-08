# Reproduce round24_v5_migration_v2

工作根：`/data1/stock-mxy/课题/一进二策略`

```bash
python tools/run_round24_v5_migration_v2.py --jobs 2
```

阻断：资金流和涨停事件的 available_at 是回填；equ_h2l_factor_t1 缺失且无凭证；g_prim 原列不在本地。
