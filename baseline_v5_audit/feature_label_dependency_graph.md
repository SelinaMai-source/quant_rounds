# 特征 / 标签依赖图

```
Hermes v7 daily_raw/post_adj/money_flow/limit_events/industry
AmazingData minute + ClickHouse micro (PCV/BCVP)
        │
        ▼
factor caches / single_model assets
        │
        ├─ Five14 features ──► XGBRanker(five) ──► label_e2_net (mature: next2_date)
        │                         │
        └─ Ten13 features ──► XGBRanker(ten) ──► label_exit_rule_net (mature: actual_exit_date)
                                  │
                                  ▼
                     fuse 90/10 (+ d1_open_gap filter)
                                  │
                                  ▼
                     clean_risk → B03 → O01 → local_executor → NAV
```

关键边：
- `d1_open_gap` 依赖 D+1 daily_raw（OPEN_INFORMATION）
- Ten 标签依赖退出规则路径，成熟晚于 Five 的 next2_date
- O01 依赖 PUT_SKEW25 日频（options 因子，非 Five/Ten 特征）
