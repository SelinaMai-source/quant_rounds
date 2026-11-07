# B03 / O01 / clean_risk 契约

## clean_risk
竞价/风格健康度缩放，乘数 ∈ {0.25, 0.375, 0.5, 0.75, 1.0}（见 round9/round8 风险层）。应用于融合分之后、B03 之前的份额缩放链。

## B03 九格（Top2 NAV 份额）
来源 `core/baseline_v5/contract.py::B03_CELL` / `core/round9`：

| agreement \ separation | LOW | MID | HIGH |
|---|---:|---:|---:|
| A | 0.00 | 0.05 | 0.10 |
| B | 0.00 | 0.00 | 0.05 |
| C | 0.00 | 0.00 | 0.00 |

## O01
`PUT_SKEW25.shift(1) > rolling(252, min_periods=126).quantile(0.90) → apply_top2_veto`：信号日将 **rank=2** 置 `MIN_KEEP≈1e-12`，**Top1 保留**。

信号日清单：`result/round16/options/treat_dates.json`（本分支已有相关产物指针）。
