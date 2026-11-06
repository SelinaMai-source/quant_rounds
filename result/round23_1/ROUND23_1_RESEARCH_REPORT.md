# Round23.1 研究报告

决策：**PREDICTIVE_ONLY**。O4 是正式主目标。O1 的 Spearman 只作为失败对照，Ten=0 不是正式候选，也不重新晋级。

Five/Ten 诊断：{'daily_spearman_five_ten': 0.5830477457333747, 'top1_disagreement': 0.6732186732186732, 'n_days': 2035, 'disagreement_days_five_exit_mean': 0.0006245250771621598, 'disagreement_days_ten_exit_mean': 0.006004745579407837, 'disagreement_days_five_e2_mean': -0.00021566154233019206, 'disagreement_days_ten_e2_mean': 0.0012374832391475927, 'note': '分歧日的实现标签是诊断，不用来事后改权重'}

内折权重：{'2018': 0.9, '2019': 0.75, '2020': 0.75, '2021': 0.9, '2022': 1.0, '2023': 0.9, '2024': 0.75, '2025': 0.5, '2026': 0.75}。与 0.90 完全相同：False。

LightGBM 多样性：Spearman=0.8405585121740321，Top1 不一致=0.39803439803439805，结论=DIVERSITY_PASS。未过门则不融合。融合用日度 rank 百分位，不用原始分数。

规则法医没有重调阈值。R2 笔数上升来自过滤后的替补或换仓。R3 是执行敏感性。

自助与 SPA：{'bootstrap': {'R231_O4_BLEND': {'status': 'OK', 'n': 2075, 'block': 20, 'n_boot': 5000, 'mean_excess': -0.00028725938777850104, 'ci95_low': -0.0007188475142652538, 'ci95_high': 0.00012507988244646402, 'share_boot_mean_gt_0': 0.0854, 'interpretation': 'block bootstrap interval, not a posterior'}, 'R231_L1_LGBM': {'status': 'OK', 'n': 2075, 'block': 20, 'n_boot': 5000, 'mean_excess': -0.0011377411902433396, 'ci95_low': -0.0017711154972002608, 'ci95_high': -0.0005024183971294242, 'share_boot_mean_gt_0': 0.0, 'interpretation': 'block bootstrap interval, not a posterior'}, 'R231_L2_50': {'status': 'OK', 'n': 2075, 'block': 20, 'n_boot': 5000, 'mean_excess': -0.00022465085891921528, 'ci95_low': -0.000586551670740536, 'ci95_high': 0.00014595198289350464, 'share_boot_mean_gt_0': 0.1184, 'interpretation': 'block bootstrap interval, not a posterior'}, 'R231_L3_INNER': {'status': 'OK', 'n': 2075, 'block': 20, 'n_boot': 5000, 'mean_excess': -0.00020063964946955774, 'ci95_low': -0.0005383605825921853, 'ci95_high': 0.000140218083323956, 'share_boot_mean_gt_0': 0.1284, 'interpretation': 'block bootstrap interval, not a posterior'}}, 'spa': {'status': 'OK', 'method': 'WHITE_REALITY_CHECK_BLOCK_BOOT', 'not_a_posterior': True, 'baseline_excluded': 'v5', 'candidates': ['R231_O4_BLEND', 'R231_L1_LGBM', 'R231_L2_50', 'R231_L3_INNER'], 'best_candidate': 'R231_L2_50', 'sample_stat': -1.076238888424038, 'p_value': 0.9928, 'n': 2075, 'block': 20, 'n_boot': 5000, 'used_labels': False}, 'bootstrap_is_posterior': False}。区间不是后验。

```
                  id result
H231_O1_NOT_PROMOTED   PASS
       H231_O4_BLEND   FAIL
  H231_LGBM_ENSEMBLE   FAIL
H231_RULES_EXPLAINED   PASS
```

没有独立 Forward。LIVE_READY=false。CURRENT 未改。
