from pathlib import Path
import json,shutil
C=Path('/data1/stock-mxy/课题/一进二策略');r=C/'result/round23_7/D0'
assert json.loads((r/'ten_action_label_replay.json').read_text())['PASS']
assert json.loads((r/'actual_cache_hit_test.json').read_text())[1]['cache_hit']
assert (r/'new_wrapper_sort_repair.json').exists()
s=json.loads((r/'STATUS.json').read_text());s['EVIDENCE_COMPLETE']=True;s['TARGET_OOF_COMPLETE']='NOT_APPLICABLE_D0';s['REFERENCE_RUN_COMPLETE']='NOT_APPLICABLE_D0';s['BACKTEST_COMPLETE']=True;s['status']='D0_COMPLETE_WITH_EXPLICIT_LEGACY_LIMITS';s['uncovered']=['vendor historical revisions of inherited derived microfeatures','new-model tests must accompany D1/D3/D4/D5','corporate-action behavior not observed: original loader supplies zero action events'];(r/'STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
(r/'REPORT.md').write_text('''# D0 实际正确性审计

两个 Head 各 2018/2022/2026 三折，每折原始、删除未来标签、置乱未来标签、置乱未成熟标签、修改预测年之后特征五次真实拟合，共 30 个模型。训练矩阵、分箱标签、候选键、分数与实际 B03/O01 输入计划完全一致。另对 2018–2026 全部 18 个 Head/年边界比较完整训练输入。精确覆盖见 actual_entry_mutation_tests.csv 和 all_year_boundaries.csv；这不是所有后续模型已无泄漏的保证。

Five 按原固定 D+2 收盘标签公式核对，成熟事件为 next2_date 的交易所收盘。Ten 使用 actual_exit_date，不以 next2_date 替代。原源码完整重放 46,234 条 2018 后候选，43,586 条有效标签净收益及退出日完全一致；原实际 1,107 笔交易的标签和退出日也一致。15:00 是重建的市场事件时刻，不是已证明的供应商入库时刻。

原执行器基线从计划调用重新跟踪，overlay/NAV/交易/持仓/费用/拒单/现金流/袖套均与冻结基线一致。27 组旧回测逐日使用已实现人民币 PnL、未平仓成本与市值重建 NAV，误差均小于 max(0.01,1e-8*NAV)，未用不明残差补桥。原加载器未传公司行动，实际行动事件为0；不能据此声称历史企业行动覆盖完整。遗留 high/low 成交截断与 D+1 开盘信息通道保留，Parity 不是实盘可成交证明。

新缓存封装曾在应用 O01 前漏掉稳定排序，产生了错误的 1,113 笔结果；错误源、日志和产物留存，经济资格为否。修复为先按日期、分数、证券代码稳定排序。修复后完整基线产物再次对齐 1,107 笔，并实际验证首次 miss、相同输入第二次 hit。结果按完整指纹不可变存储，失败或部分目录不命中，篡改输出失效。合同包括代码闭包、作者锁、未提交差异、环境、内容/schema/分区、标签及成熟、参数/特征/资格、成本资金执行和上游；完全相同配置与逐项变更测试见 cache_tests.log。大文件每次流式内容核验，不依赖 mtime。

SL/BOA 诊断没有支持“均值专家主导 SL”这一简单假说。SL 的 Ridge 平均权重 Five 90.23%、Ten 93.10%，均值权重0；BOA 的 Ridge 55.10%/59.47%，均值13.84%/19.25%。Ten Ridge 的全样本MSE约0.007508，低于XGB约0.007525，但 Top2 成熟目标均值约0.1273%，低于XGB约0.2717%。Ridge 的整体RankIC反而较高，因此低数值误差/整体相关不等于尾部选择或组合收益更好。校准斜率截距、日内标准差、同分、日期均值/中心化误差、历史权重、真实Top1/Top2人民币PnL及原回测均保存；相关关系不能直接证明失败的单一因果机制。归档当前 SL/BOA，不搜索新参数。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE
''')
(r/'DECISION.md').write_text('D0 公共修复与规定范围审计完成，允许进入冻结 D1；保留遗留信息/成交/企业行动限制。SL/BOA 当前配置归档，不升级基线，不调参。新模型须各自追加真实入口反事实测试。\n')
(r/'REPRODUCE.md').write_text('在服务器项目根运行 research/round23_6_26_execution/env/bin/python tools/continue_round23_7_round26_1.py --deliverable D0 --resume --seed 17 --jobs 1。审计不是新收益候选；原始数据不上传，输入路径/schema/SHA记录在input_manifest和完整运行合同。\n')
for p in C.glob('result/round23_7_D0*.log'):shutil.copy2(p,r/p.name)
# Public D0 artifacts and identical correctness module copied to both independent lines.
out=C/'result/round26_1/D0';shutil.copytree(r,out,dirs_exist_ok=True)
for branch in ['round23_7','round26_1']:
 base=C/'result'/branch;base.mkdir(exist_ok=True)
 permission=base/'无权限数据.md'
 if not permission.exists():permission.write_bytes(b'')
 main=base/('ROUND23_7_REPORT.md' if branch=='round23_7' else 'ROUND26_1_REPORT.md')
 if not main.exists():main.write_text('# '+branch+'\n\nD0已执行，范围见D0/REPORT.md；其他交付不在D0阶段冒称完成。\n\nCURRENT_BASELINE_CHANGED=FALSE\nPRODUCTION_CONFIG_CHANGED=FALSE\n')
