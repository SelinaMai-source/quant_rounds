from pathlib import Path
import json,shutil,hashlib
import pandas as pd
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_7/D1'
assert json.loads((R/'cache_provenance_repair.json').read_text())['all_NAV_and_trades_identical_to_prior_runs']
results=pd.read_csv(R/'risk_results.csv');results.to_csv(R/'all_results.csv',index=False)
# Final raw score lineage resolves to a model and a real next trading-day decision.
fits=pd.read_csv(R/'annual_fits.csv');d=pd.read_parquet(C/'result/round22_3/data/ten_candidate.parquet',columns=['trade_date','instrument_id','next1_date']).rename(columns={'trade_date':'signal_date'});d['prediction_at']=pd.to_datetime(d.next1_date)+pd.Timedelta(hours=9,minutes=30);registry=[]
for method in ['N1','N2']:
 pred=pd.read_parquet(R/(method+'_predictions.parquet'));pred=pred.merge(d[['signal_date','instrument_id','prediction_at']],on=['signal_date','instrument_id'],validate='one_to_one');pred['outer_year']=pred.signal_date.dt.year;meta=fits.loc[fits.method.eq(method),['year','fit_cutoff','max_pair_available_at']].rename(columns={'year':'outer_year'});pred=pred.merge(meta,on='outer_year',validate='many_to_one');pred['method']=method;pred['model_hash']=pred.outer_year.map(lambda y:hashlib.sha256((R/f'models/{method}_{y}.json').read_bytes()).hexdigest());assert pred.prediction_at.notna().all();registry.append(pred)
pd.concat(registry).to_parquet(R/'prediction_registry.parquet',index=False)
for p in C.glob('result/round23_7_D1*.log'):shutil.copy2(p,R/p.name)
main=results.loc[results.scenario.eq('S0')];table=main[['experiment_id','CAGR','Sharpe','MaxDD','UI','trades']].to_csv(index=False)
(R/'REPORT.md').write_text('''# D1 AnchorRank：REJECT_CURRENT_CONFIGURATION

真实运行已完成。N0、N-U、N-R 各执行 S0/S1/S2，共9组正式历史回测；所有代价均在订单层增加买入滑点，S3原计划封涨停限制已生效，不复制成绩。2018–2026属于反复使用的开发期，不称独立测试或实盘证明。

主成绩（收益为小数）：

```csv
'''+table+'''
```

N-U 与 N-R 均未提高收益或 Sharpe，未通过冻结收益经济门；不选择最优年份、seed或新网格。MaxDD有小幅改善，单独作为风险诊断，不能据此宣布基线替代成功。

输入固定四个原专家。2018后原v5 XGB逐条对应，LGBM使用原round23_3截断4/5的冻结预测。2016/2017预热恢复4个LGBM模型，并验证既有XGB模型文件hash、训练截止与成熟，XGB旧bank逐日排名在2018后全量与v5完全一致。2015旧bank不具有完整三年特征历史，未拿它填补训练；2018拟合只在过去三年窗口中使用真实已恢复的2016/2017，已有461个成熟配对日期、2904个配对，超过冻结最低门。完整逐年覆盖和fallback见annual_fits与active_and_fallback_period_results。

原SL/BOA bank的LGBM与旧round23_3 LGBM排序不一致，未直接把它冒充原专家；预热采用旧作者参数/float32调用约定重训，Outer始终使用原冻结预测。未在同一挖掘训练集做回填OOF。

同日固定v5 Top5后，Top1/Top2与其余股票构造去重无序配对，目标统一用D0验证的Ten执行行动，未成交为明确UNFILLED_ACTION，不把未知退出填0。pair_available_at取两者实际成熟最大值，年度只用截止前成熟对。温度0.1、正则0.1、beta上限0.25、收益差95%截断均冻结。两个内层半年另外做真正时间外诊断，不用于挑参或选择Outer权重；训练内半年度损失明确另表保留，不冒称验证。

18个主年度两参数模型均收敛，解析梯度通过合成及真实配对有限差分，目标/KKT/边界逐项保存。两次前经济测试发现数值问题：常数平移造成约1.11e-10系数差，以及少数并列股票次序变化。失败日志保留；在看任何经济结果前提交数值修正规约，系数canonicalize到8位小数并重核KKT。最终常数平移后的系数/分数/股票顺序完全相同；2018/2022/2026两个模型的未来与未成熟标签扰动，预测和B03/O01计划完全一致；beta=0的整个执行产物精确回v5。常数专家仅在固定四输入之外无通道进入分数，不增设均值专家。

逐日配对错误、纠正/新增错误、真实改选与事后已知行动Top2机会损失已保存。事后上界只做诊断，不参与选择。实际交易Top1/Top2净人民币PnL、费用、换手、平均敞口、持仓广度、年度收益、最差63日、ES95、UI和NAV桥接，以及配对Bootstrap20日5000次/10和40日各1000次、锁定arch SPA均来自真实组合收益。全部同scope尝试纳入SPA，去重相同收益路径，不混参考数据结果。

缓存审计补充发现：仅绑定分数和市场数据仍不足，必须同时绑定原始训练/候选的特征、标签和成熟列内容。补齐完整上游面板身份后，9组回测全部重放，NAV和交易与先前9组逐字节相同。原9组、旧合同和元数据问题均保留；当前9组为完整指纹版本，重复验证不是新增模型搜索。实际总经济回测执行18次，唯一配置9组。预热4个唯一模型因两次诊断失败共实际拟合12次；主Outer参数18个，另有36个真正内层诊断拟合，反事实优化调用不冒称额外经济候选。

遗留D+1开盘信息、high/low成交截断、原加载器零公司行动事件及继承微观因子的供应商历史版本证据限制均保留。未扩大本项目RankNet式适配为论文已证明的A股Alpha。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE
''')
(R/'DECISION.md').write_text('N1/N2当前冻结配置归档：收益及Sharpe均不优于R16_O01，不升级基线，不扩Top10/20、不加状态门或重拟合其他损失。完成本交付后继续独立D2，不以此失败中断研究。\n')
(R/'REPRODUCE.md').write_text('项目根：research/round23_6_26_execution/env/bin/python tools/continue_round23_7_round26_1.py --deliverable D1 --resume --jobs 1 --seed 17。完整不可变交易产物路径见run_manifest；含代码/环境/数据/schema/标签/成熟/参数/执行/上游的合同及完成标记随每组上传。原始供应商面板仅上传身份/schema/path，不上传原库。\n')
s=json.loads((R/'STATUS.json').read_text());s['economic_run_executions_including_cache_provenance_replays']=18;s['unique_economic_configurations']=9;s['cache_provenance_replays']=9;s['new_warmup_fit_executions_including_failed_diagnostic_attempt']=12;s['decision']='REJECT_CURRENT_CONFIGURATION';s['EVIDENCE_COMPLETE']=True;(R/'STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
base=C/'result/round23_7';(base/'ROUND23_7_REPORT.md').write_text('# Round23.7 执行进度\n\nD0 与 D1 已实际执行并审计。D1结论为REJECT_CURRENT_CONFIGURATION；N-U CAGR54.56%/Sharpe1.857，N-R CAGR54.10%/Sharpe1.855，均弱于v5。完整结果、失败记录和证据见D0/REPORT.md、D1/REPORT.md及STATUS.json。D2、D4尚未在本提交完成。\n\nCURRENT_BASELINE_CHANGED=FALSE\nPRODUCTION_CONFIG_CHANGED=FALSE\n')
# Updated D0 actual cache binding proof shared without any D1 model/winner.
for p in C.glob('result/round23_7_D0*.log'):shutil.copy2(p,C/'result/round23_7/D0'/p.name)
shutil.copytree(C/'result/round23_7/D0',C/'result/round26_1/D0',dirs_exist_ok=True)
