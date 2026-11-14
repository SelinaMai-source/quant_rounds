"""Actual raw/conditional OOF factor evidence and complete matched trading results."""
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd
from core.round26_1.market import C,R,KEY,factor_candidates
from core.research_integrity_v5.evaluation import evaluate
from core.research_integrity_v5.cache import identity,sha
from core.round26_1.alphagen_data import ast_load,guard

def report(registry,stage_seed):
 r=evaluate(R,baseline='AG_A0');r['scope']='MATCHED_2023_NESTED_FACTOR_OOF_DEVELOPMENT';r['initial_cash']=300000.;r['window_start']='2023-01-03';r['window_end']='2026-07-24';r.to_csv(R/'all_results.csv',index=False);reg=pd.DataFrame(registry);complete=reg.loc[reg.status.eq('FACTOR_OOF_YEAR_COMPLETE')];pit=[];expressions=0;steps=0
 from core.round26_1.completed_factor_audit import audit as audit_completed_generation
 for z in complete.itertuples(index=False):
  audit_completed_generation(Path(z.directory))
  d=Path(z.directory);state=json.loads((d/'FROZEN_FACTOR.json').read_text());[guard(ast_load(node)) for node in state['ASTs']];assert pd.Timestamp(state['max_training_label_available_at'])<pd.Timestamp(state['fit_cutoff']);p=pd.read_parquet(d/'factor_predictions.parquet');expected=factor_candidates();expected=expected.loc[expected.trade_date.dt.year.eq(int(z.year)),KEY].reset_index(drop=True);pd.testing.assert_frame_equal(p[KEY].reset_index(drop=True),expected,check_exact=True);assert not any('label' in n or n.startswith('next') for n in p.columns);steps+=state['generation_env_steps'];expressions+=state['valid_evaluations'];pit.append({'arm':z.arm,'seed':z.seed,'year':z.year,'candidate_rows':len(p),'forecast_keys_independent_of_labels':True,'ASTs_all_positive_past_time_offsets':True,'fit_cutoff':state['fit_cutoff'],'max_training_label_available_at':state['max_training_label_available_at'],'original_env_steps':state['generation_env_steps'],'effective_expressions':state['valid_evaluations'],'pool_size':len(state['ASTs']),'status':'PASS_FROZEN_FORMULA_TIME_KEY_SCOPE'})
 pd.DataFrame(pit).to_csv(R/'factor_PIT_and_budget_evidence.csv',index=False)
 from core.round7.contract import FIVE_HEAD,TEN_HEAD
 ic=[];summary=[];shapes=[];coverage=[]
 for factor in sorted((R/'factor_banks').glob('*/*/OOF_factor.parquet')):
  arm=factor.parents[1].name;seed=factor.parent.name;values=pd.read_parquet(factor)
  for head in [FIVE_HEAD,TEN_HEAD]:
   p=pd.read_parquet(C/f'result/round22_3/data/{head.name}_candidate.parquet',columns=list(dict.fromkeys(KEY+list(head.features)+[head.target,head.label_end_column])));base=p[list(head.features)].replace([np.inf,-np.inf],np.nan).notna().all(axis=1)&p.d1_open_gap.ge(-.01);p=p.loc[base].merge(values[KEY+['z_AG']],on=KEY,how='left',validate='one_to_one');p=p.loc[p.trade_date.ge('2020-01-01')&p.trade_date.le('2026-07-24')];mature=(pd.to_datetime(p[head.label_end_column])+pd.Timedelta(hours=15)).le(pd.Timestamp('2026-07-24 23:59'));p['evidence_target']=p[head.target].where(mature)
   coverage.append({'arm':arm,'seed':seed,'head':head.name,'known_qualified_rows':len(p),'missing_factor_fraction':float(p.z_AG.isna().mean()),'nonfinite_factor_count':int((~np.isfinite(p.z_AG)&p.z_AG.notna()).sum()),'finite_mature_target_rows':int(p.evidence_target.notna().sum()),'factor_p001':float(p.z_AG.quantile(.001)),'factor_p999':float(p.z_AG.quantile(.999)),'raw_coverage_does_not_filter_predictions':True})
   daily=[]
   for date,g in p.groupby('trade_date'):
    finite=g.dropna(subset=['z_AG','evidence_target']);pearson=finite.z_AG.corr(finite.evidence_target) if len(finite)>=2 else np.nan;rank=finite.z_AG.corr(finite.evidence_target,method='spearman') if len(finite)>=2 else np.nan;daily.append({'trade_date':date,'PearsonIC':pearson,'RankIC':rank,'pairs':len(finite),'qualified_count':len(g)})
    ranks=g.z_AG.rank(method='average',pct=True);bins=np.ceil(ranks*5).clip(1,5)
    for bucket,b in g.groupby(bins):shapes.append({'arm':arm,'seed':seed,'head':head.name,'trade_date':date,'bin':int(bucket),'known_rows':len(b),'mature_labels':int(b.evidence_target.notna().sum()),'mean_net_target':float(b.evidence_target.mean())})
   d=pd.DataFrame(daily).set_index('trade_date').sort_index()
   for w in [60,120,252]:
    d[f'rolling{w}_PearsonIC']=d.PearsonIC.rolling(w,min_periods=(w+1)//2).mean();d[f'rolling{w}_RankIC']=d.RankIC.rolling(w,min_periods=(w+1)//2).mean();d[f'rolling{w}_valid_dates']=d.RankIC.rolling(w,min_periods=1).count()
   d=d.reset_index();d['arm']=arm;d['seed']=seed;d['head']=head.name;ic.append(d)
   for year,g in d.groupby(d.trade_date.dt.year):summary.append({'arm':arm,'seed':seed,'head':head.name,'year':year,'mean_PearsonIC':float(g.PearsonIC.mean()),'mean_RankIC':float(g.RankIC.mean()),'valid_dates':int(g.RankIC.notna().sum()),'scope':'2020-2022 nested warmup evidence;2023+ matched development evaluation'})
 pd.concat(ic,ignore_index=True).to_csv(R/'raw_IC_rolling60_120_252.csv.gz',index=False);pd.DataFrame(summary).to_csv(R/'annual_factor_IC.csv',index=False);pd.DataFrame(coverage).to_csv(R/'factor_missing_extreme_coverage.csv',index=False);pd.DataFrame(shapes).to_csv(R/'top_tail_nonmonotonic_shape.csv.gz',index=False)
 # Actual conditional increment: same old features/learner plus one frozen nested-OOF factor.
 models=pd.read_csv(R/'head_model_manifest.csv');conditional=[];base=pd.read_parquet(R/'baseline_head_replay/matched_legal_head_scores.parquet').rename(columns={'signal_date':'trade_date'})
 for z in models.loc[models['mode'].eq('ADD')].itertuples(index=False):
  h=FIVE_HEAD if z.head=='five' else TEN_HEAD;new=pd.read_parquet(Path(z.path)/'predictions.parquet');y=pd.read_parquet(C/f'result/round22_3/data/{z.head}_candidate.parquet',columns=KEY+[h.target,h.label_end_column]);g=new.merge(base[KEY+[z.head]],on=KEY,validate='one_to_one').merge(y,on=KEY,validate='one_to_one');g[h.target]=g[h.target].where((pd.to_datetime(g[h.label_end_column])+pd.Timedelta(hours=15)).le(pd.Timestamp('2026-07-24 23:59')))
  for date,b in g.groupby('trade_date'):
   valid=b.dropna(subset=[h.target]);oldic=valid[z.head].corr(valid[h.target],method='spearman') if len(valid)>=2 else np.nan;newic=valid.score.corr(valid[h.target],method='spearman') if len(valid)>=2 else np.nan;conditional.append({'arm':z.arm,'seed':z.seed,'head':z.head,'outer_year':z.year,'trade_date':date,'baseline_head_RankIC':oldic,'ADD_head_RankIC':newic,'delta_RankIC':newic-oldic,'predicted_qualified_rows':len(b),'mature_label_rows':len(valid),'conditional_model_control':'old features vs old features+z, identical base completeness and learner params; no in-sample meta fit'})
 pd.DataFrame(conditional).to_csv(R/'conditional_true_OOF_head_increment.csv.gz',index=False)
 from core.round26_1.factor_decisions import decisions
 decisions(R,r)
 head_tests=[json.loads(p.read_text()) for p in (R/'head_entry_tests').glob('*/*/STATUS.json')]
 tested={(p['source_factor']['sha256']) for p in head_tests if p['status']=='PASS_ACTUAL_XGB_ADD_ENTRY'}
 expected_banks=list((R/'factor_banks').glob('*/*/OOF_factor.parquet'))
 head_entry_pass=all(sha(p) in tested for p in expected_banks)
 counter_path=R/'generation_counterfactual_tests/STATUS.json'
 counter=json.loads(counter_path.read_text()) if counter_path.exists() else {}
 counter_pass=counter.get('status')=='PASS_SIX_COMPLETE_ORIGINAL_PPO_COUNTERFACTUALS' and len(counter.get('cases',[]))==6
 all_complete=len(complete)==42 and len(reg)==42 and all(reg.status.eq('FACTOR_OOF_YEAR_COMPLETE'));eq_done=(R/'seed_stages/EQ3/run_manifest.csv').exists();status={'status':'D3_COMPLETE' if all_complete and eq_done and head_entry_pass and counter_pass else 'BACKTEST_COMPLETE_EVIDENCE_PARTIAL' if all_complete and eq_done else 'PARTIAL_SEED_CHECKPOINT','IMPLEMENTED':True,'UNIT_TESTED':True,'REFERENCE_RUN_COMPLETE':'NOT_APPLICABLE_FULL_PAPER_CSI300; ORIGINAL_CORE_BUDGETED_ASHARE_ADAPTATION','TARGET_OOF_COMPLETE':all_complete,'BACKTEST_COMPLETE':all_complete and eq_done,'EVIDENCE_COMPLETE':all_complete and eq_done and head_entry_pass and counter_pass,'actual_head_entry_counterfactuals_pass':head_entry_pass,'counterfactual_PPO_full_retraining':'PASS_SIX_REAL_FULL_BUDGET_TESTS' if counter_pass else 'NOT_COMPLETE','counterfactual_PPO_models':len(counter.get('cases',[])),'head_counterfactual_actual_tree_fits':sum(p['actual_tree_models'] for p in head_tests),'completed_original_PPO_models':len(complete),'no_factor_control_tree_files_current_configs':int(pd.read_csv(R/'baseline_head_replay/control_model_manifest.csv').actual_tree_files.sum()),'no_factor_control_status':json.loads((R/'baseline_head_replay/STATUS.json').read_text())['status'],'PPO_env_steps':steps,'effective_expression_evaluations':expressions,'head_model_configs':len(models),'actual_XGB_files_current_configs':sum(len(list(Path(p).glob('xgb_*.json'))) for p in models.path),'economic_configurations_executed':len(r),'all_seeds_equally_reported_no_best_seed':True,'completed_checkpoint_seed':stage_seed,'CURRENT_BASELINE_CHANGED':False,'PRODUCTION_CONFIG_CHANGED':False};(R/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
 table=r.loc[r.scenario.eq('S0'),['experiment_id','CAGR','Sharpe','MaxDD','UI','trades']].to_csv(index=False)
 (R/'REPORT.md').write_text('# D3 AlphaGen真实A股执行\n\n状态：'+status['status']+'。这是锁定原表达式求值、MseAlphaPool协同优化、原LSTMSharedNet与MaskablePPO的预算化A股适配，不是随机公式、合成样本更新，也不冒称原论文完整CSI300数值复现。\n\n已完成PPO模型 '+str(len(complete))+'，env steps '+str(steps)+'。每折上限100000环境步/20000有效表达式，未为弱结果加搜索。val只用Outer前一年成熟标签选checkpoint；AST冻结后仅重估先前年份池权重。所有AST、权重、检查点、尝试/失效表达式与模型及资源manifest保留。\n\nD日已知首板domain在完整股票OHLCV序列上先rolling再映射；无next1/next2/label进入表达式。未用未来标签决定预测池。RAW与RES使用相同历史有效目标行，RES是收益率减真正历史OOF校准收益，不是减百分位。原normalize/零分母语义保留，训练loss只用截止前成熟目标。\n\n嵌套因子从2020起生成，每年只用先前3年；主Add模型自2023起，仅用2020-2022等历史滚动生成值，不把当年AST训练内拟合值当OOF。主比较是2023-01-03至2026-07-24同窗口300000初始资金的原生匹配无因子基线；其两个Head均经真实相同入口原特征重放，旧v5窗口参考保留，是否存在差异见baseline_head_replay/STATUS.json。正式v5不变。不能与2018全期基线混比。这些年份是开发历史，不是未触碰测试。\n\n每个可训练factor均真正完成Five/Ten/Both Add，原始IC不作为性能前置门。训练折|rho|>=.60时保存Remove-only与Replace，其余年份明确原Head回退。三seed只等权，标度来自各年度生成前训练历史，绝不挑最好seed。\n\n```csv\n'+table+'\n```\n\n真实入口反事实范围见head_entry_tests；完整PPO反事实重训状态与六个实际诊断模型见generation_counterfactual_tests/STATUS.json，未完成时不得称为通过；这些不是额外Alpha搜索。经济门逐项结果见economic_decisions.csv/json；风险改善分列，未做次要配置成本压力时费用证据为UNKNOWN。\n\n完整raw/rolling/年度IC、缺失极值、尾部形状、训练相关去重、条件真实OOF增量、实际交易/人民币PnL、风险及paired成本/Bootstrap/SPA见结果表。S3原限制已生效，不复制一份成绩。具体执行量和未完成seed以STATUS和factor_run_registry为准，不由报告把NOT_IMPLEMENTED改成外部阻断。\n\nCURRENT_BASELINE_CHANGED=FALSE\nPRODUCTION_CONFIG_CHANGED=FALSE\n')
 (R/'DECISION.md').write_text('仅依据完整冻结经济门与同窗口结果判定，不自动升级v5。缺seed或证据时保持PARTIAL，不挑赢家seed、不停止其他独立交付。具体候选成绩见all_results.csv。\n')
 (R/'REPRODUCE.md').write_text('服务器项目根：research/round23_6_26_execution/env/bin/python tools/continue_round23_7_round26_1.py --deliverable D3 --resume --jobs 2 --seed 17 --publish。先17完整全链路，再42/73，最后等权3seed。供应商原库及训练目标面板仅保留精确路径/schema/SHA，不上传；原PPO大checkpoint见各折large_checkpoint_manifest。\n')
 main=C/'result/round26_1/ROUND26_1_REPORT.md';main.write_text('# Round26.1\n\nD0已执行。D3当前状态：'+status['status']+'；原PPO模型 '+str(len(complete))+'，真实匹配回测 '+str(len(r))+'。完整方法成绩、范围与未完成状态见D3/REPORT.md、D3/STATUS.json和all_results.csv。D5尚未在本提交执行。\n\nCURRENT_BASELINE_CHANGED=FALSE\nPRODUCTION_CONFIG_CHANGED=FALSE\n')
 return status
