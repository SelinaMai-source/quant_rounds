"""Refresh factual pre-Add checkpoints from completed original generation files."""
from pathlib import Path
import datetime,json,os,psutil
import pandas as pd
from core.round26_1.market import C,R
from core.round26_1.completed_factor_audit import audit


def refresh(publish=False):
    registry=pd.read_csv(R/'factor_run_registry.csv');rows=[]
    for z in registry.loc[registry.status.eq('FACTOR_OOF_YEAR_COMPLETE')].itertuples(index=False):
        directory=Path(z.directory);s=json.loads((directory/'FROZEN_FACTOR.json').read_text());u=json.loads((directory/'RESOURCES.json').read_text())
        path=R/'generation_audits'/z.arm/str(z.seed)/f'{z.year}.json'
        a=json.loads(path.read_text()) if path.exists() else audit(directory)
        assert a['source_directory']==str(directory)
        rows.append({'arm':z.arm,'seed':z.seed,'year':z.year,'env_steps':s['generation_env_steps'],'source_evaluations_including_rejections':s['valid_evaluations'],'unique_ASTs':s['unique_valid_expressions'],'source_pool_admissible_evaluations':a['source_pool_admissible_eval_counter'],'invalid_attempts_preserved':s['invalid_expressions'],'frozen_ASTs':len(s['ASTs']),'max_training_label_available_at':s['max_training_label_available_at'],'fit_cutoff':s['fit_cutoff'],'selected_checkpoint_step':s['selected_generation_step'],'selected_validation_IC_not_strategy_performance':s['selected_validation_pool_IC'],'complete_prediction_rows':a['prediction_rows'],'prediction_dates':a['prediction_dates'],'missing_factor_rows':a['missing_factor_rows'],'wall_seconds':u['wall_seconds'],'CPU_user_seconds':u['CPU_user_seconds'],'CPU_system_seconds':u['CPU_system_seconds'],'sampled_peak_RSS_bytes':u['RSS_sampled_peak_bytes'],'PPO_optimizer_minibatch_steps':s['PPO_optimizer_minibatch_steps'],'unoptimized_environment_tail_steps':a['unoptimized_environment_tail_steps'],'fingerprint':s['original_source_and_input_fingerprint']})
    models=pd.DataFrame(rows);models.to_csv(R/'completed_generation_models.csv',index=False)
    runs=pd.read_csv(R/'run_manifest.csv') if (R/'run_manifest.csv').exists() else pd.DataFrame(columns=['method','scenario'])
    candidate_runs=int(runs.method.ne('AG_A0').sum())
    s=json.loads((R/'STATUS.json').read_text());active=[]
    for proc in psutil.process_iter(['pid','cmdline','uids']):
        try:
            if proc.uids().real!=os.getuid():continue
            args=proc.cmdline()
            if 'core.round26_1.alphagen_worker' not in args:continue
            active.append({'pid':proc.pid,'arm':args[args.index('--arm')+1],'year':int(args[args.index('--year')+1]),'seed':int(args[args.index('--seed')+1])})
        except (psutil.NoSuchProcess,psutil.AccessDenied,ValueError,IndexError):continue
    s['running_folds']=active;s['full_PPO_folds_completed']=len(models);s['candidate_backtest_scenarios_executed']=candidate_runs;s['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();(R/'STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
    # Once actual Add evidence exists its scientific report owns the main report.
    if not candidate_runs:
        small=models[['arm','seed','year','env_steps','frozen_ASTs','complete_prediction_rows','missing_factor_rows','selected_validation_IC_not_strategy_performance']].to_csv(index=False)
        baseline=pd.read_csv(R/'matched_baseline_results.csv')[['experiment_id','CAGR','Sharpe','MaxDD','trades']].to_csv(index=False)
        text=('# D3 AlphaGen真实A股研究：运行中，未完成\n\n'
              f'已完成 {len(models)}/42 个原始PPO生成折，候选Add回测0。仅匹配原生基准S0/S1/S2已实际执行；无可替代v5的新候选结论。\n\n'
              '锁定作者原表达式求值、MseAlphaPool、LSTMSharedNet和MaskablePPO；预算化A股适配，不冒称原论文完整CSI300数值复现。RAW/RES目标和原始配置见CONTRACT.yaml，RAW/RES同预算，不因指标差延长或停止。\n\n'
              '每折100000环境步或20000次原求值调用先到即停；调用数包括重复和原池拒绝，另报不同AST、原池准入、异常。最后完整PPO优化rollout与未优化尾部环境步分别保存，不将env steps当更新次数。\n\n'
              '完整生成折（验证IC仅用于Outer前一年检查点选择，不能当策略成绩）：\n\n```csv\n'+small+'```\n\n'
              '匹配原生基准，仅基准成绩（2023-01-03—2026-07-24，独立300000初资、原overlay预热、同原生执行器）：\n\n```csv\n'+baseline+'```\n\n'
              '市场OHLCV先在完整股票历史rolling再映射D已知候选。63696条历史候选PIT-ST重算通过；预测不按未来标签筛选。原算子/零分母语义沿用作者实现。未来目标、未成熟目标和历史输入prefix测试已按真实数据入口执行；完整PPO六个反事实重训协议已提交，尚未执行时不得标PASS。\n\n'
              '仍缺：全部历年/seed因子、六个完整PPO反事实、历史嵌套OOF入模、Five/Ten/Both Add、触发时Remove-only/Replace、条件OOF与完整风险/费用/Bootstrap/SPA，以及实际XGB入口反事实。不可把未交付改称外部数据阻断。\n\n'
              '2010—2012基线预热微观/资金流恢复遇真实认证失败，仅限制更早预热扩展；不阻断当前OHLCV路径。权限清单严格单列。既存v5 next-open/high-low成交与供应商历史版本限制仍保留，不将parity/PIT区间重放当实盘证明。所有历史均为开发期walk-forward。\n\n'
              '复现入口、原始协议、source lock、真实日志、generation_audits、完成模型/预测/资源及大checkpoint的精确路径与哈希均保留。三seed只做预定等权，不选赢家。\n\nCURRENT_BASELINE_CHANGED=FALSE\nPRODUCTION_CONFIG_CHANGED=FALSE\n')
        (R/'REPORT.md').write_text(text)
        (C/'result/round26_1/ROUND26_1_REPORT.md').write_text('# Round26.1执行状态\n\nD0已执行（覆盖和限制见D0报告）。D3运行中，尚未完成：'+str(len(models))+'/42个完整原始PPO生成折，候选Add回测0；匹配原生基准3种成本情景已执行。D5尚未实施/运行，不属于外部数据阻断。\n\n已完成生成模型和精确资源见D3/completed_generation_models.csv；各模型冻结因子和完整预测见D3/completed_factors，审计见D3/generation_audits。D3/REPORT.md列出实际成绩范围及仍缺证据，STATUS/registry保留失败或未完成状态。当前没有已验收的v5替代方案。\n\nCURRENT_BASELINE_CHANGED=FALSE\nPRODUCTION_CONFIG_CHANGED=FALSE\n')
    print(models[['arm','seed','year','env_steps','complete_prediction_rows']].to_string(index=False),flush=True)
    if publish:
        from core.research_integrity_v5.publish import publish as push
        push('round26_1',f'D3 factual checkpoint: {len(models)} complete original PPO folds, native baseline and remaining evidence separated')
    return rows
if __name__=='__main__':refresh(publish=True)
