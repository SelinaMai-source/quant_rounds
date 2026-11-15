"""Actual TRA native ledgers, same-backbone contrasts and honest stage reports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from core.round23_7.tra_reference import R,C
from core.research_integrity_v5.evaluation import evaluate
from core.research_integrity_v5.economic import decisions
from core.research_integrity_v5.cache import identity

def risk_plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    daily=pd.read_csv(R/'daily_returns.csv.gz',parse_dates=['trade_date']).set_index('trade_date')
    metrics=results.set_index('experiment_id');checks=[]
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharex=True)
    for col,phase in enumerate(['T0','T1']):
        names=['TRA_A0']+[f'TRA_{phase}_{seed}' for seed in [17,42,73] if f'TRA_{phase}_{seed}' in daily]
        for name in names:
            wealth=pd.concat([pd.Series([1.0],index=pd.to_datetime(['2018-01-02'])),(1+daily[name]).cumprod()])
            assert wealth.index.is_unique and wealth.index.is_monotonic_increasing
            dd=wealth/wealth.cummax()-1
            assert np.isclose(wealth.iloc[-1],metrics.loc[name,'TerminalNAV']/300000,rtol=1e-10,atol=1e-10)
            assert np.isclose(dd.min(),metrics.loc[name,'MaxDD'],atol=1e-10)
            opts={'color':'black','linewidth':2} if name=='TRA_A0' else {'linewidth':1.2}
            axes[0,col].plot(wealth.index,wealth,label=name,**opts)
            axes[1,col].plot(dd.index,dd*100,label=name,**opts)
            checks.append({'experiment_id':name,'terminal_wealth':float(wealth.iloc[-1]),'MaxDD':float(dd.min()),'matches_actual_results':True})
        axes[0,col].set_title(phase+' vs matched v5; all completed fixed seeds')
        axes[0,col].set_yscale('log');axes[0,col].set_ylabel('Wealth / initial cash (log)');axes[1,col].set_ylabel('Drawdown (%)')
        for row in [0,1]:axes[row,col].grid(alpha=.25);axes[row,col].legend(fontsize=8)
    fig.suptitle('Historical development; signals end 2026-07-01, valuation ends 2026-07-24');fig.tight_layout()
    fig.savefig(R/'primary_NAV_drawdown.png',dpi=170);fig.savefig(R/'primary_NAV_drawdown.svg');plt.close(fig)
    svg=R/'primary_NAV_drawdown.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (R/'primary_plot_audit.json').write_text(json.dumps({'scope':'all completed fixed seeds S0, no historical winner selection','checks':checks,'inputs':[identity(R/'daily_returns.csv.gz'),identity(R/'all_results.csv')]},indent=2))

def report(stage_seed=None):
    registry=pd.read_csv(R/'target_run_registry.csv')
    runs=pd.read_csv(R/'run_manifest.csv')
    results=evaluate(R,baseline='TRA_A0');results['scope']='TRA_2018_2026_DEVELOPMENT';results.to_csv(R/'risk_results.csv',index=False)
    decisions(R,results,'TRA_A0')
    complete_seeds=[]
    paired=[]
    for seed in [17,42,73]:
        methods=[f'TRA_T0_{seed}',f'TRA_T1_{seed}']
        subset=runs.loc[runs.method.isin(methods)]
        if len(subset)!=6:continue
        folder=R/'paired_T1_vs_T0'/str(seed);folder.mkdir(parents=True,exist_ok=True)
        subset.to_csv(folder/'run_manifest.csv',index=False)
        contrast=evaluate(folder,baseline=methods[0]);contrast['comparison']='T1_vs_T0_same_shared_backbone'
        contrast.to_csv(folder/'all_results.csv',index=False);decisions(folder,contrast,methods[0]);paired.append(contrast)
        done=registry.loc[registry.namespace.eq('original')&registry.seed.eq(seed)&registry.status.eq('FULL_TARGET_FIT_COMPLETE')]
        if len(done)==54:complete_seeds.append(seed)
    if paired:pd.concat(paired,ignore_index=True).to_csv(R/'same_backbone_results.csv',index=False)
    formal=R/'formal_baseline'
    if (formal/'run_manifest.csv').exists():
        formal_result=evaluate(formal,baseline='TRA_FORMAL_V5')
        combined=pd.concat([results,formal_result],ignore_index=True)
    else:combined=results
    combined.to_csv(R/'all_results.csv',index=False)
    risk_plot(combined)
    diagnostic=R/'counterfactual_tests/native_order_replays/run_manifest.csv'
    diagnostic_count=0
    if diagnostic.exists():
        probes=pd.read_csv(diagnostic);diagnostic_count=len(probes)
        anchor=Path(runs.set_index('code').loc['TRA_T1_17','path'])
        for row in probes.itertuples(index=False):
            pd.testing.assert_frame_equal(pd.read_parquet(anchor/'nav.parquet'),pd.read_parquet(Path(row.path)/'nav.parquet'),check_exact=True)
        spa_path=R/'SPA.json';proof=json.loads(spa_path.read_text())
        proof['registered_nonalpha_full_account_replay_aliases']={row.code:'TRA_T1_17' for row in probes.itertuples(index=False)}
        proof['nonalpha_replay_returns_exactly_verified_against_primary_trial']=True
        spa_path.write_text(json.dumps(proof,indent=2))
    cold_path=R/'counterfactual_tests/STATUS.json'
    cold=json.loads(cold_path.read_text()) if cold_path.exists() else {}
    full=len(complete_seeds)==3 and cold.get('status')=='PASS_SIX_REAL_FULL_BUDGET_CASES'
    reference=json.loads((R/'REFERENCE_STATUS.json').read_text())
    status={'status':'D4_COMPLETE' if full else 'PARTIAL_SEED_CHECKPOINT','IMPLEMENTED':True,'UNIT_TESTED':True,'REFERENCE_RUN_COMPLETE':reference.get('status')=='REFERENCE_RUN_COMPLETE','TARGET_OOF_COMPLETE':len(complete_seeds)==3,'BACKTEST_COMPLETE':len(complete_seeds)==3,'EVIDENCE_COMPLETE':full,'completed_seed_paths':complete_seeds,'last_checkpoint_seed':stage_seed,'configured_original_target_models_completed':int((registry.namespace.eq('original')&registry.status.eq('FULL_TARGET_FIT_COMPLETE')).sum()),'configured_counterfactual_target_models_completed':int((registry.namespace.eq('cold_unavailable_targets')&registry.status.eq('FULL_TARGET_FIT_COMPLETE')).sum()),'failed_registry_rows':int(registry.status.ne('FULL_TARGET_FIT_COMPLETE').sum()),'full_model_counterfactual_status':cold.get('status','NOT_RUN'),'actual_native_economic_configurations':len(combined),'CURRENT_BASELINE_CHANGED':False,'PRODUCTION_CONFIG_CHANGED':False,'external_blocker':False,'signal_end':'2026-07-01','valuation_end':'2026-07-24','synthetic_fixtures_are_not_research_models':True}
    status['actual_native_diagnostic_configurations']=diagnostic_count
    (R/'STATUS.json').write_text(json.dumps(status,indent=2))
    summary=combined.loc[combined.scenario.eq('S0'),['experiment_id','CAGR','Sharpe','MaxDD','TerminalNAV','trades']]
    text=('# D4 TRA — '+status['status']+'\n\n'
          '公开旧作者实现参考与A股目标回测分开。作者参考两阶段完整默认停止结果见reference/original_benchmark_results.csv；原作者切片可跨股票的已知缺陷保留披露，公开参考不是目标PIT证明。目标数据按股票边界、真实成熟时间及独立预测期队列适配。\n\n'
          '目标使用原作者LSTM/TRA/Gumbel/Sinkhorn OT/MSE；六通道数据维度适配、batch1形状兼容及相同冻结权重均值的精确恒等兼容均有源码SHA和失败证据。500epoch上限/20验证早停/100batch上限不变。T0/T1同一共享预训练骨干，保持完全冻结；只改变1/10预测头和原TRA/OT机制，不归因于骨干升级。\n\n'
          '训练梯度仅使用切分截止前成熟标签，验证保留所有D日已知节点、仅对成熟标签计指标。每次验证和年度Outer预测均独立冷启动队列；Outer不携带训练内拟合误差作为历史OOF。专家输出在预测时保存，仅在真实退出成熟后反馈，并保留作者21观测排除规则。\n\n'
          '输入覆盖、无交易报价处理、完整2010起预热和短历史零padding见target_data。信号截止2026-07-01；2018-01-02..2026-07-24均为已反复开发的历史期，初始现金300000，最后时段是估值/退出尾段。正式v5不变并单列；主要对照补齐655个真实可预测键，旧38003键分数精确一致。覆盖差异不计TRA贡献。\n\n'
          'S0实际结果：\n\n```csv\n'+summary.to_csv(index=False)+'```\n\n'
          '![全部已完成固定种子的实际净值与回撤](primary_NAV_drawdown.png)\n\n'
          '全部S1/S2、年度收益、费用/滑点、敞口/换手、真实Top1/Top2净PnL、人民币NAV桥、最差63日/UI/ES95、配对Bootstrap和SPA见结果表。S3为原执行器已有开盘封板拒单约束，实际拒单身份见S3_identity_evidence.csv，不复制成绩冒充新实验。T1相对T0的同成本结果和独立配对检验见paired_T1_vs_T0；不可把T1相对v5的全部变化都算成路由贡献。\n\n'
          '原有符号经济门逐项结果见economic_decisions.csv/json；不放宽门槛、不挑选历史赢家、不自动晋升。全部三个种子独立报告。完整冷拟合与真实Outer未来/未成熟标签回放见counterfactual_tests，尚未完成的seed和测试以STATUS为准。\n\n'
          '完整Outer另做当时不可用标签删除，不永久删除之后已合法成熟的反馈。订单反事实调用原Stock完整账户：一个原始S0加六个冷拟合标签变体，共七个诊断配置；全部overlay计划输入、成交、拒单、持仓、袖套、现金流、费用及NAV须精确一致，并再次对照主要TRA_T1_17实际NAV。独立登记于counterfactual_tests/native_order_replays，SPA明示相同路径别名，不冒充新alpha备择或七个新经济候选。\n\n'
          '模型、预测和实测资源见target_models与worker_profiles，原始大输入仅提供路径/schema/字节SHA清单，不上传私人面板。训练内输出不冒称历史OOF；工程合成fixture不计真实模型。所有早期失败和澄清保留。供应商历史修订、legacy D+1开盘缺口/日内高低价成交与零公司行动覆盖限制仍保留，不声称独立未用测试或实盘证明。\n\n'
          'CURRENT_BASELINE_CHANGED=FALSE\nPRODUCTION_CONFIG_CHANGED=FALSE\n')
    (R/'REPORT.md').write_text(text)
    decision=pd.read_csv(R/'economic_decisions.csv')
    (R/'DECISION.md').write_text('固定原经济门，不自动升级；T1还必须相对同骨干T0评估。\n\n```csv\n'+decision.to_csv(index=False)+'```\n')
    (R/'REPRODUCE.md').write_text('在现有授权服务器的项目根目录运行，使用现有env及已锁定作者源码/原始数据，不升级ZyQuant。\n\n```bash\nPYTHONDONTWRITEBYTECODE=1 research/round23_6_26_execution/env/bin/python -B tools/continue_round23_7_round26_1.py --deliverable D4 --resume --jobs 2 --seed 17 --publish\n```\n\n固定17全路径→42→73，不能选优种子；恢复前检查无重复控制进程。模型缓存核验完整源码、数据/标签/成熟时间、参数、执行契约和全部完成工件。完整参考先于目标训练。实际server-only输入路径与SHA见target_data/server_only_inputs_manifest.json。\n')
    model_manifest=[]
    import torch
    for p in sorted((R/'target_models').glob('**/model.bin')):
        item=identity(p);state=torch.load(p,map_location='cpu',weights_only=True)
        item['format']='torch state dictionaries: model, tra'
        item['tensor_schema']={group:{name:{'shape':list(tensor.shape),'dtype':str(tensor.dtype)} for name,tensor in state[group].items()} for group in ['model','tra']}
        item['contract']=identity(p.parent/'contract.json');model_manifest.append(item)
    (R/'MODEL_ARTIFACTS.json').write_text(json.dumps(model_manifest,indent=2))
    parent=C/'result/round23_7/ROUND23_7_REPORT.md'
    original=parent.read_text();marker='\n<!-- D4_REAL_EXECUTION_CHECKPOINT -->'
    original=original.split(marker)[0]
    parent.write_text(original+marker+'\nD4实际状态：'+status['status']+'；已完成种子路径'+str(complete_seeds)+'。完整作者参考、真实目标模型、T0/T1同骨干回测、时间反事实与全部风险见[D4主报告](D4/REPORT.md)。不把合成集成测试叫真实研究，不修改正式v5/CURRENT。\n')
    return status
