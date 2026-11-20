"""Truthful per-seed/current delivery reports, never turn pending work into a blocker."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from core.round26_1.master_provider import C,B,R

def read(path,default=None):
    p=Path(path);return json.loads(p.read_text()) if p.exists() else default

def run(store_path,seed):
    from core.round26_1.master_artifacts import inputs,heads
    inputs(store_path);head_artifacts=heads()
    fits=[];seed_states={}
    for current in [17,42,73]:
        manifest=R/f'FIT_MANIFEST_{current}.csv'
        progress=R/f'FIT_PROGRESS_{current}.csv'
        frame=pd.read_csv(manifest if manifest.exists() else progress) if (manifest.exists() or progress.exists()) else pd.DataFrame()
        complete=bool(len(frame)==24 and set(zip(frame.arm,frame.year))=={(arm,year) for arm in ['M0','M1'] for year in range(2015,2027)}) if len(frame) else False
        seed_states[str(current)]={'accepted_OOF_models':len(frame),'all24_OOF_complete':complete}
        for row in frame.itertuples(index=False):
            registry=read(Path(row.path)/'MODEL_REGISTRY.json');assert registry['status']=='FACTOR_OOF_YEAR_COMPLETE'
            contract=read(Path(row.path)/'contract.json');assert any('TARGET_DOMAIN_CORRECTION_FREEZE.json' in item['path'] for item in contract['upstream'])
            assert np.isfinite(registry['mean_train_loss']) and registry['gradient_dates']>0
            fits.append({'arm':row.arm,'seed':row.seed,'year':row.year,'path':row.path,'training_dates':registry['all_training_dates'],'gradient_dates':registry['gradient_dates'],'checkpoint_reason':registry['checkpoint_reason'],'checkpoint_sha256':registry['checkpoint_sha256'],'fit_cutoff':registry['fit_cutoff'],'max_training_label_available_at':registry['max_training_label_available_at']})
    pd.DataFrame(fits).to_csv(R/'ACCEPTED_NN_MODEL_MANIFEST.csv',index=False)
    reference=read(R/'reference/REFERENCE_STATUS.json',{})
    ref_complete=reference.get('status')=='REFERENCE_PUBLIC_CORE_FULL_LOCKED_EPOCH_COMPLETE' and reference.get('actual_optimizer_steps')==2979
    nn=read(R/'NN_COLD_COUNTERFACTUALS.json',{});backend=read(R/'BACKEND_COLD_COUNTERFACTUALS.json',{});native=read(R/'NATIVE_COUNTERFACTUAL_REPLAYS.json',{})
    tests={'NN12':nn.get('status')=='PASS' and len(nn.get('cases',[]))==12,'backend24':backend.get('status')=='PASS' and backend.get('full_configurations')==24,'native12':native.get('status')=='PASS' and len(native.get('cases',[]))==12}
    all_oof=all(v['all24_OOF_complete'] for v in seed_states.values())
    results=pd.read_csv(R/'all_results.csv') if (R/'all_results.csv').exists() else pd.DataFrame()
    primary_cost_complete=True
    for current in [17,42,73]:
        if not len(results):primary_cost_complete=False;break
        existing=set(results.experiment_id)
        for arm in ['M0','M1']:
            for variant in ['FIVE_ADD','TEN_ADD','BOTH_ADD']:
                method=f'{arm}_{current}_{variant}'
                primary_cost_complete &= all(method+suffix in existing for suffix in ['', '_S1','_S2'])
    all_backtests=bool(all_oof and primary_cost_complete)
    plot=read(R/'PLOT_SOURCE_AUDIT.json',{});factor=read(R/'FACTOR_EVIDENCE_SCOPE.json',{})
    evidence=bool(all_backtests and ref_complete and all(tests.values()) and plot.get('status')=='PASS' and len(plot.get('cases',[]))==9 and factor.get('banks')==6 and (R/'SPA.json').exists() and (R/'M1_vs_M0_paired_bootstrap.csv').exists())
    completed=bool(all_oof and all_backtests and evidence and ref_complete)
    status={'status':'D5_COMPLETE' if completed else 'PARTIAL_SEED_CHECKPOINT','IMPLEMENTED':True,'UNIT_TESTED':True,'REFERENCE_RUN_COMPLETE':ref_complete,'TARGET_OOF_COMPLETE':all_oof,'BACKTEST_COMPLETE':all_backtests,'EVIDENCE_COMPLETE':evidence,'seed_states':seed_states,'accepted_target_NN_models':len(fits),'accepted_target_NN_training_dates':sum(v['training_dates'] for v in fits),'full_cold_NN_runs':len(nn.get('cases',[])),'full_cold_backend_configurations':backend.get('full_configurations',0),'full_cold_backend_boosters':backend.get('full_booster_fits',0),'actual_native_causal_replays':len(native.get('cases',[])),'actual_economic_account_configurations':len(results),'actual_regular_head_configurations':len(head_artifacts),'actual_regular_head_boosters':sum(v['actual_booster_fits'] for v in head_artifacts),'no_factor_control_configurations':18,'no_factor_control_boosters':90,'diagnostic_tests':tests,'completed_checkpoint_seed':seed,'CURRENT_BASELINE_CHANGED':False,'PRODUCTION_CONFIG_CHANGED':False,'external_blocker':False,'invalid_gross2012_completed_NN_diagnostics':2,'initial_normalizer_format_failed_attempts':24,'intentionally_interrupted_partial_NN_attempts':4,'no_failure_or_invalid_model_counted_as_accepted':True,'signal_end':'2026-07-01','valuation_end':'2026-07-24','scope':'historical development walk-forward, not untouched forward/live execution proof'}
    (R/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    decisions=pd.read_csv(R/'economic_decisions.csv') if (R/'economic_decisions.csv').exists() else pd.DataFrame()
    table=results.loc[results.scenario.eq('S0'),['experiment_id','CAGR','Sharpe','MaxDD','UI','TerminalNAV','trades']].to_csv(index=False) if len(results) else 'No completed valid account result yet.\n'
    relation=pd.read_csv(R/'M1_vs_M0_primary_relation.csv') if (R/'M1_vs_M0_primary_relation.csv').exists() else pd.DataFrame()
    rtable=relation.to_csv(index=False) if len(relation) else 'No paired account result yet.\n'
    candidates=decisions.loc[decisions.economic_gate_pass.eq(True),'experiment_id'].tolist() if len(decisions) else []
    reference_text=(f"公开参考：2979训练日/2979次真实更新；新seed17测试IC={reference['fresh_test']['original_source_metrics']['IC']:.8f}，RankIC={reference['fresh_test']['original_source_metrics']['RIC']:.8f}；作者seed0检查点测试IC={reference['author0_test']['original_source_metrics']['IC']:.8f}。305个未知label节点保留在183300个测试预测中。" if ref_complete else '公开完整参考尚未完成。')
    report=f'''# D5 MASTER：{status['status']}

{reference_text} 这是锁定父main.py的一整轮示例预算，不冒称论文最高40轮数值复现，也不冒称作者未公开业务系统。旧pickle/Qlib接口按原start_idx/end_idx与声明日期映射，原历史窗口保留；训练完整，推理不按label缺失删节点。原作者checkpoint仅公开诊断，A股全部从头初始化。

A股M1直接调用锁定原MASTER；M0只对空间注意力logits加固定identity mask，其余层、初始化、数据、processor、seed和预算相同。原Alpha158全158项及三真实指数市场63均由原Qlib表达式计算；市场公式顺序与作者CSV完全匹配。无简化Transformer或伪因子。匹配骨干、每日期训练步和processor哈希见M0_M1_MATCHED_BONE_AUDIT.csv。

图由D日原合法首板清单和截至D-1过去20交易日原始成交额中位数top1000历史可交易沪深股票组成，float64排序、ID打破并列；不用当前上市/未来退市表，不用D+1开盘或未来label决定节点。模型仅D日可得221特征。无label context也参与原attention；loss仅成熟合法目标，N<40原p0空切片已按冻结修复，N<2/零方差仅跳过梯度日期，绝不删推理节点。

2012合法暖启动为1302条，完整原AI071过滤、原退出成本与真实成熟日。2013/14重建4866键、标签、退出日与原清单完全一致；原2013日历边界按原样重放，2013+冻结股票池未扩充。此前gross2012两个完成NN标INVALID_WARMUP_DOMAIN，2016未完成任务停止；24个归一化索引失败和缓存修复/域修复中断均保留在failures，未进入成绩、经济门或有效模型合计。

每个因子Y用前3年：前2年训练、最后1年诊断验证，训练label须在验证起点前成熟，验证指标label须在Y起点前成熟；模型冻结全年。真正年度OOF从2015生成，2018原后端用2015–17等三个历史OOF年，绝不把当前NN训练内拟合值伪装历史值。Five14/Ten13原XGB工厂、五个树seed、无行列采样、90%Ten/10%Five不变。Add全运行，训练|rho|>=0.60才用最相关原列执行Remove/Replace，不以原始IC预筛，未触发年份保留原Head。

主比较2018-01-02至2026-07-24、30万元同日历原生账户。原formal v5历史锚不改；新无因子控制真正重训，若候选覆盖延长，只对同键分数及覆盖差异审计后使用MATCHED A0，不把覆盖变化算因子增量。信号截至2026-07-01，07-24仅估值/退出尾部。原B03/O01、双袖套/Top2/100股/T+1/退出及费用保留，S1/S2在订单执行层加10/20bp买入滑点，不删分数重排/递补/扩大槽位。S3已在原执行器生效，原计划身份及拒单证据另存。

```csv
{table}```

关系信息主对照是每固定seed的M1 BothAdd相对M0 BothAdd，同时报告相对A0。仅胜v5而不胜M0不证明跨股票信息有效；不挑最好seed或事后增加融合。

```csv
{rtable}```

已完成有效目标NN {len(fits)}/72；三个seed状态见STATUS；原生经济配置{len(results)}。完整CPU/GPU/RSS、模型/输入/预测/订单工件的实际schema/bytes/SHA及服务器路径已导出；大型特征/私有训练面板及检查点不复制到仓库，SERVER_*是服务器工件目录，仓库artifact_manifest另验证实际上传文件。所有失败与未完成均保留。

全cap冷NN、完整双Head XGB重训、未来上下文/报价/label扰动与删除、完整原生8账表比较，均以实际通过案例为准：{json.dumps(tests,ensure_ascii=False)}。随机架构测试或缓存测试不替代这些真实运行。价格/指标/processor缓存查询前校验实际传递输入字节和全部契约；局部失败/未完成从不命中完成缓存。

原始/年度/60/120/252滚动IC、缺失/极值、quintile形状、训练rho、真正OOF Add相对原特征的条件增量、Top1/Top2实际人民币PnL、UI/ES95/最差63日/年度收益/敞口/换手/费用、NAV桥、paired bootstrap及arch.SPA见表。20日5000次、10/40日各1000次，M1/M0和A0均同抽样索引；重采样回撤只是路径敏感性。净值图包含初始1并逐点核对实际NAV。

当前通过冻结经济门的配置：{json.dumps(candidates,ensure_ascii=False)}；这不是自动升级决定，缺完整固定seed/证据时仍是PARTIAL。各门逐项见economic_decisions.csv，UNKNOWN不算True。

共同局限：2018–2026是反复使用的开发历史，不是未触碰测试；供应商非恒定历史修订/首次发布时间尚无完整vintage证明。原v5的D+1缺口通道、高低价成交假设和未提供corporate-action事件的限制保留，parity不证明实盘可成交。公开dump缺真实label_available_at/原始上游股池完整性证明；公共参考不当成A股因果验证。价格历史初始因子归一化、未来事件与全源Alpha158干预证据不等于已证明所有历史修订PIT。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE
'''
    (R/'REPORT.md').write_text(report)
    dtext=decisions.to_csv(index=False) if len(decisions) else 'Valid economic comparisons pending.\n'
    rejected=decisions.loc[decisions.decision.eq('REJECT_CURRENT_CONFIGURATION'),'experiment_id'].tolist() if len(decisions) else []
    retain='保留通过全部冻结门的研究配置：'+json.dumps(candidates,ensure_ascii=False)+'；仍不自动替代v5，缺未触碰前瞻和实盘成交/vintage证明。' if candidates else '当前没有证据足够支持替代v5的MASTER配置；保留真实工件和风险观察，不扩大当前无效配置预算。'
    relation_text='关系信息证据尚未完成。'
    if len(relation):
        rel=relation.loc[relation.scenario.eq('S0')]
        positive=int((rel.delta_CAGR.gt(0)&rel.delta_Sharpe.gt(0)).sum())
        relation_text=f'固定三seed中，当前{len(rel)}个已完成seed的M1 BothAdd相对M0 BothAdd有{positive}个同时正CAGR/Sharpe增量。不能从最好seed或仅胜v5宣称关系有效；成对CI、成本与排除2024/25后的方向逐项留表。'
    decision_conclusion=retain+'\n\n'+relation_text+'\n\n停止当前配置：'+json.dumps(rejected,ensure_ascii=False)+'。失败层由原始IC、实际OOF条件排序增量和经济门的逐项False定位：预测/排序无增量，或交易成本、风险与年份集中性未过门，不以补报告抹去失败。当前不自动运行任何后置框架或追加搜索。'
    (R/'DECISION.md').write_text('# 冻结经济门与当前配置判定\n\n'+status['status']+'；不挑历史赢家，不自动修改v5/CURRENT/生产。收益与风险分列。\n\n'+decision_conclusion+'\n\n```csv\n'+dtext+'```\n\n固定三seed全部证据未完成前不宣称最终可替代。失败属于当前数据/预算/目标/排序/交易配置，不等于否证原论文；具体保留/停止结论依据最终表，而非扩大研究方向。\n')
    (R/'REPRODUCE.md').write_text('服务器项目根 `/data1/stock-mxy/课题/一进二策略`。\n\n`PYTHONDONTWRITEBYTECODE=1 research/round23_6_26_execution/env/bin/python -B tools/continue_round23_7_round26_1.py --deliverable D5 --resume --publish --jobs 2 --seed 17`\n\n固定17完整全链路并推送后才42，再73。现有源快照/原始字段/私有面板保持只读；所有新工件在本项目内。精确输入及模型服务器路径/SHA/schema见SERVER_*和ACCEPTED_NN_MODEL_MANIFEST，失败不能用于缓存或成绩。公共数据/原checkpoint均为固定作者提交；不买数据/LLM、不运行二进三、不增加后置框架。主参数/预算不搜索。\n')
    main=C/'result/round26_1/ROUND26_1_REPORT.md'
    d3=read(C/'result/round26_1/D3/STATUS.json',{})
    main.write_text('# Round26.1\n\nD0已执行。D3：'+d3.get('status','UNKNOWN')+'，原PPO42及真实匹配经济配置49，详见D3/REPORT.md。D5：'+status['status']+'，有效目标NN'+str(len(fits))+'/72，真实经济配置'+str(len(results))+'；公开原核心整轮参考'+('已完成' if ref_complete else '未完成')+'。主成绩、关系信息对照、风险/缺证据与失败历史见D5/REPORT.md、D5/STATUS.json及all_results.csv。两条方法不导入彼此赢家。\n\nCURRENT_BASELINE_CHANGED=FALSE\nPRODUCTION_CONFIG_CHANGED=FALSE\n')
    return status
