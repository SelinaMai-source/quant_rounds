"""Finalize only after all six real deliveries, and publish both authorized branches."""
from pathlib import Path
import csv,hashlib,json,shutil,subprocess,sys
from datetime import datetime,timezone
C=Path('/data1/stock-mxy/课题/一进二策略');sys.path.insert(0,str(C));sys.dont_write_bytecode=True
B=C/'research/round23_7_26_1_execution';M=C/'result/round23_7_round26_1_master'
OWN={'D0':'round23_7','D1':'round23_7','D2':'round23_7','D3':'round26_1','D4':'round23_7','D5':'round26_1'}
PROTECTED={'baseline_v5':'feda6870a26f6efe5d722fd9c3b11130ca266d85','round23_5':'2c28c2083097a28cda921c59c28b254c30cd0976','round23_6':'efcb7da982566e6aaf1e0be8a1ce8353835569ac','round26':'ec1816154347622c4ecf401c65489c9a1479f5b7'}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(w,*args):return subprocess.check_output(['git','-C',str(w),*args])
def main():
    states={d:json.loads((C/'result'/branch/d/'STATUS.json').read_text()) for d,branch in OWN.items()}
    for d,state in states.items():
        assert state['IMPLEMENTED'] is True and state['UNIT_TESTED'] is True,(d,state)
        assert state['BACKTEST_COMPLETE'] is True and state['EVIDENCE_COMPLETE'] is True,(d,state)
    assert states['D3']['status']=='D3_COMPLETE' and states['D4']['status']=='D4_COMPLETE' and states['D5']['status']=='D5_COMPLETE'
    assert states['D5']['accepted_target_NN_models']==72
    w=B/'round23_7'
    for branch,parent in {'round23_7':PROTECTED['round23_6'],'round26_1':PROTECTED['round26']}.items():
        git(B/branch,'merge-base','--is-ancestor',parent,'HEAD')
    protected={ref:git(w,'ls-remote','origin','refs/heads/'+ref).decode().split()[0] for ref in PROTECTED}
    assert protected==PROTECTED,protected
    assert (C/'result/round23_7/无权限数据.md').stat().st_size==0
    permissions=(C/'result/round26_1/无权限数据.md').read_text()
    lines=permissions.splitlines();assert len(lines)==2 and all(line.startswith(f'{i}. ') and '，数据路径：' in line for i,line in enumerate(lines,1))
    (M/'无权限数据.md').write_text(permissions)
    (M/'DELIVERY_STATES.json').write_text(json.dumps(states,ensure_ascii=False,indent=2))
    # Exact tables, with scope preserved; no cross-window performance ranking.
    tables=[]
    for d in ['D1','D2','D3','D4','D5']:
        branch=OWN[d];source=C/'result'/branch/d/'all_results.csv'
        with source.open() as f:
            for row in csv.DictReader(f):tables.append({'deliverable':d,'branch':branch,**row})
    fields=list(dict.fromkeys(k for row in tables for k in row))
    with (M/'ALL_ACTUAL_RESULTS.csv').open('w') as f:
        out=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");out.writeheader();out.writerows(tables)
    report=['# Round23.7 与 Round26.1 最终审计','',f'生成时间 UTC：{datetime.now(timezone.utc).isoformat()}。六项均以实际 STATUS、模型和原生账本完成验收；失败、无效和中断记录保留，不计作有效成果。','',
    'D0：公共缓存、成熟时间、原生账本及 SL/BOA 诊断完成，历史供应商 vintage 与旧成交假设局限分项保留。D1：AnchorRank N1/N2 完成，当前配置拒绝。D2：四单专家顶点先通过，再运行共享账户 P-EQUAL/P-FIXED 和 P-SCORE，当前经济门未通过。D3：42 个原 PPO 完整预算模型、6 个全预算冷重训、三 seed 及等权复合因子的实际 Add/Remove/Replace 和风险审计完成。D4：公开完整 TRA 参考、162 个正式及12个冷重训模型、24个经济与7个因果诊断原生账户完成；六个 T0/T1 配置拒绝。D5：公开原 MASTER 一整轮参考及三 seed 共72个实际年度 OOF 模型、12个冷重训、24个后端全配置冷重训和12个原生因果账户完成；最终经济判定和 M1/M0 关系对照见下列原表及 D5/DECISION.md。','',
    'D3 使用 2023–2026 统一嵌套 OOF 匹配窗口；D1/D2/D4/D5 使用各自明确匹配的 2018–2026 开发期窗口，不跨窗口把 CAGR 排名当方法优劣。D3 等权 RAW ReplaceFive 的历史增量尚缺预冻结协议未运行的次级成本证据，仍为 INSUFFICIENT_COST_EVIDENCE，不事后补搜或自动升级。','',
    'MASTER 公开参考沿用锁定 main 的一个完整 epoch 示例预算，2979 次真实更新；不是论文40轮最高规模数值复现。A股 M1 原跨股票层与 M0 identity mask 对照按固定三 seed 全部报告，不能选最好 seed。真正历史 OOF 进入原 XGB 后端，未知标签不决定图节点。此前 gross2012 错误域训练、归一化错误和中断均隔离保留。','',
    '共同证据边界：2018–2026 已反复用于开发；没有未触碰前瞻证明。供应商历史非恒定修订及首次可得时点尚缺完整 vintage；原 D+1 开盘缺口通道、高低价成交假设保留；未提供企业行动事件不等于覆盖完整。旧阶段 CPU/RSS 未实测缺口及 TRA 原一次内存断言的未解释失败如实保留。不能把 parity 等同于无前视或实盘成交证明。','',
    'AlphaForge、DoubleAdapt、RD-Agent、DeepLOB、RSR、完整 FLEXPOOL 为 DEFERRED_BY_PROTOCOL；没有重新安装、训练或把未做宣称证伪。不运行二进三，不自动扩展后置研究。','',
    '实际状态与数量：DELIVERY_STATES.json。全部真实成绩：ALL_ACTUAL_RESULTS.csv。各原方法明细、经济门和仍缺证据：对应 Dn/REPORT.md、DECISION.md 及源 CSV。无权限数据.md 仅两条已实际认证失败的2010–2012暖启动源；没有把网络、计算时间或代码缺陷列为权限问题。','',
    '主报告：result/round23_7/ROUND23_7_REPORT.md；result/round26_1/ROUND26_1_REPORT.md。',
    'CURRENT_BASELINE_CHANGED=FALSE','PRODUCTION_CONFIG_CHANGED=FALSE','']
    # Include all S0 scores and all explicit decisions, never pick the best seed.
    for d in ['D1','D2','D3','D4','D5']:
        report.extend([f'## {d} 全部 S0 实际成绩','', '|配置|CAGR|Sharpe|MaxDD|完成交易|','|---|---:|---:|---:|---:|'])
        for r in tables:
            if r['deliverable']==d and r.get('scenario')=='S0':report.append(f"|{r['experiment_id']}|{float(r['CAGR']):.6f}|{float(r['Sharpe']):.6f}|{float(r['MaxDD']):.6f}|{r['trades']}|")
        decision=C/'result'/OWN[d]/d/'economic_decisions.csv'
        if decision.exists():
            report.extend(['','经济门原表（UNKNOWN不算通过）：','','```csv',decision.read_text(),'```',''])
    (M/'MASTER_REPORT.md').write_text('\n'.join(report))
    (M/'protected_refs_and_shared_hashes_check.json').write_text(json.dumps({'observed_UTC':datetime.now(timezone.utc).isoformat(),'protected_remote_refs_unchanged':protected,'CURRENT_BASELINE_CHANGED':False,'PRODUCTION_CONFIG_CHANGED':False,'scope':'protected Git refs; no invented retroactive production-file baseline hashes'},indent=2))
    (M/'ARTIFACT_INDEX.json').write_text(json.dumps([{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(M.iterdir()) if p.is_file() and p.name in ['MASTER_REPORT.md','DELIVERY_STATES.json','ALL_ACTUAL_RESULTS.csv','无权限数据.md','protected_refs_and_shared_hashes_check.json','finalize_all.py']],ensure_ascii=False,indent=2))
    for branch in ['round23_7','round26_1']:
        out=C/'result'/branch/'master_summary';out.mkdir(exist_ok=True)
        for name in ['MASTER_REPORT.md','DELIVERY_STATES.json','ALL_ACTUAL_RESULTS.csv','无权限数据.md','protected_refs_and_shared_hashes_check.json','ARTIFACT_INDEX.json','finalize_all.py']:shutil.copy2(M/name,out/name)
        if (M/'failures').exists():shutil.copytree(M/'failures',out/'failures',dirs_exist_ok=True)
    from core.research_integrity_v5.publish import publish
    verified={branch:publish(branch,'Finalize D0-D5 real results, limitations and cross-branch audit') for branch in ['round23_7','round26_1']}
    shared={}
    for p in sorted((C/'core/research_integrity_v5').glob('*.py')):
        one=git(B/'round23_7','show','HEAD:core/research_integrity_v5/'+p.name);two=git(B/'round26_1','show','HEAD:core/research_integrity_v5/'+p.name)
        assert one==two==p.read_bytes(),p.name
        shared[p.name]=hashlib.sha256(one).hexdigest()
    assert verified['round23_7']['commit']!=verified['round26_1']['commit']
    for ref,expected in PROTECTED.items():assert git(w,'ls-remote','origin','refs/heads/'+ref).decode().split()[0]==expected
    final={'observed_UTC':datetime.now(timezone.utc).isoformat(),'branches':verified,'shared_source_hashes_equal':shared,'protected_remote_refs_unchanged':protected,'all_six_deliveries_complete':True,'CURRENT_BASELINE_CHANGED':False,'PRODUCTION_CONFIG_CHANGED':False}
    (M/'FINAL_REMOTE_VERIFICATION.json').write_text(json.dumps(final,ensure_ascii=False,indent=2));print(json.dumps(final,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
