"""Frozen signed economic gates against the actual matched baseline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from core.research_integrity_v2.economic_gate import evaluate_economic_gate,freeze_economic_gate


def decisions(root,results):
    root=Path(root)
    freeze_economic_gate(root/'FROZEN_INHERITED_ECONOMIC_GATE.json')
    registry=pd.read_csv(root/'run_manifest.csv').set_index('code')
    idx=results.set_index('experiment_id')
    curves=pd.read_csv(root/'daily_returns.csv.gz',parse_dates=['trade_date']).set_index('trade_date')
    base=idx.loc['AG_A0']
    def metrics(row):
        return {'code':row.name,'cagr':float(row.CAGR),'sharpe':float(row.Sharpe),'nav_end':float(row.TerminalNAV),'max_drawdown':float(row.MaxDD)}
    def plan(code):
        frame=pd.read_parquet(Path(registry.loc[code,'path'])/'overlayed.parquet')
        frame=frame.loc[pd.to_datetime(frame.signal_date).between('2023-01-03','2026-07-24')].sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort')
        frame['rank']=frame.groupby('signal_date').cumcount()+1
        return frame.loc[frame['rank'].le(2),['signal_date','rank','instrument_id']]
    baseline_plan=plan('AG_A0');rows=[];details={}
    for code,z in idx.loc[idx.scenario.eq('S0')].iterrows():
        if code=='AG_A0':continue
        changed=baseline_plan.merge(plan(code),on=['signal_date','rank'],how='outer',suffixes=('_baseline','_candidate'),indicator=True)
        changed=changed.loc[changed.instrument_id_baseline.ne(changed.instrument_id_candidate)|changed['_merge'].ne('both')]
        changed.to_csv(root/f'{code}_actual_top2_plan_changes.csv.gz',index=False)
        # Secondary Add/Remove/Replace have no paired cost run by protocol;
        # that evidence stays unknown, never defaults to True.
        stress=[code+'_S1',code+'_S2']
        has_cost=all(q in idx.index for q in stress)
        fee=all(float(idx.loc[q,'delta_CAGR'])>0 and float(idx.loc[q,'delta_Sharpe'])>0 for q in stress) if has_cost else None
        outside=~curves.index.year.isin([2024,2025])
        excess=np.log1p(curves.loc[outside,code])-np.log1p(curves.loc[outside,'AG_A0'])
        year_ok=bool(excess.sum()>0) if len(excess) else None
        corrected=evaluate_economic_gate(metrics(z),baseline=metrics(base),top_changed=bool(len(changed)),fee_same_direction=fee,not_only_year=year_ok)['CORRECTED']
        verdict='RESEARCH_CANDIDATE_REQUIRES_COMPLETE_EVIDENCE' if corrected['economically_material'] else ('INSUFFICIENT_COST_EVIDENCE' if corrected['promotion_blocked_missing_evidence'] and all(v is not False for v in corrected['flags'].values()) else 'REJECT_CURRENT_CONFIGURATION')
        details[code]={'economic_gate':corrected,'verdict':verdict,'changed_top1_top2_rows':len(changed),'outside_2024_2025_log_relative_wealth':float(excess.sum()),'paired_cost_runs_present':has_cost,'baseline_scope':'matched2023-01-03..2026-07-24,300000,legacy execution','risk_improvements_reported_separately':True,'automatic_promotion':False}
        rows.append({'experiment_id':code,'decision':verdict,'economic_gate_pass':corrected['economically_material'],'missing_economic_evidence':corrected['promotion_blocked_missing_evidence'],'changed_top1_top2_rows':len(changed),'paired_cost_direction':fee,'outside_2024_2025_log_relative_wealth':float(excess.sum()),**corrected['flags']})
    (root/'economic_decisions.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
    pd.DataFrame(rows).to_csv(root/'economic_decisions.csv',index=False)
    return rows
