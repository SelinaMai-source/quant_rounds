"""Inherited signed economic gate, with real filled Top1/Top2 identity changes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from core.research_integrity_v2.economic_gate import evaluate_economic_gate,freeze_economic_gate

def decisions(root,results,baseline):
    root=Path(root);freeze_economic_gate(root/'FROZEN_INHERITED_ECONOMIC_GATE.json')
    registry=pd.read_csv(root/'run_manifest.csv').set_index('code')
    idx=results.set_index('experiment_id');base=idx.loc[baseline]
    curves=pd.read_csv(root/'daily_returns.csv.gz',parse_dates=['trade_date']).set_index('trade_date')
    def metrics(z):return {'code':z.name,'cagr':float(z.CAGR),'sharpe':float(z.Sharpe),'nav_end':float(z.TerminalNAV),'max_drawdown':float(z.MaxDD)}
    def plan(code):
        frame=pd.read_parquet(Path(registry.loc[code,'path'])/'overlayed.parquet')
        frame=frame.sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort');frame['rank']=frame.groupby('signal_date').cumcount()+1
        return frame.loc[frame['rank'].le(2),['signal_date','rank','instrument_id']]
    def buys(code):
        costs=pd.read_parquet(Path(registry.loc[code,'path'])/'costs.parquet')
        rows=costs.loc[costs.side.eq('buy'),['cohort_id','instrument_id']].copy();rows['signal_date']=pd.to_datetime(rows.cohort_id.str.slice(0,10),errors='raise')
        return rows[['signal_date','instrument_id']].drop_duplicates()
    old_plan=plan(baseline);old_buys=buys(baseline);rows=[];details={}
    for code,z in idx.loc[idx.scenario.eq('S0')].iterrows():
        if code==baseline:continue
        changed=old_plan.merge(plan(code),on=['signal_date','rank'],how='outer',suffixes=('_baseline','_candidate'),indicator=True)
        changed=changed.loc[changed.instrument_id_baseline.ne(changed.instrument_id_candidate)|changed['_merge'].ne('both')]
        changed=changed.merge(old_buys.rename(columns={'instrument_id':'instrument_id_baseline'}).assign(baseline_actual_buy=True),on=['signal_date','instrument_id_baseline'],how='left',validate='many_to_one')
        changed=changed.merge(buys(code).rename(columns={'instrument_id':'instrument_id_candidate'}).assign(candidate_actual_buy=True),on=['signal_date','instrument_id_candidate'],how='left',validate='many_to_one')
        changed['realized_entry_identity_change']=changed.baseline_actual_buy.eq(True)|changed.candidate_actual_buy.eq(True)
        count=int(changed.realized_entry_identity_change.sum());changed.to_csv(root/f'{code}_actual_top2_plan_changes.csv.gz',index=False)
        stress=[code+'_S1',code+'_S2'];present=all(q in idx.index for q in stress)
        fee=all(float(idx.loc[q,'delta_CAGR'])>0 and float(idx.loc[q,'delta_Sharpe'])>0 for q in stress) if present else None
        outside=~curves.index.year.isin([2024,2025]);excess=np.log1p(curves.loc[outside,code])-np.log1p(curves.loc[outside,baseline])
        year_ok=bool(excess.sum()>0) if len(excess) else None
        gate=evaluate_economic_gate(metrics(z),baseline=metrics(base),top_changed=bool(count),fee_same_direction=fee,not_only_year=year_ok)['CORRECTED']
        verdict='RESEARCH_CANDIDATE_REQUIRES_COMPLETE_EVIDENCE' if gate['economically_material'] else ('INSUFFICIENT_COST_EVIDENCE' if gate['promotion_blocked_missing_evidence'] and all(v is not False for v in gate['flags'].values()) else 'REJECT_CURRENT_CONFIGURATION')
        record={'experiment_id':code,'baseline':baseline,'decision':verdict,'economic_gate_pass':gate['economically_material'],'missing_economic_evidence':gate['promotion_blocked_missing_evidence'],'changed_top1_top2_plan_rows':len(changed),'changed_top1_top2_real_entry_rows':count,'paired_cost_direction':fee,'outside_2024_2025_log_relative_wealth':float(excess.sum()),**gate['flags']}
        rows.append(record);details[code]={**record,'gate':gate,'automatic_promotion':False,'top_change_threshold':'inherited>0, no post-result threshold changes'}
    pd.DataFrame(rows).to_csv(root/'economic_decisions.csv',index=False);(root/'economic_decisions.json').write_text(json.dumps(details,indent=2));return rows
