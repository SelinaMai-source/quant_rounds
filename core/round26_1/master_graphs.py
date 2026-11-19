"""D-known target/context graph plans; labels are a separate joined artifact."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from core.round26_1.master_provider import C,B,R,KEY
from core.research_integrity_v5.cache import identity,transaction,fingerprint

def target_bank():
    from core.round23_7.tra_data import candidates,targets
    from core.round7.contract import TEN_HEAD
    q=candidates();q=q.loc[q.trade_date.ge('2013-01-01')].copy()
    labels=targets(q,TEN_HEAD).rename(columns={'target':'label_exit_rule_net'})
    labels['actual_exit_date']=pd.to_datetime(labels.label_available_at).dt.normalize()
    early=pd.read_parquet(B/'MASTER_early_targets_AI071_boundary_replay/legal_2012_2014_targets.parquet')
    early=early.loc[early.trade_date.dt.year.eq(2012),KEY+['label_exit_rule_net','actual_exit_date']]
    out=pd.concat([early,labels[KEY+['label_exit_rule_net','actual_exit_date']]],ignore_index=True).sort_values(KEY).reset_index(drop=True)
    assert not out.duplicated(KEY).any()
    out['label_available_at']=pd.to_datetime(out.actual_exit_date)+pd.Timedelta(hours=15)
    return out

def prepare(provider):
    provider=Path(provider)
    inputs=[provider/'COMPLETE.json',B/'MASTER_early_targets_AI071_boundary_replay/legal_2012_2014_targets.parquet']+[C/f'result/round22_3/data/{head}_{kind}.parquet' for head in ['five','ten'] for kind in ['training','candidate']]
    contract=dict(code=[identity(__file__),identity(C/'core/round23_7/tra_data.py')],third_party='no learned graph selection',dirty_diff='content identity',environment={'numpy':np.__version__,'pandas':pd.__version__},parameters={'context':1000,'median_window':20,'min_periods':1,'lag_calendar':1,'liquidity_ranking_dtype':'float64'},seed=None,features='amount20 median on observed exchange calendar through D-1; no future values',universe='all target keys + SH/SZ D-1 tradable liquidity top1000',calibration='none',data=[identity(p) for p in inputs],label='Ten canonical exit net, independent column',maturity='actual_exit_date15:00',windows='2012..2026-07-24',execution='graph planning only',upstream=[identity(R/'CONTRACT.yaml'),identity(R/'TARGET_DOMAIN_CORRECTION_FREEZE.json'),identity(R/'EARLY_TARGET_RECOVERY.json')])
    fp=fingerprint(contract);root=B/'MASTER_graphs'
    with transaction(root,contract) as (dest,hit):
        if not hit:
            q=target_bank();q.to_parquet(dest/'targets.parquet',index=False)
            dates=pd.DatetimeIndex(pd.to_datetime((provider/'calendars/day.txt').read_text().splitlines()))
            ids=[s.split('\t')[0] for s in (provider/'instruments/all.txt').read_text().splitlines()];idmap={s:i for i,s in enumerate(ids)}
            q['day_idx']=dates.get_indexer(q.trade_date);q['stock_idx']=q.instrument_id.map(idmap)
            assert (q.day_idx>=0).all() and q.stock_idx.notna().all()
            quotes=pd.read_parquet(provider/'observed_quotes.parquet')
            liquidity=np.full((len(dates),len(ids)),np.nan,dtype='float64')
            eligible=np.zeros((len(dates),len(ids)),dtype=bool)
            for instrument,stock in quotes.groupby('instrument_id',sort=False):
                if not instrument.endswith(('.XSHG','.XSHE')):continue
                column=idmap[instrument];index=stock.cal_idx.to_numpy('int32')
                amount=pd.Series(np.nan,index=np.arange(len(dates)))
                amount.iloc[index]=stock.amount.to_numpy()
                liquidity[:,column]=amount.rolling(20,min_periods=1).median().to_numpy('float64')
                eligible[index,column]=(stock.close.gt(0)&stock.amount.gt(0)&stock.volume.gt(0)&~stock.paused.fillna(True).astype(bool)).to_numpy()
            qgroups=q.groupby('day_idx',sort=False).indices
            nodes=[];target_indices=[];offsets=[0];dayindices=[];counts=[]
            lexical=np.array(ids)
            for day_idx in np.flatnonzero((dates>='2012-01-01')&(dates<='2026-07-24')):
                if day_idx==0:continue
                valid=np.flatnonzero(eligible[day_idx-1]&np.isfinite(liquidity[day_idx-1]))
                order=np.lexsort((lexical[valid],-liquidity[day_idx-1,valid]));context=valid[order[:1000]]
                qi=np.array(qgroups.get(day_idx,[]),dtype='int32')
                target=q.loc[qi,'stock_idx'].to_numpy('int32')
                union=np.unique(np.concatenate([context,target])).astype('int32')
                # Canonical order by instrument_id rather than liquidity ranking.
                union=union[np.argsort(lexical[union],kind='stable')]
                lookup={int(s):int(i) for i,s in enumerate(union)}
                tq=np.full(len(union),-1,dtype='int32')
                for row,s in zip(qi,target):tq[lookup[int(s)]]=row
                nodes.extend(union);target_indices.extend(tq);offsets.append(len(nodes));dayindices.append(day_idx)
                counts.append({'trade_date':str(dates[day_idx].date()),'nodes':len(union),'context':len(context),'targets':len(target),'context_target_overlap':len(set(context)&set(target))})
            np.savez(dest/'graph_plan.npz',nodes=np.array(nodes,dtype='int32'),target_indices=np.array(target_indices,dtype='int32'),offsets=np.array(offsets,dtype='int64'),day_indices=np.array(dayindices,dtype='int32'))
            (dest/'instruments.json').write_text(json.dumps(ids));pd.DataFrame(counts).to_csv(dest/'GRAPH_COUNTS.csv',index=False)
            (dest/'GRAPH_AUDIT.json').write_text(json.dumps({'graphs':len(dayindices),'nodes_total':len(nodes),'target_rows':len(q),'2012_recovered_targets':int(q.trade_date.dt.year.eq(2012).sum()),'selection_reads_labels':False,'future_delist_or_current_constituents_used':False,'2013_onward_existing_frozen_keys_only':True},indent=2))
    path=root/fp
    for name in ['GRAPH_COUNTS.csv','GRAPH_AUDIT.json','contract.json','COMPLETE.json']:
        target=R/'graphs';target.mkdir(exist_ok=True);__import__('shutil').copy2(path/name,target/name)
    print(path,flush=True);return path
if __name__=='__main__':
    import sys
    prepare(sys.argv[1])
