"""Rebuild real historical context graphs after actual future-quote interventions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from core.round26_1.master_store import GraphStore
from core.round26_1.master_provider import B,R
from core.research_integrity_v5.cache import identity

def context(block,day,calendar):
    prior=calendar[calendar<day][-20:]
    history=block.loc[block.trade_date.isin(prior)]
    last=history.loc[history.trade_date.eq(prior[-1])].copy()
    valid=last.close.gt(0)&last.volume.gt(0)&last.amount.gt(0)&~last.paused.fillna(True).astype(bool)&last.instrument_id.str.endswith(('.XSHG','.XSHE'))
    median=history.groupby('instrument_id').amount.median()
    last=last.loc[valid,['instrument_id']].join(median.rename('liquidity'),on='instrument_id').dropna()
    return last.sort_values(['liquidity','instrument_id'],ascending=[False,True],kind='mergesort').head(1000).instrument_id.tolist()

def run(store_path):
    store=GraphStore(store_path);quotes=pd.read_parquet(store.provider/'observed_quotes.parquet')
    ids=json.loads((store.graph_path/'instruments.json').read_text());records=[]
    for year in [2015,2021,2026]:
        i=int(store.indices(f'{year}-01-01',f'{year+1}-01-01')[0]);day=store.dates[i]
        history=store.calendar[store.calendar<day][-20:]
        future=store.calendar[store.calendar>day][:5]
        block=quotes.loc[quotes.trade_date.isin(history.union(future))].copy()
        original=context(block,day,store.calendar)
        changed=block.copy();mask=changed.trade_date.gt(day)
        assert mask.any()
        for column in ['amount','volume','close']:changed.loc[mask,column]=changed.loc[mask,column]+1e9
        changed.loc[mask,'paused']=True
        deleted=block.loc[~mask]
        assert original==context(changed,day,store.calendar)==context(deleted,day,store.calendar)
        q=store.q.loc[store.q.trade_date.eq(day),'instrument_id'].tolist()
        union=sorted(set(original)|set(q))
        lo,hi=store.offsets[i:i+2];stored=[ids[s] for s in store.nodes[lo:hi]]
        assert union==stored,'Actual reconstructed context/target graph differs from frozen graph plan'
        values,qi=store.batch(i,np.zeros(221,dtype='float32'),np.ones(221,dtype='float32'))
        assert np.isfinite(values).all() and values.shape==(len(stored),8,221)
        records.append({'year':year,'day':str(day.date()),'nodes':len(union),'targets':len(q),'context':len(original),'actual_future_quote_rows_perturbed_or_deleted':int(mask.sum()),'all_reconstructed_nodes_bitexact':True,'all_nodes_finite':True,'all_target_nodes_retained_independent_of_label':int((qi>=0).sum())==len(q),'clock':'median amount of prior20 exchange dates, explicit D-1 tradability'})
    result={'status':'PASS_REAL_CONTEXT_REBUILD','cases':records,'graph_plan':identity(store.graph_path/'graph_plan.npz'),'scope':'Real source quotation perturb/delete and complete context re-selection; NN cold fits and feature calculation interventions are separate evidence, not claimed by this test'}
    (R/'REAL_CONTEXT_COUNTERFACTUALS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
    return result
if __name__=='__main__':
    import sys
    run(sys.argv[1])
