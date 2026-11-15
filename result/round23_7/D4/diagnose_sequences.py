"""Audit actual invalid observations; no learner, no substitute market fields."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from zyquant.data import DataSnapshot
from core.round23_7.tra_data import SNAP,KEY,CHANNELS,R,traded_history

manifest=json.loads((R/'target_data/server_only_inputs_manifest.json').read_text())
path=Path(next(x['path'] for x in manifest if x['path'].endswith('/candidate_keys.parquet'))).parent
q=pd.read_parquet(path/'candidate_keys.parquet')
invalid=~np.load(path/'valid.npy')
records=[]
if invalid.any():
    snapshot=DataSnapshot(SNAP)
    bars=snapshot.table('daily_raw',start=pd.Timestamp('2010-01-01').date(),end=pd.Timestamp('2026-07-24').date(),cutoff=snapshot.metadata.as_of_date,fields=KEY+['open','close','high','low','volume','amount'])
    bars['trade_date']=pd.to_datetime(bars.trade_date);bars['instrument_id']=bars.instrument_id.astype(str)
    groups=bars.groupby('instrument_id',sort=False)
for instrument,signals in q.loc[invalid].groupby('instrument_id',sort=False):
    stock=traded_history(groups.get_group(instrument)) if instrument in groups.indices else None
    for row in signals.itertuples():
        reasons=[]
        if stock is None or not stock.trade_date.eq(row.trade_date).any():reasons.append('CURRENT_BAR_ABSENT')
        else:
            past=stock.loc[stock.trade_date.le(row.trade_date)].tail(60)
            current=past.iloc[-1]
            if not(current.close>0 and current.volume>0):reasons.append('CURRENT_DENOMINATOR_NONPOSITIVE')
            if (past.volume.le(0)&past.amount.eq(0)).any():reasons.append('VWAP_UNDEFINED_NONTRADING_HISTORY')
            positive=past.loc[past.volume.gt(0)]
            if positive[['open','close','high','low']].isna().any().any():reasons.append('POSITIVE_VOLUME_PRICE_SOURCE_MISSING')
            if positive.amount.isna().any():reasons.append('POSITIVE_VOLUME_AMOUNT_SOURCE_MISSING')
            vwap=positive.amount/positive.volume
            if ((vwap<positive.low-.02)|(vwap>positive.high+.02)).any():reasons.append('VWAP_SOURCE_OUTSIDE_OBSERVED_RANGE')
            if past.volume.isna().any():reasons.append('VOLUME_SOURCE_MISSING')
        if not reasons:reasons.append('OTHER_NONFINITE_SOURCE_REQUIRES_INVESTIGATION')
        records.append({'trade_date':row.trade_date,'instrument_id':row.instrument_id,'reasons':'|'.join(reasons)})
result=pd.DataFrame(records,columns=KEY+['reasons'])
result.to_csv(R/'target_data/invalid_sequence_reasons.csv.gz',index=False)
counts=result.reasons.str.split('|').explode().value_counts().to_dict()
(R/'target_data/invalid_sequence_reason_summary.json').write_text(json.dumps({'data_cache':str(path),'data_COMPLETE_sha256':__import__('hashlib').sha256((path/'COMPLETE.json').read_bytes()).hexdigest(),'raw_window_start':'2010-01-01','invalid_candidate_sequences':len(result),'reason_counts_overlap':counts,'undefined_VWAP_on_no_trade_days_is_not_an_access_permission_failure':True,'no_VWAP_proxy_or_label_based_candidate_deletion':True,'frozen_policy':'whole-day original v5 fallback on incomplete required predictions; invalid candidates remain in forecast universe'},indent=2))
print(json.dumps(counts),flush=True)
