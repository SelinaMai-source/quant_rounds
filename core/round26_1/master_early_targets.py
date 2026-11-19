"""Recover 2012 warmup with canonical first-board and exit-label code."""
import json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from core.stock_readonly import ensure_stock_src_on_path
from core.research_integrity_v5.cache import identity
C=Path('/data1/stock-mxy/课题/一进二策略')
B=C/'research/round23_7_26_1_execution/MASTER_early_targets_AI071_boundary_replay'
R=C/'result/round26_1/D5'
SNAP=Path('/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features')
KEY=['trade_date','instrument_id']

def run():
    import pyarrow
    pyarrow.set_cpu_count(2);pyarrow.set_io_thread_count(2)
    ensure_stock_src_on_path()
    from zyquant.data import DataSnapshot
    from first_board_hermes.dataset import _calendar, _compute_daily_features, _attach_direct_features
    from first_board_hermes.candidate_alignment import apply_ai071_candidate_filters
    from first_board_hermes.labels import attach_exit_aligned_labels
    from nshape.universe import resolve_daily_members, eligibility_mask
    B.mkdir(parents=True,exist_ok=True)
    snapshot=DataSnapshot(SNAP)
    kwargs=dict(start=date(2010,1,1),end=date(2015,12,31),cutoff=snapshot.metadata.as_of_date)
    raw=snapshot.table('daily_raw',**kwargs,fields=KEY+['open','high','low','close','pre_close','volume','amount','paused','limit_up','limit_down'])
    adjusted=snapshot.table('daily_post_adjusted',**kwargs,fields=KEY+['close_post'])
    calendar=_calendar(snapshot,date(2010,1,1),date(2015,12,31),snapshot.metadata.as_of_date)
    daily=_compute_daily_features(raw,adjusted,calendar)
    print('Canonical daily features',len(daily),flush=True)
    # Presence in an observed quotation is historical evidence. No delist/list metadata.
    membership=raw.groupby('instrument_id',as_index=False).trade_date.min().rename(columns={'trade_date':'effective_from'})
    membership['effective_to']=pd.NaT;membership['universe_id']='CN_ALL_A'
    st=snapshot.table('special_treatment',start=date(1990,1,1),end=date(2015,12,31),cutoff=snapshot.metadata.as_of_date)
    members=resolve_daily_members(membership,st,calendar.trade_date.dt.date)
    daily['eligible']=eligibility_mask(daily.trade_date,daily.instrument_id,members)
    q=daily.loc[daily.trade_date.ge('2012-01-01')&daily.trade_date.lt('2015-01-01')&daily.is_first_board_hard,KEY+['cal_idx','is_first_board_hard','g7_listing_days_log','eligible']].copy()
    q=q.rename(columns={'eligible':'universe_eligible'})
    q=_attach_direct_features(q,snapshot,date(2010,1,1),date(2015,12,31),snapshot.metadata.as_of_date,include_money_flow=False)
    warm,warm_detail,warm_audit=apply_ai071_candidate_filters(q.loc[q.trade_date.dt.year.eq(2012)],raw,calendar)
    # Reproduce the frozen legacy2013 sample-calendar boundary for overlap audit.
    # Recovery2012 uses genuinely available2011 preceding quotes;2013+ keys stay frozen.
    legacy,legacy_detail,legacy_audit=apply_ai071_candidate_filters(q.loc[q.trade_date.ge('2013-01-01')],raw,calendar.loc[calendar.trade_date.ge('2013-01-01')])
    q=pd.concat([warm,legacy],ignore_index=True)
    detail=pd.concat([warm_detail,legacy_detail],ignore_index=True)
    filter_audit={'warmup2012':warm_audit,'legacy2013_2014':legacy_audit,'calendar_boundary':'2013 overlap replays original sample-calendar start: original first2013 date has no previous_market_date and is rejected.2012 new warmup uses restored true prior2011 quotes. No added2013+ keys.'}
    detail.to_parquet(B/'candidate_filter_detail.parquet',index=False)
    q=q[KEY+['cal_idx']]
    print('Recovered D-known targets',len(q),flush=True)
    labels,audit=attach_exit_aligned_labels(q,daily)
    labels['label_available_at']=pd.to_datetime(labels.actual_exit_date)+pd.Timedelta(hours=15)
    labels.to_parquet(B/'legal_2012_2014_targets.parquet',index=False)
    canonical=pd.read_parquet(C/'result/round22_3/data/ten_candidate.parquet',columns=KEY+['is_first_board_hard','universe_eligible','label_exit_rule_net','actual_exit_date'])
    canonical['trade_date']=pd.to_datetime(canonical.trade_date)
    old=canonical.loc[canonical.trade_date.lt('2015-01-01')&canonical.is_first_board_hard&canonical.universe_eligible]
    overlap=labels.merge(old,on=KEY,how='outer',suffixes=('_new','_old'),indicator=True)
    both=overlap.loc[overlap._merge.eq('both')]
    known=both[['label_exit_rule_net_new','label_exit_rule_net_old']].notna().all(axis=1)
    maxerror=float((both.loc[known,'label_exit_rule_net_new']-both.loc[known,'label_exit_rule_net_old']).abs().max())
    newonly=overlap.loc[overlap._merge.eq('left_only')&overlap.trade_date.ge('2013-01-01')]
    missing=int(overlap._merge.eq('right_only').sum())
    dates_equal=bool((pd.to_datetime(both.actual_exit_date_new).fillna(pd.Timestamp('1900-01-01'))==pd.to_datetime(both.actual_exit_date_old).fillna(pd.Timestamp('1900-01-01'))).all())
    result={'status':'CANONICAL_AI071_LEGAL_RECOVERY_AUDITED','candidate_filter_audit':filter_audit,'recovered_rows':len(labels),'year_counts':labels.groupby(labels.trade_date.dt.year).size().to_dict(),'overlap_rows':len(both),'known_label_max_abs_diff':maxerror,'exit_dates_equal':dates_equal,'2013_2014_new_only':len(newonly),'2013_2014_old_only':missing,'label_audit':audit,'inputs':[identity(SNAP/'manifest.json')],'output':identity(B/'legal_2012_2014_targets.parquet'),'future_listing_or_delist_used':False,'raw_data_not_uploaded':True,'canonical_sources':[identity(Path('/data1/Stock/src/first_board_hermes')/n) for n in ['dataset.py','labels.py','candidate_alignment.py']]}
    (R/'EARLY_TARGET_RECOVERY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    newonly.to_parquet(B/'overlap_new_only.parquet',index=False)
    assert maxerror<=1e-12 and dates_equal and missing==0 and len(newonly)==0, 'Canonical overlap mismatch; preserve audit before further use'
    print(json.dumps(result,ensure_ascii=False,default=str),flush=True)
if __name__=='__main__':run()
