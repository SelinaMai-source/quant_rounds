"""D-known candidate sequences and separate, exit-mature target records for TRA."""
from pathlib import Path
import importlib.metadata,json,subprocess,hashlib
import numpy as np
import pandas as pd
from core.research_integrity_v5.cache import identity,source_manifest,fingerprint,transaction
from core.round7.contract import FIVE_HEAD,TEN_HEAD
from core.round23_7.tra_reference import C,B,R

SNAP=Path('/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features')
KEY=['trade_date','instrument_id'];CHANNELS=['open','close','high','low','volume','vwap']

def candidates():
    parts=[]
    for head in [FIVE_HEAD,TEN_HEAD]:
        p=pd.read_parquet(C/f'result/round22_3/data/{head.name}_candidate.parquet',columns=KEY+['is_first_board_hard','universe_eligible'])
        p=p.loc[p.is_first_board_hard.fillna(False).astype(bool)&p.universe_eligible.fillna(False).astype(bool),KEY]
        p['trade_date']=pd.to_datetime(p.trade_date);p['instrument_id']=p.instrument_id.astype(str)
        parts.append(p)
    q=pd.concat(parts).drop_duplicates(KEY).sort_values(KEY).reset_index(drop=True)
    assert not q.duplicated(KEY).any()
    return q

def targets(q,head):
    parts=[]
    for kind in ['training','candidate']:
        p=pd.read_parquet(C/f'result/round22_3/data/{head.name}_{kind}.parquet',columns=KEY+[head.target,head.label_end_column])
        p['trade_date']=pd.to_datetime(p.trade_date);p['instrument_id']=p.instrument_id.astype(str)
        p=p.rename(columns={head.target:'target',head.label_end_column:'exit_date'})
        p['exit_date']=pd.to_datetime(p.exit_date)
        assert not p.duplicated(KEY).any()
        parts.append(p)
    common=parts[0].merge(parts[1],on=KEY,suffixes=('_train','_candidate'),validate='one_to_one')
    both=common.target_train.notna()&common.target_candidate.notna()
    conflicts=common.loc[both&~np.isclose(common.target_train,common.target_candidate,rtol=0,atol=1e-12,equal_nan=True)]
    assert conflicts.empty,f'{head.name}: actual immutable training/candidate target version conflict'
    both_exit=common.exit_date_train.notna()&common.exit_date_candidate.notna()
    assert common.loc[both_exit,'exit_date_train'].equals(common.loc[both_exit,'exit_date_candidate'].rename('exit_date_train'))
    label=parts[0].set_index(KEY).combine_first(parts[1].set_index(KEY)).reset_index()
    label=q.merge(label,on=KEY,how='left',validate='one_to_one')
    label['label_available_at']=label.exit_date.dt.normalize()+pd.Timedelta(hours=15)
    return label[KEY+['target','label_available_at']]

def contract():
    import zyquant
    inputs=[SNAP/'manifest.json',R/'CONTRACT.yaml']+sorted((SNAP/'daily_raw').rglob('*.parquet'))
    inputs += [C/f'result/round22_3/data/{h}_{kind}.parquet' for h in ['five','ten'] for kind in ['training','candidate']]
    return {'code':source_manifest([Path(__file__),C/'core/research_integrity_v5/cache.py',C/'core/round7/contract.py',Path(zyquant.__file__).parent]), 'third_party':{'zyquant':'2.0.10 existing API, no upgrades'},'dirty_diff':{n:hashlib.sha256(subprocess.check_output(['git','-C',str(B/n),'diff','HEAD'])).hexdigest() for n in ['round23_7','round26_1']},'environment':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},'parameters':{'sequence':60,'channels':CHANNELS,'prices':'each past price/current observed close -1','volume':'each past volume/current observed volume -1','raw_window':['2014-10-01','2026-07-24'],'missing':'invalid sequence, no semantic replacement; whole-day downstream fallback'},'seed':17,'features':CHANNELS,'universe':'union of then-known hard first-board/universe-eligible head candidate keys; never read target in candidate selector','calibration':'per-window observed final close and volume, no fitted future statistics','data':[identity(p) for p in inputs],'label':'Five label_e2_net; Ten label_exit_rule_net; original training/candidate labels checked for conflicts','maturity':'Five next2_date15:00; Ten actual_exit_date15:00; local exchange event time; unavailable remains NaT','windows':'full stock daily history before mapping candidates; target years2018..2026','execution':'data preparation only, frozen original executor is downstream','upstream':[]}

def prepare():
    from zyquant.data import DataSnapshot
    c=contract();fp=fingerprint(c);root=B/'TRA_target_data'
    with transaction(root,c) as (dest,cached):
        if not cached:
            q=candidates();q=q.loc[q.trade_date.ge('2015-01-01')&q.trade_date.le('2026-07-24')].reset_index(drop=True)
            snapshot=DataSnapshot(SNAP)
            bars=snapshot.table('daily_raw',start=pd.Timestamp('2014-10-01').date(),end=pd.Timestamp('2026-07-24').date(),cutoff=snapshot.metadata.as_of_date,fields=KEY+['open','close','high','low','volume','amount'])
            bars['trade_date']=pd.to_datetime(bars.trade_date);bars['instrument_id']=bars.instrument_id.astype(str)
            assert not bars.duplicated(KEY).any()
            bars['vwap']=bars.amount/bars.volume.where(bars.volume.gt(0))
            inconsistent=bars.volume.gt(0)&((bars.vwap<bars.low-.02)|(bars.vwap>bars.high+.02))
            bars.loc[inconsistent,'vwap']=np.nan
            features=np.full((len(q),60,6),np.nan,dtype=np.float32)
            history=np.full((len(q),60),-1,dtype=np.int32)
            valid=np.zeros(len(q),dtype=bool);source_groups=bars.groupby('instrument_id',sort=False)
            for instrument,group in q.groupby('instrument_id',sort=False):
                if instrument not in source_groups.indices:continue
                stock=source_groups.get_group(instrument).sort_values('trade_date')
                dates=stock.trade_date.to_numpy('datetime64[ns]');values=stock[CHANNELS].to_numpy('float64')
                positions=np.searchsorted(dates,group.trade_date.to_numpy('datetime64[ns]'))
                signal_at={int(loc):int(idx) for idx,loc in zip(group.index,positions) if loc<len(dates) and dates[loc]==q.loc[idx,'trade_date'].to_datetime64()}
                for idx,loc in zip(group.index,positions):
                    if loc>=len(dates) or dates[loc]!=q.loc[idx,'trade_date'].to_datetime64():continue
                    stop=int(loc)+1;start=max(0,stop-60);n=stop-start
                    raw=values[start:stop].copy();close=values[loc,1];volume=values[loc,4]
                    if not(np.isfinite(raw).all() and close>0 and volume>0):continue
                    raw[:,[0,1,2,3,5]]=raw[:,[0,1,2,3,5]]/close-1
                    raw[:,4]=raw[:,4]/volume-1
                    features[idx]=0;features[idx,-n:]=raw.astype('float32')
                    for offset,bar in enumerate(range(start,stop),start=60-n):history[idx,offset]=signal_at.get(bar,-1)
                    valid[idx]=True
            np.save(dest/'features.npy',features);np.save(dest/'history_candidate_indices.npy',history);np.save(dest/'valid.npy',valid)
            q.to_parquet(dest/'candidate_keys.parquet',index=False)
            for head in [FIVE_HEAD,TEN_HEAD]:targets(q,head).to_parquet(dest/f'{head.name}_targets.parquet',index=False)
            audit={'candidate_rows':len(q),'candidate_start':str(q.trade_date.min()),'candidate_end':str(q.trade_date.max()),'candidate_selection_reads_target':False,'raw_rows':len(bars),'raw_start':str(bars.trade_date.min()),'raw_end':str(bars.trade_date.max()),'channels':CHANNELS,'valid_sequences':int(valid.sum()),'invalid_sequences':int((~valid).sum()),'missing_policy':'whole-day v5 fallback, not label-based row deletion','VWAP_inconsistent_source_rows_masked':int(inconsistent.sum()),'full_stock_history_before_candidate_mapping':True,'stock_boundary_clipped':True,'post_adjustment_fields_consumed':[],'current_listing_list_used':False,'vendor_revision_PIT_proven':False,'snapshot_manifest':identity(SNAP/'manifest.json')}
            (dest/'DATA_AUDIT.json').write_text(json.dumps(audit,indent=2))
    path=root/fp
    exported=R/'target_data';exported.mkdir(parents=True,exist_ok=True)
    for name in ['contract.json','COMPLETE.json','DATA_AUDIT.json']:__import__('shutil').copy2(path/name,exported/name)
    records=[]
    for p in sorted(path.iterdir()):
        if not p.is_file():continue
        item=identity(p)
        if p.suffix=='.npy':
            a=np.load(p,mmap_mode='r');item['array_shape']=list(a.shape);item['array_dtype']=str(a.dtype)
        records.append(item)
    (exported/'server_only_inputs_manifest.json').write_text(json.dumps(records,indent=2))
    print('TRA_TARGET_DATA',path,'cache_hit',cached,flush=True)
    return path

if __name__=='__main__':prepare()
