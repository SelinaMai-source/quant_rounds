"""Disk-backed original feature tensors and label-independent graph batches."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from core.round26_1.master_provider import B,R
from core.research_integrity_v5.cache import identity,transaction,fingerprint,hit

def build(provider,graphs,market):
    provider,graphs,market=map(Path,(provider,graphs,market))
    manifest=R/'ALPHA158_FEATURE_MANIFEST.csv'
    records=pd.read_csv(manifest)
    c=dict(code=[identity(__file__)],third_party='original Qlib Alpha158 and market63 artifacts',dirty_diff='content identity',environment={'numpy':np.__version__,'pandas':pd.__version__},parameters={'axis':['stock','exchange_day','feature'],'dtype':'float32'},seed=None,features='158 original + 63 original market, unnormalized',universe='graph plan fixed independently of targets',calibration='none; train-only normalizer applied later',data=[identity(manifest),identity(provider/'COMPLETE.json'),identity(graphs/'COMPLETE.json'),identity(market/'COMPLETE.json')],label='separate target array',maturity='separate actual exit timestamp',windows='historical 8 exchange-day slots, no forward padding',execution='input tensors only',upstream=[identity(R/'CONTRACT.yaml')])
    fp=fingerprint(c);root=B/'MASTER_store'
    with transaction(root,c) as (dest,cached):
        if not cached:
            dates=pd.DatetimeIndex(pd.to_datetime((provider/'calendars/day.txt').read_text().splitlines()))
            ids=json.loads((graphs/'instruments.json').read_text());idmap={s:i for i,s in enumerate(ids)}
            assert set(ids)==set(records.instrument_id)
            array=np.lib.format.open_memmap(dest/'equity.npy',mode='w+',dtype='float32',shape=(len(ids),len(dates),158))
            for n,row in enumerate(records.itertuples(index=False)):
                path=Path(row.path);assert hit(path,row.fingerprint),f'Incomplete feature stock {row.instrument_id}'
                frame=pd.read_parquet(path/'features.parquet').droplevel('instrument')
                day=dates.get_indexer(pd.DatetimeIndex(frame.index));assert (day>=0).all()
                stock=idmap[row.instrument_id];array[stock]=np.nan;array[stock,day,:]=frame.to_numpy('float32')
                if (n+1)%100==0:
                    array.flush();array._mmap.madvise(__import__('mmap').MADV_DONTNEED)
                    print('Feature store stocks',n+1,flush=True)
            array.flush();del array
            marketframe=pd.read_parquet(market/'market63.parquet').reindex(dates)
            assert marketframe.notna().any(axis=1).all()
            np.save(dest/'market.npy',marketframe.to_numpy('float32'))
            (dest/'STORE.json').write_text(json.dumps({'provider':str(provider),'graphs':str(graphs),'market':str(market),'shape':[len(ids),len(dates),158],'sequence_clock':'last8 exchange dates includingD; absent historical stock slots remain missing then Fillna0; no bfill from afterD'},indent=2))
    return root/fp

class GraphStore:
    def __init__(self,path):
        self.path=Path(path);self.meta=json.loads((self.path/'STORE.json').read_text())
        self.graph_path=Path(self.meta['graphs']);plan=np.load(self.graph_path/'graph_plan.npz')
        self.nodes=plan['nodes'];self.qidx=plan['target_indices'];self.offsets=plan['offsets'];self.dayindices=plan['day_indices']
        self.provider=Path(self.meta['provider'])
        self.calendar=pd.DatetimeIndex(pd.to_datetime((self.provider/'calendars/day.txt').read_text().splitlines()))
        self.dates=self.calendar[self.dayindices]
        self.q=pd.read_parquet(self.graph_path/'targets.parquet')
        self.features=np.load(self.path/'equity.npy',mmap_mode='r');self.market=np.load(self.path/'market.npy',mmap_mode='r')
    def indices(self,start,end):
        return np.flatnonzero((self.dates>=start)&(self.dates<end))
    def current(self,index):
        lo,hi=self.offsets[index:index+2];stock=self.nodes[lo:hi];day=self.dayindices[index]
        return np.concatenate([self.features[stock,day,:],np.broadcast_to(self.market[day],(len(stock),63))],axis=1)
    def batch(self,index,mean,std):
        lo,hi=self.offsets[index:index+2];stock=self.nodes[lo:hi];day=int(self.dayindices[index]);past=np.arange(day-7,day+1)
        assert past[0]>=0
        equity=self.features[stock[:,None],past[None,:],:]
        market=np.broadcast_to(self.market[past],(len(stock),8,63))
        values=np.concatenate([equity,market],axis=2)
        values=(values-mean)/std;values=np.clip(values,-3,3)
        # Original Fillna(0); inf handling has already been audited by feature tests.
        values[np.isnan(values)]=0
        assert np.isfinite(values).all()
        return values.astype('float32',copy=False),self.qidx[lo:hi]
