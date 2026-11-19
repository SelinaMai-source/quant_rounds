"""Verified snapshot -> local Qlib feature provider, no external vendor uploads."""
from pathlib import Path
from datetime import date
import json
import importlib.metadata
import numpy as np
import pandas as pd
from core.research_integrity_v5.cache import transaction, fingerprint, identity
C=Path('/data1/stock-mxy/课题/一进二策略')
B=C/'research/round23_7_26_1_execution'
R=C/'result/round26_1/D5'
SNAP=Path('/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features')
KEY=['trade_date','instrument_id']

def recipe():
    return dict(code=[identity(__file__)],third_party={'qlib':importlib.metadata.version('pyqlib'),'zyquant':importlib.metadata.version('zyquant')},dirty_diff='code content identities',environment={'numpy':np.__version__,'pandas':pd.__version__},parameters={'through':'2026-07-24','normalization':'historical initial factor/price','format':'Qlib little endian float32 indexed calendar'},seed=None,features=['open','high','low','close','volume','vwap'],universe='all historically observed equity quotations; graph selection separate',calibration='none',data=[identity(SNAP/'manifest.json')],label='not consumed',maturity='not consumed',windows='2010-01-01..2026-07-24',execution='data provider only',upstream=[identity(R/'CONTRACT.yaml')])

def prepare():
    import pyarrow
    pyarrow.set_cpu_count(2);pyarrow.set_io_thread_count(2)
    from zyquant.data import DataSnapshot
    contract=recipe();fp=fingerprint(contract);root=B/'MASTER_provider'
    with transaction(root,contract) as (dest,hit):
        if not hit:
            snapshot=DataSnapshot(SNAP)
            kwargs=dict(start=date(2010,1,1),end=date(2026,7,24),cutoff=snapshot.metadata.as_of_date)
            raw=snapshot.table('daily_raw',**kwargs,fields=KEY+['open','high','low','close','volume','amount','paused'])
            adj=snapshot.table('daily_post_adjusted',**kwargs,fields=KEY+['close_post','adjustment_factor'])
            for frame in (raw,adj):
                frame.trade_date=pd.to_datetime(frame.trade_date);frame.instrument_id=frame.instrument_id.astype(str)
                assert not frame.duplicated(KEY).any()
            raw=raw.merge(adj,on=KEY,validate='one_to_one',how='left');del adj
            cal=snapshot.table('trade_calendar',**kwargs,fields=['trade_date','exchange'])
            cal=cal.loc[cal.exchange.eq('XSHG'),'trade_date'].drop_duplicates().sort_values()
            dates=pd.DatetimeIndex(cal);mapping={d:i for i,d in enumerate(dates)}
            (dest/'calendars').mkdir();(dest/'features').mkdir();(dest/'instruments').mkdir()
            (dest/'calendars/day.txt').write_text(''.join(d.strftime('%Y-%m-%d')+'\n' for d in dates))
            raw['cal_idx']=raw.trade_date.map(mapping)
            assert raw.cal_idx.notna().all()
            raw=raw.sort_values(['instrument_id','trade_date']).reset_index(drop=True)
            audits=[];instruments=[]
            raw[KEY+['cal_idx','close','volume','amount','paused']].to_parquet(dest/'observed_quotes.parquet',index=False)
            for instrument,stock in raw.groupby('instrument_id',sort=False):
                stock=stock.copy();f=stock.adjustment_factor.to_numpy('float64');price=stock.close.to_numpy('float64')
                finite=np.isfinite(f)&(f>0)&np.isfinite(price)&(price>0)
                if not finite.any():raise ValueError('No initial valid price/factor: '+instrument)
                initial=np.flatnonzero(finite)[0];norm=f/f[initial];initial_close=price[initial]
                valid=np.isfinite(f)&(f>0)&np.isfinite(stock.close_post.to_numpy())
                error=np.abs(stock.close_post.to_numpy()[valid]-price[valid]*f[valid])
                assert np.allclose(stock.close_post.to_numpy()[valid],price[valid]*f[valid],rtol=1e-7,atol=1e-7),instrument
                vwap=stock.amount.to_numpy()/stock.volume.where(stock.volume.gt(0)).to_numpy()
                bad=(vwap<stock.low.to_numpy()-.02)|(vwap>stock.high.to_numpy()+.02)
                vwap[bad]=np.nan
                values={k:stock[k].to_numpy()*norm/initial_close for k in ['open','high','low','close']}
                values['volume']=stock.volume.to_numpy()/norm
                values['vwap']=vwap*norm/initial_close
                # Multiplying every factor by a future global anchor must cancel.
                assert np.allclose(norm,(f*7.3)/(f[initial]*7.3),equal_nan=True)
                idx=stock.cal_idx.to_numpy('int32');lo,hi=int(idx.min()),int(idx.max())
                folder=dest/'features'/instrument.lower();folder.mkdir()
                for name,series in values.items():
                    vector=np.full(hi-lo+2,np.nan,dtype='<f4');vector[0]=lo;vector[1+idx-lo]=series.astype('<f4')
                    vector.tofile(folder/(name+'.day.bin'))
                instruments.append(f'{instrument}\t{dates[lo]:%Y-%m-%d}\t{dates[hi]:%Y-%m-%d}\n')
                audits.append({'instrument_id':instrument,'rows':len(stock),'max_post_adjustment_error':float(error.max()) if len(error) else None,'bad_vwap':int(bad.sum()),'initial_date':str(stock.trade_date.iloc[initial]),'initial_close':initial_close,'initial_factor':float(f[initial])})
                if len(audits)%500==0:print('Provider stocks',len(audits),flush=True)
            (dest/'instruments/all.txt').write_text(''.join(instruments))
            pd.DataFrame(audits).to_csv(dest/'stock_normalization_audit.csv',index=False)
            (dest/'DATA_AUDIT.json').write_text(json.dumps({'status':'PROVIDER_COMPLETE','rows':len(raw),'stocks':len(audits),'calendar_dates':len(dates),'historical_vendor_revision_PIT':'PARTIALLY_VERIFIED','raw_data_not_uploaded':True,'source_snapshot':identity(SNAP/'manifest.json')},indent=2))
    path=root/fp
    for name in ['contract.json','COMPLETE.json','DATA_AUDIT.json','stock_normalization_audit.csv']:
        target=R/'provider';target.mkdir(exist_ok=True)
        __import__('shutil').copy2(path/name,target/name)
    print(path,flush=True)
    return path
if __name__=='__main__':prepare()
