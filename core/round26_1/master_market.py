"""Actual three-index market63, evaluated by original Qlib operators."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from core.round26_1.master_provider import B,R
from core.research_integrity_v5.cache import identity,transaction,fingerprint

def run():
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    sources=[B/'MASTER_index_sources'/f'{code}.parquet' for code in ['000300','000905','000906']]
    fields=['$close/Ref($close,1)-1']
    for window in [5,10,20,30,60]:
        fields += [f'Mean($close/Ref($close,1)-1,{window})',f'Std($close/Ref($close,1)-1,{window})',f'Mean($amount,{window})/$amount',f'Std($amount,{window})/$amount']
    contract=dict(code=[identity(__file__)],third_party={'qlib':__import__('importlib.metadata').metadata.version('pyqlib')},dirty_diff='content identity',environment={'numpy':np.__version__,'pandas':pd.__version__},parameters={'indices':['000300','000905','000906']},seed=None,features=fields,universe='actual index observations',calibration='none before fold-specific RobustZScoreNorm',data=[identity(s) for s in sources],label='not read',maturity='not read',windows='2010-01-04..2026-07-24',execution='feature preparation',upstream=[identity(R/'CONTRACT.yaml')])
    fp=fingerprint(contract);root=B/'MASTER_market63'
    with transaction(root,contract) as (dest,hit):
        if not hit:
            frames=[pd.read_parquet(s).set_index('trade_date').sort_index() for s in sources]
            dates=frames[0].index
            assert all(dates.equals(f.index) for f in frames)
            provider=dest/'provider';(provider/'calendars').mkdir(parents=True);(provider/'instruments').mkdir();(provider/'features').mkdir()
            (provider/'calendars/day.txt').write_text(''.join(d.strftime('%Y-%m-%d')+'\n' for d in dates))
            instruments=['SH000300','SH000905','SH000906']
            (provider/'instruments/all.txt').write_text(''.join(f'{s}\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\n' for s in instruments))
            for instrument,frame in zip(instruments,frames):
                folder=provider/'features'/instrument.lower();folder.mkdir()
                for col in ['close','amount']:
                    np.concatenate([np.array([0],dtype='<f4'),frame[col].to_numpy('<f4')]).tofile(folder/(col+'.day.bin'))
            qlib.init(provider_uri=str(provider),region=REG_CN,kernels=1,joblib_backend='threading',expression_cache=None,dataset_cache=None)
            blocks=[]
            for instrument in instruments:
                data=D.features([instrument],fields,start_time=dates[0],end_time=dates[-1],freq='day').droplevel('instrument')
                data.columns=[instrument+':'+f for f in fields];blocks.append(data)
            market=pd.concat(blocks,axis=1)
            assert market.shape==(len(dates),63)
            market.to_parquet(dest/'market63.parquet')
            np.save(dest/'market63.npy',market.to_numpy('float32'))
            (dest/'AUDIT.json').write_text(json.dumps({'status':'REAL_MARKET63_COMPLETE','rows':len(market),'columns':list(market.columns),'provider_price_dtype':'float32 as original Qlib','source_indices':['CSI300','CSI500','CSI800'],'substitute_basket':False,'original_operators':True,'no_fitted_statistics':True},indent=2))
    path=root/fp
    for name in ['AUDIT.json','contract.json','COMPLETE.json']:
        out=R/'market63';out.mkdir(exist_ok=True);__import__('shutil').copy2(path/name,out/name)
    print(path,flush=True);return path
if __name__=='__main__':run()
