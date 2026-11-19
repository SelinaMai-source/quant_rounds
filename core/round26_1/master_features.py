"""Unmodified Qlib Alpha158 expressions evaluated on the local provider."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from core.research_integrity_v5.cache import identity, fingerprint, transaction
from core.round26_1.master_provider import B,R,C
_PROVIDER=None

def init(provider):
    global _PROVIDER
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=str(provider),region=REG_CN,kernels=1,joblib_backend='threading',expression_cache=None,dataset_cache=None)
    _PROVIDER=Path(provider)

def one(task):
    instrument,provider_identity=task
    from qlib.data import D
    from qlib.contrib.data.loader import Alpha158DL
    fields,names=Alpha158DL.get_feature_config()
    c=dict(code=[identity(__file__)],third_party={'qlib_loader':identity(__import__('inspect').getfile(Alpha158DL))},dirty_diff='content identity',environment={'numpy':np.__version__,'pandas':pd.__version__},parameters={'instrument':instrument,'dtype':'float32'},seed=None,features=dict(zip(names,fields)),universe='historical quoted dates, no labels',calibration='not yet normalized',data=[provider_identity],label='not read',maturity='not read',windows='2010-01-01..2026-07-24, full history before graphs',execution='feature evaluation only',upstream=[])
    fp=fingerprint(c);root=B/'MASTER_Alpha158'/instrument
    with transaction(root,c) as (dest,hit):
        if not hit:
            frame=D.features([instrument],fields,start_time='2010-01-01',end_time='2026-07-24',freq='day')
            frame.columns=names
            frame=frame.astype('float32')
            assert len(frame)>0 and frame.shape[1]==158
            frame.to_parquet(dest/'features.parquet')
            (dest/'SCHEMA.json').write_text(json.dumps({'rows':len(frame),'columns':names,'full_calendar_slots_in_stock_quote_span':True}))
    return {'instrument_id':instrument,'path':str(root/fp),'cache_hit':hit,'fingerprint':fp}

def run(provider,jobs=2):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    provider=Path(provider);proof=identity(provider/'COMPLETE.json')
    instruments=[line.split('\t')[0] for line in (provider/'instruments/all.txt').read_text().splitlines()]
    records=[]
    with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn'),initializer=init,initargs=(provider,)) as pool:
        for record in pool.map(one,((i,proof) for i in instruments),chunksize=1):
            records.append(record)
            if len(records)%25==0:
                tmp=R/'ALPHA158_PROGRESS.partial.csv';pd.DataFrame(records).to_csv(tmp,index=False);os.replace(tmp,R/'ALPHA158_PROGRESS.csv')
                print('Original Alpha158 complete stocks',len(records),'/',len(instruments),flush=True)
    pd.DataFrame(records).to_csv(R/'ALPHA158_FEATURE_MANIFEST.csv',index=False)
    return records
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--provider',required=True);parser.add_argument('--jobs',type=int,default=2);args=parser.parse_args()
    assert 1<=args.jobs<=2
    run(args.provider,args.jobs)
