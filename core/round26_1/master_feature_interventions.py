"""Actual original Alpha158 recomputation after future corporate-action/deletion inputs."""
from pathlib import Path
import json,sys,subprocess,shutil,os
import numpy as np
import pandas as pd
from core.round26_1.master_provider import B,R
from core.round26_1.master_store import GraphStore
from core.research_integrity_v5.cache import identity

def evaluate(provider,instrument,out):
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    from qlib.contrib.data.loader import Alpha158DL
    qlib.init(provider_uri=str(provider),region=REG_CN,kernels=1,joblib_backend='threading',expression_cache=None,dataset_cache=None)
    fields,names=Alpha158DL.get_feature_config()
    data=D.features([instrument],fields,start_time='2010-01-01',end_time='2026-07-24',freq='day')
    data.columns=names;data.astype('float32').to_parquet(out)

def run(store_path):
    store=GraphStore(store_path);ids=json.loads((store.graph_path/'instruments.json').read_text());records=[]
    for year in [2015,2021,2026]:
        index=int(store.indices(f'{year}-01-01',f'{year+1}-01-01')[0]);decision=store.dates[index];day=int(store.dayindices[index])
        lo,hi=store.offsets[index:index+2];stock=int(store.nodes[lo]);instrument=ids[stock]
        for mutation in ['original','future_CA_factor3','future_delete']:
            root=B/'MASTER_feature_interventions'/str(year)/mutation;provider=root/'provider';(provider/'calendars').mkdir(parents=True,exist_ok=True);(provider/'instruments').mkdir(exist_ok=True)
            shutil.copy2(store.provider/'calendars/day.txt',provider/'calendars/day.txt')
            instrument_lines=(store.provider/'instruments/all.txt').read_text().splitlines()
            selected=[line for line in instrument_lines if line.split('\t')[0]==instrument];assert len(selected)==1
            (provider/'instruments/all.txt').write_text(selected[0]+'\n')
            folder=provider/'features'/instrument.lower();folder.mkdir(parents=True,exist_ok=True)
            changed_rows=None
            for name in ['open','high','low','close','volume','vwap']:
                source=store.provider/'features'/instrument.lower()/(name+'.day.bin')
                vector=np.fromfile(source,dtype='<f4');future=np.arange(len(vector)-1)+int(vector[0])>day
                assert future.any();changed_rows=int(future.sum())
                values=vector[1:]
                if mutation=='future_CA_factor3':values[future]=values[future]*(1/3 if name=='volume' else 3)
                elif mutation=='future_delete':values[future]=np.nan
                vector.tofile(folder/(name+'.day.bin'))
            output=root/'actual_Alpha158.parquet'
            with (root/'EVALUATION.log').open('w') as log:
                subprocess.run([sys.executable,'-B','-m','core.round26_1.master_feature_interventions','--worker',str(provider),instrument,str(output)],check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            calculated=pd.read_parquet(output).droplevel('instrument')
            dates=pd.DatetimeIndex(calculated.index);before=dates<=decision
            expected=store.features[stock,store.calendar.get_indexer(dates[before]),:]
            actual=calculated.loc[before].to_numpy('float32')
            assert np.array_equal(expected,actual,equal_nan=True),'Historical original158 features changed under '+mutation
            records.append({'year':year,'decision':str(decision.date()),'instrument':instrument,'mutation':mutation,'actual_future_stock_rows_intervened':0 if mutation=='original' else changed_rows,'historical_rows_compared':int(before.sum()),'features':158,'historical_values_bitexact_including_missingness':True,'source':identity(output),'provider_path':str(provider)})
    result={'status':'PASS_ORIGINAL_ALPHA158_REAL_SOURCE_RECOMPUTATION','cases':records,'scope':'Original Qlib full-series expressions, actual future price/volume CA-event and missing-input intervention. Context re-selection and actual cold NN fits are separate tests.'}
    (R/'FEATURE_FUTURE_INTERVENTIONS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('Actual original Alpha158 future interventions PASS',len(records),flush=True)
    return result
if __name__=='__main__':
    if sys.argv[1]=='--worker':evaluate(*sys.argv[2:])
    else:run(sys.argv[1])
