from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np,pandas as pd
from xgboost import XGBRanker
C=Path('/data1/stock-mxy/课题/一进二策略');B=C/'research/round23_6_26_execution';R=C/'result/round23_6/diagnostics/future_mutation';R.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(C));sys.dont_write_bytecode=True
from core.round7.contract import FIVE_HEAD,model_parameters
from core.stock_readonly import ensure_stock_src_on_path
from core.r225241_sleeve import apply_b03_o01
ensure_stock_src_on_path()
from first_board_hermes.baseline import daily_relevance
h=FIVE_HEAD;features=list(h.features);target=h.target
f=pd.read_parquet(C/'result/round22_3/data/five_training.parquet',columns=['trade_date','instrument_id',target,h.label_end_column,*features])
p=pd.read_parquet(C/'result/round22_3/data/five_candidate.parquet',columns=['trade_date','instrument_id',*features])
for x in [f,p]:x['trade_date']=pd.to_datetime(x.trade_date)
f['label_available_at']=pd.to_datetime(f[h.label_end_column])+pd.Timedelta(hours=22)
cut=pd.Timestamp('2018-01-01');g=f.copy();future=g.trade_date.ge(cut)|g.label_available_at.ge(cut);g.loc[future,target]=np.arange(future.sum())*1e6-1e9
outputs=[];matrix_hashes=[]
for i,frame in enumerate([f,g]):
 train=frame.loc[frame.trade_date.ge('2015-01-01')&frame.trade_date.lt(cut)&frame.label_available_at.lt(cut)].dropna(subset=features+[target]).copy()
 train=train.loc[train.groupby('trade_date').instrument_id.transform('size').ge(2)].sort_values(['trade_date','instrument_id'])
 test=p.loc[p.trade_date.dt.year.eq(2018)].dropna(subset=features).copy()
 y=daily_relevance(train,target=target,n_bins=h.rank_n_bins).to_numpy('int32')
 matrix_hashes.append(hashlib.sha256(train[features].to_numpy('float32').tobytes()+y.tobytes()).hexdigest())
 params=model_parameters(17);params['eval_metric']=h.eval_metric
 m=XGBRanker(**params);m.fit(train[features].to_numpy('float32'),y,qid=train.trade_date.factorize(sort=True)[0],verbose=False)
 m.save_model(R/f'fit_{i}.model.json')
 out=test[['trade_date','instrument_id']].rename(columns={'trade_date':'signal_date'});out['score']=m.predict(test[features].to_numpy('float32'));outputs.append(out)
 outputs[-1].to_parquet(R/f'predictions_{i}.parquet',index=False)
pd.testing.assert_frame_equal(outputs[0],outputs[1],check_exact=True)
a,_=apply_b03_o01(outputs[0]);b,_=apply_b03_o01(outputs[1]);pd.testing.assert_frame_equal(a,b,check_exact=True)
(R/'integration_result.json').write_text(json.dumps({'status':'PASS_ONE_REAL_FOLD_SCOPE_ONLY','head':'five','outer_year':2018,'seed':17,'mutated_future_or_immature_target_rows':int(future.sum()),'matrix_hashes':matrix_hashes,'prediction_rows_compared':len(outputs[0]),'overlayed_rows_compared':len(a),'new_diagnostic_model_fits':2,'scope_limit':'This does not cover the complete NNLS/BOA/neural pipeline or all years, revisions and prefixes.'},indent=2));print('Real future mutation: one Five/2018/seed17 fold passed; full integration still incomplete')
