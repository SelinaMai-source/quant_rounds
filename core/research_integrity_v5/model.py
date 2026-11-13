"""Shared model entry: eligibility independent of target; actual per-head maturity."""
import numpy as np,pandas as pd
from core.round7.contract import model_parameters
from core.stock_readonly import ensure_stock_src_on_path
ensure_stock_src_on_path()
from first_board_hermes.baseline import daily_relevance
from xgboost import XGBRanker
KEY=['trade_date','instrument_id']
def maturity(frame,head):
 # Five is explicitly fixed D+2 close; Ten is first executable close from D+2.
 # 15:00 exchange event time, not arbitrary next2+22h. Source formula audited separately.
 return pd.to_datetime(frame[head.label_end_column]).dt.normalize()+pd.Timedelta(hours=15)
def training_rows(frame,head,year):
 cutoff=pd.Timestamp(f'{year}-01-01');known=maturity(frame,head)
 a=frame.loc[frame.trade_date.ge(f'{year-3}-01-01')&frame.trade_date.lt(cutoff)&known.lt(cutoff)].dropna(subset=[*head.features,head.target]).copy()
 a=a.loc[a.groupby('trade_date').instrument_id.transform('size').ge(2)].sort_values(KEY)
 return a
def predict_rows(frame,head,year):
 # Only immutable historical qualification and contemporaneous feature availability.
 return frame.loc[frame.trade_date.dt.year.eq(year)].dropna(subset=list(head.features)).sort_values(KEY).copy()
def fit_predict(train,candidates,head,year,seed=17):
 a=training_rows(train,head,year);p=predict_rows(candidates,head,year)
 params=model_parameters(seed);params['eval_metric']=head.eval_metric
 model=XGBRanker(**params);y=daily_relevance(a,target=head.target,n_bins=head.rank_n_bins).to_numpy('int32')
 model.fit(a[list(head.features)].to_numpy('float32'),y,qid=a.trade_date.factorize(sort=True)[0],verbose=False)
 out=p[KEY].copy();out['score']=model.predict(p[list(head.features)].to_numpy('float32'))
 return model,a,out,y
