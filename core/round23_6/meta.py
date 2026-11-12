"""Past-only expert bank, cross-fitted score calibration, NNLS and delayed BOA.

The canonical labels and financial executor are accessed read-only. Output stays
inside this round. The R numerical reference is a hard dependency.
"""
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRanker
from lightgbm import LGBMRanker
import joblib

C=Path('/data1/stock-mxy/课题/一进二策略')
B=C/'research/round23_6_26_execution'
R=C/'result/round23_6'
KEY=['trade_date','instrument_id']
spec=importlib.util.spec_from_file_location('aggregation',B/'round23_6/core/research_integrity_v4/aggregation.py');agg=importlib.util.module_from_spec(spec);spec.loader.exec_module(agg)


def run_meta(resume=True):
 sys.path.insert(0,str(C))
 from core.round7.contract import FIVE_HEAD,TEN_HEAD,SEEDS,model_parameters
 from core.stock_readonly import ensure_stock_src_on_path
 ensure_stock_src_on_path()
 from first_board_hermes.baseline import daily_relevance
 parity=json.loads((R/'r_reference/numerical_parity.json').read_text())
 if parity['status']!='PASS':raise RuntimeError('Official R parity gate failed')
 registry=[];outputs={};weights_rows=[];feedback_rows=[];forecast_rows=[]
 for name,head in [('five',FIVE_HEAD),('ten',TEN_HEAD)]:
  target=head.target;features=list(head.features);maturity=head.label_end_column
  cols=list(dict.fromkeys(KEY+features+[target,maturity]))
  train=pd.read_parquet(C/f'result/round22_3/data/{name}_training.parquet',columns=cols)
  pred=pd.read_parquet(C/f'result/round22_3/data/{name}_candidate.parquet',columns=cols)
  for x in [train,pred]:
   x['trade_date']=pd.to_datetime(x['trade_date']);x['instrument_id']=x['instrument_id'].astype(str)
   x['label_available_at']=pd.to_datetime(x[maturity])+pd.Timedelta(hours=22)
  D=R/'expert_bank'/name;D.mkdir(parents=True,exist_ok=True)
  banks=[]
  for year in range(2014,2027):
   dest=D/str(year);dest.mkdir(exist_ok=True)
   cache=dest/'predictions.parquet'
   if resume and cache.exists() and (dest/'fit_manifest.json').exists():
    banks.append(pd.read_parquet(cache));registry+=json.loads((dest/'fit_manifest.json').read_text());continue
   cut=pd.Timestamp(f'{year}-01-01')
   a=train.loc[train.trade_date.ge(pd.Timestamp(f'{year-3}-01-01')) & train.trade_date.lt(cut) & train.label_available_at.lt(cut)].dropna(subset=features+[target]).copy()
   a=a.loc[a.groupby('trade_date').instrument_id.transform('size').ge(2)].sort_values(KEY)
   p=pred.loc[pred.trade_date.dt.year.eq(year) & pred.trade_date.le(pd.Timestamp('2026-07-24'))].copy()
   if len(a)<200 or p.empty:continue
   print(name,year,'fit',len(a),'predict',len(p),flush=True)
   label=daily_relevance(a,target=target,n_bins=head.rank_n_bins).astype('int32')
   ready=p[features].notna().all(axis=1)
   out=p[KEY+[target,'label_available_at']].copy();out['xgb']=np.nan;out['lgbm']=np.nan
   models=[];ranks=[]
   for seed in SEEDS:
    params=model_parameters(seed);params['eval_metric']=head.eval_metric
    m=XGBRanker(**params);m.fit(a[features].to_numpy(np.float32),label.to_numpy(),qid=a.trade_date.factorize(sort=True)[0],verbose=False)
    m.save_model(dest/f'xgb_{seed}.json');models.append(dest/f'xgb_{seed}.json')
    scores=pd.Series(m.predict(p.loc[ready,features].to_numpy(np.float32)),index=p.index[ready])
    ranks.append(scores.groupby(p.loc[ready,'trade_date']).rank(pct=True))
   out.loc[ready,'xgb']=pd.concat(ranks,axis=1).mean(axis=1)
   l=LGBMRanker(objective='lambdarank',learning_rate=.03,n_estimators=200,num_leaves=7,max_depth=3,min_child_samples=50,feature_fraction=1.,bagging_fraction=1.,random_state=17,n_jobs=2,verbosity=-1,lambdarank_truncation_level=4 if name=='five' else 5)
   l.fit(a[features],label,group=a.groupby('trade_date',sort=False).size().to_numpy())
   l.booster_.save_model(str(dest/'lgbm.txt'));models.append(dest/'lgbm.txt')
   out.loc[ready,'lgbm']=l.predict(p.loc[ready,features])
   # Imputation/normalization fitted exclusively on mature training rows.
   from sklearn.preprocessing import StandardScaler
   ridge=make_pipeline(StandardScaler(),Ridge(alpha=10))
   ridge.fit(a[features],a[target]);joblib.dump(ridge,dest/'ridge.joblib');models.append(dest/'ridge.joblib')
   out['ridge']=np.nan;out.loc[ready,'ridge']=ridge.predict(p.loc[ready,features]);out['mean']=float(a[target].mean())
   out['fit_cutoff']=cut;out['max_training_label_available_at']=a.label_available_at.max()
   records=[{'head':name,'year':year,'model_path':str(q),'model_hash':hashlib.sha256(q.read_bytes()).hexdigest(),'fit_cutoff':str(cut),'train_start':str(a.trade_date.min()),'train_end':str(a.trade_date.max()),'max_training_label_available_at':str(a.label_available_at.max()),'train_rows':len(a),'target':target,'maturity_source':maturity} for q in models]
   (dest/'fit_manifest.json').write_text(json.dumps(records,indent=2));registry+=records
   out.to_parquet(cache,index=False);banks.append(out)
  bank=pd.concat(banks,ignore_index=True);bank['trade_date']=pd.to_datetime(bank.trade_date)
  # Each score calibrator predicts a future annual block from historical OOF.
  calibrated=[];calreg=[]
  for year in range(2015,2027):
   cut=pd.Timestamp(f'{year}-01-01')
   hist=bank.loc[bank.trade_date.ge(pd.Timestamp(f'{year-3}-01-01')) & bank.trade_date.lt(cut) & bank.label_available_at.lt(cut)].dropna(subset=['xgb','lgbm',target])
   p=bank.loc[bank.trade_date.dt.year.eq(year)].copy()
   if len(hist)<200 or p.empty:continue
   for column in ['xgb','lgbm']:
    m=Ridge(alpha=10);m.fit(hist[[column]],hist[target])
    ok=p[column].notna();p.loc[ok,column]=m.predict(p.loc[ok,[column]])
    path=D/f'calibrator_{column}_{year}.joblib';joblib.dump(m,path)
    calreg.append({'head':name,'year':year,'expert':column,'fit_cutoff':str(cut),'max_training_label_available_at':str(hist.label_available_at.max()),'train_start':str(hist.trade_date.min()),'train_end':str(hist.trade_date.max()),'rows':len(hist),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
   calibrated.append(p)
  bank=pd.concat(calibrated,ignore_index=True).sort_values(KEY)
  decision_map=pd.read_parquet(C/f'result/round22_3/data/{name}_candidate.parquet',columns=KEY+['next1_date'])
  decision_map['trade_date']=pd.to_datetime(decision_map.trade_date);decision_map['instrument_id']=decision_map.instrument_id.astype(str)
  decision_map['prediction_at']=pd.to_datetime(decision_map.next1_date)+pd.Timedelta(hours=9,minutes=30)
  bank=bank.merge(decision_map[KEY+['prediction_at']],on=KEY,how='left',validate='one_to_one')
  if bank.prediction_at.isna().any():raise RuntimeError('Legacy next-open decision time unavailable')
  bank.to_parquet(D/'calibrated_oof.parquet',index=False)
  (D/'calibrator_registry.json').write_text(json.dumps(calreg,indent=2))
  outputs[name]={k:[] for k in ['SL','BOA','EQ','XGB','LGBM','RIDGE','MEAN']}
  for year in range(2018,2027):
   cut=pd.Timestamp(f'{year}-01-01')
   hist=bank.loc[bank.trade_date.ge(pd.Timestamp(f'{year-3}-01-01')) & bank.trade_date.lt(cut) & bank.label_available_at.lt(cut)].dropna(subset=['xgb','lgbm','ridge','mean',target])
   p=bank.loc[bank.trade_date.dt.year.eq(year)].copy().sort_values(KEY)
   if p.empty or len(hist)<200:continue
   columns=['xgb','lgbm','ridge','mean'];z=hist[columns].to_numpy();y=hist[target].to_numpy()
   sl=agg.superlearner_nnls(z,y)
   for i,col in enumerate(columns):weights_rows.append({'method':'SL','head':name,'outer_year':year,'expert':col,'raw':sl['raw'][i],'weight':sl['weights'][i],'meta_rows':len(hist),'max_training_label_available_at':str(hist.label_available_at.max())})
   low,high=np.quantile(y,[.01,.99])
   if high<=low:raise RuntimeError('degenerate BOA label bounds')
   transform=lambda t: np.clip((np.asarray(t)-low)/(high-low),0,1)
   boa=agg.BernsteinOnlineAggregation(4)
   # Annual reset, replay the historic stored forecasts in maturity order.
   hz=transform(z);hy=transform(y)
   # First replay predicts then consumes only date-group matured feedback.
   pending=[];stream=pd.concat([hist.assign(outer=False),p.assign(outer=True)],ignore_index=True).sort_values(KEY)
   preds={};pend=[]
   for day,block in stream.groupby('trade_date',sort=True):
    decision=pd.Timestamp(block.prediction_at.iloc[0])
    if not block.prediction_at.eq(decision).all():raise RuntimeError('Inconsistent same-day legacy decision timestamps')
    due=[r for r in pend if r['mature_at']<decision];pend=[r for r in pend if r['mature_at']>=decision]
    if due:
     # Each maturity date contributes one equal-date update, independent of stock ordering.
     for md in sorted(set(r['mature_at'].normalize() for r in due)):
      rows=[r for r in due if r['mature_at'].normalize()==md]
      boa.update_regret(np.mean([r['regret'] for r in rows],axis=0))
      feedback_rows.append({'head':name,'outer_year':year,'decision_at':str(decision),'feedback_date':str(md),'rows':len(rows),'max_mature_at':str(max(r['mature_at'] for r in rows))})
    w=boa.weights().copy()
    for idx,row in block.iterrows():
     raw=row[columns].to_numpy(dtype=float)
     if not np.isfinite(raw).all():continue
     zs=transform(raw);ps=float(zs@w)
     if row['outer']:
      preds[(row.trade_date,row.instrument_id)]=ps
      forecast_rows.append({'head':name,'outer_year':year,'signal_date':row.trade_date,'instrument_id':row.instrument_id,'prediction_at':decision,'label_available_at':row.label_available_at,'prediction':ps,**{('expert_'+col):float(zs[i]) for i,col in enumerate(columns)},**{('weight_'+col):float(w[i]) for i,col in enumerate(columns)}})
     if pd.notna(row.label_available_at) and np.isfinite(row[target]):
      target_value=float(transform(row[target]));regret=2*(ps-target_value)*(ps-zs)
      pend.append({'mature_at':pd.Timestamp(row.label_available_at),'regret':regret})
    if block['outer'].any():
     weights_rows.extend({'method':'BOA','head':name,'outer_year':year,'date':str(day),'expert':col,'weight':float(w[i]),'q01':low,'q99':high} for i,col in enumerate(columns))
   for method in outputs[name]:
    out=p[KEY].copy()
    if method=='SL':out['score']=p[columns].to_numpy()@sl['weights']
    elif method=='BOA':out['score']=[preds.get((row.trade_date,row.instrument_id),np.nan) for row in p.itertuples()]
    elif method=='EQ':out['score']=p[columns].mean(axis=1,skipna=False)
    else:out['score']=p[{'XGB':'xgb','LGBM':'lgbm','RIDGE':'ridge','MEAN':'mean'}[method]]
    outputs[name][method].append(out)
 pd.DataFrame(registry).to_csv(R/'model_registry.csv',index=False)
 pd.DataFrame(weights_rows).to_csv(R/'meta_weights_by_time_SL_BOA.csv',index=False)
 pd.DataFrame(feedback_rows).to_csv(R/'online_feedback_ledger.csv',index=False)
 pd.DataFrame(forecast_rows).to_parquet(R/'online_forecast_ledger.parquet',index=False)
 base=pd.read_parquet(R/'frozen_components.parquet');base=base[['signal_date','instrument_id','v5']].rename(columns={'signal_date':'trade_date'})
 scoresdir=R/'scores';scoresdir.mkdir(exist_ok=True)
 for method in outputs['five']:
  x=base.copy()
  for head in ['five','ten']:
   pred=pd.concat(outputs[head][method],ignore_index=True).rename(columns={'score':head})
   x=x.merge(pred,on=KEY,how='left',validate='one_to_one')
  bad=(x[['five','ten']].isna().any(axis=1)).groupby(x.trade_date).transform('any')
  ranks=x.groupby('trade_date')[['five','ten']].rank(pct=True)
  x['score']=.1*ranks.five+.9*ranks.ten;x.loc[bad,'score']=x.loc[bad,'v5'];x['fallback_entire_day']=bad
  x=x.rename(columns={'trade_date':'signal_date'})
  x[['signal_date','instrument_id','score','fallback_entire_day']].to_parquet(scoresdir/(method+'.parquet'),index=False)
 print('META COMPLETE',flush=True)

if __name__=='__main__':run_meta()
