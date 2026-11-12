from pathlib import Path
import json,hashlib,pandas as pd
B=Path(__file__).resolve().parent;C=B.parents[1];R=C/'result/round23_6'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()
parts=[];means=[]
for head in ['five','ten']:
 D=R/'expert_bank'/head;bank=pd.read_parquet(D/'calibrated_oof.parquet')
 calibrators=json.loads((D/'calibrator_registry.json').read_text())
 for year,block in bank.groupby(bank.trade_date.dt.year):
  models=json.loads((D/str(year)/'fit_manifest.json').read_text())
  raw=pd.read_parquet(D/str(year)/'predictions.parquet',columns=['mean'])
  assert raw['mean'].nunique()==1
  mean_state={'head':head,'year':int(year),'kind':'historical_training_mean','parameter':float(raw['mean'].iloc[0]),'fit_cutoff':models[0]['fit_cutoff'],'max_training_label_available_at':models[0]['max_training_label_available_at']}
  mean_state['model_hash']=digest(mean_state);means.append(mean_state)
  member_hashes=[m['model_hash'] for m in models]
  cal=[x for x in calibrators if x['year']==year]
  signature=digest({'base_member_hashes':member_hashes,'calibrator_hashes':[x['sha256'] for x in cal],'mean_model_hash':mean_state['model_hash']})
  q=block[['trade_date','instrument_id','prediction_at','fit_cutoff','xgb','lgbm','ridge','mean']].copy().rename(columns={'trade_date':'signal_date'})
  q['head']=head;q['model_hash']=signature;q['source_fold_registry']=str(D/str(year)/'fit_manifest.json')
  q['max_training_label_available_at']=max([pd.Timestamp(m['max_training_label_available_at']) for m in models]+[pd.Timestamp(x['max_training_label_available_at']) for x in cal])
  assert (q.max_training_label_available_at<pd.to_datetime(q.fit_cutoff)).all()
  q['train_start']=min(m['train_start'] for m in models);q['train_end']=max(m['train_end'] for m in models)
  parts.append(q)
# Also retain mean states for the earliest raw bank year preceding calibrated OOF.
for head in ['five','ten']:
 for p in (R/'expert_bank'/head).glob('*/fit_manifest.json'):
  year=int(p.parent.name)
  if any(m['head']==head and m['year']==year for m in means):continue
  model=json.loads(p.read_text())[0];x=pd.read_parquet(p.parent/'predictions.parquet',columns=['mean'])
  m={'head':head,'year':year,'kind':'historical_training_mean','parameter':float(x['mean'].iloc[0]),'fit_cutoff':model['fit_cutoff'],'max_training_label_available_at':model['max_training_label_available_at']};m['model_hash']=digest(m);means.append(m)
pd.concat(parts,ignore_index=True).to_parquet(R/'oof_registry.parquet',index=False)
(R/'historical_mean_model_states.json').write_text(json.dumps(means,indent=2))
print('OOF rows',sum(map(len,parts)),'historical mean states',len(means))
