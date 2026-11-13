"""Nested OOF factor -> original Five/Ten XGB Add/Remove/Replace, no IC veto."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from xgboost import XGBRanker
from core.research_integrity_v5.cache import transaction,fingerprint,identity,sha
from core.research_integrity_v5.experiment import full_contract
from core.research_integrity_v5.resources import measured
from core.round26_1.market import C,R,KEY

def build_factor_banks(registry,seeds=(17,42,73),include_composite=True):
 banks={};records=[]
 for arm in ['AG_RAW','AG_RES']:
  individual=[]
  for seed in seeds:
   folds=[]
   for year in range(2020,2027):
    row=registry.loc[registry.arm.eq(arm)&registry.seed.eq(seed)&registry.year.eq(year)]
    if len(row)!=1 or row.iloc[0].status!='FACTOR_OOF_YEAR_COMPLETE':raise RuntimeError(f'Incomplete required factor fold {arm}/{seed}/{year}')
    d=Path(row.iloc[0].directory);p=pd.read_parquet(d/'factor_predictions.parquet');state=json.loads((d/'FROZEN_FACTOR.json').read_text());p['prior_training_scale_mean']=state['prior_training_scale_mean'];p['prior_training_scale_std']=state['prior_training_scale_std'];folds.append(p);records.append({'arm':arm,'seed':seed,'year':year,'factor_file':identity(d/'factor_predictions.parquet'),'frozen_model':identity(d/'FROZEN_FACTOR.json'),'generation_steps':state['generation_env_steps'],'valid_evaluations':state['valid_evaluations'],'fit_cutoff':state['fit_cutoff'],'max_training_label_available_at':state['max_training_label_available_at']})
   bank=pd.concat(folds,ignore_index=True).sort_values(KEY);assert not bank.duplicated(KEY).any();dest=R/'factor_banks'/arm/str(seed);dest.mkdir(parents=True,exist_ok=True);bank.to_parquet(dest/'OOF_factor.parquet',index=False);banks[(arm,str(seed))]=dest/'OOF_factor.parquet';individual.append(bank)
  if not include_composite:continue
  if list(seeds)!=[17,42,73]:raise ValueError("Equal3 requires all three frozen seeds")
  merged=None
  for seed,bank in zip(seeds,individual):
   v=bank[KEY+['z_AG','prior_training_scale_mean','prior_training_scale_std']].copy();v[str(seed)]=(v.z_AG-v.prior_training_scale_mean)/v.prior_training_scale_std
   merged=v[KEY+[str(seed)]] if merged is None else merged.merge(v[KEY+[str(seed)]],on=KEY,how='outer',validate='one_to_one')
  assert merged[KEY].notna().all().all();merged['z_AG']=merged[['17','42','73']].mean(axis=1,skipna=False);merged['factor_fit_cutoff']=pd.to_datetime(merged.trade_date.dt.year.astype(str)+'-01-01');merged['factor_model_hash']=merged.trade_date.dt.year.map(lambda y:hashlib.sha256('|'.join(records[-21+i]['frozen_model']['sha256'] for i in range(21) if records[-21+i]['year']==y).encode()).hexdigest());dest=R/'factor_banks'/arm/'EQ3';dest.mkdir(parents=True,exist_ok=True);merged.to_parquet(dest/'OOF_factor.parquet',index=False);banks[(arm,'EQ3')]=dest/'OOF_factor.parquet'
 old=json.loads((R/'factor_model_registry.json').read_text()) if (R/'factor_model_registry.json').exists() else [];merged={(x['arm'],str(x['seed']),x['year']):x for x in old+records};(R/'factor_model_registry.json').write_text(json.dumps(list(merged.values()),ensure_ascii=False,indent=2));return banks

def canonical_fuse(five,ten):
 from core.round17.backtest import load_gaps
 keys=['signal_date','instrument_id'];a=five[keys+['score']].rename(columns={'score':'five'});b=ten[keys+['score']].rename(columns={'score':'ten'});x=a.merge(b,on=keys,how='inner',validate='one_to_one').merge(load_gaps(),on=keys,how='left',validate='one_to_one');x=x.loc[np.isfinite(x.d1_open_gap)&x.d1_open_gap.ge(-.01)].copy();x['pct_five']=x.groupby('signal_date').five.rank(method='average',pct=True);x['pct_ten']=x.groupby('signal_date').ten.rank(method='average',pct=True);x['score']=.90*x.pct_ten+.10*x.pct_five
 return x[keys+['score']].sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort').reset_index(drop=True)

def prepare(head,year,factor_path):
 features=list(head.features);trainpath=C/f'result/round22_3/data/{head.name}_training.parquet';predpath=C/f'result/round22_3/data/{head.name}_candidate.parquet';cols=list(dict.fromkeys(KEY+features+[head.target,head.label_end_column]));a=pd.read_parquet(trainpath,columns=cols);p=pd.read_parquet(predpath,columns=list(dict.fromkeys(KEY+features)));factor=pd.read_parquet(factor_path);extra=factor[KEY+['z_AG','factor_fit_cutoff']]
 for frame in [a,p]:frame['trade_date']=pd.to_datetime(frame.trade_date);frame['instrument_id']=frame.instrument_id.astype(str);frame[features]=frame[features].replace([np.inf,-np.inf],np.nan)
 cutoff=pd.Timestamp(f'{year}-01-01');available=pd.to_datetime(a[head.label_end_column])+pd.Timedelta(hours=15);a=a.loc[a.trade_date.ge(f'{year-3}-01-01')&a.trade_date.lt(cutoff)&available.lt(cutoff)].dropna(subset=features+[head.target]).sort_values(KEY);a=a.loc[a.groupby('trade_date').instrument_id.transform('size').ge(2)];p=p.loc[p.trade_date.dt.year.eq(year)].dropna(subset=features).sort_values(KEY);a=a.merge(extra,on=KEY,how='left',validate='one_to_one');p=p.merge(extra,on=KEY,how='left',validate='one_to_one');assert a.factor_fit_cutoff.notna().all() and p.factor_fit_cutoff.notna().all();assert (a.factor_fit_cutoff<=pd.to_datetime(a.trade_date.dt.year.astype(str)+'-01-01')).all();assert (p.factor_fit_cutoff<=pd.to_datetime(p.trade_date.dt.year.astype(str)+'-01-01')).all();a['label_available_at']=pd.to_datetime(a[head.label_end_column])+pd.Timedelta(hours=15);return a,p

def fit_head(arm,seed_label,head,year,factor_path,mode,remove=None,*,cache_namespace=None):
 from core.round7.contract import SEEDS,model_parameters
 from core.stock_readonly import ensure_stock_src_on_path
 ensure_stock_src_on_path()
 from first_board_hermes.baseline import daily_relevance
 a,p=prepare(head,year,factor_path);features=[x for x in head.features if mode not in ['REMOVE','REPLACE'] or x!=remove]
 if mode in ['ADD','REPLACE']:features.append('z_AG')
 params={'arm':arm,'factor_seed':seed_label,'head':head.name,'year':year,'mode':mode,'remove':remove,'feature_order':features,'tree_seeds':list(SEEDS),'XGB_params':model_parameters(17),'rank_bins':head.rank_n_bins,'eval_metric':head.eval_metric,'fit_cutoff':f'{year}-01-01','max_training_label_available_at':str(a.label_available_at.max()),'train_start':str(a.trade_date.min()),'train_end':str(a.trade_date.max()),'training_keys_hash':hashlib.sha256(pd.util.hash_pandas_object(a[KEY],index=False).values.tobytes()).hexdigest(),'training_values_and_labels_hash':hashlib.sha256(pd.util.hash_pandas_object(a[KEY+features+[head.target,'label_available_at']],index=False).values.tobytes()).hexdigest(),'prediction_values_hash':hashlib.sha256(pd.util.hash_pandas_object(p[KEY+features],index=False).values.tobytes()).hexdigest(),'prediction_keys_hash':hashlib.sha256(pd.util.hash_pandas_object(p[KEY],index=False).values.tobytes()).hexdigest(),'OOF_factor_assertion':'every training row factor_fit_cutoff <= row calendar-year start, not current-Outer formula fit values'}
 c=full_contract(params,[factor_path,C/f'result/round22_3/data/{head.name}_training.parquet',C/f'result/round22_3/data/{head.name}_candidate.parquet'],seed=17,features=features,label=head.target,maturity=head.label_end_column+' at exchange close15:00');fp=fingerprint(c);cache=(Path(cache_namespace) if cache_namespace is not None else R/'head_models'/arm/str(seed_label)/head.name/mode/str(year))
 with transaction(cache,c) as (dest,cached):
  if not cached:
   with measured(dest/'RESOURCES.json'):
    relevance=daily_relevance(a,target=head.target,n_bins=head.rank_n_bins).to_numpy('int32');out=p[KEY].copy();ranks=[];models=[]
    for tree_seed in SEEDS:
     cfg=model_parameters(tree_seed);cfg['eval_metric']=head.eval_metric;m=XGBRanker(**cfg);m.fit(a[features].to_numpy('float32'),relevance,qid=a.trade_date.factorize(sort=True)[0],verbose=False);path=dest/f'xgb_{tree_seed}.json';m.save_model(path);models.append({'seed':tree_seed,'sha256':sha(path),'features':features});score=pd.Series(m.predict(p[features].to_numpy('float32')),index=p.index);ranks.append(score.groupby(p.trade_date).rank(pct=True))
    out['score']=pd.concat(ranks,axis=1).mean(axis=1);out['fit_cutoff']=pd.Timestamp(f'{year}-01-01');out['max_training_label_available_at']=a.label_available_at.max();out.to_parquet(dest/'predictions.parquet',index=False);(dest/'MODEL_REGISTRY.json').write_text(json.dumps({'parameters':params,'models':models,'input_fingerprint':fp},ensure_ascii=False,indent=2))
 final=cache/fp;return pd.read_parquet(final/'predictions.parquet'),final,fp,cached

def run_models(banks):
 from core.round7.contract import FIVE_HEAD,TEN_HEAD
 from core.round23_3.contract import E0
 from core.round26_1.factor_models import canonical_fuse as fuse_heads
 from core.research_integrity_v5.window_backtest import execute
 source={h:pd.read_parquet(E0/(h+'_pred.parquet')) for h in ['five','ten']};registry=[];correlations=[];modelrecords=[]
 # A0 same explicit matched calendar/capital/execution, original full-history overlay warmup.
 base=fuse_heads(source['five'],source['ten'])
 frozen=pd.read_parquet(C/'result/round23_6/frozen_components.parquet')[['signal_date','instrument_id','v5']].rename(columns={'v5':'score'});keys=['signal_date','instrument_id'];pd.testing.assert_frame_equal(base.sort_values(keys).reset_index(drop=True),frozen.sort_values(keys).reset_index(drop=True),check_exact=True)
 for bps in [0,10,20]:
  code='AG_A0'+('' if bps==0 else '_S'+str(bps//10));path,fp,hit=execute('round26_1',code,base,bps=bps);registry.append({'method':'AG_A0','scenario':'S'+str(bps//10),'code':code,'path':str(path),'experiment_fingerprint':fp,'cache_hit':hit})
 for (arm,seed_label),factor in banks.items():
  predictions={};remove_predictions={};replace_predictions={};activated={}
  for head in [FIVE_HEAD,TEN_HEAD]:
   add=[];rem=[];rep=[];activated[head.name]=False
   for year in range(2023,2027):
    a,p=prepare(head,year,factor);rho=a[[*head.features,'z_AG']].corr(method='spearman').z_AG.drop('z_AG');finite=rho.dropna();column=finite.abs().idxmax() if len(finite) else None;value=float(rho[column]) if column else None;high=column is not None and abs(value)>=.6;correlations.append({'arm':arm,'seed':seed_label,'head':head.name,'outer_year':year,'most_correlated_old_feature':column,'rho':value,'threshold':.60,'high_correlation':high,'selection_scope':'prior3y original base-complete mature rows only','rows':len(a),'fit_cutoff':f'{year}-01-01'})
    frame,path,fp,cached=fit_head(arm,seed_label,head,year,factor,'ADD');add.append(frame);modelrecords.append({'arm':arm,'seed':seed_label,'head':head.name,'year':year,'mode':'ADD','path':str(path),'fingerprint':fp,'cache_hit':cached})
    fallback=source[head.name].loc[pd.to_datetime(source[head.name].signal_date).dt.year.eq(year)].rename(columns={'signal_date':'trade_date'})
    if high:
     activated[head.name]=True
     for mode,collect in [('REMOVE',rem),('REPLACE',rep)]:
      frame,path,fp,cached=fit_head(arm,seed_label,head,year,factor,mode,column);collect.append(frame);modelrecords.append({'arm':arm,'seed':seed_label,'head':head.name,'year':year,'mode':mode,'removed':column,'path':str(path),'fingerprint':fp,'cache_hit':cached})
    else:rem.append(fallback);rep.append(fallback)
   predictions[head.name]=pd.concat(add,ignore_index=True).rename(columns={'trade_date':'signal_date'});remove_predictions[head.name]=pd.concat(rem,ignore_index=True).rename(columns={'trade_date':'signal_date'});replace_predictions[head.name]=pd.concat(rep,ignore_index=True).rename(columns={'trade_date':'signal_date'})
  variants={'A1_FIVE_ADD':{'five':predictions['five']},'A2_TEN_ADD':{'ten':predictions['ten']},'A3_BOTH_ADD':predictions}
  for head in ['five','ten']:
   if activated[head]:variants['REMOVE_'+head.upper()]={head:remove_predictions[head]};variants['REPLACE_'+head.upper()]={head:replace_predictions[head]}
  for variant,override in variants.items():
   five=source['five'].copy();ten=source['ten'].copy()
   for h,target in [('five',five),('ten',ten)]:
    if h in override:
     v=override[h][['signal_date','instrument_id','score']];v['signal_date']=pd.to_datetime(v.signal_date);keep=~pd.to_datetime(target.signal_date).dt.year.between(2023,2026);target.drop(target.index[~keep],inplace=True)
     if h=='five':five=pd.concat([target,v],ignore_index=True)
     else:ten=pd.concat([target,v],ignore_index=True)
   scores=fuse_heads(five,ten);method=arm+'_'+str(seed_label)+'_'+variant;dest=R/'scores';dest.mkdir(exist_ok=True);scores.to_parquet(dest/(method+'.parquet'),index=False)
   for bps in ([0,10,20] if variant=='A3_BOTH_ADD' else [0]):
    code=method+('' if bps==0 else '_S'+str(bps//10));path,fp,hit=execute('round26_1',code,scores,bps=bps,upstream=[identity(factor)]);registry.append({'method':method,'scenario':'S'+str(bps//10),'code':code,'path':str(path),'experiment_fingerprint':fp,'cache_hit':hit});print('AlphaGen actual Add trial',code,flush=True)
 stage='EQ3' if any(s=='EQ3' for _,s in banks) else '_'.join(sorted({s for _,s in banks}));stage_dir=R/'seed_stages'/stage;stage_dir.mkdir(parents=True,exist_ok=True);pd.DataFrame(registry).to_csv(stage_dir/'run_manifest.csv',index=False);pd.DataFrame(modelrecords).to_csv(stage_dir/'head_model_manifest.csv',index=False);pd.DataFrame(correlations).to_csv(stage_dir/'dedup_selection.csv',index=False)
 all_runs=pd.concat([pd.read_csv(p) for p in (R/'seed_stages').glob('*/run_manifest.csv')],ignore_index=True).drop_duplicates('code',keep='last');all_runs.to_csv(R/'run_manifest.csv',index=False);pd.DataFrame(correlations).to_csv(R/'training_only_dedup_selection.csv',index=False);pd.concat([pd.read_csv(p) for p in (R/'seed_stages').glob('*/head_model_manifest.csv')],ignore_index=True).drop_duplicates(['arm','seed','head','year','mode'],keep='last').to_csv(R/'head_model_manifest.csv',index=False);pd.concat([pd.read_csv(p) for p in (R/'seed_stages').glob('*/dedup_selection.csv')],ignore_index=True).drop_duplicates(['arm','seed','head','outer_year'],keep='last').to_csv(R/'training_only_dedup_selection.csv',index=False)
 from core.round26_1.factor_entry_tests import run as test_actual_head_entry
 for (arm,seed_label),factor in banks.items():test_actual_head_entry(factor,arm,seed_label)
 return registry
