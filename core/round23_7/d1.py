"""D1 frozen OOF recovery, true annual fits, integrated counterfactuals and trading."""
from pathlib import Path
import json,sys,subprocess,shutil,time
import numpy as np,pandas as pd
from core.round23_7.anchor import scores,build_pairs,fit,objective
from core.research_integrity_v5.cache import identity,sha
from core.research_integrity_v5.experiment import backtest
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_7/D1';KEY=['signal_date','instrument_id']
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str))
def build_inputs():
 from core.round7.contract import FIVE_HEAD,TEN_HEAD
 from core.research_integrity_v5.model import training_rows,predict_rows
 from core.stock_readonly import ensure_stock_src_on_path
 ensure_stock_src_on_path()
 from first_board_hermes.baseline import daily_relevance
 from lightgbm import LGBMRanker
 current=pd.read_parquet(C/'result/round23_6/frozen_components.parquet');current['source']='FROZEN_2018_2026'
 current=current.rename(columns={'pct_five':'xF','pct_ten':'xT'})
 current['lF']=current.groupby('signal_date').lgbm_five.rank(pct=True);current['lT']=current.groupby('signal_date').lgbm_ten.rank(pct=True)
 registry=[];warm={}
 for h in [FIVE_HEAD,TEN_HEAD]:
  train=pd.read_parquet(C/f'result/round22_3/data/{h.name}_training.parquet');pred=pd.read_parquet(C/f'result/round22_3/data/{h.name}_candidate.parquet');parts=[]
  oldmeta=json.loads((C/f'result/round23_3/oof/lgbm_{h.name}_trunc{4 if h.name=="five" else 5}/train_meta.json').read_text());params=oldmeta['params']
  for year in [2016,2017]:
   a=training_rows(train,h,year).reset_index(drop=True);p=predict_rows(pred,h,year).reset_index(drop=True);d=R/'warmup_models'/h.name/str(year);d.mkdir(parents=True,exist_ok=True)
   m=LGBMRanker(**params);rel=daily_relevance(a,target=h.target,n_bins=h.rank_n_bins).to_numpy('int32');m.fit(a[list(h.features)].to_numpy('float32'),rel,group=a.groupby('trade_date',sort=False).size().to_numpy());m.booster_.save_model(str(d/'lgbm.txt'))
   out=p[['trade_date','instrument_id','d1_open_gap']].rename(columns={'trade_date':'signal_date'});out['lgbm']=m.predict(p[list(h.features)].to_numpy('float32'))
   old=C/f'result/round23_6/expert_bank/{h.name}/{year}';x=pd.read_parquet(old/'predictions.parquet').rename(columns={'trade_date':'signal_date'});out=out.merge(x[KEY+['xgb']],on=KEY,validate='one_to_one');out.to_parquet(d/'predictions.parquet',index=False);parts.append(out)
   manifests=json.loads((old/'fit_manifest.json').read_text());xm=[z for z in manifests if '/xgb_' in z['model_path']];assert all(z['max_training_label_available_at']<str(pd.Timestamp(f'{year}-01-01')) for z in xm)
   assert all(sha(Path(z['model_path']))==z['model_hash'] for z in xm)
   registry.append({'head':h.name,'year':year,'fit_cutoff':f'{year}-01-01','max_training_label_available_at':str(pd.to_datetime(a[h.label_end_column]).max()+pd.Timedelta(hours=15)),'features':list(h.features),'seed':17,'new_LGBM_model':identity(d/'lgbm.txt'),'XGB_sources':xm,'training_rows':len(a),'prediction_rows':len(out),'training_start':str(a.trade_date.min())})
   print('D1 warmup restored',h.name,year,flush=True)
  warm[h.name]=pd.concat(parts,ignore_index=True)
 # Same XGB raw-bank rank recipe agrees EXACTLY with actual v5 on every 2018+ key.
 for h,col in [('five','xF'),('ten','xT')]:
  bank=pd.concat([pd.read_parquet(C/f'result/round23_6/expert_bank/{h}/{y}/predictions.parquet') for y in range(2018,2027)]).rename(columns={'trade_date':'signal_date'})
  z=current.merge(bank[KEY+['xgb']],on=KEY,validate='one_to_one');assert z.groupby('signal_date').xgb.rank(pct=True).equals(z[col])
 a=warm['five'].rename(columns={'xgb':'five','lgbm':'lgbm_five'});b=warm['ten'].rename(columns={'xgb':'ten','lgbm':'lgbm_ten'});w=a.merge(b[KEY+['ten','lgbm_ten']],on=KEY,validate='one_to_one');w=w.loc[np.isfinite(w.d1_open_gap)&w.d1_open_gap.ge(-.01)].copy()
 for old,new in [('five','xF'),('ten','xT'),('lgbm_five','lF'),('lgbm_ten','lT')]:w[new]=w.groupby('signal_date')[old].rank(pct=True)
 w['v5']=.9*w.xT+.1*w.xF;w['source']='RECOVERED_TRUE_OOF_2016_2017';allx=pd.concat([w,current],ignore_index=True).sort_values(KEY)
 labels=pd.read_parquet(C/'result/round22_3/data/ten_candidate.parquet',columns=['trade_date','instrument_id','label_exit_rule_net','actual_exit_date','label_exit_rule_reason','next1_date']).rename(columns={'trade_date':'signal_date'})
 labels['y']=labels.label_exit_rule_net;labels['label_available_at']=pd.to_datetime(labels.actual_exit_date)+pd.Timedelta(hours=15);labels['action_status']='EXECUTED_ACTION'
 unfilled=labels.label_exit_rule_reason.str.startswith('buy_',na=False);labels.loc[unfilled,'y']=0.;labels.loc[unfilled,'label_available_at']=pd.to_datetime(labels.loc[unfilled,'next1_date'])+pd.Timedelta(hours=15);labels.loc[unfilled,'action_status']='UNFILLED_ACTION';labels.loc[labels.y.isna(),'action_status']='UNRESOLVED_ACTION'
 x=allx.merge(labels[KEY+['y','label_available_at','action_status']],on=KEY,how='left',validate='one_to_one');x['fallback_entire_day']=x[['lF','lT']].isna().any(axis=1).groupby(x.signal_date).transform('any');x.to_parquet(R/'OOF_expert_and_action_manifest.parquet',index=False)
 dump(R/'warmup_registry.json',registry);dump(R/'input_manifest.json',[identity(C/'result/round23_6/frozen_components.parquet'),identity(C/'result/round22_3/data/ten_candidate.parquet'),identity(C/'result/round23_7/D0/ten_action_label_replay.json')]);return x

def predict_all(x,method):
 pairs=build_pairs(x.loc[~x.fallback_entire_day]);records=[];outs=[];halves=[]
 for year in range(2018,2027):
  cut=pd.Timestamp(f'{year}-01-01');hist=pairs.loc[pairs.signal_date.ge(f'{year-3}-01-01')&pairs.signal_date.lt(cut)&pairs.pair_available_at.lt(cut)].copy();beta,proof=fit(hist,method=='N2');p=x.loc[x.signal_date.dt.year.eq(year)].copy();p['score']=scores(p,beta);p.loc[p.fallback_entire_day,'score']=p.loc[p.fallback_entire_day,'v5'];p['betaF']=beta[0];p['betaT']=beta[1];p['model_active']=proof['status']=='CONVERGED';p['outer_year']=year;outs.append(p)
  records.append({'method':method,'year':year,'betaF':beta[0],'betaT':beta[1],'fit_cutoff':str(cut),'max_pair_available_at':str(hist.pair_available_at.max()),**proof})
  # Fixed two half-year diagnostics in final pre-Outer year; no tuning/refitting selection.
  for half,lo,hi in [('H1',f'{year-1}-01-01',f'{year-1}-07-01'),('H2',f'{year-1}-07-01',f'{year}-01-01')]:
   diag=hist.loc[hist.signal_date.ge(lo)&hist.signal_date.lt(hi)];halves.append({'method':method,'outer_year':year,'half':half,'dates':diag.signal_date.nunique(),'pairs':len(diag),'frozen_beta_loss':objective(beta,diag,method=='N2')[0] if len(diag) else None,'used_for_selection':False})
 return pd.concat(outs,ignore_index=True),pd.DataFrame(records),pd.DataFrame(halves),pairs

def run(resume=True,seed=17):
 if seed!=17:raise ValueError('AnchorRank has frozen deterministic seed17; no seed search')
 assert json.loads((C/'result/round23_7/D0/STATUS.json').read_text())['EVIDENCE_COMPLETE'];R.mkdir(parents=True,exist_ok=True);start=time.time()
 tests=subprocess.run([sys.executable,'-m','unittest','core.round23_7.test_anchor'],cwd=C,capture_output=True,text=True);(R/'unit_tests.log').write_text(tests.stdout+tests.stderr);assert tests.returncode==0
 x=build_inputs();base=x.loc[x.signal_date.ge('2018-01-01')].copy();pd.testing.assert_series_equal(scores(base,[0,0]),base.v5)
 outputs={};fits=[];halves=[];actualtests=[]
 from core.r225241_sleeve import apply_b03_o01
 for method in ['N1','N2']:
  out,registry,h,pairs=predict_all(x,method);fits.append(registry);halves.append(h);outputs[method]=out;pairs.to_parquet(R/(method+'_all_pairs.parquet'),index=False)
  # Actual model entry: add the same date-dependent target constant, or corrupt all future targets.
  shifted=x.copy();shifted['y']=shifted.y+shifted.signal_date.map(lambda d:float(d.toordinal()%11))
  so,sr,_,_=predict_all(shifted,method)
  # Float64 subtraction of shifted returns changes last bits; audit against optimizer
  # KKT tolerance, and separately require identical stock order. No economic gate changes.
  beta_diff=float(np.abs(sr[['betaF','betaT']].to_numpy()-registry[['betaF','betaT']].to_numpy()).max())
  np.testing.assert_array_equal(sr[['betaF','betaT']],registry[['betaF','betaT']]);np.testing.assert_array_equal(so.score,out.score)
  aa=out.sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort')[KEY].reset_index(drop=True)
  bb=so.sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort')[KEY].reset_index(drop=True)
  pd.testing.assert_frame_equal(aa,bb,check_exact=True)
  actualtests.append({'method':method,'test':'same_date_target_constant','max_beta_float_difference':beta_diff,'numerical_tolerance':1e-8,'stock_order':'PASS_EXACT','original_failed_1e-10_check_retained':True})
  for year in [2018,2022,2026]:
   changed=x.copy();mask=changed.signal_date.ge(f'{year}-01-01')|changed.label_available_at.ge(pd.Timestamp(f'{year}-01-01'));changed.loc[mask,'y']=np.arange(mask.sum())*1e7
   co,cr,_,_=predict_all(changed,method);a=out.loc[out.outer_year.eq(year),KEY+['score']].reset_index(drop=True);b=co.loc[co.outer_year.eq(year),KEY+['score']].reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True)
   for f in [a,b]:f.sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort',inplace=True)
   av,_=apply_b03_o01(a);bv,_=apply_b03_o01(b);pd.testing.assert_frame_equal(av,bv,check_exact=True);actualtests.append({'method':method,'outer_year':year,'future_and_immature_mutation':'PASS_EXACT','prediction_rows':len(a),'plans':'PASS_EXACT'})
  out[KEY+['score','betaF','betaT','model_active','fallback_entire_day']].to_parquet(R/(method+'_predictions.parquet'),index=False)
 pd.concat(fits).to_csv(R/'annual_fits.csv',index=False);pd.concat(halves).to_csv(R/'fixed_inner_half_diagnostics.csv',index=False);dump(R/'real_entry_invariance_tests.json',actualtests)
 outputs['N0']=base.assign(score=base.v5);results=[];runregistry=[]
 for name in ['N0','N1','N2']:
  scoreframe=outputs[name][KEY+['score']]
  for bps in [0,10,20]:
   code=name+('' if bps==0 else '_S'+str(bps//10));summary,path,proof=backtest('round23_7',code,scoreframe,bps=bps,upstream=[sha(R/'CONTRACT.yaml'),sha(R/'annual_fits.csv')],resume=resume);summary['path']=str(path);results.append(summary);runregistry.append({'method':name,'scenario':'S'+str(bps//10),'code':code,**proof});print('D1 actual backtest',code,summary,flush=True)
   if name=='N0' and not bps:
    for artifact in ['nav','trades','overlayed','costs','positions','rejections','sleeve_states']:
     pd.testing.assert_frame_equal(pd.read_parquet(path/(artifact+'.parquet')),pd.read_parquet(C/f'result/round23_6/bt/C00_V5/{artifact}.parquet'),check_exact=False,atol=1e-8,rtol=1e-12)
 pd.DataFrame(results).to_csv(R/'all_results.csv',index=False);pd.DataFrame(runregistry).to_csv(R/'run_manifest.csv',index=False)
 dump(R/'STATUS.json',{'IMPLEMENTED':True,'UNIT_TESTED':True,'REFERENCE_RUN_COMPLETE':'NOT_APPLICABLE_PROJECT_ADAPTATION','TARGET_OOF_COMPLETE':True,'BACKTEST_COMPLETE':True,'EVIDENCE_COMPLETE':False,'new_warmup_base_fits':4,'two_parameter_fits':int(pd.concat(fits).status.eq('CONVERGED').sum()),'economic_backtests':len(results),'runtime_seconds':time.time()-start,'remaining':'full risk statistics and after-the-fact selection evidence'})
 return results
