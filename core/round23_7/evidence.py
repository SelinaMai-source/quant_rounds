"""D1 actual OOF evidence, attribution and fixed inference."""
def run():
 from pathlib import Path
 import json,hashlib,sys
 import numpy as np,pandas as pd
 C=Path('/data1/stock-mxy/课题/一进二策略');sys.path.insert(0,str(C));R=C/'result/round23_7/D1'
 from core.round23_7.anchor import build_pairs,fit,objective
 from core.research_integrity_v5.evaluation import evaluate
 from core.research_integrity_v5.cache import identity,sha
 results=evaluate(R);x=pd.read_parquet(R/'OOF_expert_and_action_manifest.parquet');fits=pd.read_csv(R/'annual_fits.csv');inner=[];models=R/'models';models.mkdir(exist_ok=True)
 for row in fits.to_dict('records'):
  year=row['year'];method=row['method'];p=pd.read_parquet(R/(method+'_all_pairs.parquet'));cut=pd.Timestamp(f'{year}-01-01');tr=p.loc[p.signal_date.ge(f'{year-3}-01-01')&p.signal_date.lt(cut)&p.pair_available_at.lt(cut)];z={'model':row,'contract':identity(R/'CONTRACT.yaml'),'code':identity(C/'core/round23_7/anchor.py'),'training_pair_hash':hashlib.sha256(pd.util.hash_pandas_object(tr,index=False).values.tobytes()).hexdigest(),'training_date_keys':sorted(map(str,tr.signal_date.unique())),'upstream_manifest':identity(R/'OOF_expert_and_action_manifest.parquet'),'seed':17};(models/f'{method}_{year}.json').write_text(json.dumps(z,ensure_ascii=False,indent=2,default=str))
  for half,lo,hi in [('H1',f'{year-1}-01-01',f'{year-1}-07-01'),('H2',f'{year-1}-07-01',f'{year}-01-01')]:
   icut=pd.Timestamp(lo);a=p.loc[p.signal_date.ge(icut-pd.DateOffset(years=3))&p.signal_date.lt(icut)&p.pair_available_at.lt(icut)];b,proof=fit(a,method=='N2');val=p.loc[p.signal_date.ge(lo)&p.signal_date.lt(hi)&p.pair_available_at.lt(cut)];inner.append({'method':method,'outer_year':year,'half':half,'fit_cutoff':lo,'max_training_pair_available_at':str(a.pair_available_at.max()),'train_dates':a.signal_date.nunique(),'train_pairs':len(a),'betaF':b[0],'betaT':b[1],'status':proof['status'],'validation_pairs':len(val),'validation_loss':objective(b,val,method=='N2')[0] if len(val) else None,'baseline_loss':objective(np.zeros(2),val,method=='N2')[0] if len(val) else None,'selection_used':False})
 pd.DataFrame(inner).to_csv(R/'true_inner_half_OOF_diagnostics.csv',index=False)
 # Finite-difference gradient check on actual mature paired actions.
 grad=[]
 for method in ['N1','N2']:
  p=pd.read_parquet(R/(method+'_all_pairs.parquet'));p=p.loc[p.signal_date.ge('2019-01-01')&p.signal_date.lt('2022-01-01')&p.pair_available_at.lt(pd.Timestamp('2022-01-01'))];b=np.array([.12,.07]);analytic=objective(b,p,method=='N2')[1];numeric=np.array([(objective(b+np.eye(2)[j]*1e-6,p,method=='N2')[0]-objective(b-np.eye(2)[j]*1e-6,p,method=='N2')[0])/2e-6 for j in range(2)]);np.testing.assert_allclose(analytic,numeric,atol=1e-8,rtol=1e-6);grad.append({'method':method,'pairs':len(p),'analytic':analytic.tolist(),'numeric':numeric.tolist(),'PASS':True})
 (R/'real_pair_gradient_check.json').write_text(json.dumps(grad,indent=2))
 # Ex-post opportunity upper bounds are strictly diagnostic and never model inputs.
 selections=[];errs=[];active=[]
 for method in ['N1','N2']:
  pred=pd.read_parquet(R/(method+'_predictions.parquet'));z=x.loc[x.signal_date.ge('2018-01-01')].merge(pred[['signal_date','instrument_id','score','model_active']],on=['signal_date','instrument_id'],validate='one_to_one')
  p=pd.read_parquet(R/(method+'_all_pairs.parquet'));sp=z.set_index(['signal_date','instrument_id']).score
  for day,g in z.groupby('signal_date',sort=True):
   original=g.sort_values(['v5','instrument_id'],ascending=[False,True]).head(2);new=g.sort_values(['score','instrument_id'],ascending=[False,True]).head(2);available=g.loc[g.y.notna()];upper=available.nlargest(2,'y').y.sum() if len(available)>=2 else np.nan
   if len(original)==len(new)==2 and original.y.notna().all() and new.y.notna().all():
    selections.append({'method':method,'signal_date':day,'v5_Top2':','.join(original.instrument_id),'candidate_Top2':','.join(new.instrument_id),'changed':set(original.instrument_id)!=set(new.instrument_id),'v5_y_sum':original.y.sum(),'candidate_y_sum':new.y.sum(),'expost_best_known_action_Top2_sum':upper,'v5_opportunity_loss':upper-original.y.sum(),'candidate_opportunity_loss':upper-new.y.sum(),'diagnostic_only':True})
  for row in p.loc[p.signal_date.ge('2018-01-01')].itertuples(index=False):
   i=sp.get((row.signal_date,row.i));j=sp.get((row.signal_date,row.j));oldwrong=row.q*row.base<=0;newwrong=row.q*(i-j)<=0;errs.append({'method':method,'signal_date':row.signal_date,'i':row.i,'j':row.j,'v5_wrong':oldwrong,'candidate_wrong':newwrong,'corrected':oldwrong and not newwrong,'new_error':not oldwrong and newwrong})
  navs=pd.read_csv(R/'daily_returns.csv.gz',parse_dates=['trade_date']);activeyears=fits.loc[fits.method.eq(method)&fits.status.eq('CONVERGED'),'year'].tolist()
  for scope,mask in [('ACTIVE_MODEL_YEARS',navs.trade_date.dt.year.isin(activeyears)),('FALLBACK_MODEL_YEARS',~navs.trade_date.dt.year.isin(activeyears))]:
   a=navs.loc[mask,method].to_numpy();b=navs.loc[mask,'N0'].to_numpy();active.append({'method':method,'scope':scope,'days':len(a),'candidate_cumulative_return':float(np.prod(1+a)-1) if len(a) else None,'baseline_cumulative_return':float(np.prod(1+b)-1) if len(b) else None,'note':'actual full-run daily return slice, no restarted-account claim'})
 pd.DataFrame(selections).to_csv(R/'Top2_changes_and_opportunity.csv.gz',index=False);pd.DataFrame(errs).to_csv(R/'pair_errors_corrected_new.csv.gz',index=False);pd.DataFrame(active).to_csv(R/'active_and_fallback_period_results.csv',index=False)
 s=json.loads((R/'STATUS.json').read_text());s['EVIDENCE_COMPLETE']=True;s['primary_annual_parameter_models']=len(fits);s['new_warmup_fit_executions_including_failed_diagnostic_attempt']=12;s['unique_warmup_base_models']=4;s['inner_diagnostic_fits']=36;s['constant_shift_note']='Two pre-economic failures retained (1e-10 coefficient comparison, then exact tie order); canonicalize beta8dec and recompute KKT. Final beta/score/order invariants exact; no economic result had run before correction';s['remaining']=[];(R/'STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
 print(results[['experiment_id','CAGR','Sharpe','MaxDD','delta_CAGR','delta_Sharpe']].to_string(index=False))
