"""Common actual-ledger risk, paired block inference and same-cost SPA."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from arch.bootstrap import SPA

def sharpe(x,axis=0):
 s=np.std(x,axis=axis,ddof=1);m=np.mean(x,axis=axis);return np.divide(m,s,out=np.zeros_like(m),where=s>0)*np.sqrt(252)
def evaluate(root,baseline='N0'):
 root=Path(root);registry=pd.read_csv(root/'run_manifest.csv');rows=[];curves={};attribution=[];yearsrows=[];ledger=[];s3=[];calendar=None
 for row in registry.itertuples(index=False):
  p=Path(row.path);n=pd.read_parquet(p/'nav.parquet');dates=pd.to_datetime(n.trade_date);a=n.total_asset.to_numpy(float);ret=a[1:]/a[:-1]-1;years=(dates.iloc[-1]-dates.iloc[0]).days/365.25
  if calendar is None:calendar=dates
  else:assert calendar.equals(dates)
  path=np.r_[300000,a];dd=a/np.maximum.accumulate(path)[1:]-1;es=ret[ret<=np.quantile(ret,.05)];t=pd.read_parquet(p/'trades.parquet');c=pd.read_parquet(p/'costs.parquet');pos=pd.read_parquet(p/'positions.parquet');ov=pd.read_parquet(p/'overlayed.parquet')
  fees=float(c[['commission','transfer_fee','stamp_tax']].sum().sum());costbasis=t.groupby('entry_date').buy_cost.sum().reindex(dates,fill_value=0).cumsum().to_numpy()-t.groupby('actual_exit_date').buy_cost.sum().reindex(dates,fill_value=0).cumsum().to_numpy();real=t.assign(pnl=t.sell_proceeds-t.buy_cost+t.cash_dividend).groupby('actual_exit_date').pnl.sum().reindex(dates,fill_value=0).cumsum().to_numpy();err=np.abs(300000+real+n.market_value-costbasis-a);assert (err<=np.maximum(.01,1e-8*a)).all()
  z={'experiment_id':row.code,'method':row.method,'scenario':row.scenario,'CAGR':float((a[-1]/300000)**(1/years)-1),'Sharpe':float(sharpe(ret)),'MaxDD':float(dd.min()),'UI':float(np.sqrt(np.mean(dd**2))),'worst63':float(np.min(np.prod(1+np.lib.stride_tricks.sliding_window_view(ret,63),axis=1)-1)),'ES95':float(-es.mean()),'TerminalNAV':float(a[-1]),'trades':len(t),'fees_RMB':fees,'slippage_RMB':float(c.slippage_cost.sum()),'mean_exposure':float((n.market_value/n.total_asset).mean()),'mean_position_breadth':float(pos.groupby('trade_date').instrument_id.nunique().reindex(dates,fill_value=0).mean()),'turnover_annualized':float((t.buy_cost.sum()+t.sell_proceeds.sum())/n.total_asset.mean()/years),'NAV_bridge_max_error':float(err.max()),'status':'EXECUTED','scope':'LEGACY_V5_COMPARISON_DEVELOPMENT'};rows.append(z);curves[row.code]=ret
  for year,g in pd.Series(ret,index=dates.iloc[1:]).groupby(dates.iloc[1:].dt.year.to_numpy()):yearsrows.append({'experiment_id':row.code,'year':year,'return':float(np.prod(1+g)-1),'Sharpe':float(sharpe(g.to_numpy()))})
  ov=ov.sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort');ov['rank']=ov.groupby('signal_date').cumcount()+1;ids=ov.loc[ov['rank'].le(2),['signal_date','instrument_id','rank']];tr=t.merge(ids,on=['signal_date','instrument_id'],how='left',validate='many_to_one');assert tr['rank'].notna().all();tr['net_PnL_RMB']=tr.sell_proceeds-tr.buy_cost+tr.cash_dividend
  tr.to_csv(root/(row.code+'_actual_trade_attribution.csv.gz'),index=False)
  for rank,g in tr.groupby('rank'):attribution.append({'experiment_id':row.code,'rank':int(rank),'trades':len(g),'net_PnL_RMB':float(g.net_PnL_RMB.sum()),'buy_cost_RMB':float(g.buy_cost.sum())})
  rej=pd.read_parquet(p/'rejections.parquet');rr=rej.loc[rej.instrument_id.notna()].merge(ids,on=['signal_date','instrument_id'],how='left');assert rr['rank'].notna().all()
  s3.append({'experiment_id':row.code,'S3':'IDENTICAL_CONSTRAINT_ALREADY_ACTIVE','open_limit_rejections':int(rej.reason.eq('open_limit_up').sum()),'original_plan_only':True,'replacement_count':0})
 allr=pd.DataFrame(rows);idx=allr.set_index('experiment_id')
 for i,z in allr.iterrows():
  name=baseline+('' if z.scenario=='S0' else '_'+z.scenario);b=idx.loc[name]
  for metric in ['CAGR','Sharpe','MaxDD','UI']:allr.loc[i,'delta_'+metric]=z[metric]-b[metric]
  allr.loc[i,'nav_improvement']=z.TerminalNAV/b.TerminalNAV-1
 allr.to_csv(root/'risk_results.csv',index=False);pd.DataFrame(yearsrows).to_csv(root/'yearly_results.csv',index=False);pd.DataFrame(attribution).to_csv(root/'top1_top2_net_PnL.csv',index=False);pd.DataFrame(s3).to_csv(root/'S3_identity_evidence.csv',index=False)
 pd.DataFrame(curves,index=calendar.iloc[1:]).rename_axis('trade_date').to_csv(root/'daily_returns.csv.gz')
 # Entire attempted S0 family, excluding baseline and numerically identical paths.
 unique=[];aliases={};base=curves[baseline]
 for z in allr.loc[allr.scenario.eq('S0')].itertuples(index=False):
  code=z.experiment_id
  if code==baseline:continue
  if np.allclose(curves[code],base,atol=1e-12,rtol=0):aliases[code]='BASELINE_IDENTICAL';continue
  match=next((q for q in unique if np.allclose(curves[code],curves[q],atol=1e-12,rtol=0)),None)
  if match:aliases[code]=match
  else:unique.append(code)
 rng=np.random.default_rng(17);boot=[]
 for block,count in [(20,5000),(10,1000),(40,1000)]:
  arrays={q:[] for q in unique}
  for start in range(0,count,100):
   k=min(100,count-start);beg=rng.integers(0,len(base),size=(k,int(np.ceil(len(base)/block))));ids=((beg[:,:,None]+np.arange(block))%len(base)).reshape(k,-1)[:,:len(base)];br=base[ids];bs=sharpe(br,axis=1);bc=np.expm1(np.log1p(br).sum(1)/years)
   for q in unique:
    cr=curves[q][ids];path=np.cumprod(1+cr,axis=1);bp=np.cumprod(1+br,axis=1);dd=path/np.maximum.accumulate(np.column_stack([np.ones(k),path]),axis=1)[:,1:]-1;bdd=bp/np.maximum.accumulate(np.column_stack([np.ones(k),bp]),axis=1)[:,1:]-1;arrays[q].append(np.column_stack([sharpe(cr,axis=1)-bs,np.expm1(np.log1p(cr).sum(1)/years)-bc,dd.min(1)-bdd.min(1)]))
  for q,a in arrays.items():
   arr=np.concatenate(a);boot.append({'experiment_id':q,'block':block,'draws':count,'delta_Sharpe_q05':float(np.quantile(arr[:,0],.05)),'delta_Sharpe_q50':float(np.quantile(arr[:,0],.5)),'delta_Sharpe_q95':float(np.quantile(arr[:,0],.95)),'delta_CAGR_q05':float(np.quantile(arr[:,1],.05)),'delta_CAGR_q95':float(np.quantile(arr[:,1],.95)),'delta_MaxDD_q05_path_sensitivity_only':float(np.quantile(arr[:,2],.05)),'delta_MaxDD_q95_path_sensitivity_only':float(np.quantile(arr[:,2],.95)),'calendar_years':years,'paired_indices':True})
 pd.DataFrame(boot).to_csv(root/'bootstrap_results.csv',index=False)
 if unique:
  spa=SPA(-base,np.column_stack([-curves[q] for q in unique]),block_size=20,reps=5000,bootstrap='circular',seed=17);spa.compute();proof={'status':'COMPUTED','pvalues':spa.pvalues.to_dict(),'alternatives':unique,'deduplicated':aliases,'loss':'negative actual portfolio daily return','historical_development':True}
 else:proof={'status':'NO_NONIDENTICAL_ALTERNATIVES','alternatives':[],'deduplicated':aliases}
 (root/'SPA.json').write_text(json.dumps(proof,indent=2));return allr
