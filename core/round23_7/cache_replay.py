"""Rebind completed D1 scores to complete model-input provenance, no model tuning."""
from pathlib import Path
import json,shutil
import pandas as pd
from core.research_integrity_v5.experiment import backtest
from core.research_integrity_v5.cache import sha
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_7/D1'
def run():
 prior=R/'superseded_cache_provenance';prior.mkdir(exist_ok=True)
 for n in ['run_manifest.csv','all_results.csv']:
  if (R/n).exists() and not (prior/n).exists():shutil.copy2(R/n,prior/n)
 base=pd.read_parquet(C/'result/round23_6/frozen_components.parquet')[['signal_date','instrument_id','v5']].rename(columns={'v5':'score'});runs=[];results=[]
 for method in ['N0','N1','N2']:
  x=base if method=='N0' else pd.read_parquet(R/(method+'_predictions.parquet'))[['signal_date','instrument_id','score']]
  for bps in [0,10,20]:
   code=method+('' if not bps else '_S'+str(bps//10));summary,path,proof=backtest('round23_7',code,x,bps=bps,upstream=[sha(R/'CONTRACT.yaml'),sha(R/'annual_fits.csv'),sha(R/'OOF_expert_and_action_manifest.parquet')]);summary['path']=str(path);results.append(summary);runs.append({'method':method,'scenario':'S'+str(bps//10),'code':code,**proof});old=pd.read_csv(prior/'run_manifest.csv');oldpath=Path(old.loc[old.code.eq(code),'path'].iloc[0]);pd.testing.assert_frame_equal(pd.read_parquet(path/'nav.parquet'),pd.read_parquet(oldpath/'nav.parquet'),check_exact=True);pd.testing.assert_frame_equal(pd.read_parquet(path/'trades.parquet'),pd.read_parquet(oldpath/'trades.parquet'),check_exact=True);print(code,'complete fingerprint replay EXACT',flush=True)
 pd.DataFrame(runs).to_csv(R/'run_manifest.csv',index=False);pd.DataFrame(results).to_csv(R/'all_results.csv',index=False)
 (R/'cache_provenance_repair.json').write_text(json.dumps({'reason':'bind original feature/label/maturity panel content even if score and execution bytes happen unchanged; normalize raw summary scenario flags','new_training_or_search':False,'replayed_backtests':9,'all_NAV_and_trades_identical_to_prior_runs':True,'prior_runs_retained':True},indent=2))
