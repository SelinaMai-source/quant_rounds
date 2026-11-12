from pathlib import Path
import sys,os,json,hashlib
sys.dont_write_bytecode=True
C=Path('/data1/stock-mxy/课题/一进二策略');sys.path.insert(0,str(C))
import core.r225241_io as io
io._ALLOWED=('round23_6','round26','round23_6_round26_master')
import pandas as pd,numpy as np
from core.round23_5.builders import rule_frame,monthly_rules,weights_by_year,apply_weights
from core.r225241_sleeve import run_round7
R=C/'result/round23_6';old=C/'result/round23_5'
components=pd.read_parquet(R/'frozen_components.parquet')
rules=rule_frame(components)
# These are frozen historical expert ledgers; not newly selected winners.
navs={n:old/'bt'/code/'nav.parquet' for n,code in [('v5','B0_V5'),('five','RULE_FIVE'),('ten','RULE_TEN'),('lgbm','RULE_LGBM')]}
if not all(p.exists() for p in navs.values()):
 print('available', [p.name for p in (old/'bt').iterdir()],flush=True)
 raise RuntimeError('resolve expert ledger names before replay')
monthly=monthly_rules(navs);weights,rows=weights_by_year(monthly,.99)
pd.DataFrame(rows).to_csv(R/'meta_weights_by_time.csv',index=False)
scores=apply_weights(rules,weights)[['signal_date','instrument_id','score']]
scores.to_parquet(R/'C01_frozen_scores.parquet',index=False)
for name,x in [('C00_V5',components[['signal_date','instrument_id','v5']].rename(columns={'v5':'score'})),('C01_P1_SCORE_A099',scores)]:
 for bps in (0,10,20):
  code=name if bps==0 else name+'_S'+str(bps//10)
  print('backtest',code,flush=True)
  hooks=None if bps==0 else {'buy_extra_slippage_bps':float(bps)}
  s=run_round7(R/'bt'/code,x,{'code':code,'round':'23.6','information_lane':'LEGACY_V5_COMPARISON','scope':'FULL_HISTORY_FIRST_TO_SECOND'},resume=True,validation_hooks=hooks,apply_v5_gate=False,scenario=bps>0)
  print(code,s,flush=True)
ref=pd.read_parquet(old/'bt/P1_BASE_A099/overlayed.parquet');got=pd.read_parquet(R/'bt/C01_P1_SCORE_A099/overlayed.parquet')
try:pd.testing.assert_frame_equal(ref,got,check_exact=False,atol=1e-12);status='PASS'
except AssertionError as e:status='FAIL';(R/'C01_parity_error.txt').write_text(str(e))
(R/'C01_parity.json').write_text(json.dumps({'score_and_allocation_parity':status,'interpretation':'utility-weighted SCORE fusion; not portfolio FLEXPOOL'},indent=2))
