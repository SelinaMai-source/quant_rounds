import sys,os,json,hashlib,traceback
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
C=Path('/data1/stock-mxy/课题/一进二策略');sys.path.insert(0,str(C))
from core.round23_5.oof_validation import load_components
import core.r225241_io as io
io._ALLOWED=("round23_6","round26","round23_6_round26_master")
from core.r225241_sleeve import run_round7
import pandas as pd,numpy as np
R=C/'result/round23_6';R.mkdir(exist_ok=True)
x=load_components();x.to_parquet(R/'frozen_components.parquet',index=False)
print('loaded',len(x),flush=True)
scores=x[['signal_date','instrument_id','v5']].rename(columns={'v5':'score'})
s=run_round7(R/'bt/C00_V5',scores,{'code':'C00_V5','round':'23.6','information_lane':'LEGACY_V5_COMPARISON'},resume=False,apply_v5_gate=False)
print(s,flush=True)
ref=C/'result/round23_5/bt/B0_V5';got=R/'bt/C00_V5'
checks={}
for name in ['overlayed','nav','trades','positions','costs','rejections','cashflows','sleeve_states']:
 a=pd.read_parquet(ref/(name+'.parquet'));b=pd.read_parquet(got/(name+'.parquet'))
 try:pd.testing.assert_frame_equal(a,b,check_exact=False,atol=1e-8,rtol=1e-12);checks[name]={'status':'PASS','rows':len(b)}
 except AssertionError as e:checks[name]={'status':'FAIL','detail':str(e)[:1200]}
checks['status']='PASS' if all(v['status']=='PASS' for v in checks.values()) else 'FAIL'
checks['candidate_key_score_and_ranking']='overlayed frame comparison includes full keys/scores/allocation and sorted order'
checks['plan_order_parity']='NOT_SEPARATELY_INSTRUMENTED; executions/rejections and sleeve states checked'
(R/'baseline_parity.json').write_text(json.dumps(checks,indent=2))
