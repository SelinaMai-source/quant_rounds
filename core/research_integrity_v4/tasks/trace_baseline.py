from pathlib import Path
import sys,json
import pandas as pd,numpy as np
C=Path('/data1/stock-mxy/课题/一进二策略');sys.path.insert(0,str(C));sys.dont_write_bytecode=True
import core.r225241_io as io
io._ALLOWED=('round23_6','round26','round23_6_round26_master')
from core.r225241_sleeve import run_round7
from core.stock_readonly import ensure_stock_src_on_path
ensure_stock_src_on_path()
import first_board_hermes.fixed_sleeve_backtest as fsb
original=fsb._open_cohort
R=C/'result/round23_6';trace=[]
def wrapped(sleeve,scheduled,day,market_lookup,config,rejections,cost_rows,entry_portfolio_nav,entry_price_lookup):
 for p in scheduled['picks']:
  trace.append({'signal_date':str(scheduled['signal_date']),'entry_date':str(day),'sleeve_id':sleeve.sleeve_id,'sleeve_cash_before':sleeve.cash,'entry_portfolio_nav':entry_portfolio_nav,'planned_exit_date':str(scheduled['planned_exit_date']),**p})
 return original(sleeve,scheduled,day,market_lookup,config,rejections,cost_rows,entry_portfolio_nav,entry_price_lookup)
fsb._open_cohort=wrapped
try:
 ref=pd.read_parquet(C/'result/round23_5/bt/B0_V5/overlayed.parquet')
 x=pd.read_parquet(R/'frozen_components.parquet')[['signal_date','instrument_id','v5']].rename(columns={'v5':'score'})
 results=[]
 for name,scores,overlay in [('REF_FROZEN_V5',ref,True),('C00_REGENERATED_V5',x,False)]:
  trace.clear();run_round7(R/'diagnostics'/name,scores,{'code':name,'round':'23.6','scope':'BASELINE_TRACE_DIAGNOSTIC'},resume=False,already_overlayed=overlay,apply_v5_gate=False,counts_as_alpha=False)
  frame=pd.DataFrame(trace);frame.to_csv(R/'diagnostics'/name/'plan_trace.csv.gz',index=False);results.append(frame)
 pd.testing.assert_frame_equal(results[0],results[1],check_exact=False,rtol=1e-12,atol=1e-8)
 q=json.loads((R/'baseline_parity.json').read_text());q['plan_order_trace']={'status':'PASS','rows':len(results[0]),'fields':results[0].columns.tolist(),'evidence':'diagnostics/*/plan_trace.csv.gz'};q['status']='PASS';q['plan_order_parity']='PASS actual original executor _open_cohort call intents and pre-trade capital compared; final quantities checked in trades'
 q['execution_time_chain']='LEGACY_OPEN_INFORMATION; not a strict contemporaneous fill proof; original engine may clip modeled fills against full-day high/low'
 (R/'baseline_parity.json').write_text(json.dumps(q,indent=2))
finally:fsb._open_cohort=original
# Independent Round26 matched frozen baseline, no model/weight winner imported.
run_round7(C/'result/round26/bt/A0_V5',x,{'code':'A0_V5','round':'26','scope':'LEGACY_V5_COMPARISON'},resume=False,apply_v5_gate=False,counts_as_alpha=False)
print('Trace parity and independent round26 baseline executed')
