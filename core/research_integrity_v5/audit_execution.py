"""Original-authority all-candidate exit replay, real plans and accounting."""
from pathlib import Path
import json,sys
import pandas as pd,numpy as np
from .cache import identity,sha
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_7/D0'
def run():
 from core.r225241_sleeve import market,apply_b03_o01,run_round7
 import core.r225241_io as io
 io._ALLOWED=tuple(set(io._ALLOWED)|{'round23_7','round26_1','round23_7_round26_1_master'})
 from core.stock_readonly import ensure_stock_src_on_path
 ensure_stock_src_on_path()
 from first_board_hermes.labels import attach_exit_aligned_labels
 import first_board_hermes.fixed_sleeve_backtest as fsb
 source=market();m=source['prepared'];panel=pd.read_parquet(C/'result/round22_3/data/ten_candidate.parquet');panel=panel.loc[panel.trade_date.ge('2018-01-01')].copy();bars=m.bars.copy()
 mapping=pd.Series(np.arange(len(m.calendar)),index=m.calendar);offset=(panel.cal_idx-panel.trade_date.map(mapping)).dropna().unique();assert len(offset)==1,offset
 bars['cal_idx']=bars.trade_date.map(mapping)+int(offset[0]);assert bars.cal_idx.notna().all()
 rebuilt,audit=attach_exit_aligned_labels(panel[['trade_date','instrument_id','cal_idx']],bars)
 compare=panel[['trade_date','instrument_id','actual_exit_date','label_exit_rule_net','label_exit_rule_reason']].merge(rebuilt,on=['trade_date','instrument_id'],suffixes=('_stored','_replayed'),validate='one_to_one')
 eligible=compare.label_exit_rule_net_stored.notna();diff=(compare.label_exit_rule_net_stored-compare.label_exit_rule_net_replayed).abs();dates=(compare.actual_exit_date_stored!=compare.actual_exit_date_replayed)&eligible
 report={'scope':'all frozen candidate actions 2018-2026 in original market snapshot','rows':len(compare),'stored_valid':int(eligible.sum()),'replayed_valid':int(compare.label_exit_rule_net_replayed.notna().sum()),'label_max_abs_error':float(diff.max()),'valid_to_missing':int((eligible&compare.label_exit_rule_net_replayed.isna()).sum()),'exit_mismatches_on_valid':int(dates.sum()),'source':identity(Path(fsb.__file__)),'snapshot':source['snapshot_path'],'corporate_action_events_in_original_prepared_market':sum(map(len,m.action_lookup.values())),'legacy_caveat':'source loader has no corporate actions passed; original fill clips high/low; parity is not live/PIT proof'}
 report['PASS']=report['label_max_abs_error']<1e-12 and report['valid_to_missing']==0 and report['exit_mismatches_on_valid']==0
 compare.to_parquet(R/'ten_action_label_replay.parquet',index=False);(R/'ten_action_label_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report,flush=True)
 # Trace actual entry plan and monkey-patch the original function without altering source.
 trace=[];original=fsb._open_cohort
 def wrapped(sleeve,scheduled,day,market_lookup,config,rejections,cost_rows,entry_portfolio_nav,entry_price_lookup):
  for pick in scheduled['picks']:trace.append({'entry_date':day,'signal_date':scheduled['signal_date'],'sleeve_id':sleeve.sleeve_id,'cash_before':sleeve.cash,'nav_before':entry_portfolio_nav,**pick})
  return original(sleeve,scheduled,day,market_lookup,config,rejections,cost_rows,entry_portfolio_nav,entry_price_lookup)
 fsb._open_cohort=wrapped
 try:
  x=pd.read_parquet(C/'result/round23_6/frozen_components.parquet')[['signal_date','instrument_id','v5']].rename(columns={'v5':'score'})
  summary=run_round7(R/'baseline_actual_trace',x,{'code':'D0_N0_TRACE','scope':'LEGACY_V5_COMPARISON'},resume=False,apply_v5_gate=False,counts_as_alpha=False)
 finally:fsb._open_cohort=original
 pd.DataFrame(trace).to_csv(R/'actual_plan_trace.csv.gz',index=False)
 parity={}
 for n in ['nav','trades','positions','costs','sleeve_states','rejections','cashflows','overlayed']:
  a=pd.read_parquet(C/f'result/round23_6/bt/C00_V5/{n}.parquet');b=pd.read_parquet(R/f'baseline_actual_trace/{n}.parquet');pd.testing.assert_frame_equal(a,b,check_exact=False,atol=1e-8,rtol=1e-12);parity[n]={'PASS':True,'rows':len(b)}
 (R/'actual_execution_parity.json').write_text(json.dumps(parity,indent=2));print('D0 actual replay parity',summary,flush=True)
 if not report['PASS']:raise RuntimeError('Ten action-label replay failed; dependent paths blocked pending matched repair')
