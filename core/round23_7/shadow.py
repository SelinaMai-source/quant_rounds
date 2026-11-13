"""Causal original-engine policy events captured before orders, never reversed trades."""
from pathlib import Path
from dataclasses import asdict
import copy,json
import pandas as pd,numpy as np
from core.research_integrity_v5.cache import sha
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_7/D2'
def generate(name,scores,*,destination=None,extra_buy_bps=0):
 from core.stock_readonly import ensure_stock_src_on_path
 ensure_stock_src_on_path()
 import first_board_hermes.fixed_sleeve_backtest as f
 from core.r225241_sleeve import market,run_round7
 import core.r225241_io as io
 io._ALLOWED=tuple(set(io._ALLOWED)|{'round23_7','round26_1','round23_7_round26_1_master'})
 m=market()['prepared'];events={};state={};orig={k:getattr(f,k) for k in ['_apply_actions','_open_cohort','_close_position','_build_report']}
 def snapshot(day,phase,sleeves):
  lots=[];qty={};prices={};nav=sum(s.cash for s in sleeves)
  for s in sleeves:
   for position in s.positions.values():
    p=asdict(position);p['sleeve_id']=s.sleeve_id;lots.append(p);bar=f._market_row(m.market_lookup,day,position.instrument_id);price=float(bar['open' if phase=='open' else 'close']) if bar is not None and f._finite(bar.get('open' if phase=='open' else 'close')) else position.last_close;prices[position.instrument_id]=price;nav+=position.quantity*price
    retained=phase=='open' or day<position.planned_exit_date
    if retained:qty[position.instrument_id]=qty.get(position.instrument_id,0)+position.quantity
  return {'day':day,'phase':phase,'pre_NAV':nav,'cash':sum(s.cash for s in sleeves),'target_quantity':qty,'prices':prices,'pre_lots':lots,'intents':[],'source_expert':name}
 def finish_previous():
  if not state:return
  day=state['day'];key=(day,'close')
  if key not in events:events[key]=snapshot(day,'close',state['sleeves'])
 def actions(day,sleeves,lookup,cashflows):
  finish_previous();orig['_apply_actions'](day,sleeves,lookup,cashflows);state.update(day=day,sleeves=sleeves);events[(day,'open')]=snapshot(day,'open',sleeves)
 def opening(sleeve,scheduled,day,lookup,config,rejections,costs,nav,override):
  ev=events[(day,'open')];picks=scheduled['picks'];budgets=[nav*config.rank_name_budget_nav_fractions[int(p['selection_rank'])-1]*float(p['allocation_multiplier']) for p in picks];scale=min(1.,sleeve.cash*config.cohort_cash_fraction/max(sum(budgets),1e-12));ev['event_allowed']=True
  for pick,budget in zip(picks,budgets):
   instrument=str(pick['instrument_id']);bar=f._market_row(lookup,day,instrument);reason=f._buy_block_reason(bar,config.limit_up_tolerance);price=float(bar['open']) if bar is not None and f._finite(bar.get('open')) else np.nan;modeled=f._modeled_buy_reference_price(bar,price,config.costs.slippage_bps+extra_buy_bps) if np.isfinite(price) else np.nan;cap=budget*scale;fractional=cap/(modeled*(1+(config.costs.commission_bps+config.costs.transfer_fee_bps)/10000)) if np.isfinite(modeled) and modeled>0 else 0.
   intent={'instrument_id':instrument,'signal_date':scheduled['signal_date'],'entry_date':day,'planned_exit_date':scheduled['planned_exit_date'],'sleeve_id':sleeve.sleeve_id,'cohort_id':f"{scheduled['signal_date'].date()}-sleeve-{sleeve.sleeve_id}",'score':pick['score'],'rank':pick['selection_rank'],'allocation_multiplier':pick['allocation_multiplier'],'entry_portfolio_nav':nav,'target_name_budget':budget,'scaled_budget':cap,'desired_fractional_quantity':fractional,'shadow_rejection_before_quantity':reason}
   ev['intents'].append(intent)
   if np.isfinite(price):ev['prices'][instrument]=price;ev['target_quantity'][instrument]=ev['target_quantity'].get(instrument,0)+fractional
  return orig['_open_cohort'](sleeve,scheduled,day,lookup,config,rejections,costs,nav,override)
 def closing(sleeve,position,day,bar,config,**kwargs):
  if (day,'close') not in events:events[(day,'close')]=snapshot(day,'close',state['sleeves'])
  events[(day,'close')]['event_allowed']=True
  return orig['_close_position'](sleeve,position,day,bar,config,**kwargs)
 def report(frames,sleeves,config,*args,**kwargs):
  finish_previous();return orig['_build_report'](frames,sleeves,config,*args,**kwargs)
 f._apply_actions=actions;f._open_cohort=opening;f._close_position=closing;f._build_report=report
 try:summary=run_round7(Path(destination) if destination is not None else R/'shadow'/name,scores,{'code':'D2_SHADOW_'+name,'scope':'LEGACY_V5_COMPARISON'},resume=False,validation_hooks={'buy_extra_slippage_bps':float(extra_buy_bps)} if extra_buy_bps else None,scenario=bool(extra_buy_bps),apply_v5_gate=False,counts_as_alpha=False)
 finally:
  for key,value in orig.items():setattr(f,key,value)
 dest=Path(destination) if destination is not None else R/'shadow'/name;ordered=[events[k] for k in sorted(events,key=lambda z:(z[0],0 if z[1]=='open' else 1))];(dest/'causal_policy_events.json').write_text(json.dumps(ordered,ensure_ascii=False,default=str,separators=(',',':')));print('D2 original shadow',name,len(ordered),summary,flush=True);return ordered
