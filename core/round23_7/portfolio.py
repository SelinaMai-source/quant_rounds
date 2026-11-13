"""One-account target adapter using original Stock fill, fee, action and lot kernels.

Shadow event inputs are causal before-order recommendations. No trade-return averaging.
"""
from pathlib import Path
from dataclasses import replace
import copy,json
import numpy as np,pandas as pd

def largest_remainders(requests,cash,lot=100):
 # Requests have already been netted. Blocked names are retained here; no redistribution.
 keys=sorted(requests);cost={k:requests[k][1]*lot for k in keys};desired={k:max(0,int(requests[k][0]//lot)) for k in keys};need=sum(desired[k]*cost[k] for k in keys);scale=min(1.,max(0,cash)/max(need,1e-30));exact={k:desired[k]*scale for k in keys};lots={k:int(np.floor(exact[k]+1e-12)) for k in keys};left=cash-sum(lots[k]*cost[k] for k in keys)
 for k in sorted(keys,key=lambda z:(-(exact[z]-lots[z]),z)):
  if lots[k]<desired[k] and cost[k]<=left+1e-8:lots[k]+=1;left-=cost[k]
 assert sum(lots[k]*cost[k] for k in keys)<=cash+1e-7
 return {k:lots[k]*lot for k in keys},scale

def recommendation(event):
 """Pure causal capital recommendation; no modeled fill/volume/high/low preview."""
 held={};day=pd.Timestamp(event['day'])
 for lot in event['pre_lots']:
  if event['phase']=='open' or day<pd.Timestamp(lot['planned_exit_date']):held[lot['instrument_id']]=held.get(lot['instrument_id'],0)+int(lot['quantity'])
 cash={}
 for intent in event['intents']:cash[intent['instrument_id']]=cash.get(intent['instrument_id'],0)+float(intent['scaled_budget'])
 return {'held_quantity':held,'new_cash_capital':cash,'pre_NAV':float(event['pre_NAV'])}

def simulate(shadows,market,weight_by_year,*,extra_buy_bps=0):
 from core.local_executor import round7_fixed_sleeve_config
 from core.stock_readonly import ensure_stock_src_on_path
 ensure_stock_src_on_path()
 import first_board_hermes.fixed_sleeve_backtest as f
 cfg=round7_fixed_sleeve_config(exit_mode='close_from_d2');account=f._Sleeve(sleeve_id=0,cash=300000.);events={name:{(pd.Timestamp(e['day']),e['phase']):e for e in ev} for name,ev in shadows.items()};names=list(shadows);calendar=market.calendar[market.calendar>=pd.Timestamp('2018-01-02')];trades=[];costs=[];rejections=[];positions=[];navs=[];targets=[];orders=[];cashflows=[];owners={};counter=0
 def mark(day,phase,instrument,fallback=0):
  bar=f._market_row(market.market_lookup,day,instrument);field='open' if phase=='open' else 'close'
  return float(bar[field]) if bar is not None and f._finite(bar.get(field)) else float(fallback)
 def netqty():
  result={}
  for p in account.positions.values():result[p.instrument_id]=result.get(p.instrument_id,0)+p.quantity
  return result
 for day in calendar:
  day=pd.Timestamp(day);f._apply_actions(day,[account],market.action_lookup,cashflows)
  for phase in ['open','close']:
   eventlist=[events[n][(day,phase)] for n in names];w=np.asarray(weight_by_year[day.year],float);assert np.all(w>=0) and abs(w.sum()-1)<1e-10
   # Context at this event is fully available under the original legacy price contract.
   first_target=len(targets)
   current=netqty();price={k:mark(day,phase,k,next(p.last_close for p in account.positions.values() if p.instrument_id==k)) for k in current};NAV=account.cash+sum(current[k]*price[k] for k in current);asset={};cash_asset={};represent={};preweights={};intentlist=[]
   for j,e in enumerate(eventlist):
    if w[j]==0:continue
    rec=recommendation(e);preNAV=rec['pre_NAV'];assert preNAV>0
    for ins,q in rec['held_quantity'].items():
     ref=float(e['prices'].get(ins,mark(day,phase,ins,0)));price.setdefault(ins,mark(day,phase,ins,ref));asset[ins]=asset.get(ins,0)+w[j]*float(q)*ref/preNAV
    for ins,amount in rec['new_cash_capital'].items():
     cash_asset[ins]=cash_asset.get(ins,0)+w[j]*amount/preNAV
     price.setdefault(ins,mark(day,phase,ins,0))
    for p in e['pre_lots']:
     item=dict(p);item['represent_weight']=w[j]
     key=p['instrument_id']
     if key not in represent or w[j]>represent[key]['represent_weight']:represent[key]=item
    for intent in e['intents']:
     item=dict(intent);item['represent_weight']=w[j];intentlist.append(item);ins=item['instrument_id'];prior=represent.get(ins)
     if prior is None or 'scaled_budget' not in prior or w[j]>prior['represent_weight']:represent[ins]=item
   desired={};raw_quantities={}
   assert sum(asset.values())+sum(cash_asset.values())<=1+1e-7
   for ins in set(current)|set(asset)|set(cash_asset):
    q=asset.get(ins,0)*NAV/price[ins] if price.get(ins,0)>0 else current.get(ins,0)
    if cash_asset.get(ins,0)>0 and phase=='open':
     bar=f._market_row(market.market_lookup,day,ins)
     if bar is not None and price.get(ins,0)>0:
      modeled=f._modeled_buy_reference_price(bar,price[ins],cfg.costs.slippage_bps+extra_buy_bps)
      unit=modeled*(1+(cfg.costs.commission_bps+cfg.costs.transfer_fee_bps)/10000)
      q+=cash_asset[ins]*NAV/unit
    raw_quantities[ins]=q
    # Preserve exactly unchanged odd lots; remove none of a newly locked lot.
    locked=sum(p.quantity for p in account.positions.values() if p.instrument_id==ins and day<=pd.Timestamp(p.entry_date));raw=q
    q=current.get(ins,0) if abs(q-current.get(ins,0))<=1e-7 else int(np.floor(q/100+1e-9))*100
    desired[ins]=max(locked,q);targets.append({'trade_date':day,'phase':phase,'instrument_id':ins,'pre_NAV':NAV,'raw_target_weight':asset.get(ins,0)+cash_asset.get(ins,0),'held_asset_weight':asset.get(ins,0),'new_allin_cash_capital_weight':cash_asset.get(ins,0),'raw_target_quantity':raw,'T1_locked_quantity':locked,'projected_quantity':desired[ins],'current_quantity':current.get(ins,0),'reference_price':price.get(ins,0),'weights':json.dumps(w.tolist())})
   # Sell net reductions first. No stock is sold through T+1 or its actual price block.
   for ins in sorted(current):
    sell=max(0,current[ins]-desired.get(ins,0));bar=f._market_row(market.market_lookup,day,ins);reason=f._sell_open_block_reason(bar,cfg.limit_up_tolerance) if phase=='open' else f._sell_block_reason(bar,cfg.limit_up_tolerance)
    if sell and reason is not None:
     remaining=sell
     for key,p in sorted(account.positions.items(),key=lambda v:(pd.Timestamp(v[1].entry_date),v[0])):
      if p.instrument_id!=ins or day<=pd.Timestamp(p.entry_date) or remaining<=0:continue
      count=min(remaining,p.quantity);remaining-=count
      rejections.append({'event':'sell_open_rejected' if phase=='open' else 'sell_rejected','reason':reason,'execution_date':day,'signal_date':p.signal_date,'cohort_id':p.cohort_id,'sleeve_id':owners.get(key,0),'instrument_id':ins,'phase':phase,'quantity':count})
     continue
    for key,p in sorted(list(account.positions.items()),key=lambda v:(pd.Timestamp(v[1].entry_date),v[0])):
     if p.instrument_id!=ins or sell<=0 or day<=pd.Timestamp(p.entry_date):continue
     count=min(p.quantity,sell);ratio=count/p.quantity;piece=copy.copy(p);piece.quantity=count;piece.initial_quantity=int(round(p.initial_quantity*ratio))
     for attr in ['buy_notional','buy_commission','buy_transfer_fee','buy_cost','cash_dividend']:setattr(piece,attr,getattr(p,attr)*ratio)
     trade,cost=f._close_position(account,piece,day,bar,cfg,reference_field='open' if phase=='open' else 'close');owner=owners.get(key,0);trade['sleeve_id']=owner
     for c in cost:c['sleeve_id']=owner
     trades.append(trade);costs.extend(cost);orders.append({'trade_date':day,'phase':phase,'instrument_id':ins,'side':'sell','quantity':count,'cohort_id':p.cohort_id});sell-=count
     if count==p.quantity:del account.positions[key];owners.pop(key,None)
     else:
      p.quantity-=count;p.initial_quantity-=piece.initial_quantity
      for attr in ['buy_notional','buy_commission','buy_transfer_fee','buy_cost','cash_dividend']:setattr(p,attr,getattr(p,attr)-getattr(piece,attr))
   current=netqty();requests={};buy_info={}
   if phase=='open':
    for ins,target in desired.items():
     quantity=max(0,target-current.get(ins,0))
     if quantity<=0:continue
     bar=f._market_row(market.market_lookup,day,ins);reference=price.get(ins,0)
     if bar is None or not f._finite(reference) or reference<=0:rejections.append({'event':'buy_rejected','reason':'missing_market_row','execution_date':day,'instrument_id':ins,'phase':phase});continue
     modeled=f._modeled_buy_reference_price(bar,reference,cfg.costs.slippage_bps+extra_buy_bps);unit=modeled*(1+(cfg.costs.commission_bps+cfg.costs.transfer_fee_bps)/10000);requests[ins]=(quantity,unit);buy_info[ins]=(bar,modeled,unit)
    quantities,scale=largest_remainders(requests,account.cash)
    # Frozen net order quantities are computed before any tradability rejection.
    for ins in sorted(quantities):
     quantity=quantities[ins]
     if quantity<=0:continue
     meta=represent[ins];bar,modeled,unit=buy_info[ins];sid=int(meta.get('sleeve_id',0));signal=pd.Timestamp(meta.get('signal_date',day));planned=pd.Timestamp(meta.get('planned_exit_date',day));allocation=float(meta.get('allocation_multiplier',1.));raw_budget=float(meta.get('target_name_budget',quantity*unit));budget=quantity*unit+max(1e-8,quantity*unit*1e-12)
     temp=f._Sleeve(sleeve_id=sid,cash=account.cash);config=replace(cfg,top_k=1,rank_name_budget_nav_fractions=None,name_budget_nav_fraction=None,name_budget=budget,cohort_cash_fraction=1.)
     if extra_buy_bps:config=replace(config,high_gap_threshold=-10.,high_gap_extra_entry_slippage_bps=float(extra_buy_bps))
     scheduled={'signal_date':signal,'planned_exit_date':planned,'name_budget_nav_fraction':None,'picks':[{'instrument_id':ins,'score':float(meta.get('score',0)),'selection_rank':1,'allocation_multiplier':1.}]};before=len(rejections);oldcost=len(costs)
     f._open_cohort(temp,scheduled,day,market.market_lookup,config,rejections,costs,NAV,None);account.cash=temp.cash
     orders.append({'trade_date':day,'phase':phase,'instrument_id':ins,'side':'buy','quantity':quantity,'sleeve_id':sid,'signal_date':signal,'planned_capital':raw_budget,'projected_capital':quantity*unit,'allocation_multiplier':allocation,'scale':scale,'selection_rank':int(meta.get('rank',0))})
     for key,p in temp.positions.items():
      counter+=1
      if key in account.positions:key=key+f'|rebalance{counter}'
      p.target_name_budget=raw_budget;p.allocation_multiplier=allocation;account.positions[key]=p;owners[key]=sid
    # Preserve attempted zero-lot orders for the real vertex, without reassigning money.
    seen_zero=set()
    for source_intent in intentlist:
     ins=source_intent['instrument_id']
     if ins in seen_zero:continue
     seen_zero.add(ins);intent=represent[ins]
     raw_delta=raw_quantities.get(ins,0)-current.get(ins,0)
     bar=f._market_row(market.market_lookup,day,ins);reason=f._buy_block_reason(bar,cfg.limit_up_tolerance)
     if ins not in requests and intent['scaled_budget']>0 and (0<raw_delta<100 or reason is not None):
      cohort=intent.get('cohort_id',f"{pd.Timestamp(intent['signal_date']).date()}-sleeve-{intent['sleeve_id']}")
      orders.append({'trade_date':day,'phase':phase,'instrument_id':ins,'side':'buy','quantity':0,'sleeve_id':intent['sleeve_id'],'signal_date':pd.Timestamp(intent['signal_date']),'planned_capital':intent['target_name_budget'],'projected_capital':0.,'allocation_multiplier':intent.get('allocation_multiplier',1.),'selection_rank':intent.get('rank',0),'scale':1.,'intent_status':'ZERO_LOT_OR_UNPRICED'})
      rejections.append({'event':'buy_rejected','reason':reason or 'insufficient_cash_lot','signal_date':pd.Timestamp(intent['signal_date']),'execution_date':day,'instrument_id':ins,'sleeve_id':intent['sleeve_id'],'cohort_id':cohort})

   assert account.cash>=-1e-7
   actual=netqty()
   for rec in targets[first_target:]:
    if rec['trade_date']==day and rec['phase']==phase:rec['actual_quantity_after']=actual.get(rec['instrument_id'],0);rec['tracking_error_RMB']=(actual.get(rec['instrument_id'],0)-rec['raw_target_quantity'])*rec['reference_price']
  mv=0.
  for key,p in account.positions.items():
   p.last_close=mark(day,'close',p.instrument_id,p.last_close);value=p.quantity*p.last_close;mv+=value;positions.append({'trade_date':day,'sleeve_id':owners.get(key,0),'cohort_id':p.cohort_id,'instrument_id':p.instrument_id,'quantity':p.quantity,'last_close':p.last_close,'market_value':value,'signal_date':p.signal_date,'planned_exit_date':p.planned_exit_date})
  navs.append({'trade_date':day,'cash':account.cash,'market_value':mv,'total_asset':account.cash+mv,'live_sleeves':len(set(owners.values()))})
 return {'nav':pd.DataFrame(navs),'trades':pd.DataFrame(trades),'costs':pd.DataFrame(costs),'positions':pd.DataFrame(positions),'rejections':pd.DataFrame(rejections),'cashflows':pd.DataFrame(cashflows),'target_projection':pd.DataFrame(targets),'net_orders':pd.DataFrame(orders)}
