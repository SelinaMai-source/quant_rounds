import unittest
import pandas as pd,numpy as np
from core.round23_7.portfolio import largest_remainders,simulate
class PortfolioTest(unittest.TestCase):
 def test_cash_scale_and_ties(self):
  result,scale=largest_remainders({'B':(1000,10.),'A':(1000,10.)},10000.)
  self.assertEqual(result,{'A':500,'B':500});self.assertEqual(scale,.5)
  result,_=largest_remainders({'B':(100,10.),'A':(100,10.)},1000.)
  self.assertEqual(result,{'A':100,'B':0})
 def test_T1_blocked_exit_netting_and_no_leverage(self):
  from core.stock_readonly import ensure_stock_src_on_path
  ensure_stock_src_on_path()
  from first_board_hermes.fixed_sleeve_backtest import prepare_fixed_sleeve_market
  dates=pd.to_datetime(['2018-01-02','2018-01-03','2018-01-04','2018-01-05']);bars=pd.DataFrame({'trade_date':dates,'instrument_id':'A','open':[10,9,10,10],'high':[10,9,10,10],'low':[10,9,10,10],'close':[10,9,10,10],'pre_close':[10,10,9,10],'volume':10000.,'paused':False,'limit_up':[11,11,11,11],'limit_down':[9,9,8,9]});m=prepare_fixed_sleeve_market(bars)
  ev=[]
  for i,day in enumerate(dates):
   for phase in ['open','close']:
    intent={'instrument_id':'A','signal_date':day,'planned_exit_date':dates[min(i+1,3)],'sleeve_id':0,'score':1.,'target_name_budget':10020.,'allocation_multiplier':1.,'desired_fractional_quantity':1000.,'scaled_budget':10020.} if i==0 and phase=='open' else None
    ev.append({'day':day,'phase':phase,'pre_NAV':300000.,'target_quantity':{'A':1000.} if intent else {},'prices':{'A':float(bars.open.iloc[i])},'pre_lots':[],'intents':[intent] if intent else []})
  got=simulate({'one':ev,'two':ev},m,{2018:[.5,.5]});self.assertEqual(int(got['trades'].initial_quantity.sum()),1000);self.assertTrue((got['trades'].actual_exit_date>got['trades'].entry_date).all());self.assertEqual(got['trades'].actual_exit_date.min(),dates[2]);self.assertTrue(got['nav'].cash.ge(-1e-8).all());self.assertTrue(got['nav'].total_asset.eq(got['nav'].cash+got['nav'].market_value).all())
 def test_corporate_action_kernel(self):
  from core.stock_readonly import ensure_stock_src_on_path
  ensure_stock_src_on_path()
  import first_board_hermes.fixed_sleeve_backtest as f
  from types import SimpleNamespace
  day=pd.Timestamp('2018-01-03');s=f._Sleeve(sleeve_id=0,cash=100.);p=SimpleNamespace(instrument_id='A',cohort_id='test-lot',quantity=100,cash_dividend=0.);s.positions['p']=p;ledger=[]
  f._apply_actions(day,[s],{(day,'A'):[{'cash_per_share':.5,'share_ratio':.2,'event_id':'synthetic_actual_kernel_test'}]},ledger)
  self.assertEqual(p.quantity,120);self.assertEqual(s.cash,150.);self.assertEqual(p.cash_dividend,50.);self.assertEqual(len(ledger),1)
if __name__=='__main__':unittest.main()
