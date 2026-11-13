import unittest
import numpy as np,pandas as pd,torch
from core.round26_1.alphagen_data import ast_dump,ast_load,PreloadedMarket,FeatureType,guard
from alphagen.data.expression import Feature,Ref,Mean,Corr,Delta,Div,Constant
class AlphaGenTest(unittest.TestCase):
 def test_AST_all_frozen_operator_roundtrip(self):
  from alphagen.config import OPERATORS
  from alphagen.data.expression import UnaryOperator,BinaryOperator,RollingOperator,PairRollingOperator
  f=Feature(FeatureType.CLOSE);g=Feature(FeatureType.VOLUME)
  for op in OPERATORS:
   if issubclass(op,UnaryOperator):expr=op(f)
   elif issubclass(op,BinaryOperator):expr=op(f,g)
   elif issubclass(op,RollingOperator):expr=op(f,5)
   else:expr=op(f,g,5)
   with self.subTest(op=op.__name__):self.assertEqual(str(expr),str(ast_load(ast_dump(expr))))
 def test_future_Ref_rejected(self):
  with self.assertRaises(ValueError):guard(Ref(Feature(FeatureType.CLOSE),-1))
 def test_whole_stock_history_before_candidate_mapping(self):
  dates=pd.bdate_range('2010-01-01',periods=130);tensor=torch.arange(130*6*2,dtype=torch.float32).reshape(130,6,2);data=PreloadedMarket(['A','B'],str(dates[100]),str(dates[-1]),max_backtrack_days=100,max_future_days=0,preloaded_data=(tensor,dates,pd.Index(['A','B'])),device=torch.device('cpu'));f=Mean(Feature(FeatureType.CLOSE),5).evaluate(data)
  self.assertAlmostEqual(float(f[0,0]),float(tensor[96:101,1,0].mean()))
  # This would differ if rolling were performed only on sparse first-board dates.
  self.assertNotEqual(float(f[10,0]),float(tensor[[100,103,105,108,110],1,0].mean()))
 def test_future_bar_append_invariance(self):
  dates=pd.bdate_range('2010-01-01',periods=130);raw=torch.arange(130*6*2,dtype=torch.float32).reshape(130,6,2);base=PreloadedMarket(['A','B'],str(dates[100]),str(dates[-1]),max_backtrack_days=100,max_future_days=0,preloaded_data=(raw,dates,pd.Index(['A','B'])),device=torch.device('cpu'));expr=Delta(Feature(FeatureType.CLOSE),5);got=expr.evaluate(base).clone();mut=raw.clone();mut[120:]=1e12;other=PreloadedMarket(['A','B'],str(dates[100]),str(dates[-1]),max_backtrack_days=100,max_future_days=0,preloaded_data=(mut,dates,pd.Index(['A','B'])),device=torch.device('cpu'));torch.testing.assert_close(got[:20],expr.evaluate(other)[:20],rtol=0,atol=0)
if __name__=='__main__':unittest.main()
