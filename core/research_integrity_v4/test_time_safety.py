import unittest
import pandas as pd,numpy as np
from .time_safety import mature_training,asof_versions,validate_formula
class TimeSafetyTests(unittest.TestCase):
 def test_actual_maturity_and_future_label_mutation(self):
  x=pd.DataFrame({'trade_date':['2017-01-01','2017-02-01','2018-01-01'],'label_available_at':['2017-01-04','2019-01-01','2018-01-04'],'target':[1.,2.,3.]})
  a=mature_training(x,'2015-01-01','2018-01-01');x.loc[1:,'target']=-100000
  pd.testing.assert_frame_equal(a,mature_training(x,'2015-01-01','2018-01-01'))
  self.assertEqual(len(a),1)
 def test_future_revision_of_old_event_is_invisible(self):
  x=pd.DataFrame({'id':[1,1],'event_time':['2010-01-01']*2,'known_at':['2010-01-02','2020-01-01'],'revision_time':['2010-01-02','2020-01-01'],'value':[1.,99.]})
  a=asof_versions(x,'2015-01-01',['id'])
  self.assertEqual(a.value.tolist(),[1.])
 def test_asof_prefix(self):
  x=pd.DataFrame({'id':[1,2],'event_time':['2010-01-01','2020-01-01'],'known_at':['2010-01-02','2020-01-02'],'revision_time':['2010-01-02','2020-01-02']})
  pd.testing.assert_frame_equal(asof_versions(x,'2015-01-01',['id']),asof_versions(x.iloc[:1],'2015-01-01',['id']))
 def test_formula_future_and_private_inputs_rejected(self):
  for s in ['Ref($close,-1)','Mean($close,0)','Add($close,$label)','open("/tmp/nav.csv")','rolling($close,center=True)']:
   with self.assertRaises((ValueError,SyntaxError)):validate_formula(s)
  self.assertTrue(validate_formula('Div(Mean($close,5),Mean($close,20))'))
 def test_score_blending_is_not_portfolio_blending(self):
  # Experts select disjoint top stocks; averaging scores ties every stock,
  # while the asset-weight mixture holds both assets.
  a=np.array([1.,0.]);b=np.array([0.,1.]);score=.5*a+.5*b
  score_top=np.eye(2)[np.argmax(score)]
  position=.5*np.eye(2)[np.argmax(a)]+.5*np.eye(2)[np.argmax(b)]
  self.assertFalse(np.array_equal(score_top,position));self.assertAlmostEqual(position.sum(),1.)
