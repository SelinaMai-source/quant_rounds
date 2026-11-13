import unittest
import numpy as np,pandas as pd
from core.round23_7.anchor import objective,scores,build_pairs,fit
class AnchorTest(unittest.TestCase):
 def test_gradient(self):
  p=pd.DataFrame({'signal_date':pd.to_datetime(['2020-01-01']*3+['2020-01-02']*3),'q':[1,-1,1,-1,1,-1],'gap':[.1,.4,.2,.1,.3,.9],'base':[.1,-.1,.3,.2,-.2,.1],'dF':[.1,.2,.3,-.2,.1,.3],'dT':[-.2,.1,.4,.3,.2,-.1]})
  b=np.array([.12,.07])
  for weighted in [False,True]:
   analytic=objective(b,p,weighted)[1];numeric=np.array([(objective(b+np.eye(2)[j]*1e-6,p,weighted)[0]-objective(b-np.eye(2)[j]*1e-6,p,weighted)[0])/2e-6 for j in range(2)]);np.testing.assert_allclose(analytic,numeric,rtol=1e-6,atol=1e-8)
 def test_zero_endpoint_and_constant_expert(self):
  f=pd.DataFrame({'v5':[.23,.71],'xF':[.5,.8],'xT':[.2,.7],'lF':[.5,.5],'lT':[.5,.5]});self.assertTrue(scores(f,[0,0]).equals(f.v5));self.assertEqual(f.v5.idxmax(),scores(f,[.25,.25]).idxmax())
 def test_insufficient_fallback(self):
  b,s=fit(pd.DataFrame(),True);np.testing.assert_array_equal(b,[0,0]);self.assertIn('FALLBACK',s['status'])
if __name__=='__main__':unittest.main()
