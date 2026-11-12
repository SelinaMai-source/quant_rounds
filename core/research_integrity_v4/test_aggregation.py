import unittest,numpy as np
from .aggregation import superlearner_nnls,BernsteinOnlineAggregation
class AggregationTests(unittest.TestCase):
 def test_exact_single_expert(self):
  x=np.arange(1.,21.);z=np.column_stack([x,x*x])
  r=superlearner_nnls(z,x)
  np.testing.assert_allclose(r['weights'],[1,0],atol=1e-10)
 def test_unmatured_feedback_keeps_weights(self):
  b=BernsteinOnlineAggregation(4);w=b.weights().copy()
  for x in range(10):b.predict(np.arange(4.))
  np.testing.assert_array_equal(w,b.weights())
 def test_day_feedback_permutation(self):
  a=BernsteinOnlineAggregation(3);b=BernsteinOnlineAggregation(3)
  r=np.array([[1.,2,3],[4,5,6],[7,8,9]])/30
  a.update_regret(r.mean(0));b.update_regret(r[::-1].mean(0))
  np.testing.assert_allclose(a.weights(),b.weights(),atol=1e-14)
