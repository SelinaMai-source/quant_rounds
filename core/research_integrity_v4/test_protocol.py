import json,unittest
from pathlib import Path
class ProtocolTest(unittest.TestCase):
 def test_frozen_scope(self):
  root=Path(__file__).resolve().parents[2]
  protocol=json.loads(next(root.glob("result/round*/experiment_prereg.yaml")).read_text())
  self.assertEqual(protocol["seeds"],[0,1,2])
  self.assertFalse(protocol["ROUND24_EXECUTED"])
  self.assertFalse(protocol["AUTO_PROMOTE"])
