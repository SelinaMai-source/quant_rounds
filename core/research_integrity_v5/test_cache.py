import unittest,tempfile,copy
from pathlib import Path
from .cache import REQUIRED,identity,fingerprint,transaction,hit
class CacheTest(unittest.TestCase):
 def test_each_contract_invalidation(self):
  base={k:{'v':1} for k in REQUIRED};fp=fingerprint(base)
  for k in REQUIRED:
   other=copy.deepcopy(base);other[k]['v']=2
   with self.subTest(field=k):self.assertNotEqual(fp,fingerprint(other))
  self.assertEqual(fp,fingerprint(copy.deepcopy(base)))
 def test_same_mtime_content_change(self):
  import os
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'data';p.write_bytes(b'aaa');s=p.stat();a=identity(p);p.write_bytes(b'bbb');os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns));self.assertNotEqual(a['sha256'],identity(p)['sha256'])
 def test_atomic_success_failure_and_tamper(self):
  c={k:1 for k in REQUIRED};fp=fingerprint(c)
  with tempfile.TemporaryDirectory() as t:
   with transaction(t,c) as (p,cached):self.assertFalse(cached);(p/'result').write_text('actual')
   self.assertTrue(hit(Path(t)/fp,fp))
   with transaction(t,c) as (p,cached):self.assertTrue(cached)
   (Path(t)/fp/'result').write_text('changed');self.assertFalse(hit(Path(t)/fp,fp))
   with self.assertRaises(RuntimeError):
    with transaction(t,c) as (p,cached):raise RuntimeError('failure')
   self.assertFalse(hit(Path(t)/fp,fp))
if __name__=='__main__':unittest.main()
