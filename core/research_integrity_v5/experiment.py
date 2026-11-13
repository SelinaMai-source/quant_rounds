"""Full input closure and completed content-addressed original-engine runs."""
from pathlib import Path
import sys,json,hashlib,subprocess,importlib.metadata,os
import pandas as pd,numpy as np
from .cache import identity,source_manifest,fingerprint,transaction,hit,sha
C=Path('/data1/stock-mxy/课题/一进二策略');B=C/'research/round23_7_26_1_execution'
_INPUT_CLOSURE=set()
def full_contract(parameters,inputs,upstream=(),seed=17,features=None,label='policy-specific',maturity='actual exchange event'):
 from core.stock_readonly import ensure_stock_src_on_path
 ensure_stock_src_on_path()
 import first_board_hermes
 code=source_manifest([C/'core',Path(first_board_hermes.__file__).parent])
 locks=json.loads((C/'result/round23_6/source_lock.json').read_text())
 environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
 diffs={name:hashlib.sha256(subprocess.check_output(['git','-C',str(B/name),'diff','HEAD'])).hexdigest() for name in ['round23_7','round26_1']}
 return {'code':code,'third_party':locks,'dirty_diff':diffs,'environment':environment,'parameters':parameters,'seed':seed,'features':features,'universe':'immutable then-known legacy-v5 candidate keys; label-independent','calibration':parameters.get('calibration','none'),'data':[identity(p) for p in sorted(set(map(Path,inputs)))],'label':label,'maturity':maturity,'windows':parameters.get('windows','2018-2026 development walk-forward; 3y prior fit'),'execution':{'engine':'original local_executor -> Stock fixed_sleeve','cost':'ROUND7_FLAT_5BP','B03_O01':'frozen original','capital':300000,'corporate_action':'original prepared market no actions supplied; zero events, no coverage claim','initial_ledger':'cash300000; no positions'},'upstream':list(upstream)}
def backtest(round_,code,scores,bps=0,upstream=(),resume=True):
 from core.r225241_sleeve import market,apply_b03_o01,run_round7
 import core.r225241_io as io
 io._ALLOWED=tuple(set(io._ALLOWED)|{'round23_7','round26_1','round23_7_round26_1_master'})
 # Trace exact pd parquet file reads for overlays; full actual market partitions also hashed.
 inputs=[];read=pd.read_parquet
 def traced(path,*a,**kw):
  if isinstance(path,(str,Path)) and Path(path).is_file():inputs.append(Path(path))
  return read(path,*a,**kw)
 pd.read_parquet=traced
 try:
  ordered=scores.sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort').copy()
  overlay,_=apply_b03_o01(ordered);m=market()
 finally:pd.read_parquet=read
 _INPUT_CLOSURE.update(inputs);inputs=list(_INPUT_CLOSURE)
 root=Path(m['snapshot_path']);inputs+=list((root/'daily_raw').rglob('*.parquet'))+[root/'manifest.json']
 # Overlay caches may be warm: include the file dependencies of the module closure.
 # These exact series are also bound directly by their data hashes, preventing stale cached context hits.
 from core.round16_5.engine import o01_signal_dates
 context={'overlay_hash':hashlib.sha256(pd.util.hash_pandas_object(overlay,index=False).values.tobytes()).hexdigest(),'o01_dates':list(map(str,o01_signal_dates()))}
 params={'round':round_,'code':code,'buy_extra_bps':bps,'context':context,'scores_hash':hashlib.sha256(pd.util.hash_pandas_object(scores,index=False).values.tobytes()).hexdigest()}
 contract=full_contract(params,inputs,upstream);fp=fingerprint(contract);cache=C/'result'/round_/'runs'/code
 with transaction(cache,contract) as (dest,cached):
  if cached:
   if not resume:raise RuntimeError('Identical immutable run exists; resume required to reuse')
   summary=json.loads((dest/'summary.json').read_text())
  else:summary=run_round7(dest,overlay,{'code':code,'scope':'LEGACY_V5_COMPARISON','experiment_fingerprint':fp},resume=False,validation_hooks={'buy_extra_slippage_bps':float(bps)} if bps else None,already_overlayed=True,apply_v5_gate=False)
 final=cache/fp
 return summary,final,{'experiment_fingerprint':fp,'cache_hit':cached,'path':str(final)}
