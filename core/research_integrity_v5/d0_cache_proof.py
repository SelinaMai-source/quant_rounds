"""Actual frozen-baseline full-fingerprint cache replay and exact original parity."""
from pathlib import Path
import json,shutil
import pandas as pd
from .experiment import backtest
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_7/D0'
def run():
 x=pd.read_parquet(C/'result/round23_6/frozen_components.parquet')[['signal_date','instrument_id','v5']].rename(columns={'v5':'score'});proofs=[]
 if (R/'actual_cache_hit_test.json').exists():shutil.copy2(R/'actual_cache_hit_test.json',R/'superseded_before_full_label_panel_binding_cache_test.json')
 for repeat in range(2):
  summary,path,proof=backtest('round23_7','D0_FULL_CONTRACT_V5',x);proofs.append(proof)
  for name in ['nav','trades','overlayed','positions','costs','rejections','sleeve_states']:
   pd.testing.assert_frame_equal(pd.read_parquet(path/(name+'.parquet')),pd.read_parquet(C/f'result/round23_6/bt/C00_V5/{name}.parquet'),check_exact=False,atol=1e-8,rtol=1e-12)
  print('D0 full contract baseline',proof,summary,flush=True)
 assert proofs[1]['cache_hit'] and proofs[0]['experiment_fingerprint']==proofs[1]['experiment_fingerprint']
 (R/'actual_cache_hit_test.json').write_text(json.dumps(proofs,indent=2))
