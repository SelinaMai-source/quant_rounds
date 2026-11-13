"""Explicit-window original Stock execution with original full-history overlays."""
from pathlib import Path
import json,hashlib
import pandas as pd
from .experiment import full_contract
from .cache import transaction,fingerprint
C=Path('/data1/stock-mxy/课题/一进二策略')
def execute(round_,code,scores,*,start='2023-01-03',end='2026-07-24',bps=0,upstream=()):
 from core.r225241_sleeve import market,apply_b03_o01
 from core.local_executor import round7_fixed_sleeve_config,run_local_fixed_sleeve_backtest,write_backtest_artifacts
 from core.round4.runner import attach_round4_fields
 import core.r225241_io as io
 io._ALLOWED=tuple(set(io._ALLOWED)|{'round23_7','round26_1','round23_7_round26_1_master'})
 ordered=scores.sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort');overlay,meta=apply_b03_o01(ordered);source=market();snapshot=Path(source['snapshot_path']);inputs=list((snapshot/'daily_raw').rglob('*.parquet'))+[snapshot/'manifest.json']+[C/f'result/round22_3/data/{h}_{kind}.parquet' for h in ['five','ten'] for kind in ['training','candidate']]
 c=full_contract({'scope':'MATCHED_NESTED_FACTOR_OOF_DEVELOPMENT','start':start,'end':end,'code':code,'extra_buy_bps':bps,'scores_hash':hashlib.sha256(pd.util.hash_pandas_object(scores,index=False).values.tobytes()).hexdigest(),'overlay_hash':hashlib.sha256(pd.util.hash_pandas_object(overlay,index=False).values.tobytes()).hexdigest(),'initial_cash':300000},inputs,upstream);cache=C/'result'/round_/'D3/backtests'/code;fp=fingerprint(c)
 with transaction(cache,c) as (dest,cached):
  if not cached:
   overlay.to_parquet(dest/'overlayed.parquet',index=False);result=run_local_fixed_sleeve_backtest(overlay,source['prepared'],config=round7_fixed_sleeve_config(exit_mode='close_from_d2'),start_date=pd.Timestamp(start).date(),end_date=pd.Timestamp(end).date(),validation_hooks={'buy_extra_slippage_bps':float(bps)} if bps else None);attach_round4_fields(result['nav'],result['trades'],result['report'].setdefault('metrics',{}));write_backtest_artifacts(result,dest,extra_report={'code':code,'scope':'MATCHED_NESTED_FACTOR_OOF_DEVELOPMENT','start':start,'end':end,'fee_schedule':'ROUND7_FLAT_5BP','restarted_matched_capital':True,'initial_cash':300000,'scenario':bool(bps),'baseline':'MATCHED_2023_V5','full_period_baseline_not_mixed':True,'experiment_fingerprint':fp})
 return cache/fp,fp,cached
