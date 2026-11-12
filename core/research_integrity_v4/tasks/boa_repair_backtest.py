from pathlib import Path
import sys,pandas as pd
C=Path('/data1/stock-mxy/课题/一进二策略');sys.path.insert(0,str(C));sys.dont_write_bytecode=True
import core.r225241_io as io
io._ALLOWED=('round23_6','round26','round23_6_round26_master')
from core.r225241_sleeve import run_round7
R=C/'result/round23_6';x=pd.read_parquet(R/'scores/BOA.parquet')
for bps in (0,10,20):
 code='C11_BOA'+('' if not bps else '_S'+str(bps//10))
 s=run_round7(R/'bt'/code,x[['signal_date','instrument_id','score']],{'code':code,'round':'23.6','scope':'FULL_HISTORY_FIRST_TO_SECOND','information_lane':'LEGACY_V5_COMPARISON','implementation_version':'NEXT1_0930_FEEDBACK_REPAIR'},resume=False,validation_hooks=None if not bps else {'buy_extra_slippage_bps':float(bps)},apply_v5_gate=False,scenario=bool(bps))
 print(code,s,flush=True)
