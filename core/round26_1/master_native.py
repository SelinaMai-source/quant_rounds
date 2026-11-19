"""MASTER scores through the unchanged authoritative Stock account."""
import hashlib
from pathlib import Path
import pandas as pd
from core.round26_1.master_provider import C,R
from core.research_integrity_v5.cache import transaction,fingerprint
from core.research_integrity_v5.experiment import full_contract
from core.research_integrity_v5.resources import measured
_INPUTS=set()

def execute(code,scores,bps=0,upstream=(),start='2018-01-02',end='2026-07-24'):
    from core.r225241_sleeve import market,apply_b03_o01
    from core.local_executor import round7_fixed_sleeve_config,run_local_fixed_sleeve_backtest,write_backtest_artifacts
    from core.round4.runner import attach_round4_fields
    import core.r225241_io as io
    io._ALLOWED=tuple(set(io._ALLOWED)|{'round23_7','round26_1','round23_7_round26_1_master'})
    reads=[];reader=pd.read_parquet
    def traced(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).is_file():reads.append(Path(path))
        return reader(path,*args,**kwargs)
    pd.read_parquet=traced
    try:
        ordered=scores.sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort')
        overlay,metadata=apply_b03_o01(ordered);source=market()
    finally:pd.read_parquet=reader
    _INPUTS.update(reads);snapshot=Path(source['snapshot_path'])
    inputs=list(_INPUTS)+[snapshot/'manifest.json']+list((snapshot/'daily_raw').rglob('*.parquet'))+[C/'result/round23_6/frozen_components.parquet']
    inputs+=[C/f'result/round22_3/data/{h}_{kind}.parquet' for h in ['five','ten'] for kind in ['training','candidate']]
    parameters={'scope':'MASTER_TRUE_NESTED_OOF_DEVELOPMENT','code':code,'windows':{'start':start,'end':end},'extra_buy_bps':bps,'capital':300000,'scores_hash':hashlib.sha256(pd.util.hash_pandas_object(scores,index=False).values.tobytes()).hexdigest(),'overlay_hash':hashlib.sha256(pd.util.hash_pandas_object(overlay,index=False).values.tobytes()).hexdigest()}
    c=full_contract(parameters,inputs,upstream);fp=fingerprint(c);root=R/'backtests'/code
    with transaction(root,c) as (dest,cached):
        if not cached:
            with measured(dest/'RESOURCES.json'):
                overlay.to_parquet(dest/'overlayed.parquet',index=False)
                result=run_local_fixed_sleeve_backtest(overlay,source['prepared'],config=round7_fixed_sleeve_config(exit_mode='close_from_d2'),start_date=pd.Timestamp(start).date(),end_date=pd.Timestamp(end).date(),validation_hooks={'buy_extra_slippage_bps':float(bps)} if bps else None)
                attach_round4_fields(result['nav'],result['trades'],result['report'].setdefault('metrics',{}))
                write_backtest_artifacts(result,dest,extra_report={'code':code,'scope':'MASTER_TRUE_NESTED_OOF_DEVELOPMENT','start':start,'end':end,'initial_cash':300000,'fee_schedule':'ROUND7_FLAT_5BP','baseline':'matched actual no-factor original XGB replay; formal historical v5 separate','experiment_fingerprint':fp,'scenario':bool(bps),'no_MASTER_specific_execution_path':True})
    return root/fp,fp,cached
