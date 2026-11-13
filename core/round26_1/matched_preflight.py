"""Real matched native baseline before any generated-factor economic trial."""
from pathlib import Path
import json
import pandas as pd
from core.round26_1.market import C,R
from core.round26_1.factor_models import canonical_fuse
from core.round23_3.contract import E0
from core.research_integrity_v5.window_backtest import execute
from core.research_integrity_v5.evaluation import evaluate


def run():
    scores=canonical_fuse(pd.read_parquet(E0/'five_pred.parquet'),pd.read_parquet(E0/'ten_pred.parquet'))
    frozen=pd.read_parquet(C/'result/round23_6/frozen_components.parquet')[['signal_date','instrument_id','v5']].rename(columns={'v5':'score'})
    keys=['signal_date','instrument_id']
    pd.testing.assert_frame_equal(scores.sort_values(keys).reset_index(drop=True),frozen.sort_values(keys).reset_index(drop=True),check_exact=True)
    records=[]
    for bps in [0,10,20]:
        code='AG_A0'+('' if bps==0 else '_S'+str(bps//10))
        path,fp,hit=execute('round26_1',code,scores,bps=bps)
        records.append({'method':'AG_A0','scenario':'S'+str(bps//10),'code':code,'path':str(path),'experiment_fingerprint':fp,'cache_hit':hit})
        print('ACTUAL_NATIVE_MATCHED_BASELINE',code,path,flush=True)
    dest=R/'seed_stages/BASELINE_PREFLIGHT';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(dest/'run_manifest.csv',index=False)
    pd.DataFrame(records).to_csv(R/'run_manifest.csv',index=False)
    result=evaluate(R,baseline='AG_A0')
    result['scope']='MATCHED_2023_NESTED_FACTOR_OOF_DEVELOPMENT';result.to_csv(R/'matched_baseline_results.csv',index=False)
    (R/'matched_baseline_STATUS.json').write_text(json.dumps({'status':'MATCHED_BASELINE_NATIVE_RUN_COMPLETE','candidate_backtests':0,'baseline_scenarios':3,'window_start':'2023-01-03','initial_cash':300000,'full_history_overlay_warmup':True,'candidate_factor_generation':'RUNNING_NOT_COMPLETE'},indent=2))
    print(result[['experiment_id','CAGR','Sharpe','MaxDD','trades']].to_string(index=False),flush=True)
if __name__=='__main__':run()
