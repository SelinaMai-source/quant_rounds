"""TRA scores into the unchanged original Stock account, with matched coverage."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from core.round23_7.tra_reference import C,R
from core.research_integrity_v5.cache import identity,fingerprint,transaction
from core.research_integrity_v5.experiment import full_contract
from core.research_integrity_v5.resources import measured
from core.round7.contract import FIVE_HEAD,TEN_HEAD

KEY=['signal_date','instrument_id'];_INPUT_CLOSURE=set()

def fuse(five,ten):
    from core.round17.backtest import load_gaps
    x=five[KEY+['score']].rename(columns={'score':'five'}).merge(ten[KEY+['score']].rename(columns={'score':'ten'}),on=KEY,validate='one_to_one')
    x=x.merge(load_gaps(),on=KEY,how='left',validate='one_to_one')
    x=x.loc[np.isfinite(x.d1_open_gap)&x.d1_open_gap.ge(-.01)].copy()
    x['score']=.9*x.groupby('signal_date').ten.rank(method='average',pct=True)+.1*x.groupby('signal_date').five.rank(method='average',pct=True)
    return x[KEY+['score']].sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort').reset_index(drop=True)

def baseline_sources():
    parts={};inputs=[]
    for head_contract in [FIVE_HEAD,TEN_HEAD]:
        head=head_contract.name
        current=[]
        for year in range(2018,2027):
            p=C/f'result/round23_6/expert_bank/{head}/{year}/predictions.parquet'
            frame=pd.read_parquet(p,columns=['trade_date','instrument_id','xgb']).rename(columns={'trade_date':'signal_date','xgb':'score'})
            frame['signal_date']=pd.to_datetime(frame.signal_date);current.append(frame);inputs.append(identity(p))
        raw=pd.concat(current,ignore_index=True)
        panel=C/f'result/round22_3/data/{head}_candidate.parquet'
        candidate=pd.read_parquet(panel,columns=['trade_date','instrument_id',*head_contract.features])
        candidate['trade_date']=pd.to_datetime(candidate.trade_date)
        candidate=candidate.loc[candidate.trade_date.dt.year.between(2018,2026)]
        eligible=candidate[list(head_contract.features)].replace([np.inf,-np.inf],np.nan).notna().all(axis=1)
        legal=candidate.loc[eligible,['trade_date','instrument_id']].rename(columns={'trade_date':'signal_date'})
        parts[head]=legal.merge(raw,on=KEY,how='left',validate='one_to_one')
        assert np.isfinite(parts[head].score).all(),'Missing real XGB prediction on original feature-complete head domain; must repair control, never select nodes using target'
        inputs.append(identity(panel))
    matched=fuse(parts['five'],parts['ten'])
    formal=pd.read_parquet(C/'result/round23_6/frozen_components.parquet',columns=['signal_date','instrument_id','v5']).rename(columns={'v5':'score'})
    formal['signal_date']=pd.to_datetime(formal.signal_date);formal=formal.loc[formal.signal_date.between('2018-01-02','2026-07-24')]
    proof=formal.merge(matched,on=KEY,how='outer',suffixes=('_formal','_matched'),indicator=True,validate='one_to_one')
    common=proof.loc[proof['_merge'].eq('both')]
    assert len(common)==len(formal) and not proof['_merge'].eq('left_only').any()
    assert np.array_equal(common.score_formal.to_numpy(),common.score_matched.to_numpy()),'Do not attribute a different baseline learner to TRA'
    folder=R/'matched_control';folder.mkdir(parents=True,exist_ok=True)
    proof.to_parquet(folder/'formal_vs_matched_scores.parquet',index=False)
    matched.to_parquet(folder/'scores.parquet',index=False);formal.to_parquet(folder/'formal_v5_scores.parquet',index=False)
    added=proof.loc[proof['_merge'].eq('right_only'),KEY]
    added.groupby('signal_date').size().rename('added_keys').to_csv(folder/'added_prediction_dates.csv')
    status={'status':'EXACT_COMMON_SCORES_COVERAGE_EXTENSION_ONLY' if len(added) else 'EXACT_FORMAL_BASELINE_SCORE_PARITY','formal_rows':len(formal),'matched_rows':len(matched),'common_exact_rows':len(common),'added_rows':len(added),'added_dates':int(added.signal_date.nunique()),'formal_score_end':str(formal.signal_date.max()),'matched_score_end':str(matched.signal_date.max()),'formal_baseline_changed':False,'score_input_is_true_historical_XGB_bank_not_neural_fit':True,'before_any_TRA_performance_comparison':True,'source_inputs':inputs}
    (folder/'COVERAGE_AUDIT.json').write_text(json.dumps(status,indent=2))
    return parts,matched,formal

def materialize(five,ten,baseline):
    """Same legal keys; any missing required prediction causes whole-day fallback."""
    legal=baseline[KEY]
    x=legal.merge(five[KEY+['score']].rename(columns={'score':'five'}),on=KEY,how='left',validate='one_to_one').merge(ten[KEY+['score']].rename(columns={'score':'ten'}),on=KEY,how='left',validate='one_to_one')
    x['invalid_head_prediction']=~np.isfinite(x.five)|~np.isfinite(x.ten)
    bad=x.groupby('signal_date').invalid_head_prediction.any()
    x['score']=.9*x.groupby('signal_date').ten.rank(method='average',pct=True)+.1*x.groupby('signal_date').five.rank(method='average',pct=True)
    failed_dates=bad.index[bad]
    x=x.merge(baseline.rename(columns={'score':'v5_score'}),on=KEY,validate='one_to_one')
    x.loc[x.signal_date.isin(failed_dates),'score']=x.loc[x.signal_date.isin(failed_dates),'v5_score']
    assert np.isfinite(x.score).all() and len(x)==len(legal)
    return x[KEY+['score']].sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort').reset_index(drop=True),bad.rename('whole_day_v5_fallback').reset_index()

def execute(code,scores,bps=0,upstream=()):
    status=json.loads((R/'REFERENCE_STATUS.json').read_text()) if (R/'REFERENCE_STATUS.json').exists() else {}
    assert status.get('status')=='REFERENCE_RUN_COMPLETE','Finish full author reference before A-share economic runs'
    from core.r225241_sleeve import market,apply_b03_o01
    from core.local_executor import round7_fixed_sleeve_config,run_local_fixed_sleeve_backtest,write_backtest_artifacts
    from core.round4.runner import attach_round4_fields
    import core.r225241_io as io
    io._ALLOWED=tuple(set(io._ALLOWED)|{'round23_7','round26_1','round23_7_round26_1_master'})
    reads=[];read=pd.read_parquet
    def traced(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).is_file():reads.append(Path(path))
        return read(path,*args,**kwargs)
    pd.read_parquet=traced
    try:
        ordered=scores.sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True],kind='mergesort')
        overlay,meta=apply_b03_o01(ordered);source=market()
    finally:pd.read_parquet=read
    _INPUT_CLOSURE.update(reads)
    snapshot=Path(source['snapshot_path'])
    inputs=list(_INPUT_CLOSURE)+[C/'result/round23_6/frozen_components.parquet',R/'CONTRACT.yaml',R/'MATCHED_COVERAGE_PROTOCOL.json']+list((snapshot/'daily_raw').rglob('*.parquet'))+[snapshot/'manifest.json']+[C/f'result/round22_3/data/{h}_{kind}.parquet' for h in ['five','ten'] for kind in ['training','candidate']]
    params={'scope':'TRA_2018_2026_DEVELOPMENT','start':'2018-01-02','end':'2026-07-24','code':code,'extra_buy_bps':bps,'scores_hash':hashlib.sha256(pd.util.hash_pandas_object(scores,index=False).values.tobytes()).hexdigest(),'overlay_hash':hashlib.sha256(pd.util.hash_pandas_object(overlay,index=False).values.tobytes()).hexdigest(),'initial_cash':300000}
    c=full_contract(params,inputs,upstream);fp=fingerprint(c);cache=R/'backtests'/code
    with transaction(cache,c) as (dest,cached):
        if not cached:
            with measured(dest/'RESOURCES.json'):
                overlay.to_parquet(dest/'overlayed.parquet',index=False)
                result=run_local_fixed_sleeve_backtest(overlay,source['prepared'],config=round7_fixed_sleeve_config(exit_mode='close_from_d2'),start_date=pd.Timestamp('2018-01-02').date(),end_date=pd.Timestamp('2026-07-24').date(),validation_hooks={'buy_extra_slippage_bps':float(bps)} if bps else None)
                attach_round4_fields(result['nav'],result['trades'],result['report'].setdefault('metrics',{}))
                write_backtest_artifacts(result,dest,extra_report={'code':code,'scope':'TRA_2018_2026_DEVELOPMENT','start':'2018-01-02','end':'2026-07-24','fee_schedule':'ROUND7_FLAT_5BP','initial_cash':300000,'scenario':bool(bps),'baseline':'original formal v5 and separately coverage-matched true XGB bank','experiment_fingerprint':fp,'no_TRA_specific_execution_path':True})
    return cache/fp,fp,cached
