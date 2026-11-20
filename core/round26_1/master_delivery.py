"""D5 frozen execution: real reference, then full17 delivery/push,42,73."""
from pathlib import Path
import json,os,sys,traceback,fcntl
import numpy as np
import pandas as pd
from core.round26_1.master_provider import C,B,R,KEY

def validate_inputs():
    from core.round26_1.master_bindings import validate_store
    locations=json.loads((R/'LOCKED_INPUT_LOCATIONS.json').read_text());store=Path(locations['store'])
    if not (store/'COMPLETE.json').is_file():
        # Actual recovery/computation if missing; existence alone never declares ready.
        from core.round26_1.master_provider import prepare as provider_prepare
        from core.round26_1.master_early_targets import run as recover_early
        from core.round26_1.master_features import run as features
        from core.round26_1.master_market import run as market
        from core.round26_1.master_graphs import prepare as graphs
        from core.round26_1.master_store import build
        provider=provider_prepare();recover_early();features(provider,jobs=2);market_path=market();graph=graphs(provider);rebuilt=build(provider,graph,market_path)
        assert rebuilt==store,'Restored inputs changed frozen identity; audit/refreeze, never silently switch data'
    validate_store(store)
    meta=json.loads((store/'STORE.json').read_text());assert Path(meta['graphs'])==Path(locations['graphs'])
    graph=Path(meta['graphs']);contract=json.loads((graph/'contract.json').read_text())
    assert contract['parameters']['liquidity_ranking_dtype']=='float64'
    targets=pd.read_parquet(graph/'targets.parquet');early=pd.read_parquet(locations['legal_early_targets'])
    a=targets.loc[targets.trade_date.dt.year.eq(2012),KEY].sort_values(KEY).reset_index(drop=True)
    b=early.loc[early.trade_date.dt.year.eq(2012),KEY].sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    audit=json.loads((R/'EARLY_TARGET_RECOVERY.json').read_text())
    assert audit['2013_2014_new_only']==audit['2013_2014_old_only']==0 and audit['known_label_max_abs_diff']==0 and audit['exit_dates_equal']
    return store

def assert_no_other_own_learners():
    import psutil
    tags={'core.round26_1.master_fit','core.round26_1.master_pipeline','core.round26_1.master_reference','core.round26_1.master_counterfactual','core.round26_1.master_backend_counterfactual'}
    live=[]
    for proc in psutil.process_iter(['pid','cmdline','status']):
        if proc.info['pid']==os.getpid() or proc.info['status']==psutil.STATUS_ZOMBIE:continue
        command=proc.info['cmdline'] or []
        if tags.intersection(command):live.append({'pid':proc.info['pid'],'command':command})
    if live:raise RuntimeError('Own D5 work already active; resume after it finishes, do not duplicate or exceed2CPU learners: '+json.dumps(live,ensure_ascii=False))

def run(jobs=2,publish=False,resume=True):
    assert 1<=jobs<=2
    B.mkdir(exist_ok=True)
    with (B/'MASTER_delivery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_no_other_own_learners()
        store=validate_inputs()
        from core.round26_1.master_reference_data import ROOT
        if not all((ROOT/f'csi300_dl_{phase}.pkl').is_file() for phase in ['train','valid','test']):
            from core.round26_1.master_public import run as download
            download()  # Actual public download/MAC/CRC, not an existence placeholder.
        from core.round26_1.master_reference import run as reference
        reference()
        from core.round26_1.master_artifacts import resolved_execution
        resolved_execution()
        from core.round26_1.master_pipeline import run as oof
        from core.round26_1.master_trials import run as trials
        from core.round26_1.master_evidence import run as evidence
        from core.round26_1.master_report import run as report
        for seed in [17,42,73]:
            print('MASTER fixed complete delivery seed',seed,flush=True)
            paths=oof(store,seed=seed,jobs=jobs,fit_only=True)
            if seed==17:
                from core.round26_1.master_counterfactual import run_nn
                run_nn(store,jobs=jobs)
            trials(seed,paths)
            if seed==17:
                from core.round26_1.master_backend_counterfactual import run as backend_tests
                backend_tests(jobs=jobs)
            evidence();status=report(store,seed)
            if publish:
                from core.research_integrity_v5.publish import publish as push
                push('round26_1',f'D5 original MASTER seed{seed} full OOF Add/cost/risk and causal audit')
            print('MASTER complete seed checkpoint',seed,status['status'],flush=True)
        assert status['status']=='D5_COMPLETE',status
        return status

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    try:run(jobs=args.jobs,publish=args.publish)
    except BaseException:
        (R/'DELIVERY_FAILURE.log').write_text(traceback.format_exc());raise
