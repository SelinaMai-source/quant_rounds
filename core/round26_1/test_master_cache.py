"""Actual transaction/cache integration; no NN-fit causality claim."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from core.round26_1.master_provider import B,R
from core.round26_1.master_bindings import _validate
from core.round26_1.master_fit import contract
from core.research_integrity_v5.cache import transaction,fingerprint,identity

def run(store):
    c=contract(store,'M0',17,2015,'none')
    root=Path(tempfile.mkdtemp(prefix='cache_integration_',dir=B))
    values=pd.read_parquet(Path(json.loads((Path(store)/'STORE.json').read_text())['graphs'])/'targets.parquet').head(64)
    data=root/'actual_target_rows.parquet';values.to_parquet(data,index=False)
    original_data=data.read_bytes();c['data'].append(identity(data))
    probe=root/'probe_code.py';probe.write_text('# base diagnostic wrapper source\n');original_code=probe.read_bytes();c['code'].append(identity(probe))
    cache=root/'completed'
    def invoke(state):
        with transaction(cache,state) as (dest,cached):
            if not cached:(dest/'CACHE_PROOF.json').write_text(json.dumps({'scope':'transaction integration only','fingerprint':fingerprint(state)}))
        return cached
    assert not invoke(c) and invoke(c)
    results=[]
    for kind in ['code','data','label','maturity','parameters','execution']:
        variant=copy.deepcopy(c)
        if kind=='code':
            probe.write_text('# distinct diagnostic wrapper source\n');variant['code'][-1]=identity(probe);assert variant['code'][-1]['sha256']!=c['code'][-1]['sha256']
        elif kind in ['data','label','maturity']:
            changed=values.copy()
            if kind=='data':changed.loc[changed.index[0],'instrument_id']+='__CACHE_DIAGNOSTIC'
            elif kind=='label':changed.loc[changed.index[0],'label_exit_rule_net']=.123456789
            else:changed.loc[changed.index[0],'label_available_at']=pd.Timestamp('2014-01-02 15:00:00')
            changed.to_parquet(data,index=False);variant['data'][-1]=identity(data);assert variant['data'][-1]['sha256']!=c['data'][-1]['sha256']
        elif kind=='parameters':variant['parameters']['lr']=2e-5
        else:variant['execution']['capital']=300001
        assert fingerprint(variant)!=fingerprint(c) and not invoke(variant)
        data.write_bytes(original_data);probe.write_bytes(original_code)
        results.append({'same_input_paths':True,'mutation':kind,'cache_hit':False,'scope':'diagnostic contract only, no NN/baseline parameter change'})
    assert invoke(c),'Restored exact inputs must hit again'
    # Actual bytes changed beneath a COMPLETE manifest must fail closed, never hit.
    completed=cache/fingerprint(c);marker=completed/'CACHE_PROOF.json';marker.write_text('changed bytes')
    rejected=False
    try:_validate(completed)
    except AssertionError:rejected=True
    assert rejected
    result={'status':'PASS','identical_complete_hit':True,'six_actual_transaction_misses':results,'changed_actual_completed_bytes_rejected':True,'full_transitive_real_NN_inputs_validated_before_constructing_base_contract':True,'scope':'Cache integration only. Real cold NN/backend fits separately required.','fixture_directory':str(root)}
    (R/'MASTER_CACHE_INTEGRATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    import sys
    run(sys.argv[1])
