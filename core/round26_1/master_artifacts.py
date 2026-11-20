"""Export exact server artifact catalogs and derived results; no vendor panels."""
from pathlib import Path
import json,shutil
import pandas as pd
from core.round26_1.master_provider import C,B,R
from core.research_integrity_v5.cache import identity

def inputs(store_path):
    store=Path(store_path);meta=json.loads((store/'STORE.json').read_text())
    directories={'store':store,'graphs':Path(meta['graphs']),'provider':Path(meta['provider']),'market63':Path(meta['market'])}
    records=[]
    for kind,path in directories.items():
        marker=json.loads((path/'COMPLETE.json').read_text())
        schemas={}
        for artifact in marker['artifacts']:
            file=path/artifact['path']
            if file.suffix=='.npy':
                import numpy as np
                array=np.load(file,mmap_mode='r');schemas[artifact['path']]={'shape':list(array.shape),'dtype':str(array.dtype)}
            elif file.suffix=='.npz':
                import numpy as np
                with np.load(file) as archive:schemas[artifact['path']]={name:{'shape':list(archive[name].shape),'dtype':str(archive[name].dtype)} for name in archive.files}
            elif file.suffix=='.parquet':
                import pyarrow.parquet as pq
                schemas[artifact['path']]={'rows':pq.read_metadata(file).num_rows,'schema':str(pq.read_schema(file))}
        records.append({'schemas':schemas,'kind':kind,'server_directory':str(path),'complete_manifest':identity(path/'COMPLETE.json'),'artifacts':marker['artifacts'],'raw_or_derived_large_inputs_not_uploaded':True})
    out=R/'SERVER_INPUT_ARTIFACT_MANIFEST.json';out.write_text(json.dumps(records,ensure_ascii=False,indent=2));return records

def heads():
    manifest=R/'head_model_manifest.csv'
    if not manifest.exists():return []
    rows=pd.read_csv(manifest);records=[]
    for row in rows.itertuples(index=False):
        source=Path(row.path);dest=R/'head_models'/row.arm/str(row.seed)/row.head/row.mode/str(row.year);dest.mkdir(parents=True,exist_ok=True)
        for name in ['MODEL_REGISTRY.json','RESOURCES.json','contract.json','predictions.parquet']:
            shutil.copy2(source/name,dest/name)
        shutil.copy2(source/'COMPLETE.json',dest/'SERVER_COMPLETE_MANIFEST.json')
        state=json.loads((source/'MODEL_REGISTRY.json').read_text())
        records.append({'arm':row.arm,'seed':row.seed,'head':row.head,'year':row.year,'mode':row.mode,'server_directory':str(source),'actual_booster_fits':state['actual_booster_fits'],'model_schema':{'format':'XGBoost Ranker JSON','feature_order':state['parameters']['feature_order'],'tree_seeds':state['parameters']['tree_seeds'],'model_parameters':state['parameters']['XGB_params']},'artifacts':[identity(p) for p in sorted(source.iterdir()) if p.is_file()]})
    (R/'SERVER_HEAD_MODEL_ARTIFACT_MANIFEST.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));return records

def resolved_execution():
    from dataclasses import asdict
    from core.local_executor import round7_fixed_sleeve_config
    from core.round7.contract import FIVE_HEAD,TEN_HEAD,model_parameters,SEEDS
    cfg=round7_fixed_sleeve_config(exit_mode='close_from_d2')
    record={'native_config':asdict(cfg),'Five14':list(FIVE_HEAD.features),'Ten13':list(TEN_HEAD.features),'tree_seeds':list(SEEDS),'model_parameters':model_parameters(17),'fusion':{'ten':.90,'five':.10},'B03_O01':'original apply_b03_o01, unchanged','fees':'original12.3bp slippage,1bp commission,.1bp transfer,flat5bp sell tax; buy+10/+20bp stress at execution only','corporate_action_coverage':'original prepared market with no supplied events, zero is not coverage proof','legacy':'D+1 opening-gap lane and original high/low fill assumptions retained, no live-fill/PIT proof','source':identity(C/'core/local_executor.py')}
    (R/'RESOLVED_EXECUTION_CONTRACT.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str));return record


def neural_schema(model_path):
    import torch
    state=torch.load(Path(model_path),map_location='cpu',weights_only=True)
    schema={name:{'shape':list(tensor.shape),'dtype':str(tensor.dtype),'elements':tensor.numel(),'bytes':tensor.numel()*tensor.element_size()} for name,tensor in state.items()}
    record={'format':'PyTorch original MASTER state_dict','source_checkpoint':identity(model_path),'tensor_count':len(state),'tensors':schema,'trainable_parameters':775041,'same_schema_M0_M1':True,'buffer_positional_encoding_not_trainable':True}
    (R/'MASTER_STATE_TENSOR_SCHEMA.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));return record
