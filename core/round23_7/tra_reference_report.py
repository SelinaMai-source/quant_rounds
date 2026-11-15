"""Execute the locked author's own public benchmark functions, separately."""
import ast,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from core.round23_7.tra_reference import R,A,D,V,verify_frozen
from core.research_integrity_v5.cache import identity,hit
from core.research_integrity_v5.resources import measured

class SingleWorkerParallel:
    def __init__(self,*args,**kwargs):
        kwargs['n_jobs']=1
        self.pool=Parallel(*args,**kwargs)
    def __call__(self,tasks):return self.pool(tasks)

def author_functions():
    notebook=A/'Reports.ipynb'
    assert not subprocess.check_output(['git','-C',str(V),'diff','HEAD','--',str(notebook.relative_to(V))]).strip()
    cells=json.loads(notebook.read_text())['cells'];functions={}
    for cell in cells:
        if cell.get('cell_type')!='code':continue
        text=''.join(cell.get('source',[]))
        text='\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('%'))
        try:tree=ast.parse(text)
        except SyntaxError:continue
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in ['func','backtest']:
                functions[node.name]=ast.get_source_segment(text,node)
    assert set(functions)=={'func','backtest'}
    namespace={'np':np,'pd':pd,'Parallel':SingleWorkerParallel,'delayed':delayed,'ret':pd.read_pickle(D/'ret.pkl').clip(-.1,.1)}
    source='\n\n'.join(functions[name] for name in ['func','backtest'])
    exec(compile(source,str(notebook)+'::<original_functions>','exec'),namespace)
    return namespace,source,notebook

def run():
    verify_frozen();status=json.loads((R/'REFERENCE_STATUS.json').read_text())
    assert status['status']=='REFERENCE_RUN_COMPLETE','Do not call partial training a full reference'
    with measured(R/'reference/original_benchmark_RESOURCES.json'):
        namespace,source,notebook=author_functions();rows=[]
        for phase,path in status['paths'].items():
            path=Path(path);marker=json.loads((path/'COMPLETE.json').read_text());assert hit(path,marker['experiment_fingerprint'])
            metrics,daily=namespace['backtest'](str(path/'pred.pkl'))
            assert all(np.isfinite(float(v)) for v in metrics.values())
            rows.append({'phase':phase,'seed':17,**{k:float(v) for k,v in metrics.items()}})
            daily.rename('author_long80_short80_daily_return').to_csv(R/'reference'/phase/'author_benchmark_daily_returns.csv')
        pd.DataFrame(rows).to_csv(R/'reference/original_benchmark_results.csv',index=False)
        provenance={'status':'COMPLETE_AUTHOR_PUBLIC_BENCHMARK','source_notebook':identity(notebook),'source_functions_sha256':hashlib.sha256(source.encode()).hexdigest(),'original_functions':source,'runtime_only_change':'joblib evaluation n_jobs=-1 routed to1; no model/metric/selection change','returns':identity(D/'ret.pkl'),'benchmark':'original top80 minus bottom80, supplied returns clipped[-.1,.1],365calendar annualization, arithmetic cumulative-return MDD; NOT Stock account NAV/fees','known_original_reference_slice_issue':'unaltered author stock-boundary behavior; target data adapter fixes it; reference does not establish target PIT','license':'MIT, Copyright(c) Microsoft Corporation; UPSTREAM_LICENSE_TRA.txt'}
        (R/'reference/ORIGINAL_BENCHMARK_PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True);return rows

if __name__=='__main__':run()
