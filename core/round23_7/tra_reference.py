"""Full locked legacy author TRA reference, with observation-only runtime hooks."""
from pathlib import Path
import copy,datetime,importlib.util,importlib.metadata,json,os,shutil,signal,subprocess,sys,threading,time
import numpy as np
import pandas as pd
import torch
import psutil
from ruamel.yaml import YAML
from core.research_integrity_v5.cache import identity,sha,source_manifest,fingerprint,transaction
from core.research_integrity_v5.resources import measured

C=Path('/data1/stock-mxy/课题/一进二策略')
B=C/'research/round23_7_26_1_execution'
V=C/'research/round23_6_26_execution/vendor/TRA'
A=V/'examples/benchmarks/TRA'
D=C/'research/round23_6_26_execution/public_data/TRA/kdd21'
R=C/'result/round23_7/D4'

def load_author():
    # Removed pandas spelling only; preserve original slice algorithm for reference.
    if not hasattr(pd.MultiIndex,'is_lexsorted'):
        pd.MultiIndex.is_lexsorted=pd.MultiIndex._is_lexsorted
    modules=[]
    for name in ['dataset','model']:
        key='locked_TRA_legacy_'+name
        if key not in sys.modules:
            spec=importlib.util.spec_from_file_location(key,A/'src'/f'{name}.py')
            module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
        modules.append(sys.modules[key])
    return tuple(modules)

def config(phase,logdir,init_state=None):
    name='config_alstm_tra_init.yaml' if phase=='init' else 'config_alstm_tra.yaml'
    cfg=YAML(typ='safe',pure=True).load((A/'configs'/name).read_text())
    model=copy.deepcopy(cfg['task']['model']['kwargs'])
    model.update(seed=17,logdir=str(logdir))
    if phase=='route':
        if init_state is None:raise ValueError('Reference TRA requires full author init checkpoint')
        model['model_init_state']=str(init_state)
    dataset=copy.deepcopy(cfg['task']['dataset']['kwargs'])
    dataset['handler']['kwargs']['data_loader']['kwargs']['config']={'feature':str(D/'feature.pkl'),'label':str(D/'label.pkl')}
    return model,dataset

def verify_frozen():
    frozen=json.loads((R/'CONTRACT.yaml').read_text())
    assert subprocess.check_output(['git','-C',str(V),'rev-parse','HEAD'],text=True).strip()==frozen['source_commit']
    for name,value in frozen['source_and_configs_sha256'].items():assert sha(V/name)==value,name
    for name,value in frozen['public_reference_inputs'].items():assert identity(name)['sha256']==value['sha256'],name
    return frozen

def reference_contract(phase,model,dataset,upstream=()):
    import qlib
    own=[Path(__file__),C/'core/research_integrity_v5/cache.py',C/'core/research_integrity_v5/resources.py']
    inputs=[D/f'{n}.pkl' for n in ['feature','label','ret']]+[R/'CONTRACT.yaml',R/'PUBLIC_REFERENCE_CPU_RUNTIME_CLARIFICATION.json',A/'configs'/('config_alstm_tra_init.yaml' if phase=='init' else 'config_alstm_tra.yaml')]
    parameters={'phase':phase,'model':copy.deepcopy(model),'dataset':dataset,'compatibility':'pandas is_lexsorted spelling alias only; source algorithm unchanged','CPU_threads':8,'RSS_limit_GiB':16}
    parameters['model'].pop('logdir',None)
    if phase=='route':parameters['model']['model_init_state']=identity(model['model_init_state'])
    return {'code':source_manifest(own+[A/'src/model.py',A/'src/dataset.py',Path(qlib.__file__).parent]),'third_party':{'locked_source_commit':verify_frozen()['source_commit']},'dirty_diff':{n:__import__('hashlib').sha256(subprocess.check_output(['git','-C',str(B/n),'diff','HEAD'])).hexdigest() for n in ['round23_7','round26_1']},'environment':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},'parameters':parameters,'seed':17,'features':list(pd.read_pickle(D/'feature.pkl').columns),'universe':'unchanged author public benchmark index; no target-domain claim','calibration':'unchanged supplied processed public features/labels','data':[identity(p) for p in inputs],'label':'author supplied label.pkl','maturity':'original benchmark fixed horizon21; not asserted as variable-policy A-share maturity','windows':dataset['segments'],'execution':{'reference':'original legacy neural fit/test; separate from target Stock execution','default_stopping':'500epochs or20validation nonimprovements;100train batches/epoch'},'upstream':list(upstream)}

def observed_model_class(author):
    class ObservedTRAModel(author.TRAModel):
        def _event(self,kind,**extra):
            p=Path(self.logdir)/'LIVE.json';p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'event':kind,'epoch':getattr(self,'observed_epoch',-1),'global_step':self.global_step,**extra},default=str,indent=2))
            print('TRA_REFERENCE',kind,'epoch',getattr(self,'observed_epoch',-1),'step',self.global_step,extra,flush=True)
        def train_epoch(self,dataset):
            self.observed_epoch=getattr(self,'observed_epoch',-1)+1
            self._event('TRAIN_BEGIN')
            out=super().train_epoch(dataset);self._event('TRAIN_END',loss=float(out));return out
        def test_epoch(self,dataset,return_pred=False):
            self._event('EVALUATE_BEGIN',return_pred=return_pred)
            out=super().test_epoch(dataset,return_pred=return_pred)
            self._event('EVALUATE_END',metrics=out[0]);return out
    return ObservedTRAModel

def export_phase(phase,path):
    dest=R/'reference'/phase;dest.mkdir(parents=True,exist_ok=True)
    artifacts=[identity(p) for p in sorted(path.rglob('*')) if p.is_file()]
    for name in ['contract.json','COMPLETE.json','info.json','logs.csv','model.bin','RESOURCES.json','LIVE.json','STOPPING_AUDIT.json']:
        if (path/name).exists():shutil.copy2(path/name,dest/name)
    p=pd.read_pickle(path/'pred.pkl');p.to_parquet(dest/'predictions.parquet')
    (dest/'server_artifacts_manifest.json').write_text(json.dumps(artifacts,indent=2))

def run_reference():
    torch.set_num_threads(8);torch.set_num_interop_threads(2)
    verify_frozen();dsmod,modelmod=load_author();root=B/'TRA_reference';paths={}
    old_handler=signal.getsignal(signal.SIGUSR1)
    def guard_failure(*_):raise MemoryError('Actual self RSS exceeded frozen16GiB guard; no budget or architecture reduction')
    signal.signal(signal.SIGUSR1,guard_failure)
    stop=threading.Event()
    def guard():
        while not stop.wait(.5):
            if psutil.Process().memory_info().rss>16*1024**3:os.kill(os.getpid(),signal.SIGUSR1);return
    thread=threading.Thread(target=guard,daemon=True);thread.start()
    try:
        for phase in ['init','route']:
            model_cfg,ds_cfg=config(phase,root/'placeholder',paths.get('init',Path())/'model.bin' if phase=='route' else None)
            contract=reference_contract(phase,model_cfg,ds_cfg,[identity(paths['init']/'model.bin')] if phase=='route' else [])
            fp=fingerprint(contract)
            with transaction(root/phase,contract) as (dest,cached):
                if not cached:
                    model_cfg['logdir']=str(dest)
                    with measured(dest/'RESOURCES.json'):
                        dataset=dsmod.MTSDatasetH(**ds_cfg)
                        model=observed_model_class(modelmod)(**model_cfg)
                        history={};model.fit(dataset,evals_result=history)
                        epochs=len(history['valid']);valid=np.array([x['IC'] for x in history['valid']],dtype=float)
                        assert epochs>1 and epochs<=500 and np.isfinite(valid).all()
                        best=int(np.argmax(valid));reason='AUTHOR_MAX_EPOCHS' if epochs==500 else 'AUTHOR_EARLY_STOP20'
                        assert epochs==500 or epochs-1-best>=20
                        audit={'status':'PASS_FULL_AUTHOR_STOPPING','epochs_completed':epochs,'best_epoch':best,'stop_reason':reason,'default_epoch_cap':500,'default_patience':20,'global_step':model.global_step,'max_train_batches_per_epoch':100,'no_test_selection':True,'reference_dataset_rows':len(dataset._index),'source':str(A/'src/model.py'),'reference_boundary_behavior':'unaltered source _create_ts_slices may include preceding instrument rows; target adapter must clip instrument boundary; reference metrics do not establish target PIT'}
                        (dest/'STOPPING_AUDIT.json').write_text(json.dumps(audit,indent=2))
                        del model,dataset
            paths[phase]=root/phase/fp;export_phase(phase,paths[phase])
            (R/'REFERENCE_PROGRESS.json').write_text(json.dumps({'completed_phase':phase,'path':str(paths[phase]),'fingerprint':fp,'cache_hit':cached,'REFERENCE_RUN_COMPLETE':phase=='route'},indent=2))
        (R/'REFERENCE_STATUS.json').write_text(json.dumps({'status':'REFERENCE_RUN_COMPLETE','original_default_stopping_both_phases':True,'paths':{k:str(v) for k,v in paths.items()},'scope':'public legacy implementation reference; not target backtest; target adaptation still pending'},indent=2))
    finally:stop.set();thread.join();signal.signal(signal.SIGUSR1,old_handler)

if __name__=='__main__':run_reference()
