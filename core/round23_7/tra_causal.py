"""Original TRA/OT with an actual-exit maturity queue and D-known data adapter."""
from pathlib import Path
import copy,hashlib,inspect,json,textwrap
import numpy as np
import pandas as pd
import torch
from core.round23_7.tra_reference import load_author

NAT=np.iinfo(np.int64).min

class Bundle:
    def __init__(self,path,head,*,target_override=None):
        self.path=Path(path)
        self.keys=pd.read_parquet(self.path/'candidate_keys.parquet')
        self.x=np.load(self.path/'features.npy',mmap_mode='r')
        self.history=np.load(self.path/'history_candidate_indices.npy',mmap_mode='r')
        self.valid=np.load(self.path/'valid.npy',mmap_mode='r')
        label=pd.read_parquet(self.path/f'{head}_targets.parquet') if target_override is None else target_override.copy()
        assert self.keys.equals(label[['trade_date','instrument_id']])
        self.y=label.target.to_numpy('float32',copy=True)
        self.available=pd.to_datetime(label.label_available_at).to_numpy('datetime64[ns]').view('i8')
        self.times=pd.to_datetime(self.keys.trade_date).to_numpy('datetime64[ns]').view('i8')
        self.index=pd.MultiIndex.from_frame(self.keys[['instrument_id','trade_date']]).set_names(['instrument','datetime'])
        assert self.x.shape==(len(self.keys),60,6) and self.history.shape==(len(self.keys),60)

class CausalDataset:
    """Author-compatible batches; predict membership never depends on a target."""
    def __init__(self,bundle,year,num_states,batch_size,*,prediction_sink=None,inference_bundle=None,feedback_provider=None):
        self.bundle=bundle;self.year=int(year);self.num_states=int(num_states)
        self.seq_len=60;self.horizon=21;self.batch_size=batch_size;self.drop_last=False;self.shuffle=True
        self.params=(batch_size,False,True);self.segment='all';self._index=bundle.index
        self.predictions=np.full((len(bundle.keys),num_states),np.nan,dtype='float32')
        self.recorded=np.zeros(len(bundle.keys),dtype=bool)
        self.record_kind=np.zeros(len(bundle.keys),dtype='int8')
        self.indices=np.arange(len(bundle.keys));self.feedback_audit={'mature_label_reads':0,'unmature_label_reads':0,'saved_expert_prediction_writes':0}
        self.all_indices=self.indices;self.loss_indices=self.indices;self.fit_cutoff=None
        self.segment_states={'train':(self.predictions,self.recorded,self.record_kind)}
        self.prediction_sink=prediction_sink
        self.inference_bundle=inference_bundle
        self.feedback_provider=feedback_provider
    def prepare(self,segments):
        if isinstance(segments,(list,tuple)):return [self.prepare(s) for s in segments]
        if segments not in ['train','valid','test']:raise ValueError(segments)
        obj=copy.copy(self);obj.segment=segments
        if segments=='test' and self.inference_bundle is not None:
            assert self.bundle.keys.equals(self.inference_bundle.keys)
            obj.bundle=self.inference_bundle
        b=obj.bundle
        train_cut=pd.Timestamp(f'{self.year-1}-01-01').value;outer_cut=pd.Timestamp(f'{self.year}-01-01').value
        if segments=='test':
            end=pd.Timestamp(f'{self.year+1}-01-01').value
            # Label, maturity and sequence validity do not select prediction nodes.
            selected=(b.times>=outer_cut)&(b.times<end)
            all_selected=selected;obj.fit_cutoff=None
        else:
            start=pd.Timestamp(f'{self.year-3 if segments=="train" else self.year-1}-01-01').value
            cutoff=train_cut if segments=='train' else outer_cut
            all_selected=(b.times>=start)&(b.times<cutoff)
            selected=all_selected&(b.available!=NAT)&(b.available<cutoff)&np.isfinite(b.y)&b.valid
            obj.fit_cutoff=cutoff
        obj.indices=np.flatnonzero(selected)
        obj.loss_indices=obj.indices;obj.all_indices=np.flatnonzero(all_selected)
        if segments not in self.segment_states:
            self.segment_states[segments]=(np.full_like(self.predictions,np.nan),np.zeros_like(self.recorded),np.zeros_like(self.record_kind))
        obj.predictions,obj.recorded,obj.record_kind=self.segment_states[segments]
        assert len(obj.indices)>0,f'No actual {segments} samples for{self.year}'
        return obj
    def train(self):
        self.batch_size,self.drop_last,self.shuffle=self.params;self.indices=self.loss_indices
    def eval(self):
        self.batch_size=-1;self.drop_last=False;self.shuffle=False;self.indices=self.all_indices
        if self.segment in ['valid','test']:self.clear_memory()
    def clear_memory(self):
        self.predictions.fill(np.nan);self.recorded.fill(False);self.record_kind.fill(0)
    def restore_index(self,index):
        if isinstance(index,torch.Tensor):index=index.detach().cpu().numpy()
        return self._index[np.asarray(index)]
    def record_predictions(self,index,predictions):
        if isinstance(index,torch.Tensor):index=index.detach().cpu().numpy()
        if isinstance(predictions,torch.Tensor):predictions=predictions.detach().cpu().numpy()
        index=np.asarray(index,dtype='int64');predictions=np.asarray(predictions,dtype='float32')
        assert predictions.shape==(len(index),self.num_states)
        self.predictions[index]=predictions;self.recorded[index]=np.isfinite(predictions).all(axis=1)
        self.record_kind[index]={'train':1,'valid':2,'test':3}[self.segment]
        self.feedback_audit['saved_expert_prediction_writes']+=len(index)
        if self.prediction_sink is not None:self.prediction_sink(self.segment,index,predictions.copy())
    def _batches(self):
        indexes=self.indices.copy()
        if self.batch_size<0:
            # Source author evaluation is daily, chronological and complete.
            dates=self.bundle.times[indexes]
            starts=np.r_[0,np.flatnonzero(dates[1:]!=dates[:-1])+1,len(indexes)]
            return [indexes[a:b] for a,b in zip(starts[:-1],starts[1:])]
        if self.shuffle:np.random.shuffle(indexes)
        return [indexes[i:i+self.batch_size] for i in range(0,len(indexes),self.batch_size) if not self.drop_last or i+self.batch_size<=len(indexes)]
    def __len__(self):return len(self._batches()) if self.batch_size<0 else (len(self.indices)+self.batch_size-1)//self.batch_size
    def batch_data(self,index):
        b=self.bundle;index=np.asarray(index,dtype='int64')
        data=np.zeros((len(index),60,6+self.num_states),dtype='float32')
        data[:,:,:6]=b.x[index]
        ids=np.asarray(b.history[index]);safe=np.maximum(ids,0)
        asof=b.times[index,None]+pd.Timedelta(hours=15).value
        available=b.available[safe]
        allowed=(ids>=0)&self.recorded[safe]&(available!=NAT)&(available<=asof)
        # Keep author recency exclusion, plus exact maturity regardless of horizon.
        allowed[:,-self.horizon:]=False
        rows,slots=np.nonzero(allowed);selected=ids[rows,slots]
        if len(selected):
            assert np.all(b.available[selected]!=NAT) and np.all(b.available[selected]<=asof[rows,0])
            # First read of labels for feedback occurs only after the above gate.
            if self.feedback_provider is None:target=b.y[selected]
            else:
                target=np.empty(len(selected),dtype='float32')
                cutoffs=asof[rows,0]
                for cutoff in np.unique(cutoffs):
                    subset=cutoffs==cutoff
                    mutated=self.feedback_provider(b.y,b.available,int(cutoff))
                    assert mutated.shape==b.y.shape
                    target[subset]=mutated[selected[subset]]
            finite=np.isfinite(target)
            self.feedback_audit['mature_missing_target_rows']=self.feedback_audit.get('mature_missing_target_rows',0)+int((~finite).sum())
            rows,slots,selected,target=rows[finite],slots[finite],selected[finite],target[finite]
            loss=(self.predictions[selected]-target[:,None])**2
            loss-=loss.min(axis=1,keepdims=True)
            data[rows,slots,6:]=loss
            self.feedback_audit['mature_label_reads']+=len(selected)
        return data
    def __iter__(self):
        for index in self._batches():
            data=self.batch_data(index)
            # Unknown Outer labels are not passed into original test metrics/loss.
            labels=np.full(len(index),np.nan,dtype='float32')
            if self.segment!='test':
                known=(self.bundle.available[index]!=NAT)&(self.bundle.available[index]<self.fit_cutoff)
                labels[known]=self.bundle.y[index[known]]
            yield {'data':torch.from_numpy(data),'label':torch.from_numpy(labels.copy()),'index':torch.from_numpy(index.copy())}

def causal_model_class():
    _,author=load_author()
    train=textwrap.dedent(inspect.getsource(author.TRAModel.train_epoch))
    test=textwrap.dedent(inspect.getsource(author.TRAModel.test_epoch))
    original_train=train;original_test=test
    old='data_set.assign_data(index, L)  # save loss to memory'
    assert train.count(old)==1
    train=train.replace(old,'data_set.record_predictions(index, all_preds.detach())  # actual-exit queue')
    old_test='''        L = (all_preds - label[:, None]).pow(2)

        L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input

        data_set.assign_data(index, L)  # save loss to memory'''
    assert test.count(old_test)==1,'locked author source changed'
    test=test.replace(old_test,'        data_set.record_predictions(index, all_preds)  # actual-exit queue; no unknown-label read')
    assert test.count('metrics.append(evaluate(pred))')==1
    test=test.replace('metrics.append(evaluate(pred))','metrics.append(evaluate(pred.loc[pred.label.notna()]))  # mature metric loss mask, all prediction nodes retained')
    namespace=dict(author.__dict__)
    exec(compile(train+'\n'+test,'<locked_TRA_exact_maturity_adapter>','exec'),namespace)
    original_forward=textwrap.dedent(inspect.getsource(author.LSTM.forward))
    forward=original_forward.replace('(rnn_out * scores).sum(dim=1).squeeze()', '(rnn_out * scores).sum(dim=1)')
    assert forward!=original_forward
    forward_namespace=dict(author.__dict__)
    exec(compile(forward,'<locked_TRA_singleton_shape_compatibility>','exec'),forward_namespace)
    class TargetLSTM(author.LSTM):
        pass
    TargetLSTM.forward=forward_namespace['forward']
    original_init=textwrap.dedent(inspect.getsource(author.TRAModel.__init__))
    init=original_init.replace('self.model = eval(model_type)(**model_config).to(device)', 'assert model_type == "LSTM", "Only the frozen author LSTM is permitted"\n    self.model = TargetLSTM(**model_config).to(device)')
    assert init!=original_init
    namespace['TargetLSTM']=TargetLSTM
    exec(compile(init,'<locked_TRA_data_cardinality_constructor>','exec'),namespace)
    def average_backbone(states,model):
        if all(not p.requires_grad for p in model.parameters()):
            first=states[0]
            assert all(all(torch.equal(first[k],state[k]) for k in first) for state in states),'Frozen backbone states differ before averaging'
            return first
        return author.average_params(states)
    original_fit=textwrap.dedent(inspect.getsource(author.TRAModel.fit))
    fit=original_fit.replace('self.model.load_state_dict(average_params(params_list["model"]))','self.model.load_state_dict(average_backbone(params_list["model"], self.model))')
    assert fit!=original_fit
    namespace['average_backbone']=average_backbone
    exec(compile(fit,'<locked_TRA_constant_frozen_backbone_identity>','exec'),namespace)
    class CausalTRAModel(author.TRAModel):
        pass
    CausalTRAModel.__init__=namespace['__init__'];CausalTRAModel.train_epoch=namespace['train_epoch'];CausalTRAModel.test_epoch=namespace['test_epoch'];CausalTRAModel.fit=namespace['fit']
    provenance={'original_source':str(Path(author.__file__)),'train_original_sha256':hashlib.sha256(original_train.encode()).hexdigest(),'test_original_sha256':hashlib.sha256(original_test.encode()).hexdigest(),'train_adapted_sha256':hashlib.sha256(train.encode()).hexdigest(),'test_adapted_sha256':hashlib.sha256(test.encode()).hexdigest(),'changes':['training memory assignment replaced by saved expert predictions','test error calculation and memory assignment replaced by saved expert predictions; Outer labels unavailable to loss/metrics'],'unchanged':['author LSTM','TRA router and stochastic Gumbel routing','MSE','Sinkhorn OT','optimizers','parameter averaging','fit and default early stopping'],'train_adapted_source':train,'test_adapted_source':test}
    provenance.update(singleton_compatibility='remove redundant squeeze after sum; retain [batch,hidden] for batch1; original reference untouched',lstm_original_sha256=hashlib.sha256(original_forward.encode()).hexdigest(),lstm_adapted_sha256=hashlib.sha256(forward.encode()).hexdigest(),constructor_original_sha256=hashlib.sha256(original_init.encode()).hexdigest(),constructor_adapted_sha256=hashlib.sha256(init.encode()).hexdigest(),lstm_adapted_source=forward,constructor_adapted_source=init)
    provenance['unchanged'].remove('parameter averaging')
    provenance['unchanged'].append('all trainable parameter averaging; constant frozen-backbone average uses exact identity compatibility')
    provenance['changes'].append('metrics mask unavailable labels while keeping every evaluation prediction node; train/validation/Outer queues isolated in the data adapter')
    provenance.update(fit_original_sha256=hashlib.sha256(original_fit.encode()).hexdigest(),fit_adapted_sha256=hashlib.sha256(fit.encode()).hexdigest(),fit_adapted_source=fit,frozen_backbone_compatibility='exact identity for a mean of tensor-identical frozen states; assertion forbids discarding distinct states; original public reference unmodified')
    return CausalTRAModel,provenance
