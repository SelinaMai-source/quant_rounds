"""Author pickle/Qlib-version adapter: retain history, honor declared segment bounds."""
from pathlib import Path
import pickle,json
import numpy as np
import pandas as pd
from core.round26_1.master_provider import B,R
from core.research_integrity_v5.cache import identity
ROOT=B/'MASTER_public/extracted/opensource'

class AuthorView:
    def __init__(self,base):
        self.base=base
        index=base.get_index();dates=pd.DatetimeIndex(index.get_level_values('datetime'))
        assert dates.is_monotonic_increasing,'Original daily sampler requires date-major order'
        self.selected=np.flatnonzero((dates>=pd.Timestamp(base.start))&(dates<=pd.Timestamp(base.end)))
        self.index=index[self.selected]
        assert np.array_equal(self.selected,np.arange(int(base.start_idx),int(base.end_idx))), 'Actual pickle declared bounds disagree with date selection'
        assert base.step_len==8 and base.data_arr.shape[1]==222
        count=int(np.isfinite(base.idx_df.loc[pd.Timestamp(base.start):pd.Timestamp(base.end)].to_numpy(dtype=float)).sum())
        assert count==len(self.index),'Do not drop any available idx_df nodes by labels'
    def __len__(self):return len(self.selected)
    def get_index(self):return self.index
    def __getitem__(self,index):return self.base[self.selected[index]]
    def labels(self):
        coordinates=self.base.idx_map[self.selected]
        rows=self.base.idx_arr[coordinates[:,0],coordinates[:,1]].astype('int64')
        labels=self.base.data_arr[rows,-1]
        checks=np.unique(np.array([0,len(self)//2,len(self)-1]))
        observed=self[checks][:,-1,-1]
        assert np.array_equal(labels[checks],observed,equal_nan=True)
        return labels

def load(phase):
    path=ROOT/f'csi300_dl_{phase}.pkl'
    with path.open('rb') as f:base=pickle.load(f)
    return AuthorView(base)

def audit():
    records=[]
    previous_end=None
    for phase in ['train','valid','test']:
        view=load(phase);base=view.base;dates=view.get_index().get_level_values('datetime')
        if previous_end is not None:assert previous_end<dates.min(),'No prediction-segment overlap'
        previous_end=dates.max()
        labels=view.labels()
        records.append({'phase':phase,'source':identity(ROOT/f'csi300_dl_{phase}.pkl'),'declared_start':str(base.start),'declared_end':str(base.end),'prediction_start':str(dates.min()),'prediction_end':str(dates.max()),'prediction_rows':len(view),'prediction_dates':int(dates.nunique()),'retained_history_rows':len(base.data_arr)-1,'warmup_rows_excluded_from_prediction_but_retained_in_history':len(base)-len(view),'all_available_idx_df_nodes_retained':True,'missing_labels_retained':int(np.isnan(labels).sum()),'features':221,'sequence':8,'author_fillna_type':base.fillna_type,'original_sampler_start_end_indices_reproduced':True,'PIT_first_publication_vintage_proven':False})
        del view,base
    result={'status':'PASS_ACTUAL_ALL_PROVIDED_NODES_AND_DECLARED_SPLITS','cases':records,'compatibility':'Old pickles carry unsliced idx_map/data_index plus start_idx/end_idx. Installed Qlib0.9.8 methods expect presliced arrays. View maps indices to original declared interval and leaves all preceding feature history intact. No label filtering or guessed300-node insertion.','scope':'All available nodes in actual author arrays. Raw upstream universe completeness/vintage is not inferred merely from dump size.'}
    (R/'PUBLIC_CORRECTED_SAMPLER_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True);return result
if __name__=='__main__':audit()
