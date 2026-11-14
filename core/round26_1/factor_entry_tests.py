"""Counterfactuals through the actual nested-factor XGB fit/predict entry.

Run only after a real generated OOF bank exists, with no PPO learner active.
Each intervention gets a fresh cache namespace, so parity compares actual fits.
"""
from pathlib import Path
from unittest.mock import patch
import json,time
import numpy as np
import pandas as pd
from core.round26_1.market import C,R,KEY
from core.round26_1 import factor_models as fm
from core.round7.contract import FIVE_HEAD,TEN_HEAD
from core.research_integrity_v5.cache import identity


def run(factor, arm, seed_label):
    tag=f'{arm}_{seed_label}'
    dest=R/'head_entry_tests'/tag/str(time.time_ns())
    dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    original_read=pd.read_parquet
    original_contract=fm.full_contract
    for head in [FIVE_HEAD,TEN_HEAD]:
        for year in [2023,2024,2026]:
            cutoff=pd.Timestamp(f'{year}-01-01')
            baseline=None
            for case in ['original','future_labels_deleted','future_labels_permuted','immature_labels_permuted','future_features_changed']:
                mutation_counts={'training_rows':0,'prediction_rows':0}
                def reader(path,*args,**kwargs):
                    frame=original_read(path,*args,**kwargs)
                    name=Path(path).name
                    if name not in [f'{head.name}_training.parquet',f'{head.name}_candidate.parquet']:
                        return frame
                    frame=frame.copy();before_intervention=frame.copy()
                    dates=pd.to_datetime(frame.trade_date)
                    rng=np.random.default_rng(231726)
                    if name.endswith('_candidate.parquet'):
                        # The real predictor must not ask this reader for target/maturity.
                        assert head.target not in frame.columns and head.label_end_column not in frame.columns
                    elif case in ['future_labels_deleted','future_labels_permuted']:
                        mask=dates.ge(cutoff)
                        mutation_counts['training_rows']=int(mask.sum())
                        if case=='future_labels_deleted':
                            frame.loc[mask,head.target]=np.nan
                            frame.loc[mask,head.label_end_column]=pd.NaT
                        else:
                            frame.loc[mask,head.target]=rng.permutation(frame.loc[mask,head.target].to_numpy())
                    elif case=='immature_labels_permuted':
                        mask=dates.lt(cutoff)&(pd.to_datetime(frame[head.label_end_column])+pd.Timedelta(hours=15)).ge(cutoff)
                        mutation_counts['training_rows']=int(mask.sum())
                        prior=frame.loc[mask,head.target].to_numpy();corrupt=rng.permutation(prior)
                        if len(prior) and np.array_equal(prior,corrupt,equal_nan=True):corrupt=np.nan_to_num(prior,nan=0.)+1e6
                        frame.loc[mask,head.target]=corrupt
                    if case=='future_features_changed':
                        mask=dates.ge(pd.Timestamp(f'{year+1}-01-01'))
                        mutation_counts['training_rows' if name.endswith('_training.parquet') else 'prediction_rows']=int(mask.sum())
                        frame.loc[mask,list(head.features)]=rng.normal(1e6,1e4,size=(int(mask.sum()),len(head.features)))
                    changed=~(frame.eq(before_intervention)|(frame.isna()&before_intervention.isna())).all(axis=1)
                    mutation_counts['training_rows' if name.endswith('_training.parquet') else 'prediction_rows']=int(changed.sum())
                    return frame
                def diagnostic_contract(parameters,inputs,*args,**kwargs):
                    # In-memory raw-panel interventions are explicit derivation parameters,
                    # rather than pretending the unchanged source files have new bytes.
                    parameters['diagnostic_context']={'case':case,'intervention_seed':231726,'source':identity(__file__),'actual_changed_rows':dict(mutation_counts),'economic_trial':False}
                    return original_contract(parameters,inputs,*args,**kwargs)
                # A separate immutable directory forces the production fit loop to run.
                with patch.object(pd,'read_parquet',reader),patch.object(fm,'full_contract',diagnostic_contract):
                    a,p=fm.prepare(head,year,factor)
                    frame,path,fp,cached=fm.fit_head(arm,seed_label,head,year,factor,'ADD',cache_namespace=dest/head.name/str(year)/case)
                assert not cached,'Counterfactual must be an actual cold fit'
                if baseline is None:
                    baseline=(a.copy(),p.copy(),frame.copy())
                else:
                    pd.testing.assert_frame_equal(a,baseline[0],check_exact=True)
                    pd.testing.assert_frame_equal(p,baseline[1],check_exact=True)
                    pd.testing.assert_frame_equal(frame,baseline[2],check_exact=True)
                rows.append({'arm':arm,'seed':str(seed_label),'head':head.name,'outer_year':year,'case':case,'perturbed_training_rows':mutation_counts['training_rows'],'perturbed_prediction_rows':mutation_counts['prediction_rows'],'perturbation_scope_status':('ORIGINAL_CONTROL' if case=='original' else 'NONEMPTY_INTERVENTION' if sum(mutation_counts.values()) else 'NOT_EVALUATED_NO_FUTURE_ROWS'),'cold_fit':True,'five_tree_models':5,'train_rows':len(a),'prediction_rows':len(p),'input_and_scores_exact':True,'fit_cutoff':str(cutoff),'max_training_label_available_at':str(a.label_available_at.max()),'model_directory':str(path),'fingerprint':fp})
    pd.DataFrame(rows).to_csv(dest/'actual_entry_counterfactuals.csv',index=False)
    result={'status':'PASS_ACTUAL_XGB_ADD_ENTRY','source_factor':identity(factor),'tested_heads':['five','ten'],'folds':[2023,2024,2026],'actual_fit_configs':len(rows),'actual_tree_models':len(rows)*5,'future_label_deletion_permutation':True,'unmature_label_permutation':True,'future_feature_prefix_invariance':'nonempty folds only; latest2026 has no2027+ data and is NOT_EVALUATED_NO_FUTURE_ROWS','counterfactual_PPO_retraining':'NOT_RUN; generation input tensor invariance tested separately','scope':'Nested OOF factor held fixed; complete real XGB ADD fit/predict entry. Not a full PPO counterfactual retraining proof.'}
    (dest/'STATUS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
