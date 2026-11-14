"""Actual no-factor learner replay before attributing any Add improvement."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from core.round26_1.market import R,KEY
from core.research_integrity_v5.cache import identity


def replay(source,factor_path=None):
    from core.round7.contract import FIVE_HEAD,TEN_HEAD
    from core.round26_1.factor_models import fit_head,canonical_fuse
    root=R/'baseline_head_replay';root.mkdir(parents=True,exist_ok=True)
    matched={};records=[];head_predictions={}
    for head in [FIVE_HEAD,TEN_HEAD]:
        parts=[]
        for year in range(2023,2027):
            p,path,fp,hit=fit_head('NO_FACTOR','CONTROL',head,year,factor_path,'BASELINE')
            config=json.loads((path/'MODEL_REGISTRY.json').read_text())
            assert config['parameters']['feature_order']==list(head.features)
            assert pd.Timestamp(config['parameters']['max_training_label_available_at'])<pd.Timestamp(f'{year}-01-01')
            parts.append(p.rename(columns={'trade_date':'signal_date'}))
            records.append({'head':head.name,'year':year,'mode':'NO_FACTOR_MATCHED_CONTROL','factor_input_to_learner':False,'actual_features':list(head.features),'path':str(path),'fingerprint':fp,'cache_hit':hit,'actual_tree_files':len(list(path.glob('xgb_*.json')))})
        joined=pd.concat(parts,ignore_index=True);head_predictions[head.name]=joined[['signal_date','instrument_id','score']]
        joined.to_parquet(root/f'{head.name}_control_predictions.parquet',index=False)
        old=source[head.name]
        matched[head.name]=pd.concat([old.loc[~pd.to_datetime(old.signal_date).dt.year.between(2023,2026)],joined],ignore_index=True)
    old=canonical_fuse(source['five'],source['ten']);new=canonical_fuse(matched['five'],matched['ten'])
    old=old.loc[old.signal_date.ge('2023-01-03')].sort_values(['signal_date','instrument_id']).reset_index(drop=True)
    new=new.loc[new.signal_date.ge('2023-01-03')].sort_values(['signal_date','instrument_id']).reset_index(drop=True)
    same=old.equals(new)
    comparison=old.merge(new,on=['signal_date','instrument_id'],how='outer',suffixes=('_legacy','_matched'),indicator=True,validate='one_to_one')
    comparison['different_score_or_key']=comparison['_merge'].ne('both')|comparison.score_legacy.ne(comparison.score_matched)
    comparison.to_parquet(root/'legacy_vs_matched_fused_predictions.parquet',index=False)
    legal=new[['signal_date','instrument_id']]
    conditional=legal
    for head in ['five','ten']:
        conditional=conditional.merge(head_predictions[head].rename(columns={'score':head}),on=['signal_date','instrument_id'],validate='one_to_one')
    conditional.to_parquet(root/'matched_legal_head_scores.parquet',index=False)
    pd.DataFrame(records).to_csv(root/'control_model_manifest.csv',index=False)
    proof={'status':'EXACT_FUSED_ENDPOINT_PARITY' if same else 'MATCHED_REPAIRED_NO_FACTOR_BASELINE_REQUIRED','original_v5_is_unchanged':True,'old_window_prediction_rows':len(old),'matched_window_prediction_rows':len(new),'different_score_or_key_rows':int(comparison.different_score_or_key.sum()),'same_original_feature_order_and_parameters':True,'no_z_AG_input':True,'head_configurations':len(records),'actual_tree_files_current_configs':sum(v['actual_tree_files'] for v in records),'primary_comparison':'matched no-factor learner replay; old original window baseline retained in BASELINE_PREFLIGHT stage','factor_input_to_baseline':False,'no_placeholder_factor_created':True,'optional_factor_manifest':identity(factor_path) if factor_path is not None else None,'no_decision_based_on_candidate_economic_results':True}
    (root/'STATUS.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2))
    return matched,proof
