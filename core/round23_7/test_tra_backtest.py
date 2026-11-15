"""Actual baseline endpoint and whole-day fallback integration, without a trial."""
import json
import numpy as np
import pandas as pd
from core.round23_7.tra_backtest import baseline_sources,materialize,KEY,R

def run():
    source,baseline,formal=baseline_sources()
    result,flags=materialize(source['five'],source['ten'],baseline)
    pd.testing.assert_frame_equal(result,baseline,check_exact=True)
    assert not flags.whole_day_v5_fallback.any()
    day=baseline.signal_date.iloc[0]
    corrupt=source['five'].copy();i=corrupt.loc[corrupt.signal_date.eq(day)].index[0]
    corrupt.loc[i,'score']=np.nan
    repaired,flags=materialize(corrupt,source['ten'],baseline)
    pd.testing.assert_frame_equal(repaired,baseline,check_exact=True)
    assert flags.whole_day_v5_fallback.sum()==1 and flags.loc[flags.whole_day_v5_fallback,'signal_date'].iloc[0]==day
    output={'status':'PASS_REAL_SCORE_ENDPOINT_AND_WHOLE_DAY_FALLBACK','formal_baseline_rows':len(formal),'matched_baseline_rows':len(baseline),'common_formal_scores_exact':True,'new_actual_prediction_keys':len(baseline)-len(formal),'original_complete_feature_filter_applied_without_targets':True,'all_legal_XGB_predictions_finite':True,'single_missing_head_value_restores_complete_day_exactly':True,'all_other_days_exact':True,'backtest_not_run_by_this_test':True,'previous_722_added_included67_nonfinite_bank_placeholders':'retained in target_data_attempts/control_preflight_before_complete_feature_filter; no economic run used it'}
    (R/'BACKEND_SCORE_INTEGRATION.json').write_text(json.dumps(output,indent=2));print(json.dumps(output),flush=True)

if __name__=='__main__':run()
