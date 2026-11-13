"""Replay original_ai071 PIT-ST qualification for all actual factor keys."""
import json
import pandas as pd
from core.stock_readonly import ensure_stock_src_on_path
from core.round26_1.market import SNAP,R,C,factor_candidates
from core.research_integrity_v5.cache import identity
from core.research_integrity_v5.resources import measured


def run():
    ensure_stock_src_on_path()
    from zyquant.data import DataSnapshot
    from nshape.universe import resolve_daily_members,eligibility_mask
    with measured(R/'candidate_PIT_membership_RESOURCES.json'):
        snapshot=DataSnapshot(SNAP)
        q=factor_candidates()
        treatment=snapshot.table('special_treatment',cutoff=snapshot.metadata.as_of_date)
        historical=pd.DataFrame({'universe_id':'CN_ALL_A','instrument_id':sorted(q.instrument_id.unique()),'effective_from':q.trade_date.min(),'effective_to':pd.NaT})
        members=resolve_daily_members(historical,treatment,q.trade_date.dt.date.unique(),universe_id='CN_A_EX_ST')
        mask=eligibility_mask(q.trade_date,q.instrument_id,members)
        assert mask.all(),f'{int((~mask).sum())} frozen candidates disagree with historical ST intervals'
        result={'status':'PASS_RECOMPUTED_PIT_ST_ALL_FACTOR_KEYS','rows':len(q),'stock_ids':q.instrument_id.nunique(),'future_labels_used_in_eligibility':False,'current_listing_list_used':False,'original_policy':'original_ai071 historical candidate-id universe minus interval-based daily special_treatment','candidate_source':identity(C/'result/round22_3/data/ten_candidate.parquet'),'authoritative_builder':identity('/data1/Stock/tools/filter_first_board_pit_universe.py'),'PIT_interval_resolver':identity('/data1/Stock/src/nshape/universe.py'),'audit_source':identity(__file__),'snapshot_manifest':identity(SNAP/'manifest.json'),'limitations':'Input file hashes cover entire files for cache binding. Eligibility reads no target columns. PIT interval replay does not prove unrecorded vendor revisions.'}
        (R/'candidate_PIT_membership_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('PASS PIT-ST eligibility:',len(q),'actual historical candidates, no future-label eligibility filter',flush=True)
    return result
if __name__=='__main__':run()
