"""Frozen paired MASTER arms: all Add arms and triggered Remove/Replace, all costs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from core.round26_1.master_provider import C,R,KEY
from core.round26_1.master_backend import fit_head,prepare
from core.round26_1.factor_models import canonical_fuse
from core.round26_1.master_native import execute
from core.research_integrity_v5.cache import identity

def baseline_sources():
    records=pd.read_csv(R/'baseline_control/head_models.csv')
    assert len(records)==18 and set(records.year)==set(range(2018,2027))
    source={}
    for head in ['five','ten']:
        parts=[]
        for row in records.loc[records['head'].eq(head)].sort_values('year').itertuples(index=False):
            p=pd.read_parquet(Path(row.path)/'predictions.parquet');parts.append(p.rename(columns={'trade_date':'signal_date'}))
        source[head]=pd.concat(parts,ignore_index=True)[['signal_date','instrument_id','score']]
    scores=canonical_fuse(source['five'],source['ten'])
    formal=pd.read_parquet(C/'result/round23_6/frozen_components.parquet',columns=['signal_date','instrument_id','v5']).rename(columns={'v5':'score'})
    comparison=formal.merge(scores,on=['signal_date','instrument_id'],how='outer',suffixes=('_formal','_matched'),indicator=True,validate='one_to_one')
    common=comparison._merge.eq('both')
    proof={'formal_rows':len(formal),'matched_rows':len(scores),'common_rows':int(common.sum()),'new_rows':int(comparison._merge.eq('right_only').sum()),'missing_formal_rows':int(comparison._merge.eq('left_only').sum()),'common_exact_score':bool(np.array_equal(comparison.loc[common,'score_formal'].to_numpy(),comparison.loc[common,'score_matched'].to_numpy())),'matched_no_factor_replay':True,'formal_v5_unchanged':True}
    out=R/'baseline_control';comparison.to_parquet(out/'formal_vs_matched_scores.parquet',index=False);scores.to_parquet(out/'scores.parquet',index=False)
    (out/'COVERAGE_AUDIT.json').write_text(json.dumps(proof,indent=2))
    return source,scores

def run(seed,bankpaths):
    from core.round7.contract import FIVE_HEAD,TEN_HEAD
    source,base=baseline_sources();records=[];models=[];rho_records=[]
    dest=R/'seed_stages'/str(seed);dest.mkdir(parents=True,exist_ok=True)
    for bps in [0,10,20]:
        code='MASTER_A0'+('' if not bps else '_S'+str(bps//10));path,fp,hit=execute(code,base,bps)
        records.append({'method':'MASTER_A0','scenario':'S'+str(bps//10),'code':code,'path':str(path),'fingerprint':fp,'cache_hit':hit})
    for arm in ['M0','M1']:
        bank=bankpaths[arm];predictions={};removed={};replaced={};triggered={}
        for head in [FIVE_HEAD,TEN_HEAD]:
            add=[];rem=[];rep=[];triggered[head.name]=False
            for year in range(2018,2027):
                a,p=prepare(head,year,bank)
                rho=a[[*head.features,'z_MASTER']].corr(method='spearman').z_MASTER.drop('z_MASTER').dropna()
                column=rho.abs().idxmax() if len(rho) else None
                value=float(rho[column]) if column else None;high=column is not None and abs(value)>=.6
                rho_records.append({'arm':arm,'seed':seed,'head':head.name,'year':year,'removed_feature':column,'rho':value,'triggered':high,'train_rows':len(a),'scope':'training only prior3y mature baseline-complete rows','threshold':.6})
                modes=['ADD']+(['REMOVE','REPLACE'] if high else [])
                for mode in modes:
                    path=fit_head(arm,seed,head,year,bank,mode,column if mode!='ADD' else None)
                    rows=pd.read_parquet(path/'predictions.parquet').rename(columns={'trade_date':'signal_date'})
                    {'ADD':add,'REMOVE':rem,'REPLACE':rep}[mode].append(rows)
                    models.append({'arm':arm,'seed':seed,'head':head.name,'year':year,'mode':mode,'path':str(path)})
                    print('MASTER actual XGB entry',arm,seed,head.name,year,mode,flush=True)
                if high:triggered[head.name]=True
                else:
                    fallback=source[head.name].loc[source[head.name].signal_date.dt.year.eq(year)]
                    rem.append(fallback);rep.append(fallback)
            predictions[head.name]=pd.concat(add,ignore_index=True)
            removed[head.name]=pd.concat(rem,ignore_index=True);replaced[head.name]=pd.concat(rep,ignore_index=True)
        variants={'FIVE_ADD':{'five':predictions['five']},'TEN_ADD':{'ten':predictions['ten']},'BOTH_ADD':predictions}
        for head in ['five','ten']:
            if triggered[head]:
                variants['REMOVE_'+head.upper()]={head:removed[head]};variants['REPLACE_'+head.upper()]={head:replaced[head]}
        for variant,override in variants.items():
            five=override.get('five',source['five']);ten=override.get('ten',source['ten'])
            scores=canonical_fuse(five,ten)
            method=f'{arm}_{seed}_{variant}';folder=R/'scores';folder.mkdir(exist_ok=True);scores.to_parquet(folder/(method+'.parquet'),index=False)
            assert set(map(tuple,scores[['signal_date','instrument_id']].to_numpy()))==set(map(tuple,base[['signal_date','instrument_id']].to_numpy())),'Paired equal keys required'
            for bps in [0,10,20]:
                code=method+('' if not bps else '_S'+str(bps//10));path,fp,hit=execute(code,scores,bps,upstream=[identity(bank)])
                records.append({'method':method,'scenario':'S'+str(bps//10),'code':code,'path':str(path),'fingerprint':fp,'cache_hit':hit})
                pd.DataFrame(records).to_csv(dest/'run_manifest.csv',index=False)
                print('MASTER actual native account',code,flush=True)
        pd.DataFrame(models).to_csv(dest/'head_model_manifest.csv',index=False)
        pd.DataFrame(rho_records).to_csv(dest/'training_only_dedup.csv',index=False)
    for file,keys in [('run_manifest.csv',['code']),('head_model_manifest.csv',['arm','seed','head','year','mode']),('training_only_dedup.csv',['arm','seed','head','year'])]:
        frames=[pd.read_csv(p) for p in (R/'seed_stages').glob('*/'+file)]
        pd.concat(frames,ignore_index=True).drop_duplicates(keys,keep='last').to_csv(R/file,index=False)
    return records
