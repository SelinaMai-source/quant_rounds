"""Raw factor and actual held-out XGB Add conditional evidence, no in-sample meta fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from core.round26_1.master_provider import C,R,KEY

def corr(frame,score,target,method):
    finite=frame.loc[np.isfinite(frame[score])&np.isfinite(frame[target])]
    return float(finite[score].corr(finite[target],method=method)) if len(finite)>=2 else np.nan

def run():
    from core.round7.contract import FIVE_HEAD,TEN_HEAD
    daily=[];coverage=[];shapes=[];annual=[];conditional=[]
    maturity_end=pd.Timestamp('2026-07-24 23:59:59')
    banks=sorted((R/'factor_banks').glob('*/*/OOF_factor.parquet'))
    assert banks,'No actual learned OOF factors'
    for path in banks:
        arm=path.parents[1].name;seed=int(path.parent.name);bank=pd.read_parquet(path)
        assert bank.trade_date.dt.year.between(2015,2026).all()
        assert (pd.to_datetime(bank.factor_fit_cutoff)==pd.to_datetime(bank.trade_date.dt.year.astype(str)+'-01-01')).all()
        for head in [FIVE_HEAD,TEN_HEAD]:
            p=pd.read_parquet(C/f'result/round22_3/data/{head.name}_candidate.parquet',columns=list(dict.fromkeys(KEY+list(head.features)+['is_first_board_hard','universe_eligible',head.target,head.label_end_column])))
            p.trade_date=pd.to_datetime(p.trade_date)
            p=p.loc[p.trade_date.ge('2015-01-01')&p.is_first_board_hard.fillna(False)&p.universe_eligible.fillna(False)].copy()
            p=p.merge(bank[KEY+['z_MASTER']],on=KEY,how='left',validate='one_to_one')
            legal=(pd.to_datetime(p[head.label_end_column])+pd.Timedelta(hours=15)).le(maturity_end)
            p['evidence_label']=p[head.target].where(legal)
            p['legacy_backend_domain']=p[list(head.features)].replace([np.inf,-np.inf],np.nan).notna().all(axis=1)&p.d1_open_gap.ge(-.01)
            assert np.isfinite(p.z_MASTER).all(),'Missing D-known target inference nodes'
            for domain,select in [('D_KNOWN_TARGET_DOMAIN',pd.Series(True,index=p.index)),('LEGACY_BACKEND_DOMAIN',p.legacy_backend_domain)]:
                frame=p.loc[select]
                coverage.append({'arm':arm,'seed':seed,'head':head.name,'domain':domain,'rows':len(frame),'missing_factor':int(frame.z_MASTER.isna().sum()),'nonfinite_factor':int((~np.isfinite(frame.z_MASTER)).sum()),'mature_metric_labels':int(frame.evidence_label.notna().sum()),'min':float(frame.z_MASTER.min()),'max':float(frame.z_MASTER.max()),'p001':float(frame.z_MASTER.quantile(.001)),'p999':float(frame.z_MASTER.quantile(.999)),'stock_selection_uses_label':False})
                rows=[]
                for date,group in frame.groupby('trade_date'):
                    row={'arm':arm,'seed':seed,'head':head.name,'domain':domain,'trade_date':date,'PearsonIC':corr(group,'z_MASTER','evidence_label','pearson'),'RankIC':corr(group,'z_MASTER','evidence_label','spearman'),'known_nodes':len(group),'mature_labels':int(group.evidence_label.notna().sum()),'scope':'WARMUP_TRUE_OOF' if date.year<2018 else 'PRIMARY_DEVELOPMENT'}
                    rows.append(row)
                    bins=np.ceil(group.z_MASTER.rank(method='average',pct=True)*5).clip(1,5)
                    for bucket,b in group.groupby(bins):
                        shapes.append({'arm':arm,'seed':seed,'head':head.name,'domain':domain,'trade_date':date,'quintile':int(bucket),'known_nodes':len(b),'mature_labels':int(b.evidence_label.notna().sum()),'mean_net_target':float(b.evidence_label.mean()),'target_median':float(b.evidence_label.median())})
                values=pd.DataFrame(rows).sort_values('trade_date')
                for window in [60,120,252]:
                    for name in ['PearsonIC','RankIC']:
                        values[f'rolling{window}_{name}']=values[name].rolling(window,min_periods=(window+1)//2).mean()
                daily.append(values)
                for year,group in values.groupby(values.trade_date.dt.year):
                    annual.append({'arm':arm,'seed':seed,'head':head.name,'domain':domain,'year':year,'dates':len(group),'valid_IC_dates':int(group.RankIC.notna().sum()),'PearsonIC':float(group.PearsonIC.mean()),'RankIC':float(group.RankIC.mean()),'RankICIR':float(group.RankIC.mean()/group.RankIC.std()) if group.RankIC.std()>0 else None})
    models=pd.read_csv(R/'head_model_manifest.csv');base=pd.read_csv(R/'baseline_control/head_models.csv')
    for row in models.loc[models['mode'].eq('ADD')].itertuples(index=False):
        head=FIVE_HEAD if row.head=='five' else TEN_HEAD
        matched=base.loc[base['head'].eq(row.head)&base.year.eq(row.year)];assert len(matched)==1
        old=pd.read_parquet(Path(matched.iloc[0].path)/'predictions.parquet')[KEY+['score']].rename(columns={'score':'baseline_score'})
        new=pd.read_parquet(Path(row.path)/'predictions.parquet')[KEY+['score']].rename(columns={'score':'ADD_score'})
        x=old.merge(new,on=KEY,validate='one_to_one');assert len(x)==len(old)==len(new)
        labels=pd.read_parquet(C/f'result/round22_3/data/{head.name}_candidate.parquet',columns=KEY+[head.target,head.label_end_column,'d1_open_gap'])
        x=x.merge(labels,on=KEY,validate='one_to_one');x=x.loc[x.d1_open_gap.ge(-.01)].copy()
        x['evidence_label']=x[head.target].where((pd.to_datetime(x[head.label_end_column])+pd.Timedelta(hours=15)).le(maturity_end))
        for date,g in x.groupby('trade_date'):
            oldic=corr(g,'baseline_score','evidence_label','spearman');newic=corr(g,'ADD_score','evidence_label','spearman')
            conditional.append({'arm':row.arm,'seed':row.seed,'head':row.head,'year':row.year,'trade_date':date,'baseline_head_RankIC':oldic,'ADD_head_RankIC':newic,'delta_RankIC':newic-oldic,'known_qualified_nodes':len(g),'mature_metric_labels':int(g.evidence_label.notna().sum()),'control':'same original features/learner vs plus one true annual nested-OOF MASTER factor; both actual heldout-year predictions','no_in_sample_NN_or_XGB_values':True})
    pd.concat(daily,ignore_index=True).to_csv(R/'factor_daily_and_rolling_IC.csv.gz',index=False)
    pd.DataFrame(coverage).to_csv(R/'factor_coverage_and_outliers.csv',index=False)
    pd.DataFrame(shapes).to_csv(R/'factor_quintile_shapes.csv.gz',index=False)
    pd.DataFrame(annual).to_csv(R/'factor_annual_IC.csv',index=False)
    pd.DataFrame(conditional).to_csv(R/'conditional_true_OOF_head_increment.csv.gz',index=False)
    (R/'FACTOR_EVIDENCE_SCOPE.json').write_text(json.dumps({'status':'REAL_OOF_FACTOR_EVIDENCE_COMPLETE_FOR_AVAILABLE_BANKS','banks':len(banks),'raw_domains':['D_KNOWN_TARGET_DOMAIN','LEGACY_BACKEND_DOMAIN'],'primary':'2018..2026 development','warmup':'2015..2017 genuinely heldout annual NN factors, separately labelled','conditional':'2018..2026 actual same-factory no-factor XGB vs real Add XGB heldout predictions, not fitted-row regressions','label_maturity_for_metrics_only':'2026-07-24 exchange23:59:59','prediction_node_selection_reads_future_labels':False},indent=2))
