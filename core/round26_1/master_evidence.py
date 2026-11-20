"""Actual MASTER account risk, cross-stock paired inference and source-linked figures."""
from pathlib import Path
import json,os
import numpy as np
import pandas as pd
from core.round26_1.master_provider import C,B,R
from core.research_integrity_v5.evaluation import evaluate,sharpe
from core.research_integrity_v5.economic import decisions
from core.research_integrity_v5.cache import identity

def matched_bones():
    rows=[]
    for seed in [17,42,73]:
        manifest=R/f'FIT_MANIFEST_{seed}.csv'
        if not manifest.exists():continue
        frame=pd.read_csv(manifest)
        for year in range(2015,2027):
            pair={}
            for arm in ['M0','M1']:
                selected=frame.loc[frame.arm.eq(arm)&frame.year.eq(year)];assert len(selected)==1
                p=Path(selected.iloc[0].path);pair[arm]=(p,json.loads((p/'contract.json').read_text()),json.loads((p/'MODEL_REGISTRY.json').read_text()))
            p0,c0,s0=pair['M0'];p1,c1,s1=pair['M1']
            for field in ['code','third_party','environment','data','features','universe','calibration','label','maturity','windows','execution','upstream','seed']:
                assert c0[field]==c1[field],f'Unmatched M0/M1 {seed}/{year}/{field}'
            x,y=dict(c0['parameters']),dict(c1['parameters']);x.pop('arm');y.pop('arm');assert x==y
            assert s0['normalizer_sha256']==s1['normalizer_sha256']
            for field in ['epochs','all_training_dates','gradient_dates','fit_cutoff','max_training_label_available_at','model_shape_parameters']:
                assert s0[field]==s1[field]
            a=pd.read_csv(p0/'TRAINING_DAYS.csv');b=pd.read_csv(p1/'TRAINING_DAYS.csv')
            columns=['date','step','nodes','legal','used','reason'];pd.testing.assert_frame_equal(a[columns],b[columns],check_exact=True)
            a=pd.read_parquet(p0/'factor_predictions.parquet');b=pd.read_parquet(p1/'factor_predictions.parquet')
            pd.testing.assert_frame_equal(a[['trade_date','instrument_id']],b[['trade_date','instrument_id']],check_exact=True)
            rows.append({'seed':seed,'year':year,'status':'PASS_SAME_BONE_INPUT_PROCESSOR_BUDGET_AND_TRAIN_SCHEDULE','normalizer_sha256':s0['normalizer_sha256'],'training_dates':s0['all_training_dates'],'gradient_dates':s0['gradient_dates'],'heldout_target_rows':len(a),'M0_model_sha256':s0['checkpoint_sha256'],'M1_model_sha256':s1['checkpoint_sha256']})
    pd.DataFrame(rows).to_csv(R/'M0_M1_MATCHED_BONE_AUDIT.csv',index=False);return rows

def paired_relation():
    curves=pd.read_csv(R/'daily_returns.csv.gz',parse_dates=['trade_date']).set_index('trade_date')
    results=pd.read_csv(R/'all_results.csv').set_index('experiment_id')
    registry=pd.read_csv(R/'run_manifest.csv').set_index('code')
    first=pd.read_parquet(Path(registry.iloc[0].path)/'nav.parquet');dates=pd.to_datetime(first.trade_date)
    years=(dates.iloc[-1]-dates.iloc[0]).days/365.25
    rng=np.random.default_rng(17);rows=[];point=[]
    for seed in [17,42,73]:
        for scenario in ['S0','S1','S2']:
            suffix='' if scenario=='S0' else '_'+scenario
            left=f'M0_{seed}_BOTH_ADD'+suffix;right=f'M1_{seed}_BOTH_ADD'+suffix
            if left not in curves or right not in curves:continue
            a=curves[left].to_numpy();b=curves[right].to_numpy();assert np.isfinite(a).all() and np.isfinite(b).all()
            outside=~curves.index.year.isin([2024,2025])
            record={'seed':seed,'scenario':scenario,'M0':left,'M1':right,'delta_CAGR':float(results.loc[right,'CAGR']-results.loc[left,'CAGR']),'delta_Sharpe':float(results.loc[right,'Sharpe']-results.loc[left,'Sharpe']),'delta_MaxDD':float(results.loc[right,'MaxDD']-results.loc[left,'MaxDD']),'outside2024_2025_log_relative_wealth':float((np.log1p(b[outside])-np.log1p(a[outside])).sum()),'same_window_and_cost':True,'selected_best_seed':False};point.append(record)
            for block,count in [(20,5000),(10,1000),(40,1000)]:
                sampled=[]
                for start in range(0,count,100):
                    size=min(100,count-start);beg=rng.integers(0,len(a),size=(size,int(np.ceil(len(a)/block))))
                    index=((beg[:,:,None]+np.arange(block))%len(a)).reshape(size,-1)[:,:len(a)]
                    ar,br=a[index],b[index];ap=np.cumprod(1+ar,axis=1);bp=np.cumprod(1+br,axis=1)
                    ad=ap/np.maximum.accumulate(np.column_stack([np.ones(size),ap]),axis=1)[:,1:]-1
                    bd=bp/np.maximum.accumulate(np.column_stack([np.ones(size),bp]),axis=1)[:,1:]-1
                    sampled.append(np.column_stack([np.expm1(np.log1p(br).sum(1)/years)-np.expm1(np.log1p(ar).sum(1)/years),sharpe(br,axis=1)-sharpe(ar,axis=1),bd.min(1)-ad.min(1)]))
                sample=np.concatenate(sampled)
                rows.append({'seed':seed,'scenario':scenario,'M0':left,'M1':right,'block':block,'draws':count,'paired_indices':True,'delta_CAGR_q05':float(np.quantile(sample[:,0],.05)),'delta_CAGR_q50':float(np.quantile(sample[:,0],.5)),'delta_CAGR_q95':float(np.quantile(sample[:,0],.95)),'delta_Sharpe_q05':float(np.quantile(sample[:,1],.05)),'delta_Sharpe_q50':float(np.quantile(sample[:,1],.5)),'delta_Sharpe_q95':float(np.quantile(sample[:,1],.95)),'delta_MaxDD_q05_path_sensitivity_only':float(np.quantile(sample[:,2],.05)),'delta_MaxDD_q95_path_sensitivity_only':float(np.quantile(sample[:,2],.95)),'calendar_years':years})
    pd.DataFrame(point).to_csv(R/'M1_vs_M0_primary_relation.csv',index=False)
    pd.DataFrame(rows).to_csv(R/'M1_vs_M0_paired_bootstrap.csv',index=False);return point

def plots():
    os.environ['MPLCONFIGDIR']=str(B/'mpl_master')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    runs=pd.read_csv(R/'run_manifest.csv').set_index('code');folder=R/'plots';folder.mkdir(exist_ok=True);records=[]
    for seed in [17,42,73]:
        codes=['MASTER_A0',f'M0_{seed}_BOTH_ADD',f'M1_{seed}_BOTH_ADD']
        if any(code not in runs.index for code in codes):continue
        fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        for code in codes:
            path=Path(runs.loc[code,'path']);nav=pd.read_parquet(path/'nav.parquet');dates=pd.DatetimeIndex(pd.to_datetime(nav.trade_date))
            wealth=np.r_[1.,nav.total_asset.to_numpy(float)/300000.];x=dates.insert(0,dates[0]-pd.Timedelta(days=1))
            dd=wealth/np.maximum.accumulate(wealth)-1
            line=axes[0].plot(x,wealth,label=code)[0];down=axes[1].plot(x,dd,label=code)[0]
            assert line.get_ydata()[0]==1 and np.array_equal(line.get_ydata()[1:],nav.total_asset.to_numpy(float)/300000.)
            assert np.array_equal(down.get_ydata(),dd)
            records.append({'seed':seed,'experiment_id':code,'nav_file':identity(path/'nav.parquet'),'initial_wealth':float(line.get_ydata()[0]),'actual_NAV_points':len(nav),'all_wealth_values_match_actual_NAV':True,'all_drawdown_values_match_actual_NAV':True,'plotted_MaxDD':float(dd.min())})
        axes[0].set_ylabel('Wealth / initial capital');axes[0].legend();axes[0].grid(alpha=.25)
        axes[1].set_ylabel('Drawdown');axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].grid(alpha=.25)
        fig.suptitle(f'MASTER fixed seed {seed}: matched native account, S0');fig.tight_layout();fig.savefig(folder/f'primary_seed{seed}_wealth1_drawdown.png',dpi=150);plt.close(fig)
    (R/'PLOT_SOURCE_AUDIT.json').write_text(json.dumps({'status':'PASS' if records else 'NO_COMPLETED_PRIMARY_PLOTS','cases':records},ensure_ascii=False,indent=2));return records

def run():
    bones=matched_bones()
    results=evaluate(R,baseline='MASTER_A0');results['scope']='MASTER_2018_2026_MATCHED_TRUE_NESTED_OOF_DEVELOPMENT';results['initial_cash']=300000.;results['window_start']='2018-01-02';results['window_end']='2026-07-24';results.to_csv(R/'all_results.csv',index=False)
    decisions(R,results,'MASTER_A0')
    from core.round26_1.master_factor_evidence import run as factor_report
    factor_report();relation=paired_relation();figures=plots()
    # Causal diagnostic account paths are exact aliases, not independent alpha trials.
    proof=R/'NATIVE_COUNTERFACTUAL_REPLAYS.json'
    if proof.exists():
        cases=json.loads(proof.read_text())['cases'];spa=json.loads((R/'SPA.json').read_text());spa['causal_diagnostic_aliases']={row['code']:f"{row['arm']}_17_BOTH_ADD" for row in cases};spa['causal_alias_all8_native_tables_verified']=all(row['status']=='PASS_ACTUAL_NATIVE_REPLAY_ALL8_TABLES_EXACT' for row in cases);(R/'SPA.json').write_text(json.dumps(spa,indent=2))
    return results
