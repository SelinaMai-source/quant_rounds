"""D0 real-data mutation, maturity, ledger and frozen aggregation diagnostics."""
from pathlib import Path
import json,hashlib,subprocess,sys,copy,time,importlib.metadata
import numpy as np,pandas as pd,joblib
from .cache import identity,source_manifest,fingerprint,transaction,sha
from .model import fit_predict,training_rows,predict_rows,maturity
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_7/D0'
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str))
def framehash(f):return hashlib.sha256(pd.util.hash_pandas_object(f,index=False).values.tobytes()).hexdigest()
def run(resume=True,seed=17):
 from core.round7.contract import FIVE_HEAD,TEN_HEAD
 from core.r225241_sleeve import apply_b03_o01
 from core.stock_readonly import SNAPSHOT_RELATIVE,stock_path
 R.mkdir(parents=True,exist_ok=True);start=time.time()
 inputs=[C/f'result/round22_3/data/{h}_{k}.parquet' for h in ['five','ten'] for k in ['training','candidate']]
 data=[identity(p) for p in inputs];locks=json.loads((C/'result/round23_6/source_lock.json').read_text())
 env={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
 code=source_manifest([C/'core',Path('/data1/stock-mxy/src/first_board_hermes')])
 contract={'code':code,'third_party':locks,'dirty_diff':hashlib.sha256(subprocess.check_output(['git','-C',str(C/'research/round23_7_26_1_execution/round23_7'),'diff','HEAD'])).hexdigest(),'environment':env,'parameters':{'early_middle_late':[2018,2022,2026],'all_boundaries':list(range(2018,2027))},'seed':seed,'features':{h.name:list(h.features) for h in [FIVE_HEAD,TEN_HEAD]},'universe':'frozen candidate file, no target mask','calibration':'none for mutation tests; frozen SL/BOA diagnostics','data':data,'label':{h.name:h.target for h in [FIVE_HEAD,TEN_HEAD]},'maturity':{h.name:h.label_end_column+' at close 15:00 Asia/Shanghai' for h in [FIVE_HEAD,TEN_HEAD]},'windows':'3y train,1y prediction, immature excluded','execution':'legacy B03/O01, original ROUND7_FLAT_5BP no rewrite','upstream':[]}
 dump(R/'experiment_contract.json',contract);dump(R/'input_manifest.json',data)
 ptest=subprocess.run([sys.executable,'-m','unittest','core.research_integrity_v5.test_cache'],cwd=C,capture_output=True,text=True);(R/'cache_tests.log').write_text(ptest.stdout+ptest.stderr);assert ptest.returncode==0
 records=[];bounds=[]
 for head in [FIVE_HEAD,TEN_HEAD]:
  cols=list(dict.fromkeys(['trade_date','instrument_id','next1_date',*head.features,head.target,head.label_end_column]))
  t=pd.read_parquet(C/f'result/round22_3/data/{head.name}_training.parquet',columns=cols);p=pd.read_parquet(C/f'result/round22_3/data/{head.name}_candidate.parquet',columns=cols)
  for f in [t,p]:f['trade_date']=pd.to_datetime(f.trade_date);f['instrument_id']=f.instrument_id.astype(str)
  for year in range(2018,2027):
   a=training_rows(t,head,year);cut=pd.Timestamp(f'{year}-01-01');masked=t.copy();mask=maturity(masked,head).ge(cut)|masked.trade_date.ge(cut);masked.loc[mask,head.target]=np.arange(mask.sum())*1234567.
   pd.testing.assert_frame_equal(a,training_rows(masked,head,year),check_exact=True)
   bounds.append({'head':head.name,'outer_year':year,'rows':len(a),'max_label_available_at':str(maturity(a,head).max()),'excluded_immature_pre_cutoff':int((t.trade_date.lt(cut)&maturity(t,head).ge(cut)).sum()),'training_input_unchanged':True,'cutoff':str(cut)})
  for year in [2018,2022,2026]:
   cut=pd.Timestamp(f'{year}-01-01');variants={};original=None
   for variant in ['original','future_labels_deleted','future_labels_permuted','immature_labels_permuted','future_feature_backfill']:
    tt=t.copy();pp=p.copy();future=tt.trade_date.ge(cut);imm=tt.trade_date.lt(cut)&(maturity(tt,head).isna()|maturity(tt,head).ge(cut));rng=np.random.default_rng(seed)
    if variant=='future_labels_deleted':tt.loc[future,head.target]=np.nan;pp[head.target]=np.nan
    if variant=='future_labels_permuted':tt.loc[future,head.target]=rng.normal(size=future.sum())*1e7;pp[head.target]=rng.normal(size=len(pp))*1e9
    if variant=='immature_labels_permuted':tt.loc[imm,head.target]=rng.normal(size=imm.sum())*1e8
    if variant=='future_feature_backfill':
     # Changes after the tested prediction block, never edits its historical event rows.
     tt.loc[tt.trade_date.ge(f'{year+1}-01-01'),list(head.features)]=1e8
     pp.loc[pp.trade_date.ge(f'{year+1}-01-01'),list(head.features)]=1e8
    model,a,out,y=fit_predict(tt,pp,head,year,seed);m=R/'models'/f'{head.name}_{year}_{variant}.json';m.parent.mkdir(exist_ok=True);model.save_model(m)
    signature=(framehash(a[list(head.features)]),hashlib.sha256(y.tobytes()).hexdigest(),framehash(out))
    # Actual B03/O01 plan path, on unchanged complete per-head output keys/scores.
    ov,_=apply_b03_o01(out.rename(columns={'trade_date':'signal_date'}));signature+= (framehash(ov),)
    if original is None:original=signature;out.to_parquet(m.with_suffix('.predictions.parquet'),index=False)
    assert signature==original,(head.name,year,variant)
    record={'head':head.name,'year':year,'variant':variant,'training_rows':len(a),'prediction_rows':len(out),'max_training_label_available_at':str(maturity(a,head).max()),'fit_cutoff':str(cut),'model_hash':sha(m),'input_hash':signature[0],'rank_label_hash':signature[1],'prediction_hash':signature[2],'plan_hash':signature[3],'PASS':True};records.append(record)
    print('D0 mutation',head.name,year,variant,'PASS',flush=True)
 pd.DataFrame(records).to_csv(R/'actual_entry_mutation_tests.csv',index=False);pd.DataFrame(bounds).to_csv(R/'all_year_boundaries.csv',index=False)
 # Verify Five source equation against stored values, not just maturity-column naming.
 five=pd.read_parquet(C/'result/round22_3/data/five_candidate.parquet');valid=five.label_e2_net.notna();calc=five.next2_close*(1-.00123)*(1-.0001-.0005)/(five.next1_open*(1+.00123)*(1+.0001))-1
 ferr=float((calc[valid]-five.loc[valid,'label_e2_net']).abs().max());assert ferr<1e-12
 ten=pd.read_parquet(C/'result/round22_3/data/ten_candidate.parquet');assert (pd.to_datetime(ten.loc[ten.label_exit_rule_net.notna(),'actual_exit_date'])>=pd.to_datetime(ten.loc[ten.label_exit_rule_net.notna(),'next2_date'])).all()
 # Existing actual executor trades independently reconcile Ten date and net-return formula.
 t=pd.read_parquet(C/'result/round23_6/bt/C00_V5/trades.parquet');z=t.merge(ten[['trade_date','instrument_id','actual_exit_date','label_exit_rule_net']].rename(columns={'trade_date':'signal_date','actual_exit_date':'label_exit'}),on=['signal_date','instrument_id'],how='left',validate='many_to_one')
 err=float((z.ret-z.label_exit_rule_net).abs().max());datebad=int(z.actual_exit_date.ne(z.label_exit).sum());assert err<1e-12 and datebad==0
 dump(R/'maturity_audit.json',{'Five':{'policy':'fixed D+1 open to D+2 close with label tradability; not Ten actual exit','checked_rows':int(valid.sum()),'formula_max_error':ferr,'matures':'next2 exchange close 15:00'},'Ten':{'policy':'first executable close from D+2, close limit-up hold; actual_exit_date','actual_executed_labels_checked':len(z),'return_max_error':err,'exit_mismatches':datebad,'unexecuted_action_labels':'audited source logic and dates; exhaustive all-candidate market replay not yet covered'},'event_time_vs_provider_known_at':'market exchange close reconstructed; exact vendor ingestion/publication versions unavailable; derived microfeatures retain source-specific PIT limitations'})
 ledger=[]
 for folder in sorted((C/'result/round23_6/bt').iterdir()):
  if not (folder/'nav.parquet').exists():continue
  n=pd.read_parquet(folder/'nav.parquet');t=pd.read_parquet(folder/'trades.parquet');p=pd.read_parquet(folder/'positions.parquet');s=pd.read_parquet(folder/'sleeve_states.parquet')
  realized=t.assign(pnl=t.sell_proceeds-t.buy_cost+t.cash_dividend).groupby('actual_exit_date').pnl.sum().reindex(n.trade_date,fill_value=0).cumsum().to_numpy()
  # Open cost basis from recorded initial purchases belonging to not-yet-closed lots.
  # In this frozen historical window terminal positions are all closed. Daily basis is reconstructed from buys/sells, separately marking dividends.
  buys=t.groupby('entry_date').buy_cost.sum().reindex(n.trade_date,fill_value=0).cumsum().to_numpy();closed=t.groupby('actual_exit_date').buy_cost.sum().reindex(n.trade_date,fill_value=0).cumsum().to_numpy();basis=buys-closed
  unreal=n.market_value.to_numpy()-basis;bridge=300000+realized+unreal
  tol=np.maximum(.01,1e-8*n.total_asset.to_numpy());bridgeerr=np.abs(bridge-n.total_asset.to_numpy());identityerr=np.abs(n.cash+n.market_value-n.total_asset)
  # Changes from dividends on open lots require explicit treatment, cannot hide residual.
  daily=pd.DataFrame({'trade_date':n.trade_date,'actual_nav':n.total_asset,'realized_pnl_cumulative':realized,'open_cost_basis':basis,'market_value':n.market_value,'unrealized_pnl':unreal,'reconstructed_nav':bridge,'difference':bridge-n.total_asset,'tolerance':tol})
  daily.to_csv(R/(folder.name+'_daily_nav_bridge.csv.gz'),index=False)
  rows={'experiment_id':folder.name,'cash_identity_max_error':float(identityerr.max()),'nav_bridge_max_error':float(bridgeerr.max()),'bridge_PASS':bool((bridgeerr<=tol).all()),'terminal_market_value':float(n.market_value.iloc[-1]),'min_cash':float(n.cash.min()),'rows':len(n),'all_trades_closed':bool(n.market_value.iloc[-1]==0)};ledger.append(rows)
 pd.DataFrame(ledger).to_csv(R/'ledger_results.csv',index=False)
 diagnostics=[];cal=[]
 for head in ['five','ten']:
  bank=pd.read_parquet(C/f'result/round23_6/expert_bank/{head}/calibrated_oof.parquet');target='label_e2_net' if head=='five' else 'label_exit_rule_net';bank=bank.loc[bank.trade_date.ge('2018-01-01')]
  for expert in ['xgb','lgbm','ridge','mean']:
   v=bank.dropna(subset=[expert,target]);e=v[expert]-v[target];means=v.groupby('trade_date')[[expert,target]].mean();cx=v[expert]-v.groupby('trade_date')[expert].transform('mean');cy=v[target]-v.groupby('trade_date')[target].transform('mean');ric=v.groupby('trade_date').apply(lambda f:f[expert].corr(f[target],method='spearman'),include_groups=False)
   top=v.sort_values(['trade_date',expert,'instrument_id'],ascending=[True,False,True]).groupby('trade_date').head(2)
   diagnostics.append({'head':head,'expert':expert,'rows':len(v),'MSE_all':float(np.mean(e**2)),'MSE_date_mean':float(np.mean((means[expert]-means[target])**2)),'MSE_centered':float(np.mean((cx-cy)**2)),'RankIC':float(ric.mean()),'intraday_std_mean':float(v.groupby('trade_date')[expert].std().mean()),'tie_fraction':float(1-v.groupby('trade_date')[expert].nunique().sum()/len(v)),'Top2_mature_net_return_mean':float(top[target].mean())})
  for path in sorted((C/f'result/round23_6/expert_bank/{head}').glob('calibrator_*.joblib')):
   m=joblib.load(path);cal.append({'head':head,'file':path.name,'slope':float(m.coef_[0]),'intercept':float(m.intercept_),'sha256':sha(path)})
 pd.DataFrame(diagnostics).to_csv(R/'SL_BOA_error_rank_diagnostics.csv',index=False);pd.DataFrame(cal).to_csv(R/'calibrator_coefficients.csv',index=False)
 for name in ['meta_weights_by_time_SL_BOA.csv','top1_top2_actual_pnl.csv','all_results.csv']:
  __import__('shutil').copy2(C/'result/round23_6'/name,R/('archived_'+name))
 status={'IMPLEMENTED':True,'UNIT_TESTED':True,'REFERENCE_RUN_COMPLETE':False,'TARGET_OOF_COMPLETE':False,'BACKTEST_COMPLETE':False,'EVIDENCE_COMPLETE':False,'actual_mutation_model_fits':len(records),'covered_heads':['five','ten'],'covered_model_folds':[2018,2022,2026],'all_year_boundaries':list(range(2018,2027)),'uncovered':['neural models not yet implemented','all-candidate Ten action market replay','vendor historical revisions of derived microfeatures','new-model integrated tests must accompany each deliverable'],'ledger_all_pass':all(x['bridge_PASS'] for x in ledger),'seconds':time.time()-start,'experiment_fingerprint':fingerprint(contract),'scope':'actual XGB entry and original overlay; not blanket no leakage'}
 dump(R/'STATUS.json',status);print(json.dumps(status),flush=True)
