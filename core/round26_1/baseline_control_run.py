"""Run the required real A0 control in a free CPU slot without a fake factor."""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time,datetime
import pandas as pd,psutil
from core.round26_1.market import C,R
from core.research_integrity_v5.resources import measured


def generation_workers():
    result=[]
    for proc in psutil.process_iter(['pid','cmdline','uids']):
        try:
            if proc.uids().real==os.getuid() and 'core.round26_1.alphagen_worker' in proc.cmdline():result.append(proc.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return result


def resume_after_exit(job_pid,job_created,controller_pid,controller_created):
    while psutil.pid_exists(job_pid):
        try:
            if psutil.Process(job_pid).create_time()!=job_created:break
        except psutil.NoSuchProcess:break
        time.sleep(.5)
    try:
        controller=psutil.Process(controller_pid)
        if controller.uids().real==os.getuid() and controller.create_time()==controller_created:os.kill(controller_pid,signal.SIGCONT)
    except psutil.NoSuchProcess:pass


def run(controller_pid):
    from core.round23_3.contract import E0
    from core.round26_1.baseline_control import replay
    from core.round26_1.factor_models import canonical_fuse
    from core.research_integrity_v5.window_backtest import execute
    from core.research_integrity_v5.evaluation import evaluate
    root=R/'baseline_head_replay';root.mkdir(parents=True,exist_ok=True)
    controller=psutil.Process(controller_pid);assert controller.uids().real==os.getuid()
    assert any(Path(a).name=='continue_round23_7_round26_1.py' for a in controller.cmdline())
    stopped=False;watchdog=None;started=None
    try:
        while True:
            if len(generation_workers())<=1 and controller.status()!=psutil.STATUS_STOPPED:
                os.kill(controller_pid,signal.SIGSTOP);stopped=True;time.sleep(.1)
                if len(generation_workers())<=1:break
                os.kill(controller_pid,signal.SIGCONT);stopped=False
            print('A0_CONTROL_WAITING_FOR_FREE_CPU_TRAINING_SLOT',generation_workers(),flush=True);time.sleep(5)
        started=time.monotonic()
        watchdog=subprocess.Popen([sys.executable,'-m','core.round26_1.baseline_control_run','--resume-after-exit',str(os.getpid()),str(psutil.Process().create_time()),str(controller_pid),str(controller.create_time())],cwd=C,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        with measured(root/'CONTROL_REPLAY_RESOURCES.json'):
            source={h:pd.read_parquet(E0/(h+'_pred.parquet')) for h in ['five','ten']}
            matched,proof=replay(source)
            score=canonical_fuse(matched['five'],matched['ten']);records=[]
            for bps in [0,10,20]:
                code='AG_A0'+('' if not bps else '_S'+str(bps//10));path,fp,hit=execute('round26_1',code,score,bps=bps)
                records.append({'method':'AG_A0','scenario':'S'+str(bps//10),'code':code,'path':str(path),'experiment_fingerprint':fp,'cache_hit':hit})
            stage=R/'seed_stages/CONTROL_REPLAY';stage.mkdir(parents=True,exist_ok=True);pd.DataFrame(records).to_csv(stage/'run_manifest.csv',index=False);pd.DataFrame(records).to_csv(R/'run_manifest.csv',index=False)
            legacy=R/'legacy_original_window_baseline_results.csv'
            if not legacy.exists():legacy.write_bytes((R/'matched_baseline_results.csv').read_bytes())
            result=evaluate(R,baseline='AG_A0');result['scope']='MATCHED_2023_ACTUAL_NO_FACTOR_LEARNER_CONTROL';result.to_csv(R/'matched_baseline_results.csv',index=False)
            (root/'ACTUAL_RUN_COMPLETE.json').write_text(json.dumps({'status':'REAL_NO_FACTOR_CONTROL_MODELS_AND_NATIVE_BACKTESTS_COMPLETE','head_configurations':8,'current_tree_files':40,'baseline_scenarios':3,'candidate_backtests':0,'parity_status':proof['status'],'CURRENT_BASELINE_CHANGED':False,'PRODUCTION_CONFIG_CHANGED':False},indent=2))
            print(result[['experiment_id','CAGR','Sharpe','MaxDD','trades']].to_string(index=False),flush=True)
    finally:
        if stopped:
            try:os.kill(controller_pid,signal.SIGCONT)
            except ProcessLookupError:pass
        if watchdog is not None:watchdog.terminate();watchdog.wait()
        (root/'scheduler_coordination.json').write_text(json.dumps({'controller_pid':controller_pid,'only_own_dispatcher_temporarily_stopped':stopped,'original_PPO_children_not_stopped':True,'dispatcher_resumed':stopped,'dispatcher_pause_seconds':time.monotonic()-started if started is not None else 0,'max_simultaneous_CPU_training_tasks':2,'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controller-pid',type=int);p.add_argument('--resume-after-exit',nargs=4);a=p.parse_args()
    if a.resume_after_exit:resume_after_exit(int(a.resume_after_exit[0]),float(a.resume_after_exit[1]),int(a.resume_after_exit[2]),float(a.resume_after_exit[3]))
    else:run(a.controller_pid)
