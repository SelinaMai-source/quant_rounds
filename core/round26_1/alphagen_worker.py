"""One true full-budget generator fold per CPU process; no data-bearing env pickle."""
from pathlib import Path
import argparse,sys,json,traceback
sys.dont_write_bytecode=True
from core.round26_1.alphagen_train import run_fold
from core.research_integrity_v5.resources import measured
C=Path('/data1/stock-mxy/课题/一进二策略')
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['AG_RAW','AG_RES'],required=True);p.add_argument('--year',type=int,required=True);p.add_argument('--seed',type=int,choices=[17,42,73],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();r=C/'result/round26_1/D3/worker_profiles';r.mkdir(parents=True,exist_ok=True)
 with measured(r/f'{a.arm}_{a.year}_{a.seed}_whole_worker.json'):
  directory=run_fold(a.arm,a.year,a.seed,resume=a.resume)
  status='FACTOR_OOF_YEAR_COMPLETE' if (directory/'COMPLETE.json').exists() else json.loads((directory/'STATUS.json').read_text())['status'];out={'arm':a.arm,'year':a.year,'seed':a.seed,'directory':str(directory),'status':status};(r/f'{a.arm}_{a.year}_{a.seed}_result.json').write_text(json.dumps(out,indent=2));print('WORKER_RESULT '+json.dumps(out),flush=True)
if __name__=='__main__':main()
