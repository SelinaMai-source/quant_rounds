#!/usr/bin/env python3
"""Sequential, explicit deliverables; unsupported paths fail without fake statuses."""
from pathlib import Path
import sys,argparse
C=Path('/data1/stock-mxy/课题/一进二策略');sys.path.insert(0,str(C));sys.dont_write_bytecode=True

def main():
 p=argparse.ArgumentParser();p.add_argument('--deliverable',choices=['D0','D1','D2','D3','D4','D5','all'],required=True);p.add_argument('--resume',action='store_true');p.add_argument('--publish',action='store_true');p.add_argument('--jobs',type=int,default=1);p.add_argument('--seed',type=int,default=17);a=p.parse_args()
 if not 1<=a.jobs<=2:p.error('CPU training jobs must be 1 or 2')
 selected=['D0','D1','D2','D3','D4','D5'] if a.deliverable=='all' else [a.deliverable]
 for d in selected:
  if d=='D0':
   from core.research_integrity_v5.d0 import run
   from core.research_integrity_v5.audit_execution import run as execution
   run(resume=a.resume,seed=a.seed);execution()
  else:raise NotImplementedError(d+' has no implemented execution entry yet; no completion status written')
  if a.publish:
   from core.research_integrity_v5.publish import publish
   for branch in (['round23_7','round26_1'] if d=='D0' else ['round23_7']):publish(branch,'research: execute and audit '+d)
if __name__=='__main__':main()
