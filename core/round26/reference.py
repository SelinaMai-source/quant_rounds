"""Thin entry to actual locked-author core tests; never a full L1/L2 alias."""
from pathlib import Path
import subprocess

METHODS=('AlphaGen','AlphaForge','RSR','MASTER','RD-Agent','DeepLOB')

def run_core_test(method):
    if method not in METHODS:raise ValueError('Method is outside the frozen Round26 list')
    root=Path('/data1/stock-mxy/课题/一进二策略/research/round23_6_26_execution')
    if method=='RD-Agent':
        raise RuntimeError('BLOCKED_LLM_ACCESS: official research loop was not run; no hand-written replacement')
    if method=='RSR':
        return subprocess.run([str(root/'rsr-env/bin/python'),str(Path(__file__).resolve().parents[1]/'research_integrity_v4/tasks/rsr_smoke.py')],check=True).returncode
    from core.research_integrity_v4.tasks.smoke import run
    return run(method)
