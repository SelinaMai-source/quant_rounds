#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.research_integrity_v4.runner import main
if __name__=='__main__':main('round26')
