"""Shared setup of the evaluation tests: no bytecode, no site roster file, and the repo's evaluation
directory first on sys.path."""

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.pop("WINDOWBENCH_ROSTER_EXTRA", None)
ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "evaluation"
if str(EVAL) in sys.path:
    sys.path.remove(str(EVAL))
sys.path.insert(0, str(EVAL))
