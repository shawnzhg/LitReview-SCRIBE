"""Test setup of windowbench: the repository's evaluation directory first on sys.path, no bytecode and
no site roster declarations."""

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
EVAL = Path(__file__).resolve().parents[2]
if str(EVAL) in sys.path:
    sys.path.remove(str(EVAL))
sys.path.insert(0, str(EVAL))
os.environ.pop("WINDOWBENCH_ROSTER_EXTRA", None)

import windowbench

if Path(windowbench.__file__).resolve().parent.parent != EVAL:
    raise ImportError(f"windowbench resolved to {windowbench.__file__}, not the repository copy under {EVAL}")
