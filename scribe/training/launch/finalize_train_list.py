#!/usr/bin/env python3
"""Writes the GRPO training list: the first 192 candidates whose base run has an outline and whose
planning prompt fits GRPO's limit. Usage: python finalize_train_list.py --base_cache <dir> --out
<file> [--model <dir>]."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
ROOT = TRAINING.parent.parent
for p in (str(TRAINING), str(ROOT / "third_party" / "kvskill")):
    if p not in sys.path:
        sys.path.insert(0, p)

N_TRAIN = 192
MAX_PROMPT_TOKENS = 20000
CANDIDATES_DEFAULT = str(ROOT / "configs/train_tasks/train_candidates.txt")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base_cache", required=True, help="the base runs of the candidates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("SCRIBE_MODEL_DIR", ""), help="backbone dir (tokenizer)")
    a = ap.parse_args(argv)
    if not a.model:
        raise SystemExit("--model (or $SCRIBE_MODEL_DIR) is required")
    import planning_task as LP
    from kvskill.chat import tokenize_prompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    cands = [l.strip() for l in open(CANDIDATES_DEFAULT) if l.strip()]
    LP.refuse_eval(cands, f"finalize_train_list {CANDIDATES_DEFAULT}")
    keep, why = [], {"no_base_run": [], "prompt_too_long": []}
    for t in cands:
        if len(keep) >= N_TRAIN:
            break
        if not LP.has_base(a.base_cache, t):
            why["no_base_run"].append(t)
            continue
        b = LP.load_base(a.base_cache, t)
        n = len(tokenize_prompt(tok, LP.plan_messages(b["spec"], b["graph"])))
        if n > MAX_PROMPT_TOKENS:
            why["prompt_too_long"].append([t, n])
            continue
        keep.append(t)
    body = "".join(t + "\n" for t in keep)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    lock = {"schema": "scribe_train_final/1",
            "rule": (f"candidates in file order; drop a candidate without a base synthesis graph and "
                     f"outline or with a planning prompt over {MAX_PROMPT_TOKENS} tokens; keep the "
                     f"first {N_TRAIN}"),
            "candidates": {"path": CANDIDATES_DEFAULT, "sha256": hashlib.sha256(Path(CANDIDATES_DEFAULT).read_bytes()).hexdigest()},
            "base_cache": a.base_cache, "n": len(keep), "complete": len(keep) == N_TRAIN,
            "dropped": {k: len(v) for k, v in why.items()}, "dropped_ids": why,
            "output": {"path": str(out), "sha256": hashlib.sha256(body.encode()).hexdigest()}}
    Path(str(out.with_suffix("")) + ".lock.json").write_text(json.dumps(lock, indent=1, sort_keys=True))
    print(f"[finalize_train_list] kept {len(keep)}/{N_TRAIN} (dropped {lock['dropped']}) -> {out}")
    return 0 if len(keep) == N_TRAIN else 4


if __name__ == "__main__":
    sys.exit(main())
