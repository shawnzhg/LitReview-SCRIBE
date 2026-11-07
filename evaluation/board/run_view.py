"""Runs the windowbench planning-axis builder, or prints roster declarations, inside a private merged
view after declaring the view's external keys; called by view_builder.py. Usage: python run_view.py
windowbench.outline_axis_add <args> | declarations <key> ...."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ALLOWED = ("windowbench.outline_axis_add", "declarations")
VIEW_MARKER = ".merged_view"
DECL_FIELDS = ("entries", "model_group", "provenance", "label", "role", "draft_is_final", "backbone", "retrieval_interface",
               "temperature", "family")


def inject(R, spec_path: str) -> dict:
    spec = json.loads(Path(spec_path).read_text())
    done = {}
    for key, d in spec["systems"].items():
        if key in R.SYSTEMS:
            raise SystemExit(f"VIEW REFUSED: {key!r} is already declared in roster.py; the injection may only add")
        kw = dict({"gpt_ref": R._GPT_REF, "gpt_self": R._GPT_SELF}[d["template"]])
        kw.update(synthesis=None, planning=None, writing=None)
        if d.get("model_group"):
            kw["model_group"] = d["model_group"]
        R.SYSTEMS[key] = R._arm(d["label"], interface=d["interface"], temperature=d["temperature"], notes=d["notes"],
                                draft_is_final=True, **kw)
        if d.get("backbone"):
            R.SYSTEMS[key]["backbone"] = d["backbone"]
        R.SYSTEMS[key]["vendor_model"] = d.get("vendor_model")
        done[key] = d["template"]
    return {"declared": done}


def declarations(R, keys: list[str]) -> dict:
    return {k: {**{f: R.SYSTEMS[k].get(f) for f in DECL_FIELDS},
                "in_datasets": [d for d, v in R.DATASETS.items() if k in v]} for k in keys}


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("-h", "--help"):
        print(f"usage: run_view.py {{{','.join(ALLOWED)}}} <args>")
        return 0
    if not argv or argv[0] not in ALLOWED:
        print(f"VIEW REFUSED: first argument must be one of {ALLOWED}", file=sys.stderr)
        return 2
    mod, args = argv[0], argv[1:]
    out = Path(os.environ.get("CCBENCH_OUT", "/nonexistent"))
    vdir = out.parent.resolve()
    if out.name != "out" or not (vdir / VIEW_MARKER).exists():
        print(f"VIEW REFUSED: CCBENCH_OUT={str(out)!r} is not a view's out directory (no view marker)", file=sys.stderr)
        return 2
    need = ["WINDOWBENCH_OUTLINE_AXIS_ROWS", "WINDOWBENCH_ROSTER_EXTRA"]
    if os.environ.get("WINDOWBENCH_ROSTER_INJECT"):
        need.append("WINDOWBENCH_ROSTER_INJECT")
    for var in need:
        p = Path(os.environ.get(var, "/nonexistent"))
        if not p.is_file() or vdir not in p.resolve().parents:
            print(f"VIEW REFUSED: {var}={str(p)!r} is not a file of the view {vdir}", file=sys.stderr)
            return 2
    from windowbench import config as C
    if C.CCB_OUT.resolve() != out.resolve():
        print(f"VIEW REFUSED: windowbench.config.CCB_OUT = {C.CCB_OUT} != CCBENCH_OUT {out}", file=sys.stderr)
        return 2
    C.SOURCES["outline_axis_rows"] = Path(os.environ["WINDOWBENCH_OUTLINE_AXIS_ROWS"])
    from windowbench import roster as R
    if os.environ.get("WINDOWBENCH_ROSTER_INJECT"):
        rec = inject(R, os.environ["WINDOWBENCH_ROSTER_INJECT"])
        print(f"view wrapper: declared {rec['declared']}", file=sys.stderr)
    if mod == "declarations":
        print(json.dumps(declarations(R, args), default=str))
        return 0
    rc = importlib.import_module(mod).main(args)
    return int(rc) if isinstance(rc, int) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
