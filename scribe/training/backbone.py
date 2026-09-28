#!/usr/bin/env python3
"""Builds the backbone dir from a Hugging Face snapshot with the shipped chat-template patch, checks
a dir against the pin, and guards GRPO against trainable backbone weights. Usage: python backbone.py
prepare <snapshot> <out> | check <model dir>."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PIN = HERE.parent.parent / "configs" / "models" / "Qwen3.8-27B-nothink.json"
TEMPLATE = "chat_template.jinja"
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
HEX64 = re.compile(r"[0-9a-f]{64}")


def sha256_file(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_pin(pin_path=None) -> dict:
    p = Path(pin_path or DEFAULT_PIN)
    pin = json.loads(p.read_text())
    pin["_path"] = str(p)
    pin["_patch"] = str(p.parent / pin["chat_template_patch"])
    return pin


def apply_patch(text: str, patch: str) -> str:
    src = text.splitlines(keepends=True)
    hunks, cur = [], None
    for line in patch.splitlines(keepends=True):
        m = HUNK.match(line)
        if m:
            cur = {"start": int(m.group(1)), "count": int(m.group(2) or 1), "lines": []}
            hunks.append(cur)
        elif cur is not None and line.startswith("\\"):
            tag, body = cur["lines"][-1]
            cur["lines"][-1] = (tag, body.rstrip("\n"))
        elif cur is not None and line[:1] in (" ", "-", "+"):
            cur["lines"].append((line[0], line[1:]))
    if not hunks:
        raise ValueError("the patch has no hunk")
    out, pos = [], 0
    for h in hunks:
        start = h["start"] - 1 if h["count"] else h["start"]
        if start < pos:
            raise ValueError("overlapping hunks")
        out.extend(src[pos:start])
        pos = start
        for tag, body in h["lines"]:
            if tag in (" ", "-"):
                if pos >= len(src) or src[pos] != body:
                    raise ValueError(f"the patch does not apply at line {pos + 1}")
                pos += 1
                if tag == " ":
                    out.append(body)
            else:
                out.append(body)
    out.extend(src[pos:])
    return "".join(out)


def _shard_hash(p: Path, full: bool) -> str:
    blob = os.path.basename(os.path.realpath(p))
    if not full and HEX64.fullmatch(blob):
        return blob
    return sha256_file(p)


def _check_weights(d: Path, pin: dict, full: bool) -> int:
    shards = sorted(f for f in os.listdir(d) if f.startswith("model-") and f.endswith(".safetensors"))
    want = pin["weight_shards"]
    if shards != sorted(want):
        raise ValueError(f"{d} has {len(shards)} weight shards, the pin has {len(want)}")
    for f in shards:
        got = _shard_hash(d / f, full)
        if got != want[f]:
            raise ValueError(f"shard {f} of {d} is not the pinned weight ({got[:16]} != {want[f][:16]})")
    return len(shards)


def _check_files(d: Path, want: dict) -> None:
    for f, h in want.items():
        if not (d / f).is_file():
            raise ValueError(f"{d} has no {f}")
        if sha256_file(d / f) != h:
            raise ValueError(f"{d}/{f} differs from the pin ({h[:16]})")


def check_source(snapshot, pin_path=None, full: bool = False) -> dict:
    pin, d = load_pin(pin_path), Path(snapshot)
    if not d.is_dir():
        raise ValueError(f"snapshot {d} does not exist")
    n = _check_weights(d, pin, full)
    _check_files(d, {f: h for f, h in pin["sha256"].items() if f != TEMPLATE})
    _check_files(d, pin["source_sha256"])
    return {"snapshot": str(d), "n_shards": n, "revision": pin["revision"]}


def check(model_dir, pin_path=None, full: bool = False) -> dict:
    pin, d = load_pin(pin_path), Path(model_dir)
    if not d.is_dir():
        raise ValueError(f"model dir {d} does not exist")
    n = _check_weights(d, pin, full)
    _check_files(d, pin["sha256"])
    return {"model_dir": str(d), "realpath": os.path.realpath(d), "n_shards": n,
            "pin": pin["_path"], "revision": pin["revision"], "pinned": True}


def prepare(snapshot, out, pin_path=None, full: bool = False) -> dict:
    pin, src, dst = load_pin(pin_path), Path(snapshot), Path(out)
    check_source(src, pin_path, full)
    if dst.exists() and any(dst.iterdir()):
        raise ValueError(f"{dst} exists and is not empty")
    patched = apply_patch((src / TEMPLATE).read_text(), Path(pin["_patch"]).read_text())
    if hashlib.sha256(patched.encode()).hexdigest() != pin["sha256"][TEMPLATE]:
        raise ValueError("the patched chat template does not match the pin")
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(os.listdir(src)):
        if f == TEMPLATE:
            (dst / f).write_text(patched)
        else:
            os.symlink(os.path.realpath(src / f), dst / f)
    return check(dst, pin_path, full)


def install_grpo_guard(pin_path=None) -> None:
    import kvskill.grpo as G
    T = G.SkillTrainer
    if getattr(T, "_frozen_backbone_guard", False):
        return
    pin = pin_path or os.environ.get("SCRIBE_BACKBONE_PIN", "")
    orig_init, orig_step = T.__init__, T.train_step

    def __init__(self, cfg, *a, **k):
        rec = check(cfg.model_name, pin) if pin else None
        orig_init(self, cfg, *a, **k)
        n_all = sum(1 for _ in self.flex.model.parameters())
        n_req = sum(1 for p in self.flex.model.parameters() if p.requires_grad)
        if n_req:
            raise RuntimeError(f"{n_req} of {n_all} backbone parameters require grad; the "
                               f"backbone must be frozen")
        print(f"[backbone] frozen: 0 of {n_all} backbone parameters require grad; "
              f"model dir {'pinned ' + json.dumps(rec) if rec else 'not pin-checked (no pin)'}",
              flush=True)

    def train_step(self, step):
        stats = orig_step(self, step)
        n = stats.get("base_params_with_grad") if isinstance(stats, dict) else None
        if n is not None:
            if int(n) != 0:
                raise RuntimeError(f"base_params_with_grad = {n} at step {step}: the backbone "
                                   f"received gradients")
            print(f"[backbone] step {step}: base_params_with_grad = 0", flush=True)
        return stats

    T.__init__, T.train_step, T._frozen_backbone_guard = __init__, train_step, True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("snapshot")
    p.add_argument("out")
    c = sub.add_parser("check")
    c.add_argument("model_dir")
    for s in (p, c):
        s.add_argument("--pin", default=str(DEFAULT_PIN))
    a = ap.parse_args(argv)
    try:
        rec = (prepare(a.snapshot, a.out, a.pin) if a.cmd == "prepare"
               else check(a.model_dir, a.pin))
    except ValueError as e:
        print(f"FATAL {e}", file=sys.stderr)
        return 3
    print(f"[backbone] {json.dumps(rec)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
