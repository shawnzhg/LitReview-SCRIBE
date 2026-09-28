#!/usr/bin/env python3
"""Shared helpers of the runners: repo and run paths, hashing and sealing of artifacts, schema
validation, host class and timestamps."""

from __future__ import annotations
import hashlib
import json
import os
import socket
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def env_path(name: str) -> Path:
    return Path(os.environ.get(name) or f"/nonexistent/{name}")


ROOT = repo_root()
SCHEMAS = ROOT / "biolitbench" / "schemas"
RUNS = env_path("SCRIBE_RUNS_ROOT")


def sha256_str(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def content_hash(obj: dict, exclude=("content_hash",)) -> str:
    d = {k: v for k, v in obj.items() if k not in exclude}
    return sha256_str(json.dumps(d, sort_keys=True))


def seal(obj: dict) -> dict:
    obj["content_hash"] = content_hash(obj)
    return obj


_VALIDATORS: dict = {}


def validator(name: str):
    if name not in _VALIDATORS:
        import jsonschema
        s = json.loads((SCHEMAS / f"{name}.schema.json").read_text())
        _VALIDATORS[name] = jsonschema.Draft202012Validator(s)
    return _VALIDATORS[name]


def validate(name: str, obj: dict) -> list:
    v = validator(name)
    return [f"{'/'.join(str(x) for x in e.path)}: {e.message}" for e in v.iter_errors(obj)]



def host_class() -> str:
    return "gpu" if os.environ.get("SLURM_JOB_ID") else "login"


def hostname() -> str:
    return socket.gethostname()



def utcnow() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")
