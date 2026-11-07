#!/usr/bin/env python3
"""Claude Science worker: claims unstarted runs from the queue, drives each once through the browser
and extracts its record. Usage: python cs_worker.py <worker id>."""

import json, os, sys, time, traceback
from pathlib import Path
from playwright.sync_api import sync_playwright
wid = sys.argv[1]
os.environ["CS_PROFILE"] = f"/tmp/pw_profile_{wid}"
import cs_browser as B
from cs_extract import extract
from cs_run import run_one
OUTROOT = Path(os.environ["COMMERCIAL_OUT"])
RUNS = OUTROOT / "runs"
CL = OUTROOT / "_claims"; CL.mkdir(parents=True, exist_ok=True)
tasks = json.loads(Path(os.environ["COMMERCIAL_TASKS"]).read_text())
jobs = [(c, k + 1, t) for k, t in enumerate(tasks) for c in ("fixinput", "samepool")]


def claim(cond, stem):
    if (RUNS / cond / stem / "frame.json").exists():
        return False
    try:
        os.mkdir(CL / f"{cond}__{stem}")
        (CL / f"{cond}__{stem}" / "by").write_text(f"{wid} {time.time()}")
        return True
    except FileExistsError:
        return False


with sync_playwright() as p:
    ctx, page = B.login(p)
    while True:
        nxt = next(((c, n, t) for c, n, t in jobs if claim(c, f"{n:02d}_{t}")), None)
        if nxt is None:
            break
        cond, nn, task = nxt; stem = f"{nn:02d}_{task}"
        out = RUNS / cond / stem
        t0 = time.time()
        try:
            url, wall = run_one(page, cond, nn, task)
            rec = extract(cond, stem, url)
        except Exception as e:
            rec = {"error": repr(e)[:500], "trace": traceback.format_exc()[-1500:]}
            try:
                page.screenshot(path=str(out / "error.png"))
            except Exception:
                pass
        rec.update({"cond": cond, "stem": stem, "worker": wid, "wall_s": round(time.time() - t0, 1)})
        print(json.dumps(rec), flush=True)
    ctx.close()
print("worker done", flush=True)
