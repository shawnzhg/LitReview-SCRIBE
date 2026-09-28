#!/usr/bin/env python3
"""Copies one Claude Science run's record out of the app's data directory and selects its final
report. Usage: python cs_extract.py <cond> <stem> <url>."""

import json, os, shutil, sqlite3, sys
from pathlib import Path
import re


def msg_text(m):
    c = m.get("content")
    return c if isinstance(c, str) else "\n".join(b.get("text", "") for b in (c or []) if isinstance(b, dict) and b.get("type") == "text")


def extract(cond, stem, url):
    fid = url.rstrip("/").split("/frames/")[-1]
    D = Path(os.environ["CS_DATA"]) / "orgs"
    db = next(D.glob("*/operon-cli.db"))
    OUT = Path(os.environ["COMMERCIAL_OUT"]) / "runs" / cond / stem
    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    row = dict(con.execute("select * from frames where id=?", (fid,)).fetchone())
    kids = [dict(r) for r in con.execute("select id,agent_name,model,effort,input_tokens,output_tokens,total_cost,status from frames where root_frame_id=? and id!=?", (fid, fid))]
    row["_child_frames"] = kids
    (OUT / "frame.json").write_text(json.dumps(row, indent=1, default=str))
    with open(OUT / "messages.jsonl", "w") as f:
        for r in con.execute("select frame_id, idx, msg_json from frame_messages where frame_id in (select id from frames where root_frame_id=?) order by frame_id, idx", (fid,)):
            f.write(json.dumps({"frame_id": r["frame_id"], "idx": r["idx"], "msg": json.loads(r["msg_json"])}) + "\n")
    ws = next(D.glob(f"*/workspaces/{fid}"), None)
    mds = sorted(ws.glob("*.md")) if ws else []
    for p in mds:
        shutil.copy(p, OUT / f"{stem}.artifact.{p.name}")
    asst = [m for m in (json.loads(l) for l in open(OUT / "messages.jsonl"))
            if m["frame_id"] == fid and m["msg"].get("role") == "assistant"]
    src = None
    ids = [a for m in asst[::-1] for a in re.findall(r"\{\{artifact:([0-9a-f-]{36})\}\}", msg_text(m["msg"]))]
    arts = []
    for a in ids[:1]:
        v = con.execute("select storage_path from artifact_versions where id=? union all "
                        "select v.storage_path from artifacts x join artifact_versions v on v.id=x.latest_version_id where x.id=?",
                        (a, a)).fetchone()
        if v:
            arts += [p for p in (db.parent / "artifacts" / v[0],) if p.exists()]
    final = [m for m in asst if re.search(r"\n#+\s*References", msg_text(m["msg"]))]
    if arts:
        best = max(arts, key=lambda p: p.stat().st_mtime)
        shutil.copy(best, OUT / "report.md"); src = f"linked artifact {best.parent.name}/{best.name}"
    elif final and len(msg_text(final[-1]["msg"]).split()) >= 800:
        (OUT / "report.md").write_text(msg_text(final[-1]["msg"])); src = f"message idx {final[-1]['idx']}"
    elif mds:
        best = max(mds, key=lambda p: p.stat().st_mtime)
        shutil.copy(best, OUT / "report.md"); src = f"workspace {best.name}"
    return {"frame": fid, "model": row["model"], "effort": row["effort"], "status": row["status"], "in": row["input_tokens"],
            "out": row["output_tokens"], "cost": row["total_cost"], "report_source": src,
            "children": [(k["agent_name"], k["model"]) for k in kids], "md_artifacts": [p.name for p in mds]}


if __name__ == "__main__":
    print(json.dumps(extract(*sys.argv[1:4])))
