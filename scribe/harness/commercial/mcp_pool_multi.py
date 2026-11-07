#!/usr/bin/env python3
"""MCP server giving an agent search and fetch over the pool service for the task in the X-Task
header, without the evaluated review and, under fixed input, only within the task's reference list.
Usage: MCP_TASKS=<json> MCP_GOLD_DIR=<dir> MCP_POOL_URL=<url> [MCP_ALLOWLIST_DIR=<dir>]
MCP_TOKEN=<secret> MCP_PORT=<port> MCP_LOG=<file> python mcp_pool_multi.py."""

from __future__ import annotations

import json
import os
import time

import anyio
import requests
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

TASKS = set(json.load(open(os.environ["MCP_TASKS"])))
GOLD = os.environ["MCP_GOLD_DIR"]
POOL = os.environ["MCP_POOL_URL"].rstrip("/")
ALLOW_DIR = os.environ.get("MCP_ALLOWLIST_DIR") or None
TOKEN = os.environ["MCP_TOKEN"]
PORT = int(os.environ.get("MCP_PORT", "31961"))
LOG = os.environ["MCP_LOG"]
K = int(os.environ.get("MCP_K", "20"))


def task_ctx() -> tuple[str, set]:
    t = (get_http_headers(include_all=True).get("x-task") or "").strip()
    if t not in TASKS:
        raise ValueError("unknown task")
    rp = str(json.load(open(f"{GOLD}/{t}.json"))["review_pmid"])
    return t, {rp}


def allowlist(task: str) -> set | None:
    if ALLOW_DIR is None:
        return None
    return {str(x) for x in json.load(open(f"{ALLOW_DIR}/{task}.json"))}


def log(rec: dict) -> None:
    rec = {"t": time.time(), **rec}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def check_allowed(task: str, pmids: list) -> None:
    allow = allowlist(task)
    if allow is not None and any(str(p) not in allow for p in pmids):
        log({"task": task, "tool": "_allowlist_violation", "pmids": [str(p) for p in pmids if str(p) not in allow]})
        raise ValueError("the pool returned papers outside the task's reference list")


def url_of(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


mcp = FastMCP(name="frozen-biomedical-pool",
              instructions="Search and fetch biomedical paper abstracts (PubMed). "
                           "search returns ranked papers; fetch returns one paper's title and abstract.")


def _search(query: str, TASK: str, EXCLUDE: set) -> dict:
    r = requests.get(f"{POOL}/t/{TASK}/search", params={"q": query, "k": K + len(EXCLUDE)}, timeout=600)
    r.raise_for_status()
    hits = r.json()
    check_allowed(TASK, [h["pmid"] for h in hits])
    kept, suppressed = [], []
    for h in hits:
        if h["pmid"] in EXCLUDE:
            suppressed.append(h["pmid"])
            continue
        kept.append(h)
    kept = kept[:K]
    results = [{"id": h["pmid"], "title": h.get("title") or "", "url": url_of(h["pmid"]),
                "text": (h.get("abstract") or "")[:300]} for h in kept]
    log({"task": TASK, "tool": "search", "query": query, "returned": [h["pmid"] for h in kept],
         "suppressed": suppressed, "n_pool_hits": len(hits)})
    return {"results": results}


def _fetch(id: str, TASK: str, EXCLUDE: set) -> dict:
    pmid = str(id).strip().removeprefix("PMID:").removeprefix("pmid:")
    if pmid in EXCLUDE:
        log({"task": TASK, "tool": "fetch", "id": id, "returned": [], "suppressed": [pmid]})
        raise ValueError("document not available")
    r = requests.get(f"{POOL}/t/{TASK}/graph/v1/paper/PMID:{pmid}",
                     params={"fields": "title,abstract,year,externalIds"}, timeout=600)
    if r.status_code == 404:
        log({"task": TASK, "tool": "fetch", "id": id, "returned": [], "status": 404})
        raise ValueError("document not available")
    r.raise_for_status()
    check_allowed(TASK, [pmid])
    d = r.json()
    title, year, abstract = d.get("title") or "", d.get("year"), d.get("abstract") or ""
    log({"task": TASK, "tool": "fetch", "id": id, "returned": [pmid]})
    return {"id": pmid, "title": title, "text": f"{title}\n({year})\n\n{abstract}",
            "url": url_of(pmid), "metadata": {"pmid": pmid, "year": year}}


@mcp.tool(description="Search the biomedical literature corpus. Returns a ranked list of papers (id = PMID).")
async def search(query: str) -> dict:
    TASK, EXCLUDE = task_ctx()
    return await anyio.to_thread.run_sync(_search, query, TASK, EXCLUDE)


@mcp.tool(description="Fetch one paper by id (PMID): title, year and abstract.")
async def fetch(id: str) -> dict:
    TASK, EXCLUDE = task_ctx()
    return await anyio.to_thread.run_sync(_fetch, id, TASK, EXCLUDE)


SUPPORTED = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")


class Bearer(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            log({"tool": "_auth_reject", "path": request.url.path})
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        pv = request.headers.get("mcp-protocol-version")
        if pv and pv not in SUPPORTED:
            request.scope["headers"] = [(k, v) for k, v in request.scope["headers"] if k != b"mcp-protocol-version"] + [
                (b"mcp-protocol-version", SUPPORTED[-1].encode())]
        resp = await call_next(request)
        if resp.status_code >= 400:
            log({"tool": "_http_error", "status": resp.status_code, "method": request.method,
                 "headers": {k: v for k, v in request.headers.items() if k not in ("authorization", "x-api-key")},
                 "protocol_version_in": pv})
        return resp


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=PORT, path=f"/{TOKEN[:16]}/mcp",
            middleware=[Middleware(Bearer)], stateless_http=True)
