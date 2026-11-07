#!/usr/bin/env python3
"""MCP connector for Claude Science with the tools of mcp_pool_multi.py, resolving the task and entry
condition from a per-run session code. Usage: CS_SESSIONS=<json> CS_POOL_URL_FIXINPUT=<url>
CS_POOL_URL_SAMEPOOL=<url> MCP_ALLOWLIST_DIR=<dir> MCP_GOLD_DIR=<dir> MCP_TOKEN=<secret>
MCP_PORT=<port> MCP_LOG=<file> python mcp_pool_session.py."""

from __future__ import annotations

import json
import os
import time

import anyio
import requests
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

SESS = json.load(open(os.environ["CS_SESSIONS"]))
GOLD = os.environ["MCP_GOLD_DIR"]
POOLS = {c: os.environ[f"CS_POOL_URL_{c.upper()}"].rstrip("/") for c in ("fixinput", "samepool")}
ALLOW_DIR = os.environ["MCP_ALLOWLIST_DIR"]
TOKEN = os.environ["MCP_TOKEN"]
PORT = int(os.environ.get("MCP_PORT", "31973"))
LOG = os.environ["MCP_LOG"]
K = int(os.environ.get("MCP_K", "20"))


def ctx(session: str) -> tuple[str, str, set]:
    s = SESS.get(str(session).strip())
    if not s:
        log({"tool": "_bad_session", "session": str(session)[:64]})
        raise ValueError("unknown session code")
    rp = str(json.load(open(f"{GOLD}/{s['task']}.json"))["review_pmid"])
    return s["task"], s["cond"], {rp}


def log(rec: dict) -> None:
    with open(LOG, "a") as f:
        f.write(json.dumps({"t": time.time(), **rec}) + "\n")


def check_allowed(task: str, cond: str, pmids: list) -> None:
    if cond != "fixinput":
        return
    allow = {str(x) for x in json.load(open(f"{ALLOW_DIR}/{task}.json"))}
    if any(str(p) not in allow for p in pmids):
        log({"task": task, "cond": cond, "tool": "_allowlist_violation", "pmids": [str(p) for p in pmids if str(p) not in allow]})
        raise ValueError("the pool returned papers outside the task's reference list")


def url_of(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


mcp = FastMCP(name="frozen-biomedical-pool",
              instructions="Search and fetch biomedical paper abstracts (PubMed). Every call needs the "
                           "session code given in your task. search returns ranked papers; fetch returns "
                           "one paper's title and abstract.")


def _search(query: str, task: str, cond: str, excl: set) -> dict:
    r = requests.get(f"{POOLS[cond]}/t/{task}/search", params={"q": query, "k": K + len(excl)}, timeout=(5, 600))
    r.raise_for_status()
    hits = r.json()
    check_allowed(task, cond, [h["pmid"] for h in hits])
    kept = [h for h in hits if h["pmid"] not in excl][:K]
    suppressed = [h["pmid"] for h in hits if h["pmid"] in excl]
    log({"task": task, "cond": cond, "tool": "search", "query": query, "returned": [h["pmid"] for h in kept],
         "suppressed": suppressed, "n_pool_hits": len(hits)})
    return {"results": [{"id": h["pmid"], "title": h.get("title") or "", "url": url_of(h["pmid"]),
                         "text": (h.get("abstract") or "")[:300]} for h in kept]}


def _fetch(id: str, task: str, cond: str, excl: set) -> dict:
    pmid = str(id).strip().removeprefix("PMID:").removeprefix("pmid:")
    if pmid in excl:
        log({"task": task, "cond": cond, "tool": "fetch", "id": id, "returned": [], "suppressed": [pmid]})
        raise ValueError("document not available")
    r = requests.get(f"{POOLS[cond]}/t/{task}/graph/v1/paper/PMID:{pmid}",
                     params={"fields": "title,abstract,year,externalIds"}, timeout=(5, 600))
    if r.status_code == 404:
        log({"task": task, "cond": cond, "tool": "fetch", "id": id, "returned": [], "status": 404})
        raise ValueError("document not available")
    r.raise_for_status()
    check_allowed(task, cond, [pmid])
    d = r.json()
    title, year, abstract = d.get("title") or "", d.get("year"), d.get("abstract") or ""
    log({"task": task, "cond": cond, "tool": "fetch", "id": id, "returned": [pmid]})
    return {"id": pmid, "title": title, "text": f"{title}\n({year})\n\n{abstract}",
            "url": url_of(pmid), "metadata": {"pmid": pmid, "year": year}}


@mcp.tool(description="Search the biomedical literature corpus. Returns a ranked list of papers (id = PMID).\n"
                       "`session` is the session code given in your task.")
async def search(session: str, query: str) -> dict:
    task, cond, excl = ctx(session)
    return await anyio.to_thread.run_sync(_search, query, task, cond, excl)


@mcp.tool(description="Fetch one paper by id (PMID): title, year and abstract. "
                       "`session` is the session code given in your task.")
async def fetch(session: str, id: str) -> dict:
    task, cond, excl = ctx(session)
    return await anyio.to_thread.run_sync(_fetch, id, task, cond, excl)


SUPPORTED = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")


class Shim(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        pv = request.headers.get("mcp-protocol-version")
        if pv and pv not in SUPPORTED:
            request.scope["headers"] = [(k, v) for k, v in request.scope["headers"] if k != b"mcp-protocol-version"] + [
                (b"mcp-protocol-version", SUPPORTED[-1].encode())]
        resp = await call_next(request)
        if resp.status_code >= 400:
            log({"tool": "_http_error", "status": resp.status_code, "method": request.method, "protocol_version_in": pv})
        return resp


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=PORT, path=f"/{TOKEN}/mcp",
            middleware=[Middleware(Shim)], stateless_http=True)
