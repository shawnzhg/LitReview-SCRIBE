#!/usr/bin/env python3
"""Audits the Gemini Deep Research reports: resolves every cited PMID through the pool service under
the task's cutoff and checks titles, reference-list membership and use of the evaluated review.
Usage: python audit_gemini.py."""

import json, os, re, statistics as st
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
G = Path(os.environ["COMMERCIAL_OUT"])
GOLD = Path(os.environ["COMMERCIAL_GOLD"])
ALLOW = Path(os.environ["COMMERCIAL_ALLOWLISTS"])
POOL = os.environ["COMMERCIAL_POOL_URL"].rstrip("/")
tasks = json.loads(Path(os.environ["COMMERCIAL_TASKS"]).read_text())
cache = {}


def toks(s): return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) > 2}


def sim(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(1, min(len(A), len(B)))


def pool_title(task, pmid):
    k = (task, pmid)
    if k not in cache:
        for _ in range(3):
            try:
                r = requests.get(f"{POOL}/t/{task}/graph/v1/paper/PMID:{pmid}", params={"fields": "title,year"}, timeout=60)
                cache[k] = r.json() if r.status_code == 200 else None
                break
            except requests.RequestException:
                cache[k] = "ERR"
    return cache[k]


def parse_refs(md):
    hs = list(re.finditer(r"^#+\s*(References|Works Cited|Bibliography|Sources)\s*$", md, flags=re.I | re.M))
    if not hs:
        return body_len(md), []
    body = md[:hs[0].start()]
    secs = []
    for i, h in enumerate(hs):
        end = hs[i + 1].start() if i + 1 < len(hs) else len(md)
        secs.append(md[h.end():end])
    refs = max(secs, key=lambda x: len(re.findall(r"pubmed\.ncbi\.nlm\.nih\.gov/\d+|PMID:?\s*\d+", x)))
    out = []
    for line in refs.splitlines():
        line = line.strip()
        if not line:
            continue
        pm = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{5,9})|PMID:?\s*(\d{5,9})", line)
        pmc = re.search(r"(PMC\d{5,9})", line)
        url = re.search(r"https?://[^\s)>\]]+", line)
        title = re.sub(r"https?://\S+", "", re.sub(r"^\s*\[?\d+\]?\.?\s*", "", line)).strip(" .")
        out.append({"pmid": (pm.group(1) or pm.group(2)) if pm else None, "pmc": pmc.group(1) if pmc else None,
                    "domain": re.sub(r"^https?://([^/]+).*", r"\1", url.group(0)) if url else None, "title": title[:300]})
    return body_len(body), out


def body_len(b): return len(b.split())


rows = []
for cond in ("samepool", "fixinput"):
    for k, t in enumerate(tasks):
        stem = f"{k+1:02d}_{t}"; md = G / "runs" / cond / f"{stem}.md"; sj = G / "runs" / cond / f"{stem}.sources.json"
        rp = str(json.loads((GOLD / f"{t}.json").read_text())["review_pmid"]); rpmc = t.replace("pmcid_", "")
        allow = set(json.loads((ALLOW / f"{t}.json").read_text()))
        text = md.read_text(); words, refs = parse_refs(text)
        src = json.loads(sj.read_text())
        su, sr = src.get("sources_used") or [], src.get("sources_read_not_used") or []
        rows.append(dict(cond=cond, task=t, stem=stem, body_words=words, refs=refs, rp=rp, rpmc=rpmc, allow=allow,
                         src_used_domains=Counter(s.get("domain") for s in su), src_read_domains=Counter(s.get("domain") for s in sr),
                         review_in_sources=any(rp in (s.get("url") or "") or rpmc in (s.get("url") or "") for s in su + sr),
                         review_in_text=(rpmc in text) or (rp in text)))
jobs = [(r["task"], x["pmid"]) for r in rows for x in r["refs"] if x["pmid"]]
with ThreadPoolExecutor(16) as ex:
    list(ex.map(lambda j: pool_title(*j), sorted(set(jobs))))
out = []
for r in rows:
    n = len(r["refs"]); pm = [x for x in r["refs"] if x["pmid"]]
    inpool = [x for x in pm if isinstance(pool_title(r["task"], x["pmid"]), dict)]
    correct = [x for x in inpool if sim(x["title"], pool_title(r["task"], x["pmid"])["title"]) >= 0.5]
    wrong = [x for x in inpool if x not in correct]
    notpool = [x for x in pm if pool_title(r["task"], x["pmid"]) is None]
    o = dict(cond=r["cond"], task=r["task"], stem=r["stem"], body_words=r["body_words"], n_refs=n, n_pmid=len(pm),
             pmid_in_pool=len(inpool), pmid_title_match=len(correct), pmid_title_mismatch=len(wrong),
             pmid_not_in_pool_before_cutoff=len(notpool), refs_without_pmid=n - len(pm),
             ref_domains=dict(Counter(x["domain"] for x in r["refs"] if not x["pmid"])),
             in_allowlist=sum(1 for x in pm if x["pmid"] in r["allow"]) if r["cond"] == "fixinput" else None,
             target_review_cited=any((x["pmid"] == r["rp"] or x["pmc"] == r["rpmc"]) and "uploaded file" not in (x["title"] or "").lower() for x in r["refs"]),
             target_review_in_sources=r["review_in_sources"], target_review_mentioned=r["review_in_text"],
             src_used=dict(r["src_used_domains"]), src_read_not_used=dict(r["src_read_domains"]),
             mismatch_examples=[(x["pmid"], x["title"][:80], pool_title(r["task"], x["pmid"])["title"][:80]) for x in wrong[:3]])
    out.append(o)
(G / "audit.json").write_text(json.dumps(out, indent=1))
for cond in ("samepool", "fixinput"):
    R = [o for o in out if o["cond"] == cond]
    tot = lambda k: sum(o[k] for o in R)
    print(f"== {cond} (n={len(R)}): body words median {st.median(o['body_words'] for o in R):.0f}; refs median {st.median(o['n_refs'] for o in R)}; zero-ref runs {[o['stem'][:2] for o in R if o['n_refs']==0]}")
    print(f"  refs {tot('n_refs')}: with PMID {tot('n_pmid')} | in the pool before the cutoff {tot('pmid_in_pool')} (title OK {tot('pmid_title_match')}, title MISMATCH {tot('pmid_title_mismatch')}) | not in the pool before the cutoff {tot('pmid_not_in_pool_before_cutoff')} | no PMID {tot('refs_without_pmid')}")
    if cond == "fixinput": print(f"  cited PMIDs in allowlist: {tot('in_allowlist')} / {tot('n_pmid')}")
    print(f"  target review: cited {[o['stem'][:2] for o in R if o['target_review_cited']]}; in sources {[o['stem'][:2] for o in R if o['target_review_in_sources']]}")
    sd = Counter(); [sd.update(o['src_used']) for o in R]; rd = Counter(); [rd.update(o['src_read_not_used']) for o in R]
    print(f"  sources_used domains top: {sd.most_common(6)}"); print(f"  read_not_used domains top: {rd.most_common(6)}")
    nd = Counter(); [nd.update(o['ref_domains']) for o in R]; print(f"  non-PMID ref domains top: {nd.most_common(6)}")
