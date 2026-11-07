"""Shared setup of the harness tests: repo imports, a jsonschema fallback, a synthetic fixed-input
task under a temporary runs root and an offline mock model client."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / "scribe" / "harness" / "runners"
RETRIEVAL = ROOT / "scribe" / "retrieval"
LAUNCHERS = ROOT / "scribe" / "launchers"
LEVERS_DIR = ROOT / "scribe" / "levers"
LEVERS_FILE = LEVERS_DIR / "writing_levers.json"

try:
    import jsonschema
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _mini_jsonschema
    sys.modules["jsonschema"] = _mini_jsonschema

for p in (str(LEVERS_DIR), str(RUNNERS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import runner as R
import windows as W
import writing_levers as LV
from common import seal, sha256_str

assert Path(R.__file__).resolve().parent == RUNNERS, R.__file__
assert Path(W.__file__).resolve().parent == RUNNERS, W.__file__
assert Path(LV.__file__).resolve().parent == LEVERS_DIR, LV.__file__

TASK = "pmcid_PMC0"
REVIEW_PMID = "90000000"


def build_task(root, task=TASK, n_papers=12, cutoff=2020, target_words=2400):
    root = Path(root)
    runs = root / "runs"
    spec = seal({"schema_version": "taskspec/1.0", "task_id": task, "split": "dev",
                 "question": "What is known about synthetic topic A?", "review_type": "narrative",
                 "scope": {"population": "", "intervention_or_topic": "Synthetic topic A", "comparators": [],
                           "outcomes": [], "inclusion": [],
                           "exclusion": [f"the review under evaluation itself (PMID {REVIEW_PMID})"]},
                 "audience": "biomedical researchers", "corpus_snapshot_id": R.SNAPSHOT,
                 "publication_cutoff": cutoff, "allowed_tools": ["pool_search", "pool_fetch"],
                 "budget": {"max_document_opens": 60},
                 "output_spec": {"target_words": target_words, "citation_style": "numeric", "required_sections": []}})
    pmids = [str(10000001 + i) for i in range(n_papers)]
    texts, papers, evidence, years = {}, [], [], {}
    for i, pid in enumerate(pmids):
        title = f"Synthetic study {i} of topic A"
        sents = [f"Study {i} measured outcome {j} in cohort {i}." for j in range(3)]
        abstract = " ".join(sents)
        sd = {f"pmid{pid}_0": title, **{f"pmid{pid}_{j + 1}": x for j, x in enumerate(sents)}}
        texts[pid] = {"title": title, "abstract": abstract, "sentences": sd}
        years[pid] = 2010 + i % 10
        papers.append({"paper_id": pid, "doi": None, "title": title, "year": years[pid], "rank": i + 1, "score": None,
                       "first_seen_step": None,
                       "retrieval_provenance": {"query": "", "tool": "canonical_allowlist", "rank_from_tool": None,
                                                "route": "cited_by_human"},
                       "decision": "include", "decision_reason": "allowlist", "post_cutoff": False})
        evidence.append({"evidence_id": f"e_{pid}", "paper_id": pid, "granularity": "abstract",
                         "locator": f"pmid{pid}", "text_hash": sha256_str(abstract), "condition": None})
        for loc, x in sd.items():
            evidence.append({"evidence_id": f"e_{pid}_{loc.rsplit('_', 1)[1]}", "paper_id": pid,
                             "granularity": "abstract_sentence", "locator": loc, "text_hash": sha256_str(x),
                             "condition": None})
    bundle = seal({"schema_version": "evidence_bundle/1.0", "task_id": task, "task_spec_hash": spec["content_hash"],
                   "provenance_tier": "reference_derived", "retrieval_status": "done", "papers": papers,
                   "evidence": evidence, "discovered_union": [], "failures": [], "budget_remaining": {},
                   "validation": {"allowlist_sha256": R.allowlist_sha256(pmids)}})
    files = {runs / "taskspecs" / "dev" / f"{task}.json": spec,
             runs / "canonical" / "campaign50_ref" / "evidence_bundle" / f"{task}.json": bundle,
             runs / "canonical" / "campaign50_ref" / "texts" / f"{task}.json": texts,
             runs / "gold" / "dev" / f"{task}.json": {"task_id": task, "review_pmid": REVIEW_PMID},
             root / "allowlists" / f"{task}.json": pmids, root / "ref_years.json": years}
    for path, obj in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=1))
    with (runs / "taskspecs" / "dev_index.jsonl").open("a") as f:
        f.write(json.dumps({"task_id": task}) + "\n")
    return {"root": root, "runs": runs, "spec": spec, "bundle": bundle, "texts": texts, "pmids": pmids}


def install_task(monkeypatch, root, task=TASK):
    t = build_task(root, task)
    runs = t["runs"]
    for name, val in (("RUNS", runs), ("ORACLE", runs / "oracle" / "dev"), ("SPECS", runs / "taskspecs" / "dev"),
                      ("CANON", runs / "canonical" / "campaign50_ref"), ("TASKSPECS_ROOT", runs / "taskspecs"),
                      ("CANON_SPLIT_ROOT", runs / "canonical"), ("GOLD_ROOT", runs / "gold"),
                      ("ALLOWLISTS", t["root"] / "allowlists"), ("REF_YEARS", t["root"] / "ref_years.json"),
                      ("_SPLIT_CACHE", {})):
        monkeypatch.setattr(R, name, val)
    return t


def install_levers(monkeypatch):
    monkeypatch.setattr(W, "writing", getattr(W, "writing", None), raising=False)
    monkeypatch.setattr(R, "PROMPT_HASH", R.PROMPT_HASH)
    return LV.install(W, R, LV.load(str(LEVERS_FILE)))


class MockClient:

    url = "mock://offline"
    model = "mock-llm"

    def __init__(self, relations=True):
        self.relations = relations
        self.calls = {}

    def describe(self):
        return {"backend": "mock-offline", "endpoints": [{"url": self.url, "served_model": self.model}]}

    def health(self):
        return True

    @staticmethod
    def kind(user):
        if "Extract the substantive findings" in user:
            return "extract"
        if "Which of these claims stand in a relation" in user:
            return "relations"
        if "Now the higher-level claims a review adds" in user:
            return "cross"
        if "Design the section structure" in user:
            return "planning"
        if "Write this section." in user:
            return "writing"
        return "other"

    def chat(self, messages, max_tokens=2048, seed=None, stop=None):
        import re
        from llm import Draw
        user = messages[-1]["content"] if messages else ""
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        k = self.kind(first_user)
        self.calls[k] = self.calls.get(k, 0) + 1
        if k == "extract":
            eids = re.findall(r"^\[(\S+)\] \(", first_user, re.M)
            out = {"claims": [{"text": f"Finding {i} of paper {e}.", "type": "study_finding", "polarity": "positive",
                               "evidence_ids": [e], "confidence": 0.8} for e in eids for i in (1, 2)]}
            text = json.dumps(out)
        elif k == "relations":
            cids = re.findall(r"^\s+(c\d+): ", first_user, re.M)
            lines = []
            if self.relations:
                for a, b in zip(cids, cids[1:]):
                    lines.append(f"{a} compares {b}")
                lines += [f"{cids[0]} supports {c}" for c in cids[2:5]] if len(cids) > 4 else []
            text = "\n".join(lines)
        elif k == "cross":
            cids = re.findall(r"^\s+(c\d+): ", first_user, re.M)
            out = {"claims": [{"text": f"Across studies, pattern {j}.", "type": t, "polarity": "mixed",
                               "from_claims": cids[j:j + 3], "confidence": 0.6}
                              for j, t in enumerate(("cross_study_synthesis", "conflict", "gap"))],
                   "unresolved_conflicts": cids[:1], "missing_information": ["long-term follow-up"]}
            text = json.dumps(out)
        elif k == "planning":
            cids = re.findall(r"^\s+(c\d+) \[", first_user, re.M)
            words = int(re.search(r"TARGET LENGTH: (\d+) words", first_user).group(1))
            n = 4
            out = {"sections": [{"section_id": f"s{i + 1}", "parent_id": None, "title": f"Section {i + 1}",
                                 "objective": f"Objective {i + 1}", "claim_ids": cids[i::n],
                                 "word_budget": max(250, words // n)} for i in range(n)]}
            text = json.dumps(out)
        elif k == "writing":
            block = first_user.split("CLAIMS TO REALISE IN THIS SECTION:", 1)[-1].split("SUPPORTING EVIDENCE:", 1)[0]
            cids = re.findall(r"^\s+(c\d+): ", block, re.M)
            pmids = re.findall(r"PMID (\d+):", first_user)
            sents = []
            for i in range(3):
                s = {"text": f"Sentence {i} of this section.", "claim_ids": cids[i:i + 1], "citations": pmids[i:i + 1]}
                if i == 2:
                    s["new_paragraph"] = True
                sents.append(s)
            text = json.dumps({"sentences": sents})
        else:
            text = "{}"
        flat = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
        return Draw(prompt=flat, completion=text, finish_reason="stop",
                    params={"max_tokens": max_tokens, "seed": seed, "backend": "mock"},
                    n_prompt_tokens=len(flat) // 4, n_completion_tokens=len(text) // 4, wall_ms=1,
                    model=self.model, role_sequence=[m["role"] for m in messages])
