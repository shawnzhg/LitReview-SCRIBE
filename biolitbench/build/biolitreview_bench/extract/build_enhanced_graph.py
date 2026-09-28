"""Builds the claim graph of one paper: embedding-selected claim pairs, eight-type relation labels,
NLI grounding against the cited abstracts, and assembly with hard negatives. Usage: python
build_enhanced_graph.py --step candidates|label|verify|assemble --paper <name> --work <dir>
[--model <dir>] [--refs <dir>] [--grounding-dir <dir>]."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

BENCH = Path(os.environ.get("SCRIBE_BENCH_RESULTS") or "/nonexistent/SCRIBE_BENCH_RESULTS") / "benchmark_claims"
BGE_PATH = os.environ.get("SCRIBE_BGE_MODEL_DIR") or "/nonexistent/SCRIBE_BGE_MODEL_DIR"
NLI_PATH = os.environ.get("SCRIBE_NLI_MODEL_DIR") or "/nonexistent/SCRIBE_NLI_MODEL_DIR"

RELATION_TYPES = ["parallel", "contrast", "elaboration", "evidence", "mechanism", "context",
                  "synthesis", "condition"]
UNRELATED = "unrelated"
LABELS = RELATION_TYPES + [UNRELATED]
TOP_PER_CLAIM = 6
MIN_SIM = 0.45
NLI_MAX_LEN = 512

_STRIP = re.compile(
    r"^\s*(?:however|nevertheless|nonetheless|conversely|whereas|although|though|"
    r"in contrast|by contrast|on the other hand|on the contrary|instead|rather|while|"
    r"despite|in spite of|yet|still|because|due to|since|owing to|as a result of|"
    r"as a consequence of|therefore|thus|hence|accordingly|consequently|as such|"
    r"as a result|moreover|furthermore|additionally|in addition|similarly|likewise|"
    r"also|besides|equally|for example|for instance|indeed|notably|in particular|"
    r"specifically|as shown|as reported|as demonstrated|as evidenced)\b[,\s]*", re.I)

PROMPT = (
    "Judge the discourse relation between two claims from a scientific review, based "
    "ONLY on their content. Return relation_type from {parallel, contrast, elaboration, "
    "evidence, mechanism, context, synthesis, condition, unrelated}, a direction "
    "(source_to_target / target_to_source / symmetric / none), and confidence 0-1.\n\n"
    "SOURCE: <SRC>\nTARGET: <TGT>\n\nReturn ONLY JSON: "
    '{"relation_type": "...", "direction": "...", "confidence": 0.0}')

FEWSHOT = (
    "Definitions & examples (choose the SINGLE best label; connectives like 'in addition',"
    " 'moreover', 'similarly', 'in particular' signal PARALLEL, not mechanism):\n"
    "- parallel: two coordinated points on the same topic; neither explains/causes/supports the other.\n"
    "    S: 'Ribosome profiling revealed many small ORFs in UTRs.'\n"
    "    T: 'In addition, ribosomes drive mRNA quality-control by degrading abnormal mRNAs.' -> parallel\n"
    "- contrast: target conflicts with / qualifies / opposes the source.\n"
    "    S: 'Codon optimality is a major determinant of mRNA half-life in yeast.'\n"
    "    T: 'In human cells, codon optimality explains little of the variation in mRNA half-life.' -> contrast\n"
    "- elaboration: target adds conceptual detail / expands the same point (not new experimental data).\n"
    "    S: 'Codon optimality affects gene expression.'\n"
    "    T: 'It acts at both the translation-efficiency and the mRNA-stability level.' -> elaboration\n"
    "- evidence: target reports a specific experimental/empirical FINDING that supports the source claim.\n"
    "    S: 'Optimal codons raise translation efficiency.'\n"
    "    T: 'Genes enriched in optimal codons showed higher ribosome-profiling efficiency.' -> evidence\n"
    "- mechanism: target explains the causal HOW/WHY of the source (a molecular process).\n"
    "    S: 'Optimal codons increase mRNA stability.'\n"
    "    T: 'Slow decoding of nonoptimal codons exposes the mRNA to deadenylation.' -> mechanism\n"
    "- context: target gives the background, setting or history in which the source is situated.\n"
    "    S: 'Codon optimality now guides the design of therapeutic mRNAs.'\n"
    "    T: 'The genetic code assigns most amino acids to several synonymous codons.' -> context\n"
    "- synthesis: target integrates the source with other findings into a more general conclusion.\n"
    "    S: 'Nonoptimal codons slow elongation.'\n"
    "    T: 'Together, elongation speed and decay rates show that translation and mRNA turnover are coupled.' -> synthesis\n"
    "- condition: target states when, where or for whom the source holds (a limit or requirement).\n"
    "    S: 'Codon-mediated decay destabilises nonoptimal transcripts.'\n"
    "    T: 'This decay requires ongoing translation and is lost when initiation is blocked.' -> condition\n"
    "- unrelated: the two claims are about different things with no writing relation.\n"
    "    S: 'CAI measures synonymous codon bias from reference genes.'\n"
    "    T: 'GC content correlates with codon usage in some species.' -> unrelated\n"
    "Distinguish evidence (new data supporting the claim) from elaboration (conceptual detail).\n\n")

SCHEMA = {
    "type": "object",
    "properties": {
        "relation_type": {"type": "string", "enum": LABELS},
        "direction": {"type": "string",
                      "enum": ["source_to_target", "target_to_source", "symmetric", "none"]},
        "confidence": {"type": "number"},
    },
    "required": ["relation_type", "direction", "confidence"],
    "additionalProperties": False,
}


def load_paper(paper):
    claims = [json.loads(l) for l in (BENCH / paper / "claims_llm.jsonl").open()]
    struct = json.loads((BENCH / paper / "review_structure_llm.json").read_text())
    return claims, struct


def importance_weights(claims):
    freq = Counter()
    for c in claims:
        for r in set(c.get("reference_ids", [])):
            freq[r] += 1
    return dict(freq)


def bge_embed(texts, device="cuda", batch_size=64):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BGE_PATH)
    model = AutoModel.from_pretrained(BGE_PATH).to(device).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h, p=2, dim=1).cpu())
    return torch.cat(out, 0)


def step_candidates(paper, work):
    claims, _ = load_paper(paper)
    id2c = {c["claim_id"]: c for c in claims}
    ids = [c["claim_id"] for c in claims]
    emb = bge_embed([c["sentence"] for c in claims])
    sim = emb @ emb.T
    seen, cands = set(), []
    for i in range(len(ids)):
        row = sorted(((float(sim[i, j]), j) for j in range(len(ids))
                      if j != i and id2c[ids[i]]["paragraph_id"] != id2c[ids[j]]["paragraph_id"]),
                     reverse=True)
        for cos, j in row[:TOP_PER_CLAIM]:
            if cos < MIN_SIM:
                break
            key = tuple(sorted((ids[i], ids[j])))
            if key in seen:
                continue
            seen.add(key)
            a, b = id2c[ids[i]], id2c[ids[j]]
            cands.append({"source_id": a["claim_id"], "target_id": b["claim_id"],
                          "source_text": a["sentence"], "target_text": b["sentence"],
                          "source_section": a["section_id"], "target_section": b["section_id"],
                          "cosine": round(cos, 4)})
    (work / "candidates.json").write_text(json.dumps(cands))
    (work / "weights.json").write_text(json.dumps(importance_weights(claims)))
    print(f"[candidates] {len(cands)} cross-paragraph pairs")


def step_label(work, model):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    cands = json.loads((work / "candidates.json").read_text())
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=4096, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=200,
                        structured_outputs=StructuredOutputsParams(json=SCHEMA))
    msgs = [[{"role": "user", "content": FEWSHOT + PROMPT
              .replace("<SRC>", _STRIP.sub("", c["source_text"]).strip())
              .replace("<TGT>", _STRIP.sub("", c["target_text"]).strip())}] for c in cands]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    for c, o in zip(cands, outs):
        try:
            j = json.loads(o.outputs[0].text)
            c["relation_type"] = j["relation_type"] if j["relation_type"] in LABELS else None
            c["direction"] = j.get("direction", "none")
            c["rel_confidence"] = float(j.get("confidence", 0.0))
        except (ValueError, KeyError, TypeError):
            c["relation_type"] = None
    (work / "labeled.json").write_text(json.dumps(cands))
    print(f"[label] {len(cands)} pairs; {dict(Counter(c['relation_type'] for c in cands))}")


def nli_classes(pairs, batch_size=64):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLI_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    names = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if not {"entailment", "contradiction"} <= set(names.values()):
        raise SystemExit(f"NLI head has no entailment/contradiction class: {model.config.id2label}")
    out = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        enc = tok([p for p, _ in chunk], [h for _, h in chunk], truncation=True,
                  max_length=NLI_MAX_LEN, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out.extend(names[int(j)] for j in model(**enc).logits.argmax(-1))
    return out


def step_verify(paper, refs_dir, grounding_dir):
    claims, _ = load_paper(paper)
    refs = json.loads((refs_dir / f"{paper}.json").read_text())
    pairs, index = [], []
    for c in claims:
        for rid in sorted({str(r) for r in c.get("reference_ids", [])}):
            ab = (refs.get(rid) or {}).get("abstract")
            if ab:
                pairs.append((ab, c["sentence"]))
                index.append((c["claim_id"], rid))
    grounding = {}
    for (cid, rid), cls in zip(index, nli_classes(pairs) if pairs else []):
        g = grounding.setdefault(cid, {"entail": 0.0, "entail_ref": None, "contradict": 0.0,
                                       "contradict_ref": None, "n_refs": 0})
        g["n_refs"] += 1
        if cls == "entailment" and g["entail_ref"] is None:
            g["entail"], g["entail_ref"] = 1.0, rid
        if cls == "contradiction" and g["contradict_ref"] is None:
            g["contradict"], g["contradict_ref"] = 1.0, rid
    grounding_dir.mkdir(parents=True, exist_ok=True)
    (grounding_dir / f"{paper}.json").write_text(json.dumps(grounding))
    n_ent = sum(1 for g in grounding.values() if g["entail"])
    print(f"[verify] {n_ent}/{len(claims)} claims entailed by at least one cited abstract")


def step_assemble(paper, work, grounding_dir):
    claims, struct = load_paper(paper)
    id2c = {c["claim_id"]: c for c in claims}
    weights = json.loads((work / "weights.json").read_text())
    labeled = json.loads((work / "labeled.json").read_text())
    grounding = json.loads((grounding_dir / f"{paper}.json").read_text())
    parent = {s["section_id"]: s.get("parent_section_id") for s in struct.get("sections", [])}

    def top_section(cid):
        sid, seen = id2c[cid]["section_id"], set()
        while parent.get(sid) and parent[sid] in parent and sid not in seen:
            seen.add(sid)
            sid = parent[sid]
        return sid

    edges, hard_negs = [], []
    for c in labeled:
        rt = c.get("relation_type")
        if rt in RELATION_TYPES:
            edges.append(c)
        elif rt == UNRELATED and top_section(c["source_id"]) != top_section(c["target_id"]):
            hard_negs.append(c)
    for c in claims:
        g = grounding.get(c["claim_id"])
        c["groundedness"] = g["entail"] if g else None
        c["max_ref_weight"] = max((weights.get(str(r), 0) for r in c.get("reference_ids", [])), default=0)
    graph = {"paper": paper, "n_claims": len(claims), "reference_weights": weights, "claims": claims,
             "review_level_edges": edges, "hard_negatives": hard_negs}
    (work / "graph_enhanced.json").write_text(json.dumps(graph, ensure_ascii=False, indent=1))
    print(f"[assemble] {len(claims)} claims, {len(edges)} typed relations, {len(hard_negs)} hard negatives")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["candidates", "label", "verify", "assemble"])
    ap.add_argument("--paper", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--model")
    ap.add_argument("--refs")
    ap.add_argument("--grounding-dir")
    a = ap.parse_args()
    need = {"label": ["model"], "verify": ["refs", "grounding_dir"], "assemble": ["grounding_dir"]}
    for k in need.get(a.step, []):
        if not getattr(a, k):
            ap.error(f"--step {a.step} needs --{k.replace('_', '-')}")
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.step == "candidates":
        step_candidates(a.paper, work)
    elif a.step == "label":
        step_label(work, a.model)
    elif a.step == "verify":
        step_verify(a.paper, Path(a.refs), Path(a.grounding_dir))
    else:
        step_assemble(a.paper, work, Path(a.grounding_dir))


if __name__ == "__main__":
    main()
