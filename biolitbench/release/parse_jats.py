#!/usr/bin/env python3
"""Parses PMC JATS article XML into the review structure, claim, reference and relation files of the benchmark."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool

_REFERENCES_TITLES = {'works cited', 'references', 'references cited', 'bibliography', 'literature cited'}


def clean_text(text: str) -> str:
    text = text.translate(str.maketrans({"ﬁ": "fi", "ﬂ": "fl"}))
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", "", text)
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def section_type_for_title(title: str) -> str:
    lower = title.lower()
    if lower.startswith("abstract"):
        return "abstract"
    if lower.startswith("keywords"):
        return "keywords"
    if lower in _REFERENCES_TITLES or lower.startswith("references"):
        return "references"
    if "introduction" in lower:
        return "background"
    if "conclusion" in lower or "discussion" in lower:
        return "discussion_or_conclusion"
    return "body"


def parent_section_id(sections: list[dict], heading_level: int) -> str | None:
    if heading_level <= 1:
        return None
    for section in reversed(sections):
        if int(section.get("heading_level") or 1) < heading_level:
            return str(section["section_id"])
    return None


_ABBREV = [
    "Fig", "Figs", "fig", "figs", "Table", "Tables", "Tab", "Eq", "Eqs",
    "ref", "refs", "Ref", "Refs", "No", "Nos", "Sec", "Suppl", "Supp",
    "e.g", "i.e", "cf", "vs", "viz", "approx", "ca", "etc", "al",
    "Dr", "Prof", "Mr", "Ms", "Mrs", "St", "Inc", "Ltd", "Co",
    "i.v", "i.p", "s.c", "p.o", "a.u", "wt", "vol", "Vol", "ed", "Eds",
]
_ABBREV_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREV) + r")\.")
_INITIAL_RE = re.compile(r"\b([A-Z])\.(?=\s+[A-Z])")
_DOT = ""
_BOUNDARY_RE = re.compile(r"[.!?][\"')\]]?(?=\s+[A-Z(\[])")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + _DOT, text)
    protected = _INITIAL_RE.sub(lambda m: m.group(1) + _DOT, protected)
    protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + _DOT + m.group(2), protected)
    pieces: list[str] = []
    last = 0
    for m in _BOUNDARY_RE.finditer(protected):
        pieces.append(protected[last:m.end()])
        last = m.end()
    pieces.append(protected[last:])
    return [p.replace(_DOT, ".").strip() for p in pieces if p.strip()]


CITATION_BRACKET_RE = re.compile(r"\[\s*\d[\d,;\s\-–—]*\]")
CITATION_SUPERSCRIPT_RE = re.compile(r"(?<=[a-z]{2})\d{1,3}(?:[,;–—\-]\d{1,3})*(?=[.,;:)\s]|$)")
CITATION_AUTHOR_RE = re.compile(
    r"\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,\s*[12]\d{3}[a-z]?"
    r"(?:;\s*[A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,\s*[12]\d{3}[a-z]?)*\)")
CITATION_AUTHOR_NOSPACE_RE = re.compile(
    r"\([A-Z][A-Za-z\-]+(?:(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-]+)|\s+et\s+al\.)?\s+[12]\d{3}[a-z]?"
    r"(?:[,;]\s*[A-Z][A-Za-z\-]+(?:(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-]+)|\s+et\s+al\.)?\s+[12]\d{3}[a-z]?)*\)")
CITATION_PAREN_NUM_RE = re.compile(r"\(\s*\d{1,3}(?:\s*[,;–—\-]\s*\d{1,3})*\s*\)")
_RANGE_RE = re.compile(r"(\d+)\s*[–—\-]\s*(\d+)")
_DOI_RE = re.compile(r"\bdoi:", re.IGNORECASE)
_AUTHOR_AFF_RE = re.compile(r"[A-Z][a-z]{2,}\d+")

_AA_CODES = {"ala", "arg", "asn", "asp", "cys", "gln", "glu", "gly", "his", "ile",
             "leu", "lys", "met", "phe", "pro", "ser", "thr", "trp", "tyr", "val",
             "sec", "pyl"}
_WORD_BEFORE_RE = re.compile(r"([A-Za-z]+)$")


def has_citation(s: str) -> bool:
    return bool(CITATION_BRACKET_RE.search(s) or CITATION_SUPERSCRIPT_RE.search(s)
                or CITATION_AUTHOR_RE.search(s) or CITATION_AUTHOR_NOSPACE_RE.search(s)
                or CITATION_PAREN_NUM_RE.search(s))


def extract_citation_markers(s: str) -> list[str]:
    ms = (CITATION_BRACKET_RE.findall(s) + CITATION_SUPERSCRIPT_RE.findall(s)
          + CITATION_AUTHOR_RE.findall(s) + CITATION_AUTHOR_NOSPACE_RE.findall(s)
          + CITATION_PAREN_NUM_RE.findall(s))
    return [m.strip() for m in ms]


def _expand_ranges(text: str) -> list[int]:
    ids: list[int] = []
    for part in re.split(r"[,;]", text):
        part = part.strip()
        if re.fullmatch(r"\d+", part):
            ids.append(int(part))
        else:
            m = _RANGE_RE.fullmatch(part)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                if hi > lo and hi - lo <= 50:
                    ids.extend(range(lo, hi + 1))
    return ids


def _is_residue(sentence: str, start: int) -> bool:
    m = _WORD_BEFORE_RE.search(sentence[:start])
    return bool(m and m.group(1).lower() in _AA_CODES)


def _superscript_is_gene(sentence: str, m: re.Match) -> bool:
    start = m.start()
    i = start
    while i > 0 and (sentence[i - 1].isalnum() or sentence[i - 1] in "-_"):
        i -= 1
    word = sentence[i:start]
    if not word:
        return False
    if any(c.isdigit() for c in word):
        return True
    if len(word) > 1 and any(c.isupper() for c in word[1:]):
        return True
    return False


def extract_reference_ids(s: str, max_ref: int | None = None) -> list[int]:
    ids: list[int] = []
    for marker in CITATION_BRACKET_RE.findall(s):
        ids.extend(_expand_ranges(marker.strip("[]")))
    for m in CITATION_SUPERSCRIPT_RE.finditer(s):
        if _superscript_is_gene(s, m) or _is_residue(s, m.start()):
            continue
        ids.extend(_expand_ranges(m.group()))
    for marker in CITATION_PAREN_NUM_RE.findall(s):
        ids.extend(_expand_ranges(marker.strip("()")))
    out = sorted(set(ids))
    if max_ref is not None:
        out = [i for i in out if i <= max_ref]
    return out


def _is_noise(sentence: str) -> bool:
    if _DOI_RE.search(sentence):
        return True
    return len(_AUTHOR_AFF_RE.findall(sentence)) >= 2


DROP_INLINE = {"fig", "table-wrap", "graphic", "inline-graphic", "disp-formula"}
NOCLAIM = {"keywords", "references", "glossary", "back_matter", "box"}


def render(el, refmap):
    out = [el.text or ""]
    for c in el:
        if c.tag == "xref":
            if c.get("ref-type") == "bibr":
                ns = [str(refmap[r]) for r in (c.get("rid") or "").split() if r in refmap]
                out.append("[" + ",".join(ns) + "]" if ns else render(c, refmap))
        elif c.tag not in DROP_INLINE:
            out.append(render(c, refmap))
        out.append(c.tail or "")
    return "".join(out)


def _txt(e):
    return clean_text("".join(e.itertext())) if e is not None else None


def parse(path):
    if str(path).endswith(".gz"):
        with gzip.open(path) as f:
            r = ET.parse(f).getroot()
    else:
        r = ET.parse(path).getroot()
    return parse_article(r if r.tag == "article" else r.find(".//article"))


def parse_article(art):
    meta = art.find("front/article-meta")
    refmap = {}
    refs = []
    for i, ref in enumerate(art.iter("ref")):
        refmap.setdefault(ref.get("id"), i + 1)
        pms = [p for p in ref.iter("pub-id") if p.get("pub-id-type") == "pmid"]
        dois = [p for p in ref.iter("pub-id") if p.get("pub-id-type") == "doi"]
        pid = pms[-1] if pms else None
        doi = dois[-1] if dois else None
        ti = next((e for e in ref.iter() if e.tag in ("article-title", "source")), None)
        refs.append({"reference_id": i + 1, "rid": ref.get("id"),
                     "pmid": (_txt(pid) or None) if pid is not None else None,
                     "doi": (_txt(doi) or None) if doi is not None else None,
                     "title": (_txt(ti) or None) if ti is not None else None,
                     "raw": clean_text("".join(ref.itertext()))[:500]})
    sections = []
    paragraphs = []

    def new_sec(title, level, stype=None):
        s = {"section_id": f"sec_{len(sections):03d}", "title": title,
             "section_type": stype or section_type_for_title(title), "page_start": None,
             "heading_level": level, "heading_number": None,
             "parent_section_id": parent_section_id(sections, level), "paragraph_ids": []}
        sections.append(s)
        return s

    def add_par(sec, text, role="body"):
        text = clean_text(text)
        if not text:
            return
        pid = f"p_{len(paragraphs):04d}"
        paragraphs.append({"paragraph_id": pid, "section_id": sec["section_id"], "page": None,
                           "paragraph_index_in_section": len(sec["paragraph_ids"]),
                           "text": text, "sentences": split_sentences(text), "central_sentence": "",
                           "central_sentence_index": None, "role": role, "source_item_index": None,
                           "claim_ids": []})
        sec["paragraph_ids"].append(pid)

    def add_item(sec, c):
        if c.tag == "p":
            add_par(sec, render(c, refmap))
        elif c.tag in ("fig", "table-wrap"):
            cap = c.find("caption")
            if cap is not None:
                add_par(sec, render(cap, refmap), "caption")

    ab = meta.findall("abstract")
    if ab:
        s = new_sec("Abstract", 1, "abstract")
        for p in ab[0].iter("p"):
            add_par(s, render(p, refmap))

    def walk_sec(secel, level):
        s = new_sec(_txt(secel.find("title")) or "Untitled", level)
        for c in secel:
            if c.tag == "sec":
                walk_sec(c, level + 1)
            else:
                add_item(s, c)

    body = art.find("body")
    if body is not None:
        loose = []
        for c in body:
            if c.tag == "sec":
                walk_sec(c, 1)
            elif c.tag == "p":
                loose.append(c)
        if loose:
            s = new_sec("Body", 1)
            for c in loose:
                add_item(s, c)
    authors = []
    for ct in art.iter("contrib"):
        if ct.get("contrib-type") != "author":
            continue
        n = ct.find(".//name")
        if n is not None:
            ge = n.find("given-names")
            sn = _txt(n.find("surname")) or ""
            authors.append((_txt(ge) + " " + sn) if ge is not None else sn)
    authors = authors[:50]
    title = _txt(meta.find("title-group/article-title"))
    stype = {s["section_id"]: s["section_type"] for s in sections}
    claims = []
    for p in paragraphs:
        st = stype.get(p["section_id"], "body")
        if st in NOCLAIM or p["role"] == "caption":
            continue
        for si, snt in enumerate(p["sentences"]):
            if _is_noise(snt) or not has_citation(snt):
                continue
            cid = f"claim_{len(claims):04d}"
            claims.append({"claim_id": cid, "paragraph_id": p["paragraph_id"], "section_id": p["section_id"],
                           "section_type": st, "sentence_index": si, "sentence": snt,
                           "citation_markers": extract_citation_markers(snt),
                           "reference_ids": extract_reference_ids(snt), "claim_function": "body_evidence"})
            p["claim_ids"].append(cid)
    struct = {"paper": {"title": title, "authors": authors, "publication": [], "parser": "jats-pmc",
                        "model_version": None, "source_content_list": None,
                        "structure_method": "xml_adapter_jats_v1", "references": refs},
              "sections": sections, "paragraphs": paragraphs, "tables": [], "table_rows": [],
              "table_contexts": [], "claims": [], "relations": []}
    refsjson = {"topic": title, "references": {str(r["reference_id"]): {"pmid": r["pmid"], "doi": r["doi"],
                                                                          "title": r["title"], "raw": r["raw"],
                                                                          "abstract": None} for r in refs}}
    return struct, claims, refsjson


def serialize(struct, claims, refsjson):
    return {"review_structure_llm.json": json.dumps(struct, ensure_ascii=False, indent=2) + "\n",
            "claims_llm.jsonl": "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in claims),
            "references.json": json.dumps(refsjson, ensure_ascii=False, indent=2) + "\n",
            "relations_llm.jsonl": ""}


def task_id_of(name: str) -> str:
    for ext in (".xml.gz", ".xml"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def write_outputs(files: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, text in files.items():
        dst = os.path.join(out_dir, name)
        tmp = f"{dst}.tmp-{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(text.encode("utf-8"))
        os.replace(tmp, dst)


def _parse_one(job):
    path, out_dir = job
    try:
        write_outputs(serialize(*parse(path)), out_dir)
        return path, None
    except Exception as e:
        return path, type(e).__name__


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if os.path.isdir(a.xml):
        files = sorted(f for f in os.listdir(a.xml) if f.endswith((".xml", ".xml.gz")))
        paths = {task_id_of(f): os.path.join(a.xml, f) for f in files}
    else:
        paths = {task_id_of(os.path.basename(a.xml)): a.xml}
    if a.tasks:
        keep = {line.strip() for line in open(a.tasks) if line.strip()}
        paths = {t: p for t, p in paths.items() if t in keep}
    jobs = [(p, os.path.join(a.out, t)) for t, p in sorted(paths.items())]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            results = pool.map(_parse_one, jobs, chunksize=8)
    else:
        results = [_parse_one(j) for j in jobs]
    failed = [(task_id_of(os.path.basename(p)), err) for p, err in results if err]
    print(json.dumps({"parsed": len(results) - len(failed), "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
