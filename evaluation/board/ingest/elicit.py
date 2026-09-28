"""Converts the Elicit Systematic Review runs in $COMMERCIAL_OUT into fixed-input scorer runs,
mapping its citation markers to PMIDs, and builds their rollouts. Usage: python elicit.py convert
--out <dir> | build --runs <dir>."""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import os
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared as S
from ccbench.adapters.autosurvey import CITE

KEY = "elicit_sr.ref"
KEYS = {"fixinput": KEY}
LABEL = "Elicit Systematic Review (fixed input only)"
BACKBONE = "proprietary (Elicit's own models, not selectable)"
ENTRY_NOTE = "Elicit reads full text where it finds it; the entry set is the task's reference list."


def _gold_refs(task):
    from ccbench import paths
    return paths.gold_refs_path(task)


MARKER = re.compile(r"\{([0-9a-f]{64})_(\d+)\}")
BRACKET = re.compile(r"(?<!\\)\[([^\[\]\n]+?)\]")
AY_ITEM = re.compile(r"^.+,\s*(?:\d{4}[a-z]?|n\.d\.)$")
REF_LINE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")
PMID_TOK = re.compile(r"PMID\s*:?\s*(\d{4,9})", re.I)
PUBMED_ID = re.compile(r"^PUBMED-(\d{4,9})$")


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", html.unescape(t or "").lower())


def norm_doi(d: str) -> str:
    d = (d or "").strip().lower()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", d)


FOLD = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "′": "'", "“": '"', "”": '"',
                      "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", " ": " "})


def fold(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").translate(FOLD)).strip()


def first_author_year(s: str) -> tuple[str, str] | None:
    m = re.match(r"^(.*?)(?: et al\.| & .*?)?,\s*(\d{4}[a-z]?|n\.d\.)$", fold(s))
    return (m.group(1).strip().lower(), m.group(2)) if m else None


def lookup(table: dict, item: str) -> tuple[dict | None, str]:
    if item in table:
        return table[item], "exact"
    f = [r for k, r in table.items() if fold(k) == fold(item)]
    if len(f) == 1:
        return f[0], "fold"
    fa = first_author_year(item)
    g = [r for k, r in table.items() if fa and first_author_year(k) == fa]
    if len(g) == 1:
        return g[0], "first_author_year"
    return None, "none"


def task_src(task: str) -> Path:
    return S.X / "runs" / task


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def search_csv(task: str) -> Path:
    ex = task_src(task) / "exports"
    if (ex / "search.csv.final").exists():
        return ex / "search.csv.final"
    return sorted(ex.glob("search.csv.2*"))[-1]


def extract_csv(task: str) -> Path:
    p = task_src(task) / "exports/extract.csv.postreport"
    if not p.exists():
        raise SystemExit(f"INGEST REFUSED: {task}: no exports/extract.csv.postreport")
    return p


def allowlist(task: str) -> list[str]:
    d = os.environ.get("COMMERCIAL_ALLOWLISTS")
    if not d:
        raise SystemExit("set COMMERCIAL_ALLOWLISTS to the reference-list directory the runs used")
    return [str(x) for x in json.load(open(Path(d) / f"{task}.json"))]


def preflight(task: str) -> tuple[dict, dict]:
    out, st = {}, Counter()
    for fn in ("preflight_search.json",):
        p = task_src(task) / fn
        if not p.exists():
            continue
        st[f"{fn}:present"] = 1
        d = json.load(open(p))
        d = d if isinstance(d, list) else [d]
        for e in d:
            if not isinstance(e, dict):
                continue
            if "error" in e:
                st[f"{fn}:error_entries"] += 1
            for q in e.get("papers") or []:
                old = out.get(q["elicitId"])
                if old and str(old.get("pmid") or "") != str(q.get("pmid") or ""):
                    raise SystemExit(f"INGEST REFUSED: {task}: elicitId {q['elicitId']} has two PMIDs across the preflight files")
                out[q["elicitId"]] = q
    st["preflight_papers"] = len(out)
    return out, dict(st)


def pmid_of_row(pid: str, r: dict, pre: dict, aset: set, by_doi: dict, by_title: dict, st: Counter) -> tuple[str | None, str | None]:
    p = pre.get(pid)
    if p and p.get("pmid") and str(p["pmid"]) in aset:
        return str(p["pmid"]), "preflight_pmid"
    if p and p.get("pmid"):
        st["preflight_pmid_outside_allowlist"] += 1
    m = PUBMED_ID.match(pid or "")
    if m and m.group(1) in aset:
        return m.group(1), "paper_id_PUBMED"
    if m:
        st["paper_id_PUBMED_outside_allowlist"] += 1
    d = norm_doi(r.get("DOI") or (p or {}).get("doi"))
    if d and d in by_doi:
        return by_doi[d], "doi->allowlist_gold_meta"
    t = norm_title(r.get("Title") or (p or {}).get("title"))
    if t and t in by_title:
        return by_title[t], "title->allowlist_gold_meta"
    return None, None


def gold_index(task: str, allow: list[str]) -> tuple[dict, dict]:
    gold = {str(r.get("pmid")): r for r in json.load(open(_gold_refs(task)))["refs"] if r.get("pmid")}
    by_doi = {norm_doi(gold[p].get("doi")): p for p in allow if p in gold and gold[p].get("doi")}
    by_title = {norm_title(gold[p].get("title")): p for p in allow if p in gold and gold[p].get("title")}
    return by_doi, by_title


def paper_table(task: str) -> tuple[dict, dict, dict]:
    ext = read_csv(extract_csv(task))
    if not ext or "Report citation" not in ext[0] or "Included in report" not in ext[0]:
        raise SystemExit(f"INGEST REFUSED: {task}: extract.csv.postreport lacks 'Report citation' / 'Included in report'")
    search_ids = {r["Paper ID"] for r in read_csv(search_csv(task))}
    pre, pst = preflight(task)
    allow = allowlist(task)
    aset = set(allow)
    by_doi, by_title = gold_index(task, allow)
    st = Counter(pst)
    table: dict[str, dict] = {}
    index = {"doi": {}, "title": {}}
    for r in ext:
        pid = r["Paper ID"]
        pm, how = pmid_of_row(pid, r, pre, aset, by_doi, by_title, st)
        rec = {"paper_id": pid, "title": r.get("Title") or "", "doi": r.get("DOI") or "", "pmid": pm, "how": how,
               "full_text": (r.get("Full text retrieved?") or "").strip() or None,
               "authors": r.get("Authors") or "", "year": (r.get("Year") or "").strip()}
        if pm:
            index.setdefault("pmid", {}).setdefault(pm, rec)
        if rec["doi"]:
            index["doi"][norm_doi(rec["doi"])] = rec
        if rec["title"]:
            index["title"][norm_title(rec["title"])] = rec
        if (r.get("Included in report") or "").strip() != "Yes":
            st["rows_not_included"] += 1
            continue
        rc = r["Report citation"].strip()
        st["rows"] += 1
        if pid not in search_ids:
            st["row_not_in_search_csv"] += 1
        st[how or "no_pmid"] += 1
        if rc in table:
            raise SystemExit(f"INGEST REFUSED: {task}: Report citation {rc!r} names two rows ({table[rc]['paper_id']}, {pid})")
        table[rc] = rec
    st["report_citation_column"] = 1
    return table, dict(st), index


def docx_paras(p: Path) -> list[tuple[str | None, str]]:
    x = zipfile.ZipFile(p).read("word/document.xml").decode()
    out = []
    for q in re.findall(r"<w:p[ >].*?</w:p>", x, flags=re.S):
        st = re.search(r'<w:pStyle w:val="([^"]+)"', q)
        out.append((st.group(1) if st else None, html.unescape("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", q)))))
    return out


def unmd(s: str) -> str:
    return fold(re.sub(r"[*_`]", "", s or ""))


def docx_groups(p: Path, first_heading: str) -> list[list[int]]:
    paras = docx_paras(p)
    start = next((i for i, (st, tx) in enumerate(paras) if st and st.startswith("Heading") and unmd(tx) == unmd(first_heading)), None)
    ends = [i for i, (st, tx) in enumerate(paras) if st == "Heading1" and tx.strip() == "References"]
    if start is None or not ends or ends[-1] <= start:
        raise SystemExit(f"INGEST FAILED: {p}: no docx anchor heading {first_heading!r} before References")
    txt = [tx for st, tx in paras[start + 1: ends[-1]] if not (st and st.startswith("Heading"))]
    out = []
    for g in re.findall(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", "\n".join(txt)):
        ns = []
        for tok in re.split(r"\s*,\s*", g):
            m = re.match(r"^(\d+)\s*[–-]\s*(\d+)$", tok)
            ns += list(range(int(m.group(1)), int(m.group(2)) + 1)) if m else [int(tok)]
        out.append(ns)
    return out


def docx_ref_pmids(p: Path, index: dict) -> dict[int, str | None]:
    paras = docx_paras(p)
    k = max(i for i, (st, t) in enumerate(paras) if st == "Heading1" and t.strip() == "References")
    out = {}
    for st, t in paras[k + 1:]:
        m = re.match(r"^\s*(\d+)\.\s*(.*)$", t)
        if not m:
            continue
        e = m.group(2)
        rec = None
        md = re.search(r"doi\.org/(\S+)", e)
        if md:
            rec = index["doi"].get(norm_doi(md.group(1).rstrip(".")))
        if rec is None:
            ne = norm_title(e)
            hit = [r for tt, r in index["title"].items() if len(tt) > 15 and tt in ne]
            rec = hit[0] if len(hit) == 1 else None
        out[int(m.group(1))] = (rec or {}).get("pmid")
    return out


def align(md: list[dict], dx: list[frozenset]) -> list[int | None]:
    n, m = len(md), len(dx)
    NEG = -10 ** 9
    S = [[NEG] * (m + 1) for _ in range(n + 1)]
    B = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        S[i][0], B[i][0] = -i, 1
    for j in range(m + 1):
        S[0][j], B[0][j] = -j, 2
    B[0][0] = 0

    def sc(t, g):
        if t["md_pmids"] is not None:
            return 2 if t["md_pmids"] == g else -1
        return 1 if len(g) == 1 else -1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, b = S[i - 1][j - 1] + sc(md[i - 1], dx[j - 1]), 0
            if S[i - 1][j] - 1 > best:
                best, b = S[i - 1][j] - 1, 1
            if S[i][j - 1] - 1 > best:
                best, b = S[i][j - 1] - 1, 2
            S[i][j], B[i][j] = best, b
    res: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        b = B[i][j]
        if i > 0 and j > 0 and b == 0:
            res[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or b == 1):
            i -= 1
        else:
            j -= 1
    return res


PIPE = re.compile(r"(?<!\\)\|")


def md_cells(line: str) -> list[tuple[int, int]]:
    pos = [m.start() for m in PIPE.finditer(line)]
    return [(pos[i] + 1, pos[i + 1]) for i in range(len(pos) - 1)]


def tokens(body: str, table: dict) -> list[dict]:
    toks = []
    lines = body.splitlines()
    first_col, ti, ri = None, -1, 0
    for ln, line in enumerate(lines):
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            spans = md_cells(line)
            nxt = lines[ln + 1] if ln + 1 < len(lines) else ""
            header = nxt.lstrip().startswith("|") and set(nxt.strip()) <= set("|-: ")
            if set(line.strip()) <= set("|-: "):
                continue
            if header:
                first_col, ti, ri = cells[0], ti + 1, 0
            else:
                ri += 1
            seen: Counter = Counter()
            for mm in MARKER.finditer(line):
                ci = next((i for i, (a, b) in enumerate(spans) if a <= mm.start() < b), None)
                mi = seen[ci]
                seen[ci] += 1
                if header:
                    kind, items = "cell_header", []
                elif first_col == "Study":
                    kind, items = "cell_study", [MARKER.sub("", cells[0]).strip()]
                else:
                    kind, items = "cell_other", []
                toks.append({"kind": kind, "line": ln, "span": mm.span(), "items": items, "table_first_col": first_col,
                             "marker_k": int(mm.group(2)), "addr": (ti, ri, ci, mi)})
            continue
        if MARKER.search(line):
            raise SystemExit(f"INGEST FAILED: line {ln}: a cell marker outside a table (no rule maps it)")
        for mm in BRACKET.finditer(line):
            items = [x.strip() for x in mm.group(1).split(";")]
            if all(AY_ITEM.match(x) for x in items):
                toks.append({"kind": "prose", "line": ln, "span": mm.span(), "items": items})
    for t in toks:
        hits = [lookup(table, it) for it in t["items"]]
        t["how"] = [h for _, h in hits]
        recs = [r for r, _ in hits]
        t["unmapped_items"] = [it for it, r in zip(t["items"], recs) if r is None or not r.get("pmid")]
        t["md_pmids"] = frozenset(r["pmid"] for r in recs if r and r.get("pmid")) if t["items"] and not t["unmapped_items"] else None
    return toks


def docx_tables(p: Path) -> list[list[list[str]]]:
    x = zipfile.ZipFile(p).read("word/document.xml").decode()
    out = []
    for tb in re.findall(r"<w:tbl[ >].*?</w:tbl>", x, flags=re.S):
        if len(re.findall(r"<w:tbl[ >]", tb)) != 1:
            raise SystemExit(f"INGEST FAILED: {p}: a nested docx table")
        out.append([[html.unescape("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", tc)))
                     for tc in re.findall(r"<w:tc[ >].*?</w:tc>", tr, flags=re.S)]
                    for tr in re.findall(r"<w:tr[ >].*?</w:tr>", tb, flags=re.S)])
    return out


def md_tables(body: str) -> list[list[str]]:
    lines, out, i = body.splitlines(), [], 0
    while i < len(lines):
        if (lines[i].lstrip().startswith("|") and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("|")
                and set(lines[i + 1].strip()) <= set("|-: ")):
            rows, j = [lines[i]], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(lines[j])
                j += 1
            out.append(rows)
            i = j
        else:
            i += 1
    return out


def _seg_re(s: str) -> str:
    s = fold(s.replace("<br>", " ").replace("<br/>", " "))
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            out.append(re.escape(s[i + 1]))
            i += 2
            continue
        if ch in "*_`":
            out.append(f"(?:{re.escape(ch)})?")
        elif ch.isspace():
            out.append(r"\s*")
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)


def cell_groups(md_cell: str, dx_cell: str) -> list[list[int]] | None:
    segs = MARKER.split(md_cell)[0::3]
    G = r"\s*\[(\d+(?:\s*[,–-]\s*\d+)*)\]\s*"

    def nums(g: str) -> list[int]:
        ns = []
        for tok in re.split(r"\s*,\s*", g):
            r = re.match(r"^(\d+)\s*[–-]\s*(\d+)$", tok)
            ns += list(range(int(r.group(1)), int(r.group(2)) + 1)) if r else [int(tok)]
        return ns
    m = re.fullmatch(r"\s*" + G.join(_seg_re(sg) for sg in segs) + r"\s*", fold(dx_cell), flags=re.S)
    if m:
        return [nums(g) for g in m.groups()]
    runs, coll = [], [segs[0]]
    for i in range(1, len(segs)):
        if i > 1 and segs[i - 1].strip() == "":
            runs[-1] += 1
            coll[-1] = segs[i]
        else:
            runs.append(1)
            coll.append(segs[i])
    if all(r == 1 for r in runs):
        return None
    m = re.fullmatch(r"\s*" + G.join(_seg_re(sg) for sg in coll) + r"\s*", fold(dx_cell), flags=re.S)
    if not m or len(m.groups()) != len(runs):
        return None
    out = []
    for g, j in zip(m.groups(), runs):
        ns = nums(g)
        if j == 1:
            out.append(ns)
        elif len(ns) == j:
            out.extend([n] for n in ns)
        else:
            return None
    return out


def docx_struct(dxp: Path, body: str, st: Counter) -> dict | None:
    D, M = docx_tables(dxp), md_tables(body)
    if len(D) != len(M) or any(len(d) != len(m) for d, m in zip(D, M)):
        st["struct_shape_mismatch_tables"] += 1
        return None
    res = {}
    for ti, (dt_, mt) in enumerate(zip(D, M)):
        for ri, (dr, line) in enumerate(zip(dt_, mt)):
            spans = md_cells(line)
            if len(spans) != len(dr):
                st["struct_row_cell_count_mismatch"] += 1
                continue
            gs = [cell_groups(line[a:b].strip(), dc) for (a, b), dc in zip(spans, dr)]
            row_ok = all(g is not None for g in gs)
            st["struct_rows_ok" if row_ok else "struct_rows_mismatch"] += 1
            for ci, ((a, b), g) in enumerate(zip(spans, gs)):
                if not MARKER.search(line[a:b]):
                    continue
                st["struct_cells_ok" if row_ok else "struct_cells_text_mismatch"] += 1
                res[(ti, ri, ci)] = g if row_ok else None
    return res


def rewrite(task: str, body: str, table: dict, index: dict) -> tuple[str, list[dict], list[dict], dict]:
    toks = tokens(body, table)
    st = Counter()
    dxp = task_src(task) / "exports/report.docx.final"
    struct, refp = None, {}
    if dxp.exists():
        refp = docx_ref_pmids(dxp, index)
        mh = re.search(r"^#{1,6}\s+(.*?)\s*#*\s*$", body, re.M)
        groups = docx_groups(dxp, mh.group(1) if mh else "Results")
        dsets = [frozenset(refp.get(n) or f"?{n}" for n in g) for g in groups]
        al = align(toks, dsets)
        st["docx_groups"] = len(groups)
        st["docx_ref_entries"] = len(refp)
        st["docx_ref_entries_without_pmid"] = sum(1 for v in refp.values() if not v)
        for t, j in zip(toks, al):
            if t["md_pmids"] is None:
                continue
            if j is None:
                st["xcheck_md_token_unaligned"] += 1
            elif t["md_pmids"] == dsets[j]:
                st["xcheck_agree"] += 1
            else:
                st["xcheck_disagree"] += 1
        st["xcheck_docx_groups_unaligned"] = len(groups) - sum(1 for j in al if j is not None)
        struct = docx_struct(dxp, body, st)
    else:
        st["docx_groups"] = -1
    for t in toks:
        t["struct_pmids"], t["struct_why"] = None, None
        if "addr" not in t:
            continue
        ti, ri, ci, mi = t["addr"]
        if struct is None:
            t["struct_why"] = "no docx" if not dxp.exists() else "docx/md table shapes differ"
            continue
        g = struct.get((ti, ri, ci))
        if g is None:
            t["struct_why"] = "docx cell is not the md cell (text / marker-count mismatch)"
            continue
        if mi >= len(g):
            t["struct_why"] = "more md markers than docx groups in the cell"
            continue
        pm = [refp.get(n) for n in g[mi]]
        if not pm or any(p is None for p in pm):
            t["struct_why"] = f"docx reference(s) {g[mi]} without a PMID"
            continue
        t["struct_pmids"] = frozenset(pm)
        if t["md_pmids"] is not None:
            st["struct_xcheck_agree" if t["md_pmids"] == t["struct_pmids"] else "struct_xcheck_disagree"] += 1
    by_pmid: dict[str, dict] = {}
    for rec in table.values():
        if rec.get("pmid"):
            by_pmid.setdefault(rec["pmid"], rec)
    allrows = index.get("pmid", {})
    num: dict[str, int] = {}
    refs: list[dict] = []
    log: list[dict] = []

    def n_of(key: str, pmid: str | None, title: str) -> int:
        if key not in num:
            num[key] = len(num) + 1
            refs.append({"n": num[key], "key": key, "pmid": pmid, "title": title})
        return num[key]

    def surname_year_ok(p: str, cell: str) -> bool:
        rr = allrows.get(p) or {}
        fa = fold(rr.get("authors") or "").split(",")[0].strip()
        sur = re.sub(r"[^\w-]", "", fa.split()[-1]).lower() if fa.split() else ""
        c = unmd(cell).lower()
        return bool(sur) and len(sur) >= 2 and sur in c and bool(rr.get("year")) and rr["year"] in c

    repl: dict[tuple[int, tuple[int, int]], str] = {}
    for t in toks:
        st[f"{t['kind']}_tokens"] += 1
        ns = []
        named = t["kind"] in ("prose", "cell_study") and not t["unmapped_items"]
        if named:
            for it in t["items"]:
                st[f"{t['kind']}_items"] += 1
                rec, how = lookup(table, it)
                st[f"match_{how}"] += 1
                n = n_of(rec["pmid"], rec["pmid"], rec["title"])
                if n not in ns:
                    ns.append(n)
        elif t["kind"] == "prose":
            for it in t["items"]:
                st["prose_items"] += 1
                rec, how = lookup(table, it)
                st[f"match_{how}"] += 1
                if rec is None or not rec.get("pmid"):
                    st["prose_items_unmapped"] += 1
                    log.append({"task": task, "kind": "prose", "line": t["line"] + 1, "item": it,
                                "why": "no extraction row with this Report citation" if rec is None else "row has no PMID"})
                    n = n_of(f"UNMAPPED::{it}", None, "")
                else:
                    n = n_of(rec["pmid"], rec["pmid"], rec["title"])
                if n not in ns:
                    ns.append(n)
        else:
            if t["kind"] == "cell_study":
                st["cell_study_items"] += 1
                st["match_none"] += 1
            src = t["struct_pmids"]
            if src is None:
                st[f"{t['kind']}_unmapped"] += 1
                log.append({"task": task, "kind": t["kind"], "line": t["line"] + 1, "marker_k": t["marker_k"],
                            "table_first_col": t["table_first_col"], "addr": list(t["addr"]),
                            **({"item": t["items"][0]} if t["items"] else {}), "why": t["struct_why"]})
                ns.append(n_of(f"UNMAPPED::cell {t['marker_k']} ({t['table_first_col']} table)", None, ""))
            else:
                st[f"{t['kind']}_via_docx_struct"] += 1
                if t["kind"] == "cell_study":
                    ok = len(src) == 1 and surname_year_ok(next(iter(src)), t["items"][0])
                    st["cell_study_struct_surname_year_" + ("ok" if ok else "not_ok")] += 1
                for p in sorted(src):
                    ns.append(n_of(p, p, (by_pmid.get(p) or allrows.get(p) or {}).get("title", "")))
        repl[(t["line"], t["span"])] = "[" + ", ".join(map(str, ns)) + "]"
    lines = body.splitlines()
    for ln in sorted({k[0] for k in repl}):
        spans = sorted((sp for (l, sp) in repl if l == ln), reverse=True)
        s = lines[ln]
        for sp in spans:
            s = s[:sp[0]] + repl[(ln, sp)] + s[sp[1]:]
        lines[ln] = s
    out = "\n".join(lines) + "\n"
    st["markers_in_body"] = len(MARKER.findall(body))
    st["markers_tokenised"] = st["cell_study_tokens"] + st["cell_other_tokens"] + st["cell_header_tokens"]
    st["markers_left_in_output"] = len(MARKER.findall(out))
    if st["markers_left_in_output"] or st["markers_tokenised"] != st["markers_in_body"]:
        raise SystemExit(f"INGEST FAILED: {task}: {st['markers_in_body']} markers in the body, {st['markers_tokenised']} tokenised, "
                         f"{st['markers_left_in_output']} left in the output")
    st["refs"] = len(refs)
    st["refs_with_pmid"] = sum(1 for r in refs if r["pmid"])
    st["unmapped_log_entries"] = len(log)
    st["nonciting_brackets"] = sum(1 for l in body.splitlines() if not l.lstrip().startswith("|")
                                   for mm in BRACKET.finditer(l)
                                   if not all(AY_ITEM.match(x.strip()) for x in mm.group(1).split(";")))
    return out, refs, log, dict(st)


def references_block(refs: list[dict]) -> str:
    lines = ["", "## References", ""]
    for r in refs:
        if r["pmid"]:
            title = re.sub(r"\s+", " ", r["title"] or "").strip().rstrip(".")
            lines.append(f"[{r['n']}] {title}. PMID: {r['pmid']}.")
        else:
            lines.append(f"[{r['n']}] {r['key'].split('::', 1)[1]} (unmapped: no extraction row or no identifier)")
    return "\n".join(lines) + "\n"


def parse_refs(text: str) -> dict[str, list[str]]:
    lines = text.splitlines()
    start = max(i for i, l in enumerate(lines) if l.strip() == "## References") + 1
    out: dict[str, list[str]] = {}
    for l in lines[start:]:
        m = REF_LINE.match(l)
        if m:
            pm = PMID_TOK.findall(m.group(2))
            if pm:
                out[m.group(1)] = [pm[-1]]
    return out


def body_of(text: str) -> str:
    lines = text.splitlines()
    cut = max(i for i, l in enumerate(lines) if l.strip() == "## References")
    return "\n".join(lines[:cut])


def utc(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def completion(task: str) -> tuple[dt.datetime, dt.datetime, int]:
    src = task_src(task)
    t0 = utc(json.load(open(src / "create_response.json"))["_t"])
    n = 0
    for l in open(src / "polls.jsonl"):
        n += 1
        d = json.loads(l)
        if (d.get("body") or {}).get("status") == "completed":
            return t0, utc(d["_t"]), n
    raise SystemExit(f"INGEST FAILED: {task}: no poll with status 'completed'")


def convert(out: Path) -> int:
    tasks = S.run_tasks()
    summ = {"schema": "elicit_convert/2", "source": str(S.X), "key": KEY, "label": LABEL, "backbone": BACKBONE, "entry_note": ENTRY_NOTE,
            "source_sha256": S.source_hashes(("audit.json",)), "per_task": {}}
    arm = out / "fixinput"
    (arm / "_arm").mkdir(parents=True)
    all_log, st_lines = [], []
    for task in tasks:
        src = task_src(task)
        fin = json.load(open(src / "final.json"))
        if fin.get("status") != "completed":
            raise SystemExit(f"INGEST FAILED: {task}: final.json status {fin.get('status')}")
        sid_file = (src / "session_id.txt").read_text().strip() if (src / "session_id.txt").exists() else None
        if sid_file is not None and sid_file != fin["sessionId"]:
            raise SystemExit(f"INGEST FAILED: {task}: session_id.txt {sid_file} != final.json sessionId {fin['sessionId']}")
        body = (src / "report_body.md").read_text()
        rb = (((fin.get("data") or {}).get("report") or {}).get("result") or {}).get("reportBody")
        if rb is not None and body.strip() != rb.strip():
            raise SystemExit(f"INGEST FAILED: {task}: report_body.md differs from final.json reportBody")
        if CITE.search(MARKER.sub("", body)):
            raise SystemExit(f"INGEST REFUSED: {task}: the original body already contains a numeric [n] mark (collision)")
        table, pst, index = paper_table(task)
        newbody, refs, log, rst = rewrite(task, body, table, index)
        td = arm / task
        td.mkdir()
        report_text = newbody + references_block(refs)
        rm = parse_refs(report_text)
        want = {str(r["n"]): [r["pmid"]] for r in refs if r["pmid"]}
        if rm != want:
            raise SystemExit(f"INGEST FAILED: {task}: the References block does not parse back ({len(rm)} vs {len(want)} entries)")
        unmapped_n = {str(r["n"]) for r in refs if not r["pmid"]}
        bad = [mk for mk in CITE.findall(body_of(report_text))
               if not all(tok.strip() in rm or tok.strip() in unmapped_n for tok in re.split(r"[,;]", mk) if tok.strip())]
        if bad:
            raise SystemExit(f"INGEST FAILED: {task}: {len(bad)} numeric marks do not resolve through the written References")
        (td / "report.md").write_text(report_text)
        (td / "report_body.orig.md").write_text(body)
        (td / "refs_map.json").write_text(json.dumps(refs, indent=1))
        (td / "allowlist.json").write_text(json.dumps(allowlist(task)))
        req = json.load(open(src / "request.json"))
        pre_map, _ = preflight(task)
        pre = {str(p["pmid"]) for p in pre_map.values() if p.get("pmid")}
        t0, t1, npoll = completion(task)
        allow = allowlist(task)
        q_union = []
        with open(td / "search_calls.jsonl", "w") as f:
            for s in req["searches"]:
                qp = re.findall(r"(\d{4,9})\[pmid\]", s["query"])
                q_union += qp
                ret = [p for p in qp if p in pre] if pre else qp
                f.write(json.dumps({"ts": iso(t0), "task": task, "route": "s2_search",
                                    "params": {"query": s["query"], "k": s.get("maxResults"), "native_route": "elicit_pubmed_keyword",
                                               "corpus": s.get("corpus")},
                                    "n_returned": len(ret), "returned": [{"pmid": p} for p in ret], "latency_ms": None,
                                    "returned_source": "preflight_search.json" if pre else "query PMIDs (no preflight)",
                                    "ts_source": "create_response._t (the gather stage runs server-side at session start)"}) + "\n")
        if sorted(set(q_union)) != sorted(set(allow)):
            raise SystemExit(f"INGEST FAILED: {task}: the gather queries' PMIDs are not exactly the allowlist")
        ft = Counter(r.get("full_text") or "n/a" for r in table.values())
        cov = {"session_id": fin["sessionId"], "created_utc": iso(t0), "completed_utc": iso(t1),
               "wall_s": round((t1 - t0).total_seconds(), 1), "polls_read_to_completion": npoll,
               "included_full_text_retrieved": dict(ft)}
        call = {"seq": 0, "t_req": t0.timestamp(), "t_done": t1.timestamp(), "wall_ms": int(round((t1 - t0).total_seconds() * 1000)),
                "status": 200, "path": "POST /api/v2/sessions/systematic-reviews (Elicit server-side pipeline)",
                "model": "elicit (own models, not selectable)", "requested_model": None, "finish_reason": fin.get("status"),
                "usage": {"prompt_tokens": None, "completion_tokens": None}, "cost_usd": None,
                "prompt": req["researchQuestion"], "completion": body,
                "params": {"session_id": fin["sessionId"], "extraction": req.get("extraction"), "screening": "none",
                           "entry_note": ENTRY_NOTE}}
        (td / "_calls.jsonl").write_text(json.dumps(call) + "\n")
        (td / "pool_health.json").write_text(json.dumps({"ok": True, "stats_fingerprint": None,
                                                         "source": "Elicit's own PubMed index; entry = the reference list"}))
        st_lines.append({"task": task, "status": "ok", "wall_s": cov["wall_s"], "session_id": fin["sessionId"]})
        prov = {"source_dir": str(src), "session_id": fin["sessionId"], "report_body_sha256": S.sha256(src / "report_body.md"),
                "extract": "exports/extract.csv.postreport",
                "extract_sha256": S.sha256(extract_csv(task)), "search_csv": search_csv(task).name,
                "docx": "exports/report.docx.final" if (src / "exports/report.docx.final").exists() else None,
                "covariates": cov, "entry_note": ENTRY_NOTE, "paper_table": pst, "rewrite": rst, "n_unmapped": len(log)}
        (td / "provenance.json").write_text(json.dumps(prov, indent=1))
        all_log += log
        summ["per_task"][task] = prov
        print(f"{task[6:]:12s} rows {pst.get('rows'):3d} pmid "
              f"{ {k: v for k, v in pst.items() if '->' in k or k in ('preflight_pmid', 'paper_id_PUBMED', 'no_pmid')} } "
              f"tok {rst.get('prose_tokens', 0)}p/{rst.get('cell_study_tokens', 0)}s/{rst.get('cell_other_tokens', 0)}o "
              f"unm p{rst.get('prose_items_unmapped', 0)} s{rst.get('cell_study_unmapped', 0)} o{rst.get('cell_other_unmapped', 0)} h{rst.get('cell_header_unmapped', 0)} "
              f"docx agr {rst.get('xcheck_agree', 0)} dis {rst.get('xcheck_disagree', 0)}", flush=True)
    S.write_status(arm, st_lines)
    with open(out / "map_log.jsonl", "w") as f:
        for r in all_log:
            f.write(json.dumps(r) + "\n")
    summ["unmapped_total"] = len(all_log)
    agg = Counter()
    for p in summ["per_task"].values():
        for k, v in p["rewrite"].items():
            if isinstance(v, int) and v >= 0:
                agg[k] += v
        for k, v in p["paper_table"].items():
            if isinstance(v, int):
                agg["rows:" + k] += v
    summ["totals"] = dict(agg)
    S.write_tasks(out, tasks)
    (out / "convert_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"unmapped citation items/markers: {len(all_log)} (log {out / 'map_log.jsonl'})")
    return 0


def adapt(task: str, task_dir: Path, key: str, cond: str):
    from ccbench.adapters import common
    from ccbench.ingest import logs
    status = logs.load_status(task_dir.parent).get(task, {})
    if status.get("status") == "failed":
        return common.bot(key, task, task_dir, "no_report: " + str(status.get("note") or "failed"))
    rp = task_dir / "report.md"
    if not rp.exists():
        return common.bot(key, task, task_dir, "no_report: report.md missing")
    text = rp.read_text(errors="replace")
    if not text.strip():
        return common.bot(key, task, task_dir, "no_report: report.md empty")
    refmap = parse_refs(text)
    report = common.report_from_markdown(text, CITE, common.numeric_resolver(refmap))
    outline = common.outline_from_markdown(body_of(text))
    papers = [str(x) for x in json.load(open(task_dir / "allowlist.json"))]
    prov = json.load(open(task_dir / "provenance.json"))
    cov = prov["covariates"]
    return common.assemble(key, task, task_dir, papers=papers, outline=outline, report=report, graph=None, final_ok=True,
                           bot_reason=None, extra_meta={"n_reference_entries": len(refmap),
                                                        "adapter": "evaluation/board/ingest/elicit.py:adapt", "draft_is_final": True,
                                                        "backbone": BACKBONE, "label": LABEL, "entry_note": ENTRY_NOTE,
                                                        "session_id": cov["session_id"],
                                                        "unmapped_citations": prov["n_unmapped"]})


def main(argv=None) -> int:
    return S.main(argv, __doc__, convert, adapt, KEYS)


if __name__ == "__main__":
    raise SystemExit(main())
