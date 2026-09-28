"""Graph-derived unit readouts through the observation channel: retrieval, synthesis, reasoning
relations, organisation, writing and form."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ccbench.gt import graphs, years
from ccbench.gt.units import Units
from ccbench.model import Rollout
from ccbench.readouts.channel import Induced

Q, S = "quality", "style"


@dataclass
class UnitScore:
    name: str
    stage: str
    family: str
    stratum: str
    criterion: str
    direction: str
    value: float | None
    n_units: int
    note: str = ""


def _frac(num, den) -> float | None:
    return None if den == 0 else num / den


def _sc(name, stage, family, stratum, crit, direction, num, den, note="") -> UnitScore:
    return UnitScore(name, stage, family, stratum, crit, direction, _frac(num, den), den, note)


def retrieval(ro: Rollout, g: graphs.GStar, u: Units) -> list[UnitScore]:
    out: list[UnitScore] = []
    provided = ro.meta.get("provenance_tier") == "reference_derived" or ro.system == "lira"
    P = set(ro.papers)
    reach = g.reachable_pmids(ro.context.cutoff)
    w = g.pmid_weights()

    def wrec(S: set[str]) -> tuple[float, int]:
        S = S & reach
        return (sum(w.get(p, 1) for p in S & P), sum(w.get(p, 1) for p in S)) if S else (0, 0)

    for stratum, subset in (("hub", u.hubs), ("tail", u.tail), ("recent", u.recent), ("classic", u.classic)):
        n, d = wrec(subset)
        s = _sc(f"ret_{stratum}_recall", "retrieval", "papers", stratum, "coverage", Q, n, d)
        if provided:
            s.value, s.note = None, "provided"
        out.append(s)
    comms = [c & reach for c in u.communities if c & reach]
    hit = [c for c in comms if c & P]
    s = _sc("ret_community_breadth", "retrieval", "communities", "all", "coverage", Q, len(hit), len(comms))
    depth = float(np.mean([len(c & P) / len(c) for c in hit])) if hit else None
    s2 = UnitScore("ret_community_depth", "retrieval", "communities", "all", "depth", S, depth, len(hit))
    if provided:
        s.value = s2.value = None
        s.note = s2.note = "provided"
    out += [s, s2]
    searches = [e for e in ro.events if e.kind == "search"]
    hubs = u.hubs & reach
    for b in (5, 10, 20):
        if provided or not searches or not hubs:
            out.append(UnitScore(f"ret_hub_recall@{b}", "retrieval", "trajectory", f"b{b}", "coverage", Q, None, len(hubs), "provided" if provided else "no_search_events"))
            continue
        seen: set[str] = set()
        for e in searches[:b]:
            seen.update(e.returned_ids)
        out.append(_sc(f"ret_hub_recall@{b}", "retrieval", "trajectory", f"b{b}", "coverage", Q, sum(w.get(p, 1) for p in seen & hubs), sum(w.get(p, 1) for p in hubs)))
    return out


def synthesis(ind: Induced, g: graphs.GStar, u: Units) -> list[UnitScore]:
    out: list[UnitScore] = []
    m = ind.claim_match
    req = u.claims_by_tier.get("required", [])
    adm = u.claims_by_tier.get("admissible", [])
    forb = u.claims_by_tier.get("forbidden", [])
    out.append(_sc("syn_required_cov", "synthesis", "claims", "required", "coverage", Q, sum(c in m for c in req), len(req)))
    out.append(_sc("syn_admissible_cov", "synthesis", "claims", "admissible", "coverage", Q, sum(c in m for c in adm), len(adm)))
    out.append(_sc("syn_forbidden_ok", "synthesis", "claims", "forbidden", "validity", Q, sum(c not in m for c in forb), len(forb)))
    out.append(_sc("syn_grounded_cov", "synthesis", "claims", "well_grounded", "coverage", Q, sum(c in m for c in u.grounded_hi), len(u.grounded_hi)))
    out.append(_sc("syn_weak_claim_cov", "synthesis", "claims", "weakly_grounded", "coverage", Q, sum(c in m for c in u.grounded_lo), len(u.grounded_lo), "reported only"))
    out.append(_sc("syn_central_cov", "synthesis", "claims", "central", "coverage", Q, sum(c in m for c in u.central), len(u.central)))
    ms_matched = [(c, ps) for c, ps in u.multi_source.items() if c in m]
    ok = sum(1 for c, ps in ms_matched if len(ind.sent_cites.get(m[c][0], set()) & ps) >= 2)
    out.append(_sc("syn_multisource_integration", "synthesis", "bipartite", "multi_source", "integration", Q, ok, len(ms_matched)))
    reuse_ok = 0
    for p, cl in u.reused_papers.items():
        sents = {m[c][0] for c in cl if c in m and p in ind.sent_cites.get(m[c][0], set())}
        reuse_ok += int(len(sents) >= 2)
    out.append(_sc("syn_paper_reuse", "synthesis", "bipartite", "reused", "integration", Q, reuse_ok, len(u.reused_papers)))
    both = [(a, b) for a, b in u.hard_negatives if a in m and b in m]
    bad = sum(1 for a, b in both if ind.coloc(a, b) == "strong")
    out.append(_sc("syn_hardneg_ok", "synthesis", "hard_negatives", "all", "validity", Q, len(both) - bad, len(both)))
    return out


def reasoning(ind: Induced, g: graphs.GStar, u: Units) -> list[UnitScore]:
    out: list[UnitScore] = []
    for typ, edges in sorted(u.edges_by_type.items()):
        if typ in ("", "synthesis"):
            continue
        rep = sum(1 for a, b in edges if ind.linked(a, b))
        out.append(_sc(f"rel_{typ}_cov", "reasoning", "edges", typ, "coverage", Q, rep, len(edges)))
    for dist, edges in sorted(u.edges_by_distance.items()):
        if dist == "unknown":
            continue
        rep = sum(1 for a, b in edges if ind.linked(a, b))
        out.append(_sc(f"rel_{dist}_cov", "reasoning", "edges_distance", dist, "coverage", Q, rep, len(edges)))
    for name, paths in u.motifs.items():
        rep = sum(1 for p in paths if all(ind.linked(p[i], p[i + 1]) for i in range(len(p) - 1)))
        out.append(_sc(f"motif_{name}_cov", "reasoning", "motifs", name, "coverage", Q, rep, len(paths)))
    return out


def organisation(ro: Rollout, ind: Induced, g: graphs.GStar, u: Units) -> list[UnitScore]:
    out: list[UnitScore] = []
    by_role: dict[str, list[dict]] = {}
    for s in u.top_sections:
        by_role.setdefault(s["role"], []).append(s)
    for role, secs in sorted(by_role.items()):
        out.append(_sc(f"org_section_match_{role}", "organisation", "sections", role, "coverage", Q, sum(s["id"] in ind.section_match for s in secs), len(secs)))
    matched = [s for s in u.top_sections if s["id"] in ind.section_match]
    if len(matched) >= 2:
        conc = disc = 0
        for i in range(len(matched)):
            for j in range(i + 1, len(matched)):
                oa = ind.section_order.get(ind.section_match[matched[i]["id"]], 0) - ind.section_order.get(ind.section_match[matched[j]["id"]], 0)
                ob = matched[i]["order"] - matched[j]["order"]
                if oa * ob > 0:
                    conc += 1
                elif oa * ob < 0:
                    disc += 1
        out.append(_sc("org_order", "organisation", "sections", "all", "order", Q, conc, conc + disc))
    else:
        out.append(UnitScore("org_order", "organisation", "sections", "all", "order", Q, None, len(matched), "fewer than 2 matched sections"))
    for dist, edges in sorted(u.edges_by_distance.items()):
        if dist == "unknown":
            continue
        both = [(a, b) for a, b in edges if a in ind.claim_match and b in ind.claim_match]
        if dist == "within":
            ok = sum(1 for a, b in both if ind.coloc(a, b) is not None)
        else:
            ok = sum(1 for a, b in both if ind.coloc(a, b) != "strong")
        out.append(_sc(f"org_placement_{dist}", "organisation", "placement", dist, "placement", Q, ok, len(both)))
    alloc = []
    for s in matched:
        sys_sec = ind.section_match[s["id"]]
        cited_here = set().union(*(ind.sent_cites.get(sid, set()) for sid, sec in ind.sent_section.items() if sec == sys_sec)) if ind.sent_section else set()
        if s["papers"]:
            alloc.append(len(cited_here & set(s["papers"])) / len(s["papers"]))
    out.append(UnitScore("org_allocation", "organisation", "allocation", "matched", "coverage", Q, float(np.mean(alloc)) if alloc else None, len(alloc)))
    hum_n = len(u.top_sections)
    levels = [sec.get("level", 1) for sec in ro.report.sections if sec.get("title")] if ro.report else []
    sys_n = sum(1 for l in levels if l == min(levels)) if levels else 0
    out.append(UnitScore("org_size_fit", "organisation", "sections", "all", "size", S, (1 - min(1.0, abs(math.log(sys_n / hum_n))) if sys_n and hum_n else None), hum_n))
    return out


def writing(ro: Rollout, ind: Induced, g: graphs.GStar, u: Units) -> list[UnitScore]:
    out: list[UnitScore] = []
    if not ro.report:
        return out
    P = set(ro.papers)
    bib = set(ro.report.bibliography)
    out.append(_sc("wr_cite_fidelity", "writing", "citations", "all", "validity", Q, len(bib & P) if P else 0, len(bib)))
    post = known = 0
    for p in bib:
        r = years.is_post_cutoff(p, ro.context.cutoff)
        if r is None:
            continue
        known += 1
        post += int(r)
    out.append(_sc("wr_post_cutoff_ok", "writing", "citations", "all", "validity", Q, known - post, known))
    dens = []
    hum_claims_by_sec = {s["id"]: s for s in u.top_sections}
    for hid, sysid in ind.section_match.items():
        sids = [sid for sid, sec in ind.sent_section.items() if sec == sysid]
        if not sids:
            continue
        sys_d = sum(len(ind.sent_cites.get(sid, ())) for sid in sids) / len(sids)
        hs = hum_claims_by_sec.get(hid)
        if hs is None:
            continue
        hum_d = sum(len(g.claims[[c.id for c in g.claims].index(c)].pmids) for c in hs["claims"]) / max(1, len(hs["claims"])) if hs["claims"] else 0
        if sys_d > 0 and hum_d > 0:
            dens.append(1 - min(1.0, abs(math.log(sys_d / hum_d))))
    out.append(UnitScore("wr_density_fit", "writing", "citations", "matched_sections", "density", S, float(np.mean(dens)) if dens else None, len(dens)))
    nm = ind.n_matched_sentences
    distinct = len({v[0] for v in ind.sent_match.values()})
    out.append(UnitScore("wr_restatement", "writing", "restatement", "all", "redundancy", S, (1 - (nm - distinct) / nm) if nm else None, nm))
    tw = ro.context.target_words or 0
    out.append(UnitScore("wr_length_fit", "writing", "length", "all", "length", S, (1 - min(1.0, abs(math.log(ro.report.words / tw)))) if tw and ro.report.words else None, 1))
    mm = [(c, sid) for c, (sid, _) in ind.claim_match.items()]
    pm = {c.id: set(c.pmids) for c in g.claims}
    withev = [(c, sid) for c, sid in mm if pm.get(c)]
    ok = sum(1 for c, sid in withev if ind.sent_cites.get(sid, set()) & pm[c])
    out.append(_sc("wr_evidence_agreement", "writing", "citations", "matched_claims", "grounding", Q, ok, len(withev)))
    return out


def form(ind: Induced, g: graphs.GStar, u: Units) -> list[UnitScore]:
    out: list[UnitScore] = []
    m = list(ind.claim_match)
    n = len(m)
    if n >= 2:
        links = sum(1 for k, v in ind.pair_cos.items() if ind.edge_tau is not None and v >= ind.edge_tau)
        possible = len(ind.pair_cos)
        out.append(UnitScore("form_link_density", "form", "graph", "all", "density", S, links / possible if possible else None, possible))
        cross = [(a, b) for a, b in u.edges_by_distance.get("distant", []) if ind.linked(a, b)]
        allrep = [(a, b) for lst in u.edges_by_type.values() for a, b in lst if ind.linked(a, b)]
        out.append(UnitScore("form_cross_section_fraction", "form", "graph", "all", "shape", S, len(cross) / len(allrep) if allrep else None, len(allrep)))
    cited = set().union(*ind.sent_cites.values()) if ind.sent_cites else set()
    r = (n / len(cited)) if cited else None
    out.append(UnitScore("form_claims_per_paper", "form", "graph", "all", "shape", S, (r / (1 + r)) if r is not None else None, len(cited)))
    return out


def all_unit_readouts(ro: Rollout, ind: Induced, g: graphs.GStar, u: Units) -> list[UnitScore]:
    out = retrieval(ro, g, u)
    if ro.is_bot or not ro.report:
        out.append(UnitScore("completion", "all", "task", "all", "completion", Q, 0.0, 1))
        return out
    out += synthesis(ind, g, u) + reasoning(ind, g, u) + organisation(ro, ind, g, u) + writing(ro, ind, g, u) + form(ind, g, u)
    out.append(UnitScore("completion", "all", "task", "all", "completion", Q, 1.0, 1))
    return out
