#!/usr/bin/env python3
"""Defines one SCRIBE run on a task: loads the TaskSpec, entry bundle and reference list, runs
acquisition and the synthesis, planning and writing windows in the chosen entry mode, and writes
sealed artifacts, manifests, the call log and integrity results."""

from __future__ import annotations
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import windows as W
from calllog import CallLog
from common import ROOT, RUNS, env_path, sha256_str, utcnow, validate, host_class
from integrity import check_chain

ORACLE = RUNS / "oracle" / "dev"
SPECS = RUNS / "taskspecs" / "dev"
CANON = RUNS / "canonical" / "campaign50_ref"
ALLOWLISTS = env_path("SCRIBE_ALLOWLISTS")
ENTRY_MODES = ("native_chain", "bundle_entry")

SPLITS = ("dev", "train")
TASKSPECS_ROOT = RUNS / "taskspecs"
CANON_SPLIT_ROOT = RUNS / "canonical"
GOLD_ROOT = RUNS / "gold"
_SPLIT_CACHE = {}
_SPLIT_CACHE_LOCK = __import__("threading").Lock()


def specs_dir(split="dev") -> Path:
    return SPECS if split == "dev" else TASKSPECS_ROOT / split


def canonical_root(task_id=None, split=None) -> Path:
    split = split or split_of(task_id)
    return CANON if split == "dev" else CANON_SPLIT_ROOT / split / "bundles"


def allowlist_dir(split="dev") -> Path:
    return ALLOWLISTS if split == "dev" else CANON_SPLIT_ROOT / split / "allowlists"


_NO_DEFAULT = object()


def split_of(task_id, default=_NO_DEFAULT) -> str:
    hit = _SPLIT_CACHE.get(task_id)
    if hit is not None:
        return hit
    found = [s for s in SPLITS if (specs_dir(s) / f"{task_id}.json").exists()]
    if len(found) > 1:
        raise KeyError(f"task {task_id!r} has a taskspec in MORE THAN ONE split {found}; the "
                       f"splits must be disjoint (the run directory does not encode the split, "
                       f"and the allowlist/bundle a run is handed would be ambiguous)")
    if found:
        with _SPLIT_CACHE_LOCK:
            _SPLIT_CACHE[task_id] = found[0]
        return found[0]
    if default is not _NO_DEFAULT:
        return default
    raise KeyError(f"task {task_id!r} belongs to no split; probed "
                   + ", ".join(str(specs_dir(s)) for s in SPLITS))


def _split_cache_clear():
    with _SPLIT_CACHE_LOCK:
        _SPLIT_CACHE.clear()


split_of.cache_clear = _split_cache_clear


GOLD_INPUTS_WITHHELD = {"synthesis_graph": True, "outline_plan": True}
ADAPTER_VERSION = "runners/1.0"
SCRIBE = "SCRIBE"
SYSTEMS = (SCRIBE,)
LEVERS_FILE = ROOT / "scribe" / "levers" / "writing_levers.json"
SNAPSHOT = "pool-bm25-263f4ee078ad2e00"
CODE_GLOBS = ("scribe/harness/runners/*.py", "scribe/harness/runners/*.sh", "scribe/harness/tools/*.py",
              "scribe/levers/*", "scribe/retrieval/*.py", "scribe/retrieval/skills/*.txt")


def code_fingerprint():
    h = hashlib.sha256()
    for p in sorted({q for g in CODE_GLOBS for q in ROOT.glob(g) if q.is_file()}):
        h.update(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes() + b"\0")
    return "code-sha256:" + h.hexdigest()[:16]


COMMIT = code_fingerprint()


PROMPT_TEMPLATES = ("ACQ_SYS", "ACQ_RANK", "SYN_SYS", "SYN_EXTRACT", "SYN_REL", "SYN_CROSS",
                    "PLAN_SYS", "PLAN_USER", "WRITE_SYS", "WRITE_USER")


def prompt_hash(write_user=None):
    src = "".join(write_user if n == "WRITE_USER" and write_user is not None else getattr(W, n)
                  for n in PROMPT_TEMPLATES)
    return sha256_str(src)


PROMPT_HASH = prompt_hash()


def run_dir(level, system, mode, task_id, seed) -> Path:
    return RUNS / f"level{level}" / system / mode / task_id / f"seed{seed}"


def load_spec(task_id, split=None):
    return json.loads((specs_dir(split or split_of(task_id)) / f"{task_id}.json").read_text())


def load_oracle(task_id, kind):
    p = ORACLE / kind / f"{task_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_canonical(task_id, kind, root=None, split=None):
    p = (root or canonical_root(task_id, split)) / kind / f"{task_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def allowlist_path(task_id) -> Path:
    return ALLOWLISTS / f"{task_id}.json"


def split_allowlist_path(task_id, split=None) -> Path:
    return allowlist_dir(split or split_of(task_id, default="dev")) / f"{task_id}.json"


def load_allowlist(task_id, split=None):
    p = split_allowlist_path(task_id, split) if split else allowlist_path(task_id)
    if not p.exists():
        return None
    return [str(x) for x in json.loads(p.read_text())]


def sealed_allowlist(allow, bundle):
    val = (bundle or {}).get("validation") or {}
    excluded = []
    rp = val.get("review_pmid_excluded")
    if rp:
        excluded.append(str(rp))
    for x in (val.get("post_cutoff_excluded_pmids") or []):
        if str(x) not in excluded:
            excluded.append(str(x))
    if not excluded:
        return allow, []
    drop = set(excluded)
    return [p for p in allow if str(p) not in drop], excluded


REF_YEARS = env_path("SCRIBE_REF_YEARS")
_YEARS_CACHE = {}


def _years_map(path):
    key = str(path)
    if key not in _YEARS_CACHE:
        if not Path(path).exists():
            _YEARS_CACHE[key] = None
        else:
            _YEARS_CACHE[key] = {str(k): v for k, v in json.loads(Path(path).read_text()).items()
                                 if v is not None}
    return _YEARS_CACHE[key]


def gold_path(task_id, split=None) -> Path:
    return GOLD_ROOT / (split or split_of(task_id)) / f"{task_id}.json"


def verify_exclusions(task_id, split, raw, bundle, excluded):
    if not excluded:
        return []
    errors = []
    val = (bundle or {}).get("validation") or {}
    raw_set = {str(x) for x in (raw or [])}
    delivered = {str(p.get("paper_id")) for p in ((bundle or {}).get("papers") or [])}

    raw_sha = allowlist_sha256(raw or [])
    got = val.get("allowlist_sha256_file")
    if got is None:
        errors.append(f"bundle claims {len(excluded)} allowlist exclusion(s) but records no "
                      f"validation.allowlist_sha256_file, so the raw allowlist it was sealed "
                      f"against cannot be pinned")
    elif got != raw_sha:
        errors.append(f"validation.allowlist_sha256_file {got} != sha256 of the allowlist file "
                      f"{raw_sha} (the bundle was sealed against a different file)")
    n_file = val.get("n_allowlisted_file")
    if n_file is not None and int(n_file) != len(raw or []):
        errors.append(f"validation.n_allowlisted_file {n_file} != {len(raw or [])} pmids in "
                      f"the allowlist file")

    not_in_raw = [x for x in excluded if x not in raw_set]
    if not_in_raw:
        errors.append(f"{len(not_in_raw)} excluded pmids are not in the allowlist file at all: "
                      f"{not_in_raw[:5]}")
    still_there = [x for x in excluded if x in delivered]
    if still_there:
        errors.append(f"{len(still_there)} pmids are recorded as excluded AND delivered: "
                      f"{still_there[:5]}")

    rp = val.get("review_pmid_excluded")
    if rp:
        gp = gold_path(task_id, split)
        if not gp.exists():
            errors.append(f"bundle excludes review_pmid {rp} but there is no gold record to "
                          f"check it against: {gp}")
        else:
            gold_rp = (json.loads(gp.read_text()) or {}).get("review_pmid")
            if gold_rp is None or str(gold_rp) != str(rp):
                errors.append(f"validation.review_pmid_excluded {rp!r} != gold review_pmid "
                              f"{gold_rp!r} ({gp}): the leakage exclusion names a different "
                              f"paper from the one the review actually is")
            if val.get("review_pmid") is not None and str(val["review_pmid"]) != str(gold_rp):
                errors.append(f"validation.review_pmid {val['review_pmid']!r} != gold "
                              f"review_pmid {gold_rp!r}")

    post = [str(x) for x in (val.get("post_cutoff_excluded_pmids") or [])]
    if post:
        ry = _years_map(REF_YEARS)
        sry = _years_map(CANON_SPLIT_ROOT / (split or "") / "ref_years.json") if split else None
        if ry is None:
            errors.append(f"cannot check {len(post)} post-cutoff exclusion(s): {REF_YEARS} "
                          f"is missing")
        else:
            dated = [x for x in post
                     if ry.get(x) is not None or (sry or {}).get(x) is not None]
            if dated:
                errors.append(f"{len(dated)} post-cutoff exclusions name a pmid a ref_years "
                              f"manifest DOES date: {dated[:5]} -- the cutoff guard is licensed "
                              f"only for a pmid neither manifest dates")

    n_rev = int(val.get("n_review_pmid_excluded") or 0)
    n_post = int(val.get("n_post_cutoff_excluded") or 0)
    if n_rev != (1 if rp else 0):
        errors.append(f"validation.n_review_pmid_excluded {n_rev} != "
                      f"{1 if rp else 0} (review_pmid_excluded={rp!r})")
    if n_post != len(post):
        errors.append(f"validation.n_post_cutoff_excluded {n_post} != "
                      f"{len(post)} post_cutoff_excluded_pmids")
    if n_file is not None and int(n_file) - n_rev - n_post != len(raw or []) - len(excluded):
        errors.append(f"exclusion counts do not reconcile: n_allowlisted_file {n_file} - "
                      f"n_review_pmid_excluded {n_rev} - n_post_cutoff_excluded {n_post} != "
                      f"{len(raw or [])} allowlisted - {len(excluded)} excluded")
    return errors


def allowlist_sha256(pmids) -> str:
    return sha256_str(json.dumps(sorted((str(x) for x in pmids), key=int)))


def texts_sha256(texts) -> str:
    return sha256_str(json.dumps({str(k): ((v or {}).get("abstract") or "")
                                  for k, v in texts.items()}, sort_keys=True))


def texts_from_canonical(task_id, root=None, split=None):
    return load_canonical(task_id, "texts", root, split) or {}


def check_bundle_delivery(task_id, spec, bundle, texts, canonical=None, allowlist=None,
                          split=None):
    errors = []
    croot = canonical_root(task_id, split) if split else CANON
    al_path = split_allowlist_path(task_id, split) if split else allowlist_path(task_id)
    excluded = []
    if allowlist is None:
        allow = load_allowlist(task_id, split)
        if allow is not None and split:
            allow, excluded = sealed_allowlist(allow, bundle)
    else:
        allow = [str(x) for x in allowlist]
    if allow is None:
        errors.append(f"allowlist missing: {al_path}")
        allow = []
    exc_errors = []
    if excluded:
        exc_errors = verify_exclusions(task_id, split, load_allowlist(task_id, split) or [],
                                       bundle, excluded)
        errors += exc_errors
    allow_set = set(allow)
    if len(allow) != len(allow_set):
        errors.append(f"duplicate pmids in allowlist file: {len(allow) - len(allow_set)}")
    papers = list((bundle or {}).get("papers") or [])
    pids = [str(p.get("paper_id")) for p in papers]
    pid_set = set(pids)
    if len(pids) != len(pid_set):
        errors.append(f"duplicate paper_id in bundle: {len(pids) - len(pid_set)}")
    if not texts:
        errors.append(f"delivered texts empty/missing: {croot / 'texts' / (task_id + '.json')}")
    absent = sorted((pid_set - set(str(k) for k in (texts or {}))),
                    key=lambda x: int(x) if x.isdigit() else -1)
    if texts and absent:
        errors.append(f"{len(absent)} bundle papers absent from delivered texts: {absent[:5]}")
    not_allowed = sorted(pid_set - allow_set, key=lambda x: int(x) if x.isdigit() else -1)
    not_delivered = sorted(allow_set - pid_set, key=int)
    if not_allowed:
        errors.append(f"{len(not_allowed)} bundle papers not in allowlist: {not_allowed[:5]}")
    if not_delivered:
        errors.append(f"{len(not_delivered)} allowlisted pmids not delivered: {not_delivered[:5]}")

    n_checked, bad_hash = 0, []
    for e in (bundle or {}).get("evidence") or []:
        pid = str(e.get("paper_id"))
        t = texts.get(pid) or {}
        if e.get("granularity") == "abstract":
            txt = t.get("abstract")
        elif e.get("granularity") == "abstract_sentence":
            txt = (t.get("sentences") or {}).get(e.get("locator"))
        else:
            txt = None
        n_checked += 1
        if txt is None or sha256_str(txt) != e.get("text_hash"):
            bad_hash.append(e.get("evidence_id"))
    if bad_hash:
        errors.append(f"{len(bad_hash)} evidence entries whose text_hash != sha256(delivered "
                      f"text): {bad_hash[:5]}")

    cutoff = spec.get("publication_cutoff") if spec else None
    late = [p["paper_id"] for p in papers
            if cutoff is not None and p.get("year") is not None and int(p["year"]) >= int(cutoff)]
    if late:
        errors.append(f"{len(late)} papers with year >= cutoff {cutoff}: {late[:5]}")
    n_year_unknown = sum(1 for p in papers if p.get("year") is None)

    bh = (bundle or {}).get("content_hash")
    from common import content_hash as _ch
    if bundle and bh != _ch(bundle):
        errors.append("bundle content_hash does not re-hash (bundle edited after sealing)")
    if canonical is None:
        canonical = (load_canonical(task_id, "evidence_bundle", split=split) if split
                     else load_canonical(task_id, "evidence_bundle", root=CANON))
    if canonical is None:
        errors.append(f"canonical bundle missing: {croot / 'evidence_bundle' / (task_id + '.json')}")
    elif canonical.get("content_hash") != bh:
        errors.append(f"entry bundle hash {bh} != canonical file hash {canonical.get('content_hash')}")

    val = (bundle or {}).get("validation") or {}
    al_sha = allowlist_sha256(allow) if allow else None
    if val.get("allowlist_sha256") != al_sha:
        errors.append(f"validation.allowlist_sha256 {val.get('allowlist_sha256')} != recomputed {al_sha}")
    if bundle and bundle.get("provenance_tier") != "reference_derived":
        errors.append(f"provenance_tier {bundle.get('provenance_tier')!r} != 'reference_derived'")
    if bundle and spec and bundle.get("task_spec_hash") != spec.get("content_hash"):
        errors.append("bundle task_spec_hash != TaskSpec content_hash")

    with_abs = {str(e["paper_id"]) for e in (bundle or {}).get("evidence") or []
                if e.get("granularity") == "abstract"}
    summary = {
        "task_id": task_id,
        "entry_bundle_hash": bh,
        "allowlist_path": str(al_path),
        "allowlist_sha256": al_sha,
        "allowlist_n": len(allow_set),
        "n_delivered_papers": len(pid_set),
        "n_with_abstract": len(pid_set & with_abs),
        "n_without_abstract": len(pid_set - with_abs),
        "n_evidence_checked": n_checked,
        "n_text_hash_mismatch": len(bad_hash),
        "texts_sha256": texts_sha256(texts),
        "delivery_rate": (round(len(pid_set & allow_set) / len(allow_set), 6) if allow_set else 0.0),
        "bundle_subset_of_allowlist": not not_allowed,
        "allowlist_subset_of_bundle": not not_delivered,
        "cutoff": cutoff,
        "cutoff_ok": not late,
        "n_year_unknown": n_year_unknown,
        "ok": not errors,
    }
    if excluded:
        summary["n_allowlist_excluded"] = len(excluded)
        summary["allowlist_excluded"] = excluded
        summary["allowlist_excluded_reconciled"] = not exc_errors
        summary["split"] = split
    return errors, summary


def delivery_fields(summary):
    return {"entry_mode": "bundle_entry",
            "entry_bundle_hash": summary.get("entry_bundle_hash"),
            "allowlist_sha256": summary.get("allowlist_sha256"),
            "allowlist_path": summary.get("allowlist_path"),
            "allowlist_n": summary.get("allowlist_n"),
            "n_delivered_papers": summary.get("n_delivered_papers"),
            "n_with_abstract": summary.get("n_with_abstract"),
            "n_without_abstract": summary.get("n_without_abstract"),
            "texts_sha256": summary.get("texts_sha256"),
            "delivery_rate": summary.get("delivery_rate"),
            "bundle_subset_of_allowlist": summary.get("bundle_subset_of_allowlist"),
            "cutoff_ok": summary.get("cutoff_ok"),
            "n_year_unknown": summary.get("n_year_unknown"),
            "delivery_ok": bool(summary.get("ok")),
            "gold_inputs_withheld": dict(GOLD_INPUTS_WITHHELD)}


def binding_record(bundle, spec, extra=None):
    rec = {"entry_artifact_hash": bundle["content_hash"],
           "task_spec_hash": spec["content_hash"],
           "provenance_tier": bundle["provenance_tier"]}
    if extra:
        rec.update(extra)
    return rec


def texts_from_oracle(task_id):
    p = ORACLE / "sentences" / f"{task_id}.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    out = {}
    for loc, txt in raw.items():
        pid = loc.split("_")[0][4:]
        d = out.setdefault(pid, {"sentences": {}, "_order": []})
        d["sentences"][loc] = txt
        d["_order"].append((int(loc.rsplit("_", 1)[1]), txt))
    for pid, d in out.items():
        d["abstract"] = " ".join(t for _, t in sorted(d.pop("_order")))
    return out


def texts_from_cache(bundle):
    raise RuntimeError("native-chain generation reads its texts from the per-task pool cache; install the pool "
                       "backend first (pool_backend.install_generation)")


def manifest(run_id, task_id, level, window, mode, system, seed, entry_hash, exit_hash,
             budget, resources, started, status, host, serving=None, extra=None):
    m = {"run_id": run_id, "task_id": task_id, "level": level, "window": window,
         "mode": mode, "system": system, "commit": COMMIT, "prompt_hash": PROMPT_HASH,
         "corpus_snapshot_id": SNAPSHOT, "seed": seed,
         "entry_hash": entry_hash or "", "exit_hash": exit_hash or "",
         "budget": budget, "observed_resources": resources,
         "started_at": started, "ended_at": utcnow(), "status": status,
         "adapter_version": ADAPTER_VERSION, "evaluator_version": "none(generation only)",
         "host_class": host, "serving": serving or {}}
    if extra:
        m.update({k: v for k, v in extra.items() if k not in m})
    errs = validate("run_manifest", m)
    if errs:
        m["_manifest_validation_errors"] = errs
    return m


def window_status(window, artifact, schema_errors):
    if schema_errors:
        return "failed"
    if artifact is None:
        return "failed"
    if window == "synthesis" and not artifact.get("claims"):
        return "failed"
    if window == "planning" and not artifact.get("sections"):
        return "failed"
    if window == "writing":
        if not artifact.get("sections"):
            return "failed"
        if not (artifact.get("terminal_audit") or {}).get("n_words"):
            return "failed"
    return "ok"


def _write(d: Path, name, obj):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def check_native_entry(d: Path, spec, bundle):
    from common import content_hash as _ch
    errors = []
    bh = (bundle or {}).get("content_hash")
    if not bh or bh != _ch(bundle):
        errors.append("entry bundle content_hash does not re-hash (bundle edited after sealing)")
    if spec and bundle.get("task_spec_hash") != spec.get("content_hash"):
        errors.append("entry bundle task_spec_hash != TaskSpec content_hash")
    mp = d / "manifests.jsonl"
    acq = [m for m in (json.loads(l) for l in mp.read_text().splitlines() if l.strip())
           if m.get("window") == "acquisition"] if mp.exists() else []
    if len(acq) != 1:
        errors.append(f"{len(acq)} acquisition manifests in {mp} (need exactly 1)")
    else:
        m = acq[0]
        if m.get("status") not in ("ok", "budget_exhausted"):
            errors.append(f"acquisition status {m.get('status')!r}")
        if m.get("exit_hash") != bh:
            errors.append(f"acquisition exit_hash {m.get('exit_hash')} != entry bundle hash {bh}")
    return errors


def run_acquisition(task_id, system, seed, client, level=1, mode="native_chain"):
    from metered_tool import MeteredRetrieval
    if system not in SYSTEMS:
        raise ValueError(f"system {system!r} not in {SYSTEMS}")
    split = split_of(task_id)
    spec = load_spec(task_id, split)
    d = run_dir(level, system, mode, task_id, seed)
    d.mkdir(parents=True, exist_ok=True)
    run_id = f"{system}.{mode}.{task_id}.s{seed}"
    log = CallLog(d / "trace.jsonl", run_id, task_id)
    started = utcnow()
    _write(d, "task_spec.json", spec)

    excl = set()
    gold_p = RUNS / "gold" / split / f"{task_id}.json"
    if gold_p.exists():
        rp = json.loads(gold_p.read_text()).get("review_pmid")
        if rp:
            excl.add(str(rp))

    tool = MeteredRetrieval(SNAPSHOT, spec["publication_cutoff"], spec["budget"], log, exclude_ids=excl)
    try:
        bundle, errs = W.acquisition(spec, tool, client, log, seed=seed)
        status = {"done": "ok", "budget_exhausted": "budget_exhausted",
                  "failed": "failed"}[bundle["retrieval_status"]]
    except Exception as e:
        log.failure("acquisition", started, type(e).__name__, str(e),
                    traceback=traceback.format_exc()[:6000])
        bundle, errs, status = None, [f"exception: {e}"], "failed"

    tool.flush()
    if bundle:
        _write(d, "evidence_bundle.json", bundle)
    _write(d, "acquisition_validation.json", {"schema_errors": errs,
                                              "retrieval_report": tool.report()})
    m = manifest(run_id + ".acquisition", task_id, level, "acquisition", mode, system, seed,
                 spec["content_hash"], bundle["content_hash"] if bundle else "",
                 spec["budget"], dict(log.totals), started, status, "login",
                 serving=client.describe(), extra={"split": split})
    with (d / "manifests.jsonl").open("a") as f:
        f.write(json.dumps(m) + "\n")
    log.close()
    return bundle, m


RETRIEVAL_MODULES = ("metered_tool", "pool_retrieval_tool")
POOL_PROBE_TIMEOUT_S = 3.0


def assert_no_retrieval_path():
    leaked = [m for m in RETRIEVAL_MODULES if m in sys.modules]
    pb = sys.modules.get("pool_backend")
    if pb is not None and (getattr(pb, "_INSTALLED", None) or {}).get("acq"):
        leaked.append("pool_backend (acquisition half installed)")
    return leaked


def probe_pool_url(url=None, timeout=POOL_PROBE_TIMEOUT_S):
    import urllib.error
    import urllib.request
    url = os.environ.get("POOL_URL", "") if url is None else url
    if not url:
        return False, {"configured": False}
    target = url.rstrip("/") + "/healthz"
    rec = {"configured": True, "url": target, "timeout_s": timeout}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(target, timeout=timeout) as r:
            rec.update(reachable=True, http_status=getattr(r, "status", None))
            return True, rec
    except urllib.error.HTTPError as e:
        rec.update(reachable=True, http_status=e.code)
        return True, rec
    except Exception as e:
        rec.update(reachable=False, error=f"{type(e).__name__}: {e}"[:300])
        return False, rec


def retrieval_capabilities():
    found = list(assert_no_retrieval_path())
    reachable, rec = probe_pool_url()
    if reachable:
        found.append(f"POOL_URL {rec['url']} answered (HTTP {rec.get('http_status')})")
    return found, rec


def assert_levers_installed():
    try:
        want = json.loads(LEVERS_FILE.read_text())["levers"]
    except (OSError, ValueError, KeyError) as e:
        raise RuntimeError(f"the release lever file {LEVERS_FILE} is unreadable ({type(e).__name__}: {e})")
    fn = getattr(W, "writing", None)
    got = getattr(fn, "_writing_levers", None)
    if got != want:
        raise RuntimeError("the writing window is not the lever writer with writing_levers "
                           f"(W.writing={getattr(fn, '__name__', fn)!r}, levers={got!r}); install it "
                           "first: writing_levers.install(W, runner, writing_levers.load(<writing_levers.json>))")
    return got


def run_generation(task_id, system, seed, client, level=1, mode="native_chain",
                   entry_bundle=None, client_write=None, entry_mode=None):
    cw = client_write or client
    if system not in SYSTEMS:
        raise ValueError(f"system {system!r} not in {SYSTEMS}")
    assert_levers_installed()
    if entry_mode is None:
        entry_mode = "native_chain"
    if entry_mode not in ENTRY_MODES:
        raise ValueError(f"entry_mode {entry_mode!r} not in {ENTRY_MODES}")
    bundle_entry = entry_mode == "bundle_entry"
    if bundle_entry:
        if mode != entry_mode:
            raise ValueError(f"entry_mode {entry_mode!r} requires mode {entry_mode!r}, got {mode!r}")
    split = split_of(task_id)
    spec = load_spec(task_id, split)
    d = run_dir(level, system, mode, task_id, seed)
    d.mkdir(parents=True, exist_ok=True)
    run_id = f"{system}.{mode}.{task_id}.s{seed}"
    log = CallLog(d / "trace.jsonl", run_id, task_id)
    started = utcnow()
    mans, out = [], {}

    leaked, pool_probe = retrieval_capabilities()
    log.validate_event("synthesis", started, "no_retrieval_path", leaked, host=host_class())
    retrieval_check = {"findings": leaked, "modules_checked": list(RETRIEVAL_MODULES) + ["pool_backend(acq)"],
                       "pool_url_probe": pool_probe}
    if leaked:
        reason = "harness_failure:retrieval_reachable"
        log.failure("synthesis", started, "RetrievalPathPresent",
                    f"{reason}: {leaked}; the no-search frontier is not enforced in this process, so the "
                    f"run is failed before the first model call", recovered=False)
        mans.append(manifest(run_id + ".synthesis", task_id, level, "synthesis", mode, system, seed,
                             (entry_bundle or {}).get("content_hash", ""), "", spec["budget"],
                             dict(log.totals), started, "failed", "gpu", serving=client.describe(),
                             extra={"entry_mode": entry_mode, "split": split,
                                    "failure_reason": reason, "retrieval_check": retrieval_check}))
        _flush(d, log, mans)
        return {}, mans

    if entry_bundle is None:
        if bundle_entry:
            entry_bundle = load_canonical(task_id, "evidence_bundle", split=split)
        else:
            p = d / "evidence_bundle.json"
            entry_bundle = json.loads(p.read_text()) if p.exists() else None
    if entry_bundle is None:
        log.failure("synthesis", started, "MissingEntryArtifact",
                    "no EvidenceBundle to enter the synthesis window with")
        log.close()
        return {}, []
    if entry_mode == "native_chain":
        t_ent = utcnow()
        eerrs = check_native_entry(d, spec, entry_bundle)
        log.validate_event("synthesis", t_ent, "native_entry_hash", eerrs, host=host_class())
        if eerrs:
            reason = "harness_failure:entry_hash"
            log.failure("synthesis", t_ent, "EntryHashMismatch", f"{reason}: " + "; ".join(eerrs)[:3500],
                        recovered=False)
            mans.append(manifest(run_id + ".synthesis", task_id, level, "synthesis", mode, system, seed,
                                 entry_bundle.get("content_hash", ""), "", spec["budget"], dict(log.totals), t_ent,
                                 "failed", "gpu", serving=client.describe(),
                                 extra={"entry_mode": entry_mode, "split": split,
                                        "failure_reason": reason, "entry_check": eerrs,
                                        "retrieval_check": retrieval_check}))
            _flush(d, log, mans)
            return {}, mans

    if bundle_entry:
        texts = texts_from_canonical(task_id, split=split)
    else:
        texts = texts_from_cache(entry_bundle)
    extra = {"entry_mode": entry_mode, "split": split, "retrieval_check": retrieval_check}
    bind_extra = None
    if bundle_entry:
        t_del = utcnow()
        derrs, dsum = check_bundle_delivery(task_id, spec, entry_bundle, texts, split=split)
        log.validate_event("synthesis", t_del, "bundle_delivery", derrs, host=host_class())
        bind_extra = delivery_fields(dsum)
        extra.update(bind_extra)
        _write(d, "delivery.json", {"schema": "bundle_delivery/1.0", "task_id": task_id,
                                    "run_id": run_id, "checked_at": t_del,
                                    "ok": not derrs, "errors": derrs, "summary": dsum,
                                    "fields": bind_extra})
    _write(d, "entry_evidence_bundle.json", entry_bundle)
    _write(d, "binding.json", binding_record(entry_bundle, spec, extra=bind_extra))
    if bundle_entry and derrs:
        reason = "harness_failure:bundle_delivery"
        log.failure("synthesis", t_del, "BundleDeliveryMismatch",
                    f"{reason}: " + "; ".join(derrs)[:3500], recovered=False)
        extra["failure_reason"] = reason
        mans.append(manifest(run_id + ".synthesis", task_id, level, "synthesis", mode, system,
                             seed, entry_bundle.get("content_hash"), "", spec["budget"],
                             dict(log.totals), t_del, "failed", "gpu",
                             serving=client.describe(), extra=extra))
        _flush(d, log, mans)
        return {}, mans

    t0 = utcnow()
    try:
        graph, errs = W.synthesis(spec, entry_bundle, texts, client, log, seed=seed)
        st = window_status("synthesis", graph, errs)
    except Exception as e:
        log.failure("synthesis", t0, type(e).__name__, str(e), traceback=traceback.format_exc()[:6000])
        graph, errs, st = None, [str(e)], "failed"
    if graph:
        _write(d, "synthesis_graph.json", graph)
        out["synthesis_graph"] = graph
    mans.append(manifest(run_id + ".synthesis", task_id, level, "synthesis", mode, system, seed,
                         entry_bundle["content_hash"], graph["content_hash"] if graph else "",
                         spec["budget"], dict(log.totals), t0, st, "gpu",
                         serving=client.describe(), extra=extra))
    if not graph:
        _flush(d, log, mans)
        return out, mans

    entry_graph = graph
    t0 = utcnow()
    try:
        plan, errs = W.planning(spec, entry_graph, client, log, seed=seed)
        st = window_status("planning", plan, errs)
    except Exception as e:
        log.failure("planning", t0, type(e).__name__, str(e), traceback=traceback.format_exc()[:6000])
        plan, errs, st = None, [str(e)], "failed"
    if plan:
        _write(d, "outline_plan.json", plan)
        out["outline_plan"] = plan
    mans.append(manifest(run_id + ".planning", task_id, level, "planning", mode, system, seed,
                         entry_graph["content_hash"], plan["content_hash"] if plan else "",
                         spec["budget"], dict(log.totals), t0, st, "gpu",
                         serving=client.describe(), extra=extra))
    if not plan:
        _flush(d, log, mans)
        return out, mans

    entry_plan = plan
    t0 = utcnow()
    try:
        report, errs = W.writing(spec, entry_graph, entry_plan, texts, entry_bundle, cw, log, seed=seed)
        st = window_status("writing", report, errs)
    except Exception as e:
        log.failure("writing", t0, type(e).__name__, str(e), traceback=traceback.format_exc()[:6000])
        report, errs, st = None, [str(e)], "failed"
    if report:
        _write(d, "report_artifact.json", report)
        out["report_artifact"] = report
    mans.append(manifest(run_id + ".writing", task_id, level, "writing", mode, system, seed,
                         entry_plan["content_hash"], report["content_hash"] if report else "",
                         spec["budget"], dict(log.totals), t0, st, "gpu",
                         serving=cw.describe(), extra=extra))

    integ = check_chain(spec, entry_bundle, graph, plan, report)
    _write(d, "integrity.json", integ)
    _flush(d, log, mans)
    return out, mans


def _flush(d, log, mans):
    with (d / "manifests.jsonl").open("a") as f:
        for m in mans:
            f.write(json.dumps(m) + "\n")
    _write(d, "resources.json", dict(log.totals))
    log.close()
