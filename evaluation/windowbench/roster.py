"""The arm roster: label, role, backbone, provenance and entries of every system key, the families
read from WINDOWBENCH_ROSTER_EXTRA on first use, and the two datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

WINDOWS = ("retrieval", "synthesis", "planning", "writing", "draft", "draft_trunc", "system", "system_trunc")


BACKBONE = {"gpt56": "gpt-5.6-luna", "drtulu": "DR-Tulu-8B", "qwen38_frozen_bundle": "Qwen3.8-27B + KV-Skill carrier"}


INTERFACE = {"autosurvey": "own_faiss_nomic768_abstract", "surveyforge": "own_faiss_gte1024_title_abstract", "surveyg": "pool_service_bm25",
             "sgi": "pool_service_bm25", "llmxmr": "pool_service_bm25", "drtulu": "pool_service_bm25",
             "lira": "given_bibliography", "ours": "pool_service_bm25"}


TEMPERATURE = {"autosurvey": "provider default", "surveyforge": "provider default", "llmxmr": "provider default",
               "lira": "provider default", "surveyg": "pipeline config", "sgi": "0", "drtulu": "pipeline config",
               "ours": "0 (greedy)"}


def _arm(label, role, model_group, provenance, *, retrieval, synthesis, planning, writing, system,
         allocation=None, family=None, notes="", interface=None, temperature=None, draft="prefix", draft_is_final=False):
    return {
        "label": label, "role": role, "model_group": model_group, "provenance": provenance, "backbone": BACKBONE.get(model_group, model_group),
        "retrieval_interface": interface, "temperature": temperature,
        "entries": {"retrieval": retrieval, "synthesis": synthesis, "planning": planning,
                    "writing": writing, "allocation": allocation,
                    "draft": system if draft else None, "draft_trunc": system if draft else None,
                    "system": system, "system_trunc": system},
        "draft_is_final": bool(draft_is_final), "family": family, "notes": notes,
    }


_ALLOC = "own_sections|own_papers"
_GPT_SELF = dict(role="baseline", model_group="gpt56", provenance="self", allocation=_ALLOC,
                 retrieval="pool_full", synthesis=None, planning="own_kept_set", writing="own_outline",
                 system="taskspec|pool_full")
_GPT_REF = dict(role="baseline", model_group="gpt56", provenance="reference_fed", allocation=_ALLOC,
                retrieval="pool_allowlist", synthesis=None, planning="own_kept_set", writing="own_outline",
                system="taskspec|pool_allowlist")

_BASE_SYSTEMS: dict[str, dict] = {
    "autosurvey": _arm("AutoSurvey", **_GPT_SELF, interface=INTERFACE["autosurvey"]),
    "surveyforge": _arm("SurveyForge", **_GPT_SELF, interface=INTERFACE["surveyforge"]),
    "surveyg": _arm("SurveyG", **_GPT_SELF, interface=INTERFACE["surveyg"]),
    "sgi": _arm("SurveyGen-I", **_GPT_SELF, interface=INTERFACE["sgi"]),
    "llmxmr": _arm("LLMxMapReduce", **_GPT_SELF, interface=INTERFACE["llmxmr"]),
    "lira": _arm("LiRA", role="baseline", model_group="gpt56", provenance="reference_fed", allocation=_ALLOC,
                 retrieval=None, synthesis=None, planning="own_kept_set", writing="own_outline",
                 system="taskspec|given_bibliography", interface=INTERFACE["lira"]),
    "drtulu": _arm("DR-Tulu", role="baseline", model_group="drtulu", provenance="self", allocation=_ALLOC,
                   retrieval="pool_full", synthesis=None, planning=None, writing=None,
                   system="taskspec|pool_full", interface=INTERFACE["drtulu"], draft_is_final=True),
    "autosurvey.ref": _arm("AutoSurvey.ref", **_GPT_REF, interface=INTERFACE["autosurvey"]),
    "surveyg.ref": _arm("SurveyG.ref", **_GPT_REF, interface=INTERFACE["surveyg"]),
    "llmxmr.ref": _arm("LLMxMapReduce.ref", **_GPT_REF, interface=INTERFACE["llmxmr"]),
    "sgi.ref": _arm("SurveyGen-I.ref", **_GPT_REF, interface=INTERFACE["sgi"]),
}

_TEMPLATES = {
    "bundle_entry": dict(role="ours", model_group="qwen38_frozen_bundle", provenance="reference_fed", draft_is_final=True,
                         allocation=_ALLOC, retrieval=None, synthesis="pool_allowlist", planning="own_graph",
                         writing="own_outline", system="taskspec|pool_allowlist"),
    "native": dict(role="ours", model_group="qwen38_frozen_bundle", provenance="self", interface=INTERFACE["ours"],
                   draft_is_final=True, allocation=_ALLOC, retrieval="pool_full", synthesis="own_papers",
                   planning="own_graph", writing="own_outline", system="taskspec|pool_full"),
}

DATASETS = {
    "fixed_input": ["autosurvey.ref", "surveyg.ref", "llmxmr.ref", "sgi.ref", "lira"],
    "same_pool": ["autosurvey", "surveyforge", "surveyg", "sgi", "llmxmr", "drtulu"],
}

REPLICATE_SEEDS = (1, 2)

_STATE: dict = {}


def _extra_families(path) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    doc = json.loads(p.read_text())
    if int(doc.get("version", 1)) != 1:
        raise ValueError(f"{p}: unsupported roster_extra version {doc.get('version')!r}")
    return list(doc.get("families", []))


def _declare(systems: dict, fam: dict, src) -> None:
    tag = fam["tag"]
    tpl_name = fam.get("template", "bundle_entry")
    if tpl_name not in _TEMPLATES:
        raise KeyError(f"{src}: family {tag!r} asks for unknown template {tpl_name!r} (have {sorted(_TEMPLATES)})")
    tpl = dict(_TEMPLATES[tpl_name])
    mg = fam.get("model_group")
    if mg is not None:
        if mg not in BACKBONE:
            raise KeyError(f"{src}: family {tag!r} has model_group {mg!r}; declared groups are {sorted(BACKBONE)}")
        tpl["model_group"] = mg
    keys = list(fam["keys"])
    labels = list(fam.get("labels") or [])
    if labels and len(labels) != len(keys):
        raise ValueError(f"{src}: family {tag!r} has {len(keys)} keys but {len(labels)} labels")
    seeds = [int(n) for n in (fam.get("seeds") or [])]
    if any(n < 1 for n in seeds):
        raise ValueError(f"{src}: family {tag!r} declares seeds {seeds}; seed 0 is the key itself")
    for i, k in enumerate(keys):
        if k in systems:
            raise KeyError(f"{src}: system key {k!r} is already declared; roster_extra.json may only add")
        label = labels[i] if labels else f"{tag} ({k})"
        systems[k] = _arm(label, family=tag, temperature=TEMPERATURE["ours"], **tpl)
        for n in seeds:
            sk = f"{k}.seed{n}"
            if sk in systems:
                raise KeyError(f"{src}: system key {sk!r} is already declared")
            systems[sk] = _arm(f"{label} seed {n}", family=None, temperature=TEMPERATURE["ours"], **tpl)


def _load(path=None) -> dict:
    if _STATE:
        return _STATE
    src = path if path is not None else os.environ.get("WINDOWBENCH_ROSTER_EXTRA")
    systems = {k: json.loads(json.dumps(v)) for k, v in _BASE_SYSTEMS.items()}
    for k, v in systems.items():
        if v.get("temperature") is None:
            v["temperature"] = TEMPERATURE.get(k.replace(".ref", ""), "undeclared")
    for fam in _extra_families(src):
        if fam["tag"] in systems or fam["tag"] in DATASETS:
            raise KeyError(f"{src}: family {fam['tag']!r} collides with a declared key or dataset")
        _declare(systems, fam, src)
    families: dict[str, list[str]] = {}
    for k, v in systems.items():
        if v.get("family"):
            families.setdefault(v["family"], []).append(k)
    _STATE.update(SYSTEMS=systems, FAMILIES=families)
    globals().update(_STATE)
    return _STATE


def reset() -> None:
    _STATE.clear()
    for name in ("SYSTEMS", "FAMILIES"):
        globals().pop(name, None)


def reload(path=None) -> dict:
    reset()
    return _load(path if path is not None else "")


def __getattr__(name):
    if name in ("SYSTEMS", "FAMILIES"):
        return _load()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def replicate_keys(key: str) -> list[str]:
    return [key] + [f"{key}.seed{n}" for n in REPLICATE_SEEDS]


def resolve_arms(spec: str) -> list[str]:
    st = _load()
    if spec in st["FAMILIES"]:
        return list(st["FAMILIES"][spec])
    if spec in DATASETS:
        return list(DATASETS[spec])
    keys = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [k for k in keys if k not in st["SYSTEMS"]]
    if unknown:
        raise KeyError(f"unknown system keys {unknown}; declare them in $WINDOWBENCH_ROSTER_EXTRA first")
    return keys


def entry(system: str, window: str):
    return _load()["SYSTEMS"][system]["entries"].get(window)


def observable(system: str, window: str) -> bool:
    return entry(system, window) is not None
