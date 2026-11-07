#!/usr/bin/env python3
"""Acquisition driver of SCRIBE under same pool: installs the ranker, query agent, budget backend
and selection rule over the windows, checks the wrapper chain and runs run_campaign_pool. Usage:
python run_acquisition.py --phase acquisition <run_campaign options>."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNERS = (HERE.parent / "harness" / "runners").resolve()


def _runners():
    sys.path.insert(0, str(RUNNERS))
    if str(HERE) not in sys.path:
        sys.path.append(str(HERE))
    return RUNNERS


def stamp_ranker(R, cfg):
    if getattr(R.manifest, "_ranker_stamped", False):
        return
    orig = R.manifest

    def manifest(*a, **k):
        extra = dict(k.get("extra") or {})
        extra.setdefault("ranker", cfg["version"])
        extra.setdefault("ranking_module_md5", cfg["module_md5"])
        k["extra"] = extra
        return orig(*a, **k)
    for attr in ("_cap_budget_stamped", "_budget_stamped"):
        if getattr(orig, attr, False):
            setattr(manifest, attr, True)
    manifest._ranker_stamped = True
    R.manifest = manifest


def selection_chain_problem(W):
    f = W.acquisition
    inner = getattr(f, "_selection_rule_inner", None)
    if not getattr(f, "_selection_rule", False) or inner is None or getattr(inner, "_ranker", False) or \
            getattr(inner, "_query_agent", False) or getattr(inner, "_selection_rule", False):
        return "### selection rule: install order check failed: windows.acquisition must be selection rule -> the cap-budget recorder"

    class _V:
        _rank_once = W._rank_once
        acquisition = inner
    why = chain_problem(_V)
    if why:
        return why
    s = W._sealed
    cl = [c.cell_contents for c in (getattr(s, "__closure__", None) or []) if callable(getattr(c, "cell_contents", None))]
    if not getattr(s, "_query_agent", False) or not any(getattr(x, "_selection_rule", False) for x in cl):
        return "### selection rule: windows._sealed must be query agent -> selection rule -> windows"
    return None


def chain_problem(W):
    for name in ("_rank_once", "acquisition"):
        f = getattr(W, name)
        inner = [c.cell_contents for c in (getattr(f, "__closure__", None) or []) if callable(getattr(c, "cell_contents", None))]
        if name == "acquisition":
            mids = [x for x in inner if getattr(x, "_query_agent", False)]
            ok = (not getattr(f, "_ranker", False) and not getattr(f, "_query_agent", False) and len(mids) == 1
                  and getattr(getattr(mids[0], "_query_agent_inner", None), "_ranker", False))
            if not ok:
                return "### query agent: install order check failed for windows.acquisition: want cap-budget recorder -> query agent -> ranker"
            continue
        if getattr(f, "_ranker", False) or getattr(f, "_query_agent", False) or not any(getattr(x, "_ranker", False) for x in inner):
            return f"### ranker: install order check failed for windows.{name}: the cap-budget recorder does not wrap the ranker"
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runners = _runners()
    import run_campaign_pool as RCP
    phase = RCP._phase_of(argv)
    if phase != "acquisition":
        sys.exit(f"### acquisition: run_acquisition.py serves --phase acquisition only (got {phase!r}); generation "
                 f"runs run_budgeted_campaign.py")
    import windows as W
    if Path(W.__file__).resolve().parent != runners:
        sys.exit(f"### acquisition: windows imported from {W.__file__}, not {runners}")
    import budget as PBK
    import ranker as CR
    import query_agent as QA
    import selection_rule as SR
    for m in (PBK, CR, QA, SR):
        if Path(m.__file__).resolve().parent != HERE:
            sys.exit(f"### acquisition: {m.__name__} imported from {m.__file__}, not {HERE}")
    try:
        kc = PBK.load_kcap()
    except PBK.KcapConfigError as e:
        sys.exit(f"### cap budget: refused before any unit: {e}")
    print(f"### K_cap file {kc['path']} sha256 {kc['sha256'][:16]} ({len(kc['tasks'])} tasks, rule "
          f"{PBK.CAP_RULE})", flush=True)
    try:
        cfg = CR.install(W)
        ecfg = SR.config()
        SR.install_sealed(W, ecfg)
        agent_cfg = QA.install(W)
        PBK.install_acquisition_kcap(kc)
        SR.install_acquisition(W, ecfg)
        import runner as R
        stamp_ranker(R, cfg)
        QA.stamp_manifests(R, agent_cfg)
        SR.stamp_manifests(R, ecfg)
    except BaseException as e:
        sys.exit(f"### acquisition: refused before any unit: {type(e).__name__}: {e}")
    why = selection_chain_problem(W)
    if why:
        sys.exit(why)
    print(f"### query agent: module md5 {agent_cfg['module_md5']}; skills {[(x['name'], x['md5']) for x in agent_cfg['skills']]}",
          flush=True)
    print(f"### selection rule: gate={ecfg['gate']} window_years={ecfg['window_years']} module md5 {ecfg['module_md5']} "
          f"edges meta {ecfg['edges_meta_sha256'][:16]}", flush=True)
    print(f"### ranker ({CR.RULE}) installed under the cap-budget recorder: {cfg['version']} module md5 "
          f"{cfg['module_md5']}; tokenizer {cfg['tokenizer_dir']} sha256 {cfg['tokenizer_sha256'][:16]} template "
          f"{cfg['template_sha256'][:16]}; max_model_len {cfg['max_model_len']}; constants {cfg['constants']}", flush=True)
    RCP.main(argv)


if __name__ == "__main__":
    main()
