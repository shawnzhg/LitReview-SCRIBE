"""Assignment of each readout to its capability axis."""

from __future__ import annotations

STAGE_GROUP = {"retrieval": "retrieval", "synthesis": "claim_coverage", "reasoning": "reasoning", "organisation": "organisation", "writing": "citation_fidelity", "form": "form", "all": "completion"}
FORM_READOUTS = {"wr_density_fit", "wr_restatement", "org_size_fit"}


def group_of(readout: str, stage: str) -> str:
    if readout in FORM_READOUTS:
        return "form"
    return STAGE_GROUP.get(stage, stage)
