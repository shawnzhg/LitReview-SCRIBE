"""Registry of the per-system adapters that turn each pipeline's run directory into a common
rollout."""

from __future__ import annotations

from ccbench.adapters import agent, autosurvey, drtulu, lira, llmxmr, sgi, surveyforge, surveyg

BASELINE_ADAPTERS = {
    "autosurvey": autosurvey.adapt,
    "surveyforge": surveyforge.adapt,
    "surveyg": surveyg.adapt,
    "sgi": sgi.adapt,
    "lira": lira.adapt,
    "llmxmr": llmxmr.adapt,
    "drtulu": drtulu.adapt,
}

AGENT_ADAPTER = agent.adapt

__all__ = ["BASELINE_ADAPTERS", "AGENT_ADAPTER"]
