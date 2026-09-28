"""Adapter for SurveyForge outputs, which share the AutoSurvey format with prefixed reference
values."""

from __future__ import annotations

from pathlib import Path

from ccbench.adapters import autosurvey
from ccbench.model import Rollout


def _pmid(v) -> str:
    s = str(v)
    return s.split(".")[-1] if "." in s else s


def adapt(task: str, task_dir: Path) -> Rollout:
    return autosurvey.adapt(task, task_dir, arm="surveyforge", ref_value_to_pmid=_pmid)
