"""Dataclasses for task contexts, events, rollouts, reports and the failure symbol."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BOT = "bot"


@dataclass
class Context:
    task: str
    topic: str
    cutoff: int
    pool_fingerprint: str | None = None
    snapshot_id: str | None = None
    target_words: int | None = None
    budget: dict = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    review_type: str | None = None


@dataclass
class Label:

    stage: str
    action_class: str
    tool_class: str
    resource_bin: str
    observable_effect: str
    retry_or_failure: str | None = None
    novelty: str | None = None

    def as_tuple(self) -> tuple:
        return (self.stage, self.action_class, self.tool_class, self.resource_bin, self.observable_effect)


@dataclass
class Event:
    seq: int
    t_start: float
    t_end: float
    kind: str
    label: Label
    resources: dict = field(default_factory=dict)
    route: str | None = None
    status: int | None = None
    finish_reason: str | None = None
    n_returned: int | None = None
    returned_ids: list[str] = field(default_factory=list)
    query: str | None = None
    prompt_chars: int = 0
    completion_chars: int = 0


@dataclass
class OutlineNode:
    id: str
    title: str
    level: int
    parent: str | None = None


@dataclass
class Sentence:
    sid: str
    section: str
    text: str
    cites: list[str] = field(default_factory=list)


@dataclass
class Report:
    sections: list[dict] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)
    bibliography: list[str] = field(default_factory=list)
    words: int = 0
    unresolved_citations: int = 0
    total_citation_marks: int = 0


@dataclass
class Rollout:
    system: str
    task: str
    panel: str
    context: Context
    status: str = "ok"
    bot_reason: str | None = None
    attempt: str = "final"
    mode: str | None = None
    events: list[Event] = field(default_factory=list)
    papers: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    outline: list[OutlineNode] = field(default_factory=list)
    report: Report | None = None
    graph: dict | None = None
    resources: dict = field(default_factory=dict)
    window_hashes: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f)

    @staticmethod
    def from_json(path: Path) -> "Rollout":
        with open(path) as f:
            d = json.load(f)
        return _rollout_from_dict(d)

    @property
    def is_bot(self) -> bool:
        return self.status == BOT


def _rollout_from_dict(d: dict) -> Rollout:
    ctx = Context(**d["context"])
    events = []
    for e in d.get("events", []):
        lab = Label(**e.pop("label"))
        events.append(Event(label=lab, **e))
    outline = [OutlineNode(**o) for o in d.get("outline", [])]
    rep = None
    if d.get("report"):
        r = d["report"]
        rep = Report(
            sections=r.get("sections", []),
            sentences=[Sentence(**s) for s in r.get("sentences", [])],
            bibliography=r.get("bibliography", []),
            words=r.get("words", 0),
            unresolved_citations=r.get("unresolved_citations", 0),
            total_citation_marks=r.get("total_citation_marks", 0),
        )
    return Rollout(
        system=d["system"],
        task=d["task"],
        panel=d["panel"],
        context=ctx,
        status=d.get("status", "ok"),
        bot_reason=d.get("bot_reason"),
        attempt=d.get("attempt", "final"),
        mode=d.get("mode"),
        events=events,
        papers=d.get("papers", []),
        evidence_ids=d.get("evidence_ids", []),
        outline=outline,
        report=rep,
        graph=d.get("graph"),
        resources=d.get("resources", {}),
        window_hashes=d.get("window_hashes", {}),
        meta=d.get("meta", {}),
    )


def bot_rollout(system: str, task: str, panel: str, context: Context, reason: str, **kw: Any) -> Rollout:
    return Rollout(system=system, task=task, panel=panel, context=context, status=BOT, bot_reason=reason, **kw)
