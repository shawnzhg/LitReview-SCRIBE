"""Registry of the tasks the trainers run; the planning task registers itself here."""

from __future__ import annotations

TASKS: dict = {}


def get_task(name: str):
    if name not in TASKS:
        raise KeyError(f"task {name!r} is not registered ({sorted(TASKS)})")
    return TASKS[name]()
