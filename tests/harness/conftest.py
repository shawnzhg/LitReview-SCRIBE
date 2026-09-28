"""Pytest fixtures of the harness tests: no pool URL in the environment, the lever writer for one
test and a synthetic task under a temporary runs root."""

from __future__ import annotations

import pytest

from harness_env import install_levers, install_task


@pytest.fixture(autouse=True)
def no_pool_url(monkeypatch):
    monkeypatch.delenv("POOL_URL", raising=False)


@pytest.fixture
def levers_installed(monkeypatch):
    return install_levers(monkeypatch)


@pytest.fixture
def task_runs(tmp_path, monkeypatch):
    return install_task(monkeypatch, tmp_path / "site")["runs"]
