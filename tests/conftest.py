"""Shared fixtures: isolated temp data dir + DB, and a harness factory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ops_harness.config import Settings
from ops_harness.core.db import Store
from ops_harness.core.orchestrator import Harness
from ops_harness.core.trace import get_logger
from ops_harness.llm.mock import MockLLM
from ops_harness.tools.base import ToolRegistry
from ops_harness.tools.incident import make_incident_tool
from ops_harness.tools.kb import make_kb_tool
from ops_harness.tools.service_status import make_service_status_tool

REPO_DATA = Path(__file__).resolve().parent.parent / "data"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    defaults = dict(
        provider="mock",
        db_path=str(tmp_path / "test.db"),
        data_dir=str(tmp_path / "data"),
        max_steps=10,
        time_limit_s=30.0,
        tool_timeout_s=5.0,
        tool_max_retries=2,
        log_level="ERROR",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_registry(data_dir: Path, **overrides) -> ToolRegistry:
    registry = ToolRegistry()
    kwargs = dict(timeout_s=5.0, max_retries=2, backoff_base_s=0.02)
    kwargs.update(overrides)
    registry.register(make_kb_tool(data_dir, **kwargs))
    registry.register(make_service_status_tool(data_dir, **kwargs))
    registry.register(make_incident_tool(data_dir, **kwargs))
    return registry


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Copy the repo mock dataset into an isolated temp dir."""
    dest = tmp_path / "data"
    shutil.copytree(REPO_DATA, dest)
    return dest


@pytest.fixture
def incidents_file(data_dir: Path) -> Path:
    return data_dir / "incidents.json"


def load_incidents(incidents_file: Path) -> list[dict]:
    with open(incidents_file, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def harness_factory(data_dir, tmp_path):
    """Factory: build a Harness with a scripted MockLLM and isolated storage."""
    created = []

    def _make(script: list, settings: Settings | None = None, tool_overrides: dict | None = None):
        settings = settings or make_settings(tmp_path)
        kwargs = dict(
            timeout_s=settings.tool_timeout_s,
            max_retries=settings.tool_max_retries,
            backoff_base_s=0.02,
        )
        kwargs.update(tool_overrides or {})
        registry = make_registry(data_dir, **kwargs)
        store = Store(settings.db_path)
        harness = Harness(MockLLM(script), registry, store, settings,
                          logger=get_logger("ops_harness.test", "ERROR"))
        created.append(harness)
        return harness, store

    return _make
