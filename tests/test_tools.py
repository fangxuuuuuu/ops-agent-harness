"""Tool-level tests: schemas, validation, fault injection, output validation."""

from __future__ import annotations

import pytest

from ops_harness.tools.base import ToolValidationError, ToolTimeoutError, execute_tool
from ops_harness.tools.kb import KBArgs
from ops_harness.tools.service_status import ServiceStatusArgs


def test_kb_search_returns_relevant_hits(data_dir):
    from ops_harness.tools.kb import make_kb_tool

    result = execute_tool(make_kb_tool(data_dir), {"query": "payment-api degraded runbook"})
    assert result.ok
    ids = [h["id"] for h in result.data["results"]]
    assert "KB-001" in ids
    assert result.data["results"][0]["score"] >= result.data["results"][-1]["score"]


def test_kb_search_empty_query_fails_validation(data_dir):
    from ops_harness.tools.kb import make_kb_tool

    result = execute_tool(make_kb_tool(data_dir), {"query": ""})
    assert not result.ok
    assert result.error["code"] == "validation_error"
    assert result.attempts == 0  # no side effect attempted


def test_service_status_found(data_dir):
    from ops_harness.tools.service_status import make_service_status_tool

    result = execute_tool(make_service_status_tool(data_dir), {"service_name": "payment-api"})
    assert result.ok
    assert result.data["status"] == "degraded"
    assert result.data["degraded"] is True


def test_service_status_unknown_service_is_non_retryable(data_dir):
    from ops_harness.tools.service_status import make_service_status_tool

    result = execute_tool(make_service_status_tool(data_dir), {"service_name": "no-such-service"})
    assert not result.ok
    assert result.error["code"] == "tool_error"
    assert result.attempts == 1


def test_service_status_transient_error_then_success(data_dir):
    from ops_harness.tools.service_status import make_service_status_tool

    tool = make_service_status_tool(data_dir)
    # fault injection: two transient failures before success
    import json

    with open(data_dir / "services.json", encoding="utf-8") as f:
        rows = json.load(f)
    next(r for r in rows if r["name"] == "auth-service")["fail_times"] = 2
    with open(data_dir / "services.json", "w", encoding="utf-8") as f:
        json.dump(rows, f)

    result = execute_tool(tool, {"service_name": "auth-service"})
    assert result.ok, result.error
    assert result.attempts == 3  # 1 initial + 2 retries


def test_incident_invalid_severity_rejected(data_dir):
    from ops_harness.tools.incident import make_incident_tool

    result = execute_tool(make_incident_tool(data_dir),
                          {"title": "x", "description": "y", "severity": "emergency"})
    assert not result.ok
    assert result.error["code"] == "validation_error"


def test_tool_output_validation_catches_bad_output(tmp_path):
    """A tool returning a non-conforming payload must produce a validation error."""
    from pydantic import BaseModel
    from ops_harness.tools.base import Tool

    class Out(BaseModel):
        value: int

    def broken_fn(args) -> dict:
        return {"value": "not-an-int"}

    class Args(BaseModel):
        pass

    tool = Tool(name="broken", description="d", args_model=Args, fn=broken_fn, output_model=Out)
    result = execute_tool(tool, {})
    assert not result.ok
    assert result.error["code"] == "validation_error"


def test_timeout_and_retries(tmp_path):
    import time as _t
    from pydantic import BaseModel
    from ops_harness.tools.base import Tool

    class Args(BaseModel):
        pass

    def slow_fn(args) -> dict:
        _t.sleep(0.8)
        return {}

    tool = Tool(name="slow", description="d", args_model=Args, fn=slow_fn,
                timeout_s=0.15, max_retries=1, backoff_base_s=0.01)
    result = execute_tool(tool, {})
    assert not result.ok
    assert result.error["code"] == "tool_timeout"
    assert result.attempts == 2
