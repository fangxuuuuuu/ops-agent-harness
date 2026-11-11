"""Failure-path tests: transient tool errors, retries, timeouts, malformed LLM responses."""

from __future__ import annotations

import json

from ops_harness.core.state import ExecutionStatus


def test_transient_tool_error_retried_and_recovers(harness_factory, data_dir):
    with open(data_dir / "services.json", encoding="utf-8") as f:
        rows = json.load(f)
    next(r for r in rows if r["name"] == "payment-api")["fail_times"] = 2
    with open(data_dir / "services.json", "w", encoding="utf-8") as f:
        json.dump(rows, f)

    script = [
        {"call": "get_service_status", "args": {"service_name": "payment-api"}},
        {"final": "payment-api is degraded after transient errors were retried."},
    ]
    harness, store = harness_factory(script)
    exec_id = harness.start("check payment-api")

    assert store.get_execution(exec_id)["status"] == ExecutionStatus.COMPLETED.value
    tool_steps = [s for s in store.list_steps(exec_id) if s["step_type"] == "tool_call"]
    assert len(tool_steps) == 1
    assert tool_steps[0]["retry_count"] == 2
    assert tool_steps[0]["status"] == "ok"


def test_tool_failure_exhausts_retries_but_loop_survives(harness_factory, data_dir):
    with open(data_dir / "services.json", encoding="utf-8") as f:
        rows = json.load(f)
    next(r for r in rows if r["name"] == "payment-api")["fail_times"] = 99
    with open(data_dir / "services.json", "w", encoding="utf-8") as f:
        json.dump(rows, f)

    script = [
        {"call": "get_service_status", "args": {"service_name": "payment-api"}},
        {"final": "Status service is down; reporting without live status."},
    ]
    harness, store = harness_factory(script)
    exec_id = harness.start("check payment-api")

    row = store.get_execution(exec_id)
    assert row["status"] == ExecutionStatus.COMPLETED.value
    tool_steps = [s for s in store.list_steps(exec_id) if s["step_type"] == "tool_call"]
    assert tool_steps[0]["status"] == "error"
    assert tool_steps[0]["retry_count"] == 2  # exhausted configured retries


def test_slow_tool_times_out_and_loop_continues(harness_factory, data_dir):
    with open(data_dir / "services.json", encoding="utf-8") as f:
        rows = json.load(f)
    next(r for r in rows if r["name"] == "search-service")["slow_ms"] = 900
    with open(data_dir / "services.json", "w", encoding="utf-8") as f:
        json.dump(rows, f)

    script = [
        {"call": "get_service_status", "args": {"service_name": "search-service"}},
        {"final": "Status lookup timed out; treating search-service as unknown."},
    ]
    harness, store = harness_factory(script, tool_overrides={"timeout_s": 0.2, "max_retries": 1})
    exec_id = harness.start("check search-service")

    row = store.get_execution(exec_id)
    assert row["status"] == ExecutionStatus.COMPLETED.value
    tool_steps = [s for s in store.list_steps(exec_id) if s["step_type"] == "tool_call"]
    assert tool_steps[0]["status"] == "timeout"
    assert tool_steps[0]["retry_count"] == 1


def test_malformed_llm_responses_fail_after_three_attempts(harness_factory):
    script = [{"bad": "not json at all"}] * 3
    harness, store = harness_factory(script)
    exec_id = harness.start("do something")

    row = store.get_execution(exec_id)
    assert row["status"] == ExecutionStatus.FAILED.value
    assert "malformed" in row["error"]
    error_steps = [s for s in store.list_steps(exec_id) if s["step_type"] == "llm_call" and s["status"] == "error"]
    assert len(error_steps) == 3


def test_malformed_then_recovery(harness_factory):
    script = [
        {"bad": "garbage once"},
        {"final": "recovered fine"},
    ]
    harness, store = harness_factory(script)
    exec_id = harness.start("do something")
    row = store.get_execution(exec_id)
    assert row["status"] == ExecutionStatus.COMPLETED.value
    assert row["final_answer"] == "recovered fine"
