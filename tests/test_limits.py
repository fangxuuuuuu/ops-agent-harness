"""Loop-limit tests: step budget, wall-clock deadline, repeated identical calls, cancel."""

from __future__ import annotations

from ops_harness.core.state import ExecutionStatus
from tests.conftest import make_settings, make_registry

SERVICES = [
    "payment-api", "checkout-service", "checkout-db", "auth-service",
    "notification-service", "search-service", "inventory-service", "reporting-service",
]


def _distinct_calls_script(n: int) -> list:
    """n scripted tool calls, each with distinct arguments (avoids the identical-call guard)."""
    from ops_harness.llm.base import LLMResponse, ToolCallRequest

    entries = []
    for i in range(n):
        service = SERVICES[i % len(SERVICES)]
        call_id = f"call_{i}"

        def entry(messages, _service=service, _call_id=call_id):
            return LLMResponse(
                tool_calls=[ToolCallRequest(call_id=_call_id, name="get_service_status",
                                            arguments={"service_name": _service})]
            )

        entries.append(entry)
    return entries


def test_step_budget_stops_infinite_loop(harness_factory, tmp_path):
    harness, store = harness_factory(
        _distinct_calls_script(20), settings=make_settings(tmp_path, max_steps=4)
    )
    exec_id = harness.start("keep checking services")

    row = store.get_execution(exec_id)
    assert row["status"] == ExecutionStatus.LIMIT_REACHED.value
    assert "step budget" in row["error"]
    llm_steps = [s for s in store.list_steps(exec_id) if s["step_type"] == "llm_call"]
    assert len(llm_steps) == 4


def test_wall_clock_deadline_stops_loop(harness_factory, tmp_path):
    harness, store = harness_factory(
        _distinct_calls_script(20),
        settings=make_settings(tmp_path, time_limit_s=0.05),
        tool_overrides={"max_retries": 0},
    )
    exec_id = harness.start("keep checking services")

    row = store.get_execution(exec_id)
    assert row["status"] == ExecutionStatus.LIMIT_REACHED.value
    assert "deadline" in row["error"]


def test_repeated_identical_tool_calls_fail_fast(harness_factory, tmp_path):
    script = [{"call": "get_service_status", "args": {"service_name": "payment-api"}}] * 3
    harness, store = harness_factory(script, settings=make_settings(tmp_path, max_steps=10))
    exec_id = harness.start("check payment-api")

    row = store.get_execution(exec_id)
    assert row["status"] == ExecutionStatus.FAILED.value
    assert "identical arguments" in row["error"]


def test_cancel_before_run(harness_factory, tmp_path):
    from ops_harness.core.db import Store
    from ops_harness.core.orchestrator import Harness
    from ops_harness.llm.mock import MockLLM

    settings = make_settings(tmp_path)
    store = Store(settings.db_path)
    harness = Harness(MockLLM([{"final": "never"}]), make_registry(tmp_path / "data"), store, settings)
    exec_id = harness.create_execution("objective")
    harness.cancel(exec_id)
    assert harness.run(exec_id) == ExecutionStatus.CANCELLED.value
