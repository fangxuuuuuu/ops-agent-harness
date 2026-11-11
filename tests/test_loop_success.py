"""End-to-end loop tests: success path, execution history, state persistence."""

from __future__ import annotations

from ops_harness.core.state import ExecutionStatus


def test_successful_kb_objective(harness_factory):
    script = [
        {"call": "search_knowledge_base", "args": {"query": "payment-api degraded runbook", "top_k": 3}},
        {"final": "payment-api is degraded; follow KB-001 runbook and roll back if recent release."},
    ]
    harness, store = harness_factory(script)
    exec_id = harness.start("payment-api is degraded, what should I do?")

    row = store.get_execution(exec_id)
    assert row["status"] == ExecutionStatus.COMPLETED.value
    assert "KB-001" in row["final_answer"]

    steps = store.list_steps(exec_id)
    types = [s["step_type"] for s in steps]
    assert "llm_call" in types and "tool_call" in types
    tool_step = next(s for s in steps if s["step_type"] == "tool_call")
    assert tool_step["name"] == "search_knowledge_base"
    assert tool_step["status"] == "ok"
    assert tool_step["latency_ms"] >= 0

    # message history persisted (system + user + assistant tool_call + tool obs + assistant final)
    roles = [m["role"] for m in store.get_messages(exec_id)]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_multi_tool_success_flow(harness_factory):
    script = [
        {"call": "get_service_status", "args": {"service_name": "payment-api"}},
        {"call": "search_knowledge_base", "args": {"query": "payment-api incident severity"}},
        {"final": "payment-api degraded; severity high per KB-002; created nothing yet."},
    ]
    harness, store = harness_factory(script)
    exec_id = harness.start("check payment-api and advise")
    assert store.get_execution(exec_id)["status"] == ExecutionStatus.COMPLETED.value
    tool_names = [s["name"] for s in store.list_steps(exec_id) if s["step_type"] == "tool_call"]
    assert tool_names == ["get_service_status", "search_knowledge_base"]


def test_unknown_tool_is_observation_not_crash(harness_factory):
    script = [
        {"call": "restart_service", "args": {"service_name": "payment-api"}},
        {"final": "The restart tool does not exist; please do it manually."},
    ]
    harness, store = harness_factory(script)
    exec_id = harness.start("restart payment-api")
    assert store.get_execution(exec_id)["status"] == ExecutionStatus.COMPLETED.value
    steps = store.list_steps(exec_id)
    assert any(s["step_type"] == "validation_error" and s["name"] == "restart_service" for s in steps)


def test_invalid_tool_arguments_become_observation(harness_factory):
    script = [
        {"call": "create_incident", "args": {"title": "", "description": "x", "severity": "high"}},
        {"final": "Validation failed; I cannot create an incident without a title."},
    ]
    harness, store = harness_factory(script)
    exec_id = harness.start("open an incident")
    row = store.get_execution(exec_id)
    assert row["status"] == ExecutionStatus.COMPLETED.value
    assert any(s["step_type"] == "validation_error" and s["name"] == "create_incident"
               for s in store.list_steps(exec_id))
    # the invalid call never reached the approval gate
    assert store.list_approvals(status="pending", exec_id=exec_id) == []
