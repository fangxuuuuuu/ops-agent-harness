"""Human-in-the-loop approval tests: pause, approve, reject, no side effect without approval."""

from __future__ import annotations

import json

from ops_harness.core.state import ExecutionStatus


INCIDENT_ARGS = {
    "title": "payment-api degraded",
    "description": "Automated escalation: payment-api reports degraded.",
    "severity": "high",
}


def _incident_script() -> list:
    return [
        {"call": "get_service_status", "args": {"service_name": "payment-api"}},
        {"call": "create_incident", "args": INCIDENT_ARGS},
    ]


def test_create_incident_pauses_for_approval(harness_factory, incidents_file):
    before = len(json.load(open(incidents_file, encoding="utf-8")))
    harness, store = harness_factory(_incident_script() + [{"final": "done"}])
    exec_id = harness.start("open an incident for payment-api")

    row = store.get_execution(exec_id)
    assert row["status"] == ExecutionStatus.AWAITING_APPROVAL.value

    # no side effect has happened yet
    assert len(json.load(open(incidents_file, encoding="utf-8"))) == before

    pending = store.list_approvals(status="pending")
    assert len(pending) == 1
    assert pending[0]["tool_name"] == "create_incident"
    assert json.loads(pending[0]["arguments_json"]) == INCIDENT_ARGS

    # execution history records the paused tool call
    steps = store.list_steps(exec_id)
    assert any(s["name"] == "create_incident" and s["status"] == "pending_approval" for s in steps)


def test_approval_executes_tool_and_completes(harness_factory, incidents_file):
    before = len(json.load(open(incidents_file, encoding="utf-8")))
    harness, store = harness_factory(_incident_script() + [{"final": "incident created and confirmed."}])
    exec_id = harness.start("open an incident for payment-api")

    approval = store.list_approvals(status="pending")[0]
    status = harness.decide_approval(approval["id"], "approved", "alice", "looks right")

    row = store.get_execution(exec_id)
    assert status == ExecutionStatus.COMPLETED.value
    assert row["status"] == ExecutionStatus.COMPLETED.value

    # side effect happened exactly once
    incidents = json.load(open(incidents_file, encoding="utf-8"))
    assert len(incidents) == before + 1
    assert incidents[-1]["severity"] == "high"

    # audit trail
    decided = store.get_approval(approval["id"])
    assert decided["decision"] == "approved"
    assert decided["decided_by"] == "alice"
    assert decided["decided_at"] is not None


def test_rejection_does_not_execute_and_loop_adapts(harness_factory, incidents_file):
    before = len(json.load(open(incidents_file, encoding="utf-8")))
    script = _incident_script() + [{"final": "User denied incident creation; reporting without a ticket."}]
    harness, store = harness_factory(script)
    exec_id = harness.start("open an incident for payment-api")

    approval = store.list_approvals(status="pending")[0]
    status = harness.decide_approval(approval["id"], "rejected", "bob", "not needed yet")

    row = store.get_execution(exec_id)
    assert status == ExecutionStatus.COMPLETED.value
    assert row["status"] == ExecutionStatus.COMPLETED.value
    assert "denied" in row["final_answer"]

    # no side effect
    assert len(json.load(open(incidents_file, encoding="utf-8"))) == before
    decided = store.get_approval(approval["id"])
    assert decided["decision"] == "rejected"

    # rejection was fed back to the model as an observation
    tool_msgs = [m for m in store.get_messages(exec_id) if m["role"] == "tool"]
    assert any("REJECTED" in m["content"] for m in tool_msgs)


def test_double_decision_rejected_as_conflict(harness_factory):
    harness, store = harness_factory(_incident_script() + [{"final": "ok"}])
    exec_id = harness.start("open an incident")
    approval = store.list_approvals(status="pending")[0]
    harness.decide_approval(approval["id"], "approved", "alice", None)
    try:
        harness.decide_approval(approval["id"], "rejected", "bob", None)
        raise AssertionError("expected ValueError for double decision")
    except ValueError as exc:
        assert "already decided" in str(exc)


def test_cancel_while_awaiting_approval(harness_factory):
    harness, store = harness_factory(_incident_script() + [{"final": "unreachable"}])
    exec_id = harness.start("open an incident")
    status = harness.cancel(exec_id)
    assert status == ExecutionStatus.CANCELLED.value
    row = store.get_execution(exec_id)
    assert row["status"] == ExecutionStatus.CANCELLED.value
