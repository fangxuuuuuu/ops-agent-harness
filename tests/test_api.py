"""REST API tests via FastAPI TestClient (offline mock LLM)."""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from ops_harness.api import create_app
from tests.conftest import make_settings


def _poll(client, exec_id: str, want: set, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        row = client.get(f"/executions/{exec_id}").json()
        if row["status"] in want:
            return row
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for {want}; last={row}")


def test_health(data_dir, tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_full_approval_flow_over_api(data_dir, tmp_path, incidents_file):
    from ops_harness.mock_plans import demo_incident_script

    before = len(json.load(open(incidents_file, encoding="utf-8")))
    app = create_app(make_settings(tmp_path), mock_script=demo_incident_script())
    with TestClient(app) as client:
        # start execution -> runs in background -> pauses at approval gate
        resp = client.post("/executions", json={"objective": "payment-api degraded; escalate"})
        assert resp.status_code == 202
        exec_id = resp.json()["execution_id"]

        row = _poll(client, exec_id, {"AWAITING_APPROVAL"})
        assert row["status"] == "AWAITING_APPROVAL"
        assert len(json.load(open(incidents_file, encoding="utf-8"))) == before  # no side effect yet

        # step trace available
        steps = client.get(f"/executions/{exec_id}/steps").json()["steps"]
        assert any(s["name"] == "get_service_status" and s["status"] == "ok" for s in steps)
        assert any(s["name"] == "create_incident" and s["status"] == "pending_approval" for s in steps)

        # pending approval listed
        approvals = client.get("/approvals", params={"status": "pending"}).json()["approvals"]
        assert len(approvals) == 1
        approval_id = approvals[0]["id"]

        # reject -> loop resumes and completes without side effect
        resp = client.post(f"/approvals/{approval_id}/decision",
                           json={"decision": "rejected", "decided_by": "api-user", "comment": "not needed"})
        assert resp.status_code == 202

        row = _poll(client, exec_id, {"COMPLETED", "FAILED", "LIMIT_REACHED"})
        assert row["status"] == "COMPLETED"
        assert len(json.load(open(incidents_file, encoding="utf-8"))) == before  # still no incident
        assert client.get(f"/approvals/{approval_id}").json()["decision"] == "rejected"


def test_approved_incident_flow_over_api(data_dir, tmp_path, incidents_file):
    from ops_harness.mock_plans import demo_incident_script

    before = len(json.load(open(incidents_file, encoding="utf-8")))
    app = create_app(make_settings(tmp_path), mock_script=demo_incident_script())
    with TestClient(app) as client:
        exec_id = client.post("/executions", json={"objective": "escalate payment-api"}).json()["execution_id"]
        row = _poll(client, exec_id, {"AWAITING_APPROVAL"})
        assert row["status"] == "AWAITING_APPROVAL"

        approval_id = client.get("/approvals", params={"status": "pending"}).json()["approvals"][0]["id"]
        client.post(f"/approvals/{approval_id}/decision",
                    json={"decision": "approved", "decided_by": "api-user"})

        row = _poll(client, exec_id, {"COMPLETED", "FAILED", "LIMIT_REACHED"})
        assert row["status"] == "COMPLETED"
        assert len(json.load(open(incidents_file, encoding="utf-8"))) == before + 1  # incident created


def test_double_decision_conflict(data_dir, tmp_path):
    from ops_harness.mock_plans import demo_incident_script

    app = create_app(make_settings(tmp_path), mock_script=demo_incident_script())
    with TestClient(app) as client:
        exec_id = client.post("/executions", json={"objective": "escalate"}).json()["execution_id"]
        _poll(client, exec_id, {"AWAITING_APPROVAL"})
        approval_id = client.get("/approvals", params={"status": "pending"}).json()["approvals"][0]["id"]
        client.post(f"/approvals/{approval_id}/decision", json={"decision": "approved", "decided_by": "u"})
        _poll(client, exec_id, {"COMPLETED", "FAILED", "LIMIT_REACHED"})
        resp = client.post(f"/approvals/{approval_id}/decision", json={"decision": "rejected", "decided_by": "u"})
        assert resp.status_code == 409


def test_unknown_execution_404(data_dir, tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/executions/nope").status_code == 404
        assert client.get("/approvals/nope").status_code == 404


def test_cancel_over_api(data_dir, tmp_path):
    from ops_harness.mock_plans import demo_incident_script

    app = create_app(make_settings(tmp_path), mock_script=demo_incident_script())
    with TestClient(app) as client:
        exec_id = client.post("/executions", json={"objective": "escalate"}).json()["execution_id"]
        _poll(client, exec_id, {"AWAITING_APPROVAL"})
        resp = client.post(f"/executions/{exec_id}/cancel")
        assert resp.json()["status"] == "CANCELLED"
