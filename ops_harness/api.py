"""FastAPI REST surface for the agent harness."""

from __future__ import annotations

import threading

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ops_harness.config import Settings, load_settings
from ops_harness.core.db import Store
from ops_harness.core.orchestrator import Harness
from ops_harness.core.trace import get_logger
from ops_harness.llm.mock import MockLLM
from ops_harness.tools.base import ToolRegistry
from ops_harness.tools.incident import make_incident_tool
from ops_harness.tools.kb import make_kb_tool
from ops_harness.tools.service_status import make_service_status_tool


# ---------------------------------------------------------------- requests/responses

class ExecutionCreate(BaseModel):
    objective: str = Field(..., min_length=1)


class ApprovalDecision(BaseModel):
    decision: str = Field(..., pattern="^(approved|rejected)$")
    decided_by: str = Field(..., min_length=1)
    comment: str | None = None


class ExecutionOut(BaseModel):
    id: str
    objective: str
    status: str
    final_answer: str | None
    error: str | None
    created_at: str
    updated_at: str


# ---------------------------------------------------------------- wiring

def build_harness(settings: Settings | None = None, mock_script: list | None = None) -> Harness:
    settings = settings or load_settings()
    store = Store(settings.db_path)
    registry = ToolRegistry()
    data_dir = settings.data_path
    registry.register(make_kb_tool(data_dir, timeout_s=settings.tool_timeout_s,
                                   max_retries=settings.tool_max_retries))
    registry.register(make_service_status_tool(data_dir, timeout_s=settings.tool_timeout_s,
                                               max_retries=settings.tool_max_retries))
    registry.register(make_incident_tool(data_dir, timeout_s=settings.tool_timeout_s,
                                         max_retries=settings.tool_max_retries))
    if settings.provider == "openai":
        from ops_harness.llm.openai_client import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM(settings.openai_base_url, settings.openai_api_key, settings.model)
    else:
        llm = MockLLM(mock_script or default_demo_script())
    logger = get_logger("ops_harness.api", settings.log_level)
    return Harness(llm, registry, store, settings, logger), store


def default_demo_script() -> list:
    """Offline demo plan for the mock provider.

    Uses the incident plan so the API demo exercises the full flow:
    service status -> knowledge base -> create_incident approval gate.
    """
    from ops_harness.mock_plans import demo_incident_script

    return demo_incident_script()


# ---------------------------------------------------------------- app

def create_app(settings: Settings | None = None, mock_script: list | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Ops Agent Harness", version="0.1.0",
                  description="Bounded LLM–tool agent loop with validation, approvals and tracing.")
    harness, store = build_harness(settings, mock_script=mock_script)

    def _run_in_background(exec_id: str) -> None:
        threading.Thread(target=harness.run, args=(exec_id,), daemon=True).start()

    @app.post("/executions", status_code=202)
    def create_execution(payload: ExecutionCreate, background: BackgroundTasks):
        exec_id = harness.create_execution(payload.objective)
        background.add_task(_run_in_background, exec_id)
        return {"execution_id": exec_id, "status": "PENDING",
                "poll": f"/executions/{exec_id}"}

    @app.get("/executions/{exec_id}", response_model=ExecutionOut)
    def get_execution(exec_id: str):
        row = store.get_execution(exec_id)
        if row is None:
            raise HTTPException(404, "execution not found")
        return ExecutionOut(
            id=row["id"], objective=row["objective"], status=row["status"],
            final_answer=row["final_answer"], error=row["error"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @app.get("/executions/{exec_id}/steps")
    def get_steps(exec_id: str):
        if store.get_execution(exec_id) is None:
            raise HTTPException(404, "execution not found")
        return {"execution_id": exec_id, "steps": store.list_steps(exec_id)}

    @app.get("/executions")
    def list_executions(limit: int = Query(50, ge=1, le=500)):
        return {"executions": store.list_executions(limit)}

    @app.get("/approvals")
    def list_approvals(status: str | None = Query(None, pattern="^(pending|approved|rejected)$"),
                       execution_id: str | None = None):
        return {"approvals": store.list_approvals(status, execution_id)}

    @app.get("/approvals/{approval_id}")
    def get_approval(approval_id: str):
        row = store.get_approval(approval_id)
        if row is None:
            raise HTTPException(404, "approval not found")
        return row

    @app.post("/approvals/{approval_id}/decision", status_code=202)
    def decide(approval_id: str, payload: ApprovalDecision, background: BackgroundTasks):
        approval = store.get_approval(approval_id)
        if approval is None:
            raise HTTPException(404, "approval not found")
        if approval["decision"] is not None:
            raise HTTPException(409, f"approval already decided ({approval['decision']})")
        try:
            background.add_task(
                harness.decide_approval, approval_id, payload.decision,
                payload.decided_by, payload.comment,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"approval_id": approval_id, "decision": payload.decision,
                "execution_id": approval["execution_id"], "status": "RESUMING"}

    @app.post("/executions/{exec_id}/cancel")
    def cancel(exec_id: str):
        try:
            return {"execution_id": exec_id, "status": harness.cancel(exec_id)}
        except ValueError as exc:
            raise HTTPException(404, str(exc))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
