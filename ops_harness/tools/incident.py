"""Mock tool: create_incident — side-effecting action gated by human approval."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ops_harness.tools.base import Tool


class IncidentArgs(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    severity: Literal["low", "medium", "high", "critical"]


class IncidentOutput(BaseModel):
    incident_id: str
    status: str
    severity: str


def _load(data_dir: Path) -> list[dict]:
    with open(data_dir / "incidents.json", encoding="utf-8") as f:
        return json.load(f)


def _save(data_dir: Path, rows: list[dict]) -> None:
    with open(data_dir / "incidents.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def make_incident_tool(data_dir: Path, timeout_s: float = 10.0, max_retries: int = 2,
                       backoff_base_s: float = 0.25) -> Tool:
    def fn(args: IncidentArgs) -> dict:
        rows = _load(data_dir)
        incident = {
            "id": f"INC-{uuid.uuid4().hex[:8]}",
            "title": args.title,
            "description": args.description,
            "severity": args.severity,
            "status": "open",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(incident)
        _save(data_dir, rows)
        return IncidentOutput(
            incident_id=incident["id"], status="created", severity=args.severity
        ).model_dump()

    return Tool(
        name="create_incident",
        description="Create an incident in the external incident system. "
        "SENSITIVE: requires explicit human approval before execution.",
        args_model=IncidentArgs,
        fn=fn,
        output_model=IncidentOutput,
        requires_approval=True,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
    )
