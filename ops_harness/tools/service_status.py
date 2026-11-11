"""Mock tool: get_service_status — service health lookup with fault injection.

Dataset rows support deterministic fault injection for tests:
- "fail_times": N   -> the first N calls raise a transient error (then succeed)
- "slow_ms": N      -> the call sleeps N ms (to trigger harness timeouts)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel

from ops_harness.tools.base import Tool, ToolError, TransientToolError


class ServiceStatusArgs(BaseModel):
    service_name: str


class ServiceStatusOutput(BaseModel):
    service: str
    status: str
    region: str
    owner: str
    last_deployment: str
    degraded: bool


def _load(data_dir: Path) -> list[dict]:
    with open(data_dir / "services.json", encoding="utf-8") as f:
        return json.load(f)


def _save(data_dir: Path, rows: list[dict]) -> None:
    with open(data_dir / "services.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def make_service_status_tool(data_dir: Path, timeout_s: float = 10.0, max_retries: int = 2,
                             backoff_base_s: float = 0.25) -> Tool:
    def fn(args: ServiceStatusArgs) -> dict:
        rows = _load(data_dir)
        row = next(
            (r for r in rows if r["name"].lower() == args.service_name.lower()), None
        )
        if row is None:
            raise ToolError(f"service '{args.service_name}' not found in service registry")

        if row.get("fail_times", 0) > 0:
            row["fail_times"] = row["fail_times"] - 1
            _save(data_dir, rows)
            raise TransientToolError(
                f"status service returned 503 for '{row['name']}' (upstream temporarily unavailable)"
            )
        if row.get("slow_ms"):
            time.sleep(row["slow_ms"] / 1000.0)

        return ServiceStatusOutput(
            service=row["name"],
            status=row["status"],
            region=row["region"],
            owner=row["owner"],
            last_deployment=row["last_deployment"],
            degraded=row["status"] != "healthy",
        ).model_dump()

    return Tool(
        name="get_service_status",
        description="Retrieve the current status of a service by name "
        "(health, region, owner team, last deployment).",
        args_model=ServiceStatusArgs,
        fn=fn,
        output_model=ServiceStatusOutput,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
    )
