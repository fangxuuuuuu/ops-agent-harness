"""The agent loop: a bounded, resumable LLM–tool control loop.

Guarantees implemented here:
- step budget (`max_steps`) and wall-clock deadline -> LIMIT_REACHED
- malformed LLM responses -> corrective observation; 3 consecutive -> FAILED
- tool input/output validation errors -> observation, not crash
- repeated identical tool calls (same tool + same args, 3x) -> FAILED
- approval gate: requires_approval tools pause the loop in AWAITING_APPROVAL
- every action is persisted as a step + a structured trace event
- the loop is re-entrant: state lives in the Store, so approval resumes work
  even from a new process.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

from ops_harness.config import Settings
from ops_harness.core.db import Store, utcnow_iso
from ops_harness.core.prompts import build_system_prompt
from ops_harness.core.state import ExecutionStatus, StepStatus, StepType
from ops_harness.core.trace import trace
from ops_harness.llm.base import LLMBadResponseError, LLMClient
from ops_harness.tools.base import ToolRegistry, ToolExecutionResult, ToolValidationError, UnknownToolError, execute_tool, validate_arguments

MAX_CONSECUTIVE_MALFORMED = 3
MAX_IDENTICAL_TOOL_CALLS = 3


class Harness:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        store: Store,
        settings: Settings,
        logger=None,
    ):
        self._llm = llm
        self._registry = registry
        self._store = store
        self._settings = settings
        self._log = logger
        self._schemas = registry.openai_schemas()

    # ------------------------------------------------------------ lifecycle

    def create_execution(self, objective: str, meta: str | None = None) -> str:
        """Persist a PENDING execution with system prompt + objective. Does not run."""
        exec_id = uuid.uuid4().hex
        deadline = datetime.now(timezone.utc) + timedelta(seconds=self._settings.time_limit_s)
        self._store.create_execution(exec_id, objective, self._settings.max_steps, deadline.isoformat(), meta)
        self._store.add_message(exec_id, "system", build_system_prompt(self._schemas))
        self._store.add_message(exec_id, "user", objective)
        self._trace("execution_created", execution_id=exec_id, objective=objective)
        return exec_id

    def run(self, exec_id: str) -> str:
        """Run (or resume) the loop for an execution; returns the resulting status."""
        row = self._store.get_execution(exec_id)
        if row is None:
            raise ValueError(f"execution '{exec_id}' not found")
        status = ExecutionStatus(row["status"])
        if status in (ExecutionStatus.PENDING, ExecutionStatus.AWAITING_APPROVAL):
            self._store.update_execution(exec_id, status=ExecutionStatus.RUNNING.value)
        return self._run_loop(exec_id)

    def start(self, objective: str, meta: str | None = None) -> str:
        """Create + run in one call (CLI convenience). Returns execution id."""
        exec_id = self.create_execution(objective, meta)
        self.run(exec_id)
        return exec_id

    def cancel(self, exec_id: str) -> str:
        row = self._store.get_execution(exec_id)
        if row is None:
            raise ValueError(f"execution '{exec_id}' not found")
        status = ExecutionStatus(row["status"])
        if status in (ExecutionStatus.PENDING, ExecutionStatus.RUNNING, ExecutionStatus.AWAITING_APPROVAL):
            self._store.update_execution(exec_id, status=ExecutionStatus.CANCELLED.value, error="cancelled by user")
            self._trace("execution_cancelled", execution_id=exec_id)
        return self._store.get_execution(exec_id)["status"]

    def decide_approval(
        self, approval_id: str, decision: str, decided_by: str, comment: str | None = None
    ) -> str:
        """Apply a human decision and resume the loop if needed. Returns execution status."""
        approval = self._store.get_approval(approval_id)
        if approval is None:
            raise ValueError(f"approval '{approval_id}' not found")
        if approval["decision"] is not None:
            raise ValueError(f"approval '{approval_id}' already decided ({approval['decision']})")
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")

        exec_id = approval["execution_id"]
        self._store.update_approval(approval_id, decision, decided_by, comment)
        self._store.add_step(
            exec_id, StepType.APPROVAL.value, approval["tool_name"], StepStatus.OK.value,
            request_json=approval["arguments_json"],
            response_json=json.dumps({"decision": decision, "decided_by": decided_by, "comment": comment}),
        )
        self._trace(
            "approval_decision", execution_id=exec_id, approval_id=approval_id,
            tool=approval["tool_name"], decision=decision, decided_by=decided_by,
        )

        if decision == "rejected":
            self._store.add_message(
                exec_id, "tool",
                json.dumps({"status": "rejected", "message":
                            "The user REJECTED this action. Do not retry it; adapt your answer without it."}),
                tool_call_id=f"approval_{approval_id}",
            )
        else:
            result = execute_tool(self._registry.get(approval["tool_name"]),
                                  json.loads(approval["arguments_json"]))
            self._record_tool_step(exec_id, approval["tool_name"], result, approval_id=approval_id)
            self._store.add_message(
                exec_id, "tool", result.observation(),
                tool_call_id=f"approval_{approval_id}",
            )

        self._store.update_execution(exec_id, status=ExecutionStatus.RUNNING.value)
        return self._run_loop(exec_id)

    # ------------------------------------------------------------ the loop

    def _run_loop(self, exec_id: str) -> str:
        deadline = datetime.fromisoformat(self._store.get_execution(exec_id)["deadline_at"])
        identical_counts: dict[str, int] = {}
        consecutive_malformed = 0

        while True:
            row = self._store.get_execution(exec_id)
            status = ExecutionStatus(row["status"])
            if status == ExecutionStatus.CANCELLED:
                return status.value
            if status in (
                ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.LIMIT_REACHED
            ):
                return status.value

            # --- loop limits -------------------------------------------
            iterations = self._store.count_steps(exec_id, StepType.LLM_CALL.value)
            if iterations >= self._settings.max_steps:
                return self._finish(exec_id, ExecutionStatus.LIMIT_REACHED,
                                    error=f"step budget exhausted ({self._settings.max_steps} LLM steps)")
            if datetime.now(timezone.utc) > deadline:
                return self._finish(exec_id, ExecutionStatus.LIMIT_REACHED, error="wall-clock deadline exceeded")

            # --- LLM call ----------------------------------------------
            messages = self._store.get_messages(exec_id)
            step_id = self._store.add_step(
                exec_id, StepType.LLM_CALL.value, "llm", StepStatus.OK.value,
                request_json=json.dumps({"messages_count": len(messages)}),
            )
            try:
                resp = self._llm.chat(messages, self._schemas)
            except LLMBadResponseError as exc:
                consecutive_malformed += 1
                self._store.update_step(step_id, status=StepStatus.ERROR.value, error=str(exc))
                self._trace("llm_malformed_response", execution_id=exec_id, step_id=step_id,
                            attempt=consecutive_malformed, detail=str(exc))
                if consecutive_malformed >= MAX_CONSECUTIVE_MALFORMED:
                    return self._finish(exec_id, ExecutionStatus.FAILED,
                                        error=f"LLM returned malformed responses {consecutive_malformed}x in a row")
                self._store.add_message(
                    exec_id, "user",
                    "Your previous response was malformed and could not be parsed. "
                    "Respond either with a tool call using the provided tool schemas or with a final answer.",
                )
                continue
            consecutive_malformed = 0
            self._store.update_step(
                step_id,
                response_json=json.dumps(resp.raw, ensure_ascii=False),
            )
            self._trace("llm_call", execution_id=exec_id, step_id=step_id,
                        is_final=resp.is_final, usage=resp.usage)

            # --- final answer? ------------------------------------------
            if resp.is_final:
                self._store.add_message(exec_id, "assistant", resp.final_answer)
                self._store.update_execution(
                    exec_id, status=ExecutionStatus.COMPLETED.value, final_answer=resp.final_answer
                )
                self._trace("execution_completed", execution_id=exec_id)
                return ExecutionStatus.COMPLETED.value

            # --- tool call (one per step) -------------------------------
            call = resp.tool_calls[0]
            if len(resp.tool_calls) > 1:
                self._trace("tool_calls_truncated", execution_id=exec_id,
                            ignored=len(resp.tool_calls) - 1)

            # repeated identical tool call guard
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
            identical_counts[key] = identical_counts.get(key, 0) + 1
            if identical_counts[key] >= MAX_IDENTICAL_TOOL_CALLS:
                return self._finish(
                    exec_id, ExecutionStatus.FAILED,
                    error=f"tool '{call.name}' called {MAX_IDENTICAL_TOOL_CALLS} times with identical arguments",
                )

            self._store.add_message(
                exec_id, "assistant",
                json.dumps({"tool_call": call.name, "arguments": call.arguments, "call_id": call.call_id}),
            )

            try:
                tool = self._registry.get(call.name)
            except UnknownToolError as exc:
                observation = json.dumps({"status": "error", **exc.to_dict()})
                self._store.add_step(exec_id, StepType.VALIDATION_ERROR.value, call.name,
                                     StepStatus.ERROR.value, error=str(exc))
                self._store.add_message(exec_id, "tool", observation, tool_call_id=call.call_id)
                self._trace("tool_unknown", execution_id=exec_id, tool=call.name)
                continue

            self._trace("tool_call_start", execution_id=exec_id, tool=call.name, arguments=call.arguments)

            # validate arguments BEFORE the approval gate so a human is never
            # asked to approve a call that would be rejected anyway
            try:
                validate_arguments(tool, call.arguments)
            except ToolValidationError as exc:
                self._store.add_step(exec_id, StepType.VALIDATION_ERROR.value, call.name,
                                     StepStatus.ERROR.value, error=str(exc))
                self._store.add_message(exec_id, "tool",
                                        json.dumps({"status": "error", **exc.to_dict()}),
                                        tool_call_id=call.call_id)
                self._trace("tool_args_invalid", execution_id=exec_id, tool=call.name, detail=str(exc))
                continue

            if tool.requires_approval:
                approval_id = self._store.create_approval(exec_id, call.name,
                                                          json.dumps(call.arguments, ensure_ascii=False))
                self._store.add_step(
                    exec_id, StepType.TOOL_CALL.value, call.name, StepStatus.PENDING_APPROVAL.value,
                    request_json=json.dumps(call.arguments, ensure_ascii=False),
                    response_json=json.dumps({"approval_id": approval_id}),
                )
                self._store.update_execution(exec_id, status=ExecutionStatus.AWAITING_APPROVAL.value)
                self._trace("approval_required", execution_id=exec_id, approval_id=approval_id,
                            tool=call.name, arguments=call.arguments)
                return ExecutionStatus.AWAITING_APPROVAL.value

            result = execute_tool(tool, call.arguments)
            self._record_tool_step(exec_id, call.name, result)
            self._store.add_message(exec_id, "tool", result.observation(), tool_call_id=call.call_id)
            self._trace("tool_call_end", execution_id=exec_id, tool=call.name,
                        ok=result.ok, attempts=result.attempts, latency_ms=result.latency_ms)

    # ------------------------------------------------------------ helpers

    def _record_tool_step(self, exec_id: str, tool_name: str, result: ToolExecutionResult,
                          approval_id: str | None = None) -> None:
        if result.ok:
            status = StepStatus.OK.value
        elif result.timed_out:
            status = StepStatus.TIMEOUT.value
        elif result.error and result.error.get("code") == "validation_error":
            status = StepStatus.ERROR.value
        else:
            status = StepStatus.ERROR.value
        self._store.add_step(
            exec_id, StepType.TOOL_CALL.value, tool_name, status,
            request_json=None,
            response_json=json.dumps(result.data, ensure_ascii=False) if result.ok else json.dumps(result.error),
            latency_ms=result.latency_ms,
            retry_count=max(result.attempts - 1, 0),
        )

    def _finish(self, exec_id: str, status: ExecutionStatus, error: str) -> str:
        self._store.update_execution(exec_id, status=status.value, error=error)
        self._trace("execution_finished", execution_id=exec_id, status=status.value, error=error)
        return status.value

    def _trace(self, event: str, **fields) -> None:
        if self._log is not None:
            trace(self._log, event, **fields)
