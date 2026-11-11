"""Execution state machine and shared enums."""

from __future__ import annotations

from enum import Enum


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    LIMIT_REACHED = "LIMIT_REACHED"


TERMINAL_STATUSES = {
    ExecutionStatus.COMPLETED,
    ExecutionStatus.FAILED,
    ExecutionStatus.CANCELLED,
    ExecutionStatus.LIMIT_REACHED,
}


class StepType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    VALIDATION_ERROR = "validation_error"
    APPROVAL = "approval"


class StepStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
