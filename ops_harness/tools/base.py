"""Tool abstraction: schema, registry, and the guarded execution wrapper.

Execution wrapper guarantees:
- input validation against a Pydantic schema (before any side effect)
- output validation against an optional output schema
- per-call timeout (executed in a worker thread)
- retries with exponential backoff for transient errors
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------- errors

class ToolError(Exception):
    """Structured tool failure surfaced to the LLM as an observation."""

    code = "tool_error"
    retryable = False

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


class TransientToolError(ToolError):
    code = "transient_tool_error"
    retryable = True


class ToolTimeoutError(ToolError):
    code = "tool_timeout"
    retryable = True


class ToolValidationError(ToolError):
    code = "validation_error"


class UnknownToolError(ToolError):
    code = "unknown_tool"


# ---------------------------------------------------------------- tool model

class Tool:
    """A single callable tool with its schema and execution policy.

    fn receives the *validated* args model instance and returns a
    JSON-serializable dict (validated against output_model when provided).
    """

    def __init__(
        self,
        name: str,
        description: str,
        args_model: Type[BaseModel],
        fn: Callable[[BaseModel], dict],
        output_model: Type[BaseModel] | None = None,
        requires_approval: bool = False,
        timeout_s: float = 10.0,
        max_retries: int = 2,
        backoff_base_s: float = 0.25,
    ):
        self.name = name
        self.description = description
        self.args_model = args_model
        self.output_model = output_model
        self.fn = fn
        self.requires_approval = requires_approval
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s

    def openai_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(f"unknown tool '{name}'") from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def openai_schemas(self) -> list[dict]:
        return [t.openai_schema() for t in self._tools.values()]


# ---------------------------------------------------------------- execution

@dataclass
class ToolExecutionResult:
    ok: bool
    data: dict | None = None
    error: dict | None = None
    attempts: int = 0
    latency_ms: int = 0
    timed_out: bool = False
    retryable_error: bool = False

    def observation(self) -> str:
        """Render the result as the observation string fed back to the LLM."""
        if self.ok:
            return json.dumps(self.data, ensure_ascii=False)
        payload = {"status": "error", **(self.error or {})}
        return json.dumps(payload, ensure_ascii=False)


def _validate(model: Type[BaseModel], payload: dict, what: str) -> BaseModel:
    try:
        return model(**payload)
    except ValidationError as exc:
        raise ToolValidationError(f"{what} failed validation: {exc.errors()[0]['msg'] if exc.errors() else exc}") from exc


def validate_arguments(tool: Tool, arguments: dict) -> BaseModel:
    """Public helper: validate raw tool-call arguments against the tool schema."""
    return _validate(tool.args_model, dict(arguments or {}), f"arguments of tool '{tool.name}'")


def execute_tool(
    tool: Tool,
    arguments: dict,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> ToolExecutionResult:
    """Validate -> run with timeout -> retry transient failures with backoff.

    Never raises: every failure is captured in the returned result so the
    agent loop can turn it into an LLM observation.
    """
    timeout_s = tool.timeout_s if timeout_s is None else timeout_s
    max_retries = tool.max_retries if max_retries is None else max_retries

    started = time.monotonic()
    try:
        args = validate_arguments(tool, arguments)
    except ToolValidationError as exc:
        return ToolExecutionResult(ok=False, error=exc.to_dict(), latency_ms=0)

    attempts = 0
    result = ToolExecutionResult(ok=False)
    while True:
        attempts += 1
        result.attempts = attempts
        error: ToolError | None = None
        raw: dict | None = None
        executor: ThreadPoolExecutor | None = None
        try:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(tool.fn, args)
            raw = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            error = ToolTimeoutError(f"tool '{tool.name}' timed out after {timeout_s}s")
            result.timed_out = True
        except ToolError as exc:
            error = exc
        except Exception as exc:  # unexpected crash inside the tool
            error = ToolError(f"tool '{tool.name}' crashed: {type(exc).__name__}: {exc}")
        finally:
            if executor is not None:
                # Do not wait for a hung worker on timeout.
                executor.shutdown(wait=False, cancel_futures=True)

        if error is None:
            try:
                if tool.output_model is not None:
                    raw = _validate(tool.output_model, dict(raw or {}), f"output of tool '{tool.name}'").model_dump()
                elif not isinstance(raw, dict):
                    raise ToolValidationError(f"output of tool '{tool.name}' is not a JSON object")
                result.ok = True
                result.data = raw
                break
            except ToolValidationError as exc:
                error = exc  # malformed tool output: surface as validation error

        result.error = error.to_dict()
        result.retryable_error = error.retryable
        if not error.retryable or attempts > max_retries:
            break
        # exponential backoff + jitter
        time.sleep(tool.backoff_base_s * (2 ** (attempts - 1)) + random.uniform(0, 0.05))

    result.latency_ms = int((time.monotonic() - started) * 1000)
    return result
