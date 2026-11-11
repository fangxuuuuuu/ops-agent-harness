"""LLM client abstraction: the loop only knows this interface."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMBadResponseError(Exception):
    """LLM produced a response that cannot be parsed into a valid action."""


@dataclass
class ToolCallRequest:
    call_id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Exactly one of `final_answer` or `tool_calls` is meaningful."""

    final_answer: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None and not self.tool_calls


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tool_schemas: list[dict]) -> LLMResponse:
        """Send the conversation + tool schemas, return a parsed response."""
        ...


def parse_arguments(name: str, raw_arguments: str | dict, call_id: str = "call_0") -> ToolCallRequest:
    """Parse tool-call arguments from a JSON string or dict; raise LLMBadResponseError on garbage."""
    if isinstance(raw_arguments, dict):
        return ToolCallRequest(call_id=call_id, name=name, arguments=raw_arguments)
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        raise LLMBadResponseError(f"tool call '{name}' arguments are not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMBadResponseError(f"tool call '{name}' arguments must be a JSON object")
    return ToolCallRequest(call_id=call_id, name=name, arguments=parsed)
