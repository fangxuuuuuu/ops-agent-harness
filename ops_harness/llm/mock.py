"""Deterministic scripted LLM for offline demos and tests.

A script is a list of entries consumed in order; each entry is one of:

- {"final": "answer text"}                       -> final answer, loop ends
- {"call": "tool_name", "args": {...}}           -> one tool call
- {"bad": "reason"}                              -> malformed response (tests harness resilience)
- callable(messages) -> LLMResponse              -> dynamic scripted behaviour
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ops_harness.llm.base import LLMBadResponseError, LLMClient, LLMResponse, ToolCallRequest

ScriptEntry = dict | Callable[[list[dict]], LLMResponse]


class MockLLM(LLMClient):
    def __init__(self, script: list[ScriptEntry]):
        if not script:
            raise ValueError("MockLLM requires at least one scripted entry")
        self._script = list(script)
        self._cursor = 0

    def chat(self, messages: list[dict], tool_schemas: list[dict]) -> LLMResponse:
        if self._cursor >= len(self._script):
            return LLMResponse(
                final_answer="Script exhausted: no more scripted steps.",
                raw={"mock": "script_exhausted"},
            )
        entry = self._script[self._cursor]
        self._cursor += 1
        if callable(entry):
            return entry(messages)
        if "bad" in entry:
            raise LLMBadResponseError(str(entry["bad"]))
        if "final" in entry:
            return LLMResponse(final_answer=str(entry["final"]), raw={"mock": "final"})
        if "call" in entry:
            call = ToolCallRequest(
                call_id=f"mock_{self._cursor}",
                name=str(entry["call"]),
                arguments=entry.get("args", {}),
            )
            return LLMResponse(tool_calls=[call], raw={"mock": "tool_call"})
        raise LLMBadResponseError(f"unrecognized script entry: {json.dumps(entry, default=str)[:200]}")

    @property
    def cursor(self) -> int:
        return self._cursor


def script_entry_to_response(entry: ScriptEntry) -> LLMResponse:
    """Helper used by tools building ad-hoc responses."""
    if isinstance(entry, dict) and "call" in entry:
        return LLMResponse(
            tool_calls=[
                ToolCallRequest(call_id="adhoc", name=entry["call"], arguments=entry.get("args", {}))
            ]
        )
    return LLMResponse(final_answer=str(entry) if not isinstance(entry, dict) else entry.get("final", ""))
