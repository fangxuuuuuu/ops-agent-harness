"""OpenAI-compatible chat-completions client (works with OpenAI, DeepSeek, vLLM, etc.)."""

from __future__ import annotations

import httpx

from ops_harness.llm.base import LLMBadResponseError, LLMClient, LLMResponse, ToolCallRequest, parse_arguments


class OpenAICompatibleLLM(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 60.0):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    def chat(self, messages: list[dict], tool_schemas: list[dict]) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": messages,
            "tool_choice": "auto",
        }
        if tool_schemas:
            payload["tools"] = [
                {"type": "function", "function": schema} for schema in tool_schemas
            ]
        try:
            resp = httpx.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout_s,
            )
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMBadResponseError(f"LLM HTTP failure: {exc}") from exc

        try:
            choice = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMBadResponseError(f"unexpected completion shape: {body!r:.300}") from exc

        usage = body.get("usage") or {}
        raw = {"id": body.get("id"), "finish_reason": body["choices"][0].get("finish_reason")}

        raw_tool_calls = choice.get("tool_calls") or []
        if raw_tool_calls:
            calls = [
                parse_arguments(
                    name=tc["function"]["name"],
                    raw_arguments=tc["function"].get("arguments", "{}"),
                    call_id=tc.get("id", f"call_{i}"),
                )
                for i, tc in enumerate(raw_tool_calls)
            ]
            return LLMResponse(tool_calls=calls, raw=raw, usage=usage)

        content = choice.get("content")
        if not content:
            raise LLMBadResponseError("empty assistant message: no content and no tool calls")
        return LLMResponse(final_answer=content, raw=raw, usage=usage)
