"""Scripted demo plans for the offline MockLLM (used by CLI and API demo mode).

`incident` plan demonstrates the full approval flow:
  1. check service status  2. search KB  3. propose create_incident (approval gate)
  4. after decision, adapt the final answer based on the tool observation.
"""

from __future__ import annotations

import json

from ops_harness.llm.base import LLMResponse, ToolCallRequest


def _last_user_content(messages: list[dict]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "ops")


def _tool_observation(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "tool":
            return m["content"]
    return "{}"


def demo_kb_script() -> list:
    def call_kb(messages):
        return LLMResponse(
            tool_calls=[ToolCallRequest(
                call_id="demo_kb", name="search_knowledge_base",
                arguments={"query": _last_user_content(messages)[:200], "top_k": 3},
            )]
        )

    def final(messages):
        obs = _tool_observation(messages)
        return LLMResponse(
            final_answer="Demo (mock LLM) — searched the knowledge base for your objective; "
                         f"retrieved: {obs[:300]}. Configure AGENT_LLM_PROVIDER=openai for real reasoning.",
        )

    return [call_kb, final]


def demo_incident_script() -> list:
    def call_status(messages):
        return LLMResponse(
            tool_calls=[ToolCallRequest(
                call_id="demo_status", name="get_service_status",
                arguments={"service_name": "payment-api"},
            )]
        )

    def call_kb(messages):
        return LLMResponse(
            tool_calls=[ToolCallRequest(
                call_id="demo_kb", name="search_knowledge_base",
                arguments={"query": "payment-api incident degraded runbook", "top_k": 2},
            )]
        )

    def propose_incident(messages):
        status = _tool_observation(messages)
        return LLMResponse(
            tool_calls=[ToolCallRequest(
                call_id="demo_incident", name="create_incident",
                arguments={
                    "title": "payment-api degraded",
                    "description": f"Automated escalation. Service status observation: {status[:400]}",
                    "severity": "high",
                },
            )]
        )

    def final(messages):
        obs = _tool_observation(messages)
        data = json.loads(obs) if obs.startswith("{") else {}
        if data.get("status") == "error" or data.get("incident_id"):
            verdict = f"Incident tool result: {obs[:300]}"
        else:
            verdict = f"Incident tool result: {obs[:300]}"
        return LLMResponse(
            final_answer="Demo (mock LLM) — checked payment-api, consulted the runbook, "
                         f"and proposed an incident. {verdict} "
                         "Configure AGENT_LLM_PROVIDER=openai for real reasoning."
        )

    return [call_status, call_kb, propose_incident, final]


SCRIPT_BUILDERS = {
    "kb": demo_kb_script,
    "incident": demo_incident_script,
}


def get_script(plan: str) -> list:
    """Build a scripted plan by name; raises ValueError for unknown plans."""
    if plan not in SCRIPT_BUILDERS:
        raise ValueError(f"unknown demo plan '{plan}' (known: {sorted(SCRIPT_BUILDERS)})")
    return SCRIPT_BUILDERS[plan]()
