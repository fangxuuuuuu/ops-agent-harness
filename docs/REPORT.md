# Agent Harness for an Operations Assistant — Design Report

Author: [fangxu] · Assessment: Senior AI Engineer — Agent Harness
Date: 2026-11-11

---

## 1. Objective

Build an **Agent Harness** for an operations assistant that runs a bounded, observable
LLM–tool execution loop with input/output validation, state persistence, failure handling,
human-in-the-loop approval for sensitive actions, and tests for success and failure paths.

The harness is deliberately model-agnostic and tool-agnostic:

- **Model-agnostic** — the loop talks to an `LLMClient` interface. A real OpenAI-compatible
  client and a deterministic `MockLLM` (scripted plans) both implement it. Tests and demos
  run fully offline; switching to a live model is an env-var change.
- **Tool-agnostic** — tools are registered objects carrying a JSON Schema (derived from a
  Pydantic model), an execution policy (timeout, retries), and a safety flag
  (`requires_approval`). Adding a fourth tool requires zero harness changes.

## 2. Architecture

```
┌────────────┐   ┌────────────────────────────────────────────────┐
│  CLI       │   │                  FastAPI (REST)                │
└─────┬──────┘   └───────────────┬────────────────────────────────┘
      │                          │ /executions  /approvals  /health
      ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Harness Orchestrator                        │
│  ┌───────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ State     │  │ Agent Loop   │  │ Guardrails                │ │
│  │ Machine   │→ │ (ReAct loop) │→ │ • input/output validation │ │
│  │ (persist) │  │ step budget  │  │ • timeout / retry policy  │ │
│  └───────────┘  │ wall clock   │  │ • approval gate           │ │
│                 └──────┬───────┘  │ • loop-injection guard    │ │
│                        ▼          └───────────────────────────┘ │
│  ┌───────────────┐      ┌─────────────────┐                     │
│  │ Trace / Logs  │      │ Tool Registry   │                     │
│  │ (JSON events) │      │ search_knowledge_base /             │
│  └───────────────┘      │ get_service_status /                │
│                          │ create_incident (approval)         │
└─────────────────────────────────────────────────────────────────┘
          │                        │                    │
          ▼                        ▼                    ▼
   ┌─────────────┐        ┌──────────────┐      ┌─────────────┐
   │ LLMClient   │        │  SQLite DB   │      │ Mock data   │
   │ mock/openai │        │ (state+audit)│      │ (JSON files)│
   └─────────────┘        └──────────────┘      └─────────────┘
```

### 2.1 Agent loop (control loop)

One loop iteration ("step") is:

1. **Assemble context** — system prompt (tool schemas, policies) + persisted message history.
2. **Call the LLM** — response must be either a final answer or a tool call, parsed from
   OpenAI-style tool-calling format (or JSON for the mock). Malformed responses are caught,
   logged, and fed back to the model as a corrective observation (max 3 strikes, then fail).
3. **Validate** — tool arguments are validated against the tool's Pydantic schema *before*
   execution; validation errors go back to the LLM as an observation, not a crash.
4. **Safety gate** — if the tool `requires_approval`, the execution pauses in
   `AWAITING_APPROVAL`; the loop resumes on `approve` (executes the call) or `reject`
   (feeds "user denied this action" as an observation so the model can still answer).
5. **Execute tool** — with per-call timeout and retry w/ exponential backoff + jitter.
   Tool errors are returned to the LLM as observations (the model may adapt), while
   transport-level exhaustion is recorded as a failed step.
6. **Record** — every LLM call, tool call, validation error, approval decision is one row in
   `steps` + one trace event (JSON logs).
7. **Check limits** — `max_steps` and a wall-clock deadline; breaching either stops the loop
   with status `LIMIT_REACHED` and a partial-answer summary.

Termination: final answer from the model, repeated-tool-call guard (same tool+args 3×),
limits reached, unrecoverable error, or explicit cancel.

### 2.2 Execution state machine

```
PENDING → RUNNING → AWAITING_APPROVAL → RUNNING → COMPLETED
                ↘ FAILED / CANCELLED / LIMIT_REACHED ↗
```

State transitions are persisted, so an execution can be **resumed after approval** even if
the API process restarted (state lives in SQLite, not in memory).

## 3. Technology Stack

| Layer        | Choice                              | Why |
|--------------|-------------------------------------|-----|
| Language     | Python 3.10+                        | JD requirement, ecosystem for LLM tooling |
| API          | FastAPI + Uvicorn                   | typed, async, auto OpenAPI docs |
| CLI          | argparse (stdlib)                   | zero-dep, covers "accept an objective through API **or** CLI" |
| Validation   | Pydantic v2                         | schemas for tool args/outputs, JSON Schema export |
| LLM access   | OpenAI-compatible HTTP interface    | portable across providers; `MockLLM` for offline determinism |
| Storage      | SQLite (stdlib `sqlite3`)           | zero-ops; WAL mode; schema below |
| Observability| stdlib logging → JSON trace events  | one structured event per step, execution_id correlation |
| Testing      | pytest + fixtures                   | success/failure/approval/limit cases |
| Packaging    | standard `pyproject.toml` (pip -e)  | reproducible install |

## 4. Database Design (SQLite)

Three tables — one row per execution, append-only per-step history, explicit approval audit:

**executions**

| column | type | notes |
|---|---|---|
| id | TEXT PK | uuid4 |
| objective | TEXT | user's goal |
| status | TEXT | PENDING / RUNNING / AWAITING_APPROVAL / COMPLETED / FAILED / CANCELLED / LIMIT_REACHED |
| max_steps | INTEGER | loop budget |
| deadline_at | TEXT | wall-clock limit |
| final_answer | TEXT NULL | set on COMPLETED |
| error | TEXT NULL | set on FAILED |
| created_at / updated_at | TEXT | UTC ISO-8601 |

**steps** (append-only execution history)

| column | type | notes |
|---|---|---|
| id | TEXT PK | uuid4 |
| execution_id | TEXT FK → executions | |
| seq | INTEGER | ordering |
| step_type | TEXT | llm_call / tool_call / validation_error / approval |
| name | TEXT | tool name or "llm" |
| request_json / response_json | TEXT | full payload (redacted of API keys) |
| status | TEXT | ok / error / timeout / pending_approval / rejected |
| latency_ms / retry_count | INTEGER | perf + retry observability |
| error | TEXT NULL | |
| created_at | TEXT | |

**approvals** (human checkpoint audit trail)

| column | type | notes |
|---|---|---|
| id | TEXT PK | |
| execution_id | TEXT FK | |
| tool_name / arguments_json | TEXT | what was about to run |
| decision | TEXT | approved / rejected |
| decided_by | TEXT | extensible identity |
| comment | TEXT NULL | |
| created_at / decided_at | TEXT | |

Indexes: `steps(execution_id, seq)`, `approvals(execution_id)`.

## 5. API Surface (see Postman collection)

| Method | Path | Purpose |
|---|---|---|
| POST | `/executions` | start execution `{objective}` → 202 with id |
| GET | `/executions/{id}` | status + final answer |
| GET | `/executions/{id}/steps` | full step/trace history |
| GET | `/approvals?status=pending` | list pending approvals |
| POST | `/approvals/{id}/decision` | `{decision: approved|rejected, decided_by, comment}` |
| POST | `/executions/{id}/cancel` | cooperative cancel |
| GET | `/health` | liveness |

CLI mirrors the same operations: `python -m ops_harness run "objective"`, `... approve <id>`, `... status <id>`, `... trace <id>`.

## 6. Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_LLM_PROVIDER` | `mock` | `mock` (offline scripted) or `openai` (any OpenAI-compatible endpoint) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | provider base URL (DeepSeek/vLLM/etc. compatible) |
| `OPENAI_API_KEY` | — | required when provider=openai |
| `AGENT_MODEL` | `gpt-4o-mini` | model id |
| `AGENT_MAX_STEPS` | `10` | loop step budget |
| `AGENT_TIME_LIMIT_SECONDS` | `120` | wall-clock deadline per execution |
| `TOOL_TIMEOUT_SECONDS` | `10` | per tool call |
| `TOOL_MAX_RETRIES` | `2` | retries per tool call (exponential backoff) |
| `OPS_DB_PATH` | `./ops_harness.db` | SQLite location |
| `LOG_LEVEL` | `INFO` | log verbosity |
| `API_HOST` / `API_PORT` | `127.0.0.1:8000` | server bind |

## 7. Mock Dataset

Shipped in the repository under `data/` and packaged as
[ops-agent-harness-mock-dataset.zip](https://drive.google.com/file/d/1inZNz2YlQEmYlNIvPWxT_vrJjZczUhLr/view?usp=sharing) (Google Drive, public):

- `kb_articles.json` — 12 internal-doc articles (runbooks, on-call guides) for `search_knowledge_base` (keyword/BM25-lite retrieval).
- `services.json` — 8 services with health, region, owner, last-deployment for `get_service_status`.
- `incidents.json` — write-target for `create_incident` (seeded with 2 historical incidents).

## 8. Safety & Error Handling Summary

- **Input validation** — Pydantic schema per tool; rejects wrong types/missing fields before any side effect.
- **Output validation** — tool return is schema-checked; a malformed tool response becomes a typed error, not a crash.
- **Timeouts** — per tool call (`TOOL_TIMEOUT_SECONDS`), simulated slow tools included in tests.
- **Retries** — transient tool errors retried with exponential backoff + jitter; non-retryable (validation) errors are not.
- **Malformed LLM responses** — repaired/validated; 3 consecutive malformed responses fail the run instead of looping.
- **Infinite-loop protection** — `max_steps`, wall-clock deadline, and repeated-identical-tool-call detection.
- **Approval gate** — `create_incident` cannot execute without an explicit human `approved` decision; rejection is fed back to the model so it can still produce a useful answer.
- **Prompt-injection containment (baseline)** — tool results are inserted as *data* (never concatenated into instructions), and the system prompt instructs the model to treat tool output as untrusted.

## 9. Limitations

1. **Single-node, single-process** — SQLite and in-process resumption; no distributed workers. Approval resumes work only via the same deployment.
2. **No authN/authZ on the API** — fine for the assessment; production needs tokens + per-user approval identity.
3. **Retrieval is keyword-based** — the KB mock uses lexical matching, not embeddings/RAG with reranking.
4. **Sequential tool calls** — one tool call per step; no parallel tool execution.
5. **Mock LLM is scripted** — deterministic plans, not real reasoning; live-model behavior (latency, cost, hallucination) is only partially represented.
6. **No streaming** — final answer only at completion; no token streaming to clients.
7. **Cost/token accounting is basic** — usage is logged when the provider returns it; no budget capping.
8. **English-only prompts/policies.**

## 10. Future Improvements

1. **Observability upgrade** — OpenTelemetry spans + Langfuse/Cloud Trace export (trace_id per execution, spans per step), Prometheus metrics for cost/latency/failure-rate.
2. **Evaluation suite in CI** — golden dataset of objectives → expected tool traces; LLM-as-judge scoring with human-calibrated rubric; regression gate on GitLab CI.
3. **Stronger guardrails** — PII detection/redaction on tool I/O, prompt-injection classifiers on tool outputs, output policy checks (schema + tone + policy rules).
4. **Platform integration** — expose tools via **MCP servers**; support **A2A** for agent-to-agent delegation (matches the platform's interoperability roadmap).
5. **Durable execution** — Postgres + worker queue (e.g., Temporal-style durable loops) so executions survive process death mid-tool-call; multi-tenant quotas.
6. **Parallel tool calls & planning** — allow the model to batch independent calls; add a lightweight planner for multi-objective requests.
7. **RAG upgrade** — embeddings + reranking for the knowledge base, incremental indexing pipeline.
8. **Streaming + webhooks** — SSE/WebSocket progress and webhook callbacks on approval requests (Slack/Teams HITL).

## 11. How to Run & Test (short)

```bash
pip install -e ".[dev]"            # install
uvicorn ops_harness.api:app --reload   # API mode
python -m ops_harness.cli run "Check the status of payment-api and open an incident if it is degraded"  # CLI mode
pytest                             # test suite
```

Postman: import `postman/OpsAgentHarness.postman_collection.json`. Environment variables are listed in `.env.example` and Section 6.
