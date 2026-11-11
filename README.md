# Ops Agent Harness

An **Agent Harness** for an operations assistant: a bounded, observable LLM–tool
execution loop with input/output validation, persisted state, failure handling
(timeouts, retries, malformed LLM responses), loop limits, and a human-in-the-loop
approval gate for sensitive actions (`create_incident`).

> Design rationale, database schema, limitations and future improvements:
> see [docs/REPORT.md](docs/REPORT.md).

## Features

- **Agent loop** — ReAct-style control loop (LLM ⇄ tool), one tool call per step,
  resumable across process restarts (state lives in SQLite, not memory).
- **Model-agnostic** — `AGENT_LLM_PROVIDER=openai` for any OpenAI-compatible endpoint
  (OpenAI, DeepSeek, vLLM…), or `mock` for a deterministic offline scripted LLM.
- **Tool validation** — Pydantic schemas for tool inputs *and* outputs; invalid
  arguments become observations for the model, never crashes. Validation runs
  **before** the approval gate.
- **Failure handling** — per-call timeouts, transient-error retries with exponential
  backoff + jitter, malformed-LLM recovery (3 strikes → fail), repeated-identical-call
  guard, `max_steps` budget and wall-clock deadline.
- **Human checkpoint** — `create_incident` pauses in `AWAITING_APPROVAL`; an explicit
  approve/reject decision resumes the loop (rejection is fed back to the model).
- **Observability** — one structured JSON trace event per action + a persisted,
  append-only step history per execution.
- **Interfaces** — REST API (FastAPI) and CLI.

## Quickstart

Requires Python 3.10+.

```bash
pip install -e ".[dev]"
```

### CLI (offline demo, no API key needed)

```bash
# Scenario 1: search the knowledge base and answer
python -m ops_harness.cli run "How do I handle payment-api degradation?" --plan kb

# Scenario 2: full approval flow (check service -> consult KB -> propose incident)
python -m ops_harness.cli run "payment-api is degraded, escalate" --plan incident
# -> pauses with: PENDING APPROVAL: id=...
python -m ops_harness.cli pending
python -m ops_harness.cli approve <approval_id> --by alice        # or: reject ...
```

Other commands: `status <execution_id>`, `trace <execution_id>`, `pending`, `serve`.

### REST API

```bash
python -m ops_harness.cli serve          # http://127.0.0.1:8000 (docs at /docs)
```

> WSL users: bind outside the VM and use the VM IP instead of localhost —
> `API_HOST=0.0.0.0 python -m ops_harness.cli serve`, then open
> `http://$(wsl hostname -I):8000/docs`.

Then import `postman/OpsAgentHarness.postman_collection.json` into Postman. The
collection auto-captures `execution_id` / `approval_id`. Typical flow:

| Step | Request |
|---|---|
| 1 | `POST /executions` `{"objective": "Check payment-api and open an incident if degraded"}` → 202 |
| 2 | poll `GET /executions/{id}` → `AWAITING_APPROVAL` |
| 3 | `GET /approvals?status=pending` → approval id + proposed arguments |
| 4 | `POST /approvals/{id}/decision` `{"decision": "approved", "decided_by": "oncall"}` → 202 |
| 5 | poll `GET /executions/{id}` → `COMPLETED` + `final_answer` |
| 6 | `GET /executions/{id}/steps` → full trace |

With a real model:

```bash
export AGENT_LLM_PROVIDER=openai
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
export AGENT_MODEL=gpt-4o-mini
python -m ops_harness.cli serve
```

## Tests

```bash
pytest          # 32 tests: success, tool failure/retry, timeout, approval (approve +
                # reject + double-decision + cancel), malformed LLM, step budget,
                # wall-clock deadline, repeated-call guard, REST API flows
```

## Project structure

```
ops_harness/
├── core/
│   ├── orchestrator.py   # the agent loop + limits + approval gate
│   ├── db.py             # SQLite: executions, steps (append-only), messages, approvals
│   ├── state.py          # execution state machine
│   ├── prompts.py        # system prompt assembly (tool schemas + policies)
│   └── trace.py          # structured JSON trace events
├── llm/
│   ├── base.py           # LLMClient protocol + response parsing
│   ├── mock.py           # deterministic scripted LLM (offline tests/demos)
│   └── openai_client.py  # OpenAI-compatible chat-completions client
├── tools/
│   ├── base.py           # Tool/registry + guarded execution (validate/timeout/retry)
│   ├── kb.py             # search_knowledge_base (keyword retrieval over data/kb_articles.json)
│   ├── service_status.py # get_service_status (fault injection: fail_times / slow_ms)
│   └── incident.py       # create_incident (requires_approval=True)
├── api.py                # FastAPI endpoints
├── cli.py                # CLI commands
├── config.py             # env-var settings
└── mock_plans.py         # offline demo scripts for the mock LLM
data/                     # mock dataset (kb_articles, services, incidents)
postman/                  # Postman collection
tests/                    # pytest suite
```

## Mock dataset

Shipped in `data/` — also packaged at
[ops-agent-harness-mock-dataset.zip](https://drive.google.com/file/d/1inZNz2YlQEmYlNIvPWxT_vrJjZczUhLr/view?usp=sharing)
(Google Drive):

- `kb_articles.json` — 12 internal docs for `search_knowledge_base`
- `services.json` — 8 services; `fail_times` and `slow_ms` fields inject
  deterministic transient failures / slowness (used by tests and demos)
- `incidents.json` — 2 seeded incidents; `create_incident` appends here

## Environment variables

See `.env.example`; defaults: mock provider, `max_steps=10`, `120s` wall clock,
`10s` tool timeout, `2` retries, SQLite at `./ops_harness.db`.
