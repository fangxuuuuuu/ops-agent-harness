Ops Agent Harness — Mock Dataset
================================

This dataset backs the three mock tools of the operations assistant agent harness.

Files
-----

1. kb_articles.json — 12 internal documentation articles (runbooks, on-call guides,
   severity policy, API references) used by `search_knowledge_base(query, top_k)`.
   Retrieval is keyword-overlap scoring over title/tags/body.

2. services.json — 8 services with health status, region, owner team and last
   deployment, used by `get_service_status(service_name)`.
   Two special fields support deterministic fault injection (used by tests/demos):
   - "fail_times": N  -> the first N calls raise a transient error, then succeed
   - "slow_ms": N     -> the call sleeps N milliseconds (to exercise timeouts)

3. incidents.json — 2 seeded historical incidents. `create_incident(title,
   description, severity)` appends new incidents here (side-effecting tool, gated
   behind human approval in the harness).

Usage
-----

The harness loads these files from the directory configured via the OPS_DATA_DIR
environment variable (default: the repository's data/ directory). Place the three
JSON files together in one directory and point OPS_DATA_DIR at it:

    export OPS_DATA_DIR=/path/to/this/folder

Repository: <your GitHub repo link — see submission notes>
