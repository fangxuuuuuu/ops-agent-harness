"""CLI interface: run, status, trace, approve, reject, pending, serve."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ops_harness", description="Agent Harness for an operations assistant"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run an objective end-to-end (CLI mode)")
    p_run.add_argument("objective")
    p_run.add_argument("--plan", choices=["kb", "incident"], default="kb",
                       help="mock-LLM demo plan to execute (ignored when AGENT_LLM_PROVIDER=openai)")

    p_status = sub.add_parser("status", help="Show execution status")
    p_status.add_argument("execution_id")

    p_trace = sub.add_parser("trace", help="Dump the full step/trace history")
    p_trace.add_argument("execution_id")

    p_appr = sub.add_parser("approve", help="Approve a pending tool call")
    p_appr.add_argument("approval_id")
    p_appr.add_argument("--by", default="cli-operator")
    p_appr.add_argument("--comment", default=None)

    p_rej = sub.add_parser("reject", help="Reject a pending tool call")
    p_rej.add_argument("approval_id")
    p_rej.add_argument("--by", default="cli-operator")
    p_rej.add_argument("--comment", default=None)

    sub.add_parser("pending", help="List pending approvals")
    sub.add_parser("serve", help="Start the REST API server")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        from ops_harness.config import load_settings

        settings = load_settings()
        uvicorn.run("ops_harness.api:app", host=settings.api_host, port=settings.api_port)
        return 0

    return _dispatch(args)


def _dispatch(args) -> int:
    import json

    from ops_harness.api import build_harness
    from ops_harness.config import load_settings
    from ops_harness.core.db import Store
    from ops_harness.mock_plans import demo_kb_script, get_script

    settings = load_settings()

    # For the offline mock provider, rebuild the scripted plan so a resumed
    # execution continues where it paused (instead of replaying from the start).
    script = None
    if settings.provider == "mock":
        probe = Store(settings.db_path)
        if args.cmd == "run":
            script = get_script(args.plan)
        elif args.cmd in ("approve", "reject"):
            approval = probe.get_approval(args.approval_id)
            exec_row = probe.get_execution(approval["execution_id"]) if approval else None
            plan = None
            if exec_row and exec_row.get("meta"):
                plan = (json.loads(exec_row["meta"]) or {}).get("plan")
            if plan and approval:
                consumed = probe.count_steps(approval["execution_id"], "llm_call")
                script = get_script(plan)[consumed:] or demo_kb_script()
        else:
            plan = None

    harness, store = build_harness(settings, mock_script=script)

    if args.cmd == "run":
        meta = json.dumps({"plan": args.plan}) if settings.provider == "mock" else None
        exec_id = harness.start(args.objective, meta=meta)
        return _print_result(store, exec_id)

    if args.cmd == "status":
        row = store.get_execution(args.execution_id)
        if row is None:
            print(f"execution '{args.execution_id}' not found", file=sys.stderr)
            return 1
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "trace":
        row = store.get_execution(args.execution_id)
        if row is None:
            print(f"execution '{args.execution_id}' not found", file=sys.stderr)
            return 1
        print(json.dumps(store.list_steps(args.execution_id), indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "pending":
        print(json.dumps(store.list_approvals(status="pending"), indent=2, ensure_ascii=False))
        return 0

    if args.cmd in ("approve", "reject"):
        decision = "approved" if args.cmd == "approve" else "rejected"
        try:
            status = harness.decide_approval(args.approval_id, decision, args.by, args.comment)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        approval = store.get_approval(args.approval_id)
        print(f"\ndecision '{decision}' applied; execution finished with status: {status}")
        return _print_result(store, approval["execution_id"])

    return 0


def _print_result(store, exec_id: str) -> int:
    row = store.get_execution(exec_id)
    print(f"\n=== execution {exec_id} ===")
    print(f"status: {row['status']}")
    if row["final_answer"]:
        print(f"final answer:\n{row['final_answer']}")
    if row["error"]:
        print(f"error: {row['error']}")
    for appr in store.list_approvals(status="pending", exec_id=exec_id):
        print(f"PENDING APPROVAL: id={appr['id']} tool={appr['tool_name']} args={appr['arguments_json']}")
        print(f"-> run: python -m ops_harness.cli approve {appr['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
