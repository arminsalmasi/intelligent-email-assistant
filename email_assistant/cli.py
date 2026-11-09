"""Interactive terminal demo: process the inbox, then review actions one by one.

email-assistant demo            # interactive review
email-assistant demo --auto     # approve everything (non-interactive walkthrough)
email-assistant metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from email_assistant.config import get_settings
from email_assistant.schemas.actions import ProposedAction, SendEmailPayload
from email_assistant.schemas.review import HumanDecision, ReviewResponse
from email_assistant.service import AssistantService

console = Console()


def _inbox_table(svc: AssistantService) -> Table:
    t = Table(title=f"Inbox (ranked) - now {svc.deps.clock.now():%a %d %b %H:%M}")
    for col in ("#", "status", "category", "urgency", "from", "subject"):
        t.add_column(col)
    colours = {"pending_review": "yellow", "quarantined": "red", "archived": "dim", "done": "green"}
    for i in svc.orchestrator.ranked():
        c = colours.get(i.status, "white")
        t.add_row(
            str(i.rank),
            f"[{c}]{i.status}[/{c}]",
            i.triage.category.value,
            i.triage.urgency.value,
            i.sender[:32],
            i.subject[:48],
        )
    return t


def _render(a: ProposedAction) -> str:
    p = a.payload
    lines = [f"[bold]{a.type.value}[/bold]  ({a.rationale})"]
    if isinstance(p, SendEmailPayload):
        lines += [
            f"To: {', '.join(p.to)}   Subject: {p.subject}   source: {p.draft_source}",
            "",
            p.body,
        ]
        if p.attachments:
            lines.append(f"[dim]attachments: {[x.filename for x in p.attachments]}[/dim]")
    else:
        lines.append(json.dumps(p.model_dump(mode="json", exclude={"kind"}), indent=1)[:600])
    for f in a.findings:
        lines.append(
            f"[{'red' if f.severity == 'hard' else 'yellow'}]{f.severity.upper()} "
            f"{f.code}: {f.message}[/]"
        )
    return "\n".join(lines)


def _review_interactive(svc: AssistantService) -> None:
    while reqs := svc.orchestrator.pending_reviews():
        req = reqs[0]
        console.rule(f"[bold]Review: {req.subject}[/bold] (round {req.round})")
        if req.summary:
            console.print(f"[dim]{req.summary}[/dim]")
        decisions = []
        for a in req.actions:
            console.print(Panel(_render(a), border_style="cyan"))
            choice = Prompt.ask(
                "[a]pprove / [e]dit / [r]eject / [s]kip thread",
                choices=["a", "e", "r", "s"],
                default="a",
            )
            if choice == "s":
                return
            if choice == "e" and isinstance(a.payload, SendEmailPayload):
                console.print("Enter the new body; finish with a single '.' line")
                body = "\n".join(iter(input, "."))
                comment = Prompt.ask("Why? (stored as feedback)", default="")
                decisions.append(
                    HumanDecision(
                        action_id=a.id,
                        decision="edit",
                        edits={"body": body},
                        comment=comment or None,
                    )
                )
            else:
                decisions.append(
                    HumanDecision(action_id=a.id, decision="reject" if choice == "r" else "approve")
                )
        item = svc.orchestrator.submit_review(req.thread_id, ReviewResponse(decisions=decisions))
        console.print(f"-> {item.status}")


def cmd_demo(
    auto: bool, llm: str | None = None, eml: str | None = None, fresh: bool = False
) -> None:
    settings = get_settings()
    if llm:
        settings = settings.model_copy(update={"llm_provider": llm})
    if eml:
        settings = settings.model_copy(update={"mail_eml_dir": Path(eml)})
    svc = AssistantService.create(settings)
    if fresh:
        svc.orchestrator.store.reset()
        svc = AssistantService.create(settings)
    elif svc.orchestrator.items:
        console.print(
            f"[dim]Resumed {len(svc.orchestrator.items)} threads from the last session "
            f"({len(svc.orchestrator.pending_reviews())} pending reviews); --fresh starts over[/dim]"
        )
    describe = getattr(svc.deps.llm, "describe", None)
    chain = " -> ".join(f"{b['provider']}:{b['name']}" for b in describe()) if describe else "none"
    console.print(
        f"[dim]LLM backends: {chain} | embedder: "
        f"{svc.deps.memory.embedder.name} | tracing: {svc.tracing}[/dim]"
    )
    with console.status("Processing inbox..."):
        svc.orchestrator.process_inbox()
    console.print(_inbox_table(svc))
    if auto:
        for req in svc.orchestrator.pending_reviews():
            svc.orchestrator.submit_review(
                req.thread_id,
                ReviewResponse(
                    decisions=[
                        HumanDecision(action_id=a.id, decision="approve") for a in req.actions
                    ]
                ),
            )
    else:
        _review_interactive(svc)
    console.print(_inbox_table(svc))
    console.print(
        Panel(
            json.dumps(svc.deps.telemetry.summary(), indent=1, default=str)[:3000], title="Metrics"
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="email-assistant")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--auto", action="store_true", help="approve every proposed action")
    d.add_argument(
        "--llm",
        choices=["auto", "gemini", "bedrock", "local", "none"],
        help="LLM backend (default: LLM_PROVIDER, i.e. auto = gemini -> bedrock -> local)",
    )
    d.add_argument("--eml", metavar="DIR", help="read the inbox from a directory of .eml files")
    d.add_argument("--fresh", action="store_true", help="forget the previous session's inbox")
    args = ap.parse_args(argv)
    if args.cmd == "demo":
        cmd_demo(args.auto, args.llm, args.eml, args.fresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
