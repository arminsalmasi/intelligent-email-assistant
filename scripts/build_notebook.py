"""Build (and execute) notebooks/demo_walkthrough.ipynb.

python scripts/build_notebook.py            # replay mode (recorded Gemini responses)
NOTEBOOK_MODE=offline python scripts/build_notebook.py
"""

from __future__ import annotations

import os

import nbformat
from nbclient import NotebookClient

from email_assistant.config import ROOT_DIR

MODE = os.getenv("NOTEBOOK_MODE", "replay")

cells: list[tuple[str, str]] = [
    (
        "md",
        """# Email assistant - end-to-end walkthrough

This notebook drives the same `AssistantService` used by the API, CLI and eval runner:

1. process a 36-thread inbox (guard -> triage -> rank -> per-thread LangGraph)
2. inspect a meeting thread: summary, scheduling decision, tentative holds, grounded draft
3. human review: approve, **edit** (learned into memory) and an edit that **leaks** data (bounced back to review)
4. prompt-injection quarantine
5. telemetry and the audit trail

`MODE = "replay"` uses recorded Gemini responses (no API key needed);
`"live"` calls the configured LLM backend(s) (`LLM_PROVIDER`, default auto); `"offline"` uses only the rule-based fallbacks.""",
    ),
    (
        "code",
        f"""import pandas as pd
from email_assistant.service import AssistantService
from email_assistant.schemas.review import HumanDecision, ReviewResponse
from evals.run import settings_for

MODE = "{MODE}"
svc = AssistantService.create(settings_for(MODE), ephemeral=True)
orc = svc.orchestrator
print(type(svc.deps.llm).__name__, svc.deps.memory.embedder.name, svc.deps.clock.now())""",
    ),
    ("md", "## 1. Process the inbox (ranked)"),
    (
        "code",
        """items = orc.process_inbox()
pd.DataFrame([{"rank": i.rank, "status": i.status, "category": i.triage.category.value,
               "urgency": i.triage.urgency.value, "by": i.triage.source,
               "from": i.sender[:30], "subject": i.subject[:50]} for i in items])""",
    ),
    ("md", "## 2. A meeting request that conflicts with the calendar"),
    (
        "code",
        """tid = next(t for t in orc.items if t.startswith("t03"))
state = orc.state(tid)
dec = state["scheduling"]
print("decision:", dec.outcome, "| source:", dec.source, "| warnings:", dec.warnings)
for s in dec.alternatives:
    print("  alternative:", s.label("Europe/Stockholm"), "/", s.label("America/New_York"))
for a in state["actions"].values():
    print(f"{a.type.value:18} {a.status.value:17} {a.rationale}")""",
    ),
    (
        "code",
        """draft = next(a for a in state["actions"].values() if a.type.value == "send_email")
print(draft.payload.body)
print("\\nungrounded claims:", draft.payload.ungrounded_claims, "| source:", draft.payload.draft_source)""",
    ),
    (
        "md",
        "## 3. Human review\nApprove the meeting reply; then edit a customer reply (the edit is learned).",
    ),
    (
        "code",
        """req = orc.items[tid].review
orc.submit_review(tid, ReviewResponse(decisions=[
    HumanDecision(action_id=a.id, decision="approve") for a in req.actions]))
print(orc.items[tid].status, "| sent:", [m["to"] for m in svc.deps.mail.sent_messages()])
print("awaiting-reply reminders:",
      [r.title for r in svc.deps.reminders.list(thread_id=tid)])""",
    ),
    (
        "code",
        """t08 = next(t for t in orc.items if t.startswith("t08"))
email = orc.items[t08].review.actions[0]
print(email.payload.body)
leaky = email.payload.body + "\\nPS: our Q4 GPU budget is EUR 350,000."
item = orc.submit_review(t08, ReviewResponse(decisions=[
    HumanDecision(action_id=email.id, decision="edit", edits={"body": leaky})]))
print("\\nafter leaky edit ->", item.status, [f.code for f in item.review.actions[0].findings])""",
    ),
    (
        "code",
        """better = email.payload.body.replace("Best,\\nAlex", "Cheers,\\nAlex")
item = orc.submit_review(t08, ReviewResponse(decisions=[
    HumanDecision(action_id=email.id, decision="edit", edits={"body": better},
                  comment="warmer sign-off for customers")]))
print(item.status)
pd.DataFrame([{"kind": r.kind.value, "key": r.key, "content": r.content[:80], "source": r.source.value}
              for r in svc.deps.memory.list() if r.source.value == "hitl_feedback"])""",
    ),
    ("md", "## 4. Prompt injection: quarantined, nothing drafted"),
    (
        "code",
        """t27 = next(t for t in orc.items if t.startswith("t27"))
i = orc.items[t27]
print(i.status, i.guard.risk_score, i.guard.signals)
print([a.type.value for a in i.actions])""",
    ),
    ("md", "## 5. Telemetry and audit trail"),
    (
        "code",
        """import json
print(json.dumps(svc.deps.telemetry.summary(), indent=1)[:2500])""",
    ),
    ("code", """pd.DataFrame(orc.audit(t08))"""),
]


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(s) if k == "md" else nbformat.v4.new_code_cell(s)
        for k, s in cells
    ]
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    NotebookClient(
        nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT_DIR)}}
    ).execute()
    out = ROOT_DIR / "notebooks" / "demo_walkthrough.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbformat.write(nb, out)
    print(out)


if __name__ == "__main__":
    main()
