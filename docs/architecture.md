# Architecture

## Service composition

`AssistantService.create` (`email_assistant/service.py`) is the single composition root used by the API, CLI, notebook, eval runner and tests.

```mermaid
flowchart LR
    S[Settings / .env] --> SVC[AssistantService.create]
    SVC --> CLK[Clock<br/>fixed or wall]
    SVC --> TEL[Telemetry<br/>SQLite events]
    SVC --> EMB[Embedder<br/>Gemini or hashing<br/>+ JSONL cache]
    SVC --> LLM[LLM client<br/>Gemini / Bedrock / Null<br/>+ JSONL cache, retry, counters]
    EMB --> MEM[MemoryStore<br/>semantic, episodic, procedural]
    SVC --> TOOLS[MailProvider<br/>MockCalendar<br/>MockReminders]
    SVC --> SAFE[ThreatDetector<br/>PolicyEngine]
    CLK & TEL & LLM & MEM & TOOLS & SAFE --> DEPS[Deps]
    DEPS --> TG[ThreadGraph.build<br/>+ SqliteSaver checkpointer]
    TG --> ORCH[InboxOrchestrator]
    ORCH --> API[FastAPI]
    ORCH --> CLI[CLI]
    API --> DASH[Streamlit dashboard]
    TEL --> API
```

`Deps` is a plain container passed to the graph and workers. Nothing reads global state, so tests inject fakes (`llm`, `embedder`, `clock`, `mail`) through `create(...)`. `ephemeral=True` moves every database to `:memory:`; the eval runner uses it.

## Inbox orchestrator

`InboxOrchestrator.process_inbox`:

1. List threads not yet processed.
2. For inbound replies, resolve awaiting-reply reminders for that thread.
3. For each thread: `ThreatDetector.assess`, then the triage agent (structured classification, optionally merged with LLM suspicion).
4. Rank by a priority score (`priority_score`) and assign ranks.
5. Run the per-thread graph in rank order with a `thread_id`-keyed checkpoint.
6. Threads that stop at `interrupt()` appear in the review queue (`pending_reviews`); `submit_review` resumes them.

## Per-thread graph

Generated from the compiled graph (`scripts/export_graph.py`).

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	guard(guard)
	triage(triage)
	supervisor(supervisor)
	labeler(labeler)
	summarizer(summarizer)
	scheduler(scheduler)
	followup(followup)
	drafter(drafter)
	archiver(archiver)
	quarantine(quarantine)
	policy(policy)
	execute_auto(execute_auto)
	review(review)
	revalidate(revalidate)
	execute(execute)
	learn(learn)
	__end__([<p>__end__</p>]):::last
	__start__ --> guard;
	archiver --> supervisor;
	drafter --> supervisor;
	execute --> learn;
	execute_auto -.-> execute;
	execute_auto -.-> review;
	followup --> supervisor;
	guard --> triage;
	labeler --> supervisor;
	policy --> execute_auto;
	quarantine --> supervisor;
	revalidate -.-> execute;
	revalidate -.-> review;
	review --> revalidate;
	scheduler --> supervisor;
	summarizer --> supervisor;
	supervisor -.-> archiver;
	supervisor -.-> drafter;
	supervisor -.-> followup;
	supervisor -.-> labeler;
	supervisor -.-> policy;
	supervisor -.-> quarantine;
	supervisor -.-> scheduler;
	supervisor -.-> summarizer;
	triage --> supervisor;
	learn --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

Routing is deterministic: the supervisor reads the triage result and builds a plan (`plan`, `completed` in state), then dispatches the next worker until the plan is empty, then goes to `policy`. Every node is wrapped by `traced_node`. For workers, an exception is appended to `state.errors`, the node is marked completed and the thread continues in degraded form. The safety-critical nodes (`policy`, `execute_auto`, `review`, `revalidate`, `execute`) fail closed: the exception aborts the run, so nothing unvalidated can reach execution.

`review` calls `interrupt()` with a `ReviewRequest`; the checkpoint holds the state until `Command(resume=ReviewResponse)` arrives. Before resuming, the orchestrator validates the response: the review round must match, every action id must be pending, edits must pass the strict payload schema (unknown fields and naive datetimes are rejected), and an action with hard findings cannot be approved, only edited or rejected. `revalidate` is the single gate before `execute`: it re-runs policy on every approved or edited approval-tier action; a hard finding routes back to `review` (after 3 rounds the action is blocked). A meeting confirmation and the email announcing it are all-or-nothing.

## Sequence: meeting request with conflict

```mermaid
sequenceDiagram
    autonumber
    participant Inbox as InboxOrchestrator
    participant G as Thread graph
    participant Sch as scheduler
    participant Cal as MockCalendar
    participant Pol as PolicyEngine
    participant H as Human (API/dashboard)
    participant Mem as Memory

    Inbox->>G: run(thread, guard, triage)
    G->>Sch: supervisor dispatches scheduler
    Sch->>Sch: extract requested time, normalise timezone
    Sch->>Cal: lookup free/busy (tool loop, bounded)
    Cal-->>Sch: conflict
    Sch-->>G: 3 alternatives + tentative calendar_hold actions
    G->>G: drafter quotes alternatives (send_email, depends on hold)
    G->>Pol: policy (hard/soft findings)
    G->>G: execute_auto (tentative holds, labels, reminders)
    G-->>H: interrupt(ReviewRequest: send_email)
    H->>G: resume(edit body)
    G->>Pol: revalidate edited action
    Pol-->>G: no hard finding
    G->>G: execute (send_email, dependency-aware)
    G->>Mem: learn (approved reply, human edit, sign-off)
```

If the reviewer rejects the email instead, `execute` runs the compensation path and releases the tentative holds that depended on it. If the edit introduces a hard finding (for example a recipient outside the thread), `revalidate` returns the thread to `review`.

## State schema

`ThreadState` (`schemas/state.py`) is a `TypedDict(total=False)`; node updates are partial dicts.

| Field | Type | Reducer | Purpose |
|---|---|---|---|
| `thread_id` | str | last write | Checkpoint key |
| `thread` | `EmailThread` | last write | Input thread |
| `guard` | `GuardResult` | last write | Risk score, signals, threat type |
| `triage` | `TriageResult` | last write | Category, urgency, reply need, language, labels |
| `summary` | `ThreadSummary` | last write | Thread summary |
| `plan` | list[str] | last write | Workers still to run |
| `completed` | list[str] | `operator.add` | Workers finished |
| `scheduling` | `SchedulingDecision` | last write | Requested time, verdict, alternatives |
| `actions` | dict[id, `ProposedAction`] | `merge_actions` (merge by id) | All proposed actions and statuses |
| `review_round` | int | last write | Review/revalidate counter |
| `audit` | list[dict] | `operator.add` | Append-only trail |
| `errors` | list[str] | `operator.add` | Captured node errors |

Action payloads are a discriminated union over `ActionType`: `send_email`, `calendar_hold`, `calendar_confirm`, `calendar_update`, `calendar_cancel`, `create_reminder`, `apply_labels`, `archive`, `quarantine`. Each action has a `RiskTier` (`auto` or `approval`), a deterministic id and an idempotency key.

## Action lifecycle

```mermaid
stateDiagram-v2
    [*] --> proposed
    proposed --> blocked: hard policy finding
    proposed --> pending_approval: approval tier
    proposed --> approved: auto tier
    pending_approval --> approved: human approves
    pending_approval --> edited: human edits
    pending_approval --> rejected: human rejects
    edited --> pending_approval: revalidate finds hard finding
    edited --> executed: revalidate clean
    approved --> pending_approval: revalidate finds hard finding
    approved --> executed
    approved --> failed
    approved --> skipped: dependency rejected / compensated
    blocked --> [*]
    rejected --> [*]
    executed --> [*]
    failed --> [*]
    skipped --> [*]
```

Statuses are defined in `ActionStatus`: `proposed`, `blocked`, `pending_approval`, `approved`, `edited`, `rejected`, `executed`, `failed`, `skipped`.

## Safety placement

| Stage | Control |
|---|---|
| Before triage | `ThreatDetector.assess` on raw thread; quarantine above threshold |
| Prompt construction | `untrusted.wrap` (normalise, strip role tags, delimit); least-privilege context by recipient audience |
| After worker output | Pydantic validation; typed payloads |
| `policy` / `revalidate` | Allow-listed action types, recipients within thread participants, protected-term leak check, attendee check, rate limit, grounding check |
| `execute` | Idempotency keys; dependency order; compensation |
