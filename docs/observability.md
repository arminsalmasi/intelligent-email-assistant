# Observability

Observability has four parts: structured logs, a telemetry event store, optional tracing backends, and per-thread audit trails. All are built in `email_assistant/observability/` and wired in `AssistantService.create`.

## Structured logs

`setup_logging` installs a JSON formatter on the `email_assistant` logger (stderr). Set `LOG_JSON=false` for a plain-text format and `LOG_LEVEL` for verbosity.

Fields on every line:

| Field | Source |
|---|---|
| `ts` | UTC, millisecond precision |
| `level`, `logger`, `msg` | Standard |
| `run_id` | Set by `InboxOrchestrator.process_inbox` via `log_context` |
| `thread_id` | Set per thread and re-set by `traced_node` |
| `node` | Set by `traced_node` (the graph node name) |
| `event` and extra fields | Passed to `log_event(event, **fields)` |
| `exception` | Formatted traceback when present |

Correlation uses `contextvars` (`log_context`), so it follows the call stack without passing IDs explicitly. Telemetry rows read the same context to fill `run_id` and `thread_id`.

Events that always log in addition to being stored: `error`, `llm_fallback`, `policy_finding`, `guard`. Node completion logs `node.done` with `latency_ms`; node failures log `node.failed` at ERROR.

## Telemetry events

`Telemetry` (`metrics.py`) stores one SQLite row per event: `ts, type, name, value, run_id, thread_id, attrs`.

| Type | Meaning | Notable attributes |
|---|---|---|
| `node` | Node execution; value is latency in ms | `ok`, `interrupted` |
| `llm_call` | One model call; value is latency in ms | model, purpose, input/output tokens, `cache_hit`, `ok` |
| `llm_fallback` | A worker used its deterministic path | worker name |
| `embedding` | Embedding request | provider, count, `cache_hit` |
| `policy_finding` | Policy engine finding | code, severity, action type |
| `guard` | Threat scan result; value is risk score | `quarantined` or `clean` |
| `human_decision` | Reviewer decision | decision, action type |
| `action` | Action outcome | action type, status |
| `memory_reuse` | Episodic reuse | mode (`template` or `example`), tokens saved |
| `error` | Captured exception | error class, truncated detail |

## Metrics endpoints

- `GET /metrics/summary`: per-node count with p50 and p95 latency; LLM calls, cache hits, failures, input and output tokens, p50 and p95 latency of non-cached calls; counts of fallbacks, human decisions, actions, policy findings, guard outcomes and errors; memory reuse (templates, examples, estimated tokens saved).
- `GET /metrics/events?type=&limit=`: raw events, newest first.

The Streamlit dashboard "Metrics" view renders these, including node latency and recent events.

## Tracing

`TRACING_BACKEND` selects the backend (`setup_tracing`):

- `mlflow`: calls `mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)`, `mlflow.set_experiment(MLFLOW_EXPERIMENT)` and `mlflow.langchain.autolog()`. LangGraph node spans and LLM calls (with token usage) are captured. `docker compose` enables this for the API against the bundled MLflow server on :5000. `make mlflow` serves a local `./mlruns` store.
- `langsmith`: sets `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT` if unset; requires `LANGSMITH_API_KEY`. Present in code but not tested end to end (no account); MLflow is the verified backend.
- `none` (default).

If the backend fails to initialise, `tracing.unavailable` is logged and the service continues without tracing.

## Evaluation runs in MLflow

`python -m evals.run --mode live --mlflow` (the `make eval-live` target) logs the metric set of the run to MLflow, so quality can be compared across models and prompt versions alongside traces. Reports are also written to `evals/reports/` and uploaded as CI artifacts.

## Error tracking

- `traced_node` wraps every node. An exception is caught, recorded as a `node` event with `ok=false` and as an `error` event, logged as `node.failed`, and appended to `state.errors`. The node is marked completed, so one failing worker degrades the thread instead of aborting the run. `GraphInterrupt` is passed through and recorded as `interrupted`.
- LLM failures and retries are counted in `llm_call` events (`ok`); when a worker falls back, an `llm_fallback` event is written. `llm_fallbacks` and `llm_cache_misses` are reported by the eval runner; `llm_cache_misses` is gated in replay mode (`llm_fallbacks` is reported only).
- The eval gate treats `thread_errors` above zero as a failure.

## Audit trail

Each thread's state carries an append-only `audit` list (`operator.add` reducer): decisions, policy findings, executions and learning steps. It is returned by `GET /threads/{thread_id}` and shown in the dashboard. Because state is checkpointed, the trail survives interrupt and resume.

## Mapping to OpenTelemetry and Coralogix

The event schema was kept simple so it maps directly onto OpenTelemetry. This mapping is a plan; no OTel exporter is implemented.

| Current | OpenTelemetry |
|---|---|
| JSON log line with `run_id`, `thread_id`, `node` | Log record with trace and span ids plus attributes; Coralogix log ingestion |
| `node` event | Span per graph node (child of a per-thread span, parent of LLM spans) |
| `llm_call` event | Span with GenAI attributes (model, token counts), plus a latency histogram |
| `policy_finding`, `guard`, `action`, `human_decision` | Counters with `code`, `severity`, `action_type` attributes |
| `error` event | Span status ERROR with exception event |
| `/metrics/summary` percentiles | Histograms computed by the backend |

Rollout: add an OTLP exporter (collector sidecar or the OTel SDK) pointing at Coralogix, set `run_id` as a trace-level attribute, and keep MLflow for experiment tracking and eval comparison. Message bodies should be excluded or redacted from exported attributes.

## Sample output

Produced by the API in replay mode (recorded Gemini responses, fixed clock 2026-11-09 09:00 Europe/Stockholm): `POST /inbox/process`, then approve t03 and t07, reject t16, and edit t08 to add the confidential Q4 budget figure (the edit is bounced back to review). The dashboard screenshots in the README come from the same run.

Structured logs (selected lines, one JSON object per line). They show backend selection, a quarantined injection (t27), the ranked inbox, a policy finding on a draft, the scheduler's verifier rejecting a model-proposed slot (t04), the meeting thread t03 through the graph, the four review submissions and the leak finding on the t08 edit:

```json
{"ts": "2026-11-10T08:36:12.217+00:00", "level": "INFO", "logger": "email_assistant", "msg": "llm.backends", "event": "llm.backends", "order": ["gemini:gemini-3.1-flash-lite"]}
{"ts": "2026-11-10T08:36:12.221+00:00", "level": "INFO", "logger": "email_assistant", "msg": "service.ready", "event": "service.ready", "llm": "LLMClient", "embedder": "gemini-embedding-001-768", "tracing": "none"}
{"ts": "2026-11-10T08:36:12.770+00:00", "level": "INFO", "logger": "email_assistant", "msg": "guard.quarantined", "run_id": "7a0ae7af", "thread_id": "t27-system-override-admin-debug", "event": "guard.quarantined", "value": 0.952}
{"ts": "2026-11-10T08:36:12.774+00:00", "level": "INFO", "logger": "email_assistant", "msg": "inbox.ranked", "run_id": "7a0ae7af", "event": "inbox.ranked", "threads": 36}
{"ts": "2026-11-10T08:36:12.786+00:00", "level": "INFO", "logger": "email_assistant", "msg": "policy_finding.confidential_internal", "run_id": "7a0ae7af", "thread_id": "t01-cfo-q4-gpu-budget-board-deck", "node": "policy", "event": "policy_finding.confidential_internal", "value": 1.0, "severity": "soft", "action_type": "send_email"}
{"ts": "2026-11-10T08:36:13.001+00:00", "level": "INFO", "logger": "email_assistant", "msg": "llm_fallback.scheduler.verify", "run_id": "7a0ae7af", "thread_id": "t04-lumen-reschedule-to-thursday", "node": "scheduler", "event": "llm_fallback.scheduler.verify", "value": 1.0, "reason": "no_alternatives"}
{"ts": "2026-11-10T08:36:13.036+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "7a0ae7af", "thread_id": "t03-acmetech-meeting-request-tuesday", "node": "triage", "event": "node.done", "latency_ms": 0.0}
{"ts": "2026-11-10T08:36:13.046+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "7a0ae7af", "thread_id": "t03-acmetech-meeting-request-tuesday", "node": "scheduler", "event": "node.done", "latency_ms": 7.9}
{"ts": "2026-11-10T08:36:13.050+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "7a0ae7af", "thread_id": "t03-acmetech-meeting-request-tuesday", "node": "drafter", "event": "node.done", "latency_ms": 2.9}
{"ts": "2026-11-10T08:36:13.051+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "7a0ae7af", "thread_id": "t03-acmetech-meeting-request-tuesday", "node": "policy", "event": "node.done", "latency_ms": 0.4}
{"ts": "2026-11-10T08:36:13.188+00:00", "level": "INFO", "logger": "email_assistant", "msg": "review.submitted", "run_id": "7a0ae7af", "thread_id": "t03-acmetech-meeting-request-tuesday", "event": "review.submitted", "decisions": ["approve"]}
{"ts": "2026-11-10T08:36:13.189+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "7a0ae7af", "thread_id": "t03-acmetech-meeting-request-tuesday", "node": "review", "event": "node.done", "latency_ms": 0.5}
{"ts": "2026-11-10T08:36:13.197+00:00", "level": "INFO", "logger": "email_assistant", "msg": "review.submitted", "run_id": "7a0ae7af", "thread_id": "t07-royalspin-deposit-webhooks-outage", "event": "review.submitted", "decisions": ["approve"]}
{"ts": "2026-11-10T08:36:13.206+00:00", "level": "INFO", "logger": "email_assistant", "msg": "review.submitted", "run_id": "7a0ae7af", "thread_id": "t16-erik-roadmap-tisdag", "event": "review.submitted", "decisions": ["reject", "reject"]}
{"ts": "2026-11-10T08:36:13.213+00:00", "level": "INFO", "logger": "email_assistant", "msg": "review.submitted", "run_id": "7a0ae7af", "thread_id": "t08-goldcrest-budget-acquisition-rumours", "event": "review.submitted", "decisions": ["edit"]}
{"ts": "2026-11-10T08:36:13.215+00:00", "level": "INFO", "logger": "email_assistant", "msg": "policy_finding.confidential_leak", "run_id": "7a0ae7af", "thread_id": "t08-goldcrest-budget-acquisition-rumours", "node": "revalidate", "event": "policy_finding.confidential_leak", "value": 1.0, "severity": "hard", "action_type": "send_email", "stage": "post_edit"}
```

`GET /metrics/summary` after the same run (excerpt). LLM latency is 0 because every call was served from the replay cache; the live-run latencies are in the README evaluation section.

```json
{
 "llm": {
  "calls": 90,
  "cache_hits": 90,
  "failures": 0,
  "input_tokens": 53823,
  "output_tokens": 9051,
  "latency_ms_p50": 0.0,
  "latency_ms_p95": 0.0
 },
 "fallbacks": {
  "scheduler.verify": 2
 },
 "human_decisions": {
  "edit": 1,
  "reject": 2,
  "approve": 2
 },
 "policy_findings": {
  "confidential_leak": 1,
  "confidential_internal": 1
 },
 "guard": {
  "quarantined": 10,
  "clean": 26
 },
 "errors": {},
 "memory_reuse": {
  "templates": 0,
  "examples": 0,
  "estimated_tokens_saved": 0
 }
}
```
