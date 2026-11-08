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
- `langsmith`: sets `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT` if unset; requires `LANGSMITH_API_KEY`.
- `none` (default).

If the backend fails to initialise, `tracing.unavailable` is logged and the service continues without tracing.

## Evaluation runs in MLflow

`python -m evals.run --mode live --mlflow` (the `make eval-live` target) logs the metric set of the run to MLflow, so quality can be compared across models and prompt versions alongside traces. Reports are also written to `evals/reports/` and uploaded as CI artifacts.

## Error tracking

- `traced_node` wraps every node. An exception is caught, recorded as a `node` event with `ok=false` and as an `error` event, logged as `node.failed`, and appended to `state.errors`. The node is marked completed, so one failing worker degrades the thread instead of aborting the run. `GraphInterrupt` is passed through and recorded as `interrupted`.
- LLM failures and retries are counted in `llm_call` events (`ok`); when a worker falls back, an `llm_fallback` event is written. `llm_fallbacks` and `llm_cache_misses` are also reported by the eval runner and gated in replay mode.
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

## Sample output

Structured logs (replay-mode API run; one JSON object per line):

```json
{"ts": "2026-11-08T16:13:26.713+00:00", "level": "INFO", "logger": "email_assistant", "msg": "guard.quarantined", "run_id": "eecae10d", "thread_id": "t27-system-override-admin-debug", "event": "guard.quarantined", "value": 0.952}
{"ts": "2026-11-08T16:13:26.716+00:00", "level": "INFO", "logger": "email_assistant", "msg": "inbox.ranked", "run_id": "eecae10d", "event": "inbox.ranked", "threads": 36}
{"ts": "2026-11-08T16:13:26.786+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "eecae10d", "thread_id": "t27-system-override-admin-debug", "node": "guard", "event": "node.done", "latency_ms": 0.0}
{"ts": "2026-11-08T16:13:26.787+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "eecae10d", "thread_id": "t27-system-override-admin-debug", "node": "triage", "event": "node.done", "latency_ms": 0.0}
{"ts": "2026-11-08T16:13:26.787+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "eecae10d", "thread_id": "t27-system-override-admin-debug", "node": "supervisor", "event": "node.done", "latency_ms": 0.0}
{"ts": "2026-11-08T16:13:26.787+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "eecae10d", "thread_id": "t27-system-override-admin-debug", "node": "labeler", "event": "node.done", "latency_ms": 0.0}
{"ts": "2026-11-08T16:13:26.788+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "eecae10d", "thread_id": "t27-system-override-admin-debug", "node": "supervisor", "event": "node.done", "latency_ms": 0.0}
{"ts": "2026-11-08T16:13:26.788+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "eecae10d", "thread_id": "t27-system-override-admin-debug", "node": "quarantine", "event": "node.done", "latency_ms": 0.0}
{"ts": "2026-11-08T16:13:26.789+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "eecae10d", "thread_id": "t27-system-override-admin-debug", "node": "supervisor", "event": "node.done", "latency_ms": 0.0}
{"ts": "2026-11-08T16:13:26.789+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "eecae10d", "thread_id": "t27-system-override-admin-debug", "node": "policy", "event": "node.done", "latency_ms": 0.0}
{"ts": "2026-11-08T16:13:26.790+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "eecae10d", "thread_id": "t27-system-override-admin-debug", "node": "execute_auto", "event": "node.done", "latency_ms": 0.6}
{"ts": "2026-11-08T16:13:26.790+00:00", "level": "INFO", "logger": "email_assistant", "msg": "node.done", "run_id": "eecae10d", "thread_id": "t27-system-override-admin-debug", "node": "execute", "event": "node.done", "latency_ms": 0.0}
```

`GET /metrics/summary` after the same run (excerpt):

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
 "memory_reuse": {
  "templates": 0,
  "examples": 0,
  "estimated_tokens_saved": 0
 }
}
```

