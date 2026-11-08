"""Golden-set evaluation.

Modes
* offline : no LLM, hashing embeddings      -> rule-based baseline (CI, no key)
* replay  : recorded Gemini responses/vectors -> LLM pipeline in CI, no key
* live    : real Gemini, records caches        -> refresh recordings (needs key)

Usage: python -m evals.run --mode replay --check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from typing import Any

from email_assistant.config import ROOT_DIR, Settings
from email_assistant.schemas.actions import ActionStatus, ActionType, SendEmailPayload
from email_assistant.schemas.review import HumanDecision, ReviewResponse
from email_assistant.service import AssistantService

GOLDEN = ROOT_DIR / "fixtures" / "golden.json"
THRESHOLDS = ROOT_DIR / "evals" / "thresholds.json"
REPORTS = ROOT_DIR / "evals" / "reports"

ACTION_MAP = {
    "send_email": {ActionType.SEND_EMAIL},
    "calendar_hold": {ActionType.CALENDAR_HOLD, ActionType.CALENDAR_CONFIRM},
    "calendar_update": {ActionType.CALENDAR_UPDATE},
    "calendar_cancel": {ActionType.CALENDAR_CANCEL},
    "create_reminder": {ActionType.CREATE_REMINDER},
    "archive": {ActionType.ARCHIVE},
    "quarantine": {ActionType.QUARANTINE},
}


def settings_for(mode: str) -> Settings:
    base: dict[str, Any] = {"log_level": "WARNING"}
    if mode == "offline":
        base |= {"llm_provider": "none", "embedding_provider": "hashing"}
    elif mode == "replay":
        base |= {
            "llm_cache_mode": "replay",
            "embedding_cache_mode": "replay",
            "gemini_api_key": None,
            "llm_max_retries": 0,
        }
    elif mode == "live":
        base |= {"llm_cache_mode": "record", "embedding_cache_mode": "record"}
    return Settings(**base)


def auto_review(svc: AssistantService, max_rounds: int = 4) -> None:
    """Simulated reviewer: approves every pending action (exercises the executor)."""
    for _ in range(max_rounds):
        pending = svc.orchestrator.pending_reviews()
        if not pending:
            return
        for req in pending:
            svc.orchestrator.submit_review(
                req.thread_id,
                ReviewResponse(
                    reviewer="eval-bot",
                    decisions=[
                        HumanDecision(action_id=a.id, decision="approve") for a in req.actions
                    ],
                ),
            )


def evaluate(mode: str) -> dict[str, Any]:
    golden: dict[str, dict] = json.loads(GOLDEN.read_text(encoding="utf-8"))
    svc = AssistantService.create(settings_for(mode), ephemeral=True)
    started = time.perf_counter()
    items = svc.orchestrator.process_inbox()
    auto_review(svc)
    elapsed = time.perf_counter() - started
    by_id = {i.thread_id: svc.orchestrator.items[i.thread_id] for i in items}
    policy = svc.deps.policy

    c: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    rows = []
    top_pred = {i.thread_id for i in svc.orchestrator.ranked()[:8]}
    top_gold = {k for k, v in golden.items() if v["rank_bucket"] == "top"}
    for tid, g in golden.items():
        item = by_id.get(tid)
        if item is None:
            failures.append({"thread": tid, "issue": "not processed"})
            continue
        t = item.triage
        problems: list[str] = []
        cats = {g["category"], *g.get("acceptable_categories", [])}
        urgs = {g["urgency"], *g.get("acceptable_urgencies", [])}
        ok_cat, ok_urg = t.category.value in cats, t.urgency.value in urgs
        ok_reply = t.requires_reply == g["requires_reply"]
        ok_lang = t.language[:2] == g["language"]
        c["n"] += 1
        c["category_ok"] += ok_cat
        c["urgency_ok"] += ok_urg
        c["reply_ok"] += ok_reply
        c["language_ok"] += ok_lang
        if not ok_cat:
            problems.append(f"category {t.category.value} != {g['category']}")
        if not ok_urg:
            problems.append(f"urgency {t.urgency.value} != {g['urgency']}")
        if not ok_reply:
            problems.append(f"requires_reply {t.requires_reply}")
        # safety
        threat_gold = g["category"] == "security_threat"
        if threat_gold:
            c["threats"] += 1
            c["threats_caught"] += item.guard.is_threat or t.category.value == "security_threat"
        else:
            c["benign"] += 1
            if item.guard.is_threat:
                c["false_positives"] += 1
                problems.append("false positive quarantine")
        # actions
        types = {a.type for a in item.actions}
        for exp in g["expected_actions"]:
            c["expected_actions"] += 1
            if ACTION_MAP[exp] & types:
                c["expected_hit"] += 1
            else:
                problems.append(f"missing action {exp}")
        for forb in g["forbidden_actions"]:
            if ACTION_MAP[forb] & types:
                c["forbidden_violations"] += 1
                problems.append(f"forbidden action {forb}")
        # drafts
        sends = [a for a in item.actions if a.type == ActionType.SEND_EMAIL]
        for a in sends:
            p: SendEmailPayload = a.payload  # type: ignore[assignment]
            body = p.body.lower()
            c["drafts"] += 1
            c["drafts_llm"] += p.draft_source == "llm"
            c["ungrounded_drafts"] += bool(p.ungrounded_claims)
            if p.ungrounded_claims:
                problems.append(f"ungrounded {p.ungrounded_claims}")
            for s in g.get("draft_must_include", []):
                c["must_include"] += 1
                if s.lower() in body:
                    c["must_include_hit"] += 1
                else:
                    problems.append(f"draft missing '{s}'")
            for s in g.get("draft_must_not_include", []):
                if s.lower() in body:
                    c["must_not_violations"] += 1
                    problems.append(f"draft contains '{s}'")
            leak = [
                f
                for f in policy.evaluate(
                    a, svc.deps.mail.get_thread(tid) or svc.deps.mail.list_threads(True)[0]
                )
                if f.code == "confidential_leak"
            ]
            if a.status == ActionStatus.EXECUTED and leak:
                c["leaks_sent"] += 1
                problems.append("CONFIDENTIAL LEAK SENT")
        # meetings
        if g.get("meeting"):
            c["meetings"] += 1
            st = svc.orchestrator.state(tid)
            dec = st.get("scheduling")
            if dec and dec.kind == g["meeting"]["kind"]:
                c["meeting_kind_ok"] += 1
            else:
                problems.append(f"meeting kind {dec.kind if dec else None}")
            if dec and dec.outcome in ("accept_requested", "propose_alternatives", "cancel"):
                c["meeting_resolved"] += 1
        # deadlines
        if g.get("deadline"):
            c["deadlines"] += 1
            gold_day = g["deadline"]["due"][:10]
            rems = [a.payload for a in item.actions if a.type == ActionType.CREATE_REMINDER]
            days = {(r.deadline or r.due).date().isoformat() for r in rems}  # type: ignore[union-attr]
            if gold_day in days:
                c["deadline_ok"] += 1
            else:
                problems.append(f"deadline {sorted(days)} != {gold_day}")
        if item.errors:
            c["thread_errors"] += 1
            problems.append(f"errors {item.errors[:1]}")
        rows.append(
            {
                "thread": tid,
                "category": t.category.value,
                "urgency": t.urgency.value,
                "source": t.source,
                "status": item.status,
                "problems": problems,
            }
        )
        if problems:
            failures.append({"thread": tid, "problems": problems})

    tel = svc.deps.telemetry.summary()

    def r(a: str, b: str) -> float:
        return round(c[a] / c[b], 3) if c[b] else 1.0

    metrics = {
        "category_accuracy": r("category_ok", "n"),
        "urgency_accuracy": r("urgency_ok", "n"),
        "requires_reply_accuracy": r("reply_ok", "n"),
        "language_accuracy": r("language_ok", "n"),
        "threat_recall": r("threats_caught", "threats"),
        "false_positive_rate": round(c["false_positives"] / c["benign"], 3) if c["benign"] else 0.0,
        "expected_action_recall": r("expected_hit", "expected_actions"),
        "forbidden_action_violations": c["forbidden_violations"],
        "draft_must_include_rate": r("must_include_hit", "must_include"),
        "draft_must_not_violations": c["must_not_violations"],
        "confidential_leaks_sent": c["leaks_sent"],
        "ungrounded_draft_rate": round(c["ungrounded_drafts"] / c["drafts"], 3)
        if c["drafts"]
        else 0.0,
        "meeting_kind_accuracy": r("meeting_kind_ok", "meetings"),
        "meeting_resolution_rate": r("meeting_resolved", "meetings"),
        "deadline_accuracy": r("deadline_ok", "deadlines"),
        "ranking_precision_at_8": round(len(top_pred & top_gold) / 8, 3),
        "thread_errors": c["thread_errors"],
        "llm_drafts_share": round(c["drafts_llm"] / c["drafts"], 3) if c["drafts"] else 0.0,
        "llm_fallbacks": sum(tel["fallbacks"].values()),
        "llm_cache_misses": getattr(svc.deps.llm, "cache_misses", 0),
        "llm_input_tokens": tel["llm"]["input_tokens"],
        "llm_output_tokens": tel["llm"]["output_tokens"],
        "wall_clock_s": round(elapsed, 1),
    }
    return {
        "mode": mode,
        "threads": c["n"],
        "metrics": metrics,
        "failures": failures,
        "rows": rows,
        "telemetry": tel,
    }


def check(report: dict[str, Any]) -> list[str]:
    th = json.loads(THRESHOLDS.read_text())[report["mode"]]
    bad = []
    for k, v in th.get("min", {}).items():
        if report["metrics"][k] < v:
            bad.append(f"{k}={report['metrics'][k]} < {v}")
    for k, v in th.get("max", {}).items():
        if report["metrics"][k] > v:
            bad.append(f"{k}={report['metrics'][k]} > {v}")
    return bad


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation report ({report['mode']}, {report['threads']} threads)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in report["metrics"].items()]
    if report["failures"]:
        lines += ["", "## Per-thread issues", ""]
        lines += [
            f"- **{f['thread']}**: {'; '.join(f.get('problems', [f.get('issue', '')]))}"
            for f in report["failures"]
        ]
    return "\n".join(lines) + "\n"


def log_mlflow(report: dict[str, Any]) -> None:
    try:
        import mlflow
    except ImportError:
        return
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"{settings.mlflow_experiment}-evals")
    with mlflow.start_run(run_name=f"golden-{report['mode']}"):
        mlflow.log_param("mode", report["mode"])
        mlflow.log_metrics({k: float(v) for k, v in report["metrics"].items()})
        mlflow.log_dict(report, "report.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offline", "replay", "live"], default="offline")
    ap.add_argument("--check", action="store_true", help="fail on threshold regressions")
    ap.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    args = ap.parse_args(argv)
    report = evaluate(args.mode)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{args.mode}.json").write_text(json.dumps(report, indent=2, default=str))
    (REPORTS / f"{args.mode}.md").write_text(to_markdown(report))
    print(to_markdown(report))
    if args.mlflow:
        log_mlflow(report)
    if args.check:
        bad = check(report)
        if bad:
            print("THRESHOLD FAILURES:\n- " + "\n- ".join(bad))
            return 1
        print("All thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
