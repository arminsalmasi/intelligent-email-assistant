"""End-to-end graph behaviour (offline, deterministic)."""

import pytest

from email_assistant.agents.triage import is_quarantined
from email_assistant.llm.fake import FakeLLM
from email_assistant.schemas.actions import ActionStatus, ActionType
from email_assistant.schemas.review import HumanDecision, ReviewResponse
from email_assistant.schemas.triage import Category, TriageResult, Urgency
from email_assistant.service import AssistantService
from tests.conftest import offline_settings, thread_id


def _decide(svc, tid, decision="approve", edits_for=None, **kw):
    req = svc.orchestrator.items[tid].review
    decisions = []
    for a in req.actions:
        if edits_for and a.type == edits_for[0]:
            decisions.append(
                HumanDecision(action_id=a.id, decision="edit", edits=edits_for[1], **kw)
            )
        else:
            decisions.append(HumanDecision(action_id=a.id, decision=decision))
    return svc.orchestrator.submit_review(tid, ReviewResponse(decisions=decisions))


def _types(item, status=None):
    return [a.type for a in item.actions if status is None or a.status == status]


def test_injections_are_quarantined_without_drafts(processed):
    for item in processed.orchestrator.ranked():
        if item.guard.is_threat:
            assert item.status == "quarantined"
            assert ActionType.SEND_EMAIL not in _types(item)
            assert ActionType.QUARANTINE in _types(item, ActionStatus.EXECUTED)
    assert not any(
        m["to"][0].endswith(("exfil.example", "partner-sync.example"))
        for m in processed.deps.mail.sent_messages()
    )


def test_quarantined_threads_rank_below_real_work(processed):
    ranked = processed.orchestrator.ranked()
    q = [i.rank for i in ranked if is_quarantined(i.guard, i.triage)]
    others = [i.rank for i in ranked if not is_quarantined(i.guard, i.triage)]
    assert q and others and min(q) > max(others)
    # the quarantined block keeps its own score order
    scores = [i.score for i in ranked if is_quarantined(i.guard, i.triage)]
    assert scores == sorted(scores, reverse=True)


def test_nothing_outgoing_before_approval(processed):
    assert processed.deps.mail.sent_messages() == []
    pending = processed.orchestrator.pending_reviews()
    assert pending and all(
        a.status == ActionStatus.PENDING_APPROVAL for r in pending for a in r.actions
    )


def test_meeting_conflict_proposes_alternatives_with_holds(processed):
    tid = thread_id(processed, "t03")
    item = processed.orchestrator.items[tid]
    holds = [a for a in item.actions if a.type == ActionType.CALENDAR_HOLD]
    assert len(holds) == 3 and all(a.status == ActionStatus.EXECUTED for a in holds)
    email = next(a for a in item.actions if a.type == ActionType.SEND_EMAIL)
    assert email.payload.ungrounded_claims == []
    _decide(processed, tid, "approve")
    assert processed.deps.mail.sent_messages()[0]["to"] == ["daniel.reed@acmetech.example"]
    # the reply asks a question -> awaiting-reply reminder
    assert any(
        r.reminder_type == "awaiting_reply" for r in processed.deps.reminders.list(thread_id=tid)
    )


def test_rejection_releases_tentative_hold(processed):
    tid = thread_id(processed, "t16")
    _decide(processed, tid, "reject")
    item = processed.orchestrator.items[tid]
    hold = next(a for a in item.actions if a.type == ActionType.CALENDAR_HOLD)
    assert processed.deps.calendar.get(hold.payload.hold_id).status == "cancelled"
    assert processed.deps.mail.sent_messages() == []


def test_reschedule_and_cancel_touch_linked_events(processed):
    for prefix, expected in (
        ("t04", ActionType.CALENDAR_UPDATE),
        ("t05", ActionType.CALENDAR_CANCEL),
    ):
        tid = thread_id(processed, prefix)
        item = processed.orchestrator.items[tid]
        if item.status == "pending_review" and expected in _types(item):
            _decide(processed, tid, "approve")
            assert expected in _types(processed.orchestrator.items[tid], ActionStatus.EXECUTED)
    cancel_ev = processed.deps.calendar.find_by_thread(thread_id(processed, "t05"))
    assert cancel_ev is None  # cancelled events are no longer "confirmed"


def test_edit_that_leaks_is_sent_back_for_review(processed):
    tid = thread_id(processed, "t08")
    body = processed.orchestrator.items[tid].review.actions[0].payload.body
    item = _decide(
        processed, tid, edits_for=(ActionType.SEND_EMAIL, {"body": body + "\nBudget: EUR 350,000"})
    )
    assert item.status == "pending_review"
    assert item.review.actions[0].findings[0].code == "confidential_leak"
    assert processed.deps.mail.sent_messages() == []


def test_edit_to_new_recipient_is_blocked(processed):
    tid = thread_id(processed, "t14")
    item = _decide(processed, tid, edits_for=(ActionType.SEND_EMAIL, {"to": ["x@evil.example"]}))
    assert item.status == "pending_review"
    assert any(f.code == "recipient_not_in_thread" for a in item.review.actions for f in a.findings)


def test_edit_is_learned_and_changes_next_draft(processed):
    tid = thread_id(processed, "t08")
    body = processed.orchestrator.items[tid].review.actions[0].payload.body
    new_body = body.replace("Best,\nAlex", "Cheers,\nAlex")
    _decide(
        processed,
        tid,
        edits_for=(ActionType.SEND_EMAIL, {"body": new_body}),
        comment="warmer sign-off for customers",
    )
    assert processed.deps.mail.sent_messages()[0]["body"] == new_body
    # a later external draft picks up the learned sign-off
    draft = processed.orchestrator.graph  # noqa: F841 - graph already built
    tid2 = thread_id(processed, "t17")
    thread = processed.deps.mail.get_thread(tid2)
    tri = processed.orchestrator.items[tid2].triage
    acts = processed.orchestrator.triage_agent  # noqa: F841
    from email_assistant.agents.drafter import DrafterAgent

    new_draft = DrafterAgent(processed.deps).run(thread, tri, None, None)[0]
    assert new_draft.payload.body.rstrip().endswith("Cheers,\nAlex")


def test_resume_is_idempotent(processed):
    tid = thread_id(processed, "t03")
    first = processed.orchestrator.items[tid].review.actions[0].id
    _decide(processed, tid, "approve")
    with pytest.raises(KeyError):  # nothing pending any more
        processed.orchestrator.submit_review(
            tid, ReviewResponse(decisions=[HumanDecision(action_id=first, decision="approve")])
        )
    assert len(processed.deps.mail.sent_messages()) == 1


def test_llm_path_is_used_when_available():
    fake = FakeLLM(
        {
            "triage": TriageResult(
                category=Category.CUSTOMER,
                urgency=Urgency.CRITICAL,
                requires_reply=True,
                language="en",
            ),
            "draft": lambda s, u: __import__(
                "email_assistant.agents.drafter", fromlist=["x"]
            ).DraftReply(body="Hi there,\n\nWe are on it.\n\nBest,\nAlex"),
        }
    )
    svc = AssistantService.create(offline_settings(), ephemeral=True, llm=fake)
    tid = "t07-royalspin-deposit-webhooks-outage"
    for t in svc.deps.mail.list_threads():
        if t.id != tid:
            svc.deps.mail.archive(t.id)
    item = svc.orchestrator.process_inbox()[0]
    assert item.triage.source == "llm"
    email = next(a for a in item.actions if a.type == ActionType.SEND_EMAIL)
    assert email.payload.draft_source == "llm" and "on it" in email.payload.body
    assert any(p == "draft" for p, _ in fake.calls)
    # untrusted content reaches the model only inside the data block
    assert all("<email_content" in u for p, u in fake.calls if p in ("triage", "draft"))


def test_worker_failure_degrades_gracefully(processed, monkeypatch):
    svc = AssistantService.create(offline_settings(), ephemeral=True)

    def boom(*a, **k):
        raise RuntimeError("calendar down")

    monkeypatch.setattr(svc.deps.calendar, "find_free_slots", boom)
    items = svc.orchestrator.process_inbox()
    t03 = next(i for i in items if i.thread_id.startswith("t03"))
    assert any("scheduler" in e for e in t03.errors)
    assert ActionType.SEND_EMAIL in _types(t03)  # drafter still ran
