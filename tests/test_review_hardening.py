"""Regression tests for review-path hardening (each maps to a review finding)."""

import pytest

from email_assistant.graph.orchestrator import ReviewRejected
from email_assistant.schemas.actions import (
    ActionStatus,
    ActionType,
    CalendarCancelPayload,
    ProposedAction,
    SendEmailPayload,
)
from email_assistant.schemas.review import HumanDecision, ReviewResponse
from tests.conftest import thread_id


def _submit(svc, tid, decisions, round_=None):
    return svc.orchestrator.submit_review(tid, ReviewResponse(decisions=decisions, round=round_))


def _email(svc, tid):
    return next(
        a for a in svc.orchestrator.items[tid].review.actions if a.type == ActionType.SEND_EMAIL
    )


def test_bounced_action_cannot_be_approved_without_change(processed):
    tid = thread_id(processed, "t08")
    a = _email(processed, tid)
    _submit(
        processed,
        tid,
        [
            HumanDecision(
                action_id=a.id, decision="edit", edits={"body": a.payload.body + " EUR 350,000"}
            )
        ],
    )
    with pytest.raises(ReviewRejected, match="hard policy findings"):
        _submit(processed, tid, [HumanDecision(action_id=a.id, decision="approve")])
    assert processed.deps.mail.sent_messages() == []


def test_unknown_edit_field_and_bad_types_are_rejected(processed):
    tid = thread_id(processed, "t14")
    a = _email(processed, tid)
    with pytest.raises(ReviewRejected, match="unknown field"):
        _submit(
            processed, tid, [HumanDecision(action_id=a.id, decision="edit", edits={"bdoy": "typo"})]
        )
    with pytest.raises(ReviewRejected):
        _submit(
            processed,
            tid,
            [HumanDecision(action_id=a.id, decision="edit", edits={"to": "notalist"})],
        )
    assert processed.orchestrator.items[tid].status == "pending_review"


def test_naive_datetime_edit_is_rejected(processed):
    tid = thread_id(processed, "t16")
    confirm = next(
        a
        for a in processed.orchestrator.items[tid].review.actions
        if a.type == ActionType.CALENDAR_CONFIRM
    )
    with pytest.raises(ReviewRejected):
        _submit(
            processed,
            tid,
            [
                HumanDecision(
                    action_id=confirm.id, decision="edit", edits={"start": "2026-11-10T03:00:00"}
                )
            ],
        )


def test_stale_round_and_unknown_action_are_rejected(processed):
    tid = thread_id(processed, "t14")
    a = _email(processed, tid)
    with pytest.raises(ReviewRejected, match="stale"):
        _submit(processed, tid, [HumanDecision(action_id=a.id, decision="approve")], round_=7)
    with pytest.raises(ReviewRejected, match="unknown"):
        _submit(processed, tid, [HumanDecision(action_id="nope", decision="approve")])


def test_rejecting_confirm_skips_the_announcing_email(processed):
    tid = thread_id(processed, "t16")
    decisions = [
        HumanDecision(
            action_id=a.id,
            decision="reject" if a.type == ActionType.CALENDAR_CONFIRM else "approve",
        )
        for a in processed.orchestrator.items[tid].review.actions
    ]
    item = _submit(processed, tid, decisions)
    email = next(a for a in item.actions if a.type == ActionType.SEND_EMAIL)
    assert email.status == ActionStatus.SKIPPED
    assert processed.deps.mail.sent_messages() == []


def test_policy_blocks_cancel_of_unlinked_event(processed):
    tid = thread_id(processed, "t03")  # not linked to the board meeting
    a = ProposedAction.create(tid, CalendarCancelPayload(event_id="evt-board-meeting"), "x", "t")
    codes = [f.code for f in processed.deps.policy.evaluate(a, processed.deps.mail.get_thread(tid))]
    assert "event_not_linked" in codes


@pytest.mark.parametrize("amount", ["350.000", "350 000", "EUR 350000", "350 000"])
def test_leak_check_survives_number_reformatting(processed, amount):
    tid = thread_id(processed, "t08")
    thread = processed.deps.mail.get_thread(tid)
    a = ProposedAction.create(
        tid,
        SendEmailPayload(
            to=["helen.cross@goldcrest-casino.example"], subject="Re", body=f"Budget {amount}"
        ),
        "x",
        "t",
    )
    assert any(f.code == "confidential_leak" for f in processed.deps.policy.evaluate(a, thread))


def test_internal_lessons_never_reach_external_drafts(processed):
    fb = processed.deps.feedback
    fb.record_edit("t01", "a1", "customer", "internal", "Hi", "Hi, budget is EUR 350,000", None)
    assert fb.lessons("budget", "customer", audience="external") == []
    assert fb.lessons("budget", "customer", audience="internal")


def test_failed_thread_is_reprocessed(processed):
    tid = thread_id(processed, "t13")  # not archived, so still in the inbox
    processed.orchestrator.items[tid].status = "error"
    processed.orchestrator.process_inbox()
    assert processed.orchestrator.items[tid].status != "error"
    assert processed.orchestrator.items[tid].attempt == 1
