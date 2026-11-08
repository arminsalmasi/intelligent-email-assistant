from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from email_assistant.clock import FixedClock, add_business_days
from email_assistant.schemas.actions import ProposedAction, SendEmailPayload
from email_assistant.schemas.email import EmailAddress
from email_assistant.schemas.review import HumanDecision
from email_assistant.schemas.scheduling import TimeSlot
from email_assistant.tools.ics import build_ics
from email_assistant.tools.reminders import MockReminders

TZ = ZoneInfo("Europe/Stockholm")


def slot(day: int, hh: int, mm: int = 0, minutes: int = 30) -> TimeSlot:
    s = datetime(2026, 11, day, hh, mm, tzinfo=TZ)
    return TimeSlot(start=s, end=s + timedelta(minutes=minutes))


def test_email_address_validation():
    assert EmailAddress(email=" A@B.Example ").email == "a@b.example"
    with pytest.raises(ValidationError):
        EmailAddress(email="not-an-address")


def test_human_decision_validation():
    with pytest.raises(ValidationError):
        HumanDecision(action_id="x", decision="edit")  # edit without edits
    with pytest.raises(ValidationError):
        HumanDecision(action_id="x", decision="approve", override_soft_findings=True)
    with pytest.raises(ValidationError):
        HumanDecision(action_id="x", decision="edit", edits={"kind": "archive"})


def test_action_ids_are_deterministic():
    p = SendEmailPayload(to=["a@b.example"], subject="s", body="one")
    a1 = ProposedAction.create("t1", p, "r", "drafter")
    a2 = ProposedAction.create("t1", p.model_copy(update={"body": "two"}), "r", "drafter")
    assert a1.id == a2.id and a1.idempotency_key == a2.idempotency_key


def test_calendar_slot_rules(svc):
    cal = svc.deps.calendar
    assert cal.check_slot(slot(10, 15)).reasons == ["conflict"]  # board prep
    assert "focus_block" in cal.check_slot(slot(10, 13)).reasons  # Tue focus
    assert "after_friday_cutoff" in cal.check_slot(slot(13, 16)).reasons
    assert "outside_working_hours" in cal.check_slot(slot(11, 7)).reasons
    assert "buffer_too_short" in cal.check_slot(slot(10, 9)).reasons  # standup at 09:30
    # 10:00 Stockholm = 04:00 New York -> outside the attendee's day
    assert "outside_attendee_hours" in cal.check_slot(slot(12, 10), "America/New_York").reasons
    assert cal.check_slot(slot(12, 16), "America/New_York").ok


def test_free_slots_respect_attendee_timezone(svc):
    cal = svc.deps.calendar
    now = svc.deps.clock.now()
    slots = cal.find_free_slots(now, 30, count=3, attendee_tz="America/New_York")
    assert len(slots) == 3
    for s in slots:
        assert cal.check_slot(s, "America/New_York").ok
        assert s.start.astimezone(ZoneInfo("America/New_York")).hour >= 9


def test_hold_confirm_release_idempotent(svc):
    cal = svc.deps.calendar
    s = slot(12, 16)
    cal.place_hold("h1", "Sync", s, ["x@y.example"], "t", "k1")
    cal.place_hold("h1", "Sync", s, ["x@y.example"], "t", "k1")
    assert [e.status for e in cal.list_events() if e.id == "h1"] == ["tentative"]
    cal.release("h1")
    assert cal.get("h1").status == "cancelled"


def test_ics_is_rfc5545_shaped():
    s = slot(12, 16)
    ics = build_ics("u1", "Sync, weekly", s.start, s.end, "a@n.example", ["b@x.example"], s.start)
    assert ics.startswith("BEGIN:VCALENDAR\r\n") and "DTSTART:20261112T150000Z" in ics
    assert "SUMMARY:Sync\\, weekly" in ics


def test_reminders_awaiting_reply_and_digest():
    clock = FixedClock(datetime(2026, 11, 9, 9, tzinfo=TZ))
    r = MockReminders()
    rid = r.create(
        "k", "t1", "Follow up", clock.now() + timedelta(days=2), "awaiting_reply", "", clock.now()
    )
    assert r.create("k", "t1", "dup", clock.now(), "deadline", "", clock.now()) == rid
    assert r.digest(clock.now())["due_soon"][0].id == rid
    assert r.resolve_awaiting_reply("t1", clock.now() + timedelta(hours=1)) == [rid]
    assert r.list(status="completed")[0].completion_reason == "reply_received"


def test_business_days_skip_weekend():
    fri = datetime(2026, 11, 13, 10, tzinfo=TZ)
    assert add_business_days(fri, 1).weekday() == 0
