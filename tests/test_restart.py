"""Pending reviews survive a service restart (durable inbox + checkpoints)."""

from email_assistant.schemas.review import HumanDecision, ReviewResponse
from email_assistant.service import AssistantService
from tests.conftest import offline_settings


def _svc(tmp_path) -> AssistantService:
    return AssistantService.create(offline_settings(data_dir=tmp_path))


def _approve_all(svc, req):
    return svc.orchestrator.submit_review(
        req.thread_id,
        ReviewResponse(
            reviewer="t",
            decisions=[HumanDecision(action_id=a.id, decision="approve") for a in req.actions],
        ),
    )


def test_pending_review_survives_restart(tmp_path):
    first = _svc(tmp_path)
    first.orchestrator.process_inbox()
    before = {r.thread_id: [a.id for a in r.actions] for r in first.orchestrator.pending_reviews()}
    assert before

    second = _svc(tmp_path)  # "restart": new process objects, same data dir
    after = {r.thread_id: [a.id for a in r.actions] for r in second.orchestrator.pending_reviews()}
    assert after == before

    second.orchestrator.process_inbox()  # nothing is re-processed or duplicated
    assert {r.thread_id for r in second.orchestrator.pending_reviews()} == set(before)

    req = second.orchestrator.pending_reviews()[0]
    item = _approve_all(second, req)
    assert item.status != "pending_review"
    assert any(a.status.value == "executed" for a in item.actions)


def test_lost_checkpoint_is_reprocessed_not_shown(tmp_path):
    first = _svc(tmp_path)
    first.orchestrator.process_inbox()
    pending = {r.thread_id for r in first.orchestrator.pending_reviews()}
    (tmp_path / "checkpoints.db").unlink()

    second = _svc(tmp_path)
    assert second.orchestrator.pending_reviews() == []
    second.orchestrator.process_inbox()
    assert {r.thread_id for r in second.orchestrator.pending_reviews()} == pending


def test_reset_starts_a_clean_slate(tmp_path):
    first = _svc(tmp_path)
    first.orchestrator.process_inbox()
    epoch = first.orchestrator.run_id
    first.orchestrator.store.reset()
    second = _svc(tmp_path)
    assert second.orchestrator.items == {} and second.orchestrator.run_id != epoch
