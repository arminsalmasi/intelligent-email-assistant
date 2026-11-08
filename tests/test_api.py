import pytest
from fastapi.testclient import TestClient

from email_assistant.api import main as api
from email_assistant.service import AssistantService
from tests.conftest import offline_settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        api, "_build", lambda: AssistantService.create(offline_settings(), ephemeral=True)
    )
    with TestClient(api.app) as c:
        yield c


def test_full_review_cycle_over_http(client):
    assert client.get("/health").json() == {"status": "ok"}
    res = client.post("/inbox/process").json()
    assert res["processed"] == 36 and res["quarantined"] >= 10
    reviews = client.get("/reviews").json()
    req = next(r for r in reviews if r["thread_id"].startswith("t03"))
    decisions = [{"action_id": a["id"], "decision": "approve"} for a in req["actions"]]
    item = client.post(f"/reviews/{req['thread_id']}", json={"decisions": decisions}).json()
    assert item["status"] == "done"
    assert client.get("/sent").json()[0]["to"] == ["daniel.reed@acmetech.example"]
    assert (
        client.post(f"/reviews/{req['thread_id']}", json={"decisions": decisions}).status_code
        == 409
    )
    summary = client.get("/metrics/summary").json()
    assert summary["human_decisions"]["approve"] >= 1
    assert client.get("/threads/" + req["thread_id"]).json()["audit"]


def test_invalid_decision_is_rejected(client):
    client.post("/inbox/process")
    req = client.get("/reviews").json()[0]
    bad = {"decisions": [{"action_id": req["actions"][0]["id"], "decision": "edit"}]}
    assert client.post(f"/reviews/{req['thread_id']}", json=bad).status_code == 422
