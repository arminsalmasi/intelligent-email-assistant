"""Faithfulness judge: scoring, least-privilege context, untrusted wrapping."""

from email_assistant.schemas.actions import SendEmailPayload
from evals.judge import FaithfulnessJudge, FaithfulnessVerdict, JudgedClaim


class _LLM:
    def __init__(self, verdict):
        self.verdict, self.seen = verdict, None

    def structured(self, schema, system, user, purpose):
        self.seen = (system, user)
        return self.verdict


def _judge(processed, verdict):
    o = processed.deps.profile.owner
    llm = _LLM(verdict)
    return FaithfulnessJudge(llm, processed.deps.memory, o.name, o.title, "now"), llm


def _draft(processed):
    for item in processed.orchestrator.items.values():
        for a in item.actions:
            if isinstance(a.payload, SendEmailPayload) and a.payload.context_facts:
                return processed.deps.mail.get_thread(item.thread_id), a.payload
    raise AssertionError("no draft with retrieved facts")


def test_score_counts_supported_claims(processed):
    thread, payload = _draft(processed)
    verdict = FaithfulnessVerdict(
        claims=[
            JudgedClaim(claim="a", verdict="supported", evidence="x"),
            JudgedClaim(claim="b", verdict="unsupported"),
            JudgedClaim(claim="c", verdict="contradicted"),
            JudgedClaim(claim="d", verdict="supported", evidence="y"),
        ]
    )
    judge, _ = _judge(processed, verdict)
    result = judge.judge(thread, payload)
    assert result.score == 0.5 and [c.claim for c in result.problems] == ["b", "c"]


def test_no_claims_is_faithful_and_failure_is_none(processed):
    thread, payload = _draft(processed)
    assert _judge(processed, FaithfulnessVerdict())[0].judge(thread, payload).score == 1.0
    assert _judge(processed, None)[0].judge(thread, payload) is None


def test_context_has_shown_facts_and_wrapped_thread(processed):
    thread, payload = _draft(processed)
    judge, llm = _judge(processed, FaithfulnessVerdict())
    judge.judge(thread, payload)
    _, user = llm.seen
    fact = processed.deps.memory.get("facts", payload.context_facts[0])
    assert fact.content in user
    assert '<email_content source="thread">' in user and payload.body in user
