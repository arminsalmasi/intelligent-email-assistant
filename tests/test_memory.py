from email_assistant.memory.embeddings import HashingEmbedder, cosine
from email_assistant.memory.episodic import adapt_template
from email_assistant.memory.feedback import extract_sign_off
from email_assistant.memory.profile import protected_terms
from email_assistant.schemas.memory import MemoryKind, MemorySource, Sensitivity
from email_assistant.schemas.profile import OrgFact


def test_hashing_embedder_similarity():
    e = HashingEmbedder()
    a, b, c = e.embed(["invoice payment due", "payment of the invoice is due", "lunch on saturday"])
    assert cosine(a, b) > cosine(a, c)


def test_source_priority_prevents_overwrite(svc):
    store = svc.deps.memory
    assert store.get("preferences", "sign_off").source == MemorySource.USER_EXPLICIT
    assert (
        store.put(
            MemoryKind.SEMANTIC, "preferences", "sign_off", "Yo", source=MemorySource.OBSERVED
        )
        is None
    )
    assert store.get("preferences", "sign_off").content.startswith("Best")


def test_semantic_search_finds_relevant_fact(svc):
    hits = svc.deps.memory.search(
        "what are your support hours", k=3, kind=MemoryKind.SEMANTIC, namespaces=["facts"]
    )
    assert any(h.record.key == "support_hours" for h in hits)


def test_protected_terms():
    f = OrgFact(
        key="b", content="Budget is EUR 350,000 for 2026.", sensitivity=Sensitivity.CONFIDENTIAL
    )
    terms = protected_terms(f)
    assert {"350,000", "350000", "350k"} <= set(terms) and "2026" not in terms
    assert protected_terms(
        OrgFact(key="p", content="Project Aurora", sensitivity="confidential")
    ) == ["Aurora"]


def test_feedback_learns_sign_off(svc):
    fb = svc.deps.feedback
    learned = fb.record_edit(
        "t",
        "a",
        "customer",
        "external",
        "Hi X,\n\nText.\n\nBest,\nAlex",
        "Hi X,\n\nText.\n\nCheers,\nAlex",
        "warmer",
    )
    assert "sign_off:external" in learned
    assert fb.sign_off("external", "Best,\nAlex") == "Cheers,\nAlex"
    assert fb.sign_off("internal", "Best,\nAlex").startswith("Best")
    assert extract_sign_off("Hi\n\nBody\n\nKind regards,\nAlex") == "Kind regards,\nAlex"


def test_template_adaptation():
    assert adapt_template("Hi Anna,\n\nThanks!\n\nBest,\nAlex", "Tom").startswith("Hi Tom,")
