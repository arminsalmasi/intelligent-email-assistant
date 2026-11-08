"""Procedural memory: learning from human edits.

Two mechanisms:
1. Edit exemplars - (original, edited, comment) stored and retrieved for the
   drafter prompt as "past corrections".
2. Deterministic preference learning - if the human consistently changes the
   sign-off for an audience, that sign-off becomes the audience preference
   (works without an LLM, so learning is observable in offline mode too).
"""

from __future__ import annotations

from email_assistant.memory.store import MemoryStore
from email_assistant.schemas.memory import MemoryHit, MemoryKind, MemorySource, Sensitivity

SIGN_OFF_MARKERS = (
    "best",
    "regards",
    "cheers",
    "thanks",
    "thank you",
    "hälsningar",
    "saludos",
    "kind regards",
    "br,",
    "/",
)


def extract_sign_off(body: str) -> str | None:
    lines = [ln.rstrip() for ln in body.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    for i in range(len(lines) - 2, max(len(lines) - 4, -1), -1):
        if lines[i].lower().startswith(SIGN_OFF_MARKERS):
            return "\n".join(lines[i:])
    return None


class FeedbackMemory:
    def __init__(self, store: MemoryStore):
        self.store = store

    def record_edit(
        self,
        thread_id: str,
        action_id: str,
        category: str,
        audience: str,
        original: str,
        edited: str,
        comment: str | None,
    ) -> list[str]:
        """Returns a list of learned items (for audit / UI)."""
        learned = []
        self.store.put(
            MemoryKind.PROCEDURAL,
            "edits",
            f"{thread_id}:{action_id}",
            f"ORIGINAL:\n{original}\n\nEDITED BY OWNER:\n{edited}\n\nCOMMENT: {comment or '-'}",
            metadata={"category": category, "audience": audience, "comment": comment or ""},
            source=MemorySource.HITL_FEEDBACK,
            sensitivity=Sensitivity.INTERNAL,
            embed_text=f"{category} {audience} {comment or ''}\n{edited}",
        )
        learned.append("edit_exemplar")
        before, after = extract_sign_off(original), extract_sign_off(edited)
        if after and after != before:
            self.store.put(
                MemoryKind.SEMANTIC,
                "preferences",
                f"sign_off:{audience}",
                after,
                metadata={"learned_from": f"{thread_id}:{action_id}"},
                source=MemorySource.HITL_FEEDBACK,
                sensitivity=Sensitivity.INTERNAL,
            )
            learned.append(f"sign_off:{audience}")
        return learned

    def lessons(self, query: str, category: str, k: int = 2) -> list[MemoryHit]:
        hits = self.store.search(query, k=6, kind=MemoryKind.PROCEDURAL, namespaces=["edits"])
        same = [h for h in hits if h.record.metadata.get("category") == category]
        return (same or hits)[:k]

    def sign_off(self, audience: str, default: str) -> str:
        learned = self.store.get("preferences", f"sign_off:{audience}")
        if learned:
            return learned.content
        base = self.store.get("preferences", "sign_off")
        return base.content if base else default
