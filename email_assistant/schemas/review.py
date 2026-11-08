"""Human-in-the-loop contract: what the graph asks, what the human answers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from email_assistant.schemas.actions import ProposedAction


class ReviewRequest(BaseModel):
    """Payload emitted with `interrupt()`; rendered by CLI / API / dashboard."""

    thread_id: str
    subject: str
    summary: str
    round: int
    actions: list[ProposedAction]
    instructions: str = (
        "Decide per action: approve, edit (with payload changes) or reject. "
        "Soft findings may be overridden with a justification; hard findings cannot."
    )


class HumanDecision(BaseModel):
    action_id: str
    decision: Literal["approve", "edit", "reject"]
    edits: dict[str, Any] = Field(default_factory=dict)
    comment: str | None = None
    override_soft_findings: bool = False
    justification: str | None = None

    @model_validator(mode="after")
    def _check(self) -> HumanDecision:
        if self.decision == "edit" and not self.edits:
            raise ValueError("decision 'edit' requires non-empty edits")
        if self.override_soft_findings and not self.justification:
            raise ValueError("overriding soft findings requires a justification")
        forbidden = {"kind", "hold_id", "event_id"} & set(self.edits)
        if forbidden:
            raise ValueError(f"fields cannot be edited: {sorted(forbidden)}")
        return self


class ReviewResponse(BaseModel):
    decisions: list[HumanDecision]
    reviewer: str = "owner"
