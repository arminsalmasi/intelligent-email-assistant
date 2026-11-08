"""Regenerate docs/thread_graph.mmd from the compiled LangGraph (single source of truth)."""

from pathlib import Path

from email_assistant.config import ROOT_DIR, Settings
from email_assistant.service import AssistantService

svc = AssistantService.create(
    Settings(llm_provider="none", embedding_provider="hashing", log_level="WARNING"), ephemeral=True
)
out = Path(ROOT_DIR / "docs" / "thread_graph.mmd")
out.write_text(svc.graph.get_graph().draw_mermaid())
print(out)
