from pydantic import BaseModel

from email_assistant.llm.cache import JsonlCache
from email_assistant.llm.client import LLMClient
from email_assistant.observability.metrics import Telemetry


class Out(BaseModel):
    answer: str


class _Raw:
    usage_metadata = {"input_tokens": 10, "output_tokens": 3}


class FlakyModel:
    """Minimal stand-in for a LangChain chat model: fails `fail` times, then answers."""

    def __init__(self, fail: int):
        self.fail = fail
        self.calls = 0

    def with_structured_output(self, schema, include_raw=True):
        model = self

        class R:
            def invoke(self, messages):
                model.calls += 1
                if model.calls <= model.fail:
                    raise ConnectionError("503")
                return {"parsed": schema(answer="ok"), "raw": _Raw()}

        return R()


def _client(model, tmp_path, mode="record", retries=2):
    return LLMClient(
        model,
        "m",
        Telemetry(),
        JsonlCache(tmp_path / "c.jsonl", mode),
        max_retries=retries,
        sleep=lambda s: None,
    )


def test_retries_then_succeeds_and_records(tmp_path):
    model = FlakyModel(fail=2)
    c = _client(model, tmp_path)
    assert c.structured(Out, "sys", "user", "t").answer == "ok"
    s = c.telemetry.summary()["llm"]
    assert model.calls == 3 and s["failures"] == 2 and s["input_tokens"] == 10


def test_replay_needs_no_model(tmp_path):
    _client(FlakyModel(0), tmp_path).structured(Out, "sys", "user", "t")
    replay = _client(None, tmp_path, mode="replay")
    assert replay.available
    assert replay.structured(Out, "sys", "user", "t").answer == "ok"
    assert replay.structured(Out, "sys", "other", "t") is None  # miss -> fallback
    assert replay.cache_misses == 1


def test_exhausted_retries_return_none(tmp_path):
    c = _client(FlakyModel(fail=10), tmp_path, retries=1)
    assert c.structured(Out, "sys", "user", "t") is None
    assert c.telemetry.summary()["fallbacks"] == {"t": 1}
