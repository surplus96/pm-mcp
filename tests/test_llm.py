"""Offline tests for the Claude LLM wrapper (no network access)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mcp_server.tools import llm


def _message(*texts: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=t) for t in texts],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        model="claude-sonnet-5",
    )


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


@pytest.fixture
def fake_client(monkeypatch):
    def install(response):
        messages = FakeMessages(response)
        monkeypatch.setattr(llm, "_client", SimpleNamespace(messages=messages))
        llm.circuit_llm.reset()
        return messages

    return install


def test_call_llm_uses_sonnet_5_without_sampling_params(fake_client):
    messages = fake_client(_message("Hello ", "world"))
    assert llm.call_llm("sys", "hi") == "Hello world"
    sent = messages.calls[0]
    assert sent["model"] == "claude-sonnet-5"
    assert sent["system"] == "sys"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in sent


def test_refusal_raises(fake_client):
    fake_client(_message("", stop_reason="refusal"))
    with pytest.raises(llm.LLMRefusalError, match="cyber"):
        llm.call_llm("sys", "hi")


def test_summarize_swallows_refusal(fake_client):
    fake_client(_message("", stop_reason="refusal"))
    assert llm.summarize_items(["a", "b"]) == ""


def test_summarize_skips_empty_input(fake_client):
    messages = fake_client(_message("unused"))
    assert llm.summarize_text("   ") == ""
    assert messages.calls == []


def test_structured_output(fake_client):
    from mcp_server.tools.news_sentiment import _call_llm_sentiment, _LLMSentiment

    parsed = _LLMSentiment(overall_sentiment="bullish", sentiment_score=1.7, key_themes=["AI"],
                           summary="Strong demand.", confidence=0.8)
    response = _message("{}")
    response.parsed_output = parsed
    messages = fake_client(response)

    out = _call_llm_sentiment([{"title": "Chip demand surges", "snippet": "..."}])
    assert out["overall_sentiment"] == "bullish"
    assert out["sentiment_score"] == 1.0  # clamped
    assert messages.calls[0]["output_format"] is _LLMSentiment
