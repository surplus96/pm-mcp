"""LLM access (Anthropic Claude).

Every LLM call in the server goes through :func:`call_llm` (free text) or
:func:`call_llm_structured` (Pydantic-validated JSON). Retries on 429 / 5xx /
connection errors are handled by the Anthropic SDK (``LLM_MAX_RETRIES``); the
local circuit breaker stops hammering the API after repeated failures.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, TypeVar

import anthropic
from pydantic import BaseModel

from mcp_server.tools.resilience import CircuitOpenError, Timeout, circuit_llm

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
LLM_MAX_OUTPUT_TOKENS_DEFAULT = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "16000"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
# Optional effort override (low|medium|high|xhigh|max); unset uses the API default.
LLM_EFFORT = os.getenv("LLM_EFFORT", "").strip() or None

ModelT = TypeVar("ModelT", bound=BaseModel)

_client: anthropic.Anthropic | None = None
_client_lock = threading.Lock()


class LLMRefusalError(RuntimeError):
    """The model declined the request (stop_reason == "refusal")."""


def get_client() -> anthropic.Anthropic:
    """Shared client. Credentials resolve from ANTHROPIC_API_KEY (or an `ant auth login` profile)."""
    global _client
    with _client_lock:
        if _client is None:
            _client = anthropic.Anthropic(timeout=Timeout.LLM, max_retries=LLM_MAX_RETRIES)
        return _client


def _request_kwargs(system: str, user: str, model: str | None, max_tokens: int | None) -> dict:
    kwargs: dict = {
        "model": model or CLAUDE_MODEL,
        "max_tokens": max_tokens or LLM_MAX_OUTPUT_TOKENS_DEFAULT,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if LLM_EFFORT:
        kwargs["output_config"] = {"effort": LLM_EFFORT}
    return kwargs


def _check_stop(response: anthropic.types.Message) -> None:
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) if response.stop_details else None
        raise LLMRefusalError(f"model declined the request (category={category})")
    if response.stop_reason == "max_tokens":
        logger.warning(
            "LLM response truncated by max_tokens (model=%s). Raise LLM_MAX_OUTPUT_TOKENS.", response.model
        )


def call_llm(system: str, user: str, *, model: str | None = None, max_tokens: int | None = None) -> str:
    """Single-turn text completion. Returns the concatenated text blocks."""

    def _do_request() -> anthropic.types.Message:
        return get_client().messages.create(**_request_kwargs(system, user, model, max_tokens))

    response = circuit_llm.call(_do_request)
    _check_stop(response)
    return "".join(b.text for b in response.content if b.type == "text").strip()


def call_llm_structured(
    system: str,
    user: str,
    output_format: type[ModelT],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
) -> ModelT:
    """Single-turn completion constrained to ``output_format`` (validated Pydantic instance)."""

    def _do_request():
        return get_client().messages.parse(
            **_request_kwargs(system, user, model, max_tokens), output_format=output_format
        )

    response = circuit_llm.call(_do_request)
    _check_stop(response)
    if response.parsed_output is None:
        raise ValueError("LLM returned no structured output")
    return response.parsed_output


def summarize_text(text: str, max_sentences: int = 6, model: str | None = None) -> str:
    """텍스트 요약. 실패하면 빈 문자열을 반환해 리포트 생성은 계속 진행됩니다."""
    if not text or not text.strip():
        return ""

    system = (
        f"You are a concise financial analyst. Summarize in {max_sentences} sentences (bullet-ready). "
        "Focus on drivers, risks, guidance, and near-term catalysts."
    )
    try:
        return call_llm(system, text[:8000], model=model, max_tokens=2048)
    except CircuitOpenError:
        logger.warning("LLM circuit open, skipping summarization")
    except LLMRefusalError as e:
        logger.warning("LLM summarization refused: %s", e)
    except anthropic.AnthropicError as e:  # API errors and missing credentials
        logger.warning("LLM summarization failed: %s", e)
    return ""


def summarize_items(lines: List[str], max_sentences: int = 6) -> str:
    """리스트 항목 요약."""
    text = "\n".join(f"- {ln}" for ln in lines if ln)
    return summarize_text(text, max_sentences=max_sentences)
