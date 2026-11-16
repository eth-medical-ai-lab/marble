"""One OpenAI-compatible chat backend for remote APIs and local servers."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class ChatResult:
    """Model answer and metadata used by planning and perception evaluation."""

    content: str
    reasoning_content: str = ""
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


class LLMBackend:
    """Send ordinary text/image conversations to a configured chat endpoint."""

    def __init__(self, model_name: str, base_url: str | None = None,
                 api_key: str | None = None) -> None:
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
            timeout=1800.0,
        )

    def chat(self, messages: list[dict[str, Any]], model: str | None = None,
             max_tokens: int = 32768, reasoning_effort: str | None = None) -> ChatResult:
        # Finish reasons and reasoning are logged locally, not sent as chat fields.
        request = {
            "model": model or self.model_name,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": max_tokens,
            "stream": False,
        }
        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort
        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message

        # Preserve reported usage, including reasoning/cache details when supplied.
        usage = response.usage.model_dump(exclude_none=True) if response.usage is not None else None
        if usage is not None:
            completion = usage.get("completion_tokens_details") or {}
            prompt = usage.get("prompt_tokens_details") or {}
            if "reasoning_tokens" in completion:
                usage["reasoning_tokens"] = completion["reasoning_tokens"]
            if "cached_tokens" in prompt:
                usage["cache_read_input_tokens"] = prompt["cached_tokens"]
        return ChatResult(
            content=(message.content or "").strip(),
            reasoning_content=getattr(message, "reasoning_content", None) or getattr(message, "reasoning", "") or "",
            usage=usage,
            finish_reason=choice.finish_reason,
        )


def make_backend(model_name: str, *, local_server_port: int | None = None,
                 local_server_url: str | None = None) -> LLMBackend:
    """An explicit URL wins over a local port; otherwise use OPENAI_BASE_URL."""
    if local_server_url:
        return LLMBackend(model_name, base_url=local_server_url)
    if local_server_port is not None:
        return LLMBackend(model_name, base_url=f"http://localhost:{local_server_port}/v1", api_key="EMPTY")
    return LLMBackend(model_name)
