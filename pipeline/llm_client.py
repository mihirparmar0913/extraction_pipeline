"""Step 4 - talk to the LLM. Everything provider-specific lives in this file.

Two clients are provided:
  - AnthropicClient  : original Anthropic SDK client (forced tool call)
  - OpenAICompatClient: works with any OpenAI-compatible endpoint
                        (NeuralWatt / OpenAI / local models etc.)
                        Uses JSON-mode structured output instead of tool calls.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Protocol

DEFAULT_MODEL = os.environ.get("EXTRACTION_MODEL", "glm-5.2")
TOOL_NAME = "record_extraction"

# NeuralWatt / OpenAI-compatible endpoint
OPENAI_COMPAT_BASE_URL = os.environ.get(
    "OPENAI_COMPAT_BASE_URL", "https://api.neuralwatt.com/v1"
)
OPENAI_COMPAT_API_KEY = os.environ.get("OPENAI_COMPAT_API_KEY", "")


class LLMError(Exception):
    """The model call failed or returned something unusable."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
        )

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens


@dataclass(frozen=True)
class LLMResult:
    data: dict
    usage: Usage


class LLMClient(Protocol):
    model: str

    def extract(self, *, system: str, blocks: list[dict], tool_schema: dict) -> LLMResult: ...


# ---------------------------------------------------------------------------
# Block helpers (provider-neutral → Anthropic API format)
# ---------------------------------------------------------------------------

def _to_anthropic_block(block: dict) -> dict:
    if block["type"] == "text":
        out = {"type": "text", "text": block["text"]}
    else:
        out = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block["media_type"],
                "data": base64.b64encode(block["data"]).decode(),
            },
        }
    if block.get("cache"):
        out["cache_control"] = {"type": "ephemeral"}
    return out


def _blocks_to_text(blocks: list[dict]) -> str:
    """Flatten blocks to plain text for OpenAI-compatible clients.

    Images become a placeholder note since the model may not support vision.
    """
    parts = []
    for b in blocks:
        if b["type"] == "text":
            parts.append(b["text"])
        else:
            parts.append(f"[IMAGE: scanned page — {len(b.get('data', b''))} bytes PNG]")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Anthropic client (original)
# ---------------------------------------------------------------------------

class AnthropicClient:
    """Forces the model to answer through a tool whose input schema IS our JSON schema,
    so the output is always structured JSON (no free-text parsing)."""

    def __init__(self, model: str | None = None, *, max_tokens: int = 8000, temperature: float = 0.0):
        import anthropic  # imported lazily so tests / dry-runs work without it configured

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.model = model or DEFAULT_MODEL
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=3)
        self._max_tokens = max_tokens
        self._temperature: float | None = temperature

    def _call(self, system, blocks, tool_schema):
        kwargs = dict(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            tools=[{
                "name": TOOL_NAME,
                "description": "Record the extracted lease data.",
                "input_schema": tool_schema,
            }],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": [_to_anthropic_block(b) for b in blocks]}],
        )
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        return self._client.messages.create(**kwargs)

    def extract(self, *, system: str, blocks: list[dict], tool_schema: dict) -> LLMResult:
        try:
            try:
                resp = self._call(system, blocks, tool_schema)
            except self._anthropic.BadRequestError as exc:
                # Some models do not accept a temperature parameter; retry without it.
                if self._temperature is not None and "temperature" in str(exc).lower():
                    self._temperature = None
                    resp = self._call(system, blocks, tool_schema)
                else:
                    raise
        except self._anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        if resp.stop_reason == "max_tokens":
            raise LLMError("Model output was truncated (max_tokens reached)")
        for block in resp.content:
            if block.type == "tool_use":
                u = resp.usage
                usage = Usage(
                    u.input_tokens,
                    u.output_tokens,
                    getattr(u, "cache_creation_input_tokens", 0) or 0,
                    getattr(u, "cache_read_input_tokens", 0) or 0,
                )
                return LLMResult(dict(block.input), usage)
        raise LLMError("Model did not return a tool call")


# ---------------------------------------------------------------------------
# OpenAI-compatible client (NeuralWatt / glm-5.2 / any OpenAI-API endpoint)
# ---------------------------------------------------------------------------

class OpenAICompatClient:
    """OpenAI-compatible chat completions client.

    Works with NeuralWatt (https://api.neuralwatt.com/v1) and any other
    service exposing the OpenAI /v1/chat/completions interface.

    Uses JSON-mode output: the model is instructed to return a JSON object
    matching the given schema. The response is parsed and validated client-side.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 8000,
        temperature: float = 0.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "openai package is required for OpenAICompatClient: pip install openai"
            ) from exc

        self.model = model or DEFAULT_MODEL
        resolved_base_url = base_url or OPENAI_COMPAT_BASE_URL
        resolved_key = api_key or OPENAI_COMPAT_API_KEY
        if not resolved_key:
            raise LLMError(
                "API key not set. Pass api_key= or set OPENAI_COMPAT_API_KEY env var."
            )
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base_url)
        self._max_tokens = max_tokens
        self._temperature = temperature

    def extract(self, *, system: str, blocks: list[dict], tool_schema: dict) -> LLMResult:
        # Flatten blocks to text (images become placeholders)
        user_content = _blocks_to_text(blocks)

        # Build the schema instruction
        schema_str = json.dumps(tool_schema, indent=2)
        instruction = (
            f"{system}\n\n"
            f"You MUST respond with a single valid JSON object that exactly matches "
            f"the following JSON Schema. Do NOT include any text before or after the JSON.\n\n"
            f"SCHEMA:\n{schema_str}"
        )

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_content},
        ]

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as exc:
            # Surface payment / auth / quota errors clearly
            msg = str(exc)
            if "402" in msg or "payment_required" in msg or "allowance" in msg.lower():
                raise LLMError(
                    "API key quota exhausted (402 payment_required). "
                    "Please top up the NeuralWatt account or use a different API key."
                ) from exc
            if "401" in msg or "unauthorized" in msg.lower():
                raise LLMError(f"API authentication failed: {exc}") from exc
            raise LLMError(f"OpenAI-compat API error: {exc}") from exc

        choice = resp.choices[0] if resp.choices else None
        if not choice:
            raise LLMError("API returned no choices")

        finish_reason = getattr(choice, "finish_reason", "")
        import logging as _log
        _log.getLogger(__name__).debug(
            "finish_reason=%s output_snippet=%r",
            finish_reason,
            (choice.message.content or "")[:200],
        )
        if finish_reason == "length":
            snippet = (choice.message.content or "")[:300]
            raise LLMError(
                f"Model output was truncated (max_tokens={self._max_tokens} reached). "
                f"Response snippet: {snippet!r}"
            )

        raw_text = (choice.message.content or "").strip()

        # Strip markdown code fences if the model wrapped JSON in ```json ... ```
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            # Remove opening ``` or ```json line
            lines = lines[1:] if lines else lines
            # Remove closing ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Model returned non-JSON response: {exc}\n"
                f"Raw output (first 500 chars): {raw_text[:500]}"
            ) from exc

        if not isinstance(data, dict):
            raise LLMError(f"Model returned {type(data).__name__}, expected a JSON object")

        usage_obj = getattr(resp, "usage", None)
        usage = Usage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
        )
        return LLMResult(data, usage)
