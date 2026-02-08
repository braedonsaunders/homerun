"""
Multi-provider LLM abstraction layer.

Supports: OpenAI, Anthropic, Google (Gemini), xAI (Grok), DeepSeek,
and any OpenAI-compatible API.

Provider is detected by model name prefix:
- gpt-*, o1-*, o3-*, o4-*, chatgpt-* -> OpenAI
- claude-* -> Anthropic
- gemini-* -> Google
- grok-* -> xAI
- deepseek-* -> DeepSeek
- Any other -> tries OpenAI-compatible with OPENAI_API_BASE

Usage:
    manager = LLMManager()
    await manager.initialize()  # Load API keys from AppSettings

    response = await manager.chat(
        messages=[LLMMessage(role="user", content="Analyze this market")],
        model="gpt-4o-mini",
    )

    # Structured output (JSON schema)
    result = await manager.structured_output(
        messages=[...],
        schema={"type": "object", "properties": {"score": {"type": "number"}}},
        model="gpt-4o-mini",
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import httpx
from sqlalchemy import func, select

from models.database import AppSettings, AsyncSessionLocal, LLMModelCache, LLMUsageLog

logger = logging.getLogger(__name__)


# ==================== PRICING ====================

# Approximate pricing per 1M tokens (input, output) as of 2025
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3-mini": (1.10, 4.40),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-opus-4-6": (15.00, 75.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
}


# ==================== DATA CLASSES ====================


@dataclass
class LLMMessage:
    """A single message in a chat conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[list] = None  # For assistant messages with tool calls
    tool_call_id: Optional[str] = None  # For tool response messages
    name: Optional[str] = None  # For tool messages


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by the LLM."""

    name: str
    description: str
    parameters: dict  # JSON Schema


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict  # Parsed JSON arguments


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    tool_calls: Optional[list[ToolCall]] = None
    usage: Optional[TokenUsage] = None
    model: str = ""
    provider: str = ""
    latency_ms: int = 0


# ==================== ENUMS ====================


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    XAI = "xai"
    DEEPSEEK = "deepseek"


# ==================== RETRY LOGIC ====================

_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds


async def _retry_with_backoff(
    coro_factory, max_retries: int = _MAX_RETRIES, base_delay: float = _BASE_DELAY
):
    """Execute an async callable with exponential backoff on retryable errors.

    Retries on HTTP 429 (rate limit) and 5xx (server errors).

    Args:
        coro_factory: A callable that returns a new coroutine each invocation.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.

    Returns:
        The httpx.Response from the successful request.

    Raises:
        httpx.HTTPStatusError: If all retries are exhausted.
        httpx.RequestError: If a non-retryable request error occurs.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = await coro_factory()
            if response.status_code == 429 or response.status_code >= 500:
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "LLM request returned %d, retrying in %.1fs (attempt %d/%d)",
                        response.status_code,
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise last_exc
            return response
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "LLM request failed (%s), retrying in %.1fs (attempt %d/%d)",
                    str(exc),
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ==================== BASE PROVIDER ====================


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    Each provider implements the raw HTTP communication with its
    respective API, handling message format conversion, tool calling,
    and structured output.
    """

    provider: LLMProvider
    api_key: str

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        model: str,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: Conversation messages.
            model: Model identifier (e.g. "gpt-4o-mini").
            tools: Optional tool definitions for function calling.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            LLMResponse with content, tool calls, and usage info.
        """
        pass

    @abstractmethod
    async def structured_output(
        self,
        messages: list[LLMMessage],
        schema: dict,
        model: str,
        temperature: float = 0.0,
    ) -> dict:
        """Get structured JSON output conforming to a schema.

        Args:
            messages: Conversation messages.
            schema: JSON Schema the output must conform to.
            model: Model identifier.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON dict conforming to the schema.
        """
        pass

    async def list_models(self) -> list[dict[str, str]]:
        """Fetch available models from the provider API.

        Returns:
            List of dicts with 'id' and 'name' keys.
        """
        return []

    def _estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost in USD based on model pricing.

        Args:
            model: Model identifier.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        pricing = PRICING.get(model)
        if pricing is None:
            # Try prefix matching for versioned model names
            for known_model, known_pricing in PRICING.items():
                if model.startswith(known_model):
                    pricing = known_pricing
                    break
        if pricing is None:
            # Default conservative estimate
            pricing = (5.0, 15.0)
        input_cost = (input_tokens / 1_000_000) * pricing[0]
        output_cost = (output_tokens / 1_000_000) * pricing[1]
        return round(input_cost + output_cost, 6)


# ==================== OPENAI PROVIDER ====================


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider using raw httpx calls.

    Supports GPT-4o, GPT-4.1, o1, o3, and other OpenAI models.
    Also serves as the base for OpenAI-compatible providers (xAI, DeepSeek).
    """

    provider = LLMProvider.OPENAI

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key.
            base_url: API base URL (override for compatible providers).
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for OpenAI API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _format_messages(self, messages: list[LLMMessage]) -> list[dict]:
        """Convert LLMMessage objects to OpenAI API format.

        Args:
            messages: List of LLMMessage objects.

        Returns:
            List of dicts in OpenAI message format.
        """
        formatted = []
        for msg in messages:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls is not None:
                entry["tool_calls"] = msg.tool_calls
            if msg.tool_call_id is not None:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.name is not None:
                entry["name"] = msg.name
            formatted.append(entry)
        return formatted

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert ToolDefinition objects to OpenAI tools format.

        Args:
            tools: List of ToolDefinition objects.

        Returns:
            List of dicts in OpenAI tools format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _parse_tool_calls(self, raw_tool_calls: list[dict]) -> list[ToolCall]:
        """Parse OpenAI tool call responses into ToolCall objects.

        Args:
            raw_tool_calls: Raw tool call dicts from the API response.

        Returns:
            List of parsed ToolCall objects.
        """
        result = []
        for tc in raw_tool_calls:
            try:
                arguments = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                arguments = {}
            result.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc["function"]["name"],
                    arguments=arguments,
                )
            )
        return result

    async def list_models(self) -> list[dict[str, str]]:
        """Fetch available models from the OpenAI API."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._build_headers(),
                )
            if response.status_code != 200:
                logger.warning("Failed to list OpenAI models: %s", response.text[:200])
                return []
            data = response.json()
            models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                if any(
                    model_id.startswith(p)
                    for p in ("gpt-", "o1-", "o3-", "o4-", "chatgpt-")
                ):
                    models.append({"id": model_id, "name": model_id})
            models.sort(key=lambda x: x["id"])
            return models
        except Exception as exc:
            logger.warning("Error listing OpenAI models: %s", exc)
            return []

    async def chat(
        self,
        messages: list[LLMMessage],
        model: str,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request to the OpenAI API.

        Args:
            messages: Conversation messages.
            model: Model identifier (e.g. "gpt-4o-mini").
            tools: Optional tool definitions.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            LLMResponse with content and usage.
        """
        start_ms = int(time.time() * 1000)

        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = self._format_tools(tools)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _retry_with_backoff(
                lambda: client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=payload,
                )
            )

        latency_ms = int(time.time() * 1000) - start_ms
        data = response.json()

        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", response.text)
            raise RuntimeError(
                f"OpenAI API error ({response.status_code}): {error_msg}"
            )

        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content", "") or ""

        parsed_tool_calls = None
        if message.get("tool_calls"):
            parsed_tool_calls = self._parse_tool_calls(message["tool_calls"])

        usage_data = data.get("usage", {})
        usage = TokenUsage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            usage=usage,
            model=model,
            provider=self.provider.value,
            latency_ms=latency_ms,
        )

    async def structured_output(
        self,
        messages: list[LLMMessage],
        schema: dict,
        model: str,
        temperature: float = 0.0,
    ) -> dict:
        """Get structured JSON output from the OpenAI API.

        Uses response_format with json_schema for models that support it,
        falling back to a system prompt instruction approach.

        Args:
            messages: Conversation messages.
            schema: JSON Schema for the response.
            model: Model identifier.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON dict conforming to the schema.
        """
        # Add system instruction for JSON output
        json_instruction = (
            "You MUST respond with valid JSON matching this schema. "
            "Do not include any text outside the JSON object.\n"
            f"Schema: {json.dumps(schema)}"
        )

        augmented_messages = list(messages)
        if augmented_messages and augmented_messages[0].role == "system":
            augmented_messages[0] = LLMMessage(
                role="system",
                content=augmented_messages[0].content + "\n\n" + json_instruction,
            )
        else:
            augmented_messages.insert(
                0, LLMMessage(role="system", content=json_instruction)
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(augmented_messages),
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _retry_with_backoff(
                lambda: client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=payload,
                )
            )

        data = response.json()

        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", response.text)
            raise RuntimeError(
                f"OpenAI API error ({response.status_code}): {error_msg}"
            )

        content = data["choices"][0]["message"].get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse structured output as JSON: %s", content[:500])
            raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc


# ==================== ANTHROPIC PROVIDER ====================


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider using raw httpx calls.

    Supports Claude Sonnet, Haiku, and Opus models with tool use
    and structured output. Handles the anthropic-version header and
    cache_control for system prompts.
    """

    provider = LLMProvider.ANTHROPIC

    def __init__(self, api_key: str):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key.
        """
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for Anthropic API requests."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _format_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[Optional[str], list[dict]]:
        """Convert LLMMessage objects to Anthropic API format.

        Anthropic separates system prompt from messages. This method
        extracts the system message and formats the rest.

        Args:
            messages: List of LLMMessage objects.

        Returns:
            Tuple of (system_prompt, formatted_messages).
        """
        system_prompt = None
        formatted = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
                continue

            if msg.role == "tool":
                # Anthropic tool results use a different format
                formatted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content,
                            }
                        ],
                    }
                )
                continue

            if msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool use
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", "") if isinstance(tc, dict) else tc.id,
                            "name": tc.get("function", {}).get("name", "")
                            if isinstance(tc, dict)
                            else tc.name,
                            "input": (
                                json.loads(
                                    tc.get("function", {}).get("arguments", "{}")
                                )
                                if isinstance(tc, dict)
                                else tc.arguments
                            ),
                        }
                    )
                formatted.append({"role": "assistant", "content": content_blocks})
                continue

            formatted.append({"role": msg.role, "content": msg.content})

        return system_prompt, formatted

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert ToolDefinition objects to Anthropic tools format.

        Args:
            tools: List of ToolDefinition objects.

        Returns:
            List of dicts in Anthropic tools format.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def _parse_tool_calls(self, content_blocks: list[dict]) -> list[ToolCall]:
        """Parse Anthropic tool use blocks into ToolCall objects.

        Args:
            content_blocks: Content blocks from the API response.

        Returns:
            List of parsed ToolCall objects.
        """
        result = []
        for block in content_blocks:
            if block.get("type") == "tool_use":
                result.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )
        return result

    async def list_models(self) -> list[dict[str, str]]:
        """Fetch available models from the Anthropic API."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._build_headers(),
                )
            if response.status_code != 200:
                logger.warning(
                    "Failed to list Anthropic models: %s", response.text[:200]
                )
                return []
            data = response.json()
            models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                display_name = m.get("display_name", model_id)
                models.append({"id": model_id, "name": display_name})
            models.sort(key=lambda x: x["id"])
            return models
        except Exception as exc:
            logger.warning("Error listing Anthropic models: %s", exc)
            return []

    async def chat(
        self,
        messages: list[LLMMessage],
        model: str,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a message request to the Anthropic API.

        Args:
            messages: Conversation messages.
            model: Model identifier (e.g. "claude-sonnet-4-5-20250929").
            tools: Optional tool definitions.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            LLMResponse with content and usage.
        """
        start_ms = int(time.time() * 1000)

        system_prompt, formatted_messages = self._format_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = self._format_tools(tools)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _retry_with_backoff(
                lambda: client.post(
                    f"{self.base_url}/messages",
                    headers=self._build_headers(),
                    json=payload,
                )
            )

        latency_ms = int(time.time() * 1000) - start_ms
        data = response.json()

        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", response.text)
            raise RuntimeError(
                f"Anthropic API error ({response.status_code}): {error_msg}"
            )

        # Extract text content and tool calls
        content_blocks = data.get("content", [])
        text_parts = []
        parsed_tool_calls = None

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
        if tool_use_blocks:
            parsed_tool_calls = self._parse_tool_calls(content_blocks)

        usage_data = data.get("usage", {})
        usage = TokenUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0)
            + usage_data.get("output_tokens", 0),
        )

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=parsed_tool_calls,
            usage=usage,
            model=model,
            provider=self.provider.value,
            latency_ms=latency_ms,
        )

    async def structured_output(
        self,
        messages: list[LLMMessage],
        schema: dict,
        model: str,
        temperature: float = 0.0,
    ) -> dict:
        """Get structured JSON output from the Anthropic API.

        Uses a system prompt instruction to guide the model to produce
        valid JSON matching the requested schema.

        Args:
            messages: Conversation messages.
            schema: JSON Schema for the response.
            model: Model identifier.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON dict conforming to the schema.
        """
        json_instruction = (
            "You MUST respond with valid JSON matching this schema. "
            "Do not include any text outside the JSON object. "
            "Do not use markdown code fences.\n"
            f"Schema: {json.dumps(schema)}"
        )

        augmented_messages = list(messages)
        if augmented_messages and augmented_messages[0].role == "system":
            augmented_messages[0] = LLMMessage(
                role="system",
                content=augmented_messages[0].content + "\n\n" + json_instruction,
            )
        else:
            augmented_messages.insert(
                0, LLMMessage(role="system", content=json_instruction)
            )

        response = await self.chat(
            messages=augmented_messages,
            model=model,
            temperature=temperature,
            max_tokens=4096,
        )

        content = response.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [line for line in lines if not line.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse Anthropic structured output: %s", content[:500]
            )
            raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc


# ==================== GOOGLE PROVIDER ====================


class GoogleProvider(BaseLLMProvider):
    """Google Gemini API provider using raw httpx calls.

    Supports Gemini 2.0 Flash, 2.5 Pro, and other Gemini models
    via the generativelanguage API.
    """

    provider = LLMProvider.GOOGLE

    def __init__(self, api_key: str):
        """Initialize the Google Gemini provider.

        Args:
            api_key: Google API key.
        """
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _format_contents(
        self, messages: list[LLMMessage]
    ) -> tuple[Optional[dict], list[dict]]:
        """Convert LLMMessage objects to Gemini API format.

        Gemini uses 'contents' with 'parts' structure and a separate
        system_instruction field.

        Args:
            messages: List of LLMMessage objects.

        Returns:
            Tuple of (system_instruction, formatted_contents).
        """
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
                continue

            role = "user" if msg.role == "user" else "model"
            contents.append(
                {
                    "role": role,
                    "parts": [{"text": msg.content}],
                }
            )

        return system_instruction, contents

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert ToolDefinition objects to Gemini tools format.

        Args:
            tools: List of ToolDefinition objects.

        Returns:
            List of dicts in Gemini function declarations format.
        """
        function_declarations = []
        for tool in tools:
            function_declarations.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            )
        return [{"function_declarations": function_declarations}]

    async def list_models(self) -> list[dict[str, str]]:
        """Fetch available models from the Google Gemini API."""
        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
            if response.status_code != 200:
                logger.warning("Failed to list Google models: %s", response.text[:200])
                return []
            data = response.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                # name is like "models/gemini-2.0-flash" - extract the model id
                model_id = (
                    name.replace("models/", "") if name.startswith("models/") else name
                )
                display_name = m.get("displayName", model_id)
                # Only include generative models that support generateContent
                supported = m.get("supportedGenerationMethods", [])
                if "generateContent" in supported:
                    models.append({"id": model_id, "name": display_name})
            models.sort(key=lambda x: x["id"])
            return models
        except Exception as exc:
            logger.warning("Error listing Google models: %s", exc)
            return []

    async def chat(
        self,
        messages: list[LLMMessage],
        model: str,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a generate content request to the Gemini API.

        Args:
            messages: Conversation messages.
            model: Model identifier (e.g. "gemini-2.0-flash").
            tools: Optional tool definitions.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            LLMResponse with content and usage.
        """
        start_ms = int(time.time() * 1000)

        system_instruction, contents = self._format_contents(messages)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if tools:
            payload["tools"] = self._format_tools(tools)

        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _retry_with_backoff(
                lambda: client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
            )

        latency_ms = int(time.time() * 1000) - start_ms
        data = response.json()

        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", response.text)
            raise RuntimeError(
                f"Google API error ({response.status_code}): {error_msg}"
            )

        # Parse response
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Google API returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = []
        parsed_tool_calls = None

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                if parsed_tool_calls is None:
                    parsed_tool_calls = []
                fc = part["functionCall"]
                parsed_tool_calls.append(
                    ToolCall(
                        id=uuid.uuid4().hex[:16],
                        name=fc.get("name", ""),
                        arguments=fc.get("args", {}),
                    )
                )

        usage_data = data.get("usageMetadata", {})
        usage = TokenUsage(
            input_tokens=usage_data.get("promptTokenCount", 0),
            output_tokens=usage_data.get("candidatesTokenCount", 0),
            total_tokens=usage_data.get("totalTokenCount", 0),
        )

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=parsed_tool_calls,
            usage=usage,
            model=model,
            provider=self.provider.value,
            latency_ms=latency_ms,
        )

    async def structured_output(
        self,
        messages: list[LLMMessage],
        schema: dict,
        model: str,
        temperature: float = 0.0,
    ) -> dict:
        """Get structured JSON output from the Gemini API.

        Uses Gemini's response_mime_type to request JSON output.

        Args:
            messages: Conversation messages.
            schema: JSON Schema for the response.
            model: Model identifier.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON dict conforming to the schema.
        """
        json_instruction = (
            "You MUST respond with valid JSON matching this schema. "
            "Do not include any text outside the JSON object.\n"
            f"Schema: {json.dumps(schema)}"
        )

        system_instruction, contents = self._format_contents(messages)

        # Add JSON instruction to system or as user prefix
        if system_instruction:
            existing_text = system_instruction["parts"][0]["text"]
            system_instruction["parts"][0]["text"] = (
                existing_text + "\n\n" + json_instruction
            )
        else:
            system_instruction = {"parts": [{"text": json_instruction}]}

        payload: dict[str, Any] = {
            "contents": contents,
            "systemInstruction": system_instruction,
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }

        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _retry_with_backoff(
                lambda: client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
            )

        data = response.json()

        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", response.text)
            raise RuntimeError(
                f"Google API error ({response.status_code}): {error_msg}"
            )

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Google API returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        content = parts[0].get("text", "") if parts else ""

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Google structured output: %s", content[:500])
            raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc


# ==================== XAI PROVIDER ====================


class XAIProvider(BaseLLMProvider):
    """xAI (Grok) provider using OpenAI-compatible API.

    xAI exposes an OpenAI-compatible endpoint at https://api.x.ai/v1.
    This provider delegates to OpenAIProvider with a custom base URL.
    """

    provider = LLMProvider.XAI

    def __init__(self, api_key: str):
        """Initialize the xAI provider.

        Args:
            api_key: xAI API key.
        """
        self.api_key = api_key
        self._delegate = OpenAIProvider(api_key=api_key, base_url="https://api.x.ai/v1")

    async def list_models(self) -> list[dict[str, str]]:
        """Fetch available models from the xAI API."""
        models = await self._delegate.list_models()
        # Filter to grok models only
        return [m for m in models if m["id"].startswith("grok-")]

    async def chat(
        self,
        messages: list[LLMMessage],
        model: str,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request via the xAI API.

        Args:
            messages: Conversation messages.
            model: Model identifier (e.g. "grok-3").
            tools: Optional tool definitions.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            LLMResponse with content and usage.
        """
        response = await self._delegate.chat(
            messages, model, tools, temperature, max_tokens
        )
        response.provider = self.provider.value
        return response

    async def structured_output(
        self,
        messages: list[LLMMessage],
        schema: dict,
        model: str,
        temperature: float = 0.0,
    ) -> dict:
        """Get structured JSON output from the xAI API.

        Args:
            messages: Conversation messages.
            schema: JSON Schema for the response.
            model: Model identifier.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON dict conforming to the schema.
        """
        return await self._delegate.structured_output(
            messages, schema, model, temperature
        )


# ==================== DEEPSEEK PROVIDER ====================


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek provider using OpenAI-compatible API.

    DeepSeek exposes an OpenAI-compatible endpoint at https://api.deepseek.com/v1.
    This provider delegates to OpenAIProvider with a custom base URL.
    """

    provider = LLMProvider.DEEPSEEK

    def __init__(self, api_key: str):
        """Initialize the DeepSeek provider.

        Args:
            api_key: DeepSeek API key.
        """
        self.api_key = api_key
        self._delegate = OpenAIProvider(
            api_key=api_key, base_url="https://api.deepseek.com/v1"
        )

    async def list_models(self) -> list[dict[str, str]]:
        """Fetch available models from the DeepSeek API."""
        models = await self._delegate.list_models()
        # Filter to deepseek models only
        return [m for m in models if m["id"].startswith("deepseek-")]

    async def chat(
        self,
        messages: list[LLMMessage],
        model: str,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request via the DeepSeek API.

        Args:
            messages: Conversation messages.
            model: Model identifier (e.g. "deepseek-chat").
            tools: Optional tool definitions.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            LLMResponse with content and usage.
        """
        response = await self._delegate.chat(
            messages, model, tools, temperature, max_tokens
        )
        response.provider = self.provider.value
        return response

    async def structured_output(
        self,
        messages: list[LLMMessage],
        schema: dict,
        model: str,
        temperature: float = 0.0,
    ) -> dict:
        """Get structured JSON output from the DeepSeek API.

        Args:
            messages: Conversation messages.
            schema: JSON Schema for the response.
            model: Model identifier.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON dict conforming to the schema.
        """
        return await self._delegate.structured_output(
            messages, schema, model, temperature
        )


# ==================== LLM MANAGER ====================


class LLMManager:
    """Central manager for LLM interactions.

    Routes requests to the appropriate provider based on the model name.
    Handles provider initialization from database settings, usage tracking,
    cost management, and spend limits.

    Usage:
        manager = LLMManager()
        await manager.initialize()

        response = await manager.chat(
            messages=[LLMMessage(role="user", content="Hello")],
            model="gpt-4o-mini",
        )
    """

    def __init__(self):
        """Initialize the LLM manager with empty provider registry."""
        self._providers: dict[LLMProvider, BaseLLMProvider] = {}
        self._initialized = False
        self._monthly_spend = 0.0
        self._spend_limit = 50.0
        self._default_model: str = "gpt-4o-mini"

    async def initialize(self) -> None:
        """Load API keys from AppSettings database table and initialize providers.

        Queries the AppSettings table for configured API keys and creates
        the corresponding provider instances. Also loads the current month's
        spend from the LLMUsageLog table.
        """
        # Reset providers so re-initialization picks up removed keys
        self._providers.clear()

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(AppSettings).where(AppSettings.id == "default")
                )
                app_settings = result.scalar_one_or_none()

                if app_settings is None:
                    logger.info("No AppSettings found, LLM providers not configured")
                    self._initialized = True
                    return

                # Initialize providers for each configured API key
                if app_settings.openai_api_key:
                    self._providers[LLMProvider.OPENAI] = OpenAIProvider(
                        api_key=app_settings.openai_api_key
                    )
                    logger.info("Initialized OpenAI LLM provider")

                if app_settings.anthropic_api_key:
                    self._providers[LLMProvider.ANTHROPIC] = AnthropicProvider(
                        api_key=app_settings.anthropic_api_key
                    )
                    logger.info("Initialized Anthropic LLM provider")

                if app_settings.google_api_key:
                    self._providers[LLMProvider.GOOGLE] = GoogleProvider(
                        api_key=app_settings.google_api_key
                    )
                    logger.info("Initialized Google LLM provider")

                if app_settings.xai_api_key:
                    self._providers[LLMProvider.XAI] = XAIProvider(
                        api_key=app_settings.xai_api_key
                    )
                    logger.info("Initialized xAI LLM provider")

                if app_settings.deepseek_api_key:
                    self._providers[LLMProvider.DEEPSEEK] = DeepSeekProvider(
                        api_key=app_settings.deepseek_api_key
                    )
                    logger.info("Initialized DeepSeek LLM provider")

                # Load spend settings
                if app_settings.ai_max_monthly_spend is not None:
                    self._spend_limit = app_settings.ai_max_monthly_spend

                _provider_default_models = {
                    LLMProvider.OPENAI: "gpt-4o-mini",
                    LLMProvider.ANTHROPIC: "claude-sonnet-4-5-20250929",
                    LLMProvider.GOOGLE: "gemini-2.0-flash",
                    LLMProvider.XAI: "grok-3-mini",
                    LLMProvider.DEEPSEEK: "deepseek-chat",
                }

                if app_settings.ai_default_model:
                    self._default_model = app_settings.ai_default_model
                elif app_settings.llm_model:
                    self._default_model = app_settings.llm_model
                else:
                    self._default_model = "gpt-4o-mini"

                # Validate that the default model's provider is actually
                # configured.  If not, fall back to the first available
                # provider's default model so we don't error at runtime.
                default_provider = self.detect_provider(self._default_model)
                if default_provider not in self._providers and self._providers:
                    for p, m in _provider_default_models.items():
                        if p in self._providers:
                            self._default_model = m
                            break

                # Load current month's spend
                now = datetime.utcnow()
                month_start = now.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                spend_result = await session.execute(
                    select(func.coalesce(func.sum(LLMUsageLog.cost_usd), 0.0)).where(
                        LLMUsageLog.requested_at >= month_start,
                        LLMUsageLog.success == True,  # noqa: E712
                    )
                )
                self._monthly_spend = float(spend_result.scalar() or 0.0)

        except Exception as exc:
            logger.error("Failed to initialize LLM providers: %s", exc)
            # Don't crash -- just run without AI features

        self._initialized = True
        logger.info(
            "LLM manager initialized: %d providers, $%.2f spent this month (limit $%.2f)",
            len(self._providers),
            self._monthly_spend,
            self._spend_limit,
        )

    def detect_provider(self, model: str) -> LLMProvider:
        """Detect the LLM provider from a model name prefix.

        Args:
            model: Model identifier string.

        Returns:
            The detected LLMProvider enum value.
        """
        model_lower = model.lower()
        if any(
            model_lower.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-", "chatgpt-")
        ):
            return LLMProvider.OPENAI
        elif model_lower.startswith("claude-"):
            return LLMProvider.ANTHROPIC
        elif model_lower.startswith("gemini-"):
            return LLMProvider.GOOGLE
        elif model_lower.startswith("grok-"):
            return LLMProvider.XAI
        elif model_lower.startswith("deepseek-"):
            return LLMProvider.DEEPSEEK
        else:
            # Fall back to the first configured provider rather than
            # assuming OpenAI is always available.
            if self._providers:
                return next(iter(self._providers))
            return LLMProvider.OPENAI

    def _get_provider(self, provider_enum: LLMProvider) -> BaseLLMProvider:
        """Get an initialized provider instance.

        Args:
            provider_enum: The provider to retrieve.

        Returns:
            The provider instance.

        Raises:
            RuntimeError: If the provider is not configured.
        """
        provider = self._providers.get(provider_enum)
        if provider is None:
            raise RuntimeError(
                f"LLM provider '{provider_enum.value}' is not configured. "
                f"Add the API key in Settings to use this provider."
            )
        return provider

    def _check_spend_limit(self) -> None:
        """Check if the monthly spend limit has been exceeded.

        A spend limit of 0 disables the check entirely.

        Raises:
            RuntimeError: If the spend limit has been exceeded.
        """
        if self._spend_limit <= 0:
            return  # Limit disabled
        if self._monthly_spend >= self._spend_limit:
            raise RuntimeError(
                f"Monthly LLM spend limit reached (${self._monthly_spend:.2f} / "
                f"${self._spend_limit:.2f}). Increase the limit in Settings or "
                f"wait until next month."
            )

    async def _log_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: int,
        purpose: Optional[str] = None,
        session_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Log LLM usage to the database for cost tracking.

        Args:
            provider: Provider name.
            model: Model identifier.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cost_usd: Estimated cost in USD.
            latency_ms: Request latency in milliseconds.
            purpose: Purpose of the request.
            session_id: Research session ID.
            success: Whether the request succeeded.
            error: Error message if the request failed.
        """
        try:
            async with AsyncSessionLocal() as session:
                log_entry = LLMUsageLog(
                    id=uuid.uuid4().hex[:16],
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    purpose=purpose,
                    session_id=session_id,
                    requested_at=datetime.utcnow(),
                    latency_ms=latency_ms,
                    success=success,
                    error=error,
                )
                session.add(log_entry)
                await session.commit()

            if success:
                self._monthly_spend += cost_usd

        except Exception as exc:
            logger.error("Failed to log LLM usage: %s", exc)

    async def chat(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> LLMResponse:
        """Send a chat completion request to the appropriate provider.

        Detects the provider from the model name, checks spend limits,
        calls the provider, logs usage, and returns the response.

        Args:
            messages: Conversation messages.
            model: Model identifier. Defaults to the configured default model.
            tools: Optional tool definitions for function calling.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            purpose: Purpose label for usage tracking.
            session_id: Research session ID for linking.

        Returns:
            LLMResponse with content, tool calls, and usage.
        """
        if not self._initialized:
            raise RuntimeError("LLM manager not initialized. Call initialize() first.")

        model = model or self._default_model
        provider_enum = self.detect_provider(model)
        provider = self._get_provider(provider_enum)
        self._check_spend_limit()

        try:
            response = await provider.chat(
                messages=messages,
                model=model,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Calculate cost and log usage
            cost = 0.0
            if response.usage:
                cost = provider._estimate_cost(
                    model,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )
                await self._log_usage(
                    provider=provider_enum.value,
                    model=model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cost_usd=cost,
                    latency_ms=response.latency_ms,
                    purpose=purpose,
                    session_id=session_id,
                )

            logger.debug(
                "LLM chat: model=%s, tokens=%d/%d, cost=$%.4f, latency=%dms",
                model,
                response.usage.input_tokens if response.usage else 0,
                response.usage.output_tokens if response.usage else 0,
                cost,
                response.latency_ms,
            )

            return response

        except RuntimeError:
            raise
        except Exception as exc:
            # Log the failed request
            await self._log_usage(
                provider=provider_enum.value,
                model=model,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=0,
                purpose=purpose,
                session_id=session_id,
                success=False,
                error=str(exc),
            )
            raise RuntimeError(f"LLM request failed: {exc}") from exc

    async def structured_output(
        self,
        messages: list[LLMMessage],
        schema: dict,
        model: Optional[str] = None,
        temperature: float = 0.0,
        purpose: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """Get structured JSON output conforming to a schema.

        Routes to the appropriate provider and returns parsed JSON.

        Args:
            messages: Conversation messages.
            schema: JSON Schema the output must conform to.
            model: Model identifier. Defaults to the configured default model.
            temperature: Sampling temperature.
            purpose: Purpose label for usage tracking.
            session_id: Research session ID for linking.

        Returns:
            Parsed JSON dict conforming to the schema.
        """
        if not self._initialized:
            raise RuntimeError("LLM manager not initialized. Call initialize() first.")

        model = model or self._default_model
        provider_enum = self.detect_provider(model)
        provider = self._get_provider(provider_enum)
        self._check_spend_limit()

        start_time = time.time()
        try:
            result = await provider.structured_output(
                messages=messages,
                schema=schema,
                model=model,
                temperature=temperature,
            )

            # Log successful structured_output usage.
            # Provider structured_output returns a dict (no usage info),
            # so estimate tokens from the schema prompt + response size.
            latency_ms = int((time.time() - start_time) * 1000)
            estimated_input = (
                sum(len(m.content) // 4 for m in messages)
                + len(json.dumps(schema)) // 4
            )
            estimated_output = len(json.dumps(result)) // 4
            cost = provider._estimate_cost(model, estimated_input, estimated_output)
            await self._log_usage(
                provider=provider_enum.value,
                model=model,
                input_tokens=estimated_input,
                output_tokens=estimated_output,
                cost_usd=cost,
                latency_ms=latency_ms,
                purpose=purpose,
                session_id=session_id,
            )

            return result

        except Exception as exc:
            latency_ms = int((time.time() - start_time) * 1000)
            await self._log_usage(
                provider=provider_enum.value,
                model=model,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=latency_ms,
                purpose=purpose,
                session_id=session_id,
                success=False,
                error=str(exc),
            )
            raise RuntimeError(f"Structured output request failed: {exc}") from exc

    async def get_usage_stats(self) -> dict:
        """Get usage statistics from the database.

        Returns a summary of LLM usage including total spend, per-provider
        breakdown, per-model breakdown, and request counts for the current
        month.

        Returns:
            Dict with usage statistics.
        """
        now = datetime.utcnow()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        try:
            async with AsyncSessionLocal() as session:
                # Total spend this month (successful requests)
                total_result = await session.execute(
                    select(
                        func.coalesce(func.sum(LLMUsageLog.cost_usd), 0.0),
                        func.coalesce(func.sum(LLMUsageLog.input_tokens), 0),
                        func.coalesce(func.sum(LLMUsageLog.output_tokens), 0),
                        func.count(LLMUsageLog.id),
                        func.coalesce(func.avg(LLMUsageLog.latency_ms), 0.0),
                    ).where(
                        LLMUsageLog.requested_at >= month_start,
                        LLMUsageLog.success == True,  # noqa: E712
                    )
                )
                total_row = total_result.one()
                total_cost = float(total_row[0])
                total_input = int(total_row[1])
                total_output = int(total_row[2])
                total_requests = int(total_row[3])
                avg_latency = float(total_row[4])

                # Per-provider breakdown
                provider_result = await session.execute(
                    select(
                        LLMUsageLog.provider,
                        func.coalesce(func.sum(LLMUsageLog.cost_usd), 0.0),
                        func.count(LLMUsageLog.id),
                    )
                    .where(
                        LLMUsageLog.requested_at >= month_start,
                        LLMUsageLog.success == True,  # noqa: E712
                    )
                    .group_by(LLMUsageLog.provider)
                )
                provider_rows = provider_result.all()

                # Per-model breakdown
                model_result = await session.execute(
                    select(
                        LLMUsageLog.model,
                        func.count(LLMUsageLog.id),
                        func.coalesce(func.sum(LLMUsageLog.input_tokens), 0),
                        func.coalesce(func.sum(LLMUsageLog.output_tokens), 0),
                        func.coalesce(func.sum(LLMUsageLog.cost_usd), 0.0),
                    )
                    .where(
                        LLMUsageLog.requested_at >= month_start,
                        LLMUsageLog.success == True,  # noqa: E712
                    )
                    .group_by(LLMUsageLog.model)
                )
                model_rows = model_result.all()

                # Error count
                error_result = await session.execute(
                    select(func.count(LLMUsageLog.id)).where(
                        LLMUsageLog.requested_at >= month_start,
                        LLMUsageLog.success == False,  # noqa: E712
                    )
                )
                error_count = error_result.scalar() or 0

            return {
                "month_start": month_start.isoformat(),
                "total_cost_usd": total_cost,
                "estimated_cost": total_cost,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "total_requests": total_requests,
                "successful_requests": total_requests,
                "failed_requests": int(error_count),
                "error_count": int(error_count),
                "avg_latency_ms": round(avg_latency, 1),
                "spend_limit_usd": self._spend_limit,
                "spend_remaining_usd": max(0.0, self._spend_limit - total_cost),
                "providers": {
                    row[0]: {"cost_usd": float(row[1]), "requests": int(row[2])}
                    for row in provider_rows
                },
                "by_model": {
                    row[0]: {
                        "requests": int(row[1]),
                        "tokens": int(row[2]) + int(row[3]),
                        "input_tokens": int(row[2]),
                        "output_tokens": int(row[3]),
                        "cost": float(row[4]),
                    }
                    for row in model_rows
                    if row[0]
                },
                "configured_providers": [p.value for p in self._providers],
            }

        except Exception as exc:
            logger.error("Failed to get usage stats: %s", exc)
            return {
                "error": str(exc),
                "configured_providers": [p.value for p in self._providers],
            }

    async def fetch_and_cache_models(
        self, provider_name: Optional[str] = None
    ) -> dict[str, list[dict[str, str]]]:
        """Fetch models from provider APIs and cache them in the database.

        Args:
            provider_name: Optional provider name to refresh. If None, refreshes all.

        Returns:
            Dict mapping provider name to list of model dicts.
        """
        results: dict[str, list[dict[str, str]]] = {}

        providers_to_refresh = {}
        if provider_name:
            for p_enum, p_inst in self._providers.items():
                if p_enum.value == provider_name:
                    providers_to_refresh[p_enum] = p_inst
                    break
        else:
            providers_to_refresh = dict(self._providers)

        for p_enum, p_inst in providers_to_refresh.items():
            try:
                models = await p_inst.list_models()
                results[p_enum.value] = models

                # Cache in database
                async with AsyncSessionLocal() as session:
                    # Delete old entries for this provider
                    from sqlalchemy import delete

                    await session.execute(
                        delete(LLMModelCache).where(
                            LLMModelCache.provider == p_enum.value
                        )
                    )
                    # Insert new entries
                    for m in models:
                        entry = LLMModelCache(
                            id=f"{p_enum.value}_{m['id']}",
                            provider=p_enum.value,
                            model_id=m["id"],
                            display_name=m.get("name", m["id"]),
                        )
                        session.add(entry)
                    await session.commit()

                logger.info(
                    "Cached %d models for provider %s",
                    len(models),
                    p_enum.value,
                )
            except Exception as exc:
                logger.error("Failed to fetch models for %s: %s", p_enum.value, exc)
                results[p_enum.value] = []

        return results

    async def get_cached_models(
        self, provider_name: Optional[str] = None
    ) -> dict[str, list[dict[str, str]]]:
        """Get cached models from the database.

        Args:
            provider_name: Optional provider to filter by.

        Returns:
            Dict mapping provider name to list of model dicts.
        """
        try:
            async with AsyncSessionLocal() as session:
                query = select(LLMModelCache)
                if provider_name:
                    query = query.where(LLMModelCache.provider == provider_name)
                query = query.order_by(LLMModelCache.provider, LLMModelCache.model_id)
                result = await session.execute(query)
                rows = result.scalars().all()

            models_by_provider: dict[str, list[dict[str, str]]] = {}
            for row in rows:
                if row.provider not in models_by_provider:
                    models_by_provider[row.provider] = []
                models_by_provider[row.provider].append(
                    {"id": row.model_id, "name": row.display_name or row.model_id}
                )
            return models_by_provider
        except Exception as exc:
            logger.error("Failed to get cached models: %s", exc)
            return {}

    def is_available(self) -> bool:
        """Check if any LLM provider is configured and ready.

        Returns:
            True if at least one provider is initialized.
        """
        return len(self._providers) > 0
