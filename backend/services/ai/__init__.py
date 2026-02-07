"""
AI Intelligence Layer for Homerun.

Inspired by virattt/dexter's autonomous research agent architecture.
Provides LLM-powered analysis for:
- Resolution criteria analysis (preventing ambiguous resolution losses)
- Opportunity scoring (LLM-as-judge replacing/augmenting logistic regression)
- Market research with sub-agent routing
- News/sentiment analysis
- Pluggable skill system for reusable analysis workflows
"""

from __future__ import annotations

from services.ai.llm_provider import (
    LLMManager,
    LLMMessage,
    LLMResponse,
    ToolDefinition,
    ToolCall,
    LLMProvider as LLMProviderEnum,
)

_llm_manager: LLMManager | None = None


async def initialize_ai() -> LLMManager:
    """Initialize the AI subsystem. Called from main.py lifespan.

    Creates and initializes the global LLMManager instance,
    loading API keys from the database and setting up providers.

    Returns:
        The initialized LLMManager instance.
    """
    global _llm_manager
    _llm_manager = LLMManager()
    await _llm_manager.initialize()
    return _llm_manager


def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance.

    Returns:
        The initialized LLMManager.

    Raises:
        RuntimeError: If initialize_ai() has not been called yet.
    """
    if _llm_manager is None:
        raise RuntimeError("AI subsystem not initialized. Call initialize_ai() first.")
    return _llm_manager


__all__ = [
    "initialize_ai",
    "get_llm_manager",
    "LLMManager",
    "LLMMessage",
    "LLMResponse",
    "ToolDefinition",
    "ToolCall",
    "LLMProviderEnum",
]
