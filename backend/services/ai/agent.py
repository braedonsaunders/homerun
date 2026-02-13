"""
Core ReAct agent loop for autonomous research.

Implements the ReAct (Reason + Act) pattern where the LLM:
1. Thinks about what to do next
2. Calls tools to gather information
3. Observes the results
4. Repeats until it has enough data
5. Generates a final answer

Inspired by virattt/dexter's agent loop architecture with typed events
and scratchpad persistence.

All steps are persisted to a SQLite scratchpad for full audit trail.
The agent yields typed ``AgentEvent`` objects so callers can display
real-time progress (e.g. streaming to a WebSocket or SSE endpoint).
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Callable, Optional

from services.ai import get_llm_manager
from services.ai.llm_provider import LLMMessage, ToolDefinition
from services.ai.scratchpad import ScratchpadService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent event types
# ---------------------------------------------------------------------------


class AgentEventType(str, Enum):
    """Types of events emitted by the agent during execution."""

    THINKING = "thinking"
    """LLM is reasoning about its next step."""

    TOOL_START = "tool_start"
    """A tool invocation is about to begin."""

    TOOL_END = "tool_end"
    """A tool invocation completed successfully."""

    TOOL_ERROR = "tool_error"
    """A tool invocation failed."""

    TOOL_LIMIT = "tool_limit"
    """A tool's per-session call limit was reached."""

    ANSWER_START = "answer_start"
    """LLM is producing its final answer (no more tool calls)."""

    DONE = "done"
    """Agent finished successfully.  ``data["result"]`` has the answer."""

    ERROR = "error"
    """Agent failed with an unrecoverable error."""


@dataclass
class AgentEvent:
    """A typed event yielded by the agent during its ReAct loop.

    The ``data`` dict contents depend on the event ``type``:

    +--------------+-----------------------------------------------------+
    | Type         | data keys                                           |
    +==============+=====================================================+
    | THINKING     | ``content`` -- LLM reasoning text                   |
    +--------------+-----------------------------------------------------+
    | TOOL_START   | ``tool`` -- tool name, ``input`` -- arguments dict   |
    +--------------+-----------------------------------------------------+
    | TOOL_END     | ``tool`` -- tool name, ``output`` -- result dict     |
    +--------------+-----------------------------------------------------+
    | TOOL_ERROR   | ``tool`` -- tool name, ``error`` -- error message    |
    +--------------+-----------------------------------------------------+
    | TOOL_LIMIT   | ``tool``, ``count``, ``max``                        |
    +--------------+-----------------------------------------------------+
    | ANSWER_START | ``content`` -- final answer text                    |
    +--------------+-----------------------------------------------------+
    | DONE         | ``result`` -- final result dict, ``session_id``     |
    +--------------+-----------------------------------------------------+
    | ERROR        | ``error`` -- error message, ``session_id``          |
    +--------------+-----------------------------------------------------+
    """

    type: AgentEventType
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent tool wrapper
# ---------------------------------------------------------------------------


@dataclass
class AgentTool:
    """Wrapper describing a tool the agent can invoke.

    Attributes:
        name: Unique tool name (used in LLM function-calling).
        description: Human-readable description shown to the LLM.
        parameters: JSON Schema dict describing the tool's parameters.
        handler: Async callable ``(arguments: dict) -> dict`` that
                 executes the tool.
        max_calls: Maximum number of times this tool can be called in
                   a single agent session (prevents runaway loops).
    """

    name: str
    description: str
    parameters: dict
    handler: Callable  # async (dict) -> dict
    max_calls: int = 5


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

# Soft limit on tool result size to avoid blowing up the context window.
_MAX_TOOL_RESULT_CHARS = 8000


class Agent:
    """Autonomous research agent using the ReAct pattern.

    Yields typed :class:`AgentEvent` objects for real-time progress
    tracking.  All steps are persisted to the SQLite scratchpad for
    audit trail.

    Usage::

        agent = Agent(
            system_prompt="You are a market resolution analyst...",
            tools=[fetch_market_tool, search_news_tool],
            model="gpt-4o-mini",
            max_iterations=10,
            session_type="resolution_analysis",
        )

        async for event in agent.run("Is this market's resolution criteria clear?"):
            if event.type == AgentEventType.DONE:
                result = event.data["result"]
    """

    def __init__(
        self,
        system_prompt: str,
        tools: Optional[list[AgentTool]] = None,
        model: Optional[str] = None,
        max_iterations: int = 10,
        session_type: str = "general",
        temperature: float = 0.0,
    ):
        """Initialise the agent.

        Args:
            system_prompt: System message that frames the LLM's role and
                           instructions.
            tools: List of tools available to the agent.
            model: LLM model identifier.  When *None* the default model
                   from AI settings is used.
            max_iterations: Maximum number of think-act-observe cycles
                            before forcing a final answer.
            session_type: Label for the research session (used for
                          filtering / caching).
            temperature: LLM sampling temperature.
        """
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = model
        self.max_iterations = max_iterations
        self.session_type = session_type
        self.temperature = temperature

        # Runtime state (reset on each run)
        self._tool_call_counts: dict[str, int] = {}
        self._messages: list[LLMMessage] = []
        self._scratchpad = ScratchpadService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        opportunity_id: Optional[str] = None,
        market_id: Optional[str] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent loop.

        Yields :class:`AgentEvent` objects for every significant step.
        The final event is always ``DONE`` (with result) or ``ERROR``.

        Args:
            query: The research question or task.
            opportunity_id: Optional link to an arbitrage opportunity.
            market_id: Optional link to a Polymarket market.

        Yields:
            :class:`AgentEvent` instances.
        """
        # Reset per-run state
        self._tool_call_counts = {}
        self._messages = []

        manager = get_llm_manager()
        if not manager.is_available():
            yield AgentEvent(
                type=AgentEventType.ERROR,
                data={"error": "No LLM provider configured"},
            )
            return

        model = self.model or get_llm_manager()._default_model

        # Create scratchpad session
        session_id = await self._scratchpad.create_session(
            session_type=self.session_type,
            query=query,
            opportunity_id=opportunity_id,
            market_id=market_id,
            model=model,
        )

        logger.info(
            "Starting agent run (session=%s, type=%s, model=%s, max_iter=%d)",
            session_id,
            self.session_type,
            model,
            self.max_iterations,
        )

        try:
            # Build initial messages
            self._messages = [
                LLMMessage(role="system", content=self.system_prompt),
                LLMMessage(role="user", content=query),
            ]

            # Convert AgentTools -> ToolDefinitions for the LLM
            tool_defs = self._build_tool_definitions()

            # --- ReAct loop ---
            for iteration in range(self.max_iterations):
                logger.debug(
                    "Agent iteration %d/%d (session=%s)",
                    iteration + 1,
                    self.max_iterations,
                    session_id,
                )

                # Call the LLM
                response = await manager.chat(
                    messages=self._messages,
                    model=model,
                    tools=tool_defs,
                    temperature=self.temperature,
                    purpose=self.session_type,
                )

                # Record thinking step
                input_tokens = response.usage.input_tokens if response.usage else 0
                output_tokens = response.usage.output_tokens if response.usage else 0

                await self._scratchpad.add_entry(
                    session_id=session_id,
                    entry_type="thinking",
                    input_data={"iteration": iteration},
                    output_data={"content": response.content or ""},
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

                if response.content:
                    yield AgentEvent(
                        type=AgentEventType.THINKING,
                        data={"content": response.content},
                    )

                # --- Tool calls ---
                if response.tool_calls:
                    # Append the assistant's message (with tool_calls) to
                    # the conversation so the LLM sees its own request.
                    self._messages.append(
                        LLMMessage(
                            role="assistant",
                            content=response.content or "",
                            tool_calls=[
                                {
                                    "id": tc.id,
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                }
                                for tc in response.tool_calls
                            ],
                        )
                    )

                    # Execute each tool call
                    for tc in response.tool_calls:
                        async for event in self._execute_tool_call(session_id, tc):
                            yield event

                    # Loop back for next iteration
                    continue

                # --- No tool calls: final answer ---
                if response.content:
                    yield AgentEvent(
                        type=AgentEventType.ANSWER_START,
                        data={"content": response.content},
                    )

                    await self._scratchpad.add_entry(
                        session_id=session_id,
                        entry_type="answer",
                        output_data={"content": response.content},
                    )

                    result = {"answer": response.content}
                    await self._scratchpad.complete_session(session_id=session_id, result=result)

                    yield AgentEvent(
                        type=AgentEventType.DONE,
                        data={"result": result, "session_id": session_id},
                    )
                    return

            # --- Max iterations exhausted ---
            logger.warning(
                "Agent hit max iterations (%d) for session %s; forcing answer",
                self.max_iterations,
                session_id,
            )
            async for event in self._force_final_answer(session_id, model, manager):
                yield event

        except Exception as e:
            logger.exception("Agent error in session %s: %s", session_id, e)
            await self._scratchpad.complete_session(session_id=session_id, error=str(e))
            yield AgentEvent(
                type=AgentEventType.ERROR,
                data={"error": str(e), "session_id": session_id},
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_tool_definitions(self) -> Optional[list[ToolDefinition]]:
        """Convert ``AgentTool`` list to ``ToolDefinition`` list for the
        LLM provider, or return *None* if no tools are configured."""
        if not self.tools:
            return None
        return [
            ToolDefinition(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
            )
            for t in self.tools
        ]

    def _get_tool(self, name: str) -> Optional[AgentTool]:
        """Find a registered tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    async def _execute_tool_call(
        self,
        session_id: str,
        tc,  # ToolCall dataclass from LLMResponse
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute a single tool call and yield progress events.

        Also appends the tool result message to ``self._messages`` and
        records the step in the scratchpad.
        """
        tool = self._get_tool(tc.name)

        # Unknown tool
        if tool is None:
            logger.warning("Agent requested unknown tool: %s", tc.name)
            yield AgentEvent(
                type=AgentEventType.TOOL_ERROR,
                data={"tool": tc.name, "error": "Unknown tool"},
            )
            self._messages.append(
                LLMMessage(
                    role="tool",
                    content=f"Error: Unknown tool '{tc.name}'",
                    tool_call_id=tc.id,
                    name=tc.name,
                )
            )
            return

        # Check per-tool call limit
        count = self._tool_call_counts.get(tc.name, 0)
        if count >= tool.max_calls:
            logger.info(
                "Tool %s hit call limit (%d/%d) in session %s",
                tc.name,
                count,
                tool.max_calls,
                session_id,
            )
            yield AgentEvent(
                type=AgentEventType.TOOL_LIMIT,
                data={
                    "tool": tc.name,
                    "count": count,
                    "max": tool.max_calls,
                },
            )
            self._messages.append(
                LLMMessage(
                    role="tool",
                    content=(f"Tool call limit reached ({tool.max_calls}). Use available data to proceed."),
                    tool_call_id=tc.id,
                    name=tc.name,
                )
            )
            return

        # Execute the tool
        yield AgentEvent(
            type=AgentEventType.TOOL_START,
            data={"tool": tc.name, "input": tc.arguments},
        )

        try:
            result = await tool.handler(tc.arguments)
            self._tool_call_counts[tc.name] = count + 1

            # Record in scratchpad
            await self._scratchpad.add_entry(
                session_id=session_id,
                entry_type="tool_call",
                tool_name=tc.name,
                input_data=tc.arguments,
                output_data=result,
            )

            yield AgentEvent(
                type=AgentEventType.TOOL_END,
                data={"tool": tc.name, "output": result},
            )

            # Serialize and truncate the result for the LLM context
            result_str = json.dumps(result, default=str) if isinstance(result, dict) else str(result)
            if len(result_str) > _MAX_TOOL_RESULT_CHARS:
                result_str = result_str[:_MAX_TOOL_RESULT_CHARS] + "\n... [truncated]"

            self._messages.append(
                LLMMessage(
                    role="tool",
                    content=result_str,
                    tool_call_id=tc.id,
                    name=tc.name,
                )
            )

        except Exception as e:
            error_msg = f"Tool error: {e}"
            logger.error(
                "Tool %s failed in session %s: %s",
                tc.name,
                session_id,
                e,
            )

            yield AgentEvent(
                type=AgentEventType.TOOL_ERROR,
                data={"tool": tc.name, "error": error_msg},
            )

            await self._scratchpad.add_entry(
                session_id=session_id,
                entry_type="tool_call",
                tool_name=tc.name,
                input_data=tc.arguments,
                output_data={"error": error_msg},
            )

            self._messages.append(
                LLMMessage(
                    role="tool",
                    content=error_msg,
                    tool_call_id=tc.id,
                    name=tc.name,
                )
            )

    async def _force_final_answer(
        self,
        session_id: str,
        model: str,
        manager,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Force the LLM to produce a final answer when max iterations
        are exhausted, by re-calling without tools."""
        self._messages.append(
            LLMMessage(
                role="user",
                content=(
                    "You've reached the maximum number of research steps. "
                    "Based on all the data you've gathered so far, provide "
                    "your final answer now."
                ),
            )
        )

        final_response = await manager.chat(
            messages=self._messages,
            model=model,
            tools=None,  # No tools -- force a text answer
            temperature=self.temperature,
            purpose=self.session_type,
        )

        answer_text = final_response.content or ""
        result = {"answer": answer_text, "max_iterations_reached": True}

        if answer_text:
            yield AgentEvent(
                type=AgentEventType.ANSWER_START,
                data={"content": answer_text},
            )

        await self._scratchpad.add_entry(
            session_id=session_id,
            entry_type="answer",
            output_data={"content": answer_text},
            input_tokens=(final_response.usage.input_tokens if final_response.usage else 0),
            output_tokens=(final_response.usage.output_tokens if final_response.usage else 0),
        )

        await self._scratchpad.complete_session(session_id=session_id, result=result)

        yield AgentEvent(
            type=AgentEventType.DONE,
            data={"result": result, "session_id": session_id},
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


async def run_agent_to_completion(
    system_prompt: str,
    query: str,
    tools: Optional[list[AgentTool]] = None,
    model: Optional[str] = None,
    max_iterations: int = 10,
    session_type: str = "general",
    opportunity_id: Optional[str] = None,
    market_id: Optional[str] = None,
) -> dict:
    """Run an agent to completion and return the final result.

    Convenience wrapper around :meth:`Agent.run` that consumes the async
    generator internally and returns only the final result dict.

    Args:
        system_prompt: System message for the LLM.
        query: The research question or task.
        tools: Optional list of tools.
        model: LLM model identifier (or *None* for default).
        max_iterations: Maximum ReAct cycles.
        session_type: Session category label.
        opportunity_id: Optional opportunity link.
        market_id: Optional market link.

    Returns:
        A dict with at least ``"result"`` and ``"session_id"`` keys.
        The ``"result"`` dict contains an ``"answer"`` key with the
        LLM's final answer text.

    Raises:
        RuntimeError: If the agent encounters an error or completes
            without producing a result.
    """
    agent = Agent(
        system_prompt=system_prompt,
        tools=tools,
        model=model,
        max_iterations=max_iterations,
        session_type=session_type,
    )

    result = None
    async for event in agent.run(
        query,
        opportunity_id=opportunity_id,
        market_id=market_id,
    ):
        if event.type == AgentEventType.DONE:
            result = event.data
        elif event.type == AgentEventType.ERROR:
            raise RuntimeError(event.data.get("error", "Agent failed"))

    if result is None:
        raise RuntimeError("Agent completed without producing a result")

    return result
