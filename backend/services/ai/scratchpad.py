"""
SQLite-backed research scratchpad.

Provides persistent audit trail for all AI agent research steps.
Each research task gets a session with ordered entries tracking
every thinking step, tool call, observation, and conclusion.

Inspired by virattt/dexter's JSONL scratchpad pattern but using
SQLite for proper persistence and queryability.
"""

import json
import logging
import uuid
from datetime import timedelta
from utils.utcnow import utcnow
from typing import Optional

from sqlalchemy import select, func, update, delete, and_, desc

from models.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy model imports
# ---------------------------------------------------------------------------
# The ResearchSession and ScratchpadEntry models are being added to
# models/database.py by a parallel effort.  We import them lazily so
# this module can be loaded even if the models haven't been merged yet.


def _get_models():
    """Import database models.  Raises ImportError with a helpful message
    if the AI-specific tables haven't been added yet."""
    from models.database import ResearchSession, ScratchpadEntry

    return ResearchSession, ScratchpadEntry


# ---------------------------------------------------------------------------
# Entry type constants
# ---------------------------------------------------------------------------

ENTRY_TYPE_THINKING = "thinking"
ENTRY_TYPE_TOOL_CALL = "tool_call"
ENTRY_TYPE_TOOL_RESULT = "tool_result"
ENTRY_TYPE_OBSERVATION = "observation"
ENTRY_TYPE_ANSWER = "answer"

VALID_ENTRY_TYPES = {
    ENTRY_TYPE_THINKING,
    ENTRY_TYPE_TOOL_CALL,
    ENTRY_TYPE_TOOL_RESULT,
    ENTRY_TYPE_OBSERVATION,
    ENTRY_TYPE_ANSWER,
}


# ---------------------------------------------------------------------------
# ScratchpadService
# ---------------------------------------------------------------------------


class ScratchpadService:
    """Manages research sessions and their scratchpad entries in SQLite.

    Each research session (e.g. "analyze market resolution criteria for
    market X") gets a unique ID and an ordered list of scratchpad entries
    recording every step the agent took: thinking, tool calls, tool
    results, observations, and the final answer.

    This gives us:
      - A full audit trail of every AI decision.
      - Token usage tracking per step and per session.
      - The ability to replay or resume a research session.
      - Cached results so we don't re-analyze the same market twice.
    """

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_type: str,
        query: str,
        opportunity_id: Optional[str] = None,
        market_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Create a new research session.

        Args:
            session_type: Category of research (e.g. "resolution_analysis",
                          "opportunity_scoring", "market_research").
            query: The question or task the agent was asked.
            opportunity_id: Optional link to an arbitrage opportunity.
            market_id: Optional link to a Polymarket market.
            model: LLM model identifier used for this session.

        Returns:
            The newly created session ID (hex string).
        """
        ResearchSession, _ = _get_models()
        session_id = uuid.uuid4().hex[:16]

        async with AsyncSessionLocal() as db:
            session = ResearchSession(
                id=session_id,
                session_type=session_type,
                query=query,
                opportunity_id=opportunity_id,
                market_id=market_id,
                status="running",
                iterations=0,
                tools_called=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost_usd=0.0,
                model_used=model,
                started_at=utcnow(),
            )
            db.add(session)
            await db.commit()

        logger.info(
            "Created research session %s (type=%s, market=%s)",
            session_id,
            session_type,
            market_id,
        )
        return session_id

    # ------------------------------------------------------------------
    # Entries
    # ------------------------------------------------------------------

    async def add_entry(
        self,
        session_id: str,
        entry_type: str,
        tool_name: Optional[str] = None,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> str:
        """Add an entry to a session's scratchpad.

        Args:
            session_id: ID of the owning research session.
            entry_type: One of "thinking", "tool_call", "tool_result",
                        "observation", "answer".
            tool_name: Name of the tool (for tool_call / tool_result entries).
            input_data: JSON-serialisable dict of input data.
            output_data: JSON-serialisable dict of output / result data.
            input_tokens: Number of LLM input tokens consumed.
            output_tokens: Number of LLM output tokens consumed.

        Returns:
            The newly created entry ID (hex string).
        """
        if entry_type not in VALID_ENTRY_TYPES:
            raise ValueError(f"Invalid entry_type '{entry_type}'. Must be one of: {VALID_ENTRY_TYPES}")

        ResearchSession, ScratchpadEntry = _get_models()
        entry_id = uuid.uuid4().hex[:16]

        async with AsyncSessionLocal() as db:
            # Determine next sequence number
            result = await db.execute(
                select(func.coalesce(func.max(ScratchpadEntry.sequence), 0)).where(
                    ScratchpadEntry.session_id == session_id
                )
            )
            next_seq = result.scalar() + 1

            entry = ScratchpadEntry(
                id=entry_id,
                session_id=session_id,
                sequence=next_seq,
                entry_type=entry_type,
                tool_name=tool_name,
                input_data=input_data,
                output_data=output_data,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                created_at=utcnow(),
            )
            db.add(entry)

            # Update session-level aggregates
            update_values = {
                ResearchSession.total_input_tokens: (ResearchSession.total_input_tokens + input_tokens),
                ResearchSession.total_output_tokens: (ResearchSession.total_output_tokens + output_tokens),
            }

            if entry_type == ENTRY_TYPE_THINKING:
                update_values[ResearchSession.iterations] = ResearchSession.iterations + 1
            elif entry_type == ENTRY_TYPE_TOOL_CALL:
                update_values[ResearchSession.tools_called] = ResearchSession.tools_called + 1

            await db.execute(update(ResearchSession).where(ResearchSession.id == session_id).values(update_values))

            await db.commit()

        logger.debug(
            "Added scratchpad entry %s to session %s (type=%s, seq=%d)",
            entry_id,
            session_id,
            entry_type,
            next_seq,
        )
        return entry_id

    # ------------------------------------------------------------------
    # Session completion
    # ------------------------------------------------------------------

    async def complete_session(
        self,
        session_id: str,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark a session as completed (or failed).

        Exactly one of *result* or *error* should be provided.

        Args:
            session_id: Session to complete.
            result: Final result dict (on success).
            error: Error message (on failure).
        """
        ResearchSession, _ = _get_models()
        now = utcnow()

        async with AsyncSessionLocal() as db:
            # Fetch started_at so we can compute duration
            row = await db.execute(select(ResearchSession.started_at).where(ResearchSession.id == session_id))
            started_at = row.scalar()

            duration = None
            if started_at is not None:
                duration = (now - started_at).total_seconds()

            status = "failed" if error else "completed"

            await db.execute(
                update(ResearchSession)
                .where(ResearchSession.id == session_id)
                .values(
                    status=status,
                    result=result,
                    error=error,
                    completed_at=now,
                    duration_seconds=duration,
                )
            )
            await db.commit()

        logger.info(
            "Completed research session %s (status=%s, duration=%.1fs)",
            session_id,
            status,
            duration or 0,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session with all its entries.

        Returns:
            Dict with session fields plus an ``entries`` list, or *None*
            if the session does not exist.
        """
        ResearchSession, ScratchpadEntry = _get_models()

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
            session = result.scalar_one_or_none()
            if session is None:
                return None

            entries_result = await db.execute(
                select(ScratchpadEntry)
                .where(ScratchpadEntry.session_id == session_id)
                .order_by(ScratchpadEntry.sequence)
            )
            entries = entries_result.scalars().all()

            return {
                "id": session.id,
                "session_type": session.session_type,
                "query": session.query,
                "opportunity_id": session.opportunity_id,
                "market_id": session.market_id,
                "status": session.status,
                "result": session.result,
                "error": session.error,
                "iterations": session.iterations,
                "tools_called": session.tools_called,
                "total_input_tokens": session.total_input_tokens,
                "total_output_tokens": session.total_output_tokens,
                "total_cost_usd": session.total_cost_usd,
                "model_used": session.model_used,
                "started_at": (session.started_at.isoformat() if session.started_at else None),
                "completed_at": (session.completed_at.isoformat() if session.completed_at else None),
                "duration_seconds": session.duration_seconds,
                "entries": [self._entry_to_dict(e) for e in entries],
            }

    async def get_session_entries(self, session_id: str) -> list[dict]:
        """Get all entries for a session, ordered by sequence.

        Returns:
            List of entry dicts.
        """
        _, ScratchpadEntry = _get_models()

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ScratchpadEntry)
                .where(ScratchpadEntry.session_id == session_id)
                .order_by(ScratchpadEntry.sequence)
            )
            entries = result.scalars().all()
            return [self._entry_to_dict(e) for e in entries]

    async def get_recent_sessions(
        self,
        session_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get recent research sessions.

        Args:
            session_type: Optional filter by session type.
            limit: Maximum number of sessions to return (default 20).

        Returns:
            List of session dicts (without entries for performance).
        """
        ResearchSession, _ = _get_models()

        async with AsyncSessionLocal() as db:
            stmt = select(ResearchSession).order_by(desc(ResearchSession.started_at))
            if session_type is not None:
                stmt = stmt.where(ResearchSession.session_type == session_type)
            stmt = stmt.limit(limit)

            result = await db.execute(stmt)
            sessions = result.scalars().all()

            return [self._session_to_dict(s) for s in sessions]

    async def get_session_for_market(
        self,
        market_id: str,
        session_type: str,
    ) -> Optional[dict]:
        """Get the most recent session of a given type for a market.

        Useful for caching -- check if we already analysed this market
        recently before kicking off a new research session.

        Args:
            market_id: Polymarket market ID.
            session_type: Category of research.

        Returns:
            Session dict (with entries) or *None*.
        """
        ResearchSession, _ = _get_models()

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ResearchSession)
                .where(
                    and_(
                        ResearchSession.market_id == market_id,
                        ResearchSession.session_type == session_type,
                    )
                )
                .order_by(desc(ResearchSession.started_at))
                .limit(1)
            )
            session = result.scalar_one_or_none()
            if session is None:
                return None

        # Fetch full session with entries via the existing method
        return await self.get_session(session.id)

    # ------------------------------------------------------------------
    # Context building (for LLM conversation replay)
    # ------------------------------------------------------------------

    async def build_context(self, session_id: str) -> list[dict]:
        """Build LLM context messages from scratchpad entries.

        Converts scratchpad entries into a conversation format suitable
        for feeding back to the LLM for continuation or final answer
        generation.

        Mapping:
            - thinking   -> assistant message
            - tool_call  -> assistant message with tool_calls
            - tool_result -> tool message
            - observation -> assistant message
            - answer      -> assistant message

        Args:
            session_id: ID of the session to reconstruct.

        Returns:
            List of message dicts compatible with the LLM chat interface.
        """
        entries = await self.get_session_entries(session_id)
        messages: list[dict] = []

        for entry in entries:
            entry_type = entry["entry_type"]
            input_data = entry.get("input_data") or {}
            output_data = entry.get("output_data") or {}

            if entry_type == ENTRY_TYPE_THINKING:
                content = output_data.get("content", "")
                if content:
                    messages.append({"role": "assistant", "content": content})

            elif entry_type == ENTRY_TYPE_TOOL_CALL:
                # Reconstruct the assistant message that requested the
                # tool call.
                tool_call_repr = {
                    "id": entry.get("id", ""),
                    "name": entry.get("tool_name", ""),
                    "arguments": input_data,
                }
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [tool_call_repr],
                    }
                )

            elif entry_type == ENTRY_TYPE_TOOL_RESULT:
                result_content = json.dumps(output_data) if isinstance(output_data, dict) else str(output_data)
                messages.append(
                    {
                        "role": "tool",
                        "content": result_content,
                        "tool_call_id": entry.get("id", ""),
                        "name": entry.get("tool_name", ""),
                    }
                )

            elif entry_type == ENTRY_TYPE_OBSERVATION:
                content = output_data.get("content", "")
                if content:
                    messages.append({"role": "assistant", "content": content})

            elif entry_type == ENTRY_TYPE_ANSWER:
                content = output_data.get("content", "")
                if content:
                    messages.append({"role": "assistant", "content": content})

        return messages

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def cleanup_old_sessions(self, days: int = 7) -> int:
        """Delete sessions (and their entries) older than *days* days.

        Args:
            days: Age threshold in days.  Sessions whose ``started_at``
                  is older than this are removed.

        Returns:
            Number of sessions deleted.
        """
        ResearchSession, ScratchpadEntry = _get_models()
        cutoff = utcnow() - timedelta(days=days)

        async with AsyncSessionLocal() as db:
            # Find sessions to delete
            result = await db.execute(select(ResearchSession.id).where(ResearchSession.started_at < cutoff))
            session_ids = [row[0] for row in result.all()]

            if not session_ids:
                return 0

            # Delete entries first (foreign key)
            await db.execute(delete(ScratchpadEntry).where(ScratchpadEntry.session_id.in_(session_ids)))

            # Delete sessions
            await db.execute(delete(ResearchSession).where(ResearchSession.id.in_(session_ids)))

            await db.commit()

        logger.info(
            "Cleaned up %d research sessions older than %d days",
            len(session_ids),
            days,
        )
        return len(session_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_to_dict(entry) -> dict:
        """Convert a ScratchpadEntry ORM object to a plain dict."""
        return {
            "id": entry.id,
            "session_id": entry.session_id,
            "sequence": entry.sequence,
            "entry_type": entry.entry_type,
            "tool_name": entry.tool_name,
            "input_data": entry.input_data,
            "output_data": entry.output_data,
            "input_tokens": entry.input_tokens,
            "output_tokens": entry.output_tokens,
            "created_at": (entry.created_at.isoformat() if entry.created_at else None),
        }

    @staticmethod
    def _session_to_dict(session) -> dict:
        """Convert a ResearchSession ORM object to a plain dict
        (without entries)."""
        return {
            "id": session.id,
            "session_type": session.session_type,
            "query": session.query,
            "opportunity_id": session.opportunity_id,
            "market_id": session.market_id,
            "status": session.status,
            "result": session.result,
            "error": session.error,
            "iterations": session.iterations,
            "tools_called": session.tools_called,
            "total_input_tokens": session.total_input_tokens,
            "total_output_tokens": session.total_output_tokens,
            "total_cost_usd": session.total_cost_usd,
            "model_used": session.model_used,
            "started_at": (session.started_at.isoformat() if session.started_at else None),
            "completed_at": (session.completed_at.isoformat() if session.completed_at else None),
            "duration_seconds": session.duration_seconds,
        }
