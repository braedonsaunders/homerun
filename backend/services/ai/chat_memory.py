"""
Persistent memory for AI copilot chat sessions.

Stores chat sessions/messages in SQLite so conversations survive refreshes
and panel close/open cycles.
"""

from __future__ import annotations

import uuid
from utils.utcnow import utcnow
from typing import Optional

from sqlalchemy import desc, select

from models.database import AIChatMessage, AIChatSession, AsyncSessionLocal


class ChatMemoryService:
    """CRUD helper for persistent copilot sessions."""

    async def create_session(
        self,
        context_type: Optional[str] = None,
        context_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> dict:
        session_id = uuid.uuid4().hex[:16]
        now = utcnow()
        row = AIChatSession(
            id=session_id,
            context_type=context_type,
            context_id=context_id,
            title=title,
            archived=False,
            created_at=now,
            updated_at=now,
        )
        async with AsyncSessionLocal() as session:
            session.add(row)
            await session.commit()
        return self._session_to_dict(row)

    async def get_session(self, session_id: str, message_limit: int = 200) -> Optional[dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(AIChatSession).where(
                    AIChatSession.id == session_id,
                    AIChatSession.archived == False,  # noqa: E712
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None

            msg_result = await session.execute(
                select(AIChatMessage)
                .where(AIChatMessage.session_id == session_id)
                .order_by(AIChatMessage.created_at.asc())
                .limit(max(1, message_limit))
            )
            messages = msg_result.scalars().all()

        out = self._session_to_dict(row)
        out["messages"] = [self._message_to_dict(m) for m in messages]
        return out

    async def get_recent_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(AIChatMessage)
                .where(AIChatMessage.session_id == session_id)
                .order_by(AIChatMessage.created_at.desc())
                .limit(max(1, limit))
            )
            rows = list(result.scalars().all())
        rows.reverse()
        return [self._message_to_dict(r) for r in rows]

    async def find_latest_for_context(
        self,
        context_type: Optional[str],
        context_id: Optional[str],
    ) -> Optional[dict]:
        if not context_type and not context_id:
            return None

        async with AsyncSessionLocal() as session:
            query = select(AIChatSession).where(AIChatSession.archived == False)  # noqa: E712
            if context_type:
                query = query.where(AIChatSession.context_type == context_type)
            if context_id:
                query = query.where(AIChatSession.context_id == context_id)
            query = query.order_by(desc(AIChatSession.updated_at)).limit(1)
            result = await session.execute(query)
            row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._session_to_dict(row)

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        model_used: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        msg = AIChatMessage(
            id=uuid.uuid4().hex[:16],
            session_id=session_id,
            role=role,
            content=content,
            model_used=model_used,
            input_tokens=max(0, int(input_tokens or 0)),
            output_tokens=max(0, int(output_tokens or 0)),
            created_at=utcnow(),
        )
        async with AsyncSessionLocal() as session:
            session.add(msg)
            result = await session.execute(
                select(AIChatSession).where(AIChatSession.id == session_id)
            )
            s = result.scalar_one_or_none()
            if s is not None:
                s.updated_at = utcnow()
                if not s.title and role == "user":
                    s.title = content.strip()[:120]
            await session.commit()

    async def archive_session(self, session_id: str) -> bool:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(AIChatSession).where(AIChatSession.id == session_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return False
            row.archived = True
            row.updated_at = utcnow()
            await session.commit()
        return True

    async def list_sessions(
        self,
        *,
        context_type: Optional[str] = None,
        context_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        async with AsyncSessionLocal() as session:
            query = select(AIChatSession).where(AIChatSession.archived == False)  # noqa: E712
            if context_type:
                query = query.where(AIChatSession.context_type == context_type)
            if context_id:
                query = query.where(AIChatSession.context_id == context_id)
            query = query.order_by(desc(AIChatSession.updated_at)).limit(max(1, limit))
            result = await session.execute(query)
            rows = result.scalars().all()
        return [self._session_to_dict(r) for r in rows]

    @staticmethod
    def _session_to_dict(row: AIChatSession) -> dict:
        return {
            "session_id": row.id,
            "context_type": row.context_type,
            "context_id": row.context_id,
            "title": row.title,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    @staticmethod
    def _message_to_dict(row: AIChatMessage) -> dict:
        return {
            "id": row.id,
            "session_id": row.session_id,
            "role": row.role,
            "content": row.content,
            "model_used": row.model_used,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }


chat_memory_service = ChatMemoryService()
