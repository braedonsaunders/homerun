"""Shared SQLAlchemy column types used across ORM models."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy.types import Numeric, TypeDecorator


class PreciseFloat(TypeDecorator):
    """Persist float-like values through Decimal-backed NUMERIC storage.

    This keeps API/service ergonomics (Python ``float`` in/out) while avoiding
    binary float serialization artifacts at the DB boundary.
    """

    impl = Numeric(24, 12, asdecimal=True)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect):
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for PreciseFloat: {value!r}") from exc

    def process_result_value(self, value: Any, dialect):
        if value is None:
            return None
        return float(value)
