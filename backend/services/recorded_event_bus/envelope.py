"""The canonical envelope every recorded observation flows through.

Goals:
  * One shape for ``polymarket.book.snapshot``, ``crypto.update.btc_eth``,
    ``news.gdelt.story``, ``wallet.trade``, etc.  No per-topic
    table layouts, no per-topic Python classes for the envelope.
    The *payload* varies; the wrapper does not.
  * Bitemporality is a first-class field, not optional.  Strategies
    backtested in ``observed_at`` order see history as it actually
    happened; backtests run in ``ingested_at`` order see history as
    the live system knew it (catches leakage).
  * Cheap to serialise to parquet and cheap to validate at publish
    time — the envelope walks the hot path so it must not allocate
    pydantic models per event.
  * Total ordering is well-defined: the ``(observed_at_us, sequence,
    topic, entity_id)`` tuple breaks every tie a reasonable consumer
    could care about.

The envelope is *not* a pydantic model on purpose.  pydantic adds
hundreds of microseconds per construction on the hot path; this
class is a frozen dataclass with explicit validation.  Strategies
that want a richer shape wrap the payload themselves at the
*subscriber* side (which is allowed to be slow — fan-out happens
once per event regardless).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


class EnvelopeValidationError(ValueError):
    """Raised at publish time when an envelope fails validation."""


# ── Topic-name discipline ────────────────────────────────────────────
#
# Topic names are structured, dotted, lowercase, ASCII.  Three-part
# minimum (``venue.kind.subkind``); five parts max.  This keeps the
# topic registry browsable and predictable.  Examples that VALIDATE:
#
#   polymarket.book.snapshot
#   polymarket.book.delta
#   polymarket.trade
#   crypto.update.btc_eth
#   news.gdelt.story
#   wallet.trade
#   external.telonex.book_snapshot_5
#
# Examples that DON'T validate (and shouldn't):
#
#   BOOK_DELTA            (no dots, uppercase)
#   polymarket-book       (hyphens)
#   polymarket.book.      (trailing dot)
#   ..foo                 (empty segments)
#
# The 5-part cap exists so we don't end up with ``foo.bar.baz.qux.
# quux.corge`` namespacing — at that depth you want a nested payload
# field, not a deeper topic.
_TOPIC_PART = re.compile(r"^[a-z][a-z0-9_]*$")
_TOPIC_MIN_PARTS = 2  # venue.kind minimum; subkind is recommended but optional
_TOPIC_MAX_PARTS = 5


def parse_topic(topic: str) -> tuple[str, ...]:
    """Validate + split a topic name into its dotted parts.

    Returns the tuple of parts on success.  Raises
    :class:`EnvelopeValidationError` with a precise message on
    anything that fails the discipline above — recorders use this
    at publish time, and the catalog uses it at registration time.
    """
    if not isinstance(topic, str):
        raise EnvelopeValidationError(
            f"topic must be str, got {type(topic).__name__}"
        )
    if not topic:
        raise EnvelopeValidationError("topic is empty")
    parts = tuple(topic.split("."))
    if not (_TOPIC_MIN_PARTS <= len(parts) <= _TOPIC_MAX_PARTS):
        raise EnvelopeValidationError(
            f"topic {topic!r} has {len(parts)} dotted parts; "
            f"must be {_TOPIC_MIN_PARTS}..{_TOPIC_MAX_PARTS}"
        )
    for i, p in enumerate(parts):
        if not _TOPIC_PART.match(p):
            raise EnvelopeValidationError(
                f"topic {topic!r} part #{i} {p!r} is invalid — "
                "must match [a-z][a-z0-9_]*"
            )
    return parts


# ── Envelope ────────────────────────────────────────────────────────


def _now_us() -> int:
    """UTC microseconds since epoch.  Used as the default for
    ``ingested_at_us`` when the caller doesn't supply one."""
    return int(datetime.now(timezone.utc).timestamp() * 1_000_000)


@dataclass(frozen=True, slots=True)
class RecordedEvent:
    """One observation in the system.

    Fields:
      topic           Dotted topic name (see :func:`parse_topic`).
      entity_id       The "key" the event is *about* — a token_id, an
                      asset_id, a wallet address, a story_id.  Stored
                      as a string so 256-bit Polymarket asset_ids
                      round-trip without precision loss.
      observed_at_us  Microseconds since epoch UTC, when the event
                      happened *in the world*.  This is the replay
                      key — the bus orders events by this field.
      ingested_at_us  Microseconds since epoch UTC, when we recorded
                      the event.  Bitemporal pair to observed_at_us.
                      Defaults to "now" if caller leaves it None.
      source          A short identifier for the producer (e.g.
                      ``polymarket_ws``, ``telonex_import``,
                      ``crypto_worker``, ``gdelt_api``).  Lets
                      consumers filter by origin when the same
                      topic has multiple emitters (e.g. an oracle
                      price coming from chainlink AND binance).
      sequence        Optional per-(topic, entity) monotonic counter
                      from the source — book ws sequence numbers,
                      blockchain block height, etc.  Used as a
                      tiebreaker when many events share the same
                      observed_at_us.  None when the source doesn't
                      provide one.
      schema_version  Integer.  Bumped when the payload shape
                      changes.  Replay knows how to up-convert older
                      versions — see :mod:`payload_upgrades`.
      payload         The topic-specific dict.  Validated against
                      the topic's registered schema at publish time
                      (cheap presence checks only — full shape
                      validation is opt-in via the catalog).

    Why a frozen dataclass instead of pydantic:
      Pydantic validates on every construction (~100µs per envelope).
      The book ingestor publishes 10k+ envelopes/sec at peak.  We
      cannot afford that here.  We get type safety from the dataclass
      declarations + explicit ``__post_init__`` validation; we get
      hash/eq for free; we save the latency.

    Why ``slots=True``:
      ~30% memory savings when the bus holds 5k envelopes in flight
      during a batched flush.  Adds up to a real number across the
      live trader's day.
    """

    topic: str
    entity_id: str
    observed_at_us: int
    payload: Mapping[str, Any]
    source: str = "unknown"
    sequence: Optional[int] = None
    ingested_at_us: int = field(default_factory=_now_us)
    schema_version: int = 1

    # ── Validation ─────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # Topic name — full discipline.  This is the most common
        # source of "I forgot to register the topic" bugs, so we
        # surface the failure precisely.
        parse_topic(self.topic)

        if not isinstance(self.entity_id, str) or not self.entity_id:
            raise EnvelopeValidationError(
                f"entity_id must be a non-empty string, got "
                f"{self.entity_id!r}"
            )
        if not isinstance(self.observed_at_us, int):
            raise EnvelopeValidationError(
                "observed_at_us must be int microseconds, got "
                f"{type(self.observed_at_us).__name__}"
            )
        if self.observed_at_us <= 0:
            raise EnvelopeValidationError(
                f"observed_at_us must be positive, got {self.observed_at_us}"
            )
        if not isinstance(self.ingested_at_us, int):
            raise EnvelopeValidationError(
                "ingested_at_us must be int microseconds, got "
                f"{type(self.ingested_at_us).__name__}"
            )
        if self.ingested_at_us <= 0:
            raise EnvelopeValidationError(
                f"ingested_at_us must be positive, got {self.ingested_at_us}"
            )
        # Knowledge time can never precede truth time by more than a
        # microsecond of clock skew.  If it does, the producer is
        # backdating ingested_at — a leakage hazard the bitemporal
        # design exists to catch.
        if self.ingested_at_us + 1_000_000 < self.observed_at_us:
            raise EnvelopeValidationError(
                f"ingested_at_us={self.ingested_at_us} is more than 1s "
                f"before observed_at_us={self.observed_at_us}; "
                "producer is backdating ingest time (leakage hazard)"
            )
        if not isinstance(self.source, str) or not self.source:
            raise EnvelopeValidationError(
                f"source must be a non-empty string, got {self.source!r}"
            )
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise EnvelopeValidationError(
                f"schema_version must be a positive int, got "
                f"{self.schema_version!r}"
            )
        if self.sequence is not None and not isinstance(self.sequence, int):
            raise EnvelopeValidationError(
                "sequence must be int or None, got "
                f"{type(self.sequence).__name__}"
            )
        if not isinstance(self.payload, Mapping):
            raise EnvelopeValidationError(
                f"payload must be a Mapping, got {type(self.payload).__name__}"
            )

    # ── Convenience constructors ───────────────────────────────────

    @classmethod
    def from_datetime(
        cls,
        *,
        topic: str,
        entity_id: str,
        observed_at: datetime,
        payload: Mapping[str, Any],
        source: str = "unknown",
        sequence: Optional[int] = None,
        ingested_at: Optional[datetime] = None,
        schema_version: int = 1,
    ) -> "RecordedEvent":
        """Construct from ``datetime`` instead of raw microseconds.

        Naive datetimes are interpreted as UTC (consistent with the
        rest of the codebase, where ``utcnow()`` returns naive UTC).
        """
        def _to_us(dt: datetime) -> int:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000)

        return cls(
            topic=topic,
            entity_id=entity_id,
            observed_at_us=_to_us(observed_at),
            payload=payload,
            source=source,
            sequence=sequence,
            ingested_at_us=_to_us(ingested_at) if ingested_at is not None else _now_us(),
            schema_version=schema_version,
        )

    # ── Total ordering ─────────────────────────────────────────────

    def order_key(self) -> tuple[int, int, str, str]:
        """The canonical sort key the bus uses for multi-topic
        replay merging.  Stable across runs; total over the envelope
        space."""
        # ``sequence`` is optional but, when present, beats topic/
        # entity ties because it's the source's own ordering.
        return (
            self.observed_at_us,
            self.sequence if self.sequence is not None else 0,
            self.topic,
            self.entity_id,
        )

    # ── Serialisation helpers ──────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Plain dict suitable for JSON serialisation or for the
        parquet writer's row-batch builder."""
        return {
            "topic": self.topic,
            "entity_id": self.entity_id,
            "observed_at_us": self.observed_at_us,
            "ingested_at_us": self.ingested_at_us,
            "source": self.source,
            "sequence": self.sequence,
            "schema_version": self.schema_version,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecordedEvent":
        return cls(
            topic=str(data["topic"]),
            entity_id=str(data["entity_id"]),
            observed_at_us=int(data["observed_at_us"]),
            payload=dict(data.get("payload") or {}),
            source=str(data.get("source") or "unknown"),
            sequence=(
                int(data["sequence"])
                if data.get("sequence") is not None else None
            ),
            ingested_at_us=int(
                data.get("ingested_at_us") or _now_us()
            ),
            schema_version=int(data.get("schema_version") or 1),
        )

    def with_payload(self, **updates: Any) -> "RecordedEvent":
        """Return a copy of this event with payload fields merged.
        Useful for payload-upgrade adapters on replay."""
        merged = dict(self.payload)
        merged.update(updates)
        return replace(self, payload=merged)
