"""Local append-only intent journal for the fast-tier order hot path.

Why this exists
---------------
The fast crypto submit path must answer one question durably *before* it
touches the venue: "have I already fired this signal?".  The venue gives
us no help — Polymarket's CLOB does not dedup by our ``metadata`` key
(see ``intent_journal`` discovery note in ``fast_idempotency``) and FAK
market orders carry no stable nonce, so the venue will happily accept a
duplicate submission after a crash-restart.  Something on our side must
record the intent durably ahead of the wire.

Previously that durable record was a synchronous Postgres commit of a
``placing`` skeleton ``TraderOrder`` row (``fast_submit`` pre-fix).  A
networked RDBMS commit on the order critical path is the wrong tool: even
with ``synchronous_commit=local`` it is a backend round-trip whose tail
spiked to multiple seconds under pool contention (see the soak captures
referenced in ``config.py``).  The institutional pattern is journal-first:
append the intent to a local append-only log with an ``fsync`` (tens of
microseconds on local NVMe), act, and treat the database as an
asynchronous downstream projection.

This module is that journal.  It is deliberately tiny, dependency-free,
and synchronous — an ``fsync`` is the whole point, so there is nothing to
``await``.

Durability model
----------------
* ``record_intent`` appends an ``intent`` record and ``fsync``s before
  returning.  This is the line that MUST be on stable storage before the
  CLOB call.
* ``record_result`` appends a ``result`` record after the wire.  Losing a
  result record is safe: replay simply re-reconciles the orphaned intent
  against the venue by its deterministic key.
* On startup ``load`` rebuilds the in-memory index and ``open_intents``
  returns every intent that never saw a matching result — the set the
  reconcile sweep must resolve against the venue.

Concurrency model
-----------------
One fast-worker process owns a disjoint set of traders (lane-based
ownership), and within that process the per-trader ``asyncio.Lock`` in
``fast_submit`` serialises intent recording for a given signal.  The
journal therefore only needs to protect its own append+index mutation
against interleaving, which a ``threading.Lock`` provides (``fsync``
drops the GIL).  Cross-process dedup for the *same* trader is out of
scope by design — two processes owning one trader is not a supported
topology.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Record kinds.
_KIND_INTENT = "intent"
_KIND_RESULT = "result"

# Terminal result statuses — an intent that reached one of these is
# fully resolved and need not be replayed against the venue.  ``recovered``
# marks an intent whose ownership was handed to a DB recovery row by the
# startup replay (the orphan-reconcile sweep then resolves it against the
# venue); it stays in the dedup index so the signal is never re-fired.
_TERMINAL_RESULT_STATUSES = frozenset(
    {"executed", "failed", "cancelled", "skipped", "recovered"}
)

# Compact the file once it grows past this many bytes.  Compaction
# rewrites the log keeping only still-open intents, so steady-state size
# tracks the number of concurrently in-flight orders (tiny).
_COMPACT_THRESHOLD_BYTES = 4 * 1024 * 1024


def _default_journal_dir() -> Path:
    """Resolve the journal directory.

    Infra path (not a user-facing trading knob): an env override for
    operators who want it on a specific fast disk, falling back to the
    repo's existing ``backend/.runtime`` convention.
    """
    override = os.environ.get("HOMERUN_INTENT_JOURNAL_DIR")
    if override:
        return Path(override)
    # ``.../backend/services/trader_orchestrator/intent_journal.py`` ->
    # ``.../backend/.runtime/intent_journal``
    backend_root = Path(__file__).resolve().parents[2]
    return backend_root / ".runtime" / "intent_journal"


class IntentJournal:
    """Append-only, fsync'd, single-process intent log with an in-memory index."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        # (trader_id, signal_id) -> "open" | "done"
        self._index: dict[tuple[str, str], str] = {}
        # (trader_id, signal_id) -> last intent record (for replay payload)
        self._open_records: dict[tuple[str, str], dict[str, Any]] = {}
        self._fd: Optional[int] = None
        self._bytes_written = 0
        self._opened = False

    # ----- lifecycle ------------------------------------------------------

    def open(self) -> None:
        """Open the journal for appending, creating parent dirs as needed."""
        with self._lock:
            if self._opened:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(
                self._path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                self._bytes_written = os.fstat(self._fd).st_size
            except OSError:
                self._bytes_written = 0
            self._opened = True

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
            self._opened = False

    # ----- load / replay --------------------------------------------------

    def load(self) -> None:
        """Rebuild the in-memory index from the on-disk log.

        Tolerates a torn final record (a crash mid-append leaves a partial
        last line); such a line is skipped.  Must be called before the
        first ``record_*`` so the dedup index reflects history.
        """
        with self._lock:
            self._index.clear()
            self._open_records.clear()
            if not self._path.exists():
                return
            try:
                raw = self._path.read_bytes()
            except OSError as exc:
                logger.error("Intent journal read failed", path=str(self._path), exc_info=exc)
                return
            for line in raw.split(b"\n"):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line.decode("utf-8"))
                except Exception:
                    # Torn/partial trailing record from a crash — ignore.
                    continue
                self._apply_record_locked(rec)

    def _apply_record_locked(self, rec: dict[str, Any]) -> None:
        kind = rec.get("k")
        trader_id = rec.get("tr")
        signal_id = rec.get("sig")
        if not trader_id or not signal_id:
            return
        key = (str(trader_id), str(signal_id))
        if kind == _KIND_INTENT:
            self._index[key] = "open"
            self._open_records[key] = rec
        elif kind == _KIND_RESULT:
            status = str(rec.get("st") or "").strip().lower()
            if status in _TERMINAL_RESULT_STATUSES:
                self._index[key] = "done"
                self._open_records.pop(key, None)
            else:
                # Non-terminal result (shouldn't normally happen) leaves
                # the intent open so replay still reconciles it.
                self._index.setdefault(key, "open")

    # ----- queries --------------------------------------------------------

    def has_intent(self, trader_id: str, signal_id: str) -> bool:
        """True if this (trader, signal) was ever recorded — open OR done.

        This is the dedup authority: a signal that has any journal record
        must not be re-fired.
        """
        return (str(trader_id), str(signal_id)) in self._index

    def is_open(self, trader_id: str, signal_id: str) -> bool:
        return self._index.get((str(trader_id), str(signal_id))) == "open"

    def open_intents(self) -> list[dict[str, Any]]:
        """Every intent that never reached a terminal result.

        These are the orphans the startup reconcile must resolve against
        the venue by their deterministic ``key``.
        """
        with self._lock:
            return list(self._open_records.values())

    def open_intent_count(self, trader_id: str) -> int:
        """Count of still-in-flight intents for a trader.

        Folded into the fast-tier ``max_open_orders`` check so an in-flight
        submission (intent recorded, venue result not yet back) counts
        against the cap — the role the committed ``placing`` skeleton row
        played before the journal cutover.
        """
        tid = str(trader_id)
        with self._lock:
            return sum(1 for (t, _s) in self._open_records if t == tid)

    # ----- writes ---------------------------------------------------------

    def record_intent(
        self,
        *,
        trader_id: str,
        signal_id: str,
        key: str,
        token_id: str | None = None,
        side: str | None = None,
        size_usd: float | None = None,
        market_id: str | None = None,
        mode: str | None = None,
    ) -> None:
        """Durably record submission intent BEFORE the venue call.

        Appends an ``intent`` record and ``fsync``s before returning.  This
        is the only call on the order critical path that touches stable
        storage, and it is the crash-recovery anchor.
        """
        rec = {
            "k": _KIND_INTENT,
            "tr": str(trader_id),
            "sig": str(signal_id),
            "key": str(key or ""),
            "tok": token_id,
            "sd": side,
            "usd": size_usd,
            "mkt": market_id,
            "md": mode,
            "ts": time.time(),
        }
        self._append(rec, fsync=True)

    def record_result(
        self,
        *,
        trader_id: str,
        signal_id: str,
        status: str,
        provider_clob_order_id: str | None = None,
        provider_order_id: str | None = None,
    ) -> None:
        """Record the venue outcome AFTER the wire.

        Loss of this record is safe — replay re-reconciles the intent — so
        we do not pay an ``fsync`` here; it rides the OS page cache and is
        flushed on the next intent or at compaction.
        """
        rec = {
            "k": _KIND_RESULT,
            "tr": str(trader_id),
            "sig": str(signal_id),
            "st": str(status or "").strip().lower(),
            "clob": provider_clob_order_id,
            "poid": provider_order_id,
            "ts": time.time(),
        }
        self._append(rec, fsync=False)

    def _append(self, rec: dict[str, Any], *, fsync: bool) -> None:
        line = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            if not self._opened or self._fd is None:
                # Lazily open so callers don't have to sequence open().
                self._lock.release()
                try:
                    self.open()
                finally:
                    self._lock.acquire()
            assert self._fd is not None
            os.write(self._fd, line)
            if fsync:
                os.fsync(self._fd)
            self._bytes_written += len(line)
            self._apply_record_locked(rec)
            should_compact = self._bytes_written >= _COMPACT_THRESHOLD_BYTES
        if should_compact:
            self._compact()

    # ----- compaction -----------------------------------------------------

    def _compact(self) -> None:
        """Rewrite the log keeping only still-open intents.

        Atomic via write-to-temp + ``os.replace``.  Bounds file growth so
        steady-state size tracks concurrent in-flight orders, not lifetime
        order count.
        """
        with self._lock:
            survivors = list(self._open_records.values())
            tmp_path = self._path.with_suffix(self._path.suffix + ".compact.tmp")
            try:
                tmp_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    buf = b"".join(
                        (json.dumps(r, separators=(",", ":")) + "\n").encode("utf-8")
                        for r in survivors
                    )
                    if buf:
                        os.write(tmp_fd, buf)
                    os.fsync(tmp_fd)
                finally:
                    os.close(tmp_fd)
                # Swap the new file in, then reopen the append fd on it.
                if self._fd is not None:
                    try:
                        os.close(self._fd)
                    except OSError:
                        pass
                    self._fd = None
                os.replace(tmp_path, self._path)
                self._fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                self._bytes_written = os.fstat(self._fd).st_size
                self._opened = True
            except OSError as exc:
                logger.error("Intent journal compaction failed", path=str(self._path), exc_info=exc)
                # Best-effort: ensure we still have an append fd.
                if self._fd is None:
                    try:
                        self._fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                        self._opened = True
                    except OSError:
                        self._opened = False
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass


# ----- process-wide singleton --------------------------------------------

_journal: Optional[IntentJournal] = None
_journal_lock = threading.Lock()


def get_intent_journal() -> IntentJournal:
    """Return the process-wide journal, opening + loading it on first use."""
    global _journal
    if _journal is not None:
        return _journal
    with _journal_lock:
        if _journal is None:
            j = IntentJournal(_default_journal_dir() / "fast_intent.log")
            j.open()
            j.load()
            _journal = j
    return _journal


def reset_intent_journal_for_tests(journal: Optional[IntentJournal]) -> None:
    """Swap the singleton (tests only)."""
    global _journal
    with _journal_lock:
        _journal = journal
