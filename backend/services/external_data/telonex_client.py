"""Async client for the Telonex API (v1).

Telonex (telonex.io) is a paid SaaS that delivers historical prediction-
market data (Polymarket) and crypto reference data (Binance) as daily
Apache Parquet files via a simple REST API.

API shape (verified against telonex.io/docs 2026-05-10):
  * Base URL:       https://api.telonex.io/v1
  * Auth:           ``Authorization: Bearer <api_key>`` (downloads only;
                    availability + datasets endpoints are public).
  * Endpoints:
      GET /availability/{exchange}                     (public)
      GET /downloads/{exchange}/{channel}/{date}       (auth required)
                                                       returns 302 → S3 URL
      GET /datasets/{exchange}/{dataset}               (public)
                                                       returns 302 → S3 URL
  * Identifier methods: ``asset_id`` | ``market_id`` + ``outcome`` |
    ``slug`` + ``outcome`` (or ``outcome_id``).  For Binance, slug is
    just the lowercase symbol (e.g. ``btcusdt``).
  * Channels (polymarket): trades, quotes, book_snapshot_5,
    book_snapshot_25, book_snapshot_full, onchain_fills.
    (binance): trades, quotes, book_snapshot_5, book_snapshot_25.
  * Free tier: 5 *total* downloads.  Plus tier: unlimited.
    The 403 response carries ``X-Downloads-Remaining``.

Downloads return a 302 to a pre-signed S3 URL that expires after 15
minutes.  The caller is expected to follow the redirect and stream the
parquet directly to disk (the S3 hop doesn't go through Telonex's API).
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


_DEFAULT_BASE_URL = "https://api.telonex.io/v1"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_MAX_DELAY = 15.0


# Channels per exchange (from the docs).  Polymarket supports the full
# set including on-chain fills; Binance is a subset (no full book, no
# on-chain).
TELONEX_CHANNELS: dict[str, tuple[str, ...]] = {
    "polymarket": (
        "trades",
        "quotes",
        "book_snapshot_5",
        "book_snapshot_25",
        "book_snapshot_full",
        "onchain_fills",
    ),
    "binance": (
        "trades",
        "quotes",
        "book_snapshot_5",
        "book_snapshot_25",
    ),
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TelonexError(RuntimeError):
    """Base for all telonex client errors."""


class TelonexNotConfiguredError(TelonexError):
    """Raised when no API key is configured (and the call needs one)."""


class TelonexAuthError(TelonexError):
    """401 / 403 from the API — bad key or download quota exhausted.

    For 403 the API includes ``X-Downloads-Remaining`` so callers can
    surface the quota state to the operator.  ``downloads_remaining`` is
    set when that header was present.
    """

    def __init__(self, msg: str, *, downloads_remaining: Optional[int] = None) -> None:
        super().__init__(msg)
        self.downloads_remaining = downloads_remaining


class TelonexNotFoundError(TelonexError):
    """404 — no data for the requested asset / date / identifier."""


class TelonexValidationError(TelonexError):
    """400 / 422 — caller-side input problem (bad slug, bad date, etc.)."""


class TelonexUpstreamError(TelonexError):
    """5xx after retries exhausted, or non-JSON / malformed response."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class TelonexClient:
    """Thin async REST wrapper.  Construct once, ``await close()`` when done.

    The api_key is optional: ``availability()`` and dataset URLs work
    without one (the docs explicitly call out that those endpoints don't
    require auth).  ``download_url()`` will raise
    :class:`TelonexNotConfiguredError` if you call it with no key set.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/") or _DEFAULT_BASE_URL
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._call_count = 0
        self._bytes_in = 0
        # Latest ``X-Downloads-Remaining`` seen in a 403 response — exposed
        # via ``stats()`` so the UI can show the operator their remaining
        # quota without making them eyeball the API headers.
        self._last_downloads_remaining: Optional[int] = None

    # ── lifecycle ──────────────────────────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {
                "Accept": "application/json",
                "User-Agent": "homerun-telonex/0.2",
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_seconds),
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except Exception:
                logger.debug("telonex client close failed", exc_info=True)

    async def __aenter__(self) -> "TelonexClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ── observability ──────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "api_calls": self._call_count,
            "bytes_downloaded": self._bytes_in,
            "last_downloads_remaining": self._last_downloads_remaining,
        }

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    # ── core request loop ─────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        follow_redirects: bool = True,
        require_auth: bool = False,
    ) -> httpx.Response:
        """Issue one request with retry handling.

        Returns the :class:`httpx.Response` so callers can inspect both
        the body and the response headers (the 403 case carries the
        ``X-Downloads-Remaining`` header we want to surface).

        Set ``follow_redirects=False`` for the downloads endpoint when
        you want the redirect *URL* without actually fetching the S3
        body (so the caller can stream it elsewhere).
        """
        if require_auth and not self._api_key:
            raise TelonexNotConfiguredError(
                "Telonex API key required — add it in Data Lab → Providers → Telonex."
            )

        client = self._get_client()
        attempt = 0
        last_exc: Optional[BaseException] = None

        clean_params: Optional[dict[str, Any]] = None
        if params is not None:
            clean_params = {k: v for k, v in params.items() if v is not None}
            if not clean_params:
                clean_params = None

        while attempt < self._max_retries:
            attempt += 1
            try:
                response = await client.request(
                    method,
                    path,
                    params=clean_params,
                    follow_redirects=follow_redirects,
                )
            except httpx.RequestError as exc:
                last_exc = exc
                delay = min(_MAX_DELAY, _BASE_DELAY * (2 ** (attempt - 1)))
                logger.warning(
                    "telonex %s %s network error, retry %d/%d in %.1fs: %s",
                    method, path, attempt, self._max_retries, delay, exc,
                )
                await asyncio.sleep(delay)
                continue

            self._call_count += 1
            self._bytes_in += len(response.content or b"")

            # 5xx → retry.  Everything else returns up to the caller so
            # specific status codes can be mapped to typed errors.
            if 500 <= response.status_code < 600:
                if attempt >= self._max_retries:
                    raise TelonexUpstreamError(
                        f"telonex {response.status_code} after {attempt} attempts on {path}"
                    )
                delay = min(_MAX_DELAY, _BASE_DELAY * (2 ** (attempt - 1)))
                await asyncio.sleep(delay)
                continue

            return response

        if last_exc is not None:
            raise TelonexUpstreamError(
                f"telonex network failure after {self._max_retries} attempts on {path}: {last_exc}"
            ) from last_exc
        raise TelonexUpstreamError(
            f"telonex exhausted retries on {path} with no recognizable response"
        )

    def _raise_for_error(self, response: httpx.Response, path: str) -> None:
        """Map an HTTP response to a typed Telonex error.

        Call this AFTER ``_request`` returns when you expect a JSON
        success body.  Skip it for the downloads endpoint when you want
        to handle the 302 yourself.
        """
        sc = response.status_code
        if 200 <= sc < 300:
            return
        # 403 — quota or permission.  Carry the remaining-count header
        # so the UI can show it.
        if sc == 403:
            remaining_raw = response.headers.get("X-Downloads-Remaining")
            remaining: Optional[int]
            try:
                remaining = int(remaining_raw) if remaining_raw is not None else None
            except (TypeError, ValueError):
                remaining = None
            if remaining is not None:
                self._last_downloads_remaining = remaining
            detail = _extract_detail(response)
            raise TelonexAuthError(
                f"telonex 403 on {path}: {detail or 'download limit exceeded or insufficient permissions'}",
                downloads_remaining=remaining,
            )
        if sc == 401:
            raise TelonexAuthError(
                f"telonex 401 on {path}: {_extract_detail(response) or 'invalid or missing API key'}"
            )
        if sc == 404:
            raise TelonexNotFoundError(
                f"telonex 404 on {path}: {_extract_detail(response) or 'no data for the requested identifier'}"
            )
        if sc in (400, 422):
            raise TelonexValidationError(
                f"telonex {sc} on {path}: {_extract_detail(response) or 'invalid request'}"
            )
        raise TelonexError(f"telonex {sc} on {path}: {_extract_detail(response)}")

    # ── endpoints ──────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Lightweight reachability + (optional) auth probe.

        Uses the *unauthenticated* ``/datasets/polymarket/markets``
        redirect endpoint so we never spend a download from the
        operator's quota.  Returns a status dict the UI renders as a
        coloured pill.

        We do NOT verify the API key here — Telonex has no documented
        free auth-check endpoint, and probing ``/downloads/...`` could
        burn a download.  Auth issues surface lazily on the first real
        ``download_url()`` call.
        """
        try:
            response = await self._request(
                "GET",
                "/datasets/polymarket/markets",
                follow_redirects=False,
            )
        except TelonexUpstreamError as exc:
            return {"ok": False, "error": str(exc)}
        # 302 to S3 is the expected success path.  200 would also be
        # fine if Telonex ever inlines the file.
        ok = response.status_code in (200, 301, 302, 303, 307, 308)
        return {
            "ok": bool(ok),
            "status_code": int(response.status_code),
            "elapsed_ms": int(response.elapsed.total_seconds() * 1000)
            if response.elapsed
            else None,
            "auth_verified": False,  # we didn't actually authenticate
        }

    async def availability(
        self,
        exchange: str,
        *,
        asset_id: Optional[str] = None,
        market_id: Optional[str] = None,
        slug: Optional[str] = None,
        outcome: Optional[str] = None,
        outcome_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """``GET /availability/{exchange}`` — public, no auth required.

        Returns the date range per channel for the resolved asset so
        the UI can populate the date-range picker and warn when a
        requested date isn't available.
        """
        ex = (exchange or "").strip().lower()
        if not ex:
            raise TelonexValidationError("exchange is required")
        params: dict[str, Any] = {}
        if asset_id:
            params["asset_id"] = asset_id
        if market_id:
            params["market_id"] = market_id
        if slug:
            params["slug"] = slug
        if outcome:
            params["outcome"] = outcome
        if outcome_id is not None:
            params["outcome_id"] = int(outcome_id)
        if not params:
            raise TelonexValidationError(
                "Provide one of: asset_id; market_id+outcome; slug+outcome (or outcome_id)"
            )
        path = f"/availability/{ex}"
        response = await self._request("GET", path, params=params)
        self._raise_for_error(response, path)
        try:
            return response.json()
        except ValueError as exc:
            raise TelonexUpstreamError(
                f"telonex returned non-JSON for {path}: {exc}"
            ) from exc

    async def download_url(
        self,
        exchange: str,
        channel: str,
        date: str,
        *,
        asset_id: Optional[str] = None,
        market_id: Optional[str] = None,
        slug: Optional[str] = None,
        outcome: Optional[str] = None,
        outcome_id: Optional[int] = None,
    ) -> str:
        """Resolve a presigned S3 download URL for one day of data.

        Counts against the operator's download quota.  The returned URL
        is valid for ~15 minutes — fetch it promptly.  We deliberately
        do NOT follow the redirect ourselves (the caller streams the
        parquet wherever it needs to land — to disk, into the importer,
        whatever).
        """
        ex = (exchange or "").strip().lower()
        ch = (channel or "").strip()
        if not ex or not ch or not date:
            raise TelonexValidationError("exchange, channel, and date are all required")
        params: dict[str, Any] = {}
        for k, v in (
            ("asset_id", asset_id),
            ("market_id", market_id),
            ("slug", slug),
            ("outcome", outcome),
            ("outcome_id", outcome_id),
        ):
            if v is not None:
                params[k] = v
        if not params:
            raise TelonexValidationError(
                "Provide one of: asset_id; market_id+outcome; slug+outcome (or outcome_id)"
            )

        path = f"/downloads/{ex}/{ch}/{date}"
        response = await self._request(
            "GET",
            path,
            params=params,
            follow_redirects=False,
            require_auth=True,
        )
        # Capture the remaining-count header when present (some 2xx
        # responses also carry it).
        rem_raw = response.headers.get("X-Downloads-Remaining")
        if rem_raw is not None:
            try:
                self._last_downloads_remaining = int(rem_raw)
            except (TypeError, ValueError):
                pass

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location") or response.headers.get("location")
            if not location:
                raise TelonexUpstreamError(
                    f"telonex {response.status_code} on {path} without Location header"
                )
            return location
        # Anything else → typed error.
        self._raise_for_error(response, path)
        # 200 with body would be unusual for this endpoint — surface it
        # as a sensible error since we can't synthesize a URL.
        raise TelonexUpstreamError(
            f"telonex {response.status_code} on {path}: expected 302 redirect"
        )

    def dataset_url(self, exchange: str, dataset: str) -> str:
        """Resolve a metadata dataset URL (markets / tags).

        These endpoints are public and produce 302 redirects to S3.
        Returning the API URL directly lets callers feed it to a
        pandas/polars/duckdb reader that follows redirects natively —
        same trick the Telonex Python SDK uses.
        """
        ex = (exchange or "").strip().lower()
        ds = (dataset or "").strip().lower()
        if not ex or not ds:
            raise TelonexValidationError("exchange and dataset are required")
        return f"{self._base_url}/datasets/{ex}/{ds}"

    async def stream_to_path(
        self,
        url: str,
        target_path: "os.PathLike[str] | str",
        *,
        chunk_size: int = 1 << 20,
        follow_redirects: bool = True,
    ) -> int:
        """Stream the bytes from a URL to disk.

        Writes to a sibling ``.tmp`` file and atomically renames on
        success — same pattern the parquet scanner expects, so the
        scanner never sees a half-written file.

        ``follow_redirects=True`` is the default because the public
        ``/datasets/...`` endpoints return a 302 → R2/S3 presigned URL
        and we want one call to do both hops.  Pass ``False`` only when
        the caller has already resolved the redirect (e.g. via
        :meth:`download_url`) and the presigned URL is the input.

        Returns the number of bytes written.  Raises
        :class:`TelonexUpstreamError` on a 4xx/5xx response after
        redirect resolution.
        """
        import os
        from pathlib import Path

        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")

        # Use a dedicated httpx client for the fetch — the base-URL'd
        # one points at telonex.io and the auth header isn't valid for
        # S3 / R2 (would actually break presigned URL signature
        # validation).
        total = 0
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, read=300.0),
            follow_redirects=follow_redirects,
        ) as raw:
            async with raw.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise TelonexUpstreamError(
                        f"telonex fetch failed {response.status_code} on {url[:120]}"
                    )
                # Guard against 3xx leaking through (when follow_redirects
                # is False but the URL still redirects).  Without this
                # the 302 body — which is empty — silently produces a
                # 0-byte file with status 200 from the caller's view.
                if 300 <= response.status_code < 400:
                    raise TelonexUpstreamError(
                        f"telonex fetch returned {response.status_code} with redirect not followed on {url[:120]}"
                    )
                with tmp.open("wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size):
                        f.write(chunk)
                        total += len(chunk)
        os.replace(tmp, target)
        return total

    async def download_to_path(
        self,
        exchange: str,
        channel: str,
        date: str,
        target_path: "os.PathLike[str] | str",
        *,
        asset_id: Optional[str] = None,
        market_id: Optional[str] = None,
        slug: Optional[str] = None,
        outcome: Optional[str] = None,
        outcome_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """Resolve the presigned URL then stream the parquet to disk.

        Returns a dict with the bytes written + last quota header value
        so callers can update operator-visible counters.  This is the
        only method on the client that spends one (1) download.
        """
        url = await self.download_url(
            exchange=exchange,
            channel=channel,
            date=date,
            asset_id=asset_id,
            market_id=market_id,
            slug=slug,
            outcome=outcome,
            outcome_id=outcome_id,
        )
        bytes_written = await self.stream_to_path(url, target_path)
        return {
            "url": url,
            "bytes": bytes_written,
            "downloads_remaining": self._last_downloads_remaining,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:
        return (response.text or "").strip()[:300]
    if isinstance(body, dict):
        # Telonex uses {"detail": "..."} for FastAPI-style errors and
        # {"error": "...", "message": "..."} for some 404s.
        return str(
            body.get("detail")
            or body.get("message")
            or body.get("error")
            or ""
        )[:300]
    return str(body)[:300]


# ---------------------------------------------------------------------------
# Settings-driven construction
# ---------------------------------------------------------------------------


async def build_client_from_settings(*, require_api_key: bool = True) -> TelonexClient:
    """Construct a configured client by reading AppSettings.

    Default behaviour matches the polybacktest client: raise
    :class:`TelonexNotConfiguredError` when no API key is set, so the
    routes layer can return 412 with a "configure first" UX.

    Pass ``require_api_key=False`` for the public endpoints
    (availability, datasets, markets / tags catalog) — those work
    without a key and we still want to use the same async client and
    base URL override.
    """
    from sqlalchemy import select
    from models.database import AppSettings, AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        row = (await session.execute(select(AppSettings))).scalar_one_or_none()
    api_key = _decrypt_or_passthrough(getattr(row, "telonex_api_key", None) if row else None)
    base_url = (
        (getattr(row, "telonex_base_url", None) or "").strip() if row else ""
    ) or _DEFAULT_BASE_URL
    if require_api_key and not api_key:
        raise TelonexNotConfiguredError(
            "Telonex API key is not configured.  Add it in Data Lab → Providers → Telonex."
        )
    return TelonexClient(api_key=api_key, base_url=base_url)


def _decrypt_or_passthrough(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        from utils.secrets import decrypt_secret

        decrypted = decrypt_secret(value)
        return decrypted if decrypted is not None else value
    except Exception:
        return value


def default_base_url() -> str:
    return _DEFAULT_BASE_URL


def supported_exchanges() -> tuple[str, ...]:
    return tuple(TELONEX_CHANNELS.keys())


def channels_for(exchange: str) -> tuple[str, ...]:
    return TELONEX_CHANNELS.get((exchange or "").strip().lower(), ())


__all__ = [
    "TelonexClient",
    "TelonexError",
    "TelonexNotConfiguredError",
    "TelonexAuthError",
    "TelonexNotFoundError",
    "TelonexValidationError",
    "TelonexUpstreamError",
    "build_client_from_settings",
    "default_base_url",
    "supported_exchanges",
    "channels_for",
    "TELONEX_CHANNELS",
]
