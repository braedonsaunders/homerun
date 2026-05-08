"""
Trading VPN/Proxy Service

Routes trading HTTP requests through a configurable proxy (SOCKS5, HTTP, HTTPS)
while leaving all scanning/data requests on the direct connection.

Settings are stored in the database (AppSettings table) and managed through the
Settings UI — no environment variables needed.

Supports:
  - SOCKS5 proxy: socks5://user:pass@host:port
  - HTTP proxy:   http://host:port
  - HTTPS proxy:  https://host:port

Usage:
  1. Configure proxy in Settings > Trading VPN/Proxy
  2. Call patch_clob_client_proxy() after ClobClient init to route trades through VPN
  3. Use get_trading_http_client() for any async trading HTTP calls
"""

import asyncio
import httpx
import threading
import time
from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger
from utils.secrets import decrypt_secret

logger = get_logger(__name__)

# Cached proxy-aware clients
_sync_proxy_client: Optional[httpx.Client] = None
_async_proxy_client: Optional[httpx.AsyncClient] = None
_sync_client_signature: Optional[tuple[bool, Optional[str], bool, float]] = None
_async_client_signature: Optional[tuple[bool, Optional[str], bool, float]] = None
_clob_patch_signature: Optional[tuple[bool, Optional[str], bool, float]] = None
_pre_trade_vpn_signature: Optional[tuple[bool, Optional[str], bool, float, bool]] = None
_pre_trade_vpn_cache_result: Optional[tuple[bool, str]] = None
_pre_trade_vpn_cache_until: float = 0.0
_PRE_TRADE_VPN_CACHE_TTL_SUCCESS_SECONDS = 120.0
_PRE_TRADE_VPN_CACHE_TTL_FAILURE_SECONDS = 10.0
_VPN_BACKGROUND_REFRESH_TASK: Optional[asyncio.Task] = None
_clob_patch_lock = threading.Lock()


@dataclass
class ProxyConfig:
    """Snapshot of proxy settings from the database."""

    enabled: bool = False
    proxy_url: Optional[str] = None
    verify_ssl: bool = True
    timeout: float = 5.0
    require_vpn: bool = True


# In-memory cache of the last-loaded config so synchronous code
# (e.g. patch_clob_client_proxy) doesn't need to await a DB read.
_cached_config: ProxyConfig = ProxyConfig()
_cached_config_loaded_at: float = 0.0
# TTL on the cached config.  ``_load_config_from_db`` is called on
# every order via ``_sync_trading_transport``; without a TTL each call
# is a fresh AppSettings SELECT under DB pool pressure (1.3-1.7s
# observed in soak as ``submit_sync_transport``).  Proxy settings
# rarely change, and the settings-update API path explicitly calls
# ``reload_proxy_settings()`` which forces a fresh load — so a 30s
# TTL is safe.  Operator changes propagate immediately via the
# explicit reload; the hot path serves from cache.
_PROXY_CONFIG_TTL_SECONDS: float = 30.0


async def _load_config_from_db(*, force: bool = False) -> ProxyConfig:
    """Load proxy settings from the AppSettings database table.

    With ``force=False`` (default), serves from the in-memory cache
    when the cache is younger than ``_PROXY_CONFIG_TTL_SECONDS``.
    Hot-path callers (``_sync_trading_transport`` per order) get the
    cached value; first call after process start or cache expiry hits
    the DB.

    Pass ``force=True`` from the settings-update API path so an
    operator's proxy-config change propagates to the hot path
    immediately.  ``reload_proxy_settings()`` does this.
    """
    global _cached_config, _cached_config_loaded_at
    if not force:
        now = time.monotonic()
        if now - _cached_config_loaded_at < _PROXY_CONFIG_TTL_SECONDS:
            return _cached_config

    try:
        from sqlalchemy import select
        from models.database import AsyncSessionLocal, AppSettings

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
            row = result.scalar_one_or_none()
            if row is None:
                _cached_config = ProxyConfig()
                _cached_config_loaded_at = time.monotonic()
                return _cached_config

            _cached_config = ProxyConfig(
                enabled=bool(row.trading_proxy_enabled),
                proxy_url=decrypt_secret(row.trading_proxy_url) or None,
                verify_ssl=row.trading_proxy_verify_ssl if row.trading_proxy_verify_ssl is not None else True,
                timeout=row.trading_proxy_timeout or 30.0,
                require_vpn=row.trading_proxy_require_vpn if row.trading_proxy_require_vpn is not None else True,
            )
            _cached_config_loaded_at = time.monotonic()
            return _cached_config
    except Exception as e:
        logger.error(f"Failed to load proxy config from DB: {e}")
        _cached_config = ProxyConfig()
        # Don't bump _cached_config_loaded_at on failure: we want the
        # next call to retry the DB read, not serve a fallback for 30s.
        return _cached_config


def _get_config() -> ProxyConfig:
    """Return the in-memory cached config (populated by _load_config_from_db)."""
    return _cached_config


def _get_proxy_url() -> Optional[str]:
    """Return the configured proxy URL if proxy is enabled, else None."""
    cfg = _get_config()
    if not cfg.enabled:
        return None
    url = cfg.proxy_url
    if not url:
        logger.warning("Trading proxy enabled but proxy_url is not set")
        return None
    return url


def _config_signature(cfg: ProxyConfig) -> tuple[bool, Optional[str], bool, float]:
    return (
        bool(cfg.enabled),
        (str(cfg.proxy_url).strip() if cfg.proxy_url else None),
        bool(cfg.verify_ssl),
        float(cfg.timeout or 30.0),
    )


def _vpn_signature(cfg: ProxyConfig) -> tuple[bool, Optional[str], bool, float, bool]:
    base = _config_signature(cfg)
    return (base[0], base[1], base[2], base[3], bool(cfg.require_vpn))


def get_sync_proxy_client() -> httpx.Client:
    """
    Get a synchronous httpx.Client configured with the trading proxy.

    Used to replace py-clob-client-v2's internal _http_client so that
    all order placement / cancellation goes through the VPN.
    """
    cfg = _get_config()
    signature = _config_signature(cfg)
    global _sync_proxy_client, _sync_client_signature
    existing_client = _sync_proxy_client
    if existing_client is not None and not existing_client.is_closed and _sync_client_signature == signature:
        return existing_client

    proxy_url = _get_proxy_url()
    # Extend httpx keepalive_expiry well beyond the observed worst-case
    # event-loop stall (10.45s in the 2026-05-07 soak). With the default
    # 5s expiry, every loop stall >5s reaped the warm CLOB connection
    # and forced the next post_order through cold TCP+TLS handshake —
    # observed as "Server disconnected" 555× in the soak. 60s gives
    # generous headroom against future stalls without keeping idle
    # connections forever.
    kwargs = {
        "http2": True,
        "timeout": cfg.timeout,
        "verify": cfg.verify_ssl,
        "limits": httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=60.0,
        ),
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url
        logger.info(
            "Created sync trading proxy client",
            proxy=_mask_proxy_url(proxy_url),
        )
    else:
        logger.info("Created sync trading client (no proxy)")

    replacement_client = httpx.Client(**kwargs)
    if existing_client is not None and not existing_client.is_closed and existing_client is not replacement_client:
        existing_client.close()
    _sync_proxy_client = replacement_client
    _sync_client_signature = signature
    return replacement_client


def get_async_proxy_client() -> httpx.AsyncClient:
    """
    Get an async httpx.AsyncClient configured with the trading proxy.

    Use this for any async HTTP calls that should go through the VPN
    (e.g., CLOB price checks during trade execution).
    """
    cfg = _get_config()
    signature = _config_signature(cfg)
    global _async_proxy_client, _async_client_signature
    existing_client = _async_proxy_client
    if existing_client is not None and not existing_client.is_closed and _async_client_signature == signature:
        return existing_client

    proxy_url = _get_proxy_url()
    # Same keepalive_expiry rationale as get_sync_proxy_client — keep
    # the warm CLOB connection alive across event-loop stalls.
    kwargs = {
        "timeout": cfg.timeout,
        "verify": cfg.verify_ssl,
        "limits": httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=60.0,
        ),
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url
        logger.info(
            "Created async trading proxy client",
            proxy=_mask_proxy_url(proxy_url),
        )
    else:
        logger.info("Created async trading client (no proxy)")

    replacement_client = httpx.AsyncClient(**kwargs)
    if existing_client is not None and not existing_client.is_closed and existing_client is not replacement_client:
        try:
            asyncio.create_task(existing_client.aclose())
        except Exception:
            pass
    _async_proxy_client = replacement_client
    _async_client_signature = signature
    return replacement_client


def patch_clob_client_proxy() -> bool:
    """
    Monkey-patch py-clob-client-v2's module-level HTTP client to use the trading proxy.

    py-clob-client-v2 uses a singleton `_http_client = httpx.Client(http2=True)` in
    `py_clob_client_v2.http_helpers.helpers` for ALL HTTP requests (order placement,
    cancellation, etc.). This function replaces it with a proxy-configured client.

    Returns True if patching succeeded, False otherwise.
    """
    cfg = _get_config()
    signature = _config_signature(cfg)
    proxy_url = _get_proxy_url()
    patching_proxy = bool(proxy_url)
    global _clob_patch_signature

    try:
        from py_clob_client_v2.http_helpers import helpers as clob_helpers
    except ImportError as exc:
        logger.error("Failed to import py-clob-client-v2 transport helpers", exc_info=exc)
        return False

    try:
        with _clob_patch_lock:
            existing = getattr(clob_helpers, "_http_client", None)
            existing_closed = bool(getattr(existing, "is_closed", False)) if existing is not None else True
            if _clob_patch_signature == signature and existing is not None and not existing_closed:
                return True

            replacement_client = get_sync_proxy_client()
            clob_helpers._http_client = replacement_client
            if existing is not None and existing is not replacement_client and not existing_closed:
                try:
                    existing.close()
                except Exception:
                    pass

            _clob_patch_signature = signature
            if patching_proxy:
                logger.info(
                    "Patched py-clob-client-v2 HTTP client with trading proxy",
                    proxy=_mask_proxy_url(proxy_url),
                )
            else:
                logger.info("Patched py-clob-client-v2 HTTP client with direct transport")
            return True
    except Exception as exc:
        logger.error("Failed to patch CLOB client proxy", exc_info=exc)
        return False


async def verify_vpn_active(cfg: Optional[ProxyConfig] = None) -> dict:
    """
    Check whether a trading proxy is configured.  No network probes — if the
    proxy URL is set and the proxy is enabled, trust the config.  A dead proxy
    will surface as a transport error on the actual order submission, which the
    retry logic already handles.
    """
    if cfg is None:
        cfg = await _load_config_from_db()

    result = {
        "proxy_enabled": cfg.enabled,
        "proxy_url": _mask_proxy_url(cfg.proxy_url) if cfg.proxy_url else None,
        "proxy_reachable": bool(cfg.proxy_url),
        "vpn_active": bool(cfg.proxy_url),
    }

    if not cfg.proxy_url:
        result["error"] = "No proxy URL configured"

    return result


async def pre_trade_vpn_check() -> tuple[bool, str]:
    """
    Pre-trade VPN verification gate.

    Returns (allowed, reason). If require_vpn is True
    and the proxy is enabled but unreachable, trades are blocked.

    Loads fresh settings from the DB.
    """
    cfg = await _load_config_from_db()

    if not cfg.enabled:
        return True, "Proxy not enabled, direct trading allowed"

    if not cfg.require_vpn:
        return True, "VPN verification not required"

    signature = _vpn_signature(cfg)
    now = time.monotonic()
    global _pre_trade_vpn_signature, _pre_trade_vpn_cache_result, _pre_trade_vpn_cache_until
    if (
        _pre_trade_vpn_cache_result is not None
        and _pre_trade_vpn_signature == signature
        and now < _pre_trade_vpn_cache_until
    ):
        return _pre_trade_vpn_cache_result

    status = await verify_vpn_active(cfg)
    if not status["proxy_reachable"]:
        result = (
            False,
            f"Trading proxy unreachable: {status.get('proxy_ip_error', 'unknown error')}",
        )
    elif not status["vpn_active"]:
        result = (False, "VPN not active: proxy IP matches direct IP")
    else:
        result = (True, f"VPN active, trading through {status.get('proxy_ip') or status.get('proxy_url') or 'configured proxy'}")

    ttl_seconds = (
        _PRE_TRADE_VPN_CACHE_TTL_SUCCESS_SECONDS if result[0] else _PRE_TRADE_VPN_CACHE_TTL_FAILURE_SECONDS
    )
    _pre_trade_vpn_signature = signature
    _pre_trade_vpn_cache_result = result
    _pre_trade_vpn_cache_until = now + ttl_seconds
    _schedule_vpn_background_refresh(ttl_seconds)
    return result


def _schedule_vpn_background_refresh(ttl_seconds: float) -> None:
    """Schedule a background task to refresh the VPN cache before it expires."""
    global _VPN_BACKGROUND_REFRESH_TASK
    if _VPN_BACKGROUND_REFRESH_TASK and not _VPN_BACKGROUND_REFRESH_TASK.done():
        _VPN_BACKGROUND_REFRESH_TASK.cancel()
    refresh_in = max(1.0, ttl_seconds - 10.0)
    _VPN_BACKGROUND_REFRESH_TASK = asyncio.create_task(
        _vpn_background_refresh(refresh_in), name="vpn-cache-refresh"
    )


async def _vpn_background_refresh(delay: float) -> None:
    """Wait, then silently refresh the VPN cache so trades always hit cache."""
    try:
        await asyncio.sleep(delay)
        await pre_trade_vpn_check()
    except asyncio.CancelledError:
        return
    except Exception:
        pass


async def reload_proxy_settings():
    """
    Reload proxy config from DB and recreate HTTP clients.

    Called by the settings API after a user updates proxy config.
    """
    await close()
    global _pre_trade_vpn_signature, _pre_trade_vpn_cache_result, _pre_trade_vpn_cache_until
    _pre_trade_vpn_signature = None
    _pre_trade_vpn_cache_result = None
    _pre_trade_vpn_cache_until = 0.0
    # ``force=True`` bypasses the TTL cache so an operator's settings
    # update propagates to the hot path (``_sync_trading_transport``)
    # immediately on the next order submission.
    await _load_config_from_db(force=True)
    cfg = _get_config()
    patched = patch_clob_client_proxy()
    if cfg.enabled and cfg.proxy_url:
        if patched:
            logger.info("Trading proxy reloaded from DB settings")
        else:
            logger.warning("Trading proxy enabled but py-clob transport patch failed")
    else:
        if patched:
            logger.info("Trading proxy disabled; py-clob transport restored to direct mode")
        else:
            logger.info("Trading proxy disabled or not configured after reload")


def _mask_proxy_url(url: Optional[str]) -> Optional[str]:
    """Mask credentials in a proxy URL for safe logging."""
    if not url:
        return None
    try:
        # Mask password in URLs like socks5://user:pass@host:port
        if "@" in url:
            scheme_and_creds, host_part = url.rsplit("@", 1)
            if ":" in scheme_and_creds:
                # Find the last : before @ which is the password separator
                scheme_part = scheme_and_creds.rsplit(":", 1)[0]
                return f"{scheme_part}:****@{host_part}"
        return url
    except Exception:
        return "****"


async def close():
    """Close proxy clients and free resources."""
    global _sync_proxy_client, _async_proxy_client
    global _sync_client_signature, _async_client_signature, _clob_patch_signature
    if _async_proxy_client and not _async_proxy_client.is_closed:
        await _async_proxy_client.aclose()
        _async_proxy_client = None
    if _sync_proxy_client and not _sync_proxy_client.is_closed:
        _sync_proxy_client.close()
        _sync_proxy_client = None
    _sync_client_signature = None
    _async_client_signature = None
    _clob_patch_signature = None
    logger.info("Trading proxy clients closed")
