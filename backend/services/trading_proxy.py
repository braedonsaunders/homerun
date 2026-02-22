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


@dataclass
class ProxyConfig:
    """Snapshot of proxy settings from the database."""

    enabled: bool = False
    proxy_url: Optional[str] = None
    verify_ssl: bool = True
    timeout: float = 30.0
    require_vpn: bool = True


# In-memory cache of the last-loaded config so synchronous code
# (e.g. patch_clob_client_proxy) doesn't need to await a DB read.
_cached_config: ProxyConfig = ProxyConfig()


async def _load_config_from_db() -> ProxyConfig:
    """Load proxy settings from the AppSettings database table."""
    global _cached_config
    try:
        from sqlalchemy import select
        from models.database import AsyncSessionLocal, AppSettings

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(AppSettings).where(AppSettings.id == "default"))
            row = result.scalar_one_or_none()
            if row is None:
                _cached_config = ProxyConfig()
                return _cached_config

            _cached_config = ProxyConfig(
                enabled=bool(row.trading_proxy_enabled),
                proxy_url=decrypt_secret(row.trading_proxy_url) or None,
                verify_ssl=row.trading_proxy_verify_ssl if row.trading_proxy_verify_ssl is not None else True,
                timeout=row.trading_proxy_timeout or 30.0,
                require_vpn=row.trading_proxy_require_vpn if row.trading_proxy_require_vpn is not None else True,
            )
            return _cached_config
    except Exception as e:
        logger.error(f"Failed to load proxy config from DB: {e}")
        _cached_config = ProxyConfig()
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


def get_sync_proxy_client() -> httpx.Client:
    """
    Get a synchronous httpx.Client configured with the trading proxy.

    Used to replace py-clob-client's internal _http_client so that
    all order placement / cancellation goes through the VPN.
    """
    cfg = _get_config()
    signature = _config_signature(cfg)
    global _sync_proxy_client, _sync_client_signature
    if _sync_proxy_client is not None and not _sync_proxy_client.is_closed:
        if _sync_client_signature == signature:
            return _sync_proxy_client
        _sync_proxy_client.close()
        _sync_proxy_client = None

    proxy_url = _get_proxy_url()
    kwargs = {
        "http2": True,
        "timeout": cfg.timeout,
        "verify": cfg.verify_ssl,
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url
        logger.info(
            "Created sync trading proxy client",
            proxy=_mask_proxy_url(proxy_url),
        )
    else:
        logger.info("Created sync trading client (no proxy)")

    _sync_proxy_client = httpx.Client(**kwargs)
    _sync_client_signature = signature
    return _sync_proxy_client


def get_async_proxy_client() -> httpx.AsyncClient:
    """
    Get an async httpx.AsyncClient configured with the trading proxy.

    Use this for any async HTTP calls that should go through the VPN
    (e.g., CLOB price checks during trade execution).
    """
    cfg = _get_config()
    signature = _config_signature(cfg)
    global _async_proxy_client, _async_client_signature
    if _async_proxy_client is not None and not _async_proxy_client.is_closed:
        if _async_client_signature == signature:
            return _async_proxy_client
        old_client = _async_proxy_client
        _async_proxy_client = None
        try:
            asyncio.create_task(old_client.aclose())
        except Exception:
            pass

    proxy_url = _get_proxy_url()
    kwargs = {
        "timeout": cfg.timeout,
        "verify": cfg.verify_ssl,
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url
        logger.info(
            "Created async trading proxy client",
            proxy=_mask_proxy_url(proxy_url),
        )
    else:
        logger.info("Created async trading client (no proxy)")

    _async_proxy_client = httpx.AsyncClient(**kwargs)
    _async_client_signature = signature
    return _async_proxy_client


def patch_clob_client_proxy() -> bool:
    """
    Monkey-patch py-clob-client's module-level HTTP client to use the trading proxy.

    py-clob-client uses a singleton `_http_client = httpx.Client(http2=True)` in
    `py_clob_client.http_helpers.helpers` for ALL HTTP requests (order placement,
    cancellation, etc.). This function replaces it with a proxy-configured client.

    Returns True if patching succeeded, False otherwise.
    """
    cfg = _get_config()
    signature = _config_signature(cfg)
    proxy_url = _get_proxy_url()
    patching_proxy = bool(proxy_url)
    global _clob_patch_signature

    try:
        from py_clob_client.http_helpers import helpers as clob_helpers

        existing = getattr(clob_helpers, "_http_client", None)
        existing_closed = bool(getattr(existing, "is_closed", False)) if existing is not None else True
        if _clob_patch_signature == signature and existing is not None and not existing_closed:
            return True

        if existing is not None:
            try:
                existing.close()
            except Exception:
                pass

        # Replace with configured transport (proxy when enabled, direct otherwise)
        clob_helpers._http_client = get_sync_proxy_client()
        _clob_patch_signature = signature
        if patching_proxy:
            logger.info(
                "Patched py-clob-client HTTP client with trading proxy",
                proxy=_mask_proxy_url(proxy_url),
            )
        else:
            logger.info("Patched py-clob-client HTTP client with direct transport")
        return True

    except ImportError:
        logger.warning("py-clob-client not installed, cannot patch HTTP client for proxy")
        return False
    except Exception as e:
        logger.error(f"Failed to patch CLOB client proxy: {e}")
        return False


async def verify_vpn_active() -> dict:
    """
    Verify the VPN proxy is active by checking the external IP through the proxy
    vs. the direct connection. Returns status dict.

    Loads fresh settings from the DB each time to pick up UI changes.
    """
    cfg = await _load_config_from_db()

    result = {
        "proxy_enabled": cfg.enabled,
        "proxy_url": _mask_proxy_url(cfg.proxy_url) if cfg.proxy_url else None,
        "proxy_reachable": False,
        "direct_ip": None,
        "proxy_ip": None,
        "vpn_active": False,
    }

    if not cfg.enabled or not cfg.proxy_url:
        result["vpn_active"] = False
        result["error"] = "Proxy not configured"
        return result

    ip_check_url = "https://api.ipify.org?format=json"

    # Get direct IP
    try:
        async with httpx.AsyncClient(timeout=10.0) as direct_client:
            resp = await direct_client.get(ip_check_url)
            result["direct_ip"] = resp.json().get("ip")
    except Exception as e:
        result["direct_ip_error"] = str(e)

    # Get proxy IP
    try:
        proxy_client = get_async_proxy_client()
        resp = await proxy_client.get(ip_check_url)
        proxy_ip = resp.json().get("ip")
        result["proxy_ip"] = proxy_ip
        result["proxy_reachable"] = True

        # VPN is active if proxy IP differs from direct IP
        if result["direct_ip"] and proxy_ip:
            result["vpn_active"] = result["direct_ip"] != proxy_ip
        elif proxy_ip:
            # Can't verify direct IP but proxy works
            result["vpn_active"] = True
    except Exception as e:
        result["proxy_ip_error"] = str(e)
        result["proxy_reachable"] = False

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

    status = await verify_vpn_active()

    if not status["proxy_reachable"]:
        return (
            False,
            f"Trading proxy unreachable: {status.get('proxy_ip_error', 'unknown error')}",
        )

    if not status["vpn_active"]:
        return False, "VPN not active: proxy IP matches direct IP"

    return True, f"VPN active, trading through {status['proxy_ip']}"


async def reload_proxy_settings():
    """
    Reload proxy config from DB and recreate HTTP clients.

    Called by the settings API after a user updates proxy config.
    """
    await close()
    await _load_config_from_db()
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
