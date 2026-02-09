"""
Trading VPN/Proxy Service

Routes trading HTTP requests through a configurable proxy (SOCKS5, HTTP, HTTPS)
while leaving all scanning/data requests on the direct connection.

Supports:
  - SOCKS5 proxy: socks5://user:pass@host:port
  - HTTP proxy:   http://host:port
  - HTTPS proxy:  https://host:port

Usage:
  1. Set TRADING_PROXY_ENABLED=true and TRADING_PROXY_URL in .env
  2. Call patch_clob_client_proxy() after ClobClient init to route trades through VPN
  3. Use get_trading_http_client() for any async trading HTTP calls
"""

import httpx
from typing import Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Cached proxy-aware clients
_sync_proxy_client: Optional[httpx.Client] = None
_async_proxy_client: Optional[httpx.AsyncClient] = None


def _get_proxy_url() -> Optional[str]:
    """Return the configured proxy URL if proxy is enabled, else None."""
    if not settings.TRADING_PROXY_ENABLED:
        return None
    url = settings.TRADING_PROXY_URL
    if not url:
        logger.warning("TRADING_PROXY_ENABLED=true but TRADING_PROXY_URL is not set")
        return None
    return url


def get_sync_proxy_client() -> httpx.Client:
    """
    Get a synchronous httpx.Client configured with the trading proxy.

    Used to replace py-clob-client's internal _http_client so that
    all order placement / cancellation goes through the VPN.
    """
    global _sync_proxy_client
    if _sync_proxy_client is not None and not _sync_proxy_client.is_closed:
        return _sync_proxy_client

    proxy_url = _get_proxy_url()
    kwargs = {
        "http2": True,
        "timeout": settings.TRADING_PROXY_TIMEOUT,
        "verify": settings.TRADING_PROXY_VERIFY_SSL,
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
    return _sync_proxy_client


def get_async_proxy_client() -> httpx.AsyncClient:
    """
    Get an async httpx.AsyncClient configured with the trading proxy.

    Use this for any async HTTP calls that should go through the VPN
    (e.g., CLOB price checks during trade execution).
    """
    global _async_proxy_client
    if _async_proxy_client is not None and not _async_proxy_client.is_closed:
        return _async_proxy_client

    proxy_url = _get_proxy_url()
    kwargs = {
        "timeout": settings.TRADING_PROXY_TIMEOUT,
        "verify": settings.TRADING_PROXY_VERIFY_SSL,
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
    return _async_proxy_client


def patch_clob_client_proxy() -> bool:
    """
    Monkey-patch py-clob-client's module-level HTTP client to use the trading proxy.

    py-clob-client uses a singleton `_http_client = httpx.Client(http2=True)` in
    `py_clob_client.http_helpers.helpers` for ALL HTTP requests (order placement,
    cancellation, etc.). This function replaces it with a proxy-configured client.

    Returns True if patching succeeded, False otherwise.
    """
    if not settings.TRADING_PROXY_ENABLED:
        logger.debug("Trading proxy not enabled, skipping CLOB client patch")
        return False

    proxy_url = _get_proxy_url()
    if not proxy_url:
        return False

    try:
        from py_clob_client.http_helpers import helpers as clob_helpers

        # Close the existing client to free connections
        if hasattr(clob_helpers, "_http_client") and clob_helpers._http_client:
            try:
                clob_helpers._http_client.close()
            except Exception:
                pass

        # Replace with proxy-configured client
        clob_helpers._http_client = get_sync_proxy_client()
        logger.info(
            "Patched py-clob-client HTTP client with trading proxy",
            proxy=_mask_proxy_url(proxy_url),
        )
        return True

    except ImportError:
        logger.warning(
            "py-clob-client not installed, cannot patch HTTP client for proxy"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to patch CLOB client proxy: {e}")
        return False


async def verify_vpn_active() -> dict:
    """
    Verify the VPN proxy is active by checking the external IP through the proxy
    vs. the direct connection. Returns status dict.

    If TRADING_PROXY_REQUIRE_VPN is True and the proxy is enabled,
    this should be called before placing trades.
    """
    result = {
        "proxy_enabled": settings.TRADING_PROXY_ENABLED,
        "proxy_url": _mask_proxy_url(settings.TRADING_PROXY_URL) if settings.TRADING_PROXY_URL else None,
        "proxy_reachable": False,
        "direct_ip": None,
        "proxy_ip": None,
        "vpn_active": False,
    }

    if not settings.TRADING_PROXY_ENABLED or not settings.TRADING_PROXY_URL:
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

    Returns (allowed, reason). If TRADING_PROXY_REQUIRE_VPN is True
    and the proxy is enabled but unreachable, trades are blocked.
    """
    if not settings.TRADING_PROXY_ENABLED:
        return True, "Proxy not enabled, direct trading allowed"

    if not settings.TRADING_PROXY_REQUIRE_VPN:
        return True, "VPN verification not required"

    status = await verify_vpn_active()

    if not status["proxy_reachable"]:
        return False, f"Trading proxy unreachable: {status.get('proxy_ip_error', 'unknown error')}"

    if not status["vpn_active"]:
        return False, "VPN not active: proxy IP matches direct IP"

    return True, f"VPN active, trading through {status['proxy_ip']}"


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
    if _async_proxy_client and not _async_proxy_client.is_closed:
        await _async_proxy_client.aclose()
        _async_proxy_client = None
    if _sync_proxy_client and not _sync_proxy_client.is_closed:
        _sync_proxy_client.close()
        _sync_proxy_client = None
    logger.info("Trading proxy clients closed")
