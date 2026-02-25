from api.routes_orchestrator_live import router as orchestrator_live_router
from api.routes_settings import router as settings_router
from services.live_execution_service import (
    LiveExecutionService,
    OrderSide,
    OrderStatus,
    OrderType,
    live_execution_service,
)


def _route_surface(router):
    surface = set()
    for route in router.routes:
        path = getattr(route, "path", "")
        methods = set(getattr(route, "methods", set()))
        for method in methods:
            method_key = str(method or "").upper()
            if method_key in {"HEAD", "OPTIONS"}:
                continue
            surface.add((method_key, path))
    return surface


def test_orchestrator_live_route_surface_is_complete():
    actual = _route_surface(orchestrator_live_router)
    expected = {
        ("GET", "/trader-orchestrator/live/status"),
        ("GET", "/trader-orchestrator/live/vpn-status"),
        ("POST", "/trader-orchestrator/live/initialize"),
        ("POST", "/trader-orchestrator/live/approve-allowance"),
        ("POST", "/trader-orchestrator/live/orders"),
        ("GET", "/trader-orchestrator/live/orders"),
        ("GET", "/trader-orchestrator/live/orders/open"),
        ("GET", "/trader-orchestrator/live/orders/{order_id}"),
        ("DELETE", "/trader-orchestrator/live/orders/{order_id}"),
        ("DELETE", "/trader-orchestrator/live/orders"),
        ("GET", "/trader-orchestrator/live/positions"),
        ("GET", "/trader-orchestrator/live/balance"),
        ("POST", "/trader-orchestrator/live/execute-opportunity"),
        ("POST", "/trader-orchestrator/live/emergency-stop"),
    }
    assert actual == expected


def test_settings_live_execution_route_surface_is_cut_over():
    actual = _route_surface(settings_router)
    assert ("GET", "/settings/live-execution") in actual
    assert ("PUT", "/settings/live-execution") in actual
    assert ("GET", "/settings/trading") not in actual
    assert ("PUT", "/settings/trading") not in actual


def test_live_execution_service_surface_is_complete():
    assert isinstance(live_execution_service, LiveExecutionService)
    assert OrderSide.BUY.value == "BUY"
    assert OrderType.GTC.value == "GTC"
    assert OrderStatus.PENDING.value == "pending"

    required_methods = {
        "initialize",
        "ensure_initialized",
        "is_ready",
        "get_stats",
        "place_order",
        "get_recent_orders",
        "get_open_orders",
        "get_order",
        "cancel_order",
        "cancel_all_orders",
        "sync_positions",
        "get_balance",
        "execute_opportunity",
        "get_execution_wallet_address",
        "get_order_snapshots_by_clob_ids",
        "prepare_sell_balance_allowance",
        "_resolve_polymarket_credentials",
        "_approve_clob_allowance",
        "_funder_for_signature_type",
        "_get_wallet_address",
    }

    missing = [name for name in sorted(required_methods) if not callable(getattr(LiveExecutionService, name, None))]
    assert not missing
