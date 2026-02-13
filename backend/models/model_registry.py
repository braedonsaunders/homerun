"""Central registry for SQLAlchemy models defined outside models.database."""


def register_all_models() -> None:
    """Import all modules that declare Base subclasses as a side effect."""
    from services.category_buffers import CategoryBufferLog  # noqa: F401
    from services.credential_manager import StoredCredential  # noqa: F401
    from services.depth_analyzer import DepthCheck  # noqa: F401
    from services.execution_tiers import TierAssignment  # noqa: F401
    from services.fill_monitor import FillEvent  # noqa: F401
    from services.latency_tracker import PipelineLatencyLog  # noqa: F401
    from services.live_market_detector import MarketLiveStatus  # noqa: F401
    from services.market_cache import CachedMarket, CachedUsername  # noqa: F401
    from services.price_chaser import OrderRetryLog  # noqa: F401
    from services.sport_classifier import SportTokenClassification  # noqa: F401
    from services.token_circuit_breaker import TokenTrip  # noqa: F401
    from services.wallet_ws_monitor import WalletMonitorEvent  # noqa: F401

    _ = (
        CategoryBufferLog,
        StoredCredential,
        DepthCheck,
        TierAssignment,
        FillEvent,
        PipelineLatencyLog,
        MarketLiveStatus,
        CachedMarket,
        CachedUsername,
        OrderRetryLog,
        SportTokenClassification,
        TokenTrip,
        WalletMonitorEvent,
    )
