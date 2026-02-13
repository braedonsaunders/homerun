from importlib import import_module

__all__ = [
    "polymarket_client",
    "PolymarketClient",
    "scanner",
    "ArbitrageScanner",
    "wallet_tracker",
    "WalletTracker",
    "smart_wallet_pool",
    "SmartWalletPoolService",
]

_LAZY_EXPORTS = {
    "polymarket_client": ("services.polymarket", "polymarket_client"),
    "PolymarketClient": ("services.polymarket", "PolymarketClient"),
    "scanner": ("services.scanner", "scanner"),
    "ArbitrageScanner": ("services.scanner", "ArbitrageScanner"),
    "wallet_tracker": ("services.wallet_tracker", "wallet_tracker"),
    "WalletTracker": ("services.wallet_tracker", "WalletTracker"),
    "smart_wallet_pool": ("services.smart_wallet_pool", "smart_wallet_pool"),
    "SmartWalletPoolService": ("services.smart_wallet_pool", "SmartWalletPoolService"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'services' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
