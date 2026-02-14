from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceAdapter:
    key: str
    label: str
    description: str
    domains: list[str] = field(default_factory=list)
    signal_types: list[str] = field(default_factory=list)


_SOURCE_ADAPTERS: dict[str, SourceAdapter] = {
    "scanner": SourceAdapter(
        key="scanner",
        label="General Opportunities",
        description="Scanner-originated arbitrage opportunities.",
        domains=["event_markets"],
        signal_types=["opportunity"],
    ),
    "crypto": SourceAdapter(
        key="crypto",
        label="Crypto Markets",
        description="Crypto microstructure and 5m/15m market signals.",
        domains=["crypto"],
        signal_types=["crypto_market"],
    ),
    "news": SourceAdapter(
        key="news",
        label="News Workflow",
        description="News-driven intents and event reactions.",
        domains=["event_markets"],
        signal_types=["news_intent"],
    ),
    "weather": SourceAdapter(
        key="weather",
        label="Weather Workflow",
        description="Weather forecast probability dislocations.",
        domains=["event_markets"],
        signal_types=["weather_intent"],
    ),
    "world_intelligence": SourceAdapter(
        key="world_intelligence",
        label="World Intelligence",
        description="Geopolitical conflict/tension opportunity signals.",
        domains=["event_markets"],
        signal_types=["world_intelligence"],
    ),
    "insider": SourceAdapter(
        key="insider",
        label="Insider Signals",
        description="Insider/tracked wallet behavior intents.",
        domains=["event_markets"],
        signal_types=["insider_intent"],
    ),
    "tracked_traders": SourceAdapter(
        key="tracked_traders",
        label="Tracked Traders",
        description="Signals synthesized from tracked trader activity.",
        domains=["event_markets"],
        signal_types=["tracked_trader"],
    ),
}

_SOURCE_ALIASES: dict[str, str] = {
    # Legacy UI/source keys mapped to canonical adapters.
    "pool_traders": "tracked_traders",
}


def _normalize_source_key(value: str) -> str:
    key = str(value or "").strip().lower()
    return _SOURCE_ALIASES.get(key, key)


def list_source_adapters() -> list[SourceAdapter]:
    return sorted(_SOURCE_ADAPTERS.values(), key=lambda item: item.key)


def list_source_aliases() -> dict[str, str]:
    return dict(_SOURCE_ALIASES)


def get_source_adapter(source_key: str) -> SourceAdapter | None:
    return _SOURCE_ADAPTERS.get(_normalize_source_key(source_key))


def normalize_sources(raw_sources: list[str] | None) -> list[str]:
    if raw_sources is None:
        return [adapter.key for adapter in list_source_adapters()]
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_sources:
        key = _normalize_source_key(str(raw or ""))
        if not key or key in seen:
            continue
        if key not in _SOURCE_ADAPTERS:
            continue
        seen.add(key)
        out.append(key)
    return out
