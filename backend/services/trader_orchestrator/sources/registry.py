from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    "manual": SourceAdapter(
        key="manual",
        label="Manual Positions",
        description="Manually adopted live positions managed without new entries.",
        domains=["event_markets"],
        signal_types=["manual_position"],
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
    "traders": SourceAdapter(
        key="traders",
        label="Traders",
        description="Tracked/pool/individual/group trader activity signals.",
        domains=["event_markets"],
        signal_types=["traders_opportunity", "confluence", "copy_trade"],
    ),
}


def normalize_source_key(value: Any) -> str:
    key = str(value or "").strip().lower()
    return key


def list_source_adapters() -> list[SourceAdapter]:
    return sorted(_SOURCE_ADAPTERS.values(), key=lambda item: item.key)


def get_source_adapter(source_key: str) -> SourceAdapter | None:
    return _SOURCE_ADAPTERS.get(normalize_source_key(source_key))


def normalize_sources(raw_sources: list[str] | None) -> list[str]:
    if raw_sources is None:
        return [adapter.key for adapter in list_source_adapters()]
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_sources:
        key = normalize_source_key(raw)
        if not key or key in seen:
            continue
        if key not in _SOURCE_ADAPTERS:
            continue
        seen.add(key)
        out.append(key)
    return out


def list_source_keys() -> list[str]:
    return [adapter.key for adapter in list_source_adapters()]
