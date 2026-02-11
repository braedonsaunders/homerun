"""Weather workflow orchestrator.

Independent from scanner/news/crypto pipelines. Runs in dedicated weather worker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from config import settings as app_settings
from models import ArbitrageOpportunity, StrategyType
from models.opportunity import ROIType
from services.polymarket import polymarket_client
from utils.logger import get_logger

from .adapters.base import WeatherForecastInput
from .adapters.open_meteo import OpenMeteoWeatherAdapter
from .contract_parser import parse_weather_contract
from .intent_builder import build_weather_intent
from .signal_engine import WeatherSignal, build_weather_signal
from . import shared_state

logger = get_logger("weather_workflow")


class WeatherWorkflowOrchestrator:
    """Runs weather discovery + signal generation + intent creation."""

    def __init__(self) -> None:
        self._adapter = OpenMeteoWeatherAdapter()
        self._cycle_count = 0
        self._last_run: Optional[datetime] = None
        self._last_opportunity_count = 0
        self._last_intent_count = 0

    async def run_cycle(self, session) -> dict[str, Any]:
        started = datetime.now(timezone.utc)
        settings = await shared_state.get_weather_settings(session)

        if not settings.get("enabled", True):
            await shared_state.write_weather_snapshot(
                session,
                opportunities=[],
                status={
                    "running": True,
                    "enabled": False,
                    "interval_seconds": settings.get("scan_interval_seconds", 14400),
                    "last_scan": started.isoformat(),
                    "current_activity": "Weather workflow disabled",
                },
                stats={
                    "markets_scanned": 0,
                    "contracts_parsed": 0,
                    "signals_generated": 0,
                    "intents_created": 0,
                },
            )
            return {"status": "disabled"}

        markets = await self._fetch_weather_markets(settings.get("max_markets_per_scan", 200))
        opportunities: list[ArbitrageOpportunity] = []
        intents_created = 0
        contracts_parsed = 0
        signals_generated = 0
        min_liquidity = float(settings.get("min_liquidity", 500.0))

        activity = f"Scanning {len(markets)} weather markets..."
        stats = {
            "markets_scanned": len(markets),
            "contracts_parsed": 0,
            "signals_generated": 0,
            "intents_created": 0,
        }
        await shared_state.write_weather_snapshot(
            session,
            opportunities=[],
            status={
                "running": True,
                "enabled": True,
                "interval_seconds": settings.get("scan_interval_seconds", 14400),
                "last_scan": None,
                "current_activity": activity,
            },
            stats=stats,
        )

        for market in markets:
            try:
                liquidity = float(getattr(market, "liquidity", 0.0) or 0.0)
                if liquidity < min_liquidity:
                    continue

                parsed = parse_weather_contract(market.question, market.end_date)
                if parsed is None:
                    continue
                contracts_parsed += 1

                fc_input = WeatherForecastInput(
                    location=parsed.location,
                    target_time=parsed.target_time,
                    metric=parsed.metric,
                    operator=parsed.operator,
                    threshold_c=parsed.threshold_c,
                )
                forecast = await self._adapter.forecast_probability(fc_input)

                signal = build_weather_signal(
                    market_id=market.condition_id or market.id,
                    yes_price=float(market.yes_price),
                    no_price=float(market.no_price),
                    forecast=forecast,
                    entry_max_price=float(settings.get("entry_max_price", 0.25)),
                    min_edge_percent=float(settings.get("min_edge_percent", 8.0)),
                    min_confidence=float(settings.get("min_confidence", 0.6)),
                    min_model_agreement=float(settings.get("min_model_agreement", 0.75)),
                )
                signals_generated += 1

                if not signal.should_trade:
                    continue

                opp = self._signal_to_opportunity(signal, market, parsed, forecast, settings)
                opportunities.append(opp)

                intent = build_weather_intent(
                    signal=signal,
                    market_id=market.condition_id or market.id,
                    market_question=market.question,
                    settings=settings,
                    metadata={
                        "weather": {
                            "location": parsed.location,
                            "metric": parsed.metric,
                            "operator": parsed.operator,
                            "threshold_c": parsed.threshold_c,
                            "raw_threshold": parsed.raw_threshold,
                            "raw_unit": parsed.raw_unit,
                            "target_time": parsed.target_time.isoformat(),
                            "gfs_value": forecast.gfs_value,
                            "ecmwf_value": forecast.ecmwf_value,
                        },
                        "market": {
                            "id": market.id,
                            "condition_id": market.condition_id,
                            "slug": market.slug,
                            "event_slug": market.event_slug,
                            "liquidity": market.liquidity,
                            "clob_token_ids": market.clob_token_ids,
                            "volume": market.volume,
                        },
                    },
                )
                await shared_state.upsert_weather_intent(session, intent)
                intents_created += 1
            except Exception as exc:
                logger.debug("Weather market skipped", market_id=market.id, error=str(exc))
                continue

        await session.commit()

        self._cycle_count += 1
        self._last_run = datetime.now(timezone.utc)
        self._last_opportunity_count = len(opportunities)
        self._last_intent_count = intents_created

        final_stats = {
            "markets_scanned": len(markets),
            "contracts_parsed": contracts_parsed,
            "signals_generated": signals_generated,
            "intents_created": intents_created,
            "cycle_count": self._cycle_count,
            "last_elapsed_seconds": round(
                (datetime.now(timezone.utc) - started).total_seconds(), 2
            ),
        }

        await shared_state.write_weather_snapshot(
            session,
            opportunities=sorted(opportunities, key=lambda o: o.roi_percent, reverse=True),
            status={
                "running": True,
                "enabled": True,
                "interval_seconds": settings.get("scan_interval_seconds", 14400),
                "last_scan": datetime.now(timezone.utc).isoformat(),
                "current_activity": (
                    f"Weather scan complete: {len(opportunities)} opportunities, "
                    f"{intents_created} intents"
                ),
            },
            stats=final_stats,
        )

        return {
            "status": "completed",
            "markets": len(markets),
            "contracts_parsed": contracts_parsed,
            "opportunities": len(opportunities),
            "intents": intents_created,
            "stats": final_stats,
        }

    async def _fetch_weather_markets(self, limit: int) -> list:
        events = await polymarket_client.get_all_events(closed=False)
        weather_events = [
            e for e in events if (e.category or "").strip().lower() == "weather"
        ]

        markets: list = []
        seen: set[str] = set()
        for ev in weather_events:
            for m in ev.markets:
                if m.closed or not m.active:
                    continue
                cid = m.condition_id or m.id
                if cid in seen:
                    continue
                seen.add(cid)
                # Ensure event_slug/category stay attached even when event payload is sparse.
                if not m.event_slug:
                    m.event_slug = ev.slug
                markets.append(m)
                if len(markets) >= limit:
                    return markets

        if markets:
            return markets

        # Fallback path when category metadata is sparse.
        searched = await polymarket_client.search_markets("weather", limit=min(limit, 100))
        for m in searched:
            if not m.active or m.closed:
                continue
            cid = m.condition_id or m.id
            if cid in seen:
                continue
            seen.add(cid)
            markets.append(m)
            if len(markets) >= limit:
                break
        return markets

    def _signal_to_opportunity(
        self,
        signal: WeatherSignal,
        market,
        parsed,
        forecast,
        settings: dict[str, Any],
    ) -> ArbitrageOpportunity:
        outcome = "YES" if signal.direction == "buy_yes" else "NO"
        token_id = None
        if market.clob_token_ids and len(market.clob_token_ids) >= 2:
            token_id = market.clob_token_ids[0] if outcome == "YES" else market.clob_token_ids[1]

        total_cost = signal.market_price
        expected_payout = signal.model_probability
        gross_profit = expected_payout - total_cost
        fee = expected_payout * app_settings.POLYMARKET_FEE
        net_profit = gross_profit - fee
        roi_percent = (net_profit / total_cost) * 100 if total_cost > 0 else 0.0

        now = datetime.now(timezone.utc)
        title = f"Weather Edge: {market.question[:110]}"
        description = (
            f"{signal.direction.replace('_', ' ').upper()} @ ${signal.market_price:.2f} | "
            f"Edge {signal.edge_percent:.2f}% | "
            f"GFS {signal.gfs_probability:.0%} / ECMWF {signal.ecmwf_probability:.0%}"
        )

        market_payload = {
            "id": market.condition_id or market.id,
            "question": market.question,
            "slug": market.slug,
            "event_slug": market.event_slug,
            "platform": "polymarket",
            "yes_price": market.yes_price,
            "no_price": market.no_price,
            "liquidity": market.liquidity,
            "volume": market.volume,
            "weather": {
                "location": parsed.location,
                "metric": parsed.metric,
                "operator": parsed.operator,
                "threshold_c": parsed.threshold_c,
                "raw_threshold": parsed.raw_threshold,
                "raw_unit": parsed.raw_unit,
                "target_time": parsed.target_time.isoformat(),
                "gfs_probability": signal.gfs_probability,
                "ecmwf_probability": signal.ecmwf_probability,
                "gfs_value": forecast.gfs_value,
                "ecmwf_value": forecast.ecmwf_value,
                "agreement": signal.model_agreement,
            },
        }

        max_position_size = min(
            float(settings.get("max_size_usd", 50.0)),
            max(float(market.liquidity) * 0.05, float(settings.get("default_size_usd", 10.0))),
        )

        return ArbitrageOpportunity(
            strategy=StrategyType.WEATHER_EDGE,
            title=title,
            description=description,
            total_cost=total_cost,
            expected_payout=expected_payout,
            gross_profit=gross_profit,
            fee=fee,
            net_profit=net_profit,
            roi_percent=roi_percent,
            is_guaranteed=False,
            roi_type=ROIType.DIRECTIONAL_PAYOUT.value,
            risk_score=max(0.0, min(1.0, 1.0 - signal.confidence)),
            risk_factors=[
                "Directional weather forecast edge",
                f"Model agreement: {signal.model_agreement:.2f}",
            ],
            markets=[market_payload],
            event_slug=market.event_slug,
            event_title=market.question,
            category="weather",
            min_liquidity=float(market.liquidity),
            max_position_size=max_position_size,
            detected_at=now,
            resolution_date=market.end_date,
            positions_to_take=[
                {
                    "action": "BUY",
                    "outcome": outcome,
                    "market": market.question,
                    "price": signal.market_price,
                    "token_id": token_id,
                    "market_id": market.condition_id or market.id,
                    "platform": "polymarket",
                }
            ],
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "cycle_count": self._cycle_count,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_opportunities": self._last_opportunity_count,
            "last_intents": self._last_intent_count,
        }


weather_workflow_orchestrator = WeatherWorkflowOrchestrator()
