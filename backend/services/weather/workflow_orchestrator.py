"""Weather workflow orchestrator.

Independent from scanner/news/crypto pipelines. Runs in dedicated weather worker.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
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
MIN_MINUTES_TO_RESOLUTION = 90


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
        self._adapter.clear_cycle_cache()

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

        configured_limit = int(settings.get("max_markets_per_scan", 200) or 200)
        market_limit = max(10, min(500, configured_limit))
        markets = await self._fetch_weather_markets(market_limit)
        opportunities: list[ArbitrageOpportunity] = []
        report_only_findings: list[ArbitrageOpportunity] = []
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
        existing_opportunities, _ = await shared_state.read_weather_snapshot(session)
        await shared_state.write_weather_snapshot(
            session,
            opportunities=existing_opportunities,
            status={
                "running": True,
                "enabled": True,
                "interval_seconds": settings.get("scan_interval_seconds", 14400),
                "last_scan": None,
                "current_activity": activity,
            },
            stats=stats,
        )

        concurrency = max(
            1,
            min(16, int(settings.get("evaluation_concurrency", 8) or 8)),
        )
        sem = asyncio.Semaphore(concurrency)

        async def _bounded_evaluate(market_obj):
            async with sem:
                return await self._evaluate_market(
                    market_obj, settings=settings, min_liquidity=min_liquidity
                )

        evaluations = await asyncio.gather(
            *[_bounded_evaluate(m) for m in markets],
            return_exceptions=True,
        )

        for result in evaluations:
            if isinstance(result, Exception):
                logger.debug("Weather market evaluation failed", error=str(result))
                continue

            contracts_parsed += int(result.get("contracts_parsed", 0) or 0)
            signals_generated += int(result.get("signals_generated", 0) or 0)
            opp = result.get("opportunity")
            intent = result.get("intent")
            if opp is not None:
                if self._is_executable_opportunity(opp):
                    opportunities.append(opp)
                else:
                    report_only_findings.append(opp)
            if intent is not None:
                await shared_state.upsert_weather_intent(session, intent)
                intents_created += 1

        await session.commit()
        await self._attach_market_price_history(opportunities)

        self._cycle_count += 1
        self._last_run = datetime.now(timezone.utc)
        self._last_opportunity_count = len(opportunities)
        self._last_intent_count = intents_created

        final_stats = {
            "markets_scanned": len(markets),
            "contracts_parsed": contracts_parsed,
            "signals_generated": signals_generated,
            "intents_created": intents_created,
            "report_only_findings": len(report_only_findings),
            "report_only_top_reasons": self._summarize_report_only_findings(
                report_only_findings
            ),
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
                    f"{len(report_only_findings)} findings, {intents_created} intents"
                ),
            },
            stats=final_stats,
        )

        return {
            "status": "completed",
            "markets": len(markets),
            "contracts_parsed": contracts_parsed,
            "opportunities": len(opportunities),
            "findings": len(report_only_findings),
            "intents": intents_created,
            "stats": final_stats,
        }

    async def _evaluate_market(
        self,
        market,
        *,
        settings: dict[str, Any],
        min_liquidity: float,
    ) -> dict[str, Any]:
        """Evaluate one weather market and return optional opportunity + intent."""
        now = datetime.now(timezone.utc)
        liquidity = float(getattr(market, "liquidity", 0.0) or 0.0)
        if liquidity < min_liquidity:
            return {"contracts_parsed": 0, "signals_generated": 0, "opportunity": None, "intent": None}
        if not self._is_market_candidate_tradable(market):
            return {"contracts_parsed": 0, "signals_generated": 0, "opportunity": None, "intent": None}

        resolution_dt = getattr(market, "end_date", None)
        if isinstance(resolution_dt, str):
            try:
                resolution_dt = datetime.fromisoformat(
                    str(resolution_dt).replace("Z", "+00:00")
                )
            except Exception:
                resolution_dt = None
        if isinstance(resolution_dt, datetime):
            if resolution_dt.tzinfo is None:
                resolution_dt = resolution_dt.replace(tzinfo=timezone.utc)
            else:
                resolution_dt = resolution_dt.astimezone(timezone.utc)
            if resolution_dt <= now + timedelta(minutes=MIN_MINUTES_TO_RESOLUTION):
                return {"contracts_parsed": 0, "signals_generated": 0, "opportunity": None, "intent": None}

        parsed = parse_weather_contract(
            market.question,
            market.end_date,
            getattr(market, "group_item_title", None),
        )
        if parsed is None:
            return {"contracts_parsed": 0, "signals_generated": 0, "opportunity": None, "intent": None}

        fc_input = WeatherForecastInput(
            location=parsed.location,
            target_time=parsed.target_time,
            metric=parsed.metric,
            operator=parsed.operator,
            threshold_c=parsed.threshold_c,
            threshold_c_low=parsed.threshold_c_low,
            threshold_c_high=parsed.threshold_c_high,
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
            operator=parsed.operator,
            threshold_c=parsed.threshold_c,
            threshold_c_low=parsed.threshold_c_low,
            threshold_c_high=parsed.threshold_c_high,
        )
        opp = self._signal_to_opportunity(signal, market, parsed, forecast, settings)
        if not signal.should_trade:
            reasons = signal.reasons or ["filtered by weather thresholds"]
            opp.description = f"REPORT ONLY | {opp.description} | {'; '.join(reasons)}"
            opp.max_position_size = 0.0
            existing_risks = list(opp.risk_factors or [])
            opp.risk_factors = existing_risks + [
                "Report only: does not meet trade thresholds",
                *reasons,
            ]
            return {"contracts_parsed": 1, "signals_generated": 1, "opportunity": opp, "intent": None}

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
                    "threshold_c_low": parsed.threshold_c_low,
                    "threshold_c_high": parsed.threshold_c_high,
                    "raw_threshold": parsed.raw_threshold,
                    "raw_threshold_low": parsed.raw_threshold_low,
                    "raw_threshold_high": parsed.raw_threshold_high,
                    "raw_unit": parsed.raw_unit,
                    "target_time": parsed.target_time.isoformat(),
                    "gfs_value": forecast.gfs_value,
                    "ecmwf_value": forecast.ecmwf_value,
                    "consensus_probability": forecast.consensus_probability,
                    "consensus_value_c": signal.consensus_temperature_c,
                    "consensus_value_f": self._c_to_f(signal.consensus_temperature_c),
                    "market_implied_temp_c": signal.market_implied_temperature_c,
                    "market_implied_temp_f": self._c_to_f(signal.market_implied_temperature_c),
                    "source_count": signal.source_count,
                    "source_spread_c": signal.source_spread_c,
                    "source_spread_f": (
                        (signal.source_spread_c * 9.0 / 5.0)
                        if signal.source_spread_c is not None
                        else None
                    ),
                    "forecast_sources": self._build_forecast_sources_payload(forecast),
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
        return {"contracts_parsed": 1, "signals_generated": 1, "opportunity": opp, "intent": intent}

    async def _fetch_weather_markets(self, limit: int) -> list:
        """Fetch markets that are parseable by the weather contract parser.

        We intentionally avoid relying only on event category labels because
        Polymarket tagging is often sparse or inconsistent for weather pages.
        """
        markets: list = []
        stale_markets: list = []
        seen: set[str] = set()
        scanned_events = 0
        offset = 0
        page_size = 100
        # Weather pages can be deep in the event feed; stop only after a wider crawl.
        max_events_offset = 15000
        now = datetime.now(timezone.utc)

        while offset < max_events_offset and len(markets) < limit:
            events = await polymarket_client.get_events(
                closed=False, limit=page_size, offset=offset
            )
            if not events:
                break
            scanned_events += len(events)
            offset += page_size

            for ev in events:
                for m in ev.markets:
                    if not self._is_market_candidate_tradable(m):
                        continue
                    cid = m.condition_id or m.id
                    if cid in seen:
                        continue
                    seen.add(cid)

                    if not m.event_slug:
                        m.event_slug = ev.slug

                    parsed = parse_weather_contract(
                        m.question,
                        m.end_date,
                        getattr(m, "group_item_title", None),
                    )
                    if parsed is None:
                        continue

                    end_dt = m.end_date
                    if isinstance(end_dt, str):
                        try:
                            end_dt = datetime.fromisoformat(
                                str(end_dt).replace("Z", "+00:00")
                            )
                        except Exception:
                            end_dt = None
                    if isinstance(end_dt, datetime):
                        if end_dt.tzinfo is None:
                            end_dt = end_dt.replace(tzinfo=timezone.utc)
                        else:
                            end_dt = end_dt.astimezone(timezone.utc)
                    if isinstance(end_dt, datetime) and end_dt <= now:
                        stale_markets.append(m)
                    else:
                        markets.append(m)
                    if len(markets) >= limit:
                        break
                if len(markets) >= limit:
                    break

        if markets:
            logger.info(
                "Weather market discovery complete",
                scanned_events=scanned_events,
                parseable_markets=len(markets),
                stale_markets=len(stale_markets),
            )
            return markets

        # Fallback path when event payloads are sparse for weather terms.
        fallback_queries = [
            "highest temperature in",
            "lowest temperature in",
            "will it rain in",
            "weather",
        ]
        for query in fallback_queries:
            searched = await polymarket_client.search_markets(
                query, limit=min(max(limit * 2, 50), 300)
            )
            for m in searched:
                if not self._is_market_candidate_tradable(m):
                    continue
                cid = m.condition_id or m.id
                if cid in seen:
                    continue
                seen.add(cid)
                parsed = parse_weather_contract(
                    m.question,
                    m.end_date,
                    getattr(m, "group_item_title", None),
                )
                if parsed is None:
                    continue

                end_dt = m.end_date
                if isinstance(end_dt, str):
                    try:
                        end_dt = datetime.fromisoformat(
                            str(end_dt).replace("Z", "+00:00")
                        )
                    except Exception:
                        end_dt = None
                if isinstance(end_dt, datetime):
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=timezone.utc)
                    else:
                        end_dt = end_dt.astimezone(timezone.utc)
                if isinstance(end_dt, datetime) and end_dt <= now:
                    stale_markets.append(m)
                else:
                    markets.append(m)

                if len(markets) >= limit:
                    break
            if len(markets) >= limit:
                break

        if markets:
            logger.info(
                "Weather market discovery complete",
                scanned_events=scanned_events,
                parseable_markets=len(markets),
                stale_markets=len(stale_markets),
                used_fallback=True,
            )
            return markets

        # Last resort: return stale markets so UI can still render weather coverage
        # when the feed has not rolled to fresh contracts yet.
        if stale_markets:
            logger.info(
                "Weather market discovery returned stale-only set",
                scanned_events=scanned_events,
                stale_markets=len(stale_markets),
                used_fallback=True,
            )
            return stale_markets[:limit]

        logger.info(
            "Weather market discovery complete",
            scanned_events=scanned_events,
            parseable_markets=0,
            used_fallback=True,
        )
        return markets

    @staticmethod
    def _market_tradability_payload(market) -> dict[str, Any]:
        return {
            "active": getattr(market, "active", None),
            "closed": getattr(market, "closed", None),
            "archived": getattr(market, "archived", None),
            "accepting_orders": getattr(market, "accepting_orders", None),
            "enable_order_book": getattr(market, "enable_order_book", None),
            "resolved": getattr(market, "resolved", None),
            "end_date": getattr(market, "end_date", None),
            "winner": getattr(market, "winner", None),
            "winning_outcome": getattr(market, "winning_outcome", None),
            "status": getattr(market, "status", None),
        }

    def _is_market_candidate_tradable(self, market) -> bool:
        """Fast local tradability guard using market metadata already in feed rows."""
        info = self._market_tradability_payload(market)
        try:
            return bool(polymarket_client.is_market_tradable(info))
        except Exception:
            return bool(getattr(market, "active", True)) and not bool(
                getattr(market, "closed", False)
            )

    async def _attach_market_price_history(
        self, opportunities: list[ArbitrageOpportunity]
    ) -> None:
        """Hydrate per-market YES/NO price history via scanner shared backfill."""
        if not opportunities:
            return
        try:
            # Use the same scanner backfill/cache pipeline as regular markets.
            from services.scanner import scanner as market_scanner

            attached = await market_scanner.attach_price_history_to_opportunities(
                opportunities,
                now=datetime.now(timezone.utc),
                timeout_seconds=12.0,
            )
            if attached:
                logger.debug(
                    "Attached weather sparkline history",
                    markets_with_history=attached,
                )
        except Exception as exc:
            logger.debug("Weather sparkline hydration skipped", error=str(exc))

    @staticmethod
    def _c_to_f(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return (float(value) * 9.0 / 5.0) + 32.0

    def _build_forecast_sources_payload(self, forecast) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        snapshots = getattr(forecast, "source_snapshots", None) or []
        for snap in snapshots:
            value_c = getattr(snap, "value_c", None)
            payload.append(
                {
                    "source_id": getattr(snap, "source_id", ""),
                    "provider": getattr(snap, "provider", ""),
                    "model": getattr(snap, "model", ""),
                    "value_c": value_c,
                    "value_f": self._c_to_f(value_c),
                    "probability": getattr(snap, "probability", None),
                    "weight": getattr(snap, "weight", None),
                    "target_time": getattr(snap, "target_time", None),
                }
            )

        if payload:
            return payload

        # Backward-compatible fallback if source snapshots are unavailable.
        if forecast.gfs_value is not None:
            payload.append(
                {
                    "source_id": "open_meteo:gfs_seamless",
                    "provider": "open_meteo",
                    "model": "gfs_seamless",
                    "value_c": forecast.gfs_value,
                    "value_f": self._c_to_f(forecast.gfs_value),
                    "probability": forecast.gfs_probability,
                    "weight": None,
                    "target_time": forecast.metadata.get("target_time"),
                }
            )
        if forecast.ecmwf_value is not None:
            payload.append(
                {
                    "source_id": "open_meteo:ecmwf_ifs04",
                    "provider": "open_meteo",
                    "model": "ecmwf_ifs04",
                    "value_c": forecast.ecmwf_value,
                    "value_f": self._c_to_f(forecast.ecmwf_value),
                    "probability": forecast.ecmwf_probability,
                    "weight": None,
                    "target_time": forecast.metadata.get("target_time"),
                }
            )
        return payload

    @staticmethod
    def _is_executable_opportunity(opp: ArbitrageOpportunity) -> bool:
        if (opp.max_position_size or 0.0) <= 0.0:
            return False
        description = str(opp.description or "").strip().lower()
        return not description.startswith("report only")

    @staticmethod
    def _summarize_report_only_findings(
        findings: list[ArbitrageOpportunity],
        *,
        top_n: int = 8,
    ) -> list[dict[str, Any]]:
        if not findings:
            return []
        counts: dict[str, int] = {}
        for finding in findings:
            for factor in finding.risk_factors or []:
                text = str(factor or "").strip()
                if not text:
                    continue
                if text.lower().startswith("report only"):
                    continue
                counts[text] = counts.get(text, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [{"reason": reason, "count": count} for reason, count in ordered]

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
        title = market.question[:110]
        consensus_temp_f = self._c_to_f(signal.consensus_temperature_c)
        market_temp_f = self._c_to_f(signal.market_implied_temperature_c)
        temp_segment = ""
        if consensus_temp_f is not None:
            temp_segment = f" | Consensus {consensus_temp_f:.1f}F"
        if market_temp_f is not None:
            temp_segment += f" vs Mkt {market_temp_f:.1f}F"
        source_summary = self._source_probability_summary(forecast, signal)
        description = (
            f"{signal.direction.replace('_', ' ').upper()} @ ${signal.market_price:.2f} | "
            f"Edge {signal.edge_percent:.2f}% | "
            f"{source_summary}"
            f"{temp_segment}"
        )

        market_probability = float(market.yes_price) if signal.direction == "buy_yes" else float(market.no_price)
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
            "clob_token_ids": market.clob_token_ids,
            "weather": {
                "location": parsed.location,
                "metric": parsed.metric,
                "operator": parsed.operator,
                "threshold_c": parsed.threshold_c,
                "threshold_c_low": parsed.threshold_c_low,
                "threshold_c_high": parsed.threshold_c_high,
                "raw_threshold": parsed.raw_threshold,
                "raw_threshold_low": parsed.raw_threshold_low,
                "raw_threshold_high": parsed.raw_threshold_high,
                "raw_unit": parsed.raw_unit,
                "target_time": parsed.target_time.isoformat(),
                "gfs_probability": signal.gfs_probability,
                "ecmwf_probability": signal.ecmwf_probability,
                "gfs_value": forecast.gfs_value,
                "ecmwf_value": forecast.ecmwf_value,
                "forecast_sources": self._build_forecast_sources_payload(forecast),
                "source_weights": forecast.metadata.get("source_weights"),
                "source_count": signal.source_count,
                "source_spread_c": signal.source_spread_c,
                "source_spread_f": (
                    (signal.source_spread_c * 9.0 / 5.0)
                    if signal.source_spread_c is not None
                    else None
                ),
                "consensus_probability": forecast.consensus_probability,
                "consensus_temp_c": signal.consensus_temperature_c,
                "consensus_temp_f": self._c_to_f(signal.consensus_temperature_c),
                "market_probability": market_probability,
                "market_implied_temp_c": signal.market_implied_temperature_c,
                "market_implied_temp_f": self._c_to_f(signal.market_implied_temperature_c),
                "agreement": signal.model_agreement,
                "model_confidence": signal.confidence,
            },
        }

        max_position_size = min(
            float(settings.get("max_size_usd", 50.0)),
            max(float(market.liquidity) * 0.05, float(settings.get("default_size_usd", 10.0))),
        )
        min_edge = max(1.0, float(settings.get("min_edge_percent", 8.0)))
        edge_scale = max(0.15, min(1.0, signal.edge_percent / (min_edge * 2.0)))
        confidence_scale = max(0.20, min(1.0, signal.confidence))
        entry_soft = float(settings.get("entry_max_price", 0.25))
        soft_price_excess = max(0.0, signal.market_price - entry_soft)
        price_scale = max(0.25, 1.0 - soft_price_excess)
        size_scale = max(
            0.20,
            min(1.0, (edge_scale * 0.45) + (confidence_scale * 0.35) + (price_scale * 0.20)),
        )
        max_position_size = max(
            float(settings.get("default_size_usd", 10.0)),
            min(float(settings.get("max_size_usd", 50.0)), max_position_size * size_scale),
        )

        risk_factors = [
            "Directional weather forecast edge",
            f"Model agreement: {signal.model_agreement:.2f}",
        ]
        risk_factors.extend(signal.notes or [])

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
            risk_factors=risk_factors,
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

    def _source_probability_summary(self, forecast, signal: WeatherSignal) -> str:
        snapshots = getattr(forecast, "source_snapshots", None) or []
        ranked = [
            snap
            for snap in snapshots
            if getattr(snap, "probability", None) is not None
        ]
        if ranked:
            ranked.sort(key=lambda s: float(getattr(s, "weight", 0.0) or 0.0), reverse=True)
            parts: list[str] = []
            for snap in ranked[:2]:
                provider = str(getattr(snap, "provider", "") or "")
                model = str(getattr(snap, "model", "") or "")
                prob = float(getattr(snap, "probability"))
                if provider == "open_meteo":
                    label = model.upper() if model else "OPEN_METEO"
                else:
                    label = f"{provider}:{model}" if model else provider
                parts.append(f"{label} {prob:.0%}")
            if parts:
                return " / ".join(parts)
        return f"GFS {signal.gfs_probability:.0%} / ECMWF {signal.ecmwf_probability:.0%}"

    def get_status(self) -> dict[str, Any]:
        return {
            "cycle_count": self._cycle_count,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_opportunities": self._last_opportunity_count,
            "last_intents": self._last_intent_count,
        }


weather_workflow_orchestrator = WeatherWorkflowOrchestrator()
