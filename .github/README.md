<div align="center">
  <img src="../homerun-logo.png" alt="Homerun" width="340" />
  <h1>Homerun</h1>
  <p><strong>Wire any data source into any trading strategy for prediction markets.</strong></p>
  <p>Polymarket + Kalshi scanning, DB-managed Python runtimes, and paper/live execution orchestration.</p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.109.0-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-18.2.0-61DAFB?logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/TypeScript-5.3.0-3178C6?logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

![Homerun Dashboard — Market Scanner](../screenshots/dashboard-scanner.png)

## TL;DR

Homerun is a full-stack platform for building and running prediction-market trading systems where:

- Data sources are first-class, DB-managed Python classes (`python`, `rss`, `rest_api`) with validation, run history, and normalized records.
- Strategies are first-class, DB-managed Python classes with full lifecycle hooks (`detect`, `evaluate`, `should_exit`).
- Strategies can consume any source directly through `StrategySDK` (`get_data_records`, `run_data_source`, `list_data_sources`, etc.).
- Trader intelligence is available through a formal `TradersSDK` (`get_firehose_signals`, `get_strategy_filtered_signals`, `get_confluence_signals`, etc.).
- You can validate and backtest strategy code before enabling it.
- Opportunities flow into a risk-gated orchestrator for paper or live execution.

If you can fetch the signal, you can trade the signal.

## Why Homerun

Most trading bots force a rigid pipeline or a toy DSL. Homerun does not.

- Full Python strategies, not expression-language rules.
- Full Python data sources, not fixed connectors only.
- Runtime editing via API/UI with source validation and hot reload.
- Unified opportunity model across scanner/news/weather/events/trader flows.
- Built for prediction markets first (binary contracts, event settlement, cross-venue dislocations).

## Quick Links

- Start local dev: `make setup && make dev`
- Start terminal UI: `make run`
- Strategy API docs: `GET /api/strategy-manager/docs`
- Data source API docs: `GET /api/data-sources/docs`
- FastAPI swagger: `http://localhost:8000/docs`

## Wire Any Source Into Any Strategy

![Global Event Map — Data Sources](../screenshots/map.png)

Two ways to connect source data into strategy logic:

1. Event-driven: subscribe strategy to `EventType.DATA_SOURCE_UPDATE`.
2. Direct read: call `StrategySDK.get_data_records(...)` inside your strategy.

In practice, teams combine both: event for wake-up signal, SDK reads for filtered payload access.

## Write a Data Source (full Python)

```python
from services.data_source_sdk import BaseDataSource


class MacroFeedSource(BaseDataSource):
    name = "Macro Feed"
    description = "Example REST ingest for macro events"
    default_config = {
        "endpoint": "https://example.com/macro",
        "limit": 200,
    }

    async def fetch_async(self):
        endpoint = str(self.config.get("endpoint") or "").strip()
        if not endpoint:
            return []

        payload = await self._http_get_json(endpoint, default=[])
        rows = payload if isinstance(payload, list) else payload.get("items", [])

        out = []
        for row in rows[: self._as_int(self.config.get("limit"), 200, 1, 5000)]:
            observed = self._parse_datetime(row.get("timestamp"))
            out.append(
                {
                    "external_id": str(row.get("id") or ""),
                    "title": str(row.get("title") or "").strip(),
                    "summary": str(row.get("summary") or "").strip(),
                    "category": "macro",
                    "source": "macro_feed",
                    "url": row.get("url"),
                    "observed_at": observed.isoformat() if observed else None,
                    "payload": row,
                    "tags": ["macro", "custom"],
                }
            )
        return out
```

## Write a Strategy (full Python)

![Strategy Editor — Source Code & Settings](../screenshots/strategy-editor.png)

```python
from models import Opportunity
from services.data_events import EventType
from services.strategies.base import BaseStrategy
from services.strategy_sdk import StrategySDK


class MacroShockStrategy(BaseStrategy):
    strategy_type = "macro_shock"
    name = "Macro Shock"
    description = "Uses custom macro feed records to find directional mispricing"

    source_key = "scanner"
    subscriptions = [EventType.MARKET_DATA_REFRESH]

    default_config = {
        "source_slug": "macro_feed_source",
        "min_confidence": 0.55,
    }

    async def detect_async(self, events, markets, prices) -> list[Opportunity]:
        source_slug = str(self.config.get("source_slug") or "macro_feed_source")
        records = await StrategySDK.get_data_records(source_slug=source_slug, limit=100)
        if not records:
            return []

        opportunities: list[Opportunity] = []
        confidence = max(0.0, min(1.0, float(self.config.get("min_confidence", 0.55))))

        for market in markets:
            if market.closed or not market.active:
                continue
            if not market.clob_token_ids:
                continue

            yes_token = market.clob_token_ids[0]
            yes_price = market.yes_price
            if yes_token in prices:
                yes_price = float(prices[yes_token].get("mid", yes_price) or yes_price)

            if yes_price >= 0.60:
                continue

            opp = self.create_opportunity(
                title=f"Macro Shock: {market.question[:80]}",
                description="Directional YES based on external macro feed signal",
                total_cost=yes_price,
                expected_payout=1.0,
                is_guaranteed=False,
                markets=[market],
                positions=[
                    {
                        "action": "BUY",
                        "outcome": "YES",
                        "price": yes_price,
                        "token_id": yes_token,
                    }
                ],
                confidence=confidence,
            )
            if opp:
                opp.strategy_context = {
                    "source_slug": source_slug,
                    "records_used": len(records),
                }
                opportunities.append(opp)

        return opportunities
```

## Architecture

```text
External APIs / RSS / Python Sources
                |
                v
       Data Source Runtime
  (fetch -> transform -> upsert)
                |
                v
      data_source_records table  <-----+
                |                      |
                |                  StrategySDK
                |             (read/run/list sources)
                v                      |
   Event Dispatcher + Workers          |
  (market/news/weather/events/traders) |
                |                      |
                +----------> Strategy Runtime (DB-loaded Python)
                               detect / evaluate / should_exit
                                          |
                                          v
                                    Opportunity + Signal
                                          |
                                          v
                                 Trader Orchestrator
                              (risk gates + order manager)
                                          |
                                          v
                              Paper / Live execution paths
```

## Safety and Risk

- This software can drive real-money trading decisions.
- Strategy code should be validated and backtested before enabling execution.
- Start in paper mode and graduate carefully.
- Nothing in this repository is financial advice.

## Contributing

- Read `./docs/CONTRIBUTING.md`
- Security policy: `./docs/SECURITY.md`

## License

MIT. See `./LICENSE`.
