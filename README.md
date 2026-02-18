# Homerun

**Autonomous prediction market trading platform for Polymarket and Kalshi.**

Homerun detects mispricings in real time — guaranteed-spread arbitrage, cross-platform dislocations, news-driven edge, weather ensemble signals, geopolitical intelligence — and executes against them autonomously with a multi-gate risk orchestrator.

---

## What it does

| Layer | Description |
|---|---|
| **Scanning** | Continuous market scan across Polymarket + Kalshi via CLOB WebSocket feeds |
| **Detection** | 30+ live strategies covering arb, ML, news, weather, geo-intelligence, copy trading |
| **Filtering** | 10-stage quality filter pipeline with per-strategy override hooks |
| **Execution** | Paper or live trading via orchestrator with portfolio risk gates |
| **Replay** | Frozen strategy context on every signal — replayable audit trail |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Scanner Worker      News Worker    Crypto Worker   Weather Worker
│  (market scan)      (RSS/LLM)      (exchange WS)   (Open-Meteo)
└──────────────┬──────────────┬───────────────┬────────────┬─────┘
               │              │               │            │
               ▼              ▼               ▼            ▼
          DataEvent ──────► EventDispatcher ◄──────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      Strategy SDK          │
                    │  on_event() → evaluate()   │
                    │  QualityFilterPipeline     │
                    └─────────────┬─────────────┘
                                  │ ArbitrageOpportunity
                    ┌─────────────▼─────────────┐
                    │   Trader Orchestrator      │
                    │  Risk gates → Order Mgr    │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   Polymarket CLOB / Kalshi │
                    └───────────────────────────┘
```

**Backend**: FastAPI + SQLAlchemy 2.0 (async) + SQLite, worker subprocesses via `subprocess.Popen`
**Frontend**: React 18 + TypeScript, Jotai, TanStack Query, Recharts, shadcn/ui, Framer Motion
**Feeds**: Polymarket CLOB WebSocket (sub-second prices), Kalshi CLOB WebSocket
**AI**: LLM semantic matching for news → market alignment, on-device ML embeddings

---

## Strategy SDK

Every strategy is a class that declares its subscriptions and implements hooks into the composable evaluation pipeline. No boilerplate.

```python
from services.strategies.base import BaseStrategy, ScoringWeights, SizingConfig
from services.data_events import DataEvent, EventType

class BreakingNewsEdge(BaseStrategy):
    strategy_type = "news_edge"
    subscriptions = [EventType.NEWS_UPDATE]

    scoring_weights = ScoringWeights(
        edge_weight=0.55,
        confidence_weight=30.0,
        risk_penalty=8.0,
    )
    sizing_config = SizingConfig(
        base_divisor=100.0,
        confidence_offset=0.75,
        risk_scale_factor=0.35,
    )

    async def on_event(self, event: DataEvent) -> None:
        opportunities = await self.detect_from_news(event.payload)
        for opp in opportunities:
            await self.emit_opportunity(opp)

    def custom_checks(self, signal, context, params, payload):
        # Strategy-specific gates run inside the evaluate pipeline
        return [
            DecisionCheck("relevance_score", payload.relevance >= 0.7, payload.relevance),
            DecisionCheck("freshness_secs", payload.age_seconds <= 120, payload.age_seconds),
        ]

    def compute_score(self, edge, confidence, risk_score, market_count, payload):
        # Override scoring formula — or omit to use the composable default
        return edge * confidence * payload.relevance - risk_score * 0.08
```

### Event types

```python
class EventType(str, Enum):
    PRICE_CHANGE         = "price_change"         # Individual token WS update
    MARKET_DATA_REFRESH  = "market_data_refresh"  # Periodic full market batch
    CRYPTO_UPDATE        = "crypto_update"         # Exchange prices + volatility
    WEATHER_UPDATE       = "weather_update"        # Open-Meteo ensemble forecast
    NEWS_UPDATE          = "news_update"           # LLM-matched article + intent
    TRADER_ACTIVITY      = "trader_activity"       # Smart wallet copy signal
    WORLD_INTEL_UPDATE   = "world_intel_update"    # Geopolitical signal payload
```

### Quality filter pipeline

Every opportunity runs through 10 cascading hard-rejection filters before reaching the orchestrator. Each filter emits a `FilterResult` — failures are stored with reasons for audit.

```
1. Edge threshold         (ROI ≥ MIN_ANNUALIZED_ROI)
2. Plausibility cap       (ROI ≤ MAX_PLAUSIBLE_ROI — catches stale CLOB artifacts)
3. Liquidity              (min depth per leg)
4. Time-to-resolution     (excludes markets expiring too soon)
5. NegRisk exhaustivity   (validates mutual-exclusivity of bundle legs)
6. Settlement lag guard   (already-resolved market filter)
7. Tradability            (API-confirmed market is open)
8. Confidence threshold   (detect-time signal strength ≥ floor)
9. Spread sanity          (bid/ask spread sanity on execution legs)
10. Strategy override     (per-strategy custom reject logic)
```

Strategies can override filter thresholds for their category. Crypto strategies relax liquidity floors; weather strategies tighten time-to-resolution windows.

---

## 30+ Live Strategies

### Arbitrage
| Strategy | Edge |
|---|---|
| `basic` | YES + NO price sum < $1.00 guaranteed spread |
| `negrisk` | Mutually-exclusive outcome bundle arbitrage |
| `mutually_exclusive` | Multi-market guaranteed payout |
| `contradiction` | Logically impossible co-occurrence pricing |
| `settlement_lag` | Resolved-but-unsettled market gap exploitation |
| `spread_dislocation` | Cross-venue price dislocations |
| `cross_platform` | Polymarket ↔ Kalshi routing + hedge coordination |

### Market Structure
| Strategy | Edge |
|---|---|
| `btc_eth_highfreq` | 5m/15m/1h/4h crypto binary arb, maker-mode execution |
| `liquidity_vacuum` | Severe bid/ask imbalances |
| `entropy_arb` | Pricing entropy in multi-leg structures |
| `bayesian_cascade` | Dependency graph update propagation |
| `correlation_arb` | Correlation breakdown trades |
| `stat_arb` | Statistical mean-reversion |
| `combinatorial` | Multi-leg execution with hedge orchestration and fill optimization |

### Intelligence-Driven
| Strategy | Edge |
|---|---|
| `news_edge` | LLM semantic matching of breaking news to markets |
| `weather_ensemble_edge` | Ensemble forecast agreement → market mispricing |
| `weather_conservative_no` | High-confidence defensive NO on weather outcomes |
| `weather_distribution` | Ensemble disagreement → implied probability spread |
| `traders_confluence` | Confluence signals from tracked smart wallets |

### Temporal & Dynamics
| Strategy | Edge |
|---|---|
| `temporal_decay` | Time-decay repricing with certainty shock detection |
| `tail_end_carry` | Long-tail event carry |
| `event_driven` | Event timeline-based mispricing |
| `must_happen` | Logical necessity arbitrage |
| `miracle` | Directional contrarian on extremely low-probability NO |

---

## World Intelligence

Homerun integrates live geopolitical signal feeds as trading alpha:

| Source | Signal |
|---|---|
| **ACLED** | Conflict events, military actions, protest activity |
| **GDELT** | Media-derived geopolitical event stream |
| **AIS** | Naval vessel movements (ship tracking) |
| **OpenSky** | Military aircraft activity |
| **USGS** | Earthquake events |
| **Open-Meteo Ensemble** | 50-member weather forecast agreement scoring |

Signals are normalized into `WORLD_INTEL_UPDATE` DataEvents and matched against open Polymarket + Kalshi markets by the world intelligence strategies.

---

## Orchestrator

The `TraderOrchestrator` is an autonomous execution engine. Every signal passes through sequential gates before an order is placed:

```
Signal received
   │
   ├─ Trading window check      (time-of-day / day-of-week gates)
   ├─ Stacking guard            (existing open positions in same market)
   ├─ Portfolio risk check      (overall exposure, drawdown limits)
   ├─ Position size cap         (per-signal MAX_TRADE_SIZE_USD)
   ├─ Tradability re-validation (live CLOB confirms market is open)
   └─ Order Manager             → maker-mode limit orders → fills
```

Hooks called at override points:
- `strategy.on_blocked(signal, reason)` — strategy learns why it was gated
- `strategy.on_size_capped(signal, original, capped)` — size reduction notification

---

## Data Model

The central object is `ArbitrageOpportunity`:

```python
@dataclass
class ArbitrageOpportunity:
    id: str                      # UUID
    stable_id: str               # Fingerprint-based deduplication key
    strategy: str                # Which strategy produced this
    roi_percent: float           # Theoretical edge
    realistic_roi: float         # VWAP-adjusted after slippage model
    risk_score: float            # 0.0–1.0 composite risk
    confidence: float            # Detection-time signal strength
    markets: list[MarketLeg]     # Each leg with price, liquidity, volume
    positions_to_take: list[PositionSpec]
    execution_plan: ExecutionPlan
    strategy_context_json: dict  # Frozen context for replay/analysis
```

Every `TradeSignal` stored in the DB links back to a frozen `strategy_context_json` so decisions can be replayed and audited offline.

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/your-org/homerun
cd homerun
make setup         # Creates venv, installs deps, runs migrations

# Configure (copy and edit)
cp backend/.env.example backend/.env

# Start dev servers (API on :8000, frontend on :5173)
make dev
```

**Minimum `.env`:**
```env
POLYMARKET_API_KEY=your_key
POLYMARKET_SECRET=your_secret
POLYMARKET_PASSPHRASE=your_passphrase
KALSHI_EMAIL=your_email
KALSHI_PASSWORD=your_password
```

**Optional (enables extra strategies):**
```env
OPENAI_API_KEY=...         # news_edge LLM matching
ACLED_API_KEY=...          # world intelligence conflict feed
OPENSKY_USERNAME=...       # military aircraft tracking
```

---

## Configuration

230+ tunable settings in `backend/config.py` via `.env` or the database settings table:

```env
# Scanning
SCAN_INTERVAL_SECONDS=30
MAX_MARKETS_TO_SCAN=500

# Quality filters
MIN_LIQUIDITY_HARD=100
MIN_ANNUALIZED_ROI=0.05
MAX_PLAUSIBLE_ROI=0.95

# Trading limits
MAX_TRADE_SIZE_USD=50
MAX_DAILY_TRADE_VOLUME=500
MAX_OPEN_POSITIONS=20

# Risk
RISK_SHORT_DAYS=3
RISK_LONG_LOCKUP_DAYS=30
```

All settings are hot-reloadable from the frontend settings panel without restarting.

---

## Writing a Strategy

1. Create `backend/services/strategies/my_strategy.py`
2. Subclass `BaseStrategy`
3. Declare `subscriptions`, `scoring_weights`, `sizing_config`
4. Implement `on_event()` (event-driven) or `detect()` (scan-driven)
5. Register in `backend/services/strategy_catalog.py`

The strategy will be picked up on next scanner reload, hot-reloaded on source change, and immediately visible in the frontend strategies panel with enable/disable + threshold controls.

---

## Tech Stack

| | |
|---|---|
| **FastAPI 0.109** | Async REST + WebSocket |
| **SQLAlchemy 2.0** | Async ORM, type-safe queries |
| **SQLite + aiosqlite** | Zero-dependency persistence |
| **Alembic** | Schema migrations |
| **React 18 + TypeScript** | Frontend |
| **Vite 5** | Build tooling |
| **TailwindCSS + shadcn/ui** | Styling + headless components |
| **Jotai** | Atomic client state |
| **TanStack Query** | Server state + caching |
| **Recharts + Lightweight Charts** | Real-time + historical charts |
| **Framer Motion** | UI animations |

---

## Status

Paper trading and live trading are both functional. The platform is under active development. Known architectural issues:

- **Process boundary**: EventDispatcher subscriptions exist only in the API process. Worker subprocesses (crypto, news, weather) dispatch DataEvents but no strategy handlers fire. Fix requires IPC (Redis pub/sub or DB polling bridge).
- **Hidden platform logic**: Dedupe, caps, expiry, orchestrator gates all modify signal outcomes after `strategy.evaluate()` returns. Not yet documented in the strategy contract.

---

## License

Private — not for redistribution.
