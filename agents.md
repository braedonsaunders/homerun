# Homerun - Agent Codex

This is a **financial application** — a full-stack prediction market arbitrage platform. Every line of code you write will handle real money decisions. You must write production-quality, reference-grade code. No stubs. No TODOs. No placeholders. No "implement later" comments. Every function must be complete and correct on first write.

---

## Core Principles

**1. Ship complete code only.** Every function, method, route, and component must be fully implemented. If you cannot implement something fully, say so — do not leave a stub. Partial implementations in financial systems cause silent losses.

**2. Clean cut, not backwards compatible.** When changing something, replace it completely. Do not preserve legacy code paths, re-export old names, add compatibility shims, or leave dead code behind. Duplicate implementations and dead code branches are how financial bugs hide. Delete the old, write the new.

**3. No speculative abstractions.** Do not create helpers, utilities, base classes, or wrappers for one-time operations. Do not add configurability that wasn't asked for. Three similar lines of code is better than a premature abstraction. Add complexity only when the current task demands it.

**4. Trust internal code.** This is a single-user, locally-deployed application. Do not add authentication, authorization, CSRF protection, input sanitization for internal service calls, or defensive error handling for impossible states. Only validate at true system boundaries (external API responses, user-facing form inputs).

**5. Delete, don't deprecate.** When removing functionality, delete it entirely — the code, the route, the database column, the frontend component, the type definition. No `// deprecated`, no `_unused` prefixes, no tombstone comments.

---

## Architecture Overview

```
homerun/
├── backend/                    # Python FastAPI async server
│   ├── api/                    # Route handlers
│   │   ├── routes.py           # Core opportunity CRUD
│   │   ├── routes_simulation.py
│   │   ├── routes_strategies.py
│   │   ├── routes_traders.py
│   │   ├── routes_trader_orchestrator.py
│   │   ├── routes_copy_trading.py
│   │   ├── routes_discovery.py
│   │   ├── routes_anomaly.py
│   │   ├── routes_ai.py
│   │   ├── routes_news.py
│   │   ├── routes_crypto.py
│   │   └── websocket.py        # WebSocket connection manager
│   ├── models/
│   │   ├── opportunity.py      # ArbitrageOpportunity Pydantic model
│   │   └── database.py         # SQLAlchemy ORM models + session factory
│   ├── services/
│   │   ├── strategies/         # Arbitrage detection strategies
│   │   │   ├── base.py         # BaseStrategy ABC
│   │   │   ├── basic.py        # BasicArbStrategy
│   │   │   └── ...             # 30+ strategy implementations
│   │   ├── simulation.py       # Paper trading engine
│   │   ├── copy_trader.py      # Copy trading service
│   │   ├── credential_manager.py
│   │   ├── event_bus.py        # Async pub/sub singleton
│   │   └── ...
│   ├── workers/                # Background async event loops
│   │   ├── scanner_worker.py
│   │   ├── discovery_worker.py
│   │   └── ...
│   ├── utils/
│   │   ├── logger.py           # Structured logging
│   │   ├── rate_limiter.py
│   │   ├── retry.py
│   │   └── utcnow.py
│   ├── config.py               # 180+ Pydantic Settings fields
│   └── main.py                 # FastAPI app with lifespan
├── frontend/                   # React 18 + TypeScript + Vite
│   └── src/
│       ├── components/         # 50+ React components
│       ├── services/           # Axios API clients
│       │   ├── api.ts          # Main API service + type definitions
│       │   └── discoveryApi.ts # Discovery-specific API
│       ├── hooks/              # Custom hooks (useWebSocket, etc.)
│       ├── store/atoms.ts      # Jotai atoms for global UI state
│       └── lib/                # Utilities (cn, timestamps)
├── alembic/                    # Database migrations
│   └── versions/               # Timestamped migration files
├── data/                       # SQLite database (auto-created)
├── tui.py                      # Terminal UI (Textual + Rich)
└── docs/                       # Strategy research documentation
```

---

## Tech Stack (exact versions matter)

| Layer | Technology | Version |
|-------|-----------|---------|
| Backend framework | FastAPI | 0.109.0 |
| Server | Uvicorn | 0.27.0 |
| ORM | SQLAlchemy (async) | 2.0.25 |
| Validation | Pydantic | 2.5.0 |
| Database | SQLite | (via aiosqlite) |
| Frontend framework | React | 18.2.0 |
| Language | TypeScript | 5.3.0 |
| Build | Vite | 5.0.0 |
| Styling | TailwindCSS + shadcn/ui | 3.4.0 |
| State | Jotai | 2.17.1 |
| Data fetching | TanStack React Query | 5.17.0 |
| HTTP client | Axios | 1.6.0 |
| Charts | Recharts + Lightweight Charts | 3.7.0 / 5.1.0 |
| Animations | Framer Motion | 12.33.0 |

---

## Domain Model

### What Homerun Does

Homerun scans prediction markets (Polymarket, Kalshi) for mispricings and arbitrage opportunities. It detects when:

- **WITHIN_MARKET**: YES + NO prices sum to less than $1.00 on the same market
- **CROSS_MARKET**: The same event is priced differently on different platforms
- **SETTLEMENT_LAG**: A market's outcome is known but the price hasn't settled yet
- **NEWS_INFORMATION**: Breaking news implies a probability different from the market price

It then allows paper trading, copy trading, or live execution of these opportunities.

### Key Domain Types

```python
# Strategy types (StrategyType enum in models/opportunity.py)
BASIC           # YES+NO sum < $1.00
NEGRISK         # Mutually exclusive bundle arbitrage
BTC_ETH_HIGHFREQ # 5m/15m crypto binary arbitrage
NEWS_EDGE       # LLM probability vs market price
WEATHER_EDGE    # Weather forecast vs market pricing
# ... 20+ more

# Mispricing classification
MispricingType: WITHIN_MARKET | CROSS_MARKET | SETTLEMENT_LAG | NEWS_INFORMATION

# ROI classification
ROIType: GUARANTEED_SPREAD | DIRECTIONAL_PAYOUT

# Trade lifecycle
TradeStatus: PENDING | OPEN | RESOLVED_WIN | RESOLVED_LOSS | CANCELLED | FAILED

# Position sides
PositionSide: YES | NO

# Copy trading modes
CopyTradingMode: ALL_TRADES | ARB_ONLY
```

### The Core Data Object: ArbitrageOpportunity

Defined in `backend/models/opportunity.py`. This is the single most important model in the system. Every strategy produces these, every UI component consumes them. Key fields:

- `id` / `stable_id` — auto-generated from market fingerprint + strategy name
- `strategy` — which strategy detected it
- `total_cost`, `expected_payout`, `gross_profit`, `fee`, `net_profit`, `roi_percent` — the financials
- `is_guaranteed` — true for pure arbitrage, false for directional bets
- `risk_score` (0.0-1.0), `risk_factors` — risk assessment
- `markets` — list of market dicts with prices, tokens, liquidity
- `positions_to_take` — list of position dicts (action, outcome, price, token_id)
- `ai_analysis` — optional AIAnalysis with scores and recommendation
- `strategy_context` — strategy-specific payload dict

---

## Code Patterns — Follow These Exactly

### Strategy Implementation

Every strategy extends `BaseStrategy` from `backend/services/strategies/base.py`.

Required:
- Set class attributes: `strategy_type`, `name`, `description`
- Implement `detect(events, markets, prices) -> list[ArbitrageOpportunity]` for sync work
- OR implement `async detect_async(events, markets, prices) -> list[ArbitrageOpportunity]` for I/O-bound work

Optional overrides:
- `evaluate(signal, context) -> StrategyDecision` — execution gating during orchestrator phase
- `should_exit(position, market_state) -> ExitDecision` — exit logic for open positions
- `calculate_risk_score(markets, resolution_date) -> tuple[float, list[str]]`
- `configure(config: dict) -> None` — apply user config overrides

Use `self.create_opportunity(...)` helper to construct opportunities with proper ID generation.

Decision dataclasses:
```python
@dataclass
class StrategyDecision:
    decision: str  # "selected" | "skipped" | "blocked" | "failed"
    reason: str
    score: float = None
    size_usd: float = None
    checks: list[DecisionCheck] = field(default_factory=list)

@dataclass
class ExitDecision:
    action: str  # "close" | "hold" | "reduce"
    reason: str
    close_price: float = None
    reduce_fraction: float = None  # 0-1 for partial exit
```

### Route Handlers

Pattern in `backend/api/routes_*.py`:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

class CreateThingRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    value: float = Field(default=0.0, ge=0.0)

@router.post("/things")
async def create_thing(request: CreateThingRequest):
    try:
        result = await some_service.create(name=request.name, value=request.value)
    except OperationalError as exc:
        if simulation_service.is_sqlite_lock_error(exc):
            raise HTTPException(status_code=503, detail="Database is busy; please retry.") from exc
        raise
    return {"id": result.id, "name": result.name}
```

Key conventions:
- Request/response models are Pydantic `BaseModel` with `Field()` validation
- HTTP errors use `HTTPException` with specific status codes (400, 404, 409, 503)
- SQLite lock errors get 503 with retry hint
- Exception chaining with `from exc`
- Return plain dicts (FastAPI serializes to JSON)

### Database Operations

Session factory in `backend/models/database.py`:

```python
from sqlalchemy.ext.asyncio import AsyncSession
from models.database import AsyncSessionLocal

# Always use async context manager
async with AsyncSessionLocal() as session:
    result = await session.execute(select(Model).where(Model.id == some_id))
    obj = result.scalar_one_or_none()

    # For writes
    session.add(new_obj)
    await session.commit()
    await session.refresh(new_obj)
```

SQLite lock retry pattern (use when writing):
```python
for attempt in range(RETRY_ATTEMPTS):
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(text(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}"))
            # ... do work ...
            await session.commit()
            return result
        except OperationalError as exc:
            await session.rollback()
            if not is_sqlite_lock_error(exc) or attempt >= RETRY_ATTEMPTS - 1:
                raise
            delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
            await asyncio.sleep(delay)
```

### ORM Models

Defined in `backend/models/database.py` using SQLAlchemy declarative base:

```python
class SomeModel(Base):
    __tablename__ = "some_models"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    value = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships use back_populates (not backref)
    children = relationship("ChildModel", back_populates="parent")
```

### Alembic Migrations

File naming: `{YYYYMMDD}{NNNN}_{description}.py` in `backend/alembic/versions/`.

```python
revision = "202602170001"
down_revision = "<previous_revision>"

def _column_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}

def upgrade() -> None:
    existing = _column_names("target_table")
    columns = [sa.Column("new_field", sa.String(), nullable=True, server_default=sa.text("'default'"))]
    for col in columns:
        if col.name not in existing:
            op.add_column("target_table", col)

def downgrade() -> None:
    pass  # SQLite safety — no downgrade
```

Always check `_column_names` before adding columns (idempotent migrations).

### Background Workers

Pattern in `backend/workers/*_worker.py`:

```python
async def _run_loop() -> None:
    logger.info("Worker started")

    # Write initial snapshot to DB
    async with AsyncSessionLocal() as session:
        await write_worker_snapshot(session, "worker_name", running=True, ...)

    while True:
        # Read control state from DB
        async with AsyncSessionLocal() as session:
            control = await read_worker_control(session)

        paused = bool(control.get("is_paused", False))
        enabled = bool(control.get("is_enabled", True))

        if not enabled or paused:
            await asyncio.sleep(1)
            continue

        try:
            async with AsyncSessionLocal() as session:
                result = await do_actual_work(session)
                await write_worker_snapshot(session, "worker_name", ...)
        except Exception as e:
            logger.error("Worker error", exc_info=e)

        await asyncio.sleep(interval_seconds)
```

Workers read control state from DB each iteration. They write snapshot state back for the UI to display.

### WebSocket Messages

All WebSocket messages are JSON with `type` and `data` fields:

```python
await manager.broadcast({
    "type": "opportunities_update",
    "data": {
        "count": len(opportunities),
        "opportunities": [serialize(o) for o in opportunities[:20]],
    },
})
```

Common types: `init`, `opportunities_update`, `scanner_status`, `trade_executed`, `ping`/`pong`, `subscribed`.

### Event Bus

Singleton in `backend/services/event_bus.py`:

```python
from services.event_bus import event_bus

# Subscribe
event_bus.subscribe("trade.executed", my_callback)
event_bus.subscribe("*", catch_all_callback)  # wildcard

# Publish (fire-and-forget via asyncio.create_task)
await event_bus.publish("trade.executed", {"trade_id": "...", "pnl": 42.0})
```

### Structured Logging

```python
from utils.logger import get_logger
logger = get_logger(__name__)

logger.info("Created account", account_id=account.id, capital=10000.0)
logger.warning("SQLite lock; retrying", attempt=3, delay=0.8)
logger.error("Worker failed", exc_info=e)
```

Always include relevant context as keyword arguments. Never log raw exceptions as strings — use `exc_info=e`.

### Frontend Components

React components in `frontend/src/components/`:

```typescript
import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useAtom } from 'jotai'
import { SomeIcon } from 'lucide-react'
import { cn } from '../lib/utils'
import { someAtom } from '../store/atoms'
import { getSomeData } from '../services/api'

export default function MyComponent() {
  const [atomValue, setAtomValue] = useAtom(someAtom)
  const { data = [], isLoading } = useQuery({
    queryKey: ['some-data'],
    queryFn: getSomeData,
  })

  // Component logic and JSX
}
```

Conventions:
- Default exports for components
- `useQuery` with `queryKey` array and `queryFn`
- Jotai atoms for global UI state (theme, selections, modal visibility)
- `cn()` utility for conditional Tailwind classes
- Lucide React for icons
- shadcn/ui components from `../components/ui/`

### Frontend API Services

```typescript
import axios from 'axios'
import { normalizeUtcTimestampsInPlace } from '../lib/timestamps'

const api = axios.create({ baseURL: '/api', timeout: 60000 })

api.interceptors.response.use((response) => {
  normalizeUtcTimestampsInPlace(response.data)
  return response
})

export interface SomeThing {
  id: string
  name: string
  value: number
}

export async function getThings(): Promise<SomeThing[]> {
  const response = await api.get('/things')
  return response.data
}
```

Always run `normalizeUtcTimestampsInPlace` on response data. Define TypeScript interfaces for all API response shapes in the same service file.

### Frontend State (Jotai Atoms)

Defined in `frontend/src/store/atoms.ts`:

```typescript
import { atom } from 'jotai'
import { atomWithStorage } from 'jotai/utils'

// Persisted to localStorage
export const themeAtom = atomWithStorage<'dark' | 'light'>('theme', 'dark')
export const selectedAccountIdAtom = atomWithStorage<string | null>('selectedAccountId', null)

// Ephemeral
export const copilotOpenAtom = atom(false)

// Derived
export const isLiveAccountAtom = atom((get) => {
  const id = get(selectedAccountIdAtom)
  return id?.startsWith('live:') ?? false
})
```

Use `atomWithStorage` for anything that should survive page refresh. Use plain `atom` for transient UI state.

---

## Naming Conventions

### Python
- Classes: `PascalCase` — `SimulationService`, `BasicArbStrategy`
- Functions/methods: `snake_case` — `create_account`, `detect_async`
- Constants: `UPPER_SNAKE_CASE` — `POLYMARKET_FEE`, `MAX_RETRY_ATTEMPTS`
- Private: leading underscore — `_detect_negrisk_event`, `_running`
- Enums: `PascalCase` class, `UPPER_SNAKE_CASE` values — `TradeStatus.RESOLVED_WIN`
- Files: `snake_case.py` — `routes_simulation.py`, `scanner_worker.py`

### TypeScript
- Interfaces/Types: `PascalCase` — `Opportunity`, `WebSocketMessage`
- Functions: `camelCase` — `getOpportunities`, `normalizeUtcTimestampsInPlace`
- Constants: `UPPER_SNAKE_CASE` — `MAX_RECONNECT_DELAY`, `CLIENT_PING_INTERVAL_MS`
- Components: `PascalCase` files and exports — `OpportunityCard.tsx`
- Hooks: `camelCase` with `use` prefix — `useWebSocket.ts`

---

## Common Gotchas

1. **SQLite locking** — All write operations must use the retry-with-backoff pattern. SQLite allows one writer at a time. Never hold a session open across an `await` that could take more than a few ms.

2. **Timezone handling** — All datetimes are UTC. Use `datetime.now(timezone.utc)` or the `utcnow()` utility, never bare `datetime.utcnow()` (it returns naive datetimes). The frontend calls `normalizeUtcTimestampsInPlace` on all API responses to ensure consistent parsing.

3. **Opportunity IDs** — `stable_id` persists across scans (same strategy + same markets = same stable_id). `id` includes a timestamp and is unique per detection. Use `stable_id` for deduplication and UI tracking.

4. **Live price data** — Strategies receive a `prices` dict mapping token IDs to price objects. Always prefer live prices over static `market.yes_price`/`market.no_price` when the token ID exists in the prices dict.

5. **WebSocket singleton** — The frontend uses a shared WebSocket connection (module-level singleton in `useWebSocket.ts`). Multiple hook consumers share one connection. Do not create additional WebSocket connections.

6. **Settings are hot-reloadable** — The 180+ settings in `config.py` can be changed at runtime via the settings API. Workers re-read configuration each loop iteration from DB. Do not cache settings at import time.

7. **Pydantic v2** — This project uses Pydantic v2 (`BaseModel`, `Field`, `model_validator`). Do not use v1 patterns like `@validator`, `class Config`, or `.dict()`. Use `@field_validator`, `model_config`, and `.model_dump()`.

---

## What NOT to Do

- Do not add type annotations, docstrings, or comments to code you did not change
- Do not refactor surrounding code when fixing a bug
- Do not add error handling for impossible states
- Do not create wrapper functions for single-use operations
- Do not add feature flags or environment variable toggles unless asked
- Do not add logging for routine successful operations (log errors and warnings, not happy paths)
- Do not re-export removed symbols for backwards compatibility
- Do not add `# TODO`, `# FIXME`, `# HACK`, `pass`, `raise NotImplementedError`, or `...` as implementation
- Do not add empty `except: pass` blocks
- Do not add input validation for internal service-to-service calls
- Do not preserve old function signatures when changing them — update all callers
