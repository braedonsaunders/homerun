<p align="center">
  <img src="homerun-logo.png" alt="Homerun Logo" width="200"/>
</p>

<h1 align="center">Homerun</h1>

<p align="center">
  <strong>$40M+ in arbitrage profits have been extracted from Polymarket.</strong><br/>
  This tool finds those opportunities in real-time — and can trade them automatically.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#the-8-strategies">Strategies</a> •
  <a href="#features">Features</a> •
  <a href="#autonomous-trading">Auto Trading</a> •
  <a href="#api">API</a>
</p>

---

## What Is This?

Prediction markets misprice things. When mutually exclusive outcomes sum to more than 100%, or the same event trades at different prices across markets, there's free money on the table.

Homerun is an autonomous scanner and trading bot that:

- **Detects** pricing inefficiencies across 8 distinct arbitrage strategies
- **Scores** opportunities by profit, risk, and liquidity
- **Executes** trades automatically — or lets you paper trade first
- **Tracks** profitable wallets and copies their moves
- **Alerts** you via Telegram when opportunities appear

It's a full-stack app: FastAPI backend, React dashboard, real-time WebSocket updates, and direct integration with Polymarket's CLOB orderbook.

---

## The 8 Strategies

### 1. Basic Arbitrage
Buy YES + NO on the same binary market when total cost < $1.00. One side must win. Guaranteed profit.

> Historical: **$10.58M** extracted from 7,051 exploitable conditions

### 2. Mutually Exclusive Arbitrage
Two events where only one can happen (e.g., Democrat vs Republican wins). Buy YES on both for < $1.00.

### 3. Contradiction Arbitrage
Markets saying opposite things. "BTC above $100K by March" and "BTC below $100K in March" — buy YES on both when prices don't add up.

### 4. NegRisk Arbitrage (Most Profitable)
One-of-many markets where all outcomes sum to $1.00. Buy positions across all outcomes when total cost is under $1.00. This covers date-sweep markets, exhaustive event outcomes, and any NegRisk-flagged event group.

> Historical: **$28.99M** from 662 markets — 29x more capital-efficient than basic arb

### 5. Must-Happen Arbitrage
Exhaustive outcomes where one *must* win. Buy YES on all outcomes when total < $1.00.

### 6. Miracle Scanner
Bet NO on events with near-zero probability of happening:
- Aliens making contact this month
- WW3 starting by Friday
- Bitcoin hitting $10M this week

Almost free money, with a tiny tail risk.

### 7. Combinatorial Arbitrage
Cross-market arbitrage using integer programming. Finds complex multi-leg opportunities that span different event groups where the combined positions create a guaranteed profit.

### 8. Settlement Lag Exploitation
Exploits delayed price updates after an outcome has been effectively determined but markets haven't fully adjusted. Speed advantage on information that's already public.

---

## Quick Start

```bash
git clone https://github.com/braedonsaunders/homerun.git
cd homerun
./setup.sh && ./run.sh
```

Open **http://localhost:3000** — you're scanning for arbitrage.

### Other Setup Methods

**Docker:**
```bash
docker-compose up --build
```

**Make:**
```bash
make setup && make run
```

**Manual:**
```bash
# Backend
cd backend && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && uvicorn main:app --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm run dev
```

### Requirements
- Python 3.10+
- Node.js 18+
- Docker (optional)

### Endpoints After Startup

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| WebSocket | ws://localhost:8000/ws |
| Health Check | http://localhost:8000/health |

---

## Features

### Core
- **8 Arbitrage Strategies** — from basic same-market arb to cross-market integer programming
- **Real-time Scanning** — WebSocket-powered live market monitoring
- **Autonomous Trading** — paper, shadow, or live mode
- **Copy Trading** — auto-replicate trades from profitable wallets
- **Anomaly Detection** — flag wallets with suspicious win rates, wash trading, or front-running
- **Wallet Tracking** — monitor any wallet's positions and trades in real-time
- **Cross-platform Scanning** — Kalshi integration for cross-market arbitrage
- **ML Opportunity Ranking** — machine learning classifier to score and prioritize opportunities
- **Telegram Notifications** — real-time alerts on your phone

### Trading Infrastructure
- **Direct CLOB Integration** — execute on Polymarket's orderbook
- **Position Sizing** — fixed, Kelly criterion, or volatility-adjusted
- **Circuit Breakers** — auto-pause after consecutive losses
- **Emergency Stop** — kill switch for all open orders
- **Daily Limits** — cap trades, volume, and losses per day
- **VWAP Execution** — volume-weighted average price for better fills

### Advanced Optimization
- **Frank-Wolfe Algorithm** — portfolio-level position optimization
- **Constraint Solver** — find optimal allocations across correlated markets
- **Dependency Detection** — identify hidden correlations between markets
- **Bregman Divergence** — information-theoretic optimization
- **Parallel Execution** — multi-threaded trade execution
- **Decay Analysis** — track how fast opportunities disappear
- **Parameter Optimization** — auto-tune scanner and trading parameters

### Production-Ready
- **Rate Limiting** — respects Polymarket API limits
- **Structured Logging** — JSON logs for debugging and aggregation
- **Health Checks** — Kubernetes-compatible liveness and readiness probes
- **Prometheus Metrics** — monitor everything at `/metrics`
- **Database Maintenance** — automatic cleanup of stale data
- **Auto-migration** — schema updates applied on startup without data loss

---

## Autonomous Trading

Start the auto-trader in three modes:

| Mode | Description |
|------|-------------|
| `paper` | Virtual money. Learn the system risk-free. |
| `shadow` | Tracks trades without executing. See what you'd make. |
| `live` | Real money. Real profits. Real risk. |

### Configuration

```bash
min_roi_percent: 2.5          # Minimum ROI to trigger trade
max_risk_score: 0.5           # Maximum acceptable risk (0-1)
base_position_size_usd: 10    # Default position size
max_position_size_usd: 100    # Maximum per trade
max_daily_trades: 50          # Daily trade limit
max_daily_loss_usd: 100       # Stop-loss if exceeded
```

### Live Trading Setup

1. Get API credentials from [Polymarket Settings](https://polymarket.com/settings/api-keys)
2. Copy `.env.example` to `.env` and add your keys:
```bash
TRADING_ENABLED=true
POLYMARKET_PRIVATE_KEY=your_key
POLYMARKET_API_KEY=your_api_key
POLYMARKET_API_SECRET=your_secret
POLYMARKET_API_PASSPHRASE=your_passphrase
```
3. Start in paper mode first. Verify it works.
4. Switch to live via the dashboard or API.

---

## Anomaly Detection

Find wallets doing the statistically impossible:

| Anomaly | What It Means |
|---------|---------------|
| `impossible_win_rate` | >95% win rate over many trades |
| `unusual_roi` | Returns far above market average |
| `perfect_timing` | Consistently buys lows, sells highs |
| `statistically_impossible` | Zero losses across 100+ trades |
| `wash_trading` | Rapid buy/sell cycles in the same market |
| `front_running` | Suspicious timing before price moves |
| `arbitrage_only` | Exclusively executes arb trades (bot signature) |

Use this to find wallets worth copy-trading — or to identify manipulation.

---

## API

### Opportunities
```
GET  /api/opportunities          # Current arbitrage opportunities (with filtering)
GET  /api/opportunities/{id}     # Specific opportunity details
POST /api/scan                   # Trigger manual scan
GET  /api/strategies             # Available strategies
```

### Scanner Control
```
GET  /api/scanner/status         # Scanner status and stats
POST /api/scanner/start          # Start scanning
POST /api/scanner/pause          # Pause scanning
POST /api/scanner/configure      # Update scanner settings
```

### Paper Trading
```
POST /api/simulation/accounts              # Create paper account
POST /api/simulation/accounts/{id}/execute # Execute opportunity
GET  /api/simulation/accounts/{id}/performance
GET  /api/simulation/accounts/{id}/equity  # Equity curve data
```

### Auto Trading
```
POST /api/auto-trader/start      # Start (paper/live/shadow)
POST /api/auto-trader/stop       # Stop trading
GET  /api/auto-trader/status     # Current status
POST /api/auto-trader/configure  # Update config
POST /api/auto-trader/emergency-stop  # Kill switch
```

### Live Trading
```
POST /api/trading/orders         # Place order
GET  /api/trading/positions      # Open positions
POST /api/trading/execute-opportunity
```

### Copy Trading
```
POST /api/copy-trading/configs              # Create config
POST /api/copy-trading/configs/{id}/enable  # Enable config
GET  /api/copy-trading/configs              # List configs
```

### Anomaly Detection
```
GET  /api/anomaly/analyze/{address}    # Analyze a wallet
POST /api/anomaly/find-profitable      # Find profitable wallets
```

### Health & Monitoring
```
GET  /health              # Basic health check
GET  /health/live         # Liveness probe
GET  /health/ready        # Readiness probe
GET  /health/detailed     # Full system diagnostics
GET  /metrics             # Prometheus metrics
```

### WebSocket
```
WS /ws    # Real-time opportunity updates
```

Interactive API documentation at `http://localhost:8000/docs` when running.

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SCAN_INTERVAL_SECONDS` | How often to scan (seconds) | `60` |
| `MIN_PROFIT_THRESHOLD` | Minimum profit to surface | `0.025` (2.5%) |
| `MAX_MARKETS_TO_SCAN` | Markets analyzed per cycle | `500` |
| `MIN_LIQUIDITY` | Minimum market liquidity (USD) | `1000` |
| `DATABASE_URL` | Database connection string | `sqlite+aiosqlite:///./data/arbitrage.db` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `TRADING_ENABLED` | Enable live trading | `false` |
| `MAX_TRADE_SIZE_USD` | Max single trade size | `100` |
| `MAX_DAILY_TRADE_VOLUME` | Max daily volume | `1000` |
| `MAX_OPEN_POSITIONS` | Max concurrent positions | `10` |

See `.env.example` for all options including Telegram, wallet tracking, and API credentials.

---

## Architecture

```
backend/
├── main.py                      # FastAPI entry point, health checks, metrics
├── config.py                    # Pydantic settings with env overrides
├── api/
│   ├── routes.py                # Opportunities, scanner, strategies
│   ├── routes_auto_trader.py    # Auto trading control
│   ├── routes_simulation.py     # Paper trading
│   ├── routes_trading.py        # Live trading
│   ├── routes_copy_trading.py   # Copy trading
│   ├── routes_anomaly.py        # Wallet anomaly detection
│   ├── routes_settings.py       # Settings management
│   ├── routes_maintenance.py    # Database maintenance
│   └── websocket.py             # WebSocket handlers
├── services/
│   ├── scanner.py               # Main arbitrage detection orchestrator
│   ├── auto_trader.py           # Autonomous trade execution engine
│   ├── trading.py               # CLOB API integration
│   ├── simulation.py            # Paper trading service
│   ├── copy_trader.py           # Copy trading logic
│   ├── wallet_tracker.py        # Wallet monitoring
│   ├── anomaly_detector.py      # Wallet analysis
│   ├── notifier.py              # Telegram notifications
│   ├── opportunity_recorder.py  # Track opportunity outcomes
│   ├── maintenance.py           # Database cleanup
│   ├── decay_analyzer.py        # Opportunity decay tracking
│   ├── ml_classifier.py         # ML-based opportunity ranking
│   ├── param_optimizer.py       # Auto-tune parameters
│   ├── kalshi_client.py         # Kalshi API integration
│   ├── cross_platform_scanner.py # Cross-market arbitrage
│   ├── polymarket.py            # Polymarket API client
│   ├── strategies/
│   │   ├── base.py              # Base strategy class
│   │   ├── basic.py             # Strategy 1: Basic Arbitrage
│   │   ├── mutually_exclusive.py # Strategy 2: Mutually Exclusive
│   │   ├── contradiction.py     # Strategy 3: Contradiction
│   │   ├── negrisk.py           # Strategy 4: NegRisk (most profitable)
│   │   ├── must_happen.py       # Strategy 5: Must-Happen
│   │   ├── miracle.py           # Strategy 6: Miracle Scanner
│   │   ├── combinatorial.py     # Strategy 7: Cross-market Integer Programming
│   │   └── settlement_lag.py    # Strategy 8: Settlement Lag
│   └── optimization/
│       ├── frank_wolfe.py       # Frank-Wolfe algorithm
│       ├── constraint_solver.py # Constraint satisfaction
│       ├── dependency_detector.py # Market dependency detection
│       ├── bregman.py           # Bregman divergence optimization
│       ├── parallel_executor.py # Multi-threaded execution
│       └── vwap.py              # Volume-weighted average price
├── models/
│   ├── opportunity.py           # ArbitrageOpportunity, StrategyType
│   ├── market.py                # Market, Event, Token models
│   └── database.py              # SQLAlchemy ORM models
└── utils/
    ├── logger.py                # JSON structured logging
    ├── rate_limiter.py          # API rate limiting
    ├── retry.py                 # Exponential backoff
    └── validation.py            # Input validation

frontend/
├── src/
│   ├── App.tsx                  # Main dashboard
│   ├── components/
│   │   ├── OpportunityCard.tsx      # Opportunity display
│   │   ├── TradeExecutionModal.tsx  # Trade execution UI
│   │   ├── SimulationPanel.tsx      # Paper trading interface
│   │   ├── TradingPanel.tsx         # Auto/live trading controls
│   │   ├── PositionsPanel.tsx       # Open positions
│   │   ├── PerformancePanel.tsx     # P&L analytics & equity curves
│   │   ├── RecentTradesPanel.tsx    # Trade history
│   │   ├── WalletTracker.tsx        # Wallet monitoring
│   │   ├── WalletAnalysisPanel.tsx  # Deep wallet analysis
│   │   ├── AnomalyPanel.tsx         # Suspicious activity alerts
│   │   └── SettingsPanel.tsx        # Configuration UI
│   ├── services/
│   │   └── api.ts               # Typed API client
│   └── hooks/
│       └── useWebSocket.ts      # Real-time WebSocket hook
├── package.json
└── vite.config.ts               # Build config with API proxy
```

### Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.10+ / FastAPI / Uvicorn |
| Frontend | React 18 / TypeScript / Vite / TailwindCSS |
| Database | SQLite (dev) / PostgreSQL (production) |
| ORM | SQLAlchemy 2.0 with async support |
| Real-time | WebSockets |
| Data Fetching | React Query / Axios / httpx |
| Trading | py-clob-client / web3.py / eth-account |
| Notifications | Telegram Bot API |
| CI/CD | GitHub Actions (Ruff lint + TypeScript checks + build) |
| Deployment | Docker / Docker Compose |

---

## Risk Disclosure

**This is not risk-free.** Understand before trading real money:

- **Fees**: Polymarket charges 2% on winnings. You need >2.5% spread to profit after fees.
- **Resolution Risk**: Markets can resolve unexpectedly or ambiguously.
- **Liquidity Risk**: You may not be able to exit at expected prices.
- **Oracle Risk**: Subjective resolution criteria can go against you.
- **Timing Risk**: Opportunities can close before your order executes.
- **Platform Risk**: Polymarket can change rules, fees, or delist markets.

**Start with paper trading. Never risk money you can't afford to lose.**

---

## Contributing

PRs welcome. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full guide covering:

- Development setup (backend + frontend)
- Code standards (Ruff for Python, TypeScript strict mode)
- How to add strategies, endpoints, and components
- PR checklist and CI requirements

Quick version:
1. Fork the repo
2. Create a feature branch
3. Run `make dev` to test locally
4. Submit a pull request

---

## Security

Found a vulnerability? **Do not open a public issue.** See **[SECURITY.md](SECURITY.md)** for responsible disclosure instructions.

---

## License

MIT License. See **[LICENSE](LICENSE)** for the full text.
