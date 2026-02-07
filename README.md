<p align="center">
  <img src="homerun-logo.png" alt="Homerun" width="180"/>
</p>

<h1 align="center">Homerun</h1>

<p align="center">
  <strong>Autonomous arbitrage scanner & trading bot for Polymarket prediction markets.</strong>
</p>

<p align="center">
  <a href="https://github.com/braedonsaunders/homerun/actions/workflows/sloppy.yml"><img src="https://github.com/braedonsaunders/homerun/actions/workflows/sloppy.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/braedonsaunders/homerun/blob/main/LICENSE"><img src="https://img.shields.io/github/license/braedonsaunders/homerun?style=flat" alt="License"></a>
  <a href="https://github.com/braedonsaunders/homerun/stargazers"><img src="https://img.shields.io/github/stars/braedonsaunders/homerun?style=flat" alt="Stars"></a>
  <a href="https://github.com/braedonsaunders/homerun/issues"><img src="https://img.shields.io/github/issues/braedonsaunders/homerun?style=flat" alt="Issues"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#strategies">Strategies</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#features">Features</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#autonomous-trading">Trading</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#api-reference">API</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#contributing">Contributing</a>
</p>

<br/>

<p align="center">
  <strong>$40M+ in arbitrage profits have been extracted from Polymarket.</strong><br/>
  <sub>Homerun finds those opportunities in real-time — and can trade them automatically.</sub>
</p>

<br/>

---

<br/>

## Why Homerun?

Prediction markets misprice things. When mutually exclusive outcomes sum to more than 100%, or the same event trades at different prices across markets, there's mathematically guaranteed profit on the table.

Most traders miss these windows — they last seconds to minutes and require monitoring hundreds of markets simultaneously. Homerun does that for you.

- **8 arbitrage strategies** — from basic same-market arb to cross-market integer programming
- **3 trading modes** — paper trade risk-free, shadow trade to backtest, or go live
- **Full-stack dashboard** — React frontend with real-time WebSocket updates
- **Production-grade** — rate limiting, circuit breakers, health checks, Prometheus metrics

<br/>

## Quick Start

```bash
git clone https://github.com/braedonsaunders/homerun.git
cd homerun
./setup.sh && ./run.sh
```

Open **http://localhost:3000** — you're scanning for arbitrage.

<details>
<summary><strong>Other setup methods</strong></summary>
<br/>

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

</details>

<details>
<summary><strong>Requirements</strong></summary>
<br/>

- Python 3.10+ (3.11 recommended)
- Node.js 18+
- Docker (optional)

</details>

### Endpoints

| Service | URL |
|---------|-----|
| Dashboard | `http://localhost:3000` |
| Backend API | `http://localhost:8000` |
| Swagger Docs | `http://localhost:8000/docs` |
| WebSocket | `ws://localhost:8000/ws` |
| Health Check | `http://localhost:8000/health` |

<br/>

## Strategies

Homerun runs 8 distinct strategies that target different types of mispricings.

### 1. Basic Arbitrage
Buy YES + NO on the same binary market when total cost < $1.00. One side must win. Guaranteed profit.

> **$10.58M** extracted historically from 7,051 exploitable conditions

### 2. Mutually Exclusive
Two events where only one can happen (e.g., Democrat vs Republican wins). Buy YES on both for < $1.00.

### 3. Contradiction
Markets saying opposite things. Buy YES on both when prices don't add up.

### 4. NegRisk (Most Profitable)
One-of-many markets where all outcomes sum to $1.00. Buy positions across all outcomes when total cost is under $1.00. Covers date-sweeps, exhaustive events, and NegRisk-flagged groups.

> **$28.99M** extracted from 662 markets — **29x more capital-efficient** than basic arb

### 5. Must-Happen
Exhaustive outcomes where one *must* win. Buy YES on all when total < $1.00.

### 6. Miracle Scanner
Bet NO on events with near-zero probability — aliens making contact, WW3 starting by Friday, Bitcoin hitting $10M this week. Almost free money, tiny tail risk.

### 7. Combinatorial
Cross-market arbitrage using integer programming. Finds multi-leg opportunities spanning different event groups where combined positions create guaranteed profit.

### 8. Settlement Lag
Exploits delayed price updates after an outcome is effectively determined but markets haven't adjusted.

<br/>

## Features

### Core

| Feature | Description |
|---------|-------------|
| **Real-time Scanning** | WebSocket-powered live market monitoring across hundreds of markets |
| **Autonomous Trading** | Paper, shadow, or live mode with configurable strategies |
| **Copy Trading** | Auto-replicate trades from profitable wallets |
| **Anomaly Detection** | Flag wallets with suspicious win rates, wash trading, or front-running |
| **Wallet Tracking** | Monitor any wallet's positions and trades in real-time |
| **Cross-platform** | Kalshi integration for cross-market arbitrage |
| **ML Ranking** | Machine learning classifier to score and prioritize opportunities |
| **Telegram Alerts** | Real-time notifications on your phone |

### Trading Infrastructure

| Feature | Description |
|---------|-------------|
| **CLOB Integration** | Direct execution on Polymarket's orderbook |
| **Position Sizing** | Fixed, Kelly criterion, or volatility-adjusted |
| **Circuit Breakers** | Auto-pause after consecutive losses |
| **Emergency Stop** | Kill switch for all open orders |
| **Daily Limits** | Cap trades, volume, and losses per day |
| **VWAP Execution** | Volume-weighted average price for better fills |

<details>
<summary><strong>Advanced Optimization</strong></summary>
<br/>

| Feature | Description |
|---------|-------------|
| **Frank-Wolfe Algorithm** | Portfolio-level position optimization |
| **Constraint Solver** | Optimal allocations across correlated markets |
| **Dependency Detection** | Identify hidden correlations between markets |
| **Bregman Divergence** | Information-theoretic optimization |
| **Parallel Execution** | Multi-threaded trade execution |
| **Decay Analysis** | Track how fast opportunities disappear |
| **Parameter Optimization** | Auto-tune scanner and trading parameters |

</details>

<details>
<summary><strong>Production-Ready Infrastructure</strong></summary>
<br/>

| Feature | Description |
|---------|-------------|
| **Rate Limiting** | Respects Polymarket API limits |
| **Structured Logging** | JSON logs for debugging and aggregation |
| **Health Checks** | Kubernetes-compatible liveness/readiness probes |
| **Prometheus Metrics** | Monitor everything at `/metrics` |
| **Database Maintenance** | Automatic cleanup of stale data |
| **Auto-migration** | Schema updates on startup without data loss |

</details>

<br/>

## Autonomous Trading

Three modes, one system:

| Mode | Description |
|------|-------------|
| **Paper** | Virtual money. Learn the system risk-free. |
| **Shadow** | Tracks trades without executing. Backtest your strategy. |
| **Live** | Real money. Real profits. Real risk. |

### Configuration

```yaml
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

3. **Start with paper mode first.** Verify it works before going live.

<br/>

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

<br/>

## API Reference

<details>
<summary><strong>Opportunities</strong></summary>

```
GET  /api/opportunities          # Current arbitrage opportunities (with filtering)
GET  /api/opportunities/{id}     # Specific opportunity details
POST /api/scan                   # Trigger manual scan
GET  /api/strategies             # Available strategies
```

</details>

<details>
<summary><strong>Scanner Control</strong></summary>

```
GET  /api/scanner/status         # Scanner status and stats
POST /api/scanner/start          # Start scanning
POST /api/scanner/pause          # Pause scanning
POST /api/scanner/configure      # Update scanner settings
```

</details>

<details>
<summary><strong>Paper Trading</strong></summary>

```
POST /api/simulation/accounts              # Create paper account
POST /api/simulation/accounts/{id}/execute # Execute opportunity
GET  /api/simulation/accounts/{id}/performance
GET  /api/simulation/accounts/{id}/equity  # Equity curve data
```

</details>

<details>
<summary><strong>Auto Trading</strong></summary>

```
POST /api/auto-trader/start      # Start (paper/live/shadow)
POST /api/auto-trader/stop       # Stop trading
GET  /api/auto-trader/status     # Current status
POST /api/auto-trader/configure  # Update config
POST /api/auto-trader/emergency-stop  # Kill switch
```

</details>

<details>
<summary><strong>Live Trading</strong></summary>

```
POST /api/trading/orders         # Place order
GET  /api/trading/positions      # Open positions
POST /api/trading/execute-opportunity
```

</details>

<details>
<summary><strong>Copy Trading</strong></summary>

```
POST /api/copy-trading/configs              # Create config
POST /api/copy-trading/configs/{id}/enable  # Enable config
GET  /api/copy-trading/configs              # List configs
```

</details>

<details>
<summary><strong>Anomaly Detection</strong></summary>

```
GET  /api/anomaly/analyze/{address}    # Analyze a wallet
POST /api/anomaly/find-profitable      # Find profitable wallets
```

</details>

<details>
<summary><strong>Health & Monitoring</strong></summary>

```
GET  /health              # Basic health check
GET  /health/live         # Liveness probe
GET  /health/ready        # Readiness probe
GET  /health/detailed     # Full system diagnostics
GET  /metrics             # Prometheus metrics
WS   /ws                  # Real-time opportunity updates
```

</details>

Interactive API docs available at `http://localhost:8000/docs` when running.

<br/>

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SCAN_INTERVAL_SECONDS` | How often to scan | `60` |
| `MIN_PROFIT_THRESHOLD` | Minimum profit to surface | `0.025` (2.5%) |
| `MAX_MARKETS_TO_SCAN` | Markets analyzed per cycle | `500` |
| `MIN_LIQUIDITY` | Minimum market liquidity (USD) | `1000` |
| `DATABASE_URL` | Database connection string | SQLite (dev) |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `TRADING_ENABLED` | Enable live trading | `false` |
| `MAX_TRADE_SIZE_USD` | Max single trade size | `100` |
| `MAX_DAILY_TRADE_VOLUME` | Max daily volume | `1000` |
| `MAX_OPEN_POSITIONS` | Max concurrent positions | `10` |

See [`.env.example`](.env.example) for all options including Telegram, wallet tracking, and API credentials.

<br/>

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+ &middot; FastAPI &middot; Uvicorn |
| **Frontend** | React 18 &middot; TypeScript &middot; Vite &middot; TailwindCSS |
| **Database** | SQLite (dev) &middot; PostgreSQL (prod) |
| **ORM** | SQLAlchemy 2.0 (async) |
| **Real-time** | WebSockets |
| **Data** | React Query &middot; Axios &middot; httpx |
| **Trading** | py-clob-client &middot; web3.py &middot; eth-account |
| **Notifications** | Telegram Bot API |
| **CI/CD** | GitHub Actions &middot; Ruff &middot; TypeScript strict &middot; [Sloppy](https://github.com/braedonsaunders/sloppy) |
| **Deployment** | Docker &middot; Docker Compose |

<details>
<summary><strong>Architecture</strong></summary>
<br/>

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
│   ├── scanner.py               # Arbitrage detection orchestrator
│   ├── auto_trader.py           # Autonomous trade execution
│   ├── trading.py               # CLOB API integration
│   ├── simulation.py            # Paper trading service
│   ├── copy_trader.py           # Copy trading logic
│   ├── wallet_tracker.py        # Wallet monitoring
│   ├── anomaly_detector.py      # Wallet analysis
│   ├── notifier.py              # Telegram notifications
│   ├── strategies/              # 8 arbitrage strategy implementations
│   └── optimization/            # Frank-Wolfe, VWAP, constraint solving
├── models/                      # SQLAlchemy ORM + Pydantic schemas
└── utils/                       # Logging, rate limiting, retry, validation

frontend/
├── src/
│   ├── App.tsx                  # Main dashboard
│   ├── components/              # 11 panel components
│   ├── services/api.ts          # Typed API client
│   └── hooks/useWebSocket.ts    # Real-time updates
└── vite.config.ts               # Build config with API proxy
```

</details>

<br/>

## Risk Disclosure

**This is not risk-free.** Understand before trading real money:

- **Fees** — Polymarket charges 2% on winnings. You need >2.5% spread to profit after fees.
- **Resolution Risk** — Markets can resolve unexpectedly or ambiguously.
- **Liquidity Risk** — You may not be able to exit at expected prices.
- **Oracle Risk** — Subjective resolution criteria can go against you.
- **Timing Risk** — Opportunities can close before your order executes.
- **Platform Risk** — Polymarket can change rules, fees, or delist markets.

**Start with paper trading. Never risk money you can't afford to lose.**

<br/>

## Contributing

PRs welcome. See [**CONTRIBUTING.md**](CONTRIBUTING.md) for the full guide covering development setup, code standards, and the PR checklist.

```bash
# Quick start for contributors
git clone https://github.com/braedonsaunders/homerun.git
cd homerun
make setup && make dev
```

<br/>

## Security

Found a vulnerability? **Do not open a public issue.** See [**SECURITY.md**](SECURITY.md) for responsible disclosure instructions.

<br/>

## License

[MIT](LICENSE)
