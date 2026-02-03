# Polymarket Arbitrage Scanner

A **production-ready** web-based tool for detecting arbitrage opportunities, paper trading simulation, copy trading, and anomaly detection on Polymarket prediction markets.

Based on documented strategies that have generated **$40M+ in extracted profits** from mathematical pricing inefficiencies.

## Features

- **5 Arbitrage Detection Strategies** - Automatically find profitable opportunities
- **Paper Trading Simulation** - Practice with virtual money, track returns
- **Copy Trading** - Auto-copy trades from profitable wallets
- **Anomaly Detection** - Find wallets with "impossible" trading records
- **Real-time Updates** - WebSocket-powered live data
- **Production Ready** - Logging, rate limiting, error handling, health checks

## The 5 Arbitrage Strategies

### 1. Basic Arbitrage
Buy YES + NO on the same binary market when total cost < $1.00. Guaranteed profit since one must win.

### 2. Mutually Exclusive Arbitrage
Two events where only one can be true (e.g., Democrat vs Republican wins). Buy YES on both for < $1.00.

### 3. Contradiction Arbitrage
Two markets that say opposite things. Buy YES in one, NO in the other when prices contradict.

### 4. NegRisk / One-of-Many Arbitrage (Most Profitable)
Same event with multiple date cutoffs (e.g., "US strikes Iran by March/April/June"). Buy NO on all dates when total < $1.00.

**This is the "$1M in 7 days" strategy used by anoin123.**

### 5. Must-Happen Arbitrage
Multiple exhaustive outcomes where one must win. Buy YES on all when total < $1.00.

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env

# Run the server
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 to view the dashboard.

## API Endpoints

### Opportunities
- `GET /api/opportunities` - Get current arbitrage opportunities
- `POST /api/scan` - Trigger a manual scan
- `GET /api/scanner/status` - Get scanner status
- `GET /api/strategies` - List available strategies

### Paper Trading Simulation
- `POST /api/simulation/accounts` - Create simulation account
- `GET /api/simulation/accounts` - List all accounts
- `GET /api/simulation/accounts/{id}` - Get account details
- `POST /api/simulation/accounts/{id}/execute` - Execute opportunity
- `GET /api/simulation/accounts/{id}/trades` - Get trade history
- `GET /api/simulation/accounts/{id}/performance` - Get performance metrics

### Copy Trading
- `POST /api/copy-trading/configs` - Create copy trading config
- `GET /api/copy-trading/configs` - List all configs
- `POST /api/copy-trading/configs/{id}/enable` - Enable copy trading
- `POST /api/copy-trading/configs/{id}/disable` - Disable copy trading
- `GET /api/copy-trading/status` - Get service status

### Anomaly Detection
- `GET /api/anomaly/analyze/{address}` - Analyze wallet for anomalies
- `POST /api/anomaly/find-profitable` - Find profitable wallets to copy
- `GET /api/anomaly/anomalies` - Get detected anomalies
- `GET /api/anomaly/check/{address}` - Quick suspicious check

### Wallets
- `GET /api/wallets` - Get tracked wallets
- `POST /api/wallets?address=0x...&label=name` - Add wallet to track
- `DELETE /api/wallets/{address}` - Remove tracked wallet

### Health & Monitoring
- `GET /health` - Basic health check
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe
- `GET /health/detailed` - Full system status
- `GET /metrics` - Prometheus metrics

### WebSocket
- `WS /ws` - Real-time updates for opportunities and wallet trades

## Configuration

Edit `.env` or set environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SCAN_INTERVAL_SECONDS` | How often to scan | 60 |
| `MIN_PROFIT_THRESHOLD` | Minimum profit % (decimal) | 0.025 (2.5%) |
| `MAX_MARKETS_TO_SCAN` | Maximum markets to analyze | 500 |
| `MIN_LIQUIDITY` | Minimum liquidity filter | 1000 |
| `LOG_LEVEL` | Logging level | INFO |
| `DATABASE_URL` | SQLite database path | sqlite+aiosqlite:///./arbitrage.db |
| `TRACKED_WALLETS` | Comma-separated wallet addresses | |
| `TELEGRAM_BOT_TOKEN` | Telegram bot for notifications | |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | |

## Production Features

### Structured Logging
- JSON-formatted logs for parsing
- Context-aware logging with request tracking
- Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Rate Limiting
- Token bucket algorithm for Polymarket API limits
- Automatic backoff on 429 responses
- Per-endpoint rate tracking

### Error Handling
- Automatic retry with exponential backoff
- Graceful degradation on API failures
- Global exception handling

### Database Persistence
- SQLite for local development
- PostgreSQL-compatible for production
- Tracks simulation accounts, trades, anomalies

### Health Checks
- Kubernetes-compatible liveness/readiness probes
- Detailed system status endpoint
- Prometheus metrics export

## Architecture

```
backend/
├── main.py              # FastAPI entry point
├── config.py            # Configuration
├── api/
│   ├── routes.py        # Core REST endpoints
│   ├── routes_simulation.py
│   ├── routes_copy_trading.py
│   ├── routes_anomaly.py
│   └── websocket.py
├── services/
│   ├── polymarket.py    # Polymarket API client
│   ├── scanner.py       # Arbitrage scanner
│   ├── simulation.py    # Paper trading
│   ├── copy_trader.py   # Copy trading service
│   ├── anomaly_detector.py
│   ├── wallet_tracker.py
│   └── strategies/      # 5 arbitrage strategies
├── models/
│   ├── database.py      # SQLAlchemy models
│   ├── market.py
│   └── opportunity.py
└── utils/
    ├── logger.py        # Structured logging
    ├── retry.py         # Retry logic
    ├── rate_limiter.py  # Rate limiting
    └── validation.py    # Input validation

frontend/
├── src/
│   ├── App.tsx
│   ├── components/
│   │   ├── OpportunityCard.tsx
│   │   ├── WalletTracker.tsx
│   │   ├── SimulationPanel.tsx
│   │   └── AnomalyPanel.tsx
│   ├── hooks/
│   │   └── useWebSocket.ts
│   └── services/
│       └── api.ts
```

## Anomaly Detection

The system detects these anomaly types:

| Type | Description |
|------|-------------|
| `impossible_win_rate` | Win rate > 95% is statistically impossible |
| `unusual_roi` | Average ROI significantly above normal |
| `perfect_timing` | Always buys at lows, sells at highs |
| `statistically_impossible` | Zero losses over many trades |
| `wash_trading` | Rapid buy/sell in same market |
| `front_running` | Suspicious timing before price moves |
| `arbitrage_only` | Only executes arbitrage (bot indicator) |

## Key Considerations

### Fees
- Polymarket charges **2% on winnings**
- Need >2.5% spread to profit after fees

### Risk Factors
- **Resolution Risk**: Market resolved unexpectedly
- **Liquidity Risk**: Can't exit position at expected price
- **Oracle Risk**: Subjective resolution criteria
- **Timing Risk**: Opportunity closes before execution

### Execution
- Paper trading simulation for practice
- Copy trading requires manual verification
- Always verify opportunities before real trading

## Research Sources

- [Polymarket Official Docs](https://docs.polymarket.com/)
- [py-clob-client GitHub](https://github.com/Polymarket/py-clob-client)
- [BeInCrypto: Arbitrage Bots Making Millions](https://beincrypto.com/arbitrage-bots-polymarket-humans/)
- [ChainCatcher: Silent Profits Through Arbitrage](https://www.chaincatcher.com/en/article/2212288)

## Disclaimer

This tool is for educational and personal use only. Trading on prediction markets involves risk. Do your own research and never invest more than you can afford to lose.
