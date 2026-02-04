<p align="center">
  <img src="homerun-logo.png" alt="Homerun Logo" width="200"/>
</p>

<h1 align="center">Homerun</h1>

<p align="center">
  Autonomous arbitrage scanner and trading bot for Polymarket prediction markets.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#the-strategies">Strategies</a> •
  <a href="#features">Features</a> •
  <a href="#autonomous-trading">Auto Trading</a> •
  <a href="#api">API</a>
</p>

---

## Why Homerun?

Prediction markets have pricing inefficiencies. When the same event trades at different prices, or when mutually exclusive outcomes sum to more than 100%, there's an arbitrage opportunity.

Homerun finds these opportunities in real-time and can execute trades automatically.

---

## The Strategies

### 1. Basic Arbitrage
Buy YES + NO on the same market when total cost < $1.00. Guaranteed profit—one side must win.

### 2. Mutually Exclusive Arbitrage
Two events where only one can happen (Democrat vs Republican wins). Buy YES on both for < $1.00.

### 3. Contradiction Arbitrage
Markets saying opposite things. Buy YES in one, NO in the other when prices contradict.

### 4. NegRisk Arbitrage
Same event with multiple date cutoffs. Buy NO on all dates when total < $1.00.

### 5. Must-Happen Arbitrage
Exhaustive outcomes where one must win. Buy YES on all when total < $1.00.

### 6. Miracle Scanner
Bet NO on events that will almost certainly never happen:
- Aliens landing on Earth
- WW3 starting by Friday
- Bitcoin hitting $1M this week

---

## Quick Start

```bash
git clone https://github.com/braedonsaunders/homerun.git
cd homerun
./setup.sh && ./run.sh
```

Open **http://localhost:3000**

That's it. You're scanning for arbitrage.

### Alternative Setup Methods

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
cd backend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && uvicorn main:app --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm run dev
```

### Requirements
- Python 3.10+
- Node.js 18+
- Docker (optional)

---

## Features

### Core
- **6 Arbitrage Strategies** — All the documented profitable approaches
- **Real-time Scanning** — WebSocket-powered live market data
- **Autonomous Trading** — Paper, shadow, or live mode
- **Copy Trading** — Auto-copy trades from profitable wallets
- **Anomaly Detection** — Find wallets with suspicious win rates
- **Direct CLOB Integration** — Execute on Polymarket's orderbook

### Production-Ready
- **Rate Limiting** — Respects Polymarket API limits
- **Circuit Breakers** — Auto-pause after consecutive losses
- **Emergency Stop** — Kill switch for all orders
- **Structured Logging** — JSON logs for debugging
- **Health Checks** — Kubernetes-compatible probes
- **Prometheus Metrics** — Monitor everything

### Safety
- **Paper Trading** — Practice with virtual money first
- **Shadow Mode** — See what would trade without executing
- **Daily Limits** — Cap trades and losses per day
- **Position Sizing** — Fixed, Kelly criterion, or volatility-adjusted

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
max_risk_score: 0.5           # Maximum acceptable risk
base_position_size_usd: 10    # Default position size
max_position_size_usd: 100    # Maximum per trade
max_daily_trades: 50          # Daily trade limit
max_daily_loss_usd: 100       # Stop if exceeded
```

### Live Trading Setup

1. Get API credentials from [Polymarket Settings](https://polymarket.com/settings/api-keys)
2. Add to `.env`:
```bash
TRADING_ENABLED=true
POLYMARKET_PRIVATE_KEY=your_key
POLYMARKET_API_KEY=your_api_key
POLYMARKET_API_SECRET=your_secret
POLYMARKET_API_PASSPHRASE=your_passphrase
```
3. Start in paper mode, verify it works
4. Enable live via the UI or API

---

## Anomaly Detection

Find wallets doing the impossible:

| Anomaly | What It Means |
|---------|---------------|
| `impossible_win_rate` | >95% win rate over many trades |
| `unusual_roi` | Returns way above market average |
| `perfect_timing` | Always buys lows, sells highs |
| `statistically_impossible` | Zero losses across 100+ trades |
| `wash_trading` | Rapid buy/sell in same market |
| `front_running` | Suspicious timing before price moves |
| `arbitrage_only` | Only executes arb trades (bot) |

Use this to find wallets worth copy-trading—or to identify potential manipulation.

---

## API

### Opportunities
```
GET  /api/opportunities       # Current arbitrage opportunities
POST /api/scan               # Trigger manual scan
GET  /api/strategies         # Available strategies
```

### Paper Trading
```
POST /api/simulation/accounts              # Create account
POST /api/simulation/accounts/{id}/execute # Execute opportunity
GET  /api/simulation/accounts/{id}/performance
```

### Auto Trading
```
POST /api/auto-trader/start   # Start (paper/live/shadow)
POST /api/auto-trader/stop    # Stop trading
GET  /api/auto-trader/status  # Current status
POST /api/auto-trader/emergency-stop
```

### Real Trading
```
POST /api/trading/orders      # Place order
GET  /api/trading/positions   # Open positions
POST /api/trading/execute-opportunity
```

### Copy Trading
```
POST /api/copy-trading/configs            # Create config
POST /api/copy-trading/configs/{id}/enable
```

### Anomaly Detection
```
GET  /api/anomaly/analyze/{address}
POST /api/anomaly/find-profitable
```

### WebSocket
```
WS /ws    # Real-time opportunity updates
```

Full API documentation available at `http://localhost:8000/docs` when running.

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SCAN_INTERVAL_SECONDS` | Scan frequency | 60 |
| `MIN_PROFIT_THRESHOLD` | Minimum profit % | 0.025 (2.5%) |
| `MAX_MARKETS_TO_SCAN` | Markets to analyze | 500 |
| `MIN_LIQUIDITY` | Liquidity filter | 1000 |
| `MAX_TRADE_SIZE_USD` | Max single trade | 100 |
| `MAX_DAILY_TRADE_VOLUME` | Max daily volume | 1000 |

See `.env.example` for all options.

---

## Architecture

```
backend/
├── main.py                 # FastAPI entry
├── services/
│   ├── scanner.py          # Arbitrage detection
│   ├── auto_trader.py      # Autonomous execution
│   ├── trading.py          # CLOB API integration
│   ├── copy_trader.py      # Copy trading logic
│   ├── anomaly_detector.py # Wallet analysis
│   └── strategies/         # 6 arbitrage strategies
└── api/                    # REST endpoints

frontend/
├── src/
│   ├── App.tsx             # Main dashboard
│   ├── components/         # UI components
│   └── hooks/useWebSocket.ts
```

---

## Risk Disclosure

**This is not risk-free.** Understand before trading:

- **Fees**: Polymarket charges 2% on winnings. Need >2.5% spread to profit.
- **Resolution Risk**: Markets can resolve unexpectedly.
- **Liquidity Risk**: May not exit at expected prices.
- **Oracle Risk**: Subjective resolution criteria.
- **Timing Risk**: Opportunities close before execution.

Start with paper trading. Never risk money you can't afford to lose.

---

## Contributing

PRs welcome. Please:
1. Fork the repo
2. Create a feature branch
3. Submit a pull request

---

## License

MIT

