<p align="center">
  <img src="homerun-logo.png" alt="Homerun" width="180"/>
</p>

<h1 align="center">Homerun</h1>

<p align="center">
  <strong>$40M+ in arbitrage has been extracted from Polymarket.<br/>This is the open-source bot that finds those trades.</strong>
</p>

<p align="center">
  <a href="https://github.com/braedonsaunders/homerun/actions/workflows/sloppy.yml"><img src="https://github.com/braedonsaunders/homerun/actions/workflows/sloppy.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/braedonsaunders/homerun/blob/main/LICENSE"><img src="https://img.shields.io/github/license/braedonsaunders/homerun?style=flat" alt="License"></a>
  <a href="https://github.com/braedonsaunders/homerun/stargazers"><img src="https://img.shields.io/github/stars/braedonsaunders/homerun?style=flat" alt="Stars"></a>
  <a href="https://github.com/braedonsaunders/homerun/issues"><img src="https://img.shields.io/github/issues/braedonsaunders/homerun?style=flat" alt="Issues"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#18-arbitrage-strategies">Strategies</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#features">Features</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#autonomous-trading">Trading</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#copy-trading--wallet-intelligence">Copy Trading</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#api-reference">API</a>
</p>

<br/>

<p align="center">
  <img src="screenshot.png" alt="Homerun Dashboard" width="800"/>
</p>

<br/>

## The Problem

Prediction markets misprice things constantly. Market makers optimize for speed over accuracy, creating windows where:

- Mutually exclusive outcomes sum to **less than $1.00** (free money)
- The same event trades at **different prices** across markets
- Prices **don't update for hours** after outcomes are effectively decided

These windows last seconds to minutes. You'd need to monitor hundreds of markets simultaneously to catch them. That's what Homerun does — and it can trade them automatically.

<br/>

## The Numbers

| Metric | Value |
|--------|-------|
| Historical arbitrage extracted (NegRisk) | **$28.99M** from 662 markets |
| Historical arbitrage extracted (Basic) | **$10.58M** from 7,051 conditions |
| Capital efficiency (NegRisk vs Basic) | **29x** |
| Arbitrage strategies | **18** |
| Lines of code | **38,000+** |
| API endpoints | **50+** |
| Database tables | **25+** |

> Based on [published research](https://arxiv.org/abs/2503.18773) (Kroer et al.) analyzing real Polymarket data.

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

**Windows (PowerShell):**
```powershell
.\setup.ps1
.\run.ps1
```

See [WINDOWS.md](WINDOWS.md) for full Windows instructions.

</details>

<details>
<summary><strong>Requirements</strong></summary>
<br/>

- Python 3.10+ (3.11 recommended)
- Node.js 18+
- Docker (optional)

</details>

| Service | URL |
|---------|-----|
| Dashboard | `http://localhost:3000` |
| API | `http://localhost:8000` |
| Swagger Docs | `http://localhost:8000/docs` |
| WebSocket | `ws://localhost:8000/ws` |
| Prometheus Metrics | `http://localhost:8000/metrics` |

<br/>

## 18 Arbitrage Strategies

Most bots run one strategy. Homerun runs 18.

### Core Strategies

| # | Strategy | How It Works | Edge |
|---|----------|-------------|------|
| 1 | **Basic Arbitrage** | Buy YES + NO on same market when sum < $1.00 | $10.58M historical profit |
| 2 | **NegRisk** | Buy YES on all outcomes in mutually exclusive events | **$28.99M profit, 29x capital efficiency** |
| 3 | **Mutually Exclusive** | Two events where only one can happen — buy both | Cross-event mispricing |
| 4 | **Contradiction** | Markets saying opposite things — buy both sides | Logical inconsistency |
| 5 | **Must-Happen** | Exhaustive outcomes where one must occur | Probability sum < 100% |
| 6 | **Miracle Scanner** | Bet NO on near-impossible events (aliens, WW3 by Friday) | Garbage collection for free money |
| 7 | **Combinatorial** | Integer programming across multiple markets | Multi-leg guaranteed profit |
| 8 | **Settlement Lag** | Trade after outcome is known but prices haven't updated | Hours-long windows |

### Advanced Strategies

| # | Strategy | How It Works |
|---|----------|-------------|
| 9 | **BTC/ETH High-Frequency** | 15min/1hr binary crypto market arbitrage |
| 10 | **Cross-Platform** | Arbitrage between Polymarket and Kalshi |
| 11 | **Bayesian Cascade** | Belief propagation across correlated market graphs |
| 12 | **Liquidity Vacuum** | Order book imbalance exploitation |
| 13 | **Entropy Arbitrage** | Information-theoretic mispricing detection |
| 14 | **Event-Driven** | Price lag after news catalysts |
| 15 | **Temporal Decay** | Time-decay mispricing near deadlines |
| 16 | **Correlation Arbitrage** | Mean-reversion on correlated pair spreads |
| 17 | **Market Making** | Earn bid-ask spread as liquidity provider |
| 18 | **Statistical Arbitrage** | Ensemble probability signals and pattern matching |

<br/>

## Features

### Autonomous Trading

Four modes, one system:

| Mode | What It Does |
|------|-------------|
| **Paper** | Virtual $10k account — learn risk-free |
| **Shadow** | Tracks every trade without executing — full backtest |
| **Live** | Real money on Polymarket's CLOB |
| **Mock** | Full pipeline with simulated execution |

Trading infrastructure includes direct CLOB integration, Kelly Criterion position sizing, VWAP execution, circuit breakers, emergency kill switch, daily loss limits, price chasing, and configurable order types (GTC, FOK, GTD).

### Copy Trading & Wallet Intelligence

Don't build a strategy — copy someone who already has one.

- **Full copy trading** — mirror all trades from profitable wallets (proportional sizing)
- **Arb-only copy** — only replicate trades matching detected arbitrage
- **Whale filtering** — ignore noise, only copy trades above configurable thresholds
- **Wallet discovery engine** — background scan to find wallets with proven track records
- **Wallet intelligence** — analyze trading patterns, detect strategy types
- **Real-time monitoring** — WebSocket-powered wallet activity tracking

### AI Intelligence Layer

LLM-powered analysis that goes beyond math:

- **ReAct agent loop** — reasoning + tool use for market analysis
- **Opportunity scoring** — profit viability, resolution safety, execution feasibility
- **Resolution analysis** — detect ambiguity and dispute risk in market criteria
- **5 built-in skills** — wallet evaluation, arbitrage assessment, miracle validation, market correlation, resolution analysis
- **Multi-provider** — plug in OpenAI, Claude, or any compatible LLM

### Anomaly Detection

Find wallets doing the statistically impossible:

| Anomaly | What It Means |
|---------|---------------|
| `impossible_win_rate` | >95% win rate over many trades |
| `perfect_timing` | Consistently buys lows, sells highs |
| `statistically_impossible` | Zero losses across 100+ trades |
| `wash_trading` | Rapid buy/sell cycles in same market |
| `front_running` | Suspicious timing before price moves |
| `coordinated_trading` | Multiple wallets trading in lockstep |
| `insider_pattern` | Timing aligned with external info release |

Use this to find wallets worth copy-trading — or to identify manipulation.

### Portfolio Optimization

- **Frank-Wolfe algorithm** for portfolio-level position optimization
- **Constraint solver** (integer programming) for optimal cross-market allocations
- **Dependency detection** for hidden market correlations
- **Bregman divergence** for information-theoretic capital allocation
- **Kelly Criterion** with configurable max fraction (default 5% per trade)
- **Parallel execution** for multi-threaded trade placement

### Production Infrastructure

| Feature | Details |
|---------|---------|
| **Rate Limiting** | Respects Polymarket limits (Gamma: 300-4000/10s, CLOB: 500-1500/10s) |
| **Circuit Breakers** | Per-token and portfolio-level trip mechanisms |
| **Health Checks** | Kubernetes-compatible liveness + readiness probes |
| **Prometheus Metrics** | Full system observability at `/metrics` |
| **Structured Logging** | JSON logs for aggregation and analysis |
| **Auto-migration** | Schema updates on startup, zero data loss |
| **Telegram Alerts** | Real-time notifications to your phone |
| **CSV Audit Log** | Append-only trade audit trail |

<br/>

## Autonomous Trading Setup

### Configuration

```yaml
min_roi_percent: 2.5          # Minimum ROI to trigger trade
max_risk_score: 0.5           # Maximum acceptable risk (0-1)
base_position_size_usd: 10    # Default position size
max_position_size_usd: 100    # Maximum per trade
max_daily_trades: 50          # Daily trade limit
max_daily_loss_usd: 100       # Stop-loss if exceeded
```

### Going Live

1. Get API credentials from [Polymarket Settings](https://polymarket.com/settings/api-keys)
2. Copy `.env.example` to `.env` and add your keys:

```bash
TRADING_ENABLED=true
POLYMARKET_PRIVATE_KEY=your_key
POLYMARKET_API_KEY=your_api_key
POLYMARKET_API_SECRET=your_secret
POLYMARKET_API_PASSPHRASE=your_passphrase
```

3. **Start with paper mode first.** Verify it works before risking real money.

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
<summary><strong>Wallet Intelligence</strong></summary>

```
GET  /api/anomaly/analyze/{address}    # Analyze a wallet
POST /api/anomaly/find-profitable      # Find profitable wallets
GET  /api/discovery/leaderboard        # Wallet leaderboard
GET  /api/discovery/wallets/{address}  # Wallet details
```

</details>

<details>
<summary><strong>AI</strong></summary>

```
POST /api/ai/judge                     # AI opportunity scoring
POST /api/ai/analyze                   # Market analysis
GET  /api/ai/skills                    # Available AI skills
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
WS   /ws                  # Real-time updates
```

</details>

Interactive API docs available at `http://localhost:8000/docs` when running.

<br/>

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+ &middot; FastAPI &middot; SQLAlchemy 2.0 (async) |
| **Frontend** | React 18 &middot; TypeScript &middot; Vite &middot; TailwindCSS &middot; shadcn/ui |
| **Database** | SQLite (dev) &middot; PostgreSQL (prod) |
| **Real-time** | WebSockets &middot; React Query |
| **Trading** | Polymarket CLOB &middot; Kalshi API &middot; web3.py |
| **AI** | Multi-provider LLM (OpenAI, Claude) &middot; ReAct agent |
| **Notifications** | Telegram Bot API |
| **Infra** | Docker &middot; GitHub Actions &middot; Prometheus |

<details>
<summary><strong>Architecture</strong></summary>
<br/>

```
backend/
├── main.py                      # FastAPI entry, health checks, metrics
├── config.py                    # 180+ env variables via Pydantic
├── api/                         # REST + WebSocket endpoints (11 route modules)
├── services/
│   ├── scanner.py               # Orchestrates all 18 strategies
│   ├── auto_trader.py           # Autonomous execution engine
│   ├── trading.py               # CLOB API execution
│   ├── copy_trader.py           # Copy trading orchestration
│   ├── wallet_intelligence.py   # Pattern analysis
│   ├── anomaly_detector.py      # Impossible win rate detection
│   ├── strategies/              # 18 strategy implementations
│   ├── optimization/            # Frank-Wolfe, VWAP, constraint solver
│   └── ai/                      # LLM agent, skills, judgment
├── models/                      # 25+ SQLAlchemy tables + Pydantic schemas
└── utils/                       # Logging, rate limiting, retry

frontend/
├── src/
│   ├── App.tsx                  # 8-tab dashboard
│   ├── components/              # 20+ UI components
│   ├── services/                # Typed API client
│   └── hooks/                   # WebSocket, keyboard shortcuts
```

</details>

<br/>

## Risk Disclosure

**This is not risk-free.** Understand before trading real money:

- **Fees** — Polymarket charges 2% on winnings. You need spreads above that to profit.
- **Resolution Risk** — Markets can resolve unexpectedly or ambiguously.
- **Liquidity Risk** — You may not exit at expected prices.
- **Oracle Risk** — Subjective resolution criteria can go against you.
- **Timing Risk** — Opportunities can close before your order fills.
- **Platform Risk** — Rules, fees, or market availability can change.

**Start with paper trading. Never risk money you can't afford to lose.**

<br/>

## Contributing

PRs welcome. See [**CONTRIBUTING.md**](CONTRIBUTING.md) for the full guide.

```bash
git clone https://github.com/braedonsaunders/homerun.git
cd homerun
make setup && make dev
```

<br/>

## Security

Found a vulnerability? **Do not open a public issue.** See [**SECURITY.md**](SECURITY.md) for responsible disclosure.

<br/>

## License

[MIT](LICENSE)
