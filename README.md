<p align="center">
  <img src="homerun-logo.png" alt="Homerun" width="180"/>
</p>

<h1 align="center">Homerun</h1>

<p align="center">
  <strong>Open-source prediction market arbitrage scanner and trading platform.<br/>Clone, run, and start finding mispriced markets in under a minute.</strong>
</p>

<p align="center">
  <a href="https://github.com/braedonsaunders/homerun/actions/workflows/sloppy.yml"><img src="https://github.com/braedonsaunders/homerun/actions/workflows/sloppy.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/braedonsaunders/homerun/blob/main/LICENSE"><img src="https://img.shields.io/github/license/braedonsaunders/homerun?style=flat" alt="License"></a>
  <a href="https://github.com/braedonsaunders/homerun/stargazers"><img src="https://img.shields.io/github/stars/braedonsaunders/homerun?style=flat" alt="Stars"></a>
  <a href="https://github.com/braedonsaunders/homerun/issues"><img src="https://img.shields.io/github/issues/braedonsaunders/homerun?style=flat" alt="Issues"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#what-it-does">What It Does</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#strategies">Strategies</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#dashboard">Dashboard</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#copy-trading--wallet-intelligence">Copy Trading</a>&nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="#api-reference">API</a>
</p>

<br/>

<p align="center">
  <img src="screenshots/screenshot.png" alt="Homerun Dashboard" width="800"/>
</p>

<br/>

## What It Does

Homerun scans Polymarket and Kalshi for arbitrage opportunities — cases where market prices are internally inconsistent or diverge across platforms — and can execute trades automatically.

It ships with a set of built-in detection strategies (basic arbitrage, NegRisk bundles, cross-platform discrepancies, news-driven edges, and more), but strategies are **fully user-definable** — you can write your own in Python and plug them in at runtime.

The scanner runs continuously, pushes results to a real-time dashboard, and supports paper trading out of the box with no API keys required.

<br/>

## Quick Start

```bash
git clone https://github.com/braedonsaunders/homerun.git
cd homerun
./run.sh
```

Open **http://localhost:3000**. The scanner starts immediately with a virtual $10k paper account.

No `.env` file. No database setup. No API keys. SQLite is created automatically on first run.

> **Windows?** Run `.\run.ps1` in PowerShell, or double-click `Homerun.bat`.

<details>
<summary><strong>One-click launchers</strong></summary>
<br/>

- **macOS:** `Homerun.command`
- **Windows:** `Homerun.bat`
- **Linux:** `Homerun.desktop` (run `chmod +x run.sh Homerun.desktop` first)

</details>

<details>
<summary><strong>Requirements</strong></summary>
<br/>

- Python 3.10+ (3.11 recommended)
- Node.js 18+

</details>

| Service | URL |
|---------|-----|
| Dashboard | `http://localhost:3000` |
| API | `http://localhost:8000` |
| Swagger Docs | `http://localhost:8000/docs` |
| WebSocket | `ws://localhost:8000/ws` |
| Prometheus Metrics | `http://localhost:8000/metrics` |

<br/>

## Strategies

Homerun ships with built-in strategies covering the most common prediction market mispricings:

| Strategy | What It Detects |
|----------|----------------|
| **Basic Arbitrage** | YES + NO on the same market sum to less than $1.00 |
| **NegRisk** | Mutually exclusive outcome bundles priced below guaranteed payout |
| **Mutually Exclusive** | Two events where only one can happen, both underpriced |
| **Contradiction** | Markets implying logically inconsistent outcomes |
| **Must-Happen** | Exhaustive outcome sets where probabilities sum below 100% |
| **Miracle Scanner** | Near-impossible events priced too high on YES |
| **Combinatorial** | Multi-leg integer programming across related markets |
| **Settlement Lag** | Markets that haven't repriced after an outcome is already known |
| **BTC/ETH High-Frequency** | Short-window crypto binaries drifting from Chainlink oracle prices |
| **Cross-Platform** | Price discrepancies between Polymarket and Kalshi |
| **News Edge** | LLM reads breaking news, estimates probability, flags the gap |
| **Bayesian Cascade** | Belief propagation across correlated market graphs |
| **Liquidity Vacuum** | Order book imbalance exploitation |
| **Entropy Arbitrage** | Information-theoretic mispricing detection |
| **Event-Driven** | Price lag after news catalysts |
| **Temporal Decay** | Time-decay mispricing near resolution deadlines |
| **Correlation Arbitrage** | Mean-reversion on correlated pair spreads |
| **Market Making** | Earn bid-ask spread as liquidity provider |

### Custom Strategies

Strategies are pluggable. Write a Python class extending `BaseStrategy`, submit it through the UI, and it runs alongside the built-ins every scan cycle:

```python
from services.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    name = "My Custom Strategy"
    description = "What this detects"

    def detect(self, events, markets, prices):
        opportunities = []
        # your detection logic
        return opportunities
```

Plugins are stored in the database and loaded dynamically at runtime.

<br/>

## Dashboard

Eight tabs, each with its own real-time data pipeline via WebSocket:

| Tab | What It Does |
|-----|-------------|
| **Opportunities** | Live arbitrage feed — card, table, terminal, and Polymarket search views |
| **Trading** | Trader orchestrator cockpit + copy trading configuration |
| **Accounts** | Paper and live account management |
| **Traders** | Wallet discovery, tracking, and deep analysis |
| **Positions** | Open positions across all accounts |
| **Performance** | Equity curves, P&L charts, trade history |
| **AI** | LLM copilot, opportunity scoring, resolution analysis |
| **Settings** | 180+ configurable parameters, persisted to database |

### Trading Modes

| Mode | Description |
|------|-------------|
| **Paper** | Virtual $10k account — no API keys needed |
| **Shadow** | Tracks every trade without executing |
| **Live** | Real money on Polymarket CLOB |
| **Mock** | Full pipeline with simulated execution |

Trading infrastructure includes Kelly Criterion sizing, VWAP execution, circuit breakers, emergency kill switch, daily loss limits, maker mode for rebates, and configurable order types (GTC, FOK, GTD, FAK).

<br/>

## Copy Trading & Wallet Intelligence

- **Full copy trading** — mirror trades from profitable wallets with proportional sizing
- **Arb-only copy** — only replicate trades matching detected arbitrage patterns
- **Whale filtering** — ignore noise below configurable thresholds
- **Wallet discovery** — scans the blockchain for wallets with strong track records
- **Anomaly detection** — flags impossible win rates, front-running, wash trading
- **Real-time monitoring** — WebSocket-powered activity tracking

<br/>

## AI Layer

Multi-provider LLM support (OpenAI, Claude, Gemini, Grok, DeepSeek, Ollama, LM Studio):

- **ReAct agent** — reasoning + tool use for market analysis
- **LLM-as-judge** — scores opportunities on profit viability, resolution safety, execution feasibility
- **Resolution analysis** — detects ambiguity and dispute risk in market criteria
- **News Edge** — estimates true probability from breaking news, flags mispriced markets
- **AI Copilot** — context-aware chat for any opportunity, wallet, or market

<br/>

## Data Sources

| Layer | Sources |
|-------|---------|
| **Prediction markets** | Polymarket (Gamma + CLOB + Data APIs), Kalshi |
| **Crypto resolution data** | Polymarket RTDS, Chainlink oracle feeds |
| **News** | Google News RSS, GDELT DOC 2.0, custom RSS feeds |
| **Wallet intelligence** | On-chain activity via Polygon RPC, anomaly filtering |
| **World intelligence** | Military aircraft (OpenSky), maritime vessels (AIS Stream), conflict data (ACLED, UCDP), earthquakes (USGS) |

<br/>

## Going Live

Paper trading works with zero configuration. To trade real money:

1. Get API credentials from [Polymarket Settings](https://polymarket.com/settings/api-keys)
2. Add keys via the **Settings** tab, or set environment variables:

```bash
TRADING_ENABLED=true
POLYMARKET_PRIVATE_KEY=your_key
POLYMARKET_API_KEY=your_api_key
POLYMARKET_API_SECRET=your_secret
POLYMARKET_API_PASSPHRASE=your_passphrase
```

3. **Start with paper mode first.** Verify everything works before risking real money.

<br/>

## API Reference

50+ endpoints. Full interactive docs at `http://localhost:8000/docs` when running.

<details>
<summary><strong>Opportunities</strong></summary>

```
GET  /api/opportunities          # Current arbitrage opportunities
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
<summary><strong>Trader Orchestrator</strong></summary>

```
GET    /api/trader-orchestrator/overview
POST   /api/trader-orchestrator/start
POST   /api/trader-orchestrator/stop
POST   /api/trader-orchestrator/kill-switch
POST   /api/trader-orchestrator/live/preflight
POST   /api/trader-orchestrator/live/arm
POST   /api/trader-orchestrator/live/start
POST   /api/trader-orchestrator/live/stop
GET    /api/traders
POST   /api/traders
GET    /api/traders/{trader_id}
PUT    /api/traders/{trader_id}
DELETE /api/traders/{trader_id}
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
<summary><strong>News & Crypto</strong></summary>

```
GET  /api/news/articles          # Latest matched articles
GET  /api/news/edges             # News-driven opportunities
GET  /api/crypto/markets         # Live crypto binary markets
GET  /api/crypto/prices          # Chainlink oracle prices
```

</details>

<details>
<summary><strong>Settings & Health</strong></summary>

```
GET  /api/settings               # Current settings
PUT  /api/settings               # Update settings
GET  /health                     # Basic health check
GET  /health/live                # Liveness probe
GET  /health/ready               # Readiness probe
GET  /health/detailed            # Full system diagnostics
GET  /metrics                    # Prometheus metrics
WS   /ws                         # Real-time updates
```

</details>

<br/>

## Architecture

```
backend/
├── main.py                      # FastAPI entry, lifespan, health checks, metrics
├── config.py                    # 180+ settings via Pydantic Settings
├── alembic/                     # Versioned schema/data migrations
├── api/                         # REST + WebSocket endpoints
├── services/
│   ├── scanner.py               # Orchestrates all strategies per scan cycle
│   ├── strategies/              # Built-in strategy implementations
│   ├── trader_orchestrator/     # Multi-trader orchestration engine
│   ├── trading.py               # Polymarket CLOB execution
│   ├── copy_trader.py           # Copy trading orchestration
│   ├── crypto_service.py        # Dedicated crypto market pipeline
│   ├── chainlink_feed.py        # Oracle price feeds
│   ├── wallet_intelligence.py   # Pattern analysis
│   ├── anomaly_detector.py      # Statistical anomaly detection
│   ├── ml_classifier.py         # False-positive classifier
│   ├── plugin_loader.py         # Dynamic strategy loading
│   ├── optimization/            # Frank-Wolfe, VWAP, constraint solver
│   ├── news/                    # Feed aggregation, semantic matching, edge detection
│   ├── world_intelligence/      # Geopolitical signal tracking
│   └── ai/                      # LLM agent, skills, copilot, judgment
├── models/
│   ├── database.py              # 25+ SQLAlchemy tables
│   └── market.py                # Market + opportunity schemas
└── workers/                     # Background scan loops (scanner, news, crypto, discovery, etc.)

frontend/
├── src/
│   ├── App.tsx                  # 8-tab dashboard
│   ├── components/              # 50+ components (shadcn/ui + custom)
│   ├── services/api.ts          # Typed API client
│   ├── hooks/                   # WebSocket, keyboard shortcuts
│   └── store/                   # Jotai atoms

tui.py                           # Terminal UI (Textual + Rich)
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+ · FastAPI · SQLAlchemy 2.0 (async) · Pydantic |
| **Frontend** | React 18 · TypeScript · Vite · TailwindCSS · shadcn/ui · Jotai · TanStack Query |
| **Database** | SQLite (zero-config, auto-created) |
| **Real-time** | WebSockets · Framer Motion |
| **Trading** | Polymarket CLOB · Kalshi API · Polygon RPC |
| **AI** | Multi-provider LLM (OpenAI, Claude, Gemini, Grok, DeepSeek, Ollama, LM Studio) · ReAct agent |
| **ML** | Custom logistic regression · FAISS (optional) · sentence-transformers (optional) |
| **Crypto** | Chainlink oracle feeds · Binary market scanner |
| **Infra** | GitHub Actions · Prometheus · Textual TUI · Telegram alerts |

<br/>

## Risk Disclosure

**This is not risk-free.** Understand before trading real money:

- **Fees** — Polymarket charges 2% on winnings. Spreads must exceed that to profit.
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
./setup.sh
./run.sh
```

<br/>

## Security

Found a vulnerability? **Do not open a public issue.** See [**SECURITY.md**](SECURITY.md) for responsible disclosure.

<br/>

## License

[MIT](LICENSE)
