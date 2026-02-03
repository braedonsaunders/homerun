# Polymarket Arbitrage Scanner

A private web-based tool to detect and alert on arbitrage opportunities in Polymarket prediction markets.

Based on documented strategies that have generated **$40M+ in extracted profits** from mathematical pricing inefficiencies.

## The 5 Arbitrage Strategies

### 1. Basic Arbitrage
Buy YES + NO on the same binary market when total cost < $1.00. Guaranteed profit since one must win.

### 2. Mutually Exclusive Arbitrage
Two events where only one can be true (e.g., Democrat vs Republican wins). Buy YES on both for < $1.00.

### 3. Contradiction Arbitrage
Two markets that say opposite things. Buy YES in one, NO in the other when prices contradict.

### 4. NegRisk / One-of-Many Arbitrage (Most Profitable)
Same event with multiple date cutoffs (e.g., "US strikes Iran by March/April/June"). Buy NO on all dates when total < $1.00.

**This is the "$1M in 7 days" strategy.**

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

### Scanner
- `GET /api/scanner/status` - Get scanner status
- `GET /api/strategies` - List available strategies

### Wallets
- `GET /api/wallets` - Get tracked wallets
- `POST /api/wallets?address=0x...&label=name` - Add wallet to track
- `DELETE /api/wallets/{address}` - Remove tracked wallet
- `GET /api/wallets/{address}/positions` - Get wallet positions
- `GET /api/wallets/{address}/trades` - Get wallet trades

### Markets
- `GET /api/markets` - Get Polymarket markets
- `GET /api/events` - Get Polymarket events

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
| `TRACKED_WALLETS` | Comma-separated wallet addresses | |
| `TELEGRAM_BOT_TOKEN` | Telegram bot for notifications | |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | |

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
This tool **detects opportunities only** - manual execution required.
Always verify opportunities before trading.

## Architecture

```
backend/
├── main.py              # FastAPI entry point
├── config.py            # Configuration
├── api/
│   ├── routes.py        # REST endpoints
│   └── websocket.py     # WebSocket handlers
├── services/
│   ├── polymarket.py    # Polymarket API client
│   ├── scanner.py       # Main arbitrage scanner
│   ├── wallet_tracker.py
│   └── strategies/
│       ├── basic.py
│       ├── negrisk.py
│       ├── mutually_exclusive.py
│       ├── contradiction.py
│       └── must_happen.py
└── models/
    ├── market.py
    └── opportunity.py

frontend/
├── src/
│   ├── App.tsx
│   ├── components/
│   │   ├── OpportunityCard.tsx
│   │   └── WalletTracker.tsx
│   ├── hooks/
│   │   └── useWebSocket.ts
│   └── services/
│       └── api.ts
```

## Research Sources

- [Polymarket Official Docs](https://docs.polymarket.com/)
- [py-clob-client GitHub](https://github.com/Polymarket/py-clob-client)
- [BeInCrypto: Arbitrage Bots Making Millions](https://beincrypto.com/arbitrage-bots-polymarket-humans/)
- [ChainCatcher: Silent Profits Through Arbitrage](https://www.chaincatcher.com/en/article/2212288)

## Disclaimer

This tool is for educational and personal use only. Trading on prediction markets involves risk. Do your own research and never invest more than you can afford to lose.
