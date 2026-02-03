# Polymarket Arbitrage Tool - Implementation Plan

## Executive Summary

A private web-based tool to detect and alert on arbitrage opportunities in Polymarket prediction markets. Based on documented strategies that have generated **$40M+ in extracted profits** from mathematical pricing inefficiencies.

## Research Sources

- [Polymarket Official Docs](https://docs.polymarket.com/)
- [py-clob-client GitHub](https://github.com/Polymarket/py-clob-client)
- [Polymarket Arbitrage Bot Reference](https://github.com/runesatsdev/polymarket-arbitrage-bot)
- [BeInCrypto: How Bots Make Millions](https://beincrypto.com/arbitrage-bots-polymarket-humans/)
- [ChainCatcher: Silent Profits Through Arbitrage](https://www.chaincatcher.com/en/article/2212288)

---

## The 5 Arbitrage Strategies

### 1. Basic Arbitrage (Single-Condition)
**Concept:** Buy YES + NO on the same binary market when total < $1.00

```
Example:
- YES price: $0.48
- NO price: $0.48
- Total cost: $0.96
- Guaranteed payout: $1.00
- Profit: $0.04 (4.17% return)
```

**Detection Logic:**
```python
def detect_basic_arb(market):
    yes_price = get_best_ask("YES")
    no_price = get_best_ask("NO")
    total = yes_price + no_price
    fee = 0.02  # 2% winner fee

    if total < (1.0 - fee):
        return {
            "type": "basic",
            "profit": 1.0 - total - fee,
            "roi": (1.0 - total - fee) / total * 100
        }
```

**Historical Performance:** $10.58M from 7,051 exploitable conditions

---

### 2. Mutually Exclusive Arbitrage
**Concept:** Two events where only one can be true, buy YES on both for < $1.00

```
Example:
- "Democrats win 2028" YES: $0.45
- "Republicans win 2028" YES: $0.52
- Total: $0.97
- One MUST win = $1.00 payout
- Profit: $0.03
```

**Detection Logic:**
```python
def detect_mutually_exclusive_arb(event_group):
    """Find events that are logically mutually exclusive"""
    # Requires semantic analysis or manual tagging of related markets
    total_yes = sum(m["yes_price"] for m in event_group)
    if total_yes < 0.98:  # After fees
        return {"profit": 1.0 - total_yes - 0.02}
```

---

### 3. Contradiction Arbitrage
**Concept:** Two markets say opposite things - buy YES in one, NO in the other

```
Example:
- Market A: "BTC above $100K by March" YES: $0.30
- Market B: "BTC below $100K in March" YES: $0.65
- These contradict! Buy YES on A ($0.30) + YES on B ($0.65) = $0.95
- One must be true = $1.00
```

**Detection:** Requires NLP/semantic matching to find contradicting markets

---

### 4. One-of-Many Arbitrage (NegRisk) - MOST PROFITABLE
**Concept:** Same event with multiple date cutoffs. Buy NO on all dates.

```
Example (Iran Strikes):
- "US strikes Iran by March" NO: $0.89
- "US strikes Iran by April" NO: $0.03  (if March NO wins, April also wins)
- "US strikes Iran by June" NO: $0.02
- Total NO cost: $0.94

If NO strike by June: All NO positions pay out
If strike in April: March NO pays, you lose April/June (~$0.05 loss)
If strike in March: You lose March (~$0.89 loss)

Expected value calculation favors NO positions when sum < $1.00
```

**This is the "$1M in 7 days" strategy from anoin123**

**Detection Logic:**
```python
def detect_negrisk_arb(event):
    """NegRisk markets: all outcomes sum to $1"""
    if not event.get("negRisk"):
        return None

    outcomes = event["markets"]
    total = sum(float(m["no_price"]) for m in outcomes)

    # For date-based markets, NO on all dates
    if total < 0.97:
        return {
            "type": "negrisk_date_sweep",
            "profit": 1.0 - total - 0.02,
            "markets": outcomes
        }
```

**Historical Performance:** $28.99M from 662 markets (29x more efficient than basic arb)

---

### 5. Must-Happen Arbitrage
**Concept:** Buy YES on ALL possible outcomes when total < $1.00

```
Example (Multi-outcome market):
- "Winner is Candidate A" YES: $0.30
- "Winner is Candidate B" YES: $0.35
- "Winner is Candidate C" YES: $0.32
- Total: $0.97
- One MUST win = $1.00
- Profit: $0.03
```

**Detection Logic:**
```python
def detect_must_happen_arb(event):
    """All YES positions on exhaustive outcomes"""
    if len(event["markets"]) < 2:
        return None

    total_yes = sum(float(m["yes_price"]) for m in event["markets"])
    if total_yes < 0.98:
        return {"profit": 1.0 - total_yes - 0.02}
```

---

## Technical Architecture

### Stack
```
Backend:  Python 3.11+ / FastAPI
Frontend: React + TypeScript + TailwindCSS
Database: SQLite (local) / PostgreSQL (production)
Realtime: WebSockets (Polymarket + internal)
Notifications: Telegram Bot API (optional)
```

### Project Structure
```
/home/user/homerun/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration management
│   ├── api/
│   │   ├── routes.py           # API endpoints
│   │   └── websocket.py        # WebSocket handlers
│   ├── services/
│   │   ├── polymarket.py       # Polymarket API client
│   │   ├── scanner.py          # Main arbitrage scanner
│   │   ├── strategies/
│   │   │   ├── basic.py        # Strategy 1: Basic Arb
│   │   │   ├── mutually_exclusive.py  # Strategy 2
│   │   │   ├── contradiction.py       # Strategy 3
│   │   │   ├── negrisk.py      # Strategy 4: One-of-Many
│   │   │   └── must_happen.py  # Strategy 5
│   │   ├── wallet_tracker.py   # Track profitable wallets
│   │   └── notifications.py    # Telegram/alerts
│   ├── models/
│   │   ├── market.py           # Market data models
│   │   ├── opportunity.py      # Arbitrage opportunity model
│   │   └── database.py         # DB models
│   └── utils/
│       ├── math.py             # Probability calculations
│       └── logger.py           # Logging setup
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── OpportunityCard.tsx
│   │   │   ├── MarketTable.tsx
│   │   │   ├── WalletTracker.tsx
│   │   │   └── PriceChart.tsx
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts
│   │   └── services/
│   │       └── api.ts
│   ├── package.json
│   └── vite.config.ts
├── requirements.txt
├── package.json
├── docker-compose.yml
└── README.md
```

---

## API Integration

### Polymarket APIs (No Auth Required for Reading)

| API | Base URL | Purpose |
|-----|----------|---------|
| **Gamma** | `https://gamma-api.polymarket.com` | Market metadata, events |
| **CLOB** | `https://clob.polymarket.com` | Order books, prices |
| **Data** | `https://data-api.polymarket.com` | Wallet positions, trades |

### Key Endpoints

```python
# Gamma API - Get all active markets
GET https://gamma-api.polymarket.com/markets?active=true&limit=100&offset=0

# Gamma API - Get events (grouped markets)
GET https://gamma-api.polymarket.com/events?closed=false&limit=100

# CLOB API - Get midpoint price
GET https://clob.polymarket.com/midpoint?token_id={token_id}

# CLOB API - Get order book
GET https://clob.polymarket.com/book?token_id={token_id}

# Data API - Get wallet positions
GET https://data-api.polymarket.com/positions?user={address}

# Data API - Get wallet trades
GET https://data-api.polymarket.com/trades?user={address}&limit=100
```

### WebSocket Real-Time Data
```python
# Connect to market updates
WSS: wss://ws-subscriptions-clob.polymarket.com/ws/market

# Subscribe message
{
    "assets_ids": ["token_id_1", "token_id_2"],
    "type": "market"
}
```

### Rate Limits
| API | Limit |
|-----|-------|
| Gamma (general) | 4,000 req/10s |
| Gamma (markets) | 300 req/10s |
| CLOB | 500-1,500 req/10s |
| Data | 1,000 req/10s |

---

## Core Algorithm

### Main Scanner Loop
```python
class ArbitrageScanner:
    def __init__(self):
        self.gamma_client = GammaAPIClient()
        self.clob_client = CLOBClient()
        self.strategies = [
            BasicArbStrategy(),
            MutuallyExclusiveStrategy(),
            ContradictionStrategy(),
            NegRiskStrategy(),  # Most profitable
            MustHappenStrategy()
        ]

    async def scan(self):
        """Main scanning loop"""
        # 1. Fetch all active events with markets
        events = await self.gamma_client.get_events(closed=False)

        # 2. Get real-time prices for all tokens
        all_tokens = self.extract_token_ids(events)
        prices = await self.clob_client.get_prices_batch(all_tokens)

        # 3. Run each strategy
        opportunities = []
        for strategy in self.strategies:
            opps = strategy.detect(events, prices)
            opportunities.extend(opps)

        # 4. Filter by profitability threshold
        min_profit = 0.02  # 2% minimum after fees
        profitable = [o for o in opportunities if o.roi > min_profit]

        # 5. Score by risk
        scored = self.score_opportunities(profitable)

        return sorted(scored, key=lambda x: x.score, reverse=True)
```

### Risk Scoring
```python
def calculate_risk_score(opportunity):
    score = 0.0

    # Time to resolution
    days_until = (opportunity.resolution_date - now()).days
    if days_until < 2:
        score += 0.4  # Higher risk - less time to exit
    elif days_until < 7:
        score += 0.2

    # Market complexity
    if opportunity.num_outcomes > 5:
        score += 0.2

    # Liquidity (can we actually execute?)
    if opportunity.liquidity < 10000:
        score += 0.3

    # Subjective resolution risk
    if opportunity.is_subjective:
        score += 0.3

    return min(score, 1.0)
```

---

## Wallet Tracking

### Track Profitable Wallets (like anoin123)
```python
class WalletTracker:
    def __init__(self):
        self.tracked_wallets = [
            "0x...",  # anoin123's wallet
        ]

    async def get_wallet_positions(self, address: str):
        """Get current open positions for a wallet"""
        response = await self.data_client.get(
            f"/positions?user={address}"
        )
        return response.json()

    async def get_recent_trades(self, address: str, limit=100):
        """Get recent trades to detect new entries"""
        response = await self.data_client.get(
            f"/trades?user={address}&limit={limit}"
        )
        return response.json()

    async def monitor_wallet_activity(self, address: str):
        """Alert when wallet enters new positions"""
        last_trade_id = None
        while True:
            trades = await self.get_recent_trades(address)
            new_trades = [t for t in trades if t["id"] > last_trade_id]

            for trade in new_trades:
                await self.notify(f"Wallet {address} entered: {trade}")
                last_trade_id = trade["id"]

            await asyncio.sleep(30)  # Check every 30 seconds
```

---

## Frontend Dashboard

### Key Components

1. **Live Opportunities Feed**
   - Real-time list of detected arbitrage opportunities
   - Sorted by profit potential / risk score
   - One-click details expansion

2. **Strategy Breakdown**
   - Filter by strategy type
   - Show historical performance per strategy
   - Enable/disable specific strategies

3. **Market Explorer**
   - Browse all active markets
   - Highlight pricing anomalies
   - Show related markets for manual analysis

4. **Wallet Tracker**
   - Add wallets to track
   - Show their current positions
   - Alert on new trades

5. **Notifications Panel**
   - Configure alert thresholds
   - Telegram integration
   - Sound/browser notifications

---

## Implementation Phases

### Phase 1: Core Scanner (MVP)
- [ ] Project setup (Python backend, React frontend)
- [ ] Gamma API integration (fetch markets/events)
- [ ] CLOB API integration (fetch prices)
- [ ] Basic Arbitrage detection (Strategy 1)
- [ ] NegRisk Arbitrage detection (Strategy 4)
- [ ] Simple CLI output of opportunities

### Phase 2: Web Dashboard
- [ ] FastAPI REST endpoints
- [ ] React dashboard with TailwindCSS
- [ ] Live opportunities table
- [ ] Market detail views
- [ ] WebSocket for real-time updates

### Phase 3: Advanced Strategies
- [ ] Mutually Exclusive detection (Strategy 2)
- [ ] Contradiction detection with NLP (Strategy 3)
- [ ] Must-Happen detection (Strategy 5)
- [ ] Cross-market relationship mapping

### Phase 4: Wallet Tracking & Alerts
- [ ] Wallet position tracking
- [ ] Trade history analysis
- [ ] Telegram bot notifications
- [ ] Configurable alert thresholds

### Phase 5: Optimization
- [ ] Historical backtesting
- [ ] Liquidity analysis
- [ ] Execution simulation
- [ ] Performance dashboards

---

## Key Considerations

### Profitability Thresholds
- **Polymarket Fee:** 2% on winnings
- **Minimum Spread:** Need >2.5% spread to profit after fees
- **Execution Slippage:** Order book depth matters

### Timing
- Opportunities exist for seconds to hours
- Most profitable windows: during news events
- Date-based markets: more stable opportunities

### Risks
1. **Resolution Risk:** Market resolved unexpectedly
2. **Liquidity Risk:** Can't exit position
3. **Oracle Risk:** Subjective resolution criteria
4. **Platform Risk:** Polymarket changes rules

---

## Dependencies

### Python (backend/requirements.txt)
```
fastapi==0.109.0
uvicorn==0.27.0
httpx==0.26.0
websockets==12.0
py-clob-client==0.1.0
pydantic==2.5.0
python-dotenv==1.0.0
sqlalchemy==2.0.25
aiosqlite==0.19.0
```

### Node.js (frontend/package.json)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@tanstack/react-query": "^5.17.0",
    "axios": "^1.6.0",
    "tailwindcss": "^3.4.0",
    "recharts": "^2.10.0",
    "lucide-react": "^0.303.0"
  }
}
```

---

## Quick Start Commands

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

---

## Next Steps

1. **Approve this plan** - confirm architecture decisions
2. **Build Phase 1** - Core scanner with basic + negrisk detection
3. **Iterate** - Add strategies and UI based on results

Ready to proceed with implementation?
