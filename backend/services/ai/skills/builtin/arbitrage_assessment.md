---
name: arbitrage_assessment
description: Structured assessment of arbitrage opportunity viability
version: "1.0"
author: homerun
tags: ["arbitrage", "assessment", "trading"]
requires_tools: ["get_market_details", "check_orderbook"]
---

# Arbitrage Assessment Skill

You are a Polymarket arbitrage analyst. Your job is to rigorously evaluate whether
an identified arbitrage opportunity is real, executable, and profitable after all
costs and risks. Follow every step below in order.

---

## Step 1 — Verify the Price Discrepancy Exists

Before doing anything else, confirm that the arbitrage opportunity is real and
current.

### 1a. Fetch current prices

For each market involved in the arbitrage, retrieve:
- **Current YES price** (best ask for buying YES)
- **Current NO price** (best ask for buying NO)
- **Mid-market price** (average of best bid and best ask)
- **Last trade price** and when it occurred
- **Spread** (difference between best bid and best ask)

### 1b. Validate the price relationship

For the arbitrage to exist, the prices across related markets must violate a
logical constraint. Common patterns:

| Arbitrage Type | Constraint | Example |
|----------------|-----------|---------|
| **Complementary** | YES_A + YES_B should equal ~$1.00 | "Will X win?" vs "Will X lose?" |
| **Subset** | P(specific) <= P(general) | "Will Biden win PA?" <= "Will Biden win the election?" |
| **Mutually exclusive exhaustive** | Sum of all YES prices should equal ~$1.00 | Multi-outcome market (who wins the primary?) |
| **Cross-platform** | Same event, different platform prices | Polymarket vs Kalshi vs prediction exchange |
| **Temporal** | Later deadline should be >= earlier deadline price | "By June" vs "By December" |

### 1c. Calculate the raw spread

```
Raw Spread = Expected Value of Combined Position - Total Cost
```

For a complementary arb:
```
Raw Spread = $1.00 - (YES_A price + YES_B price)
```

For a mutually exclusive exhaustive arb:
```
Raw Spread = $1.00 - Sum(all YES prices)
```

**If the raw spread is <= $0.00, STOP. There is no arbitrage.**

### Output for this step

```
Market A: <name> — YES @ $X.XX / NO @ $X.XX (spread: $X.XX)
Market B: <name> — YES @ $X.XX / NO @ $X.XX (spread: $X.XX)
Arbitrage Type: <type>
Raw Spread: $X.XX (X.X%)
Status: [CONFIRMED / NOT FOUND / STALE]
```

---

## Step 2 — Calculate True Profit After Fees

Polymarket charges fees that eat into arbitrage profits. You must account for all
of them.

### 2a. Polymarket fee structure

- **Trading fee**: 0% on limit orders that provide liquidity (maker), variable on
  taker orders (typically 0-1% depending on the market).
- **Winner fee**: **2% of winnings** — this is the big one. When your winning
  position pays out, Polymarket takes 2% of the profit (not the principal).
  Specifically, if you buy YES at $0.60 and it resolves YES, you get
  $1.00 - 2% * ($1.00 - $0.60) = $1.00 - $0.008 = $0.992.

### 2b. Calculate net profit for each scenario

For a complementary arbitrage where you buy YES on Market A at price `a` and
YES on Market B (the complement) at price `b`:

**Scenario 1: Market A resolves YES (Market B resolves NO)**
```
Payout from A = $1.00 - 0.02 * ($1.00 - a) = $1.00 - 0.02 + 0.02a
Loss from B = -b
Net = (0.98 + 0.02a) - b - a = 0.98 - a - b + 0.02a = 0.98 - 0.98a - b
```

**Scenario 2: Market B resolves YES (Market A resolves NO)**
```
Payout from B = $1.00 - 0.02 * ($1.00 - b) = 0.98 + 0.02b
Loss from A = -a
Net = (0.98 + 0.02b) - a - b = 0.98 - a - 0.98b
```

**True arbitrage exists only if BOTH scenarios are profitable.**

### 2c. Calculate the minimum profitable spread

Given Polymarket's 2% winner fee, the minimum raw spread for a risk-free arb is
approximately:
```
Minimum Spread > 2% * max(1 - a, 1 - b)
```

For most arbs, you need at least a 2-3 cent spread ($0.02-$0.03) to clear fees.

### 2d. Factor in gas costs (if applicable)

- USDC deposits/withdrawals on Polygon are cheap but not free.
- If you need to bridge funds, include bridge fees (~$0.50-$2.00).
- If you need to swap tokens, include DEX fees.

### Output for this step

```
Scenario 1 Net Profit: $X.XX (X.X%)
Scenario 2 Net Profit: $X.XX (X.X%)
Worst-Case Net Profit: $X.XX (X.X%)
Fee Breakdown:
  - Winner fee impact: $X.XX
  - Trading fees (est): $X.XX
  - Gas/bridge costs: $X.XX
Verdict: [PROFITABLE / MARGINAL / UNPROFITABLE]
```

---

## Step 3 — Confirm Markets Are Actually Related

One of the most dangerous errors in prediction market arbitrage is assuming two
markets are related when they are not.

### 3a. Resolution criteria comparison

Read the FULL resolution description for each market. Compare:
- Do they reference the **same event**?
- Do they use the **same resolution source**?
- Do they have the **same resolution date** or compatible dates?
- Are the conditions **logically complementary, subset, or mutually exclusive**?

### 3b. Common traps

| Trap | Example |
|------|---------|
| **Different resolution sources** | Market A uses AP call, Market B uses certified results |
| **Different timeframes** | "Will X happen in Q1?" vs "Will X happen in 2025?" — not complementary |
| **Subtle scope differences** | "Will Congress pass X?" vs "Will X become law?" — passing is not the same as becoming law (presidential veto) |
| **Different definitions** | "Recession by NBER definition" vs "Two consecutive quarters of negative GDP" |
| **Platform-specific rules** | Polymarket vs Kalshi may have different resolution rules for the same event |

### 3c. Verify logical constraint

Write out the logical relationship explicitly:

```
IF Market_A resolves YES, THEN Market_B MUST resolve [YES/NO] because: <reason>
IF Market_A resolves NO, THEN Market_B MUST resolve [YES/NO] because: <reason>
```

If either statement has a "maybe" or "probably" instead of "MUST", **this is not
a true arbitrage**. It is a correlated trade with risk.

### Output for this step

```
Relationship: [TRUE COMPLEMENT / TRUE SUBSET / CORRELATED BUT NOT GUARANTEED / UNRELATED]
Confidence in relationship: [CERTAIN / HIGHLY LIKELY / UNCERTAIN]
Key difference found: <if any>
```

---

## Step 4 — Evaluate Order Book Depth

Quoted prices mean nothing if you cannot execute at those prices. Order book
depth is critical.

### 4a. Check available liquidity

For each market, check:
- **Size at best price**: How many shares are available at the quoted best
  ask/bid?
- **Depth at 1 cent worse**: How many additional shares at best + $0.01?
- **Depth at 2 cents worse**: How many additional shares at best + $0.02?
- **Total available within your target size**: Can you fill your desired position
  at an acceptable average price?

### 4b. Calculate realistic execution prices

If you want to buy $500 worth of YES shares and the order book is:

```
$0.55 x 200 shares
$0.56 x 300 shares
$0.57 x 500 shares
```

Your average fill price for 500 shares would be:
```
(200 * 0.55 + 300 * 0.56) / 500 = $0.556
```

Not $0.55 as the quoted best ask suggests.

### 4c. Thin book warning signs

- Less than $200 available at the best price.
- Spread > $0.05 between best bid and best ask.
- Order book is one-sided (all asks, no bids — or vice versa).
- Large gap between the top of book and the next level.

### Output for this step

```
Market A:
  Size at best ask: X shares @ $X.XX
  Realistic fill for $Y position: $X.XX avg price
  Liquidity rating: [DEEP / ADEQUATE / THIN / EMPTY]

Market B:
  Size at best ask: X shares @ $X.XX
  Realistic fill for $Y position: $X.XX avg price
  Liquidity rating: [DEEP / ADEQUATE / THIN / EMPTY]

Combined execution feasibility: [EXECUTABLE / PARTIAL / NOT EXECUTABLE]
```

---

## Step 5 — Assess Slippage Risk

Slippage is the difference between the expected execution price and the actual
execution price. In prediction markets, slippage can be severe.

### 5a. Sources of slippage

- **Size impact**: Your order consumes multiple price levels.
- **Timing**: Prices move between when you see the opportunity and when your order
  hits the book. On Polymarket (Polygon blockchain), this can be 2-10 seconds.
- **Front-running**: Other bots may see the same opportunity and race you.
  Polymarket uses a CLOB (central limit order book) via the Polymarket exchange
  contract, but MEV is still possible on Polygon.
- **Information asymmetry**: The arb may exist because one market has priced in
  new information that the other hasn't yet.

### 5b. Estimate slippage impact

- For the position size you want, calculate the difference between best price and
  average fill price (from Step 4).
- Add a slippage buffer of 0.5-1% for timing risk.
- If the net profit after slippage < 1%, the arb is likely not worth executing.

### 5c. Slippage mitigation strategies

- Use **limit orders** instead of market orders to cap your maximum price.
- **Split execution** across time if the book is thin.
- Set **price alerts** to catch the arb when spreads widen temporarily.
- Consider the **two-leg risk**: you may fill one side but not the other, leaving
  you with a directional position.

### Output for this step

```
Estimated slippage (Market A): X.X%
Estimated slippage (Market B): X.X%
Combined slippage impact: -$X.XX
Profit after slippage: $X.XX (X.X%)
Two-leg risk: [LOW / MEDIUM / HIGH]
```

---

## Step 6 — Check Resolution Date Alignment

For a true arbitrage, both markets must resolve based on the same event at the
same time (or in a predictable sequence).

### 6a. Resolution date comparison

| | Market A | Market B |
|---|---------|---------|
| Resolution date | | |
| Resolution source | | |
| Expected reporting lag | | |
| Dispute period | | |
| **Effective resolution** | | |

### 6b. Temporal risk scenarios

- **Gap risk**: Market A resolves on Jan 15, Market B resolves on Jan 31.
  Between Jan 15 and Jan 31, you have capital locked in Market B with
  directional exposure.
- **Revision risk**: Market A resolves based on a preliminary report. Market B
  resolves based on a revised report. The revision could flip the outcome.
- **Early resolution**: One market may resolve early (e.g., a candidate drops
  out), while the other stays open.

### 6c. Capital lockup cost

If there's a gap between resolution dates, calculate the opportunity cost:
```
Lockup cost = Position_size * Daily_opportunity_rate * Days_between_resolutions
```

A reasonable daily opportunity rate for Polymarket is 0.02-0.05% (7-18% APY).

### Output for this step

```
Resolution date alignment: [SAME DAY / WITHIN 1 WEEK / SIGNIFICANT GAP]
Gap duration: X days
Capital lockup cost: $X.XX
Temporal risk rating: [LOW / MEDIUM / HIGH]
```

---

## Step 7 — Evaluate Counterparty and Platform Risk

Even "risk-free" arbitrage has platform risk.

### 7a. Polymarket-specific risks

- **Smart contract risk**: Polymarket's conditional token framework (CTF) is
  audited but not risk-free. A bug could lock funds.
- **Regulatory risk**: Polymarket has faced CFTC scrutiny. A regulatory action
  could freeze markets or force early resolution.
- **Oracle failure**: The UMA oracle could malfunction or resolve incorrectly.
  While disputes exist, they are not instant.
- **Liquidity withdrawal**: Market makers could withdraw liquidity before you can
  exit, widening spreads dramatically.

### 7b. Cross-platform risks (if applicable)

If the arb spans Polymarket and another platform (Kalshi, Metaculus, etc.):
- Different resolution criteria between platforms.
- Different settlement times.
- Counterparty risk on the less-established platform.
- Regulatory differences (Kalshi is CFTC-regulated; Polymarket operates offshore).

### 7c. Black swan scenarios

Consider extreme scenarios:
- Platform goes offline during a critical period.
- Blockchain congestion prevents order execution.
- A market is voided/cancelled by Polymarket (positions refunded at cost basis).
- A resolution dispute drags on for weeks, locking capital.

### Output for this step

```
Platform risk: [LOW / MODERATE / ELEVATED / HIGH]
Cross-platform risk: [N/A / LOW / MODERATE / HIGH]
Black swan scenarios identified: <count>
Most concerning scenario: <description>
```

---

## Step 8 — Check for Recent Price Movement

An arb that is closing quickly may not be worth chasing. Conversely, an arb that
has persisted for hours may have a structural reason for existing.

### 8a. Price history analysis

For each market in the arb:
- **Price 1 hour ago**: Is the spread narrowing or widening?
- **Price 24 hours ago**: How long has this opportunity existed?
- **Volume in the last hour**: Is someone actively trading to close the arb?
- **Large recent trades**: Did a whale move one side, creating a temporary
  dislocation?

### 8b. Closing velocity

Calculate how fast the spread is narrowing:
```
Closing velocity = (Spread_1h_ago - Spread_now) / 1 hour
```

If the closing velocity suggests the arb will disappear in < 5 minutes, it is
likely not executable manually.

### 8c. Why the arb might persist

Some arbs persist because of structural reasons:
- **Capital lockup**: Traders see the arb but don't want to lock capital until
  resolution.
- **Resolution risk**: The arb compensates for resolution ambiguity (see
  resolution_analysis skill).
- **Low absolute profit**: $5 profit isn't worth the effort for most traders.
- **Complexity**: Multi-leg arbs require simultaneous execution that most traders
  can't do.

### Output for this step

```
Spread trend: [WIDENING / STABLE / NARROWING / RAPIDLY CLOSING]
Time the arb has existed: ~X hours/days
Closing velocity: $X.XX per hour
Likely reason for persistence: <explanation>
Urgency: [EXECUTE NOW / NORMAL / NO RUSH / LIKELY GONE]
```

---

## Step 9 — Calculate Optimal Position Size

Even a confirmed arb should be sized appropriately.

### 9a. Kelly criterion for arbitrage

For a true arbitrage (guaranteed profit), Kelly says bet your entire bankroll.
In practice, model uncertainty means you should use **fractional Kelly**.

```
Recommended size = Bankroll * Kelly_fraction * Confidence
```

Where:
- `Kelly_fraction` = 0.25 (quarter Kelly — conservative for prediction markets)
- `Confidence` = your confidence that this is a true arb (0.0 to 1.0)

### 9b. Practical constraints

- **Liquidity constraint**: You can't buy more than the order book supports.
  Your maximum size is the lesser of your desired size and the available
  liquidity (from Step 4).
- **Concentration limit**: Never put more than 10-20% of your bankroll in a
  single arbitrage, even if it looks risk-free.
- **Minimum profit threshold**: If the total dollar profit is < $5, the
  operational overhead likely isn't worth it.

### 9c. Position sizing recommendation

```
Maximum theoretical size: $X (based on Kelly)
Liquidity-constrained size: $X (based on order books)
Recommended position size: $X (min of above, capped at 15% of bankroll)
Expected profit at recommended size: $X.XX
Expected ROI: X.X%
```

### Output for this step

Produce the position sizing recommendation above.

---

## Step 10 — Final Recommendation

Compile everything into a clear, actionable recommendation.

### Decision framework

| Factor | Weight | Score (1-10) | Weighted |
|--------|--------|-------------|----------|
| Profit after fees | 25% | | |
| Execution feasibility | 20% | | |
| Market relationship certainty | 20% | | |
| Resolution alignment | 15% | | |
| Slippage risk | 10% | | |
| Platform risk | 10% | | |
| **Total** | **100%** | | **/10** |

### Recommendation categories

- **EXECUTE**: Score >= 7.5. High confidence, profitable after all costs, good
  liquidity. Proceed with recommended position size.
- **MONITOR**: Score 5.0-7.4. The opportunity exists but has concerns. Set
  alerts and wait for conditions to improve (wider spread, more liquidity).
- **SKIP**: Score 3.0-4.9. Too many risks or too little profit. Not worth the
  capital lockup.
- **AVOID**: Score < 3.0. Not a real arbitrage, or the risks far outweigh the
  potential profit.

### Final output format

```
RECOMMENDATION: [EXECUTE / MONITOR / SKIP / AVOID]

Overall Score: X.X / 10

Expected Profit: $X.XX (X.X% ROI)
Position Size: $X.XX per leg
Confidence: [HIGH / MEDIUM / LOW]

Execution Plan:
1. [First leg: Buy YES on Market A at limit $X.XX, size X shares]
2. [Second leg: Buy YES on Market B at limit $X.XX, size X shares]
3. [Expected resolution date: YYYY-MM-DD]
4. [Expected payout: $X.XX]

Key Risks:
1. <most important risk and mitigation>
2. <second risk and mitigation>
3. <third risk and mitigation>

Notes:
<any additional context, caveats, or alternative strategies>
```

---

## General Guidelines

- **Be skeptical**: Most apparent arbs are not real after accounting for fees,
  slippage, and resolution differences. Your job is to find the ones that ARE
  real.
- **Show your math**: Every profit calculation should be explicit and verifiable.
  Do not round prematurely.
- **Think about execution**: A profitable arb on paper is worthless if you cannot
  execute both legs. Always check liquidity first.
- **Consider the counterfactual**: If this arb is so obvious, why hasn't someone
  else taken it? The answer to that question often reveals the real risk.
- **Time is money**: Factor in capital lockup. A 2% arb that locks capital for
  6 months has a very different risk-reward than a 2% arb that resolves tomorrow.
- **Document everything**: Your analysis should be detailed enough that another
  trader could review it and reach the same conclusion.
