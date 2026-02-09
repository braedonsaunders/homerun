# Algorithm Recommendations Analysis: Lessons from False Positives

Analysis of 5 opportunities that the detection algorithms surfaced but the AI judge
correctly flagged as bad trades (scores 15-30/100). Each reveals a systemic weakness.

---

## Executive Summary

| # | Strategy | Score | Root Cause |
|---|----------|-------|------------|
| 1 | Event-Driven (SFA/McNeese parlay) | 25 | Category-level matching treats unrelated parlays as correlated |
| 2 | Cross-Platform (player props) | 30 | No modeling of platform-specific resolution rules (DNP risk) |
| 3 | Liquidity Vacuum (Orlando parlay) | N/A | Velocity detector repackages past price moves as fake order book signal |
| 4 | Event-Driven (Southern U/Arizona) | 20 | Same as #1 + AI judge JSON parse crash on complex parlays |
| 5 | NegRisk (Warner Bros acquisition) | 15 | Non-exhaustive outcome set not detected (acquisition not in keyword list) |

---

## Detailed Root Cause Analysis

### 1. Event-Driven: Loose "Relatedness" Matching (event_driven.py)

**What happened:** Orlando/Southern University/Central Arkansas parlay moved DOWN 43%.
The algo flagged Stephen F. Austin/McNeese/Louisville parlay as "lagging" because it
didn't move.

**Code path:** `_find_related_markets()` (line 245-283) uses three connection types:
1. Same event (by event_id) — fine
2. Same category — **too broad**: ALL NCAA basketball parlays match each other
3. Shared keywords (≥2) — **too loose for parlays**: multi-team questions generate many
   keywords, and any 2 shared words trigger relatedness

**Why it's wrong:** A 43% move in Parlay A says nothing about Parlay B when the teams
are completely different. Category-level matching (line 272-274) short-circuits without
checking structural connection: `if catalyst_category and mid_category and
catalyst_category == mid_category: related[mid] = True`.

**Suggested fixes:**
- Disable same-category matching for multi-leg/parlay markets (questions containing
  comma-separated conditions)
- Increase `_MIN_SHARED_KEYWORDS` from 2 to 4+ for parlay markets
- Require shared keywords to be the "entity" keywords (team names, player names), not
  generic terms
- Add a minimum keyword overlap *ratio* (Jaccard-like) rather than just count

### 2. Cross-Platform: Missing Resolution Rule Divergence (cross_platform.py)

**What happened:** Player prop combo (Knueppel 20+, LaMelo Ball 10+, Cade Cunningham)
matched across platforms with similarity=1.00, but different platforms have different
DNP rules.

**Code path:** `_calculate_arb()` (line 473-565) assumes symmetric resolution:
"Guaranteed payout is $1.00 (one side always wins)." No concept of partial void.

**Why it's wrong:** If Polymarket voids a player prop on DNP but Kalshi resolves as NO,
the arbitrageur loses the Kalshi leg without the Polymarket hedge. The "guaranteed
profit" becomes a guaranteed loss on one leg.

**Suggested fixes:**
- Add a market type classifier that detects player props, game props, and other
  categories where platform resolution rules commonly diverge
- Apply a "resolution divergence penalty" to the risk score for player props, contingent
  events, and multi-condition markets
- Consider flagging cross-platform player prop matches with a mandatory "review" status
  regardless of score

### 3. Liquidity Vacuum: Synthetic Velocity = Circular Signal (liquidity_vacuum.py)

**What happened:** The same 43% drop in the Orlando parlay triggered a "87x order book
imbalance" via the velocity detector.

**Code path:** `_analyze_velocity()` (line 371-418):
```
net_move = yes_delta - no_delta  ≈  -0.43 - 0.43 = -0.86
imbalance_ratio = 1.0 + abs(-0.86) * 100 = 87.0
```

**Why it's wrong:** This is circular reasoning. The price already moved 43% — the velocity
detector sees this past movement and says "buy in the direction it already moved." There's
no forward-looking information about order book depth. The conversion formula
`1.0 + abs(net_move) * 100` turns any significant price change into a massive fake
imbalance ratio.

**Suggested fixes:**
- Cap synthetic imbalance ratios (e.g., max 20x for velocity-only signals)
- Require velocity signals to persist across 2+ consecutive scans (momentum confirmation)
  rather than triggering on a single move
- Down-weight or disable velocity detection for markets that had a large discrete move
  (likely event resolution, not order flow)
- Add a "staleness" check: if the move happened in the same scan cycle, the price has
  already adjusted — there's no vacuum left

### 4. Event-Driven #2: Same Bug + AI Judge Parse Failure

Same root cause as #1 (category matching). Additionally, the AI judge crashed:
"Automated judgment was unable to complete (LLM returned list instead of dict)."

**Code path:** `opportunity_judge.py` — LLM structured output parsing failed on a
complex multi-team parlay question.

**Suggested fix:**
- Add explicit type validation/coercion when parsing LLM judge responses
- If the LLM returns a list, extract the first element if it's a dict
- Log the raw LLM response for debugging when parsing fails

### 5. NegRisk: Non-Exhaustive Acquisition Outcomes (negrisk.py)

**What happened:** Warner Bros acquisition market with 4 outcomes (Comcast, Netflix,
Paramount, "No listed company"). Total YES = $0.952, barely above the 0.95 threshold.
If Disney/Amazon/Apple/Sony acquires, all 4 resolve to NO.

**Code path:** `_is_open_ended_event()` (line 207-234) checks for keywords like "nobel",
"oscar", "mvp" but has no keywords for acquisitions, M&A, mergers, or takeovers.

**Why it's wrong:** The 4.8% gap ($1.00 - $0.952) is the market pricing in the probability
of an unlisted acquirer. The NEGRISK_MIN_TOTAL_YES threshold of 0.95 is too permissive
for markets with very small outcome sets. With only 4 outcomes covering a universe of
dozens of potential acquirers, a 4.8% non-exhaustiveness probability is rational.

**Suggested fixes:**
- Add acquisition/M&A keywords to `_is_open_ended_event()`: "acquisition", "acquire",
  "merger", "takeover", "buyout", "purchase"
- Implement outcome-count-aware thresholds: markets with ≤5 outcomes should require
  higher total_yes (e.g., 0.98) since small sets are more likely non-exhaustive
- Consider a ratio: `total_yes_threshold = 1.0 - (0.01 * num_outcomes)` so larger
  outcome sets tolerate more gap

---

## Cross-Cutting Systemic Issues

### A. Directional Bets Surfaced Alongside Arbitrage

Three of five (Event-Driven x2 + Liquidity Vacuum) are pure directional bets with no
structural edge. They set `is_guaranteed=False` and add warning text, but reach users
alongside genuine arbitrage opportunities. Consider:
- Separate UI sections for "Arbitrage" vs "Statistical Edge" strategies
- Different default sort/filter that deprioritizes non-guaranteed opportunities

### B. No Cross-Strategy Deduplication

The Orlando/Southern University/Central Arkansas 43% move generated 3 bad
recommendations (Event-Driven x2, Liquidity Vacuum x1) from a single price movement.
The scanner runs all 18 strategies in parallel but has no mechanism to detect when
multiple strategies fire on the same underlying signal.

**Suggested fix:** After all strategies run, deduplicate by market_id. If multiple
strategies flag the same market within the same scan cycle, keep only the highest-scoring
one (or merge them into a "multi-signal" opportunity).

### C. AI Judge Is Effective But Late

All 5 were correctly scored low (15-30/100) by the AI judge. The judge is the most
reliable quality gate, but it runs *after* detection and filtering. Opportunities that
are structurally impossible (uncorrelated parlays, non-exhaustive outcome sets) should
be rejected before they reach the judge. The judge should be the last line of defense
for edge cases, not the primary quality filter.

### D. ROI Display Inconsistency

The backend `MAX_PLAUSIBLE_ROI` filter uses `(net_profit / total_cost) * 100`, which
gives 5-17% for these trades. But the UI displays 503-1286% ROI. This discrepancy
suggests the frontend calculates ROI differently (possibly using the complement price as
cost basis). Users see wildly inflated ROI numbers that the backend filter never
evaluates against.

**Suggested fix:** Ensure frontend ROI calculation matches `base.py:152` exactly, or add
a `displayed_roi` field to ArbitrageOpportunity that the frontend uses directly.
