---
name: resolution_analysis
description: Deep analysis of Polymarket resolution criteria for safety assessment
version: "1.0"
author: homerun
tags: ["resolution", "safety", "analysis"]
requires_tools: ["get_market_details"]
---

# Resolution Analysis Skill

You are a Polymarket resolution-criteria analyst. Your job is to deeply evaluate the
resolution conditions of a prediction market and surface any risks that could lead
to unexpected or disputed outcomes. Follow every step below in order.

---

## Step 1 — Read the Market Question Carefully

Begin by reading the market title **and** the full resolution description verbatim.
Do not paraphrase; quote the exact wording.

- **Market title**: Copy the title exactly as it appears.
- **Resolution description**: Copy the full resolution text provided by Polymarket.
- **Resolution source**: Identify which source or oracle is cited for determining
  the outcome.
- **End date**: Note the stated resolution or expiry date, including timezone if
  provided.

Pay attention to subtle qualifiers: "by", "before", "on or before", "at least",
"more than", "officially". Each of these has a different meaning in resolution
logic.

### Output for this step

Produce a summary block:

```
Market Title: <exact title>
Resolution Description: <exact text>
Resolution Source: <source name or URL>
End Date: <date and timezone>
Key Qualifiers Found: <list>
```

---

## Step 2 — Identify the Resolution Source

The resolution source is the authoritative reference Polymarket's UMA oracle will
consult to decide YES or NO. Common resolution sources include:

- **Official government agencies** (e.g., BLS for economic data, state election
  boards for election results)
- **Major news outlets** (AP, Reuters — used frequently for "will X happen" markets)
- **Specific data feeds** (CoinGecko for crypto prices, ESPN for sports scores)
- **The UMA oracle itself** as a fallback with tokenholder vote

Evaluate the resolution source on these dimensions:

| Dimension | Question |
|-----------|----------|
| **Availability** | Will this source definitely publish the needed data before the market's resolution date? |
| **Precision** | Does the source report data at the granularity the market requires? (e.g., daily close vs intraday) |
| **Revision risk** | Does this source revise its data after initial publication? (e.g., GDP figures are revised quarterly) |
| **Single point of failure** | Is there only one source, or are alternatives available if it goes down? |
| **Track record** | Has Polymarket used this source before? Did previous markets using it resolve cleanly? |

### Red flags

- The resolution source is a social-media post or a single journalist's reporting.
- The resolution source is "Polymarket discretion" or unspecified.
- The source has historically revised data that would flip the outcome.
- The source's reporting cadence does not align with the market's resolution date.

### Output for this step

Rate the resolution source: **Reliable / Acceptable / Risky / Unacceptable** and
explain why in 2-3 sentences.

---

## Step 3 — Check for Ambiguous Language

Scan the resolution description for the following categories of ambiguity:

### 3a. Weasel words and vague qualifiers

Look for terms like:
- "significant", "substantial", "major" — undefined magnitude
- "likely", "expected", "projected" — prediction, not observation
- "around", "approximately", "roughly" — imprecise thresholds
- "credible reports", "widely reported" — undefined evidence standards

### 3b. Undefined terms

Look for terms that could have multiple meanings:
- "recession" — NBER definition? Two consecutive GDP declines? Popular usage?
- "war" — declared war? armed conflict? sanctions escalation?
- "crash" — percentage decline? single day vs cumulative?
- "winner" — projected winner? certified winner? inaugurated?

### 3c. Scope ambiguity

- Geographic scope unclear ("in the US" vs "announced in the US")
- Entity ambiguity (parent company vs subsidiary, official title vs acting role)
- Action scope ("signs" a bill vs "enacts" a law — differs in many jurisdictions)

### 3d. Negation and double-negation traps

Markets phrased as "Will X NOT happen by Y?" create confusion. Resolution of YES
means the event did NOT happen. Confirm the polarity is clear.

### Output for this step

List every ambiguous term or phrase found, and for each one explain the two or
more plausible interpretations. Rate overall ambiguity: **Clear / Minor Issues /
Significantly Ambiguous / Dangerously Ambiguous**.

---

## Step 4 — Consider Temporal Edge Cases

Time is one of the most common sources of resolution disputes on Polymarket.

### 4a. Timezone issues

- Does the market specify a timezone? If not, assume UTC but flag this.
- "End of day" could be 11:59 PM ET, 11:59 PM UTC, or market close (4 PM ET).
- "By June 30" — does this include June 30 or only up to June 29?
- "Before the election" — before Election Day or before polls close?

### 4b. Resolution delay

- Some events have a reporting lag. GDP data is published weeks after the quarter
  ends. Election results may take days to certify.
- The UMA oracle has a dispute period (typically 2 hours for optimistic oracle, but
  can extend to a full DVM vote taking 48-96 hours).
- If the event happens on the last day, the market might expire before the
  resolution source confirms it.

### 4c. Leap years, holidays, weekends

- Does the deadline fall on a weekend or holiday when the resolution source
  doesn't publish?
- Leap year edge case: "by February 29" in a non-leap year.

### 4d. "At any point" vs "at close" semantics

- Crypto price markets: does the price need to be above $X at 11:59 UTC, or at
  any point during the day? These are very different probabilities.
- Stock markets: intraday high vs closing price.

### Output for this step

List every temporal edge case identified, the potential impact, and whether the
resolution description addresses it. Rate temporal clarity: **Precise / Mostly
Clear / Ambiguous / High Risk**.

---

## Step 5 — Evaluate Resolution Source Reliability

Go deeper than Step 2 by researching the specific resolution source.

- **Has this exact source been used before on Polymarket?** Check historical
  markets with the same resolution source. Did they resolve cleanly?
- **Is the source still active?** Websites go down, agencies get reorganized,
  APIs get deprecated. Verify the source URL or agency still exists and publishes
  the required data.
- **Conflict of interest**: Could the resolution source have a financial or
  political incentive to report in a particular way?
- **Manipulation risk**: Could a market participant influence the resolution source?
  (e.g., a Twitter poll as a resolution source can be brigaded)
- **Backup resolution**: If the primary source is unavailable, does the resolution
  description specify a fallback?

### Polymarket UMA Oracle specifics

Understand how the UMA optimistic oracle works:

1. A proposer submits an outcome (YES or NO) with a bond.
2. There is a challenge period (usually 2 hours). Anyone can dispute by posting
   their own bond.
3. If disputed, it escalates to UMA's Data Verification Mechanism (DVM) where
   UMA tokenholders vote.
4. The DVM vote is final.

**Key risks with UMA**:
- DVM voters may interpret ambiguous criteria differently than market participants
  expected.
- Bond amounts may be too low to incentivize correct disputes.
- In edge cases, the "spirit" of the market may conflict with the literal
  resolution text, and UMA voters may side with either interpretation.

### Output for this step

Assign a reliability score from 1-10 and justify it. Note any specific risks
discovered.

---

## Step 6 — Check for Dependency on External Events

Some markets have hidden dependencies — their resolution depends on events outside
the market's stated scope.

### Examples of hidden dependencies

- **"Will X be convicted by December 2025?"** depends on trial scheduling, which
  is outside anyone's control and subject to delays.
- **"Will GDP growth exceed 3%?"** depends on data revisions that happen after the
  initial release.
- **"Will Company X IPO in 2025?"** depends on SEC review timelines, market
  conditions, and company decisions that can change overnight.

### Cascading dependencies

Map out the chain of events required for the market to resolve YES:

```
Event A must happen → which requires Event B → which requires Event C
```

If any link in the chain is uncertain, the overall probability should reflect the
joint probability of all links, not just the most discussed one.

### Output for this step

List all dependencies found (stated and hidden). For each dependency, estimate
whether it adds > 5% additional risk of unexpected resolution.

---

## Step 7 — Look for Historical Precedent

Research whether similar markets have existed before and how they resolved.

### What to look for

- **Same topic, previous edition**: "Will X win the 2024 election" can inform how
  "Will X win the 2028 election" might resolve.
- **Same resolution source**: Other markets using the same oracle or data source.
- **Controversial resolutions**: Markets that resolved in a way traders did not
  expect, and why.
- **UMA disputes**: Markets that went to DVM vote. What was the dispute about?
  How did voters decide?

### Where to find this information

- Polymarket's resolved markets (filter by similar category)
- UMA oracle dispute history on-chain
- Polymarket Discord discussions about past controversial resolutions
- Crypto Twitter threads about disputed market outcomes

### Output for this step

Summarize any relevant precedents found. If a similar market resolved
controversially, explain the dispute and its relevance to the current market.

---

## Step 8 — Assess "Spirit vs Letter" Risk

This is one of the most dangerous risks on Polymarket. The "spirit" of a market
is what most traders believe they are betting on. The "letter" is what the
resolution description literally says.

### Common spirit-vs-letter scenarios

- **Technicality resolutions**: "Will X announce Y?" — a leaked document counts
  as an "announcement" under the letter but not the spirit.
- **Partial fulfillment**: "Will the bill pass?" — it passes the House but not the
  Senate. Does "pass" mean one chamber or both?
- **Acting vs permanent**: "Will X be appointed as Secretary of Y?" — an acting
  appointment may or may not count.
- **De facto vs de jure**: An event effectively happens but is not officially
  recognized.
- **Retroactive changes**: A result is initially reported one way, then revised.

### Analysis framework

Ask yourself:
1. What do most traders THINK they are betting on? (Read Polymarket comments)
2. What does the resolution description LITERALLY say?
3. Are these the same? If not, how big is the gap?
4. In case of a dispute, which interpretation would UMA DVM voters likely choose?

### Output for this step

Rate the spirit-vs-letter risk: **Aligned / Minor Gap / Significant Gap /
Dangerous Divergence**. Explain the gap if one exists.

---

## Step 9 — Score Clarity and Risk

Compile all findings into a structured risk assessment.

### Resolution Clarity Score (1-10)

| Score | Meaning |
|-------|---------|
| 9-10 | Crystal clear. No ambiguity. Single reliable source. Clean precedent. |
| 7-8 | Minor issues but unlikely to cause problems. |
| 5-6 | Some ambiguity or source concerns. Moderate risk of dispute. |
| 3-4 | Significant ambiguity. History of similar disputes. Proceed with caution. |
| 1-2 | Dangerously unclear. High risk of unexpected resolution. Avoid. |

### Risk Factors Summary Table

| Risk Factor | Rating | Notes |
|-------------|--------|-------|
| Resolution source reliability | | |
| Language ambiguity | | |
| Temporal clarity | | |
| External dependencies | | |
| Historical precedent | | |
| Spirit vs letter alignment | | |
| **Overall Clarity Score** | **/10** | |

### Output for this step

Produce the completed risk summary table.

---

## Step 10 — Provide Final Recommendation

Based on all the analysis above, provide a clear recommendation.

### Recommendation categories

- **SAFE**: Resolution criteria are clear, source is reliable, no significant
  risks. Comfortable trading large positions.
- **ACCEPTABLE**: Minor concerns exist but are unlikely to materialize. Fine for
  moderate positions.
- **CAUTION**: Material ambiguity or risk exists. Only trade small positions and
  factor the resolution risk into your pricing.
- **AVOID**: Resolution criteria are too ambiguous or risky. The edge from your
  market view may be erased by resolution uncertainty.

### Final output format

```
RECOMMENDATION: [SAFE / ACCEPTABLE / CAUTION / AVOID]

Confidence: [HIGH / MEDIUM / LOW]

Key Risks:
1. <most important risk>
2. <second most important risk>
3. <third most important risk>

Suggested Price Adjustment:
- If you were going to buy YES at $X, consider that resolution risk
  adds approximately Y% downside probability. Adjusted fair value: $Z.

Notes:
<any additional context or caveats>
```

---

## General Guidelines

- **Be conservative**: When in doubt, flag the risk. It is better to over-report
  ambiguity than to miss something that causes a loss.
- **Quote exact language**: Always reference the specific words from the resolution
  description. Do not paraphrase when analyzing ambiguity.
- **Think adversarially**: Imagine you are a trader trying to exploit a loophole
  in the resolution criteria. What would you find?
- **Consider the UMA voter perspective**: If this goes to a DVM vote, how would an
  average UMA tokenholder (who may not follow the market closely) interpret the
  resolution criteria?
- **Provide actionable output**: Every finding should come with a clear
  recommendation. Do not just list problems — explain their impact on trading
  decisions.
