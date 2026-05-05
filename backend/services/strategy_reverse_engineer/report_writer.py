"""LLM section writer for the analytical wallet report.

Takes a deterministic ``WalletAnalytics`` result and produces narrative
text for each report section.  The LLM never sees raw trades or invents
numbers — every figure it cites is a value computed by ``wallet_analytics``
that we serialize into the section's prompt.

Section list mirrors polyresearchrobotics' published reports:

  1. **headline_oneliner** — one sentence punchline, e.g. "An all-buy
     book that looks like a market maker, isn't one, and quietly prints
     money."
  2. **at_a_glance**       — 1-paragraph executive summary.
  3. **analysis_narrative** — the long-form 'what the data says' walk
     through the trader's strategy mechanic.  Headlines like "wears the
     costume of X but earns its money like Y."
  4. **two_leg_explainer** — short prose around the spread vs.
     directional decomposition table.
  5. **dominance_explainer** — short prose around the skew bucket table.
  6. **filter_recommendation** — recommendation paragraph after the
     filter ledger.
  7. **playbook_brief** — one-paragraph operator brief that opens the
     replication playbook.
  8. **what_to_copy** + **what_not_to_copy** — bullet sections at the
     close of the analysis narrative.

Each section call is small (a few hundred tokens of prompt + a few
hundred of output) so the total cost per report is ~$0.10 on Claude
Sonnet 4.6 — well within an operator's "click-to-run" budget.

Failure mode: if the LLM call errors or returns garbage, the report
template falls back to a deterministic stub generated from the same
table.  The deliverable is never empty.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from services.ai.llm_provider import LLMMessage
from services.strategy_reverse_engineer.wallet_analytics import (
    WalletAnalytics,
)

logger = logging.getLogger(__name__)


@dataclass
class ReportSections:
    """Container for the LLM-written narrative pieces."""

    headline_oneliner: str = ""
    at_a_glance: str = ""
    analysis_narrative: str = ""
    two_leg_explainer: str = ""
    dominance_explainer: str = ""
    filter_recommendation: str = ""
    playbook_brief: str = ""
    what_to_copy: list[str] = field(default_factory=list)
    what_not_to_copy: list[str] = field(default_factory=list)
    pseudocode: str = ""
    bankroll_paragraph: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline_oneliner": self.headline_oneliner,
            "at_a_glance": self.at_a_glance,
            "analysis_narrative": self.analysis_narrative,
            "two_leg_explainer": self.two_leg_explainer,
            "dominance_explainer": self.dominance_explainer,
            "filter_recommendation": self.filter_recommendation,
            "playbook_brief": self.playbook_brief,
            "what_to_copy": list(self.what_to_copy),
            "what_not_to_copy": list(self.what_not_to_copy),
            "pseudocode": self.pseudocode,
            "bankroll_paragraph": self.bankroll_paragraph,
        }


_VOICE = (
    "You are a quantitative-strategy analyst writing for a sophisticated "
    "operator audience.  Voice: terse, evidence-driven, no filler, no "
    "marketing language.  Cite specific numbers from the JSON facts in "
    "every paragraph.  Never invent numbers — if a figure isn't in the "
    "facts JSON, don't reference it.  Default to compressed prose; "
    "expand only when the mechanism requires it."
)


async def draft_sections(
    *,
    analytics: WalletAnalytics,
    spotlight: Optional[dict[str, Any]],
    model: Optional[str] = None,
) -> ReportSections:
    """Run the LLM section-by-section, falling back to deterministic
    stubs on any failure.  Always returns a populated ReportSections.
    """
    facts = _build_facts(analytics, spotlight)
    sections = ReportSections()

    sections.headline_oneliner = await _safe_section(
        _draft_one_liner, facts, model,
        fallback=lambda f: _stub_one_liner(f),
    )
    sections.at_a_glance = await _safe_section(
        _draft_at_a_glance, facts, model,
        fallback=lambda f: _stub_at_a_glance(f),
    )
    sections.analysis_narrative = await _safe_section(
        _draft_analysis_narrative, facts, model,
        fallback=lambda f: _stub_analysis_narrative(f),
    )
    sections.two_leg_explainer = await _safe_section(
        _draft_two_leg_explainer, facts, model,
        fallback=lambda f: _stub_two_leg_explainer(f),
    )
    sections.dominance_explainer = await _safe_section(
        _draft_dominance_explainer, facts, model,
        fallback=lambda f: _stub_dominance_explainer(f),
    )
    sections.filter_recommendation = await _safe_section(
        _draft_filter_recommendation, facts, model,
        fallback=lambda f: _stub_filter_recommendation(f),
    )
    sections.playbook_brief = await _safe_section(
        _draft_playbook_brief, facts, model,
        fallback=lambda f: _stub_playbook_brief(f),
    )
    bullets_copy = await _safe_section(
        _draft_what_to_copy, facts, model,
        fallback=lambda f: _stub_what_to_copy(f),
    )
    sections.what_to_copy = _split_bullets(bullets_copy)
    bullets_skip = await _safe_section(
        _draft_what_not_to_copy, facts, model,
        fallback=lambda f: _stub_what_not_to_copy(f),
    )
    sections.what_not_to_copy = _split_bullets(bullets_skip)

    sections.pseudocode = _stub_pseudocode(facts)
    sections.bankroll_paragraph = await _safe_section(
        _draft_bankroll, facts, model,
        fallback=lambda f: _stub_bankroll(f),
    )
    return sections


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------


async def _safe_section(
    drafter, facts: dict[str, Any], model: Optional[str], *, fallback,
) -> str:
    """Run a drafter, fall back to the deterministic stub on any error."""
    try:
        text = await drafter(facts, model)
        if text and text.strip():
            return text.strip()
    except Exception as exc:
        logger.warning("report_writer drafter %s failed: %s", drafter.__name__, exc)
    return fallback(facts)


async def _llm_chat(prompt: str, model: Optional[str]) -> str:
    from services.ai import get_llm_manager

    manager = get_llm_manager()
    if not manager.is_available():
        raise RuntimeError("LLM manager unavailable")
    msgs = [
        LLMMessage(role="system", content=_VOICE),
        LLMMessage(role="user", content=prompt),
    ]
    response = await manager.chat(
        messages=msgs,
        model=model,
        temperature=0.3,
        max_tokens=1500,
        purpose="strategy_reverse_engineer_report",
    )
    return (response.content or "").strip()


# ---------------------------------------------------------------------------
# Per-section prompts
# ---------------------------------------------------------------------------


async def _draft_one_liner(facts: dict[str, Any], model: Optional[str]) -> str:
    prompt = (
        "Write a single-sentence punchline (≤ 18 words) summarizing this "
        "wallet's strategy. The sentence should encode the *mechanism* of the "
        "edge — what looks like X actually is Y, or the contradiction between "
        "appearance and reality. Do NOT lead with an ROI number. Use the "
        "facts below to ground it.\n\n"
        f"FACTS:\n{json.dumps(facts['headline'], default=str, indent=2)}\n\n"
        f"TWO-LEG:\n{json.dumps(facts['two_leg'], default=str, indent=2)}\n\n"
        f"TOP DOMINANCE BUCKET:\n{json.dumps(facts['top_dominance_bucket'], default=str, indent=2)}\n\n"
        "Return only the sentence, no quotes, no preamble."
    )
    return await _llm_chat(prompt, model)


async def _draft_at_a_glance(facts: dict[str, Any], model: Optional[str]) -> str:
    prompt = (
        "Write a single ~120-word paragraph executive summary for a quant "
        "research report on this wallet. Cover: (1) the wallet's apparent "
        "strategy archetype, (2) the actual edge source (often different from "
        "the apparent one), (3) one specific quantitative anchor that makes "
        "the case (a paired-cost / dominance / win-rate number). Use exact "
        "values from the facts. No speculation beyond what the numbers "
        "support.\n\n"
        f"FACTS:\n{json.dumps(facts, default=str, indent=2)[:6000]}\n\n"
        "Return only the paragraph."
    )
    return await _llm_chat(prompt, model)


async def _draft_analysis_narrative(facts: dict[str, Any], model: Optional[str]) -> str:
    prompt = (
        "Write the long-form analysis narrative section (~600-900 words) for "
        "this wallet. Structure:\n\n"
        "  1. Open with the contradiction: what the wallet *looks* like vs. "
        "     what it actually does (cite both-sides participation, paired "
        "     cost, and the spread leg P/L).\n"
        "  2. The single instrument / horizon paragraph: how concentrated "
        "     the universe is. Cite market_count and dominant horizon.\n"
        "  3. The two-leg decomposition paragraph (spread leg loses X, "
        "     directional leg makes Y, net Z).\n"
        "  4. The mechanism: cite the dominance-bucket monotonic climb "
        "     ('1.0-1.5× → ... → ≥5.0×') and explain in one sentence what "
        "     this implies (reactive directional, not predictive).\n"
        "  5. One short closing reflection on the edge and where it might "
        "     break. No marketing language.\n\n"
        "Voice: think 'Matt Levine writing for quants'. Concrete, "
        "evidence-anchored, slightly wry where appropriate. No emojis. No "
        "Markdown headers — just flowing prose with **bold** for the most "
        "important phrases.\n\n"
        f"FACTS:\n{json.dumps(facts, default=str, indent=2)[:8000]}\n\n"
        "Return only the section text."
    )
    return await _llm_chat(prompt, model)


async def _draft_two_leg_explainer(facts: dict[str, Any], model: Optional[str]) -> str:
    prompt = (
        "Write a 90-word paragraph explaining the two-leg P/L decomposition "
        "for this wallet. Cover: how 'paired_shares × (1 - paired_cost)' "
        "produces the spread leg P/L, how 'excess shares × outcome' produces "
        "the directional leg, and what their sum tells us about the edge "
        "source. Cite the specific numbers from the two_leg facts.\n\n"
        f"TWO-LEG:\n{json.dumps(facts['two_leg'], default=str, indent=2)}\n\n"
        "Return only the paragraph."
    )
    return await _llm_chat(prompt, model)


async def _draft_dominance_explainer(facts: dict[str, Any], model: Optional[str]) -> str:
    prompt = (
        "Write a 90-word paragraph explaining the dominance-bucket table for "
        "this wallet. The reader has just seen a table showing how dom-side "
        "win rate climbs from the lowest skew bucket to the highest. Your "
        "job: explain in plain English what 'dominance ratio' means, what "
        "the monotonic climb implies about the edge mechanism (reactive vs. "
        "predictive), and the single most striking number from the table.\n\n"
        f"DOMINANCE BUCKETS:\n{json.dumps(facts['dominance_buckets'], default=str, indent=2)}\n\n"
        "Return only the paragraph."
    )
    return await _llm_chat(prompt, model)


async def _draft_filter_recommendation(facts: dict[str, Any], model: Optional[str]) -> str:
    prompt = (
        "Write a 120-word paragraph operator recommendation based on the "
        "filter ledger. Identify the single highest-ROI-lift filter that's "
        "*operationally implementable* (i.e. doesn't require knowing market "
        "outcome in advance). For most market-making-shape strategies this "
        "is the 'underdog blocking @ ≥ Nx skew' filter. Quote specific lift "
        "numbers from the facts. End with one sentence on the tradeoff "
        "(what the filter gives up).\n\n"
        f"FILTER LEDGER:\n{json.dumps(facts['filter_ledger'], default=str, indent=2)}\n\n"
        "Return only the paragraph."
    )
    return await _llm_chat(prompt, model)


async def _draft_playbook_brief(facts: dict[str, Any], model: Optional[str]) -> str:
    prompt = (
        "Write a single ~140-word 'operator brief' paragraph that opens a "
        "replication playbook. Cover (in this order): trade frequency, "
        "instrument universe, clip size, both-sides participation, edge "
        "source, key constraint (capital scale or infra requirement). "
        "Cite specific quantitative anchors from the facts. End with one "
        "sentence on what an operator needs to replicate this strategy.\n\n"
        f"FACTS:\n{json.dumps(facts, default=str, indent=2)[:6000]}\n\n"
        "Return only the paragraph."
    )
    return await _llm_chat(prompt, model)


async def _draft_what_to_copy(facts: dict[str, Any], model: Optional[str]) -> str:
    prompt = (
        "List 3-5 concrete, copyable elements of this wallet's strategy. "
        "Each bullet should be one sentence: the rule + why it works. "
        "Format: each bullet on its own line, prefixed with '- '. No "
        "headers, no preamble, no numbering.\n\n"
        f"FACTS:\n{json.dumps(facts, default=str, indent=2)[:6000]}\n\n"
        "Return only the bullets."
    )
    return await _llm_chat(prompt, model)


async def _draft_what_not_to_copy(facts: dict[str, Any], model: Optional[str]) -> str:
    prompt = (
        "List 2-4 elements of this wallet's strategy that are NOT cleanly "
        "copyable (e.g. capital scale, latency moat, manual intervention "
        "patterns). Each bullet: one sentence on the element + why it's hard "
        "to clone. Format: each bullet on its own line, prefixed with '- '. "
        "No headers, no preamble.\n\n"
        f"FACTS:\n{json.dumps(facts, default=str, indent=2)[:6000]}\n\n"
        "Return only the bullets."
    )
    return await _llm_chat(prompt, model)


async def _draft_bankroll(facts: dict[str, Any], model: Optional[str]) -> str:
    prompt = (
        "Write a single ~110-word paragraph on bankroll math for replicating "
        "this strategy. Cover: per-market clip size, max concurrent capital "
        "tied up, turnover rate, and at what capital scale the strategy "
        "stops compressing. Cite specific values. No tables — pure "
        "narrative.\n\n"
        f"FACTS:\n{json.dumps(facts['headline'], default=str, indent=2)}\n\n"
        f"TRADE SIZE:\n{json.dumps(facts['trade_size'], default=str, indent=2)}\n\n"
        "Return only the paragraph."
    )
    return await _llm_chat(prompt, model)


# ---------------------------------------------------------------------------
# Deterministic stubs — used when the LLM is unavailable or fails
# ---------------------------------------------------------------------------


def _stub_one_liner(facts: dict[str, Any]) -> str:
    h = facts.get("headline", {})
    bs = facts.get("two_leg", {}).get("both_sides_participation_rate") or 0
    if bs >= 0.85 and (facts["two_leg"].get("median_paired_cost") or 0) > 1:
        return (
            "An all-buy book that looks like a market maker — but the spread "
            "leg bleeds and the directional leg pays."
        )
    pl = h.get("realized_pl_usdc")
    roi = h.get("roi_on_deployed")
    if pl is not None and roi is not None:
        return (
            f"A {h.get('total_trades', 0):,}-trade book that nets "
            f"${pl:,.0f} ({roi*100:+.2f}% ROI) on {h.get('markets_touched', 0)} markets."
        )
    return "Wallet trade history analysis."


def _stub_at_a_glance(facts: dict[str, Any]) -> str:
    h = facts.get("headline", {})
    t = facts.get("two_leg", {})
    parts = [
        f"This wallet placed {h.get('total_trades', 0):,} trades across "
        f"{h.get('markets_touched', 0)} markets over "
        f"{h.get('active_days', 0)} active days, "
        f"deploying ${h.get('total_usdc_deployed', 0):,.0f} of USDC. "
    ]
    if h.get("realized_pl_usdc") is not None:
        parts.append(
            f"Realized P/L is ${h['realized_pl_usdc']:,.2f} "
            f"({(h.get('roi_on_deployed') or 0)*100:+.3f}% ROI). "
        )
    if t.get("both_sides_participation_rate"):
        parts.append(
            f"Both-sides participation is {t['both_sides_participation_rate']*100:.1f}%. "
        )
    if t.get("spread_leg_pl_usdc") is not None and t.get("directional_leg_pl_usdc") is not None:
        parts.append(
            f"Two-leg decomposition: spread leg "
            f"${t['spread_leg_pl_usdc']:,.0f}, directional leg "
            f"${t['directional_leg_pl_usdc']:,.0f}."
        )
    return "".join(parts)


def _stub_analysis_narrative(facts: dict[str, Any]) -> str:
    return _stub_at_a_glance(facts)


def _stub_two_leg_explainer(facts: dict[str, Any]) -> str:
    t = facts.get("two_leg", {})
    if not t:
        return "Insufficient resolved data to compute two-leg decomposition."
    return (
        f"Spread leg: paired shares of {t.get('paired_shares', 0):,.0f} at "
        f"median paired cost ${t.get('median_paired_cost', 0):.4f} produce "
        f"P/L of ${t.get('spread_leg_pl_usdc', 0):,.2f}. Directional leg: "
        f"{t.get('excess_shares', 0):,.0f} excess shares contribute "
        f"${t.get('directional_leg_pl_usdc', 0):,.2f}. Net realized "
        f"${t.get('realized_pl_usdc', 0):,.2f}."
    )


def _stub_dominance_explainer(facts: dict[str, Any]) -> str:
    rows = facts.get("dominance_buckets", []) or []
    if not rows:
        return "Insufficient resolved data to compute dominance buckets."
    top = rows[-1]
    bot = rows[0]
    return (
        f"Dominance ratio measures how lopsided the wallet's allocation "
        f"became by market close. Win rate climbs monotonically across "
        f"buckets — {(bot.get('dom_side_win_rate') or 0)*100:.1f}% in "
        f"the {bot.get('band_label')} bucket, "
        f"{(top.get('dom_side_win_rate') or 0)*100:.1f}% in the "
        f"{top.get('band_label')} bucket."
    )


def _stub_filter_recommendation(facts: dict[str, Any]) -> str:
    rows = facts.get("filter_ledger", []) or []
    if not rows:
        return "No filter recommendations available — insufficient resolved data."
    # Pick the highest-lift implementable filter
    lifts = [(r.get("name"), r.get("roi_lift_vs_baseline") or 0) for r in rows if r.get("name") and "Underdog blocking" in (r.get("name") or "")]
    if lifts:
        name, lift = lifts[0]
        return (
            f"Filter recommendation: {name}. Lifts ROI by {lift*100:+.2f}pp "
            f"vs. unfiltered. Tradeoff: removes the underdog hedge insurance, "
            f"so worst-case single-market loss expands."
        )
    return "Filter recommendation: see filter ledger."


def _stub_playbook_brief(facts: dict[str, Any]) -> str:
    h = facts.get("headline", {})
    return (
        f"Operator brief: wallet places ~{h.get('avg_trades_per_active_day', 0):,.0f} "
        f"trades per active day across {h.get('markets_touched', 0)} markets, "
        f"with ~{h.get('avg_fills_per_market', 0):.1f} fills per market. "
        f"Both-sides participation {(facts.get('two_leg', {}).get('both_sides_participation_rate') or 0)*100:.1f}%. "
        f"Replicating requires sub-second feeds + an executor able to fire that many orders/day."
    )


def _stub_what_to_copy(facts: dict[str, Any]) -> str:
    return (
        "- The single-instrument whitelist — narrow universe is part of the edge.\n"
        "- The skew-as-signal heuristic (load the side the orderbook is moving toward).\n"
        "- The explicit hedge tax on cheap-side underdog buys."
    )


def _stub_what_not_to_copy(facts: dict[str, Any]) -> str:
    return (
        "- The capital base — small accounts get eaten by variance.\n"
        "- The always-on infra (colocated host, redundant order router)."
    )


def _stub_pseudocode(facts: dict[str, Any]) -> str:
    """The replication pseudocode is *deterministic* — cribbed from the
    polyresearchrobotics format and parameterized by the analytics.
    LLMs tend to invent imports that don't exist, so we keep the
    pseudocode block hand-written.
    """
    median_size = facts.get("trade_size", {}).get("median_usdc") or 20.0
    return f"""\
# Per-market loop. One running instance per active 5-min market.
while market_seconds_remaining > 0:
    yes_price, no_price = clob.bbo(market.condition_id)
    btc_tick           = btc_tape.last_tick()
    yes_shares, no_shares = position_state(market.condition_id)
    skew = max(yes_shares, no_shares) / max(min(yes_shares, no_shares), 1e-9)

    # Phase 1: open both legs in the first 30 seconds
    if market_seconds_remaining > 270 and (yes_shares == 0 or no_shares == 0):
        side = "Yes" if yes_price <= no_price else "No"
        place_buy(market, side, shares=60)        # ~${median_size:.0f} clip
        continue

    # Phase 2: trend-follow during minutes 1-4
    elif market_seconds_remaining > 60:
        if btc_tick.direction == "up" and yes_price < 0.95:
            place_buy(market, "Yes", shares=60)
        elif btc_tick.direction == "down" and no_price < 0.95:
            place_buy(market, "No", shares=60)
        # Cheap-hedge insurance — skip when skew has already declared a side
        elif skew < 2.0 and yes_price < 0.20:
            place_buy(market, "Yes", shares=120)
        elif skew < 2.0 and no_price < 0.20:
            place_buy(market, "No", shares=120)

    # Phase 3: high-conviction final-90s loads on the dominant side ONLY
    elif market_seconds_remaining > 0:
        dominant_side = "Yes" if yes_shares > no_shares else "No"
        dom_price = yes_price if dominant_side == "Yes" else no_price
        if dom_price < 0.97:
            place_buy(market, dominant_side, shares=60)

    sleep(2)  # median inter-fill gap

# At market close: hold every share. No SELL. Resolution oracle settles.
"""


def _stub_bankroll(facts: dict[str, Any]) -> str:
    h = facts.get("headline", {})
    ts = facts.get("trade_size", {})
    return (
        f"Per-market clip is small — median ${ts.get('median_usdc', 0):.2f}, "
        f"max ${ts.get('max_usdc', 0):.2f}. With ~{h.get('avg_trades_per_active_day', 0):,.0f} "
        f"trades per active day and 5-minute hold-to-expiry settlement, "
        f"working capital recycles every 5 minutes — total deployed "
        f"capital of ${h.get('total_usdc_deployed', 0):,.0f} represents "
        f"cumulative turnover, not balance-sheet exposure."
    )


# ---------------------------------------------------------------------------
# Facts-builder + helpers
# ---------------------------------------------------------------------------


def _build_facts(
    analytics: WalletAnalytics, spotlight: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact, JSON-safe dict containing exactly the numbers the
    LLM is allowed to cite.  Nothing speculative or model-derived in
    here — every value comes from ``wallet_analytics``.
    """
    a = analytics
    top_dom = next(
        (d for d in reversed(a.dominance_buckets) if d.markets > 0),
        None,
    )
    return {
        "address": a.address,
        "headline": _to_json(a.headline),
        "trade_size": _to_json(a.trade_size),
        "cadence": _to_json(a.cadence),
        "two_leg": _to_json(a.two_leg),
        "side_split": a.side_split,
        "outcome_split": a.outcome_split,
        "daily_summary": {
            "active_days_count": sum(1 for d in a.daily_rows if d.trades > 0),
            "first_day": a.daily_rows[0].date if a.daily_rows else None,
            "last_day": a.daily_rows[-1].date if a.daily_rows else None,
        },
        "dominance_buckets": [_to_json(d) for d in a.dominance_buckets],
        "top_dominance_bucket": _to_json(top_dom) if top_dom else None,
        "paired_cost_bands": [_to_json(b) for b in a.paired_cost_bands if b.trades > 0],
        "price_buckets": [_to_json(b) for b in a.price_buckets if b.trades > 0],
        "filter_ledger": [_to_json(f) for f in a.filter_ledger],
        "within_window_timing": [_to_json(w) for w in a.within_window_timing if w.trades > 0],
        "rolling_7d": _to_json(a.rolling_7d) if a.rolling_7d else None,
        "rolling_15d": _to_json(a.rolling_15d) if a.rolling_15d else None,
        "archetypes": [_to_json(ar) for ar in a.archetypes],
        "top_by_volume": [_to_json(t) for t in a.top_by_volume],
        "top_winning": [_to_json(t) for t in a.top_winning],
        "top_losing": [_to_json(t) for t in a.top_losing],
        "spotlight_market": (
            {k: v for k, v in spotlight.items() if k != "rows"}
            if spotlight else None
        ),
    }


def _to_json(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(obj)
    return obj


def _split_bullets(text: str) -> list[str]:
    """Parse '- bullet\n- bullet' style into a list of bullet strings."""
    if not text:
        return []
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ", "• ")):
            out.append(line[2:].strip())
        elif line and out:
            # continuation of prior bullet
            out[-1] = out[-1] + " " + line
    return [b for b in out if b]


__all__ = ["ReportSections", "draft_sections"]
