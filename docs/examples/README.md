# Live Tennis match-state feed (example)

A self-contained example of a **live-tennis match-state feed** for homerun,
modeled on the pollable-REST shape of `services/chainlink_direct_feed.py`.

It polls the [Live Tennis API](https://livetennisapi.com) live-scores and
fixtures endpoints, normalizes per-match state (set/game/point score, who is
serving, a three-valued break-point flag, retirement/walkover), and keys each
record for matching against Polymarket tennis market slugs — the kind of
ground-truth signal the `sports_overreaction_fader` strategy works *without*
today (it trades on price action alone).

**This is a docs example, not a core service.** Nothing in `backend/` imports
it. It runs on its own with just `httpx`; the availability-latch and logger
that a real feed would take from `utils.feed_availability` / `utils.logger` are
inlined here as minimal shims so the file is readable and runnable standalone.
To wire it into core, swap those shims back for the project's `utils.*` and move
it under `services/`.

Disclosure: contributed by the Live Tennis API team. It uses the **free** tier
(30 req/min, 100/day, no card): https://livetennisapi.com/subscribe/free — key
via the `LIVETENNIS_API_KEY` env var. It degrades to a slow, fixtures-only
cadence on a 429 and latches off (never hammers) when the key is missing.

## Run the tests

```bash
pip install httpx pytest pytest-asyncio
pytest docs/examples/test_livetennis_feed.py
```
