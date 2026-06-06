# Plane-isolation phases — status & handoff

Goal: stop the trading event loop from saturating by isolating heavy subsystems
into their own OS-process planes, and by cutting CPU/contention on the hot path.

**Do not push to origin until the operator has tested.** All commits local-only.

## Shipped & verified this session (local commits, not pushed)

### Phase B — read-only orchestrator control read — `c4b967d4`
`ensure/read_orchestrator_control(..., read_only=True)` skips the per-cycle
settings-normalization WRITE (it only re-orders enabled_strategies) on the
orchestrator hot path. All three hot-path reads use it. Net: zero hot-path
writes to the single `orchestrator_control` row from the cycle loop. (Did NOT
cache AppSettings/list_traders SELECTs — kill-switch/creds staleness risk, low
gain. The per-cycle write was the contention.)

### Phase D — hot-loop SLO alarm — `2dadecd5`
`host._run_orchestrator_slo_monitor_loop` (trading plane) reads orchestrator
cycle lag (`now - last_run_at`) from in-process runtime_status and raises a
throttled operator alert when it exceeds the SLA. Threshold is a UI-configurable
`global_runtime.orchestrator_cycle_slo_seconds` (default 30s, 0=off) — no hidden
constant. D's CPU-pool half was the earlier cpu-pool=56 work.

### C-backed order signing (coincurve) — `5a5fa7e0`
Verified `eth-keys` was on its pure-Python `NativeECCBackend` because coincurve
wasn't installed — every order signature did ECDSA over secp256k1 in pure
Python (1.26ms, holding the GIL on the trading loop). Installed `coincurve`
(libsecp256k1 C bindings); eth-keys auto-flips to it. Verified: signatures
byte-identical (RFC 6979), eth_account sign+recover intact, **23x faster
(1.26ms → 0.055ms; 794/s → 18,290/s)**. keccak256 already C-backed
(pycryptodome). Pinned in requirements-trading.txt.

## Corrected architecture (verified — supersedes the earlier "C before A.3")

The earlier note assumed reconcile had to move with exits, forcing a C-first
execution-core extraction. **Verified false.** Exits are already isolated:

- `services/trader_orchestrator/exit_risk_loop.py` is a **dedicated fast exit
  engine** on the trading plane: 2s sweep, its own `FastAsyncSessionLocal` pool
  (2.5s stmt / 500ms lock timeout — never queues behind the saturated worker
  pool), **WS price-change wake (sub-second)**, evaluates every open live
  position, sells via the shared `execute_position_exit`. This is the primary
  stop-loss/take-profit path, and it stays on trading. (Operator's call, correct.)
- `reconcile_live_positions` (position_lifecycle.py:6102, ~4.5k lines) is ~95%
  COLD housekeeping — wallet REST syncs, terminal audit, drift detection, mark
  refresh, bulk DB reconcile — plus a **backstop** that fires only when
  exit_risk_loop is stale >10s (line 10602; logs "reconcile STOP BACKSTOP
  fired"). In normal operation the backstop is a no-op.

**Consequences:**
- **Phase C (execution-core process) is unnecessary and dropped.** Only the
  trading plane ever needs to place orders (exit_risk_loop + backstop), so
  execution has no multi-writer problem to solve and no reason to be its own
  process. Order submit is IO-bound (awaits the CLOB, yields the loop) — not a
  saturation source; the signing CPU was, and coincurve fixed that.
- **The real saturation is the per-candidate CPU loop in reconcile + its DB-pool
  contention**, both sharing the orchestrator's process/GIL/pool. That is what
  moving the cold reconcile to its own plane removes.

## Cold-reconcile split — execution plan (the remaining work, A.3 redefined)

Move the heavy reconcile to a new `reconciliation` plane in **detection-only,
no-execution** mode. Keep exit_risk_loop + the backstop on the trading plane.

**Why this is a focused, tested change, not a rush:** `reconcile_live_positions`
contains ~12 mutating live-execution call sites across distinct close contexts
(external-close, wallet-flatten, inactivity, resolution-extreme, backstop,
ladder, simple exit). Grep `live_execution_service.cancel_order|execute_live_order
|run_exit_pass|prepare_sell_balance_allowance` in position_lifecycle.py — the
ones inside 6102–11466 are: 8774, 8833, 8857, 9274, 9310/9312, 9332/9334, 9736/
9738, 10719, 10824, 10831/10832, 10922/10924. A cold plane must touch NONE of
them.

**Recommended mechanism (M1):** add `place_exits: bool = True` to
`reconcile_live_positions`, gating every mutating execution call site (and the
backstop block at 10602). When `place_exits=False`: still compute "would-close"
detail/counters (useful telemetry), but never call live_execution_service /
execute_live_order / run_exit_pass / prepare_sell_allowance. Default True keeps
ALL existing callers byte-identical.

**The safety invariant that makes M1 shippable** — a unit test that mocks
live_execution_service, execute_live_order, run_exit_pass, and
_prepare_sell_allowance_bounded, runs reconcile_live_positions with
`place_exits=False` over a fixture of live positions (including stop-loss /
trailing / wallet-flatten triggers), and asserts **call_count == 0** on all of
them. That converts "12 scattered sites" into one verifiable guarantee.

**Then:**
1. Add a `reconciliation` plane to `host._PLANE_CONFIGS`: `worker_modules =
   ("workers.trader_reconciliation_worker",)`, its own DB pool, **no
   live_execution init**, empty runtime_names. Add to GUI `_WORKER_PLANES` /
   `_WORKER_PLANE_BY_NAME`.
2. Gate the reconciliation worker by `HOMERUN_WORKER_PLANE`:
   - `reconciliation` plane: run the heavy cycle with `place_exits=False`.
   - `trading` plane: run a cheap backstop pass (`place_exits=True`, skip the
     heavy REST/terminal-audit/bulk-reconcile-write) so the safety net stays
     local — OR drop the trading reconcile entirely and have the cold plane
     enqueue stale-loop closes via `publish_signal_batch` (runtime_signal_queue)
     for the trading plane to act on. Prefer keeping the cheap local backstop;
     it's simpler and never weakens the net.
   - Plane-distinct heartbeat name so the per-plane watchdog can detect a hung
     worker (don't let two planes write the same `worker_snapshot` row).
3. Dual-writer scoping: the cold plane must not write exit/status columns the
   trading backstop owns. Money path stays single-executor on trading; a cold
   bookkeeping race is self-healing on the next cycle (worst case a mismarked
   reconcile field, never a missed/duplicate live order).

**Wallet-cache reseeder** (`_run_wallet_cache_reseeder_loop`) stays on trading
(WalletStateCache freshness gate) AND runs on the reconciliation plane (the cold
reconcile reads the cache).

### Verification gate
1. The `place_exits=False ⇒ zero execution calls` unit test (above) is green.
2. Existing reconcile tests pass unchanged (defaults preserved).
3. Live exits still fire after the cutover: force a stop-loss on a live test
   position; confirm exit_risk_loop sells. Kill exit_risk_loop; confirm the
   backstop still closes.
4. Cold reconcile drift output matches pre-split on the same inputs.
