# Plane-isolation phases — status & handoff (A.3, B, C, D)

Context: reduce trading-loop saturation by isolating heavy subsystems into
their own OS-process "planes". Phases A.1/A.2 (recording, scanner/detection)
landed earlier. This doc covers A.3, B, C, D.

**Do not push to origin until the operator has tested.** All commits below are
local-only by request.

## Done (committed locally, not pushed)

### Phase B — read-free orchestrator hot path — `c4b967d4`
The orchestrator called `read_orchestrator_control()` every cycle (both lanes)
and on every runtime-trigger spec build. That path runs
`ensure_orchestrator_control()`, which **writes** a normalized `settings_json`
row whenever the normalized form differs (it re-orders `enabled_strategies`) —
i.e. a steady stream of UPDATEs to the single `orchestrator_control` row on the
hot path, contending with the API and the crypto lane.

Fix: `ensure_orchestrator_control(session, *, read_only=False)` and
`read_orchestrator_control(session, *, read_only=False)`. With `read_only=True`
the normalization write is skipped (the cycle does not care about strategy
ordering). All three orchestrator hot-path reads now pass `read_only=True`.
Normalization still happens on API/startup writes. Net: zero hot-path writes to
`orchestrator_control`.

Not done (deliberately): caching the per-cycle `AppSettings` (creds) and
`list_traders` SELECTs. Those are cheap reads; caching them adds kill-switch /
creds-staleness risk for little gain. The per-cycle *write* was the contention,
and it is gone.

### Phase D — hot-loop SLO alarm — `2dadecd5`
`host._run_orchestrator_slo_monitor_loop()` (trading plane only) reads the
orchestrator's per-lane cycle lag (`now - last_run_at`) from in-process
`runtime_status` and raises a throttled operator alert (`notifier_bridge.publish_alert`,
5-min per-lane throttle) when the lag exceeds the SLA — surfacing resource-pressure
stalls immediately instead of only as degraded fill latency, and well under the
~180 s heartbeat-restart line.

Threshold is a first-class UI-configurable `global_runtime` field,
`orchestrator_cycle_slo_seconds` (default 30 s; `0` disables), normalized in
`_normalize_global_runtime_settings` alongside `trader_cycle_timeout_seconds` —
no hidden constant (per the no-hardcoded-settings rule).

Phase D's other half — pushing more CPU-bound work onto the existing process
pool (`cpu-pool=56`, landed under the earlier saturation task) — is incremental
and out of scope here.

## Blocked / foundational: C must precede A.3

### The verified dependency
A.3 ("move reconciliation to its own plane") **cannot be done before C**, and a
prior "split-only" note that assumed `reconcile_live_positions` was read-only
drift detection was wrong. Verified in code:

`services/trader_orchestrator/position_lifecycle.py:6102 reconcile_live_positions`
is the **live position-lifecycle / exit-execution engine**, not read-only
detection. Its docstring: *"Handles: stop-loss, take-profit, trailing stop, max
hold, market inactivity, and resolution detection."* When an exit trigger fires
it **places and cancels real orders** via `live_execution_service`
(`cancel_order`, ladder `place_order` at lines ~5751, ~8769–8857, ~9274) and
does hot-path `WalletStateCache` reads (`get_wallet_state_cache`, ~3730–3794).

`live_execution_service` (CLOB client, `_orders` cache, nonce/allowance state)
is trading-plane-resident. Therefore moving `reconcile_live_positions` to a
separate plane **before** execution is its own IPC-accessible process would
orphan it from `live_execution_service` — live positions would stop getting
their protective stop-loss / take-profit exits. Catastrophic. So: **C first,
then A.3.**

### Phase C — dedicated execution-core process (the foundational change)
Extract `live_execution_service` into its own OS process so order submission is
off the trading event loop and callable from any plane.

- New plane `execution` running an execution-core that **owns** the CLOB client,
  the `_orders` cache, and nonce/allowance/signature state. It is the **single
  writer** of orders — no split-brain on nonces/allowances across planes.
- In-process **client proxy** with the same surface as today's
  `live_execution_service` so call sites (`position_lifecycle`, the orchestrator,
  the fast trader) are unchanged. The proxy forwards mutating calls
  (`place_order`, `cancel_order`, `prepare_sell_balance_allowance`) over IPC
  (Redis req/reply or a dedicated socket) to the execution-core; idempotency
  keys on every submit so a retried IPC call cannot double-submit.
- Reads (`get_order`, `get_order_snapshots_by_clob_ids`) served from a
  replicated read cache or the same IPC.
- Exit hot path latency budget: stop-loss must still fire fast — measure IPC
  round-trip and keep it well inside the exit cadence.

### Phase A.3 — reconciliation plane (after C)
Once C lands: move `reconcile_live_positions` / `_run_reconciliation_cycle` to a
new `reconciliation` plane that reaches execution through the **same client
proxy** as trading. Keep `_run_wallet_cache_reseeder_loop` on trading (the
`WalletStateCache` freshness gate — trading halts without it) **and** run a
reseeder on the reconciliation plane (the heavy reconcile does hot-path cache
reads, so that plane needs a fresh cache too). Add the plane to
`host._PLANE_CONFIGS` and the GUI `_WORKER_PLANES` / `_WORKER_PLANE_BY_NAME`.
Heartbeat must be plane-distinct or the per-plane watchdog can't detect a hung
worker (don't let two planes write the same `worker_snapshot` row).

### Verification gate for C + A.3
1. Live exits still fire after C: force a stop-loss / take-profit on a live test
   position; confirm the sell submits and fills.
2. No duplicate / dropped orders under IPC retry (kill the execution-core
   mid-submit; confirm idempotency).
3. After A.3: reconciliation-plane drift output matches what trading produced
   pre-split, byte-for-byte on the same inputs.

## Why C/A.3 weren't rushed here
C rewrites the live order-submission path (real money). Per the institutional-
grade / no-shortcut rule, it needs fresh context and the verification gate
above, not a rushed end-of-context attempt. B and D were independent and safe to
land now; A.3 is hard-blocked on C.
