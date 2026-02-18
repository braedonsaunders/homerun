# Traders Strategy Integration Tasks

## Completed in code
- [x] Formalized trader filter contract in `backend/services/strategy_sdk.py` (tier/side/source enums, schema, defaults, config validation).
- [x] Wired strategy config validation into strategy create/update flows in `backend/api/routes_strategies.py`.
- [x] Normalized trader signal ingestion at source in `backend/services/smart_wallet_pool.py` and added normalization counters.
- [x] Removed fallback filtering and made trader firehose filtering strategy-owned in `backend/services/traders_firehose_pipeline.py`.
- [x] Moved traders opportunity construction into strategy code in `backend/services/strategies/traders_confluence.py` using `create_opportunity` and full `strategy_context` payload.
- [x] Added dedicated traders snapshot path in `backend/services/shared_state.py`.
- [x] Refactored tracked-traders worker to produce and persist strategy-derived opportunities in `backend/workers/tracked_traders_worker.py`.
- [x] Unified API consumption through `GET /api/opportunities?source=traders` in `backend/api/routes.py`.
- [x] Removed legacy tracked-traders opportunities route from `backend/api/routes_discovery.py`.
- [x] Cut frontend traders opportunities panel over to unified opportunities payload in `frontend/src/components/RecentTradesPanel.tsx`.
- [x] Removed client-side source/tier/side gating in traders opportunities UI and moved ownership to strategy config UI.
- [x] Added migration to normalize persisted trader filter enums in `backend/alembic/versions/202602180006_traders_filter_contract_normalization.py`.

## Runtime rollout tasks
- [ ] Run Alembic upgrade so trader filter config values are normalized in existing DBs.
- [ ] Restart API and `tracked_traders_worker` so runtime strategy loader and worker snapshot path are active.
- [ ] Tail tracked-traders logs and verify per-cycle counts: raw signals, filtered signals, opportunities written.
- [ ] Confirm `GET /api/opportunities?source=traders` returns opportunities with `strategy_context.source_key=traders`.
- [ ] Confirm Opportunities -> Traders subtab populates from unified opportunities endpoint.

## Post-rollout checks
- [ ] Validate strategy-config edit loop: change traders strategy config in DB UI and confirm output changes next worker cycle.
- [ ] Confirm no route/UI source/tier/side filtering affects traders results outside strategy config.
- [ ] Capture 24h parity/quality metrics from worker logs and snapshot counts.
