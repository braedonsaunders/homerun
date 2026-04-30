# Contributing to Homerun

Thanks for your interest in contributing. This document covers the process for contributing to Homerun and how to get your development environment set up.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/homerun.git
   cd homerun
   ```
3. Set up your development environment:
   ```bash
   ./scripts/infra/setup.sh
   ./scripts/infra/run.sh
   ```
4. Create a feature branch:
   ```bash
   git checkout -b your-feature-name
   ```

## Development Setup

### Requirements

- Python 3.10+
- Node.js 18+

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install ruff pytest  # dev tools
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
cd backend
pytest tests/
pytest tests/ -v  # verbose output
```

## Code Standards

### Python (Backend)

- **Linter/Formatter**: [Ruff](https://docs.astral.sh/ruff/) — enforced in CI
- Run before committing:
  ```bash
  ruff check backend/
  ruff format backend/
  ```
- Follow existing code patterns and conventions
- Use type hints for function signatures
- Use async/await for all I/O operations

### TypeScript (Frontend)

- **Type checking**: `tsc --noEmit` — enforced in CI
- Run before committing:
  ```bash
  cd frontend
  npx tsc --noEmit
  ```
- Use TypeScript strict mode
- Define types for API responses in `services/api.ts`

## Making Changes

### Adding a New Strategy

Strategies are stored in the database (`strategies` table), not as Python
files in this repo. The `.py` files under `backend/services/strategies/`
are *system* (built-in) strategies that ship with the platform and are
seeded into the DB at migration time — end users never edit them
directly.

#### As a user (creating a custom strategy in your install)

1. Open the **Strategies** screen in the UI and click **New Strategy**.
2. Start from `GET /strategies/template` (the form prefills it). The
   endpoint also returns curated examples — including a multi-timeframe
   *Compound Movement* example demonstrating
   `StrategySDK.MultiWindow`, `on_timeframe_close()`, and
   `StrategySDK.PersistentState`.
3. Extend `BaseStrategy` from `services.strategies.base` and implement
   `detect()` (or `detect_async()` for I/O-bound work). Optionally
   override `evaluate()` and `should_exit()`.
4. Save. The backend AST-validates the source (no `os` / `subprocess` /
   `eval` / arbitrary imports) and persists it into the `strategies`
   table. `StrategyLoader` compiles and hot-reloads it without a restart.
5. `GET /strategies/docs` is the live, machine-readable reference for
   `BaseStrategy`, the `StrategySDK` surface, hooks, and config schema.

API equivalents (CLI / scripting): `POST /strategies/validate` for a
pre-flight check, `POST /strategies` to create, `PUT /strategies/{id}`
to edit, `POST /strategies/{id}/reload` to force a hot-reload.

#### As a platform maintainer (adding a built-in / system strategy)

1. Add a new file in `backend/services/strategies/` extending
   `BaseStrategy`.
2. Register a seed entry in
   `backend/services/opportunity_strategy_catalog.py`
   (`SYSTEM_OPPORTUNITY_STRATEGY_SEEDS`) — slug, source_key, import path,
   config schema. The catalog converts seeds into `Strategy` rows on
   migration.
3. Add tests in `backend/tests/`.

System strategies are loaded the same way as user strategies (through
the `strategies` table); the only difference is that system rows have
`is_system=True` and are seeded from this repo at migration time.

### Adding an API Endpoint

1. Create or update a route file in `backend/api/`
2. Register the router in `backend/main.py`
3. Add TypeScript types in `frontend/src/services/api.ts` if consumed by the frontend

### Adding a Frontend Component

1. Create the component in `frontend/src/components/`
2. Follow existing patterns (React Query for data fetching, TailwindCSS for styling)
3. Wire it into `App.tsx` or the relevant parent component

## Pull Request Process

1. Ensure your code passes CI:
   - `ruff check backend/` (no lint errors)
   - `ruff format --check backend/` (properly formatted)
   - `npx tsc --noEmit` in `frontend/` (no type errors)
   - `npm run build` in `frontend/` (builds successfully)
2. Write a clear PR description explaining **what** changed and **why**
3. Keep PRs focused — one feature or fix per PR
4. Update the README if your change affects setup, configuration, or public APIs

## Reporting Bugs

Open an issue with:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python/Node versions and OS

## Suggesting Features

Open an issue with:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## Questions?

Open a discussion or issue — happy to help.
