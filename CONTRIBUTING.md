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
   ./setup.sh
   ./run.sh
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

1. Create a new file in `backend/services/strategies/`
2. Extend `BaseStrategy` from `backend/services/strategies/base.py`
3. Implement the `detect()` method
4. Register the strategy in the scanner (`backend/services/scanner.py`)
5. Add tests in `backend/tests/`

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
