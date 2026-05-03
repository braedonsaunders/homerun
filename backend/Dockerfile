# syntax=docker/dockerfile:1.7
#
# Homerun backend image — used for the API, all worker planes, and the
# one-shot Alembic migration container.  The compose file overrides the
# CMD per service.
#
# Working tree inside the container mirrors the repo:
#   /app/
#     backend/   <- this image's source
#     data/      <- bind-mounted at runtime (postgres state lives in its
#                   own container; this is for FAISS indexes, ML weights,
#                   runtime logs)
#
# Source code in services/machine_learning_sdk.py uses
# ``Path(__file__).resolve().parents[2]`` to find the repo root, so the
# /app + /app/backend layout is load-bearing — don't flatten it.

ARG PYTHON_VERSION=3.12

# ---------- builder ----------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps for native wheels (web3/eth-hash, lifelines→scipy, faiss).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt requirements-trading.txt ./
RUN pip install --upgrade pip wheel \
    && pip install -r requirements.txt \
    && pip install -r requirements-trading.txt

# ---------- runtime ----------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HOMERUN_PROCESS_ROLE=server

# libgomp1 is required by faiss-cpu, xgboost and lightgbm at runtime.
# curl is included so healthchecks can hit /health without an extra image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1000 homerun \
    && useradd  --system --uid 1000 --gid 1000 --create-home --shell /bin/bash homerun

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app/backend
COPY --chown=homerun:homerun . /app/backend

# Bind-mount target for runtime state (FAISS indexes, ML weights, logs).
RUN install -d -o homerun -g homerun /app/data

USER homerun

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
