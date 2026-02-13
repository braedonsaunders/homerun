"""Initial schema.

Revision ID: 202602130001
Revises:
Create Date: 2026-02-13 00:00:00.000000
"""

from __future__ import annotations

import sys
from pathlib import Path

from alembic import op

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.database import Base
from models.model_registry import register_all_models


# revision identifiers, used by Alembic.
revision = "202602130001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    register_all_models()
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    register_all_models()
    Base.metadata.drop_all(bind=op.get_bind())
