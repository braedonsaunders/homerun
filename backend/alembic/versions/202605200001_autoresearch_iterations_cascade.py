"""Add ON DELETE CASCADE to autoresearch_iterations.experiment_id FK.

When a trader is force-deleted, the cascade chain is:

    traders (DELETE) -> autoresearch_experiments (CASCADE; existing)
                     -> autoresearch_iterations  (no cascade; this migration)

Without the cascade on the iterations FK, postgres rejects the parent
delete with:

    ForeignKeyViolationError: update or delete on table
    "autoresearch_experiments" violates foreign key constraint
    "autoresearch_iterations_experiment_id_fkey" on table
    "autoresearch_iterations"

Reproduced 2026-05-20 trying to force-delete the "Crypto Entropy Maker"
trader (id=fb1e2fc1e6bb47fbb5dd199dafc671d2) — the trader had a
historical experiment row (f1d15e2c-fa56-4ad0-bdc2-6c289f46c5d2) which
in turn had iterations.  None of the related rows are useful after the
trader is gone, so cascade is the right semantic.

Drop + re-add the FK with ON DELETE CASCADE.  Cheap on this table —
it has only a few hundred rows at most.
"""

from alembic import op


revision = "202605200001"
down_revision = "202605160001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "autoresearch_iterations_experiment_id_fkey",
        "autoresearch_iterations",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "autoresearch_iterations_experiment_id_fkey",
        "autoresearch_iterations",
        "autoresearch_experiments",
        ["experiment_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(
        "autoresearch_iterations_experiment_id_fkey",
        "autoresearch_iterations",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "autoresearch_iterations_experiment_id_fkey",
        "autoresearch_iterations",
        "autoresearch_experiments",
        ["experiment_id"],
        ["id"],
    )
