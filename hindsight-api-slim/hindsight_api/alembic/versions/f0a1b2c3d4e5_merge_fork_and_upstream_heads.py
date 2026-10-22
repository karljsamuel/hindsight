"""Merge the fork's sentinel migrations with upstream's scheduled-refresh head

Revision ID: f0a1b2c3d4e5
Revises: b7d2e9f4c1a8, d4f7a2c9e1b5
Create Date: 2026-10-17

The fork added a chain of its own after the last synced upstream commit
(``s4t5u6v7w8x9`` -> ``b7d2e9f4c1a8``), while upstream added a separate chain
ending at ``d4f7a2c9e1b5``. Nothing linked the two, so the merged tree carried
two heads. ``hindsight_api/migrations.py`` deliberately upgrades ``"heads"``
(plural), so two heads do not break startup, but every other tool — notably
``alembic revision --autogenerate`` — assumes exactly one. This revision joins
them and carries no DDL.

Both dialect slots are empty on purpose: the merge is pure bookkeeping, and
``run_for_dialect`` is present because ``tests/test_migration_shape.py`` fails CI
for any migration that omits the dispatcher.
"""

from collections.abc import Sequence

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f0a1b2c3d4e5"
down_revision: str | Sequence[str] | None = ("b7d2e9f4c1a8", "d4f7a2c9e1b5")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    """No DDL: this revision only joins two migration chains."""


def _pg_downgrade() -> None:
    """No DDL: a downgrade re-splits the chains, so there is nothing to undo."""


def _oracle_upgrade() -> None:
    """No DDL: this revision only joins two migration chains."""


def _oracle_downgrade() -> None:
    """No DDL: a downgrade re-splits the chains, so there is nothing to undo."""


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
