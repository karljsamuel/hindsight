"""Reject year-0001 timestamps at the database level.

The writer now refuses year 0001 in ``safe_entity_event_date`` and at every
parse entry point, and the predecessor migration repairs rows written by older
builds. This is the third layer: a CHECK constraint, so a future code path that
reintroduces the value fails at the database instead of corrupting a bank.

Why 0001 and nothing else
-------------------------
Only year 0001 is rejected. python-oracledb cannot convert it back -- Julian-day
arithmetic lands on -1, outside Python's datetime range -- so one stored value
makes entity resolution, recall and consolidation fail for that bank. The
corpus holds genuine dates from 1825, 1920 and 1958, and an earlier revision
that floored at 1970 would have silently discarded them; those pass.

The predicate is TO_CHAR(col, 'YYYY') <> '0001', not a comparison against a
timestamp literal, because Oracle cannot materialise a year-0001 timestamp in a
comparison (ORA-01843, surfaced through a parallel worker as ORA-12801) and
CAST cannot construct one at all (ORA-01847). Rendering the year never
materialises the value, which is what makes this enforceable.

ENABLE NOVALIDATE, deliberately
-------------------------------
The constraint is added without validating existing rows. Validating would fail
on any environment where a sentinel survives, and one can: the predecessor
migration keeps sentinel entities that are still referenced (mention_count > 0
or a unit_entities posting), so it can legitimately leave rows behind. Failing
the migration there would break a deployment to gain nothing, since the
constraint's job is to stop NEW writes. Existing rows are handled by that
migration plus the guarded read path.

A NULL column value passes: Oracle treats a CHECK evaluating to NULL as
satisfied, so the nullable memory_units columns need no explicit IS NULL guard.

Oracle only
-----------
PostgreSQL can represent year 0001 -- the failure there is a Python conversion
error, not a storage one -- and the predecessor migration already replaces the
sentinels with the epoch floor. Adding the constraint there is a separate
decision with its own validation, so this migration is a no-op on PostgreSQL.

Revision ID: b7d2e9f4c1a8
Revises: s4t5u6v7w8x9
Create Date: 2026-10-06
"""

from collections.abc import Sequence

from alembic import op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b7d2e9f4c1a8"
down_revision: str | Sequence[str] | None = "s4t5u6v7w8x9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table, column, constraint name). Names are kept under 30 characters, the
# pre-12.2 identifier limit, so they are portable.
_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("entities", "first_seen", "CK_ENTITIES_FIRST_SEEN"),
    ("entities", "last_seen", "CK_ENTITIES_LAST_SEEN"),
    ("entity_cooccurrences", "last_cooccurred", "CK_ENTCOOC_LAST_COOCCURRED"),
    ("memory_units", "event_date", "CK_MEMUNITS_EVENT_DATE"),
    ("memory_units", "occurred_start", "CK_MEMUNITS_OCCURRED_START"),
    ("memory_units", "occurred_end", "CK_MEMUNITS_OCCURRED_END"),
    ("memory_units", "mentioned_at", "CK_MEMUNITS_MENTIONED_AT"),
)


def _oracle_upgrade() -> None:
    # Oracle migrations run with CURRENT_SCHEMA set per tenant, so table names
    # stay unqualified; the constraint-name lookup is schema-scoped to match.
    bind = op.get_bind()
    for table, column, name in _CHECKS:
        existing = bind.exec_driver_sql(
            "SELECT COUNT(*) FROM all_constraints "
            "WHERE owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') "
            f"AND constraint_name = '{name}'"
        ).scalar()
        if existing:
            # Already present (re-run or manual apply); adding again is ORA-02264.
            continue
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {name} "
            f"CHECK (TO_CHAR({column}, 'YYYY') <> '0001') ENABLE NOVALIDATE"
        )


def _oracle_downgrade() -> None:
    for table, _column, name in _CHECKS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {name}")


def _pg_noop() -> None:
    # See the module docstring: PostgreSQL can represent year 0001 and its
    # predecessor migration already repairs the values, so this is a no-op.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_noop, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_noop, oracle=_oracle_downgrade)
