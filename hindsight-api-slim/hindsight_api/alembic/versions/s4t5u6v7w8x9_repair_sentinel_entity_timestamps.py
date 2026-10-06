"""Repair Python-incompatible year-0001 timestamps after entity resolution.

LLM date extraction can return ``datetime.min`` (0001-01-01). It is truthy,
so every fallback of the form ``event_date or now()`` lets it through. Oracle
stores it, but python-oracledb cannot convert the year back: Julian-day
arithmetic lands on -1, outside Python's datetime range. A single entity row
then kills entity resolution/recall for its bank.

The writer now rejects only year 0001 via ``safe_entity_event_date``; valid
historical dates (including pre-1970) are preserved. This migration repairs
sentinels that were written by older builds.

PostgreSQL can UPDATE the values to the epoch. Oracle cannot UPDATE a corrupt
timestamp at all (ORA-01843, typically wrapped as ORA-12801), so its migration
deletes only orphan entity rows (mention_count 0 and no unit_entities refs) and
its derived co-occurrence rows. Referenced entity timestamps are guarded on the
read path, and will be replaced with the next real mention date where possible.

Revision ID: s4t5u6v7w8x9
Revises: e5b1c7d3a902
Create Date: 2026-10-05
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "s4t5u6v7w8x9"
down_revision: str | Sequence[str] | None = "e5b1c7d3a902"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for PostgreSQL multi-tenant migration runs."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # PostgreSQL can represent year 0001, so preserve entities and memory rows
    # while replacing only the broken sentinel with the neutral epoch floor.
    for table, col in (
        ("entities", "first_seen"),
        ("entities", "last_seen"),
        ("entity_cooccurrences", "last_cooccurred"),
        ("memory_units", "event_date"),
        ("memory_units", "occurred_start"),
        ("memory_units", "occurred_end"),
        ("memory_units", "mentioned_at"),
    ):
        op.execute(
            f"UPDATE {schema}{table} SET {col} = TIMESTAMPTZ '1970-01-01 00:00:00+00' "
            f"WHERE TO_CHAR({col}, 'YYYY') = '0001'"
        )


def _pg_downgrade() -> None:
    # Not reversible: the original per-entity dates are unrecoverable, and
    # guessing them back would be worse than leaving the epoch floor.
    pass


def _oracle_upgrade() -> None:
    # Oracle migrations run with CURRENT_SCHEMA set to each tenant, so table names
    # intentionally stay unqualified here.
    #
    # Oracle cannot UPDATE a year-0001 timestamp at all: evaluating the old value
    # in the update fails with ORA-01843 (usually wrapped as ORA-12801). Delete
    # only sentinel entity rows that are genuine orphans. Keep any entity with a
    # mention or unit_entities posting; entity reads project year 0001 to NULL,
    # and its writer replaces the date on the next mention.
    op.execute(
        "DELETE FROM entities "
        "WHERE (TO_CHAR(first_seen, 'YYYY') = '0001' "
        "OR TO_CHAR(last_seen, 'YYYY') = '0001') "
        "AND mention_count <= 0 "
        "AND NOT EXISTS (SELECT 1 FROM unit_entities ue WHERE ue.entity_id = entities.id)"
    )

    # Co-occurrence timestamps are derived cache/recency data; they can be
    # regenerated from unit_entities and do not own any content.
    op.execute(
        "DELETE FROM entity_cooccurrences "
        "WHERE TO_CHAR(last_cooccurred, 'YYYY') = '0001'"
    )


def _oracle_downgrade() -> None:
    # The removed rows held no real data; there is nothing to restore.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)